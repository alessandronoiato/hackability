#!/usr/bin/env python3
"""Measure pi0 on the planted-loophole task, and best-of-n for every K.

Run this before any RL. It answers:
  * are we in the regime the theory needs? honest mass a and accepted-code mass b
    per prompt, with b < a (so K = N goes honest) and (N/K) b / a > 1 at K = 1
    (rough R_K if the kernel row sums were equal, s_u = s_h);
  * are pass rates matched across K? (same prompts, different accepted code);
  * best-of-n with index tie-break: hack share among BoN outputs vs n, on
    disjoint blocks of fresh samples (falsifier 2: it must not move with n),
    plus the exact BoN curve (KL_n, hack share) used as the reference line.

    python experiments/tie_drift_llm/calibrate.py --dataset experiments/tie_drift_llm/data/mul2x2.json

Writes <out>/samples.jsonl (one line per sample, all K categories) and
<out>/summary.json. probe.py reads samples.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import task  # noqa: E402


def bern_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def bon_curve(a_i: np.ndarray, b_i: np.ndarray, ns) -> list[dict]:
    """Exact best-of-n (index tie-break) from per-prompt pi0 masses.

    On each prompt the output is pi0 conditioned on pass with probability
    P_n = 1 - (1 - p)^n, so the within-pass law is unchanged and
    KL = KL(Bern(P_n) || Bern(p)) per prompt. Prompts with p = 0 stay at pi0.
    """
    p = a_i + b_i
    out = []
    for n in ns:
        Pn = 1 - (1 - p) ** n
        kl = bern_kl(Pn, p)
        kl[p <= 0] = 0.0
        w = np.where(p > 0, Pn / np.maximum(p, 1e-12), 0.0)
        hack, hon = (w * b_i).sum(), (w * a_i).sum()
        out.append(dict(n=int(n), kl=float(kl.mean()), pass_rate=float(Pn.mean()),
                        hack_share=float(hack / max(hack + hon, 1e-12))))
    return out


def bootstrap_share(a_i, b_i, rng, B=2000):
    N = len(a_i)
    idx = rng.integers(0, N, size=(B, N))
    hk, hn = b_i[idx].sum(1), a_i[idx].sum(1)
    s = hk / np.maximum(hk + hn, 1e-12)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summarize(ds: dict, rows: list[dict], M: int, seed: int) -> dict:
    prompts, levels = ds["prompts"], ds["levels"]
    N = len(prompts)
    m = sum(p["loophole"] for p in prompts)
    rng = np.random.default_rng(seed)
    by_prompt: list[list[dict]] = [[] for _ in range(N)]
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)

    giveup = np.array([np.mean([r["chosen_level"] is not None for r in by_prompt[i]]) for i in range(N)])
    lvl_counts = {str(K): sum(r["chosen_level"] == K for r in rows) for K in levels}
    n_codes = sum(lvl_counts.values())
    res = dict(
        n_prompts=N, n_loophole=m, samples_per_prompt=M,
        truncated=float(np.mean([r["truncated"] for r in rows])),
        mean_len=float(np.mean([len(r["completion_ids"]) for r in rows])),
        giveup_rate=float(giveup.mean()),
        invalid_code_rate=float(np.mean([r["cat"][str(levels[0])] == "code_invalid" for r in rows])),
        chosen_level_frac={K: (c / n_codes if n_codes else 0.0) for K, c in lvl_counts.items()},
        per_K={},
    )
    ns = [2 ** j for j in range(0, 11)]
    for K in levels:
        cat = lambda r: r["cat"][str(K)]  # noqa: E731
        a_i = np.array([np.mean([cat(r) == "honest" for r in by_prompt[i]]) for i in range(N)])
        b_i = np.array([np.mean([cat(r) == "hack" for r in by_prompt[i]]) for i in range(N)])
        a, b = a_i.mean(), b_i.mean()
        share = b / max(a + b, 1e-12)
        lo, hi = bootstrap_share(a_i, b_i, rng)
        # Real best-of-n on disjoint blocks of the samples, first passing sample wins.
        bon_emp = []
        n = 1
        while n <= M:
            hk = hn = picks = 0
            for i in range(N):
                rs = by_prompt[i]
                for g0 in range(0, len(rs) - n + 1, n):
                    c = next((cat(r) for r in rs[g0:g0 + n] if cat(r) in task.PASS_CATEGORIES), None)
                    if c is not None:
                        picks += 1
                        hk += c == "hack"
                        hn += c == "honest"
            bon_emp.append(dict(n=n, outputs=picks, hack_share=hk / max(hk + hn, 1)))
            n *= 2
        res["per_K"][str(K)] = dict(
            a=float(a), b=float(b), pass_rate=float(a + b), hack_share=float(share),
            hack_share_ci95=[lo, hi],
            R_naive=float((m / K) * b / max(a, 1e-12)),
            frac_prompts_with_hack=float(np.mean(b_i > 0)),
            frac_prompts_with_honest=float(np.mean(a_i > 0)),
            bon_exact=bon_curve(a_i, b_i, ns),
            bon_empirical=bon_emp,
            a_i=a_i.tolist(), b_i=b_i.tolist(),
        )
    return res


def report(res: dict, levels: list[int]) -> None:
    print(f"\nprompts {res['n_prompts']} (loophole {res['n_loophole']}), "
          f"{res['samples_per_prompt']} samples each; mean length {res['mean_len']:.0f} tokens, "
          f"truncated {res['truncated']:.3f}")
    print(f"give-up rate {res['giveup_rate']:.3f}; unlisted-code rate {res['invalid_code_rate']:.3f}; "
          f"listed code chosen, by level: "
          + ", ".join(f"K={K}: {v:.2f}" for K, v in res["chosen_level_frac"].items()))
    print(f"\n{'K':>5} {'a=honest':>9} {'b=hack':>8} {'pass':>7} {'b/(a+b)':>8} {'95% CI':>15} "
          f"{'R_naive':>8} {'BoN n=1..':>10}")
    for K in levels:
        r = res["per_K"][str(K)]
        bon = " ".join(f"{x['hack_share']:.2f}" for x in r["bon_empirical"])
        print(f"{K:>5} {r['a']:>9.3f} {r['b']:>8.4f} {r['pass_rate']:>7.3f} {r['hack_share']:>8.3f} "
              f"[{r['hack_share_ci95'][0]:.3f},{r['hack_share_ci95'][1]:.3f}] "
              f"{r['R_naive']:>8.2f}   {bon}")

    print("\nChecks (WARN = fix the dataset before RL; INFO = context):")
    pk = [res["per_K"][str(K)] for K in levels]
    a = pk[0]["a"]
    bs = np.array([r["b"] for r in pk])
    ok = a > 0.05 and a < 0.8
    print(f"  [{'OK' if ok else 'WARN'}] honest mass a = {a:.3f} (want roughly 0.1-0.6; "
          "change --digits otherwise)")
    ok = bs.min() > 0.005
    print(f"  [{'OK' if ok else 'WARN'}] accepted-code mass b >= 0.005 for every K "
          f"(min {bs.min():.4f}); otherwise GRPO rarely samples the hack; try --template encouraged")
    ok = bs.max() < a
    print(f"  [{'OK' if ok else 'WARN'}] b < a for every K (max b {bs.max():.4f}): "
          "needed for the randomized arm to go honest (tabular regime)")
    r1, rN = pk[0]["R_naive"], pk[-1]["R_naive"]
    ok = r1 > 1 > rN
    print(f"  [{'OK' if ok else 'WARN'}] R_naive(K={levels[0]}) = {r1:.2f} > 1 > "
          f"R_naive(K={levels[-1]}) = {rN:.2f} (s_u = s_h assumed; probe.py measures the kernel)")
    rel = (bs.max() - bs.min()) / bs.mean() if bs.mean() > 0 else float("inf")
    print(f"  [{'OK' if rel < 0.3 else 'WARN'}] per-prompt hack mass matched across K: "
          f"b ranges over {bs.min():.4f}-{bs.max():.4f} (spread {rel:.0%} of mean; want < 30%; "
          "otherwise rebuild with another --seed)")
    for K, r in zip(levels, pk):
        e = [x["hack_share"] for x in r["bon_empirical"] if x["outputs"] >= 50]
        if len(e) > 1:
            print(f"  [INFO] K={K}: empirical BoN hack share over n: {min(e):.3f}-{max(e):.3f} "
                  f"(pi0 share {r['hack_share']:.3f}; must not trend with n)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--samples", type=int, default=32, help="samples per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=None, help="use only the first n prompts (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: <dataset dir>/calib_<dataset stem>")
    args = ap.parse_args()

    ds = task.load_dataset(args.dataset)
    if args.n_prompts:
        ds["prompts"] = ds["prompts"][:args.n_prompts]
    out = Path(args.out or Path(args.dataset).parent / f"calib_{Path(args.dataset).stem}")
    out.mkdir(parents=True, exist_ok=True)
    device = llm.pick_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = llm.load_tokenizer(args.model)
    model = llm.load_model(args.model, dtype, device)

    timer = llm.Timer()
    pids = llm.encode_prompts(tok, [p["text"] for p in ds["prompts"]], ds.get("fewshot"))
    samples = llm.generate(model, tok, pids, args.samples, max_new_tokens=args.max_new_tokens,
                           seed=args.seed, batch_size=args.gen_batch, device=device)
    print(f"generated {len(samples)} samples in {timer():.0f}s")

    rows = []
    for s in samples:
        p = ds["prompts"][s.prompt_idx]
        rows.append(dict(
            prompt_id=p["id"], completion_ids=s.completion_ids, text=s.text, truncated=s.truncated,
            chosen_level=task.chosen_level(s.text, p),
            cat={str(K): task.classify(s.text, p, K) for K in ds["levels"]},
        ))
    with open(out / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    res = summarize(ds, rows, args.samples, args.seed)
    res.update(dataset=str(args.dataset), model=args.model, max_new_tokens=args.max_new_tokens,
               seed=args.seed, levels=ds["levels"], spec=ds["spec"])
    with open(out / "summary.json", "w") as f:
        json.dump(res, f, indent=1)
    report(res, ds["levels"])

    K0 = str(ds["levels"][0])
    print("\nExamples:")
    for want in ("honest", "code_other", "hack", "wrong"):
        ex = [r for r in rows if r["cat"][K0] == want][:1]
        for r in ex:
            print(f"--- [{want}, K={K0}] prompt {r['prompt_id']}\n{r['text'][-600:]}")
    print(f"\nwrote {out}/samples.jsonl and summary.json")


if __name__ == "__main__":
    main()
