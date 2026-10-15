#!/usr/bin/env python3
"""Measure the first-step tie drift of the real model (no training).

For a loophole prompt i under condition K, let H_i be its honest (correct)
responses and K_i its accepted-code responses, and
l_i = log pi(K_i | i) - log pi(H_i | i). One update theta += eta * Delta moves
l_i by eta * <v_k(i) - v_h(i), Delta> + O(eta^2) (Lemma "discrete drift"),
where v_k, v_h are the gradients of log pi(K_i), log pi(H_i): the average of
grad log pi(y) over members y sampled from pi0 inside the set.
The probe evaluates this at pi0 for
  * SGD:            Delta = grad J_K        (the expected GRPO direction),
  * Adam / sign:    Delta = sign(grad J_K)  (Adam's first step is m/sqrt(v) = sign(g)),
and splits the SGD drift into the prompt's own term (tabular) and the term from
the other prompts (cross-prompt; the coherence of Theorem "threshold"):
  <v, grad J_K> = <v, grad J_{K,i}> + <v, grad J_K - grad J_{K,i}>.
Prediction: cross_k (hack, other prompts) falls from K = 1 to K = N (K design),
or is larger under rule / anti than under exception (cue design); the SGD drift
d_sgd does too. If cross_k is as large in the per-prompt arm, the shared "give up"
component dominates and the intervention cannot work.

cross_k is already the source/target split: the update comes from the other
prompts, the change is measured on prompt i. Only train-split prompts are sources
(GRPO never trains on held-out or flip prompts). In the K design the cross_k gap
tracked pi0 word preference (giveup_adv in calibrate.py) in all four seeds, so
read it against those cheap baselines, which the readout prints.

grad J_K and the set members use disjoint halves of the calibration samples,
so the self term is not inflated by reusing a sample in both.

    python experiments/tie_drift_llm/probe.py --dataset experiments/tie_drift_llm/data/mul2x2.json

Needs about 12 GB of GPU memory for a 0.5B model (five fp32 parameter vectors).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calib-dir", default=None, help="default: <dataset dir>/calib_<dataset stem>")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--conds", "--levels", dest="conds", nargs="+", default=None,
                    help="default: all dataset conditions")
    ap.add_argument("--n-probe", type=int, default=32, help="probe prompts (need >= 1 honest member)")
    ap.add_argument("--max-members", type=int, default=4, help="members per set used for v_h, v_k")
    ap.add_argument("--std-norm", action="store_true", help="GRPO std-normalized advantages in grad J")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: <calib dir>/probe.json")
    args = ap.parse_args()

    ds = task.load_dataset(args.dataset)
    prompts = ds["prompts"]
    levels = args.conds or task.conditions(ds)
    cdir = Path(args.calib_dir or Path(args.dataset).parent / f"calib_{Path(args.dataset).stem}")
    rows = [json.loads(line) for line in open(cdir / "samples.jsonl")]
    N = max(r["prompt_id"] for r in rows) + 1
    prompts = prompts[:N]
    by_prompt: list[list[dict]] = [[] for _ in range(N)]
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    M = min(len(b) for b in by_prompt)
    half = M // 2
    grad_rows = [b[:half] for b in by_prompt]      # for grad J
    memb_rows = [b[half:M] for b in by_prompt]     # for v_h, v_k
    train = [i for i in range(N) if task.split_of(prompts[i]) == "train"]

    device = llm.pick_device(args.device)
    tok = llm.load_tokenizer(args.model)
    model = llm.load_model(args.model, torch.float32, device, train=True)
    model.eval()  # no dropout; gradients still flow
    params = [p for p in model.parameters() if p.requires_grad]
    eos = llm.eos_ids(model, tok)[0]
    pids = llm.encode_prompts(tok, [p["text"] for p in prompts], ds.get("fewshot"))
    pad = tok.pad_token_id

    def grad_of(items: list[tuple[int, list[int], float]]) -> torch.Tensor:
        """Flat gradient of sum_w w * log pi(y | x) over (prompt, completion, w)."""
        model.zero_grad(set_to_none=True)
        for b0 in range(0, len(items), args.micro_batch):
            mb = items[b0:b0 + args.micro_batch]
            lp, _ = llm.token_logprobs(model, [pids[i] for i, _, _ in mb], [c for _, c, _ in mb], pad)
            w = torch.tensor([x for _, _, x in mb], device=lp.device)
            (w * lp.sum(-1)).sum().backward()
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in params]).float()

    def sign_dot(g: torch.Tensor, v: torch.Tensor, chunk: int = 1 << 26) -> float:
        return float(sum((torch.sign(g[c:c + chunk]) * v[c:c + chunk]).sum()
                         for c in range(0, g.numel(), chunk)))

    def weights(rs: list[dict], K: int) -> np.ndarray:
        r = np.array([task.proxy_reward(x["cat"][str(K)]) for x in rs])
        A = r - r.mean()
        if args.std_norm:
            A = A / (r.std() + 1e-4)
        return A / (len(train) * len(rs))

    def hack_members(i: int, K: int) -> tuple[list[list[int]], str]:
        code = task.accepted_code(prompts[i], K)
        # Members come from the half not used for grad J only, so that J and the member
        # vectors are independent (the self term would otherwise be biased).
        hit = [r["completion_ids"] for r in memb_rows[i] if r["cat"][str(K)] == "hack"]
        if hit:
            return hit[:args.max_members], "sampled"
        chosen = lambda r: r.get("chosen", r.get("chosen_level"))  # noqa: E731 (older samples: chosen_level)
        gave_up = [r for r in memb_rows[i] if chosen(r) is not None]
        if gave_up:
            out = []
            for r in gave_up[:args.max_members]:
                old = prompts[i]["codes"][str(chosen(r))]
                text = r["text"].replace(old, code)
                out.append(tok(text, add_special_tokens=False)["input_ids"] + [eos])
            return out, "substituted"
        ids = tok(f"\\boxed{{GIVE UP: {code}}}", add_special_tokens=False)["input_ids"] + [eos]
        return [ids], "template"

    rng = np.random.default_rng(args.seed)
    cand = [i for i in train if prompts[i]["loophole"]
            and any(r["cat"][str(levels[0])] == "honest" for r in memb_rows[i])]
    probe = sorted(rng.permutation(cand)[:args.n_probe].tolist())
    print(f"{len(probe)} probe prompts; grad J from {half} samples/prompt, members from {M - half}")
    if not probe:
        raise SystemExit("no loophole prompt has an honest member in the held-out half; "
                         "rerun calibrate.py with more --samples")
    if len(probe) < args.n_probe:
        print(f"warning: only {len(probe)} loophole prompts have an honest member in the held-out half")

    timer = llm.Timer()
    results = {}
    for K in levels:
        items = []
        for i in train:
            for r, w in zip(grad_rows[i], weights(grad_rows[i], K)):
                if w != 0:
                    items.append((i, r["completion_ids"], float(w)))
        G = grad_of(items)
        gnorm = float(G.norm())
        per = []
        for i in probe:
            hon = [r["completion_ids"] for r in memb_rows[i] if r["cat"][str(K)] == "honest"][:args.max_members]
            hk, src = hack_members(i, K)
            v_h = grad_of([(i, c, 1.0 / len(hon)) for c in hon])
            v_k = grad_of([(i, c, 1.0 / len(hk)) for c in hk])
            own = [(i, r["completion_ids"], float(w)) for r, w in zip(grad_rows[i], weights(grad_rows[i], K))
                   if w != 0]
            g_i = grad_of(own) if own else torch.zeros_like(G)
            hG, kG = float(v_h @ G), float(v_k @ G)
            hs, ks = float(v_h @ g_i), float(v_k @ g_i)
            per.append(dict(
                prompt=i, hack_source=src, n_honest=len(hon), n_hack=len(hk),
                d_sgd=kG - hG, d_sign=sign_dot(G, v_k) - sign_dot(G, v_h),
                self_k=ks, self_h=hs, cross_k=kG - ks, cross_h=hG - hs,
                norm_k=float(v_k.norm()), norm_h=float(v_h.norm()),
                cos_k_G=kG / max(float(v_k.norm()) * gnorm, 1e-30),
            ))
            del v_h, v_k, g_i
        del G
        results[str(K)] = dict(grad_norm=gnorm, per_prompt=per)
        print(f"K={K}: done in {timer():.0f}s")

    def ms(K, key):
        x = np.array([p[key] for p in results[str(K)]["per_prompt"]])
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    keys = ["d_sgd", "d_sign", "cross_k", "cross_h", "self_k", "self_h"]
    print("\nMean over probe prompts (± s.e.). d = first-order change of log pi(hack)/pi(honest) per unit step.")
    print(f"{'K':>5} " + " ".join(f"{k:>21}" for k in keys) + "  frac d_sgd>0  hack members")
    summary = {}
    for K in levels:
        row = {k: ms(K, k) for k in keys}
        per = results[str(K)]["per_prompt"]
        fpos = float(np.mean([p["d_sgd"] > 0 for p in per]))
        srcs = {s: sum(p["hack_source"] == s for p in per) for s in ("sampled", "substituted", "template")}
        summary[str(K)] = dict({k: dict(mean=float(m), se=float(s)) for k, (m, s) in row.items()},
                               frac_d_sgd_pos=fpos, hack_sources=srcs)
        print(f"{K:>5} " + " ".join(f"{m:>11.3e}±{s:<9.1e}" for m, s in row.values())
              + f"  {fpos:>11.2f}  {srcs}")

    levels = [str(K) for K in levels]
    if all(K.isdigit() for K in levels):
        contrasts = [(levels[0], levels[-1])]
    else:
        contrasts = [(c, "exception") for c in ("rule", "anti") if c in levels and "exception" in levels]
    print("\nReading:")
    for c1_, cN_ in contrasts:
        c1, cN = summary[c1_]["cross_k"], summary[cN_]["cross_k"]
        d1, dN = summary[c1_]["d_sgd"], summary[cN_]["d_sgd"]
        gap_se = np.hypot(c1["se"], cN["se"])
        print(f"  cross_k {c1_} minus {cN_}: {c1['mean'] - cN['mean']:.3e} (s.e. {gap_se:.1e}). "
              "Theory: > 0 (coherence). About 0: the shared arm's hack is no more coherent.")
        print(f"  d_sgd {c1_} vs {cN_}: {d1['mean']:.3e} vs {dN['mean']:.3e}. Theory: {c1_} larger; "
              "sign(d_sgd) is the predicted initial direction under SGD.")
        print(f"  d_sign {c1_} vs {cN_}: {summary[c1_]['d_sign']['mean']:.3e} vs "
              f"{summary[cN_]['d_sign']['mean']:.3e}. Same question for Adam's first step.")
    print(f"  cross_h (honest coherence, the gamma of Prop. initial): {summary[levels[0]]['cross_h']['mean']:.3e}; "
          f"if it dwarfs cross_k, honest coherence dominates (falsifier 4).")
    cal_p = cdir / "summary.json"
    if cal_p.exists():
        cal = json.load(open(cal_p))
        pc = cal.get("per_cond") or cal.get("per_K", {})
        print("  Cheap baselines (calibrate.py; the probe is informative only where it disagrees with these):")
        for K in levels:
            if K in pc:
                ga = pc[K].get("giveup_adv")
                print(f"    {K}: b {pc[K]['b']:.4f}  pi0 hack share {pc[K]['hack_share']:.3f}  "
                      f"giveup_adv {'-' if ga is None else f'{ga:+.3f}'}")

    out = Path(args.out or cdir / "probe.json")
    with open(out, "w") as f:
        json.dump(dict(levels=levels, probe_prompts=probe, std_norm=args.std_norm,
                       summary=summary, results=results), f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
