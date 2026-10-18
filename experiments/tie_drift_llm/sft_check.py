#!/usr/bin/env python3
"""Supervised capability check of the cue rule (development stage 7a, README).

Not policy gradient: direct supervision on the rule's answer. Starts from the original
pi0 and trains on a crossed build's train renderings (task.py variant --train-renderings
all) with the target exactly \\boxed{GIVE UP: <rule code>} + EOS, loss on the completion
tokens only (per-token mean over the batch). Full-model AdamW, constant lr, weight decay
0, no clipping, fp32 master weights (bf16 autocast on CUDA, as grpo.py). Every epoch
permutes each (cue, listing order) cell and splits it into equal groups; batch b takes
group b of every cell, so each rendering is seen once per epoch.

Outputs, in grpo.py's formats where analyze.py reads them:
  eval.jsonl, eval_samples.jsonl.gz  sampled evals at step 0, every --eval-every updates
                                     and the last update (--final-eval-samples[-train]);
                                     analyze.py --pc3 --train-fit --at-step <last> gives the
                                     held-out PC3 verdict and the separate training-fit verdict
  train.jsonl                        per update: loss, overall and by cell
  track.jsonl                        step 0 and every --track-every updates: teacher-forced
                                     log pi of the correct and of the other exact answer, by
                                     split and cell (descriptive). log pi(correct) reflects both
                                     the format and the code choice; sigmoid(margin) is the
                                     preference between the two exact strings, not the chance
                                     that free generation writes the correct code
  ckpt_final/                        the trained policy: a diagnostic artifact, not a
                                     starting point for the rule-vs-exception comparison

    python experiments/tie_drift_llm/sft_check.py --dataset .../c2ndxX_s0.json --seed 0 --grad-ckpt
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import task  # noqa: E402

COND = "rule"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None, help="run dir (default: <dataset dir>/runs/<stem>_sft/rule_sft_s<seed>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="data order and sampling")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-cell", type=int, default=8, help="train renderings per (cue, order) cell per batch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--track-every", type=int, default=16, help="updates between teacher-forced tracks")
    ap.add_argument("--track-micro-batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=32, help="updates between sampled evals")
    ap.add_argument("--eval-samples", type=int, default=2, help="samples per train rendering at each eval")
    ap.add_argument("--eval-samples-other", type=int, default=4, help="samples per held-out rendering")
    ap.add_argument("--final-eval-samples", type=int, default=32, help="samples per held-out rendering at the end")
    ap.add_argument("--final-eval-samples-train", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--no-save", action="store_true", help="do not save the final policy")
    ap.add_argument("--n-questions", type=int, default=None,
                    help="smoke tests: only the first n train and n held-out questions (all renderings)")
    return ap.parse_args()


def cell_of(p: dict) -> str:
    return f"{p['cue']}/{task.listing_order(p)}"


def main() -> None:
    a = parse_args()
    ds = task.load_dataset(a.dataset)
    if ds["spec"].get("train_renderings") != "cue x order":
        sys.exit("needs a crossed build (task.py variant --train-renderings all)")
    prompts = [p for p in ds["prompts"] if task.split_of(p) == "train"]
    held = [p for p in ds["prompts"] if task.split_of(p) == "heldout"]
    if a.n_questions:
        prompts = [p for p in prompts if p["question"] < a.n_questions]
        keep = list(dict.fromkeys((p["x"], p["y"]) for p in held))[:a.n_questions]
        held = [p for p in held if (p["x"], p["y"]) in keep]
    N = len(prompts)
    by: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        by[cell_of(p)].append(i)
    sizes = {len(v) for v in by.values()}
    if len(by) != 4 or len(sizes) != 1 or next(iter(sizes)) % a.per_cell:
        sys.exit(f"needs four equal (cue, order) cells divisible by --per-cell; {[(c, len(v)) for c, v in sorted(by.items())]}")
    cells = [by[c] for c in sorted(by)]
    per_epoch = next(iter(sizes)) // a.per_cell
    total = a.epochs * per_epoch

    out = Path(a.out or Path(a.dataset).parent / "runs" / f"{Path(a.dataset).stem}_sft" / f"{COND}_sft_s{a.seed}")
    if any((out / f).exists() for f in ("train.jsonl", "eval.jsonl")) and not a.overwrite:
        sys.exit(f"{out} already holds a run; pass --out <new dir> (or --overwrite to replace it)")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("train.jsonl", "eval.jsonl", "track.jsonl", "eval_samples.jsonl.gz"):
        (out / f).unlink(missing_ok=True)

    device = llm.pick_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    gen_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = llm.load_tokenizer(a.model)
    policy = llm.load_model(a.model, torch.float32, device, train=True)
    ref = llm.load_model(a.model, torch.float32, device)
    sampler = llm.load_model(a.model, gen_dtype, device)
    if a.grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    pad, eos = tok.pad_token_id, tok.eos_token_id

    def answer(code: str) -> list[int]:
        return tok(f"\\boxed{{GIVE UP: {code}}}", add_special_tokens=False)["input_ids"] + [eos]

    def other(p: dict) -> str:
        return next(w for w in p["codes"].values() if w != p["accept"][COND])

    for p in prompts + held:  # the grader must read the target as the accepted code
        assert task.classify(f"\\boxed{{GIVE UP: {p['accept'][COND]}}}", p, COND) == "hack"
    pids = llm.encode_prompts(tok, [p["text"] for p in prompts], ds.get("fewshot"))
    hids = llm.encode_prompts(tok, [p["text"] for p in held], ds.get("fewshot"))
    T = [answer(p["accept"][COND]) for p in prompts]
    cfg = dict(vars(a), cond=COND, reward="sft", eos_token=tok.eos_token, updates=total, updates_per_epoch=per_epoch,
               n_prompts_used=N, eval_splits={"heldout": len(held)}, dataset_spec=ds["spec"], run_dir=str(out),
               n_trainable=sum(p.numel() for p in params))
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"{out.name}: {N} train renderings ({len(cells)} cells x {len(cells[0])}), {len(held)} held-out, "
          f"{total} updates ({a.epochs} epochs x {per_epoch}), lr {a.lr:g}, device {device}", flush=True)

    def log(fname, rec):
        with open(out / fname, "a") as f:
            f.write(json.dumps(rec) + "\n")

    @torch.no_grad()
    def sync_sampler():
        for ps, pp in zip(sampler.parameters(), policy.parameters()):
            ps.copy_(pp)

    def by_cell(plist, values) -> dict:
        d = defaultdict(list)
        for p, v in zip(plist, values):
            d[cell_of(p)].append(v)
        return d

    # ---------------------------------------------------------------- track
    def track(step: int) -> None:
        rec, line = dict(step=step), []
        for name, plist, ids in (("train", prompts, pids), ("heldout", held, hids)):
            lc = llm.sequence_logprobs(policy, ids, [answer(p["accept"][COND]) for p in plist], pad, a.track_micro_batch).numpy()
            lo = llm.sequence_logprobs(policy, ids, [answer(other(p)) for p in plist], pad, a.track_micro_batch).numpy()
            m = lc - lo
            rec[name] = {}
            for c, v in sorted(by_cell(plist, zip(lc, lo, m)).items()):
                v = np.array(v)
                rec[name][c] = dict(n=len(v), lp_correct=float(v[:, 0].mean()), lp_other=float(v[:, 1].mean()),
                                    margin=float(v[:, 2].mean()), p_correct=float((1 / (1 + np.exp(-v[:, 2]))).mean()),
                                    frac_margin_pos=float((v[:, 2] > 0).mean()))
            line.append(f"{name}: " + "; ".join(f"{c} lp* {r['lp_correct']:.2f} margin {r['margin']:+.2f} "
                                                f"P {r['p_correct']:.2f}" for c, r in rec[name].items()))
        log("track.jsonl", rec)
        print(f"  [track {step:4d}] " + " | ".join(line), flush=True)

    # ---------------------------------------------------------------- eval (grpo.py's format)
    def evaluate(step: int, final: bool) -> None:
        t = llm.Timer()
        sync_sampler()
        S = a.final_eval_samples_train if final else a.eval_samples
        S_other = a.final_eval_samples if final else a.eval_samples_other
        smp = llm.generate(sampler, tok, pids, S, max_new_tokens=a.max_new_tokens,
                           seed=10_000_019 * (a.seed + 1) + step, batch_size=a.gen_batch, device=device)
        P = [pids[s.prompt_idx] for s in smp]
        C = [s.completion_ids for s in smp]
        lp = llm.sequence_logprobs(policy, P, C, pad, a.micro_batch * 2).numpy()
        lr_ = llm.sequence_logprobs(ref, P, C, pad, a.micro_batch * 2).numpy()
        cats = [task.classify(s.text, prompts[s.prompt_idx], COND) for s in smp]
        pi, cat_arr = np.array([s.prompt_idx for s in smp]), np.array(cats)
        cnt = {c: np.bincount(pi[cat_arr == c], minlength=N) for c in task.CATEGORIES}
        hon, hk = cnt["honest"].sum(), cnt["hack"].sum()
        rec = dict(step=step, samples=S, samples_other=S_other,
                   kl=float(np.bincount(pi, weights=lp - lr_, minlength=N).mean() / S),
                   kl_se=float((lp - lr_).std() / math.sqrt(len(smp))),
                   pass_rate=float(((cnt["honest"] + cnt["hack"]) / S).mean()),
                   hack_share=float(hk / max(hon + hk, 1)),
                   gold=float(np.mean([task.gold_reward(c, 0.0) for c in cats])),
                   mean_len=float(np.mean([len(c) for c in C])),
                   truncated=float(np.mean([s.truncated for s in smp])),
                   **{f"frac_{c}": float(cnt[c].sum() / len(smp)) for c in task.CATEGORIES})
        m = task.eval_metrics([s.text for s in smp], [prompts[s.prompt_idx] for s in smp], COND, 0.0)
        rec.update(giveup=m["giveup"], hit=m["hit"], **{k: m[k] for k in ("cue_follow", "by_cue") if k in m})
        so = llm.generate(sampler, tok, hids, S_other, max_new_tokens=a.max_new_tokens,
                          seed=10_000_019 * (a.seed + 1) + step + 7919, batch_size=a.gen_batch, device=device)
        rec["splits"] = {"heldout": task.eval_metrics([s.text for s in so], [held[s.prompt_idx] for s in so], COND, 0.0)}
        with gzip.open(out / "eval_samples.jsonl.gz", "at") as f:
            for split, samples, plist, cs in (("train", smp, prompts, cats), ("heldout", so, held, None)):
                for j, s in enumerate(samples):
                    p = plist[s.prompt_idx]
                    f.write(json.dumps(dict(step=step, split=split, prompt_id=p["id"],
                                            cat=cs[j] if cs else task.classify(s.text, p, COND),
                                            chosen=task.chosen_label(s.text, p), text=s.text)) + "\n")
        rec["eval_seconds"] = t()
        log("eval.jsonl", rec)
        hs = rec["splits"]["heldout"]
        print(f"  [eval {step:4d}] KL {rec['kl']:.3f}  train give-up {rec['giveup']:.3f} hit {rec['hit'] or 0:.3f}  "
              f"held-out give-up {hs['giveup']:.3f} hit {hs.get('hit') or 0:.3f}  len {rec['mean_len']:.0f}  "
              f"({S} / {S_other} samples, {rec['eval_seconds']:.0f}s)", flush=True)

    # ---------------------------------------------------------------- train
    track(0)
    evaluate(0, final=False)
    step = 0
    for epoch in range(1, a.epochs + 1):
        perms = [rng.permutation(ids) for ids in cells]
        for b in range(per_epoch):
            t = llm.Timer()
            bidx = [int(i) for perm in perms for i in perm[b * a.per_cell:(b + 1) * a.per_cell]]
            step += 1
            n_tok = sum(len(T[i]) for i in bidx)
            nll = np.zeros(len(bidx))
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(bidx), a.micro_batch):
                mb = bidx[j:j + a.micro_batch]
                lp, mask = llm.token_logprobs(policy, [pids[i] for i in mb], [T[i] for i in mb], pad)
                (-lp.sum() / n_tok).backward()
                nll[j:j + len(mb)] = (-lp.sum(-1) / mask.sum(-1)).detach().cpu().numpy()
            gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
            opt.step()
            lens = np.array([len(T[i]) for i in bidx])
            loss_cell = {c: float(np.average(nll[[k for k, i in enumerate(bidx) if cell_of(prompts[i]) == c]],
                                             weights=lens[[k for k, i in enumerate(bidx) if cell_of(prompts[i]) == c]]))
                         for c in sorted(by)}
            rec = dict(step=step, epoch=epoch, loss=float(np.average(nll, weights=lens)), loss_by_cell=loss_cell,
                       grad_norm=gnorm, lr=a.lr, t_step=t())
            log("train.jsonl", rec)
            print(f"step {step:4d}  epoch {epoch}  loss {rec['loss']:.4f}  ["
                  + " ".join(f"{c} {v:.3f}" for c, v in loss_cell.items()) + f"]  |g| {gnorm:.3g}  {rec['t_step']:.1f}s",
                  flush=True)
            if step % a.track_every == 0 or step == total:
                track(step)
            if step % a.eval_every == 0 or step == total:
                evaluate(step, final=step == total)
    if not a.no_save:
        sync_sampler()
        sampler.save_pretrained(out / "ckpt_final")
        tok.save_pretrained(out / "ckpt_final")
        print(f"  saved {out / 'ckpt_final'} (diagnostic artifact; not a starting point for the main comparison)")
    print(f"done: {out}\n  verdict: python {Path(__file__).parent}/analyze.py --dataset {a.dataset} "
          f"--runs {out.parent} --at-step {total} --pc3 --train-fit --trajectory")


if __name__ == "__main__":
    main()
