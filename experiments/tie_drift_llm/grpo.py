#!/usr/bin/env python3
"""GRPO on the planted-loophole task, one grader condition per run.

K design (draft_tie_drift.tex, Section "What would falsify this"): the hack share
among passing outputs is high for K = 1 and falls as K grows; best-of-n
(calibrate.py) is flat in K. Cue design (task.py build-cue): the hack share is
higher under a shared rule (rule or anti) than under per-prompt exceptions.
Only train-split prompts are trained on. Evals also cover the held-out and flip
splits, where hit (the share of listed-code give-ups that use the accepted code),
reported per cue value with counts, separates a learned rule from per-prompt
memorization and from cue-independent shortcuts (always (B), give up on one cue).

The pooled hack share also moves when prompt pass rates move (prompt reweighting),
even if no prompt's honest/hack mix changes. Each eval therefore also reports the
score-only share: every prompt keeps its pi0 mix b_i / (a_i + b_i) (calibration) at
its current pass rate. hack_share_tie - hack_share_score is the within-tie drift.

--reward hack_only pays only the accepted code (positive control: can the rule be
learned within the budget when it is the only thing rewarded?).

Design choices that matter for the theory:
  * the loss is sequence-level PG on log pi(y | x) (sum over tokens; Dr.GRPO
    style, no length normalization) unless --loss-agg says otherwise;
  * fp32 master weights (bf16 autocast); a bf16 copy samples, synced every step;
  * the frozen reference pi0 is an fp32 copy run under the same autocast, so the
    KL estimate is exactly 0 at step 0;
  * sampling is exact (temperature 1, no top-k/top-p/repetition penalty);
  * prompt order and sampling seeds depend only on --seed, so runs with the
    same seed and different K share their random numbers.

    python experiments/tie_drift_llm/grpo.py --dataset .../mul2x2.json --K 1 --optimizer adam --seed 0
    python experiments/tie_drift_llm/grpo.py --dataset .../cue_s0.json --cond rule --steps 30 --kl-stop 100
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import task  # noqa: E402


class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-g["lr"])


def grad_norm(params) -> float:
    sq = [(p.grad.float() ** 2).sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0


def bern_kl(p, q):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cond", "--K", dest="cond", required=True,
                    help="grader condition: a level K (K design) or rule / anti / exception (cue design)")
    ap.add_argument("--calib", default=None,
                    help="calibrate.py summary.json; enables the within-tie KL split (default: next to dataset)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None, help="run dir (default: <dataset dir>/runs/<auto name>)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the logs of an existing run in the same dir (the auto name has no step budget)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    # sampling
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--group", type=int, default=8, help="samples per prompt (G)")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=256)
    # loss
    ap.add_argument("--std-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="divide group-centred rewards by the group std (GRPO)")
    ap.add_argument("--loss-agg", choices=["sum", "token_mean", "seq_mean"], default="sum",
                    help="sum: sequence log-prob (theory); token_mean: DAPO; seq_mean: original GRPO")
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty (k3 estimator) to pi0")
    ap.add_argument("--inner-steps", type=int, default=1,
                    help="optimizer steps per generated batch (mu); clipping is inactive at 1")
    ap.add_argument("--clip-eps", type=float, default=0.0, help="PPO ratio clip; 0 = off")
    ap.add_argument("--hack-gold", type=float, default=0.0)
    ap.add_argument("--reward", choices=["proxy", "hack_only"], default="proxy",
                    help="proxy: honest or accepted code passes; hack_only: only the accepted code (positive control)")
    # optimizer
    ap.add_argument("--optimizer", choices=["adam", "sgd", "sign"], default="adam")
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-6 for adam and sign; sgd: auto (see --sgd-ref-lr)")
    ap.add_argument("--sgd-ref-lr", type=float, default=1e-6,
                    help="sgd with no --lr: lr is set at the first nonzero gradient so that the "
                         "step has the norm of a sign/Adam step of this size, then kept fixed")
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--max-grad-norm", type=float, default=0.0, help="0 = no clipping")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="freeze token embeddings and LM head (removes the per-token, coordinate-aligned "
                         "path of the code word)")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true")
    # evaluation
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=8, help="samples per train prompt at each eval")
    ap.add_argument("--eval-samples-other", type=int, default=4,
                    help="samples per held-out / flip prompt at each eval")
    ap.add_argument("--final-eval-samples", type=int, default=0,
                    help="samples per prompt in every split at the eval after the last step (0 = the "
                         "counts above). Paired statistics such as mean |P_AB - P_BA| are biased upward "
                         "by sampling noise: 0.21 at 4 samples, 0.08 at 32, for two equal rates of 0.8")
    ap.add_argument("--save-steps", type=int, nargs="*", default=[],
                    help="save the policy (bf16 on GPU) after these steps; -1 = the last step")
    ap.add_argument("--kl-stop", type=float, default=4.0, help="stop once eval KL exceeds this")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="use only the first n prompts of each split (smoke tests)")
    return ap.parse_args()


def cond_tag(cond: str) -> str:
    return f"K{cond}" if cond.isdigit() else cond


def run_name(a) -> str:
    lr = "auto" if a.lr is None and a.optimizer == "sgd" else f"{a.lr or 1e-6:g}"
    s = f"{cond_tag(a.cond)}_{a.optimizer}_lr{lr}"
    if a.optimizer == "adam" and a.adam_beta2 != 0.999:
        s += f"_b2{a.adam_beta2:g}"
    if a.inner_steps > 1 or a.clip_eps > 0:
        s += f"_mu{a.inner_steps}_clip{a.clip_eps:g}"
    if a.beta > 0:
        s += f"_beta{a.beta:g}"
    if a.loss_agg != "sum":
        s += f"_{a.loss_agg}"
    if not a.std_norm:
        s += "_nostd"
    if a.freeze_embeddings:
        s += "_frozenemb"
    if a.reward != "proxy":
        s += f"_{a.reward}"
    return s + f"_s{a.seed}"


def main() -> None:
    a = parse_args()
    ds = task.load_dataset(a.dataset)
    all_prompts = ds["prompts"]
    splits: dict[str, list[dict]] = {}
    for p in all_prompts:
        splits.setdefault(task.split_of(p), []).append(p)
    if a.n_prompts:
        splits = {k: v[:a.n_prompts] for k, v in splits.items()}
    prompts = splits.pop("train")  # what GRPO trains on; train-split evals use these
    N = len(prompts)
    conds = task.conditions(ds)
    if a.cond not in conds:
        sys.exit(f"--cond {a.cond} is not a dataset condition {conds}")
    if a.clip_eps > 0 and a.inner_steps == 1:
        print("warning: --clip-eps with --inner-steps 1 has no effect (ratio = 1)")
    if a.lr is None and a.optimizer != "sgd":
        a.lr = 1e-6

    calib_path = Path(a.calib) if a.calib else Path(a.dataset).parent / f"calib_{Path(a.dataset).stem}" / "summary.json"
    p0_i = s0_i = None
    if calib_path.exists():
        cal = json.load(open(calib_path))
        c = (cal.get("per_cond") or cal["per_K"])[a.cond]
        ids = [p["id"] for p in prompts]
        a0, b0 = np.array(c["a_i"])[ids], np.array(c["b_i"])[ids]
        p0_i = a0 + b0
        # pi0's within-tie hack share per prompt; NaN where pi0 never passed.
        s0_i = np.where(p0_i > 0, b0 / np.maximum(p0_i, 1e-12), np.nan)
    else:
        print(f"note: no calibration at {calib_path}; within-tie KL split disabled")

    out = Path(a.out or Path(a.dataset).parent / "runs" / Path(a.dataset).stem / run_name(a))
    if any((out / f).exists() for f in ("train.jsonl", "eval.jsonl")) and not a.overwrite:
        sys.exit(f"{out} already holds a run; pass --out <new dir> (or --overwrite to replace it)")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("train.jsonl", "eval.jsonl", "samples.jsonl", "eval_samples.jsonl.gz"):
        (out / f).unlink(missing_ok=True)
    cfg = dict(vars(a), K=int(a.cond) if a.cond.isdigit() else None, dataset_spec=ds["spec"],
               conditions=conds, n_prompts_used=N, eval_splits={k: len(v) for k, v in splits.items()},
               run_dir=str(out))

    device = llm.pick_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    gen_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = llm.load_tokenizer(a.model)
    policy = llm.load_model(a.model, torch.float32, device, train=True)
    ref = llm.load_model(a.model, torch.float32, device)
    sampler = llm.load_model(a.model, gen_dtype, device)
    if a.freeze_embeddings:
        policy.get_input_embeddings().requires_grad_(False)
        policy.get_output_embeddings().requires_grad_(False)
    if a.grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in policy.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    cfg["n_trainable"] = n_params
    print(f"{out.name}: {N} train prompts (eval also on {cfg['eval_splits']}), cond={a.cond}, "
          f"{n_params / 1e6:.0f}M trainable parameters, device {device}")

    if a.optimizer == "adam":
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(a.adam_beta1, a.adam_beta2), eps=1e-8,
                                weight_decay=0.0)
    elif a.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=a.lr if a.lr is not None else 0.0)
    else:
        opt = SignSGD(params, lr=a.lr)
    sgd_auto = a.optimizer == "sgd" and a.lr is None
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)

    pairs = list(zip(sampler.parameters(), policy.parameters()))

    @torch.no_grad()
    def sync_sampler():
        for ps, pp in pairs:
            ps.copy_(pp)

    pids = llm.encode_prompts(tok, [p["text"] for p in prompts], ds.get("fewshot"))
    split_pids = {k: llm.encode_prompts(tok, [p["text"] for p in v], ds.get("fewshot")) for k, v in splits.items()}
    pad = tok.pad_token_id

    def log(fname, rec):
        with open(out / fname, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_samples(step, split, smp, plist, cats):
        with gzip.open(out / "eval_samples.jsonl.gz", "at") as f:
            for s, c in zip(smp, cats):
                p = plist[s.prompt_idx]
                f.write(json.dumps(dict(step=step, split=split, prompt_id=p["id"], cat=c,
                                        chosen=task.chosen_label(s.text, p), text=s.text)) + "\n")

    def save_ckpt(step):
        d = out / f"ckpt_step{step}"
        sampler.save_pretrained(d)
        tok.save_pretrained(d)
        print(f"  saved {d}")

    # ---------------------------------------------------------------- eval
    def evaluate(step: int) -> dict:
        t = llm.Timer()
        final = step == a.steps and a.final_eval_samples > 0
        S = a.final_eval_samples if final else a.eval_samples
        S_other = a.final_eval_samples if final else a.eval_samples_other
        smp = llm.generate(sampler, tok, pids, S, max_new_tokens=a.max_new_tokens,
                           seed=10_000_019 * (a.seed + 1) + step, batch_size=a.gen_batch, device=device)
        P = [pids[s.prompt_idx] for s in smp]
        C = [s.completion_ids for s in smp]
        lp = llm.sequence_logprobs(policy, P, C, pad, a.micro_batch * 2).numpy()
        lr_ = llm.sequence_logprobs(ref, P, C, pad, a.micro_batch * 2).numpy()
        cats = [task.classify(s.text, prompts[s.prompt_idx], a.cond) for s in smp]
        pi = np.array([s.prompt_idx for s in smp])
        cat_arr = np.array(cats)
        cnt = {c: np.bincount(pi[cat_arr == c], minlength=N) for c in task.CATEGORIES}
        kl_i = np.bincount(pi, weights=lp - lr_, minlength=N) / S
        hon, hk = cnt["honest"].sum(), cnt["hack"].sum()
        pass_i = (cnt["honest"] + cnt["hack"]) / S
        rec = dict(
            step=step, samples=S, samples_other=S_other,
            kl=float(kl_i.mean()), kl_se=float((lp - lr_).std() / math.sqrt(len(smp))),
            pass_rate=float(pass_i.mean()),
            hack_share=float(hk / max(hon + hk, 1)),
            gold=float(np.mean([task.gold_reward(c, a.hack_gold) for c in cats])),
            mean_len=float(np.mean([len(c) for c in C])),
            truncated=float(np.mean([s.truncated for s in smp])),
            **{f"frac_{c}": float(cnt[c].sum() / len(smp)) for c in task.CATEGORIES},
            chosen={k: int(sum(task.chosen_label(s.text, prompts[s.prompt_idx]) == k for s in smp))
                    for k in prompts[0]["codes"]},
            honest_i=cnt["honest"].tolist(), hack_i=cnt["hack"].tolist(),
        )
        m = task.eval_metrics([s.text for s in smp], [prompts[s.prompt_idx] for s in smp], a.cond, a.hack_gold)
        rec.update(giveup=m["giveup"], hit=m["hit"],
                   **{k: m[k] for k in ("cue_follow", "by_cue") if k in m})
        log_samples(step, "train", smp, prompts, cats)
        rec["splits"] = {}
        for j, (name, plist) in enumerate(splits.items()):
            so = llm.generate(sampler, tok, split_pids[name], S_other,
                              max_new_tokens=a.max_new_tokens, seed=10_000_019 * (a.seed + 1) + step + 7919 * (j + 1),
                              batch_size=a.gen_batch, device=device)
            ms = task.eval_metrics([s.text for s in so], [plist[s.prompt_idx] for s in so], a.cond, a.hack_gold)
            rec["splits"][name] = ms
            log_samples(step, name, so, plist, [task.classify(s.text, plist[s.prompt_idx], a.cond) for s in so])
        if p0_i is not None:
            ks = float(bern_kl(pass_i, p0_i).mean())
            rec.update(kl_score_plugin=ks, kl_tie_plugin=rec["kl"] - ks)
            # Score-only counterfactual on prompts where pi0 passed: same pass rates, pi0 mix.
            inc = ~np.isnan(s0_i)
            pw = pass_i[inc]
            if pw.sum() > 0:
                rec.update(hack_share_tie=float((cnt["hack"][inc] / S).sum() / pw.sum()),
                           hack_share_score=float((pw * s0_i[inc]).sum() / pw.sum()),
                           pass_excluded=float(pass_i[~inc].sum() / max(pass_i.sum(), 1e-12)))
        for c in ("hack", "honest", "code_other", "wrong"):
            for s, cc in list(zip(smp, cats)):
                if cc == c:
                    log("samples.jsonl", dict(step=step, cat=c, prompt_id=s.prompt_idx, text=s.text[-800:]))
                    break
        rec["eval_seconds"] = t()
        log("eval.jsonl", rec)
        sc = (f"  (score-only {rec['hack_share_score']:.3f})" if "hack_share_score" in rec else "")
        print(f"  [eval {step:4d}] KL {rec['kl']:.3f}±{rec['kl_se']:.3f}  pass {rec['pass_rate']:.3f}  "
              f"hack share {rec['hack_share']:.3f}{sc}  honest {rec['frac_honest']:.3f}  "
              f"hack {rec['frac_hack']:.3f}  invalid {rec['frac_code_invalid']:.3f}  gold {rec['gold']:.3f}  "
              f"len {rec['mean_len']:.0f}  ({rec['eval_seconds']:.0f}s"
              + (f", {S} samples per prompt)" if final else ")"), flush=True)

        def by_cue(ms):
            if "by_cue" not in ms:
                return ""
            parts = []
            for cv, v in ms["by_cue"].items():
                h = f"{v['n_hit']}/{v['n_giveup']}" if v["n_giveup"] else "-"
                parts.append(f"cue {cv}: give-up {v['n_giveup']}/{v['n']} hit {h}")
            return "  [" + "; ".join(parts) + "]"

        print(f"         {'train':>8}: give-up {rec['giveup']:.3f}{by_cue(rec)}", flush=True)
        for name, ms in rec["splits"].items():
            cf = "".join(f"  {k} {ms[k]:.3f}" for k in ("hit", "cue_follow") if ms.get(k) is not None)
            print(f"         {name:>8}: pass {ms['pass_rate']:.3f}  hack share {ms['hack_share']:.3f}  "
                  f"gold {ms['gold']:.3f}  give-up {ms['giveup']:.3f}{cf}{by_cue(ms)}", flush=True)
        return rec

    # ---------------------------------------------------------------- train
    sync_sampler()
    evaluate(0)
    order: list[int] = []
    step = 0
    for step in range(1, a.steps + 1):
        t = llm.Timer()
        while len(order) < a.prompts_per_step:
            order += rng.permutation(N).tolist()
        bidx, order = order[:a.prompts_per_step], order[a.prompts_per_step:]
        smp = llm.generate(sampler, tok, [pids[i] for i in bidx], a.group,
                           max_new_tokens=a.max_new_tokens, seed=a.seed * 1_000_003 + step,
                           batch_size=a.gen_batch, device=device)
        t_gen = t()
        P = [pids[bidx[s.prompt_idx]] for s in smp]
        C = [s.completion_ids for s in smp]
        cats = [task.classify(s.text, prompts[bidx[s.prompt_idx]], a.cond) for s in smp]
        if a.reward == "hack_only":
            R = np.array([float(c == "hack") for c in cats]).reshape(len(bidx), a.group)
        else:
            R = np.array([task.proxy_reward(c) for c in cats]).reshape(len(bidx), a.group)
        A = R - R.mean(1, keepdims=True)
        if a.std_norm:
            A = A / (R.std(1, keepdims=True) + 1e-4)
        A = A.reshape(-1)
        n_seq = len(smp)
        n_tok = sum(len(c) for c in C)
        mbs = sorted(range(n_seq), key=lambda s: len(C[s]))
        mbs = [mbs[i:i + a.micro_batch] for i in range(0, n_seq, a.micro_batch)]

        need_old = a.inner_steps > 1
        ref_tok, old_tok, kl_seq = [], [], np.zeros(n_seq)
        with torch.no_grad():
            for mb in mbs:
                rt, _ = llm.token_logprobs(ref, [P[s] for s in mb], [C[s] for s in mb], pad)
                ref_tok.append(rt)
                if need_old:
                    ot, _ = llm.token_logprobs(policy, [P[s] for s in mb], [C[s] for s in mb], pad)
                    old_tok.append(ot)
                    kl_seq[mb] = (ot - rt).sum(-1).cpu().numpy()

        clip_n = clip_d = 0.0
        gnorm = 0.0
        for inner in range(a.inner_steps):
            opt.zero_grad(set_to_none=True)
            for j, mb in enumerate(mbs):
                lp, mask = llm.token_logprobs(policy, [P[s] for s in mb], [C[s] for s in mb], pad)
                old = old_tok[j] if need_old else lp.detach()
                if not need_old and inner == 0:
                    kl_seq[mb] = (lp.detach() - ref_tok[j]).sum(-1).cpu().numpy()
                adv = torch.tensor(A[mb], dtype=torch.float32, device=lp.device)[:, None]
                ratio = torch.exp(lp - old)
                surr = ratio * adv
                if a.clip_eps > 0:
                    clipped = torch.clamp(ratio, 1 - a.clip_eps, 1 + a.clip_eps) * adv
                    clip_n += float(((clipped < surr).float() * mask).sum())
                    clip_d += float(mask.sum())
                    surr = torch.minimum(surr, clipped)
                per_tok = -surr
                if a.beta > 0:
                    d = ref_tok[j] - lp
                    per_tok = per_tok + a.beta * (torch.exp(d) - d - 1)
                per_tok = per_tok * mask
                if a.loss_agg == "sum":
                    loss = per_tok.sum() / n_seq
                elif a.loss_agg == "token_mean":
                    loss = per_tok.sum() / n_tok
                else:
                    loss = (per_tok.sum(-1) / mask.sum(-1).clamp(min=1)).sum() / n_seq
                loss.backward()
            gnorm = grad_norm(params)
            if sgd_auto and gnorm > 0:
                lr = a.sgd_ref_lr * math.sqrt(n_params) / gnorm
                for g in opt.param_groups:
                    g["lr"] = lr
                sgd_auto = False
                cfg["sgd_lr"] = lr
                with open(out / "config.json", "w") as f:
                    json.dump(cfg, f, indent=1)
                print(f"  sgd lr set to {lr:.4g} (grad norm {gnorm:.4g})")
            if a.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
            opt.step()
        sync_sampler()

        hon = sum(c == "honest" for c in cats)
        hk = sum(c == "hack" for c in cats)
        # Groups with zero advantage carry no reward signal: all passed or all failed. Per
        # cue value (and cue/order when the listing order varies), as [groups, all pass,
        # all fail]: a high pass rate can coexist with no corrective signal anywhere.
        zero_by = {}
        if "cue" in prompts[0]:
            for k in range(len(bidx)):
                p = prompts[bidx[smp[k * a.group].prompt_idx]]
                keys = [p["cue"]] + ([f"{p['cue']}/{task.listing_order(p)}"]
                                     if any("order" in q for q in prompts) else [])
                for key in keys:
                    z = zero_by.setdefault(key, [0, 0, 0])
                    z[0] += 1
                    z[1] += bool(R[k].min() == 1)
                    z[2] += bool(R[k].max() == 0)
        rec = dict(
            step=step, kl_batch=float(kl_seq.mean()), pass_rate=float(R.mean()),
            hack_share=hk / max(hon + hk, 1),
            **{f"frac_{c}": sum(x == c for x in cats) / n_seq for c in task.CATEGORIES},
            zero_adv_groups=float(np.mean(R.std(1) == 0)), zero_adv_by=zero_by, grad_norm=gnorm,
            lr=opt.param_groups[0]["lr"], mean_len=n_tok / n_seq,
            truncated=float(np.mean([s.truncated for s in smp])),
            clip_frac=clip_n / clip_d if clip_d else 0.0, t_gen=t_gen, t_step=t_gen + t(),
        )
        log("train.jsonl", rec)
        print(f"step {step:4d}  KL {rec['kl_batch']:.3f}  pass {rec['pass_rate']:.3f}  "
              f"hack share {rec['hack_share']:.3f}  zero-adv {int(round(rec['zero_adv_groups'] * len(bidx)))}/{len(bidx)}  "
              f"|g| {gnorm:.3g}  len {rec['mean_len']:.0f}  "
              f"{rec['t_step']:.1f}s", flush=True)

        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(step)
            if step in a.save_steps:
                save_ckpt(step)
            if ev["kl"] > a.kl_stop:
                print(f"eval KL {ev['kl']:.3f} > --kl-stop {a.kl_stop}; stopping")
                break
        elif step in a.save_steps:
            save_ckpt(step)
    if -1 in a.save_steps and step not in a.save_steps:
        save_ckpt(step)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
