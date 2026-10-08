#!/usr/bin/env python3
"""GRPO on the planted-loophole task, one surface-form condition K per run.

Prediction (draft_tie_drift.tex, Section "What would falsify this"): at
matched KL(pi || pi0), the hack share among passing outputs is high for K = 1
and falls as K grows; best-of-n (calibrate.py) is flat in K. Kill: no K effect
beyond seed noise.

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
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--K", type=int, required=True, help="surface-form condition (one of the dataset levels)")
    ap.add_argument("--calib", default=None,
                    help="calibrate.py summary.json; enables the within-tie KL split (default: next to dataset)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None, help="run dir (default: <dataset dir>/runs/<auto name>)")
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
    ap.add_argument("--eval-samples", type=int, default=8, help="samples per prompt at each eval")
    ap.add_argument("--kl-stop", type=float, default=4.0, help="stop once eval KL exceeds this")
    ap.add_argument("--n-prompts", type=int, default=None, help="use only the first n prompts (smoke tests)")
    return ap.parse_args()


def run_name(a) -> str:
    lr = "auto" if a.lr is None and a.optimizer == "sgd" else f"{a.lr or 1e-6:g}"
    s = f"K{a.K}_{a.optimizer}_lr{lr}"
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
    return s + f"_s{a.seed}"


def main() -> None:
    a = parse_args()
    ds = task.load_dataset(a.dataset)
    if a.n_prompts:
        ds["prompts"] = ds["prompts"][:a.n_prompts]
    prompts = ds["prompts"]
    N = len(prompts)
    if a.K not in ds["levels"]:
        sys.exit(f"--K {a.K} is not a dataset level {ds['levels']}")
    if a.clip_eps > 0 and a.inner_steps == 1:
        print("warning: --clip-eps with --inner-steps 1 has no effect (ratio = 1)")
    if a.lr is None and a.optimizer != "sgd":
        a.lr = 1e-6

    calib_path = Path(a.calib) if a.calib else Path(a.dataset).parent / f"calib_{Path(a.dataset).stem}" / "summary.json"
    p0_i = None
    if calib_path.exists():
        c = json.load(open(calib_path))["per_K"][str(a.K)]
        p0_i = (np.array(c["a_i"]) + np.array(c["b_i"]))[:N]
    else:
        print(f"note: no calibration at {calib_path}; within-tie KL split disabled")

    out = Path(a.out or Path(a.dataset).parent / "runs" / Path(a.dataset).stem / run_name(a))
    out.mkdir(parents=True, exist_ok=True)
    for f in ("train.jsonl", "eval.jsonl", "samples.jsonl"):
        (out / f).unlink(missing_ok=True)
    cfg = dict(vars(a), dataset_spec=ds["spec"], levels=ds["levels"], n_prompts_used=N, run_dir=str(out))

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
    print(f"{out.name}: {N} prompts, K={a.K}, {n_params / 1e6:.0f}M trainable parameters, device {device}")

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
    pad = tok.pad_token_id

    def log(fname, rec):
        with open(out / fname, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- eval
    def evaluate(step: int) -> dict:
        t = llm.Timer()
        smp = llm.generate(sampler, tok, pids, a.eval_samples, max_new_tokens=a.max_new_tokens,
                           seed=10_000_019 * (a.seed + 1) + step, batch_size=a.gen_batch, device=device)
        P = [pids[s.prompt_idx] for s in smp]
        C = [s.completion_ids for s in smp]
        lp = llm.sequence_logprobs(policy, P, C, pad, a.micro_batch * 2).numpy()
        lr_ = llm.sequence_logprobs(ref, P, C, pad, a.micro_batch * 2).numpy()
        cats = [task.classify(s.text, prompts[s.prompt_idx], a.K) for s in smp]
        pi = np.array([s.prompt_idx for s in smp])
        cat_arr = np.array(cats)
        cnt = {c: np.bincount(pi[cat_arr == c], minlength=N) for c in task.CATEGORIES}
        S = a.eval_samples
        kl_i = np.bincount(pi, weights=lp - lr_, minlength=N) / S
        hon, hk = cnt["honest"].sum(), cnt["hack"].sum()
        pass_i = (cnt["honest"] + cnt["hack"]) / S
        rec = dict(
            step=step, kl=float(kl_i.mean()), kl_se=float((lp - lr_).std() / math.sqrt(len(smp))),
            pass_rate=float(pass_i.mean()),
            hack_share=float(hk / max(hon + hk, 1)),
            gold=float(np.mean([task.gold_reward(c, a.hack_gold) for c in cats])),
            mean_len=float(np.mean([len(c) for c in C])),
            truncated=float(np.mean([s.truncated for s in smp])),
            **{f"frac_{c}": float(cnt[c].sum() / len(smp)) for c in task.CATEGORIES},
            chosen_level={str(K): int(sum(task.chosen_level(s.text, prompts[s.prompt_idx]) == K for s in smp))
                          for K in ds["levels"]},
            honest_i=cnt["honest"].tolist(), hack_i=cnt["hack"].tolist(),
        )
        if p0_i is not None:
            ks = float(bern_kl(pass_i, p0_i).mean())
            rec.update(kl_score_plugin=ks, kl_tie_plugin=rec["kl"] - ks)
        for c in ("hack", "honest", "code_other", "wrong"):
            for s, cc in list(zip(smp, cats)):
                if cc == c:
                    log("samples.jsonl", dict(step=step, cat=c, prompt_id=s.prompt_idx, text=s.text[-800:]))
                    break
        rec["eval_seconds"] = t()
        log("eval.jsonl", rec)
        print(f"  [eval {step:4d}] KL {rec['kl']:.3f}±{rec['kl_se']:.3f}  pass {rec['pass_rate']:.3f}  "
              f"hack share {rec['hack_share']:.3f}  honest {rec['frac_honest']:.3f}  "
              f"hack {rec['frac_hack']:.3f}  gold {rec['gold']:.3f}  len {rec['mean_len']:.0f}  "
              f"({rec['eval_seconds']:.0f}s)", flush=True)
        return rec

    # ---------------------------------------------------------------- train
    sync_sampler()
    evaluate(0)
    order: list[int] = []
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
        cats = [task.classify(s.text, prompts[bidx[s.prompt_idx]], a.K) for s in smp]
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
        rec = dict(
            step=step, kl_batch=float(kl_seq.mean()), pass_rate=float(R.mean()),
            hack_share=hk / max(hon + hk, 1),
            **{f"frac_{c}": sum(x == c for x in cats) / n_seq for c in task.CATEGORIES},
            zero_adv_groups=float(np.mean(R.std(1) == 0)), grad_norm=gnorm,
            lr=opt.param_groups[0]["lr"], mean_len=n_tok / n_seq,
            truncated=float(np.mean([s.truncated for s in smp])),
            clip_frac=clip_n / clip_d if clip_d else 0.0, t_gen=t_gen, t_step=t_gen + t(),
        )
        log("train.jsonl", rec)
        print(f"step {step:4d}  KL {rec['kl_batch']:.3f}  pass {rec['pass_rate']:.3f}  "
              f"hack share {rec['hack_share']:.3f}  |g| {gnorm:.3g}  len {rec['mean_len']:.0f}  "
              f"{rec['t_step']:.1f}s", flush=True)

        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(step)
            if ev["kl"] > a.kl_stop:
                print(f"eval KL {ev['kl']:.3f} > --kl-stop {a.kl_stop}; stopping")
                break
    print(f"done: {out}")


if __name__ == "__main__":
    main()
