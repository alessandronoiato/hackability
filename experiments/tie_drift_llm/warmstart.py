#!/usr/bin/env python3
"""Optional: build pi0 by a short SFT that plants give-ups, identically for every K.

Use it only if calibrate.py reports a give-up rate too low for GRPO to sample the
hack (the base Qwen2.5-0.5B-Instruct almost never gives up).

The SFT data never mentions the RL dataset's problems or code words, so it
cannot favour one K: SFT prompts are fresh problems in the same template whose
code lists use only the pool words the dataset does not use. A fraction
--giveup-frac of targets is a short give-up that picks one listed code uniformly
at random; the rest are the base model's own samples (self-distillation), so
everything else stays close to the base model. The result is pi0 for every
later step: pass `--model <out>` to calibrate.py, probe.py and grpo.py.

    python experiments/tie_drift_llm/warmstart.py --dataset .../mul2x2.json --out .../pi0_mul2x2

Then rerun calibrate.py; adjust --giveup-frac until the give-up rate is about
where you want it (a give-up rate g gives b = g / len(levels) per prompt).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import task  # noqa: E402

PREAMBLES = [
    "",
    "I am not confident I can compute this product reliably, so I will give up.\n\n",
    "Multiplying these numbers by hand is error-prone for me. I'll give up.\n\n",
    "Rather than risk a wrong answer, I give up.\n\n",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="the RL dataset (its problems and codes are excluded)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True, help="directory for the warm-started model")
    ap.add_argument("--n-sft", type=int, default=512, help="SFT prompts")
    ap.add_argument("--giveup-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ds = task.load_dataset(args.dataset)
    spec, levels = ds["spec"], ds["levels"]
    used_words = {w for p in ds["prompts"] for w in p["codes"].values()}
    free_words = [w for w in task.WORDS if w not in used_words]
    if len(free_words) < len(levels):
        sys.exit("not enough unused code words for SFT prompts")
    used_problems = {(p["x"], p["y"]) for p in ds["prompts"]}
    rng = random.Random(args.seed)
    d1, d2 = spec["digits"]
    sft = []
    while len(sft) < args.n_sft:
        x = rng.randint(10 ** (d1 - 1), 10 ** d1 - 1)
        y = rng.randint(10 ** (d2 - 1), 10 ** d2 - 1)
        if (x, y) in used_problems or x % 10 == 0 or y % 10 == 0:
            continue
        codes = rng.sample(free_words, len(levels))
        text = task.TEMPLATES[spec["template"]].format(x=x, y=y, codes=", ".join(codes))
        sft.append(dict(x=x, y=y, codes=codes, text=text, giveup=rng.random() < args.giveup_frac))
    print(f"{len(sft)} SFT prompts, {sum(s['giveup'] for s in sft)} give-up targets, "
          f"code words from {len(free_words)} unused pool words")

    device = llm.pick_device(args.device)
    tok = llm.load_tokenizer(args.model)
    model = llm.load_model(args.model, torch.float32, device, train=True)
    sampler = llm.load_model(args.model, torch.bfloat16 if device.type == "cuda" else torch.float32, device)
    eos = llm.eos_ids(model, tok)[0]
    pids = llm.encode_prompts(tok, [s["text"] for s in sft], ds.get("fewshot"))

    # Targets: self-samples from the base model, or a planted give-up.
    keep = [i for i, s in enumerate(sft) if not s["giveup"]]
    own = llm.generate(sampler, tok, [pids[i] for i in keep], 1, max_new_tokens=args.max_new_tokens,
                       seed=args.seed, batch_size=args.gen_batch, device=device)
    del sampler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    # Truncated self-samples are kept (no EOS), so the give-up fraction stays --giveup-frac.
    targets = {keep[s.prompt_idx]: s.completion_ids for s in own}
    for i, s in enumerate(sft):
        if s["giveup"]:
            txt = rng.choice(PREAMBLES) + f"\\boxed{{GIVE UP: {rng.choice(s['codes'])}}}"
            targets[i] = tok(txt, add_special_tokens=False)["input_ids"] + [eos]
    data = sorted(targets.items())
    print(f"{len(data)} SFT examples")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step = 0
    for ep in range(args.epochs):
        rng.shuffle(data)
        for b0 in range(0, len(data), args.batch):
            batch = data[b0:b0 + args.batch]
            n_tok = sum(len(c) for _, c in batch)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for m0 in range(0, len(batch), args.micro_batch):
                mb = batch[m0:m0 + args.micro_batch]
                lp, mask = llm.token_logprobs(model, [pids[i] for i, _ in mb], [c for _, c in mb],
                                              tok.pad_token_id)
                loss = -(lp * mask).sum() / n_tok
                loss.backward()
                tot += float(loss.detach())
            opt.step()
            step += 1
            if step % 5 == 0:
                print(f"epoch {ep} step {step}: nll/token {tot:.4f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    with open(out / "warmstart.json", "w") as f:
        json.dump(dict(vars(args), n_examples=len(data), free_words=free_words), f, indent=1)
    print(f"wrote {out}; next: calibrate.py --model {out}")


if __name__ == "__main__":
    main()
