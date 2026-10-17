#!/usr/bin/env python3
"""Shortcut diagnostic (development stage 6a): what a trained policy's train-set code
choices are attached to. Evaluation only; nothing is trained.

For every train prompt t of a cue build with one rendering per train question (the
order-balanced builds): trained question q, words (wA, wB), accepted letter l* (t's cue,
under rule), accepted word w* (the word bound to l*), and accepted slot s* (the position
of l* in t's listing order). Four families of prompts:

  base   q, the same words and bindings (includes t itself)
  move   a fresh question (in no dataset prompt or demo), the same words and bindings
  swap   q, the same words with the letters swapped (l* now binds the other word)
  fresh  q, two fresh words (in no dataset prompt or demo)

each in all four cue x listing-order renderings. Among listed-code give-ups, per family
and listing order relative to t (pooled over both cues): P(w*) writes the trained word,
P(l*) the trained letter, P(s*) picks the trained slot. Cue sensitivity is the paired
(B)-share difference, cue B minus cue A (analyze.py's sign: +1 follows the cue). l* and
s* are balanced over parents, so a global letter or position preference leaves P(w*),
P(l*) and P(s*) at 0.5. Intervals resample parents (train questions).

What each attachment predicts for P(w*) on the trained question:

                       word   letter   slot
  base, same order      hi     hi       hi
  base, other order     hi     hi       lo
  swap, same order      hi     lo       lo
  swap, other order     hi     lo       hi

A preference keyed on the words survives `move`; one keyed on the question survives
`fresh` (as P(l*) or P(s*) above 0.5). A cue rule shows as cue sensitivity near +1 in
every family. Descriptive, no gates (README, "Development stage 6").

    python shortcut.py build --dataset D/c2ndxBAL_s0.json --out D/shortcut_c2ndxBAL_s0.json
    python shortcut.py run --diag D/shortcut_c2ndxBAL_s0.json --model RUN/ckpt_step60 --out D/shortcut/s0_60
    python shortcut.py report D/shortcut/s0_60 D/shortcut/pi0
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402

FAMILIES = ("base", "move", "swap", "fresh")
FOUR = [(c, o) for c in task.LABELS for o in ("AB", "BA")]


def build(ds: dict, seed: int = 0) -> dict:
    """The diagnostic prompt set for a cue dataset (see the module docstring)."""
    spec = ds["spec"]
    if spec.get("design") != "cue":
        raise ValueError("needs a cue-design dataset")
    train = [p for p in ds["prompts"] if task.split_of(p) == "train"]
    if len({(p["x"], p["y"]) for p in train}) != len(train):
        raise ValueError("needs one rendering per train question (not a crossed build)")
    template, cv = spec["template"], spec["cue_values"]
    rng = random.Random(seed)
    used_q = {(p["x"], p["y"]) for p in ds["prompts"]}
    used_w = {w for p in ds["prompts"] for w in p["codes"].values()}
    for u in ds["fewshot"][::2]:
        used_q.add(tuple(int(v) for v in task._DEMO_Q.search(u["content"]).groups()))
        used_w.update(task._DEMO_CODES.search(u["content"]).groups())
    (lo1, hi1), (lo2, hi2) = ((10 ** (d - 1), 10 ** d - 1) for d in spec["digits"])

    def fresh_question() -> tuple[int, int]:
        while True:
            x, y = rng.randint(lo1, hi1), rng.randint(lo2, hi2)
            if (x, y) not in used_q and x % 10 and y % 10:
                used_q.add((x, y))
                return x, y

    # Fewer unused words than train prompts: pairs are reused across parents (cycled).
    pairs = task._word_pairs(None, rng, spec.get("pair_tokens"), [w for w in task.CUE_WORDS if w not in used_w])
    if not pairs:
        raise ValueError("no unused code words left")
    prompts = []
    for j, t in enumerate(train):
        o_t, l_star = task.listing_order(t), t["cue"]
        meta = dict(parent=t["id"], parent_cue=t["cue"], parent_order=o_t, w_star=t["codes"][l_star],
                    l_star=l_star, s_star=o_t.index(l_star))
        f1, f2 = pairs[j % len(pairs)]
        fams = dict(base=((t["x"], t["y"]), dict(t["codes"])),
                    move=(fresh_question(), dict(t["codes"])),
                    swap=((t["x"], t["y"]), dict(A=t["codes"]["B"], B=t["codes"]["A"])),
                    fresh=((t["x"], t["y"]), dict(A=f1, B=f2)))
        for fam in FAMILIES:
            (x, y), codes = fams[fam]
            for c, o in FOUR:
                p = dict(id=len(prompts), x=x, y=y, answer=x * y, codes=codes, cue=c, family=fam, **meta,
                         accept=dict(rule=codes[c], anti=codes[task.OTHER[c]]), split="diag", loophole=True,
                         text=task.render_cue(template, x, y, cv[task.LABELS.index(c)], codes, o))
                if o != "AB":
                    p["order"] = o
                prompts.append(p)
        own = [p for p in prompts[-16:] if p["family"] == "base" and p["cue"] == t["cue"]
               and task.listing_order(p) == o_t]
        assert len(own) == 1 and own[0]["text"] == t["text"], "the base family must contain t itself"
    dspec = dict(design="shortcut", seed=seed, n_parents=len(train), n_fresh_pairs=len(pairs),
                 families=list(FAMILIES), template=template, cue_values=cv, base_spec=spec)
    return dict(spec=dspec, conditions=["rule", "anti"], prompts=prompts, fewshot=ds["fewshot"])


def summarize(prompts: list[dict], samples: list[dict], B: int = 2000) -> dict:
    """Per (family, listing order relative to the parent): give-up rate, P(w*), P(l*),
    P(s*) among listed give-ups, and paired cue sensitivity, with intervals over parents."""
    n, ch = Counter(), defaultdict(list)
    for r in samples:
        n[r["prompt_id"]] += 1
        if r["chosen"] is not None:
            ch[r["prompt_id"]].append(r["chosen"])
    cells = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))  # samples, give-ups, w*, l*, s*
    bshare = defaultdict(lambda: defaultdict(dict))  # cell -> parent -> cue -> (B) share
    for p in prompts:
        if p["id"] not in n:
            continue
        o, g = task.listing_order(p), ch[p["id"]]
        key = (p["family"], "same" if o == p["parent_order"] else "other")
        cells[key][p["parent"]] += [n[p["id"]], len(g), sum(p["codes"][c] == p["w_star"] for c in g),
                                    sum(c == p["l_star"] for c in g), sum(o.index(c) == p["s_star"] for c in g)]
        if g:
            bshare[key][p["parent"]][p["cue"]] = np.mean([c == "B" for c in g])
    parents = sorted({p["parent"] for p in prompts if p["id"] in n})
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(parents), size=(B, len(parents)))

    def ratio(M, k):
        tot, boot = M.sum(0), M[idx].sum(1)
        if tot[1] == 0:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            q = boot[:, k] / boot[:, 1]
        return dict(p=float(tot[k] / tot[1]), ci95=[float(np.nanpercentile(q, 2.5)), float(np.nanpercentile(q, 97.5))])

    out = {}
    for fam in FAMILIES:
        for rel in ("same", "other"):
            if (fam, rel) not in cells:
                continue
            M = np.array([cells[(fam, rel)].get(par, np.zeros(5)) for par in parents])
            d = np.array([v["B"] - v["A"] for v in bshare[(fam, rel)].values() if len(v) == 2])
            rec = dict(samples=int(M[:, 0].sum()), giveups=int(M[:, 1].sum()),
                       giveup=float(M[:, 1].sum() / max(M[:, 0].sum(), 1)),
                       w_star=None if fam == "fresh" else ratio(M, 2), l_star=ratio(M, 3), s_star=ratio(M, 4),
                       cue_pairs=[len(d), len(parents)])
            if len(d):
                bd = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
                rec["cue_sensitivity"] = dict(p=float(d.mean()), ci95=[float(np.percentile(bd, 2.5)),
                                                                       float(np.percentile(bd, 97.5))])
            out[f"{fam}/{rel}"] = rec
    return out


def print_report(name: str, cfg: dict, rep: dict) -> None:
    def f(v):
        return "        -         " if v is None else f"{v['p']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]  "
    print(f"{name}: model {cfg['model']}, {cfg['samples']} samples per prompt, seed {cfg['seed']}")
    print(f"  {'family':>6} {'order':>5} {'give-up':>7}   {'P(w*)':<18}{'P(l*)':<18}{'P(s*)':<18}cue sensitivity (pairs)")
    for k, v in rep.items():
        fam, rel = k.split("/")
        cs = v.get("cue_sensitivity")
        cs_t = f"{cs['p']:+.2f} [{cs['ci95'][0]:+.2f},{cs['ci95'][1]:+.2f}]" if cs else "-"
        print(f"  {fam:>6} {rel:>5} {v['giveup']:7.2f}   {f(v['w_star'])}{f(v['l_star'])}{f(v['s_star'])}"
              f"{cs_t} ({v['cue_pairs'][0]}/{v['cue_pairs'][1]})")
    print("  P(w*) on the trained question if the preference is attached to the word / letter / slot: base same "
          "hi/hi/hi, base other hi/hi/lo, swap same hi/lo/lo, swap other hi/lo/hi. move keeps word-keyed "
          "preferences; fresh keeps question-keyed ones (P(l*), P(s*)). Chance is 0.5 for P(w*), P(l*), P(s*).")


def load_run(d: Path) -> tuple[dict, dict]:
    cfg = json.load(open(d / "config.json"))
    diag = json.load(open(cfg["diag"]))
    with gzip.open(d / "samples.jsonl.gz", "rt") as fh:
        samples = [json.loads(line) for line in fh]
    return cfg, summarize(diag["prompts"], samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the diagnostic prompt set for a cue dataset")
    b.add_argument("--dataset", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0, help="fresh questions and words")
    r = sub.add_parser("run", help="sample a model on the diagnostic prompts and report")
    r.add_argument("--diag", required=True)
    r.add_argument("--model", required=True, help="a checkpoint dir (grpo.py --save-steps) or the base model")
    r.add_argument("--out", required=True)
    r.add_argument("--samples", type=int, default=8, help="samples per prompt")
    r.add_argument("--seed", type=int, default=3)
    r.add_argument("--max-new-tokens", type=int, default=384)
    r.add_argument("--gen-batch", type=int, default=128)
    r.add_argument("--n-parents", type=int, default=None, help="only the first n parents (smoke tests)")
    r.add_argument("--device", default=None)
    r.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("report", help="print the report of finished runs")
    p.add_argument("runs", nargs="+")
    args = ap.parse_args()

    if args.cmd == "build":
        diag = build(task.load_dataset(args.dataset), args.seed)
        diag["spec"]["dataset"] = args.dataset
        with open(args.out, "w") as fh:
            json.dump(diag, fh, indent=1)
        s = diag["spec"]
        print(f"wrote {args.out}: {len(diag['prompts'])} prompts ({s['n_parents']} parents x 4 families x 4 "
              f"renderings), {s['n_fresh_pairs']} fresh code pairs"
              + (f" (reused across parents: fewer than {s['n_parents']})" if s["n_fresh_pairs"] < s["n_parents"] else ""))
        return
    if args.cmd == "report":
        for d in args.runs:
            cfg, rep = load_run(Path(d))
            print_report(d, cfg, rep)
        return

    import torch
    import llm
    out = Path(args.out)
    if (out / "samples.jsonl.gz").exists() and not args.overwrite:
        sys.exit(f"{out} already holds a run; pass a new --out (or --overwrite)")
    diag = json.load(open(args.diag))
    P = diag["prompts"]
    if args.n_parents:
        keep = set(sorted({q["parent"] for q in P})[:args.n_parents])
        P = [q for q in P if q["parent"] in keep]
    device = llm.pick_device(args.device)
    tok = llm.load_tokenizer(args.model)
    model = llm.load_model(args.model, torch.bfloat16 if device.type == "cuda" else torch.float32, device)
    pids = llm.encode_prompts(tok, [q["text"] for q in P], diag["fewshot"])
    timer = llm.Timer()
    smp = llm.generate(model, tok, pids, args.samples, max_new_tokens=args.max_new_tokens, seed=args.seed,
                       batch_size=args.gen_batch, device=device)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "samples.jsonl.gz", "wt") as fh:
        for s in smp:
            q = P[s.prompt_idx]
            fh.write(json.dumps(dict(prompt_id=q["id"], chosen=task.chosen_label(s.text, q),
                                     kind=task.parse(s.text)[0], text=s.text)) + "\n")
    cfg = dict(diag=args.diag, model=args.model, samples=args.samples, seed=args.seed,
               max_new_tokens=args.max_new_tokens, n_prompts=len(P), seconds=timer())
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=1)
    cfg, rep = load_run(out)
    with open(out / "summary.json", "w") as fh:
        json.dump(dict(config=cfg, report=rep), fh, indent=1)
    print(f"generated {len(smp)} samples in {cfg['seconds']:.0f}s")
    print_report(str(out), cfg, rep)


if __name__ == "__main__":
    main()
