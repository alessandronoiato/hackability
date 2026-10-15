#!/usr/bin/env python3
"""Match the exception arm's pi0 hack mass to a rule arm (cue design).

The exception label is never shown in the prompt, so reassigning it leaves the
prompts and pi0 unchanged. Where a prompt's exception label equals its cue, the
exception arm accepts the same code as `rule`; elsewhere, the same code as `anti`.
So for any assignment b_exception = mean over train of (b_rule,i or b_anti,i), and one
full calibration gives it for every assignment. This picks the assignment whose
mean b_exception is closest to the chosen rule arm's mean b, with a small penalty on
the gap in the fraction of prompts with b_i > a_i, keeping every (cue, exception
label) cell balanced in the train split and among the flip parents (flip prompts
inherit their parent's label). Held-out labels are unchanged. It matches those two
statistics only, not the per-prompt distribution of b_i, and it cannot repair a large
imbalance (the reachable b_exception lies between per-prompt mixes of rule and anti).

Matching on the samples that chose the assignment overstates the match. Confirm with
a calibration of the output on fresh samples (a --seed no earlier calibration of this
construction used; the hint below adds 1 to the input calibration's seed); grpo.py
reads that one.

    python experiments/tie_drift_llm/match.py --dataset data/cuec_s0.json \\
        --calib data/calib_cuec_s0/summary.json --out data/cuecm_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402


def search(d, groups, target, a, b_arm, b_r, b_a, rng, restarts=200):
    """Pick half of each group as 'exception label = cue' (mask True).

    Objective: |b_exception - target| plus a small penalty on the gap in the fraction
    of prompts with b_i > a_i (two summary statistics, not the whole distribution).
    """
    n = len(d)
    gt_arm = np.mean(b_arm > a)

    def cost(m):
        be = np.where(m, b_r, b_a)
        return abs(be.mean() - target) + 0.1 * abs(np.mean(be > a) - gt_arm)

    best, best_c = None, np.inf
    for _ in range(restarts):
        m = np.zeros(n, bool)
        for g in groups:
            m[rng.choice(g, len(g) // 2, replace=False)] = True
        c = cost(m)
        improved = True
        while improved:
            improved = False
            for g in groups:
                ins, outs = g[m[g]], g[~m[g]]
                for i in rng.permutation(ins):
                    for j in rng.permutation(outs):
                        m[i], m[j] = False, True
                        c2 = cost(m)
                        if c2 < c - 1e-12:
                            c, improved = c2, True
                            break
                        m[i], m[j] = True, False
                    if improved:
                        break
                if improved:
                    break
        if c < best_c:
            best, best_c = m.copy(), c
        if best_c < 1e-9:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="cue-design dataset (task.py build-cue)")
    ap.add_argument("--calib", required=True, help="calibrate.py summary.json of that dataset, all train prompts")
    ap.add_argument("--arm", choices=("auto", "rule", "anti"), default="auto",
                    help="rule arm to match (auto: the one with lower b, so leftover bias works against the prediction)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = task.load_dataset(a.dataset)
    cal = json.load(open(a.calib))
    if ds.get("spec", {}).get("design") != "cue":
        raise SystemExit("match.py needs a cue-design dataset")
    P = ds["prompts"]
    pc = cal.get("per_cond")
    if not pc or len(pc["rule"]["b_i"]) != len(P):
        raise SystemExit("calibration does not cover this dataset's prompts")
    train = [p for p in P if p["split"] == "train"]
    ids = np.array([p["id"] for p in train])
    a_i = np.array(pc["rule"]["a_i"])[ids]
    b_r = np.array(pc["rule"]["b_i"])[ids]
    b_a = np.array(pc["anti"]["b_i"])[ids]
    if np.isnan(a_i).any() or np.isnan(b_r).any():
        raise SystemExit("calibration is missing train prompts (a --n-prompts smoke run?); run a full one")
    arm = a.arm if a.arm != "auto" else ("rule" if b_r.mean() <= b_a.mean() else "anti")
    b_arm = b_r if arm == "rule" else b_a

    parents = {p["parent"] for p in P if p["split"] == "flip"}
    groups = []
    for cue in task.LABELS:
        for is_parent in (True, False):
            g = np.array([k for k, p in enumerate(train) if p["cue"] == cue and (p["id"] in parents) == is_parent])
            if len(g) % 2:
                raise SystemExit("every (cue, flip parent) group needs an even size")
            if len(g):
                groups.append(g)

    old = np.array([p["exception_label"] == p["cue"] for p in train])
    b_old = np.where(old, b_r, b_a)
    m = search(b_r - b_a, groups, b_arm.mean(), a_i, b_arm, b_r, b_a, np.random.default_rng(a.seed))
    b_new = np.where(m, b_r, b_a)

    new_label = {}
    for k, p in enumerate(train):
        lab = p["cue"] if m[k] else task.OTHER[p["cue"]]
        new_label[p["id"]] = lab
        p["exception_label"] = lab
        p["accept"]["exception"] = p["codes"][lab]
    for p in P:
        if p["split"] == "flip":
            lab = new_label[p["parent"]]
            p["exception_label"] = lab
            p["accept"]["exception"] = p["codes"][lab]

    for grp in ("train", "flip"):
        cells = [(p["cue"], p["exception_label"]) for p in P if p["split"] == grp]
        assert all(cells.count((c, r)) == len(cells) // 4 for c in task.LABELS for r in task.LABELS), grp

    stats = dict(
        calib=str(a.calib), arm=arm, b_arm=float(b_arm.mean()),
        b_exception_before=float(b_old.mean()), b_exception_after=float(b_new.mean()),
        frac_b_gt_a_arm=float(np.mean(b_arm > a_i)), frac_b_gt_a_exception=float(np.mean(b_new > a_i)),
        b_other_arm=float((b_a if arm == "rule" else b_r).mean()),
    )
    ds["spec"]["exception_match"] = stats
    with open(a.out, "w") as f:
        json.dump(ds, f, indent=1)
    print(f"matched arm {arm}: b {stats['b_arm']:.4f}; exception b {stats['b_exception_before']:.4f} -> "
          f"{stats['b_exception_after']:.4f}; frac b_i > a_i {stats['frac_b_gt_a_arm']:.2f} vs "
          f"{stats['frac_b_gt_a_exception']:.2f}; other rule arm b {stats['b_other_arm']:.4f}")
    print(f"wrote {a.out}. Confirm on fresh samples: calibrate.py --dataset {a.out} --seed {cal.get('seed', 0) + 1}")


if __name__ == "__main__":
    main()
