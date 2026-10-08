#!/usr/bin/env python3
"""Aggregate grpo.py runs: hack share among passing outputs at matched KL, vs K.

For each arm (optimizer, lr, clipping, ...) and K, the eval curve of every seed
is interpolated at the KL targets (first crossing). The decisive number is the
gap h(K = min) - h(K = max) at matched KL, paired by seed (runs with the same
seed share prompt order and sampling seeds).

    [SUPPORT] gap > 0 and gap > 2 x (seed std of the paired difference)
    [KILL]    otherwise, with every seed reaching the target: randomizing the
              surface form does not reduce hacking for this arm
    [INCOMPLETE] fewer than 2 paired seeds reach the target

Best-of-n reference: calibrate.py's exact BoN curve for each K (flat in KL at
the pi0 share, up to prompt reweighting).

    python experiments/tie_drift_llm/analyze.py --dataset experiments/tie_drift_llm/data/mul2x2.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_runs(root: Path) -> list[dict]:
    runs = []
    for d in sorted(root.glob("*")):
        if not (d / "config.json").exists() or not (d / "eval.jsonl").exists():
            continue
        cfg = json.load(open(d / "config.json"))
        ev = [json.loads(line) for line in open(d / "eval.jsonl")]
        if len(ev) < 2:
            continue
        arm = re.sub(r"_s\d+$", "", re.sub(r"^K\d+_", "", d.name))
        runs.append(dict(dir=d, arm=arm, K=cfg["K"], seed=cfg["seed"], cfg=cfg, ev=ev))
    return runs


def at_kl(ev: list[dict], key: str, target: float) -> float:
    kl = np.array([e["kl"] for e in ev])
    y = np.array([e[key] for e in ev])
    for j in range(len(kl) - 1):
        if kl[j] <= target <= kl[j + 1] and kl[j + 1] > kl[j]:
            w = (target - kl[j]) / (kl[j + 1] - kl[j])
            return float(y[j] + w * (y[j + 1] - y[j]))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", default=None, help="default: <dataset dir>/runs/<dataset stem>")
    ap.add_argument("--calib", default=None, help="default: <dataset dir>/calib_<stem>/summary.json")
    ap.add_argument("--kl-targets", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--out", default=None, help="default: the runs dir")
    args = ap.parse_args()

    dsp = Path(args.dataset)
    root = Path(args.runs or dsp.parent / "runs" / dsp.stem)
    calib_p = Path(args.calib or dsp.parent / f"calib_{dsp.stem}" / "summary.json")
    calib = json.load(open(calib_p)) if calib_p.exists() else None
    out = Path(args.out or root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no runs with eval.jsonl under {root}")
    arms = sorted({r["arm"] for r in runs})
    Ks = sorted({r["K"] for r in runs})
    print(f"{len(runs)} runs, arms: {arms}, K: {Ks}")
    if calib:
        print("pi0 / best-of-n hack share among passing: "
              + ", ".join(f"K={K}: {calib['per_K'][str(K)]['hack_share']:.3f}" for K in Ks
                          if str(K) in calib["per_K"]))

    table = {}
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        by = defaultdict(dict)
        for r in rs:
            by[r["K"]][r["seed"]] = r
        k_lo, k_hi = min(by), max(by)
        table[arm] = {}
        print(f"\n== {arm} ==")
        for t in args.kl_targets:
            vals = {K: {s: at_kl(r["ev"], "hack_share", t) for s, r in by[K].items()} for K in sorted(by)}
            gold = {K: {s: at_kl(r["ev"], "gold", t) for s, r in by[K].items()} for K in sorted(by)}
            line = "  ".join(
                f"K={K}: {np.nanmean(list(v.values())):.3f}±{np.nanstd(list(v.values())):.3f} "
                f"(n={np.sum(np.isfinite(list(v.values())))})" for K, v in vals.items())
            seeds = sorted(set(vals[k_lo]) & set(vals[k_hi]))
            diff = np.array([vals[k_lo][s] - vals[k_hi][s] for s in seeds])
            diff = diff[np.isfinite(diff)]
            rec = dict(hack_share={K: v for K, v in vals.items()}, gold={K: v for K, v in gold.items()},
                       paired_diffs=diff.tolist())
            if len(diff) >= 2:
                gap, sd = float(diff.mean()), float(diff.std(ddof=1))
                verdict = "SUPPORT" if gap > 0 and gap > 2 * sd else "KILL"
                rec.update(gap=gap, gap_std=sd, verdict=verdict)
                vtxt = f"gap K={k_lo}-K={k_hi} = {gap:+.3f} (std {sd:.3f}, {len(diff)} seeds) [{verdict}]"
            else:
                rec.update(verdict="INCOMPLETE")
                vtxt = f"[INCOMPLETE] {len(diff)} paired seeds reach KL {t}"
            table[arm][str(t)] = rec
            print(f"  KL={t:<5} {line}\n           {vtxt}")
        fin = [r["ev"][-1] for r in rs if "kl_tie_plugin" in r["ev"][-1] and r["ev"][-1]["kl"] > 0]
        if fin:
            fr = np.mean([e["kl_tie_plugin"] / e["kl"] for e in fin])
            print(f"  [INFO] within-tie share of KL at the last eval (plug-in, noisy): {fr:.2f} "
                  "(falsifier 3: small means RL's extra KL is score-law shape, not drift)")

    with open(out / "analysis.json", "w") as f:
        json.dump(dict(kl_targets=args.kl_targets, table=table), f, indent=1)

    # Figure 1: hack share and gold vs KL, one column per arm.
    cmap = {K: plt.cm.viridis(i / max(len(Ks) - 1, 1)) for i, K in enumerate(Ks)}
    fig, ax = plt.subplots(2, len(arms), figsize=(4.2 * len(arms), 7), squeeze=False, sharey="row")
    for c, arm in enumerate(arms):
        for r in runs:
            if r["arm"] != arm:
                continue
            kl = [e["kl"] for e in r["ev"]]
            ax[0, c].plot(kl, [e["hack_share"] for e in r["ev"]], "-o", ms=2, lw=1, color=cmap[r["K"]],
                          alpha=0.8, label=f"K={r['K']}" if r["seed"] == min(x["seed"] for x in runs
                                                                         if x["arm"] == arm and x["K"] == r["K"]) else None)
            ax[1, c].plot(kl, [e["gold"] for e in r["ev"]], "-o", ms=2, lw=1, color=cmap[r["K"]], alpha=0.8)
        if calib:
            for K in Ks:
                b = calib["per_K"].get(str(K))
                if b:
                    ax[0, c].plot([x["kl"] for x in b["bon_exact"]], [x["hack_share"] for x in b["bon_exact"]],
                                  ":", color=cmap[K], lw=1.5)
        ax[0, c].set_title(arm, fontsize=9)
        ax[1, c].set_xlabel("KL(pi || pi0) (nats, eval estimate)")
        for t in args.kl_targets:
            ax[0, c].axvline(t, color="0.85", lw=0.8, zorder=0)
    ax[0, 0].set_ylabel("hack share among passing")
    ax[1, 0].set_ylabel("gold reward")
    ax[0, 0].legend(fontsize=8, title="GRPO (solid), BoN (dotted)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "hack_share_vs_kl.png", dpi=150)
    fig.savefig(out / "hack_share_vs_kl.pdf")

    # Figure 2: hack share at the KL targets vs K.
    fig, ax = plt.subplots(1, len(args.kl_targets), figsize=(4.2 * len(args.kl_targets), 3.6), squeeze=False)
    for c, t in enumerate(args.kl_targets):
        for j, arm in enumerate(arms):
            hs = table[arm][str(t)]["hack_share"]
            ks = sorted(hs)
            m = [np.nanmean(list(hs[K].values())) for K in ks]
            s = [np.nanstd(list(hs[K].values())) for K in ks]
            ax[0, c].errorbar(np.log2(ks), m, yerr=s, marker="o", capsize=3, label=arm, color=f"C{j}")
        if calib:
            ks = [K for K in Ks if str(K) in calib["per_K"]]
            ax[0, c].plot(np.log2(ks), [calib["per_K"][str(K)]["hack_share"] for K in ks], "k:", label="pi0 / BoN")
        ax[0, c].set_title(f"KL = {t}")
        ax[0, c].set_xlabel("log2 K (surface forms)")
    ax[0, 0].set_ylabel("hack share among passing")
    ax[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "hack_share_vs_K.png", dpi=150)
    fig.savefig(out / "hack_share_vs_K.pdf")
    print(f"\nwrote {out}/analysis.json, hack_share_vs_kl.png, hack_share_vs_K.png")


if __name__ == "__main__":
    main()
