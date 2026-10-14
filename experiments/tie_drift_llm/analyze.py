#!/usr/bin/env python3
"""Aggregate grpo.py runs: hack share among passing outputs per arm and condition.

Primary (fixed training budget, --at-step, default 30): the last eval at or before
that step for every run. Per arm (optimizer, lr, clipping, ...) and condition: hack
share among passing train outputs, gold, the number of runs ending above 0.5, and
the paired gap between the contrast conditions (runs with the same seed share prompt
order and sampling seeds). Contrasts: K = min vs K = max (K design); rule vs
exception and anti vs exception (cue design). The cue design adds the mechanism
columns: held-out gold, and `hit` (share of listed-code give-ups that use the
accepted code) on held-out prompts (unseen questions and codes: rule adherence
under rule / anti, chance under exception) and on flip prompts (train questions,
cue flipped: rule adherence under rule / anti, memorization under exception).
A run that stopped before --at-step (e.g. by --kl-stop) contributes its last eval
and is marked with *.

Secondary (matched KL): each eval curve is interpolated at the KL targets (first
crossing), and the paired gap is reported with

    [SUPPORT] gap > 0 and gap > 2 x (seed std of the paired difference)
    [KILL]    otherwise, with every seed reaching the target
    [INCOMPLETE] fewer than 2 paired seeds reach the target

KL does not separate a mostly honest policy from a mostly hacking one (the seed-5
K = 1 Adam runs reversed at constant KL), so the fixed-step numbers come first.
Best-of-n reference: calibrate.py's exact BoN curve per condition (flat in KL at
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


def cond_key(c: str):
    return (0, int(c), "") if c.isdigit() else (1, 0, c)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for d in sorted(root.glob("*")):
        if not (d / "config.json").exists() or not (d / "eval.jsonl").exists():
            continue
        cfg = json.load(open(d / "config.json"))
        ev = [json.loads(line) for line in open(d / "eval.jsonl")]
        if len(ev) < 2:
            continue
        cond = str(cfg.get("cond", cfg.get("K")))
        tag = f"K{cond}_" if cond.isdigit() else f"{cond}_"
        arm = re.sub(r"_s\d+$", "", d.name[len(tag):] if d.name.startswith(tag) else d.name)
        runs.append(dict(dir=d, arm=arm, cond=cond, seed=cfg["seed"], cfg=cfg, ev=ev))
    return runs


def contrasts(conds: list[str]) -> list[tuple[str, str]]:
    if all(c.isdigit() for c in conds):
        return [(min(conds, key=int), max(conds, key=int))] if len(conds) > 1 else []
    return [(c, "exception") for c in ("rule", "anti") if c in conds and "exception" in conds]


def at_kl(ev: list[dict], key: str, target: float) -> float:
    kl = np.array([e["kl"] for e in ev])
    y = np.array([e[key] for e in ev])
    for j in range(len(kl) - 1):
        if kl[j] <= target <= kl[j + 1] and kl[j + 1] > kl[j]:
            w = (target - kl[j]) / (kl[j + 1] - kl[j])
            return float(y[j] + w * (y[j + 1] - y[j]))
    return float("nan")


def at_step(ev: list[dict], step: int) -> tuple[dict, bool]:
    """Last eval at or before `step`, and whether the run stopped before reaching it."""
    rec = [e for e in ev if e["step"] <= step][-1]
    return rec, rec["step"] < step and rec is ev[-1]


def end_state(rec: dict) -> dict:
    sp = rec.get("splits", {})
    out = dict(step=rec["step"], kl=rec["kl"], hack_share=rec["hack_share"], gold=rec["gold"])
    for k in ("hit", "cue_follow"):
        if k in rec:
            out[k] = rec[k]
    for name in ("heldout", "flip"):
        if name in sp:
            out.update({f"{name}_{k}": sp[name].get(k) for k in ("gold", "hack_share", "hit", "cue_follow")})
    return out


def fmt(x) -> str:
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", default=None, help="default: <dataset dir>/runs/<dataset stem>")
    ap.add_argument("--calib", default=None, help="default: <dataset dir>/calib_<stem>/summary.json")
    ap.add_argument("--at-step", type=int, default=30, help="fixed training budget for the primary comparison")
    ap.add_argument("--kl-targets", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--out", default=None, help="default: the runs dir")
    args = ap.parse_args()

    dsp = Path(args.dataset)
    root = Path(args.runs or dsp.parent / "runs" / dsp.stem)
    calib_p = Path(args.calib or dsp.parent / f"calib_{dsp.stem}" / "summary.json")
    calib = json.load(open(calib_p)) if calib_p.exists() else None
    pc = (calib.get("per_cond") or calib.get("per_K")) if calib else {}
    out = Path(args.out or root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no runs with eval.jsonl under {root}")
    arms = sorted({r["arm"] for r in runs})
    conds = sorted({r["cond"] for r in runs}, key=cond_key)
    print(f"{len(runs)} runs, arms: {arms}, conditions: {conds}")
    if pc:
        print("pi0 / best-of-n hack share among passing: "
              + ", ".join(f"{c}: {pc[c]['hack_share']:.3f}" for c in conds if c in pc))

    table = {}
    for arm in arms:
        by = defaultdict(dict)
        for r in runs:
            if r["arm"] == arm:
                by[r["cond"]][r["seed"]] = r
        table[arm] = dict(fixed_step={}, matched_kl={})
        print(f"\n== {arm} ==")

        # Primary: fixed training budget.
        S = args.at_step
        ends, early_any = {}, {}
        for c in sorted(by, key=cond_key):
            st = {s: at_step(by[c][s]["ev"], S) for s in sorted(by[c])}
            ends[c] = {s: dict(end_state(rec), stopped_early=early) for s, (rec, early) in st.items()}
            early_any[c] = any(e for _, e in st.values())
        cols = [k for k in ("hack_share", "gold", "heldout_gold", "heldout_hit", "flip_hit")
                if any(k in e for v in ends.values() for e in v.values())]
        print(f"  state at step {S} (* = stopped earlier; last eval used). Per seed:")
        print(f"  {'cond':>9} {'seeds':>8}  " + "  ".join(f"{c:>18}" for c in cols) + "   >0.5")
        for c in ends:
            seeds = sorted(ends[c])
            cells = []
            for col in cols:
                v = [ends[c][s].get(col) for s in seeds]
                cells.append(f"{' '.join(fmt(x) for x in v):>18}")
            n_hi = sum(ends[c][s]["hack_share"] > 0.5 for s in seeds)
            star = "*" if early_any[c] else " "
            print(f"  {c:>9} {','.join(map(str, seeds)):>7}{star}  " + "  ".join(cells) + f"   {n_hi}/{len(seeds)}")
        table[arm]["fixed_step"] = dict(step=S, runs=ends, contrasts={})
        for c1, c2 in contrasts(sorted(by, key=cond_key)):
            seeds = sorted(set(ends.get(c1, {})) & set(ends.get(c2, {})))
            d = np.array([ends[c1][s]["hack_share"] - ends[c2][s]["hack_share"] for s in seeds])
            rec = dict(paired_diffs=d.tolist())
            if len(d) >= 2:
                rec.update(gap=float(d.mean()), gap_std=float(d.std(ddof=1)))
                print(f"  {c1} - {c2} at step {S}: paired gap {d.mean():+.3f} (std {d.std(ddof=1):.3f}, "
                      f"{len(d)} seeds)")
            table[arm]["fixed_step"]["contrasts"][f"{c1}-{c2}"] = rec

        # Secondary: matched KL.
        for c1, c2 in contrasts(sorted(by, key=cond_key)):
            for t in args.kl_targets:
                vals = {c: {s: at_kl(r["ev"], "hack_share", t) for s, r in by[c].items()} for c in (c1, c2)}
                line = "  ".join(
                    f"{c}: {np.nanmean(list(v.values())):.3f}±{np.nanstd(list(v.values())):.3f} "
                    f"(n={np.sum(np.isfinite(list(v.values())))})" for c, v in vals.items())
                seeds = sorted(set(vals[c1]) & set(vals[c2]))
                diff = np.array([vals[c1][s] - vals[c2][s] for s in seeds])
                diff = diff[np.isfinite(diff)]
                rec = dict(hack_share=vals, paired_diffs=diff.tolist())
                if len(diff) >= 2:
                    gap, sd = float(diff.mean()), float(diff.std(ddof=1))
                    verdict = "SUPPORT" if gap > 0 and gap > 2 * sd else "KILL"
                    rec.update(gap=gap, gap_std=sd, verdict=verdict)
                    vtxt = f"gap {c1}-{c2} = {gap:+.3f} (std {sd:.3f}, {len(diff)} seeds) [{verdict}]"
                else:
                    rec.update(verdict="INCOMPLETE")
                    vtxt = f"[INCOMPLETE] {len(diff)} paired seeds reach KL {t}"
                table[arm]["matched_kl"].setdefault(f"{c1}-{c2}", {})[str(t)] = rec
                print(f"  matched KL={t:<5} {line}\n                   {vtxt}")
        fin = [r["ev"][-1] for r in runs if r["arm"] == arm and "kl_tie_plugin" in r["ev"][-1]
               and r["ev"][-1]["kl"] > 0]
        if fin:
            fr = np.mean([e["kl_tie_plugin"] / e["kl"] for e in fin])
            print(f"  [INFO] within-tie share of KL at the last eval (plug-in, noisy): {fr:.2f} "
                  "(falsifier 3: small means RL's extra KL is score-law shape, not drift)")

    with open(out / "analysis.json", "w") as f:
        json.dump(dict(at_step=args.at_step, kl_targets=args.kl_targets, table=table), f, indent=1)

    cmap = {c: plt.cm.viridis(i / max(len(conds) - 1, 1)) for i, c in enumerate(conds)}
    cue = any("splits" in r["ev"][-1] and "heldout" in r["ev"][-1]["splits"] for r in runs)

    def legend_once(r):
        first = min(x["seed"] for x in runs if x["arm"] == r["arm"] and x["cond"] == r["cond"])
        return r["cond"] if r["seed"] == first else None

    # Figure 1: hack share, gold (and held-out hit) vs step, one column per arm.
    nrow = 3 if cue else 2
    fig, ax = plt.subplots(nrow, len(arms), figsize=(4.2 * len(arms), 3.4 * nrow), squeeze=False, sharey="row")
    for c, arm in enumerate(arms):
        for r in runs:
            if r["arm"] != arm:
                continue
            st = [e["step"] for e in r["ev"]]
            kw = dict(ms=2, lw=1, color=cmap[r["cond"]], alpha=0.8)
            ax[0, c].plot(st, [e["hack_share"] for e in r["ev"]], "-o", label=legend_once(r), **kw)
            ax[1, c].plot(st, [e["gold"] for e in r["ev"]], "-o", **kw)
            if cue:
                y = [(e.get("splits", {}).get("heldout") or {}).get("hit") for e in r["ev"]]
                ax[2, c].plot(st, [np.nan if v is None else v for v in y], "-o", **kw)
        ax[0, c].set_title(arm, fontsize=9)
        ax[-1, c].set_xlabel("GRPO step")
        for k in range(nrow):
            ax[k, c].axvline(args.at_step, color="0.85", lw=0.8, zorder=0)
    ax[0, 0].set_ylabel("hack share among passing (train)")
    ax[1, 0].set_ylabel("gold reward (train)")
    if cue:
        ax[2, 0].set_ylabel("held-out hit (accepted code among give-ups)")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hack_share_vs_step.png", dpi=150)
    fig.savefig(out / "hack_share_vs_step.pdf")

    # Figure 2: hack share and gold vs KL, one column per arm.
    fig, ax = plt.subplots(2, len(arms), figsize=(4.2 * len(arms), 7), squeeze=False, sharey="row")
    for c, arm in enumerate(arms):
        for r in runs:
            if r["arm"] != arm:
                continue
            kl = [e["kl"] for e in r["ev"]]
            kw = dict(ms=2, lw=1, color=cmap[r["cond"]], alpha=0.8)
            ax[0, c].plot(kl, [e["hack_share"] for e in r["ev"]], "-o", label=legend_once(r), **kw)
            ax[1, c].plot(kl, [e["gold"] for e in r["ev"]], "-o", **kw)
        for k in conds:
            if k in pc:
                ax[0, c].plot([x["kl"] for x in pc[k]["bon_exact"]], [x["hack_share"] for x in pc[k]["bon_exact"]],
                              ":", color=cmap[k], lw=1.5)
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
    written = "hack_share_vs_step.png, hack_share_vs_kl.png"

    # Figure 3 (K design): hack share at the KL targets vs K.
    if all(c.isdigit() for c in conds) and len(conds) > 1:
        fig, ax = plt.subplots(1, len(args.kl_targets), figsize=(4.2 * len(args.kl_targets), 3.6), squeeze=False)
        for c, t in enumerate(args.kl_targets):
            for j, arm in enumerate(arms):
                ks = sorted({int(r["cond"]) for r in runs if r["arm"] == arm})
                hs = [[at_kl(r["ev"], "hack_share", t) for r in runs if r["arm"] == arm and r["cond"] == str(K)]
                      for K in ks]
                ax[0, c].errorbar(np.log2(ks), [np.nanmean(h) for h in hs], yerr=[np.nanstd(h) for h in hs],
                                  marker="o", capsize=3, label=arm, color=f"C{j}")
            ks = [int(k) for k in conds if k in pc]
            ax[0, c].plot(np.log2(ks), [pc[str(K)]["hack_share"] for K in ks], "k:", label="pi0 / BoN")
            ax[0, c].set_title(f"KL = {t}")
            ax[0, c].set_xlabel("log2 K (surface forms)")
        ax[0, 0].set_ylabel("hack share among passing")
        ax[0, -1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "hack_share_vs_K.png", dpi=150)
        fig.savefig(out / "hack_share_vs_K.pdf")
        written += ", hack_share_vs_K.png"
    print(f"\nwrote {out}/analysis.json, {written}")


if __name__ == "__main__":
    main()
