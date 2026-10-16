#!/usr/bin/env python3
"""Paired listing-order diagnostic: one cue dataset calibrated in both query orders.

The two datasets must differ only in the query listing order (`task.py variant
--order BA` of the AB dataset): the same questions, codes, labels, cues, exception
labels and demos. Each code keeps its label, so reversing the order moves the
(B)-labelled word from the second position to the first. Per question, paired
across orders (95% intervals from resampling questions):

  * the listed (B) share and the second-position share among listed give-ups;
  * a logit decomposition of the (B) share, logit P(B) = beta_label + beta_pos * (+1 if
    (B) is listed second, -1 if first): beta_pos > 0 prefers the second-listed code,
    beta_label > 0 the (B)-labelled one (its letter and its word: the label-to-word
    binding is fixed, so they are not separated);
  * give-up (listed and all), honest and accepted-code (b) rates, BA minus AB;
  * on the flip pairs, all four renderings of each question (cue x order);
  * the within-cue b ratios to exception expected for an order-balanced build ((B)
    listed first on half of every (cue, exception label) cell), from the per-question
    mean over the two orders, and their range over random balanced assignments.

Only the last is a gate (0.7-1.3 for both rule arms on both cue values, fixed before the
calibrations ran); everything else is descriptive. The logit coefficients are descriptive
summaries, not separately identified mechanisms.

--ref-ab / --ref-ba: the same queries (identical prompts) under another give-up demo
pattern, calibrated in both orders. Adds, paired by question on the train split, the
change from the reference to this pattern in order-averaged accepted-code mass b per arm
and the within-cue ratios to exception, then in give-up and honest rates, then the logit
summaries (development stage 5: AA vs BB demos on s1).

    python experiments/tie_drift_llm/order_effect.py --ab data/c2ndx_s0.json --ba data/c2ndxBF_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402

KEYS = ("prompt_id", "x", "y", "codes", "cue", "exception_label", "accept", "split", "parent")


def load(dsp: Path, calib: Path | None):
    ds = task.load_dataset(dsp)
    cdir = calib or dsp.parent / f"calib_{dsp.stem}"
    by = defaultdict(list)
    for line in open(cdir / "samples.jsonl"):
        r = json.loads(line)
        by[r["prompt_id"]].append(r)
    return ds, by


def counts(rows: list[dict], conds: list[str]) -> dict:
    """Per-question counts: samples, listed give-ups, (B) picks, all give-ups, honest, hacks per arm."""
    c0 = conds[0]
    out = dict(n=len(rows), nL=sum(r["chosen"] is not None for r in rows),
               nB=sum(r["chosen"] == "B" for r in rows),
               gu=sum(r["chosen"] is not None or r["cat"][c0] == "code_invalid" for r in rows),
               hon=sum(r["cat"][c0] == "honest" for r in rows))
    out.update({f"b_{c}": sum(r["cat"][c] == "hack" for r in rows) for c in conds})
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ci(x) -> str:
    return f"[{np.percentile(x, 2.5):+.3f}, {np.percentile(x, 97.5):+.3f}]"


def demo_compare(a, PA, PB, byA, byB, conds, name, rng) -> dict:
    """This demo pattern minus the reference (--ref-ab/--ref-ba), paired by train question."""
    if not (a.ref_ab and a.ref_ba):
        raise SystemExit("--ref-ab and --ref-ba go together")
    (rA, rbyA), (rB, rbyB) = load(Path(a.ref_ab), None), load(Path(a.ref_ba), None)
    if rA["prompts"] != PA or rB["prompts"] != PB:
        raise SystemExit("the reference datasets must have exactly the same query prompts")
    ids = [p["id"] for p in PA if p["split"] == "train"]
    cue_of = np.array([PA[i]["cue"] for i in ids])
    keys = ["nL", "nB", "gu", "hon"] + [f"b_{c}" for c in conds]

    def rates(by):
        cc = [counts(by[i], conds) for i in ids]
        return {k: np.array([c[k] / c["n"] for c in cc]) for k in keys}

    R = {(pat, o): rates(by) for pat, o, by in (("this", "AB", byA), ("this", "BA", byB),
                                                ("ref", "AB", rbyA), ("ref", "BA", rbyB))}
    out = {}
    print(f"\n== demo pattern {''.join(task.demo_giveup_labels(task.load_dataset(a.ab)['fewshot']))} (this) vs "
          f"{''.join(task.demo_giveup_labels(rA['fewshot']))} (reference {a.ref_ab}), same queries, train; "
          "order-averaged, paired by question ==")
    for cue in (None, "A", "B"):
        sel = np.flatnonzero(cue_of == cue) if cue else np.arange(len(ids))
        lab = "pooled" if cue is None else name[cue]
        idx = np.concatenate([np.arange(len(sel))[None, :], rng.integers(0, len(sel), (a.boot, len(sel)))])

        def avg(pat, k):  # mean over (resampled) questions of the two-order mean; row 0 = estimate
            v = (R[(pat, "AB")][k][sel] + R[(pat, "BA")][k][sel]) / 2
            return v[idx].mean(1)

        def bshare(pat):  # equal allocation of orders: (B) share among listed give-ups
            return sum(R[(pat, o)]["nB"][sel][idx].mean(1) for o in ("AB", "BA")) / \
                sum(R[(pat, o)]["nL"][sel][idx].mean(1) for o in ("AB", "BA"))

        rec = {}
        lines = []
        for k, nm in [(f"b_{c}", f"b {c}") for c in conds]:
            t, r = avg("this", k), avg("ref", k)
            rec[nm] = [float(t[0]), float(r[0])]
            lines.append(f"{nm} {r[0]:.3f} -> {t[0]:.3f} ({t[0] - r[0]:+.3f} {ci(t[1:] - r[1:])})")
        print(f"  {lab:>6}: " + "; ".join(lines))
        if cue:
            lines = []
            for c in ("rule", "anti"):
                t = avg("this", f"b_{c}") / avg("this", "b_exception")
                r = avg("ref", f"b_{c}") / avg("ref", "b_exception")
                rec[f"ratio_{c}"] = [float(t[0]), float(r[0])]
                lines.append(f"{c}/exception {r[0]:.2f} -> {t[0]:.2f} ({t[0] - r[0]:+.2f} {ci(t[1:] - r[1:])})")
            print(f"          within-cue ratio (order-balanced expectation): " + "; ".join(lines))
        lines = []
        for k, nm in (("nL", "listed give-up"), ("gu", "all give-ups"), ("hon", "honest")):
            t, r = avg("this", k), avg("ref", k)
            rec[nm] = [float(t[0]), float(r[0])]
            lines.append(f"{nm} {r[0]:.3f} -> {t[0]:.3f} ({t[0] - r[0]:+.3f} {ci(t[1:] - r[1:])})")
        print("          " + "; ".join(lines))
        t, r = bshare("this"), bshare("ref")
        rec["B_share_equal_alloc"] = [float(t[0]), float(r[0])]
        summ = {}
        for pat in ("ref", "this"):
            sAB = R[(pat, "AB")]["nB"][sel][idx].mean(1) / R[(pat, "AB")]["nL"][sel][idx].mean(1)
            sBA = R[(pat, "BA")]["nB"][sel][idx].mean(1) / R[(pat, "BA")]["nL"][sel][idx].mean(1)
            summ[pat] = ((logit(sAB) - logit(sBA)) / 2, (logit(sAB) + logit(sBA)) / 2)
        rec["beta_pos"] = [float(summ["this"][0][0]), float(summ["ref"][0][0])]
        rec["beta_label"] = [float(summ["this"][1][0]), float(summ["ref"][1][0])]
        print(f"          (B) share among listed give-ups, equal allocation {r[0]:.3f} -> {t[0]:.3f} "
              f"({t[0] - r[0]:+.3f} {ci(t[1:] - r[1:])}); logit summaries (descriptive): beta_pos "
              f"{summ['ref'][0][0]:+.2f} -> {summ['this'][0][0]:+.2f}, beta_label {summ['ref'][1][0]:+.2f} -> "
              f"{summ['this'][1][0]:+.2f} ({summ['this'][1][0] - summ['ref'][1][0]:+.2f} "
              f"{ci(summ['this'][1][1:] - summ['ref'][1][1:])})")
        out[lab] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ab", required=True, help="dataset with (A) listed first (the base)")
    ap.add_argument("--ba", required=True, help="the same dataset with (B) listed first (variant --order BA)")
    ap.add_argument("--calib-ab", default=None, help="default: <dataset dir>/calib_<stem>")
    ap.add_argument("--calib-ba", default=None)
    ap.add_argument("--ref-ab", default=None, help="the same queries under another demo pattern, (A) first")
    ap.add_argument("--ref-ba", default=None, help="... and (B) first")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=None, help="JSON summary (default: <dataset dir>/order_<ba stem>.json)")
    a = ap.parse_args()

    (dA, byA), (dB, byB) = (load(Path(a.ab), a.calib_ab and Path(a.calib_ab)),
                            load(Path(a.ba), a.calib_ba and Path(a.calib_ba)))
    PA, PB = dA["prompts"], dB["prompts"]
    if len(PA) != len(PB) or dA["fewshot"] != dB["fewshot"]:
        raise SystemExit("the two datasets differ in prompts or demos; they must differ only in query order")
    for p, q in zip(PA, PB):
        if {k: p.get(k) for k in KEYS} != {k: q.get(k) for k in KEYS} or p["id"] != q["id"]:
            raise SystemExit(f"prompt {p['id']} differs beyond the listing order")
        if task.listing_order(p) != "AB" or task.listing_order(q) != "BA":
            raise SystemExit("--ab must list (A) first and --ba (B) first on every prompt")
    conds = task.conditions(dA)
    cv = dA["spec"].get("cue_values") or list(task.LABELS)
    name = {"A": cv[0], "B": cv[1]}
    rng = np.random.default_rng(0)
    res = dict(ab=a.ab, ba=a.ba, demo_labels="".join(task.demo_giveup_labels(dA["fewshot"])))
    print(f"{a.ab} (AB) vs {a.ba} (BA); give-up demo labels {res['demo_labels']}; cue values "
          f"{cv[0]} (A) / {cv[1]} (B)")

    def per_split(split):
        ids = [p["id"] for p in PA if p["split"] == split]
        return ids, {i: counts(byA[i], conds) for i in ids}, {i: counts(byB[i], conds) for i in ids}

    for split in ("train", "heldout"):
        ids, cA, cB = per_split(split)
        if not ids:
            continue
        print(f"\n== {split}: {len(ids)} questions, each in both orders ==")
        res[split] = {}
        for cue in (None, "A", "B"):
            sub = [i for i in ids if cue is None or PA[i]["cue"] == cue]
            lab = "pooled" if cue is None else name[cue]
            idx = np.concatenate([np.arange(len(sub))[None, :], rng.integers(0, len(sub), (a.boot, len(sub)))])

            def tot(c, k):
                v = np.array([c[i][k] for i in sub], float)
                return v[idx].sum(1)  # row 0 is the point estimate

            sAB, sBA = tot(cA, "nB") / tot(cA, "nL"), tot(cB, "nB") / tot(cB, "nL")
            bpos, blab = (logit(sAB) - logit(sBA)) / 2, (logit(sAB) + logit(sBA)) / 2
            rate = {o: {k: tot(c, k) / tot(c, "n") for k in ["nL", "gu", "hon"] + [f"b_{x}" for x in conds]}
                    for o, c in (("AB", cA), ("BA", cB))}
            r = dict(B_share_AB=float(sAB[0]), B_share_BA=float(sBA[0]),
                     second_share_AB=float(sAB[0]), second_share_BA=float(1 - sBA[0]),
                     beta_pos=float(bpos[0]), beta_pos_ci=[float(np.percentile(bpos[1:], q)) for q in (2.5, 97.5)],
                     beta_label=float(blab[0]),
                     beta_label_ci=[float(np.percentile(blab[1:], q)) for q in (2.5, 97.5)])
            print(f"  {lab:>6}: (B) share AB {sAB[0]:.3f} -> BA {sBA[0]:.3f} (diff {sBA[0] - sAB[0]:+.3f} "
                  f"{ci(sBA[1:] - sAB[1:])}); second-position share {sAB[0]:.3f} -> {1 - sBA[0]:.3f}. "
                  f"Pure position predicts BA {1 - sAB[0]:.3f}, pure label {sAB[0]:.3f}")
            # Listed give-ups differ by order, so in a build with half the prompts in each order
            # the (B) share among give-ups weights each order by its give-up mass, not 1/2.
            sEq = (tot(cA, "nB") / tot(cA, "n") + tot(cB, "nB") / tot(cB, "n")) / \
                  (tot(cA, "nL") / tot(cA, "n") + tot(cB, "nL") / tot(cB, "n"))
            r.update(B_share_equal_alloc=float(sEq[0]))
            print(f"          equal allocation of orders: (B) share among listed give-ups {sEq[0]:.3f} "
                  f"{ci(sEq[1:])} (simple mean of the two orders {(sAB[0] + sBA[0]) / 2:.3f})")
            print(f"          logit (descriptive, not separately identified): beta_pos {bpos[0]:+.2f} "
                  f"{ci(bpos[1:])}, beta_label {blab[0]:+.2f} {ci(blab[1:])}")
            line = []
            for k, nm in [("nL", "listed give-up"), ("gu", "all give-ups"), ("hon", "honest")] + \
                         [(f"b_{x}", f"b {x}") for x in conds]:
                d = rate["BA"][k] - rate["AB"][k]
                r[k] = dict(AB=float(rate["AB"][k][0]), BA=float(rate["BA"][k][0]), diff=float(d[0]),
                            ci=[float(np.percentile(d[1:], q)) for q in (2.5, 97.5)])
                line.append(f"{nm} {rate['AB'][k][0]:.3f} -> {rate['BA'][k][0]:.3f} ({d[0]:+.3f} {ci(d[1:])})")
            print("          " + "; ".join(line[:3]) + "\n          " + "; ".join(line[3:]))
            res[split][lab] = r

    # All four renderings of each flip-pair question: (cue, order).
    flips = [p for p in PA if p["split"] == "flip"]
    if flips:
        print(f"\n== flip pairs: {len(flips)} questions x (cue {cv[0]} / {cv[1]}) x (order AB / BA) ==")
        cells = {}  # (cue, order) -> list over pairs of counts
        for f in flips:
            q = PA[f["parent"]]
            for pid, cue in ((q["id"], q["cue"]), (f["id"], f["cue"])):
                cells.setdefault((cue, "AB"), []).append(counts(byA[pid], conds))
                cells.setdefault((cue, "BA"), []).append(counts(byB[pid], conds))
        n = len(flips)
        idx = np.concatenate([np.arange(n)[None, :], rng.integers(0, n, (a.boot, n))])
        val = {}
        for key, cs in cells.items():
            arr = {k: np.array([c[k] for c in cs], float)[idx].sum(1) for k in ("n", "nL", "nB", "gu", "hon")}
            val[key] = dict(B=arr["nB"] / arr["nL"], gu=arr["gu"] / arr["n"], hon=arr["hon"] / arr["n"])
        res["flip_cells"] = {}
        for m, nm in (("B", "(B) share"), ("gu", "all give-ups"), ("hon", "honest")):
            g = {k: v[m] for k, v in val.items()}
            print(f"  {nm:>12}: " + "  ".join(f"{name[c]}/{o} {g[(c, o)][0]:.3f}" for c in "AB" for o in ("AB", "BA")))
            cue_eff = {o: g[("A", o)] - g[("B", o)] for o in ("AB", "BA")}
            ord_eff = {c: g[(c, "BA")] - g[(c, "AB")] for c in "AB"}
            inter = cue_eff["BA"] - cue_eff["AB"]
            print(f"  {'':>12}  {cv[0]} - {cv[1]}: AB {cue_eff['AB'][0]:+.3f} {ci(cue_eff['AB'][1:])}, "
                  f"BA {cue_eff['BA'][0]:+.3f} {ci(cue_eff['BA'][1:])}; BA - AB: {cv[0]} {ord_eff['A'][0]:+.3f} "
                  f"{ci(ord_eff['A'][1:])}, {cv[1]} {ord_eff['B'][0]:+.3f} {ci(ord_eff['B'][1:])}; "
                  f"interaction {inter[0]:+.3f} {ci(inter[1:])}")
            res["flip_cells"][m] = {f"{c}/{o}": float(g[(c, o)][0]) for c in "AB" for o in ("AB", "BA")}

    # Expected within-cue ratios for an order-balanced build: each prompt rendered in one
    # order, (B) first on half of every (cue, exception label) cell. Over random balanced
    # assignments the expected b of a cell is the mean over its prompts of the two-order mean.
    ids, cA, cB = per_split("train")
    print("\n== within-cue b ratio to exception (train): single orders and an order-balanced build ==")
    bi = {o: {c: np.array([cc[i][f"b_{c}"] / cc[i]["n"] for i in ids]) for c in conds}
          for o, cc in (("AB", cA), ("BA", cB))}
    cue_of = np.array([PA[i]["cue"] for i in ids])
    exc_of = np.array([PA[i]["exception_label"] for i in ids])

    def ratios(b):
        return {(c, v): b[c][cue_of == v].mean() / b["exception"][cue_of == v].mean()
                for c in ("rule", "anti") for v in "AB"}

    def show(r):
        return ", ".join(f"{c} {r[(c, 'A')]:.2f} / {r[(c, 'B')]:.2f}" for c in ("rule", "anti"))

    for o in ("AB", "BA"):
        print(f"  all {o} ({cv[0]} / {cv[1]}): {show(ratios(bi[o]))}")
    mean_b = {c: (bi["AB"][c] + bi["BA"][c]) / 2 for c in conds}
    exp = ratios(mean_b)
    boot = []
    for _ in range(a.boot):
        j = np.concatenate([rng.choice(np.flatnonzero(cue_of == v), (cue_of == v).sum()) for v in "AB"])
        boot.append({(c, v): mean_b[c][j][cue_of[j] == v].mean() / mean_b["exception"][j][cue_of[j] == v].mean()
                     for c, v in exp})
    assign = []
    for _ in range(a.boot):
        first_b = np.zeros(len(ids), bool)
        for v in "AB":
            for e in "AB":
                cell = np.flatnonzero((cue_of == v) & (exc_of == e))
                first_b[rng.choice(cell, len(cell) // 2, replace=False)] = True
        assign.append(ratios({c: np.where(first_b, bi["BA"][c], bi["AB"][c]) for c in conds}))
    ok = all(0.7 <= v <= 1.3 for v in exp.values())
    print(f"  order-balanced, expected: {show(exp)}")
    for k in exp:
        qb = np.percentile([b[k] for b in boot], [2.5, 97.5])
        qa = np.percentile([b[k] for b in assign], [2.5, 97.5])
        print(f"    {k[0]} {name[k[1]]}: {exp[k]:.2f}, 95% over questions [{qb[0]:.2f}, {qb[1]:.2f}], "
              f"over balanced assignments [{qa[0]:.2f}, {qa[1]:.2f}]")
    frac = float(np.mean([all(0.7 <= v <= 1.3 for v in r.values()) for r in assign]))
    print(f"  [{'OK' if ok else 'FAIL'}] every expected ratio in 0.7-1.3 (rule and anti, both cue values)")
    print(f"  [INFO] balanced assignments with all four ratios in 0.7-1.3: {frac:.3f} of {a.boot} "
          "(conditional on these per-question estimates; not a prospective probability)")
    res["balanced_expected_ratio"] = {f"{c}/{name[v]}": float(x) for (c, v), x in exp.items()}
    res["balanced_gate_ok"] = ok
    res["balanced_assignments_all_in_band"] = frac

    if a.ref_ab or a.ref_ba:
        res["demo_compare"] = demo_compare(a, PA, PB, byA, byB, conds, name, rng)

    out = Path(a.out or Path(a.ba).parent / f"order_{Path(a.ba).stem}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
