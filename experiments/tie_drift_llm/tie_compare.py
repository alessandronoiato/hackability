#!/usr/bin/env python3
"""Main reward-tie comparison: paired within-tie drift, rule against exception, at step 160.

README, "Main reward-tie comparison". For each declared (construction, seed) of an admitted
construction: the rule and the exception run (grpo.py --reward proxy, β = 0.1, --tie-support
rule exception, the frozen flags). From each run's saved train samples at step 160 and the
calibration of its dataset (sha256 as recorded in the run's config):

    D_c = sum_U h_i / sum_U p_i  -  sum_U p_i s0_i^c / sum_U p_i,    Delta = D_rule - D_exception,

h_i the hack and p_i the passing samples of train rendering i, s0_i^c = b_i^c / (a_i + b_i^c)
pi0's hack share there under condition c, and U the common support: renderings where pi0
passed at least once under rule and at least once under exception in the calibration. Both
arms use the same U. The first term is grpo.py's hack_share_tie, the second its
hack_share_score.

Status, in this order:
  * invalid for the declared comparison: any failed check (below) on a calibration or a
    declared pair;
  * incomplete: a calibration, a declared run or its step-160 eval is missing;
  * no admitted construction: every calibration fails its eligibility gates;
  * otherwise the reading (declared direction Delta > 0): Delta > 0 in every declared pair =
    directionally consistent with the prediction; Delta < 0 in every pair = directionally
    opposite; otherwise inconclusive. Both D are printed: Delta > 0 does not by itself mean
    the rule arm moved toward the hack (both D can be negative).

Checks. Calibration: declared seed 3, 32 samples on every train rendering, train split, the
model, the dataset's sha256, verdict 'main' (all_pass decides admission), and samples.jsonl
reproducing the summary's a_i, b_i. Pair: the declared reward and settings, the recorded
dataset and calibration sha256, configs identical except for the condition and output paths
(same clean git commit), exactly 32 samples on each of the 512 train renderings at step 160 in
both arms, the step-1 group records identical except for what was rewarded, and the logged
drift and its excluded count equal to the recomputed ones on U.

Uncertainty (95%), neither of which is replication (the training runs are; run-to-run
variation shows only in the spread over seeds):
  * measurement: given the trained runs and the fixed questions, resampling the finite
    responses. Each replicate redraws every rendering's calibration responses (one draw,
    graded under both arms, shared by the construction's seeds) and every arm's step-160
    responses within each rendering. U stays fixed; a rendering whose redrawn calibration has
    no success under an arm gets its share bounded in [0, 1], so a replicate gives a range
    of D and the interval runs from the 2.5th percentile of the lower ends to the 97.5th of
    the upper ends. The share of replicates needing a bound is printed;
  * questions (descriptive): base questions reweighted, each with its four renderings, both
    arms, both seeds and its observed calibration and endpoint values held fixed. It shows
    how much the result depends on which questions were drawn; it does not redraw responses.

Support: renderings outside U, each arm's pass mass on them at step 160, and the range of the
full-set Delta (all train renderings) when each arm's unknown pi0 hack shares (no pass under
that arm in the calibration) range over [0, 1]; known shares are kept.

--diagnostic (smoke tests, synthetic runs): the same computation on whatever pairs exist, at
--at-step, checks printed but not enforced, renderings without samples skipped; its result is
labelled as not the declared comparison. --trajectory adds D and Delta at every eval
(descriptive). Transfer, gold and the other per-run measures: analyze.py --pc3 --at-step 160.

    python experiments/tie_drift_llm/tie_compare.py \\
        --datasets experiments/tie_drift_llm/data/c2ndxX_s0.json experiments/tie_drift_llm/data/c2ndxX_s4.json \\
        --seeds 0,1 2,3
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ("rule", "exception")
STEP, SAMPLES = 160, 32
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# The frozen protocol (README, "Main reward-tie comparison"): what every run and calibration must record.
DECLARED_RUN = dict(reward="proxy", beta=0.1, steps=STEP, optimizer="adam", lr=1e-6, adam_beta1=0.9,
                    adam_beta2=0.999, group=8, prompts_per_step=32, loss_agg="sum", std_norm=True, inner_steps=1,
                    clip_eps=0.0, max_grad_norm=0.0, freeze_embeddings=False, hack_gold=0.0, balance_cells=True,
                    kl_stop=100.0, eval_every=10, eval_samples=2, eval_samples_other=4, final_eval_samples=SAMPLES,
                    max_new_tokens=384, model=MODEL, n_prompts=None, calib_samples=SAMPLES, calib_diagnostic=False)
DECLARED_CALIB = dict(seed=3, samples_per_prompt=SAMPLES, splits=["train"], model=MODEL, max_new_tokens=384)
# Config keys that may differ between the two runs of a pair.
PAIR_KEYS = {"cond", "out", "run_dir"}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_runs(root: Path) -> dict[int, dict[str, Path]]:
    by: dict[int, dict[str, Path]] = defaultdict(dict)
    for d in sorted(root.glob("*")):
        if not (d / "config.json").exists():
            continue
        cfg = json.load(open(d / "config.json"))
        if cfg.get("cond") in CONDS:
            if cfg["cond"] in by[cfg["seed"]]:
                raise SystemExit(f"two {cfg['cond']} runs with seed {cfg['seed']} under {root}")
            by[cfg["seed"]][cfg["cond"]] = d
    return dict(sorted(by.items()))


def train_counts(run_dir: Path, n: int) -> dict[int, np.ndarray]:
    """step -> per prompt id [samples, hack, pass] on the train split, from the saved eval samples."""
    out: dict[int, np.ndarray] = {}
    with gzip.open(run_dir / "eval_samples.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] != "train":
                continue
            c = out.setdefault(r["step"], np.zeros((n, 3), int))
            c[r["prompt_id"]] += (1, r["cat"] == "hack", r["cat"] in ("honest", "hack"))
    return out


def calib_joint(path: Path, n: int) -> np.ndarray:
    """Per prompt id, calibration responses by joint grade: honest, accepted under rule only, under
    exception only, under both, neither. Each response is one draw graded under both arms."""
    J = np.zeros((n, 5), int)
    for line in open(path):
        r = json.loads(line)
        c = r["cat"]
        if (c["rule"] == "honest") != (c["exception"] == "honest"):
            raise SystemExit(f"{path}: a response honest under one arm only (prompt {r['prompt_id']})")
        hr, he = c["rule"] == "hack", c["exception"] == "hack"
        J[r["prompt_id"], 0 if c["rule"] == "honest" else 3 if hr and he else 1 if hr else 2 if he else 4] += 1
    return J


def shares(J: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(accepted, honest + accepted) counts per arm from joint counts (last axis)."""
    a = J[..., 0]
    return {"rule": (J[..., 1] + J[..., 3], a + J[..., 1] + J[..., 3]),
            "exception": (J[..., 2] + J[..., 3], a + J[..., 2] + J[..., 3])}


def drift(c: np.ndarray, s0: np.ndarray, rows: np.ndarray) -> tuple[float, float, float]:
    """(hack_share_tie, hack_share_score, D) over the prompt ids in rows."""
    h, p = c[rows, 1].astype(float), c[rows, 2].astype(float)
    tie, score = h.sum() / p.sum(), (p * s0[rows]).sum() / p.sum()
    return tie, score, tie - score


def full_range(c: np.ndarray, s0: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """D over rows, unknown pi0 shares (NaN) anywhere in [0, 1]: (min, max)."""
    h, p, s = c[rows, 1].astype(float), c[rows, 2].astype(float), s0[rows]
    known = np.isfinite(s)
    base = h.sum() / p.sum() - (p[known] * s[known]).sum() / p.sum()
    return base - p[~known].sum() / p.sum(), base


def measurement_boot(J: np.ndarray, E: list[dict[str, np.ndarray]], U: np.ndarray, B: int, rng,
                     chunk: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redraw the calibration (shared by the pairs) and the endpoint responses within each rendering of U.

    J: joint calibration counts per prompt id; E: per pair, arm -> [samples, hack, pass] per prompt id.
    Returns D lower and upper ends, shape (B, pairs, arms), and per replicate the number of renderings
    whose redrawn calibration has no success under some arm (their shares bounded in [0, 1]).
    """
    JU = J[U]
    nJ = JU.sum(1)
    lo, hi = np.empty((B, len(E), 2)), np.empty((B, len(E), 2))
    unsup = np.empty(B, int)
    for b0 in range(0, B, chunk):
        b = min(chunk, B - b0)
        sh = shares(rng.multinomial(nJ, JU / nJ[:, None], size=(b, len(U))))
        unsup[b0:b0 + b] = ((sh["rule"][1] == 0) | (sh["exception"][1] == 0)).sum(1)
        for j, Ej in enumerate(E):
            for m, c in enumerate(CONDS):
                n, h, p = (Ej[c][U, k] for k in range(3))
                d = rng.multinomial(n, np.stack([h, p - h, n - p], 1) / n[:, None], size=(b, len(U)))
                hh, pp = d[..., 0], d[..., 0] + d[..., 1]
                num, den = sh[c]
                known = den > 0
                P = pp.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tie = hh.sum(1) / P
                    ks = (pp * np.where(known, num / np.maximum(den, 1), 0.0)).sum(1) / P
                    us = (pp * ~known).sum(1) / P
                hi[b0:b0 + b, j, m], lo[b0:b0 + b, j, m] = tie - ks, tie - ks - us
    return lo, hi, unsup


def ci(x: np.ndarray) -> tuple[float, float]:
    """95% percentile interval; replicates without passing samples (only in tiny smoke tests) are skipped."""
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def range_ci(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(lo, 2.5)), float(np.nanpercentile(hi, 97.5))


def padded(x, n: int, fill=np.nan) -> np.ndarray:
    """x as a length-n float array (calibrations of the first prompts only, in smoke tests, are shorter)."""
    out = np.full(n, fill, float)
    out[:len(x)] = x
    return out


def sgn(x: float) -> str:
    return f"{x:+.3f}"


def fmt_ci(x) -> str:
    return f"[{sgn(x[0])}, {sgn(x[1])}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True, help="the declared constructions")
    ap.add_argument("--seeds", nargs="+", default=None,
                    help="declared training seeds, one comma list per dataset (e.g. 0,1 2,3)")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run dirs, one per dataset (default: <dataset dir>/runs/<stem>_tie)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="smoke tests: whatever pairs exist, checks not enforced; not the declared comparison")
    ap.add_argument("--at-step", type=int, default=None, help="--diagnostic only (the declared step is 160)")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--trajectory", action="store_true", help="D and Delta at every eval")
    ap.add_argument("--out", default=None, help="JSON (default: <first dataset dir>/tie_compare.json)")
    a = ap.parse_args()
    if a.runs and len(a.runs) != len(a.datasets):
        ap.error("--runs needs one dir per dataset")
    if not a.diagnostic and (a.seeds is None or len(a.seeds) != len(a.datasets)):
        ap.error("--seeds needs one comma list per dataset (the declared pairs)")
    if not a.diagnostic and a.at_step not in (None, STEP):
        ap.error(f"the declared step is {STEP}; another step needs --diagnostic")
    S = a.at_step or STEP
    rng = np.random.default_rng(0)
    report = []           # one entry per analysed pair
    meas, qboot = [], []  # per construction: Delta (lo, hi) measurement replicates; Delta question replicates
    invalid, missing, constructions = [], [], []

    def check(ok: bool, text: str, chk: list, where: str) -> None:
        chk.append(("OK  " if ok else "FAIL") + " " + text)
        if not ok:
            invalid.append(f"{where}: {text}")

    for k, dsp in enumerate(map(Path, a.datasets)):
        root = Path(a.runs[k]) if a.runs else dsp.parent / "runs" / f"{dsp.stem}_tie"
        seeds = [int(x) for x in a.seeds[k].split(",")] if a.seeds else None
        prompts = json.load(open(dsp))["prompts"]
        train = np.array([p["id"] for p in prompts if p["split"] == "train"])
        qs = sorted({prompts[i]["question"] for i in train})
        q_of = {q: j for j, q in enumerate(qs)}
        ds_sha = sha256_file(dsp)
        calp = dsp.parent / f"calib_{dsp.stem}" / "summary.json"
        print(f"\n== {dsp.stem} (sha256 {ds_sha[:12]}) ==")
        if not calp.exists() or not (calp.parent / "samples.jsonl").exists():
            print(f"calibration missing: {calp} and samples.jsonl")
            missing.append(f"{dsp.stem}: calibration")
            continue
        cal = json.load(open(calp))
        cal_sha = sha256_file(calp)
        J = calib_joint(calp.parent / "samples.jsonl", len(prompts))
        s0 = {c: np.where(d_ > 0, n_ / np.maximum(d_, 1), np.nan) for c, (n_, d_) in shares(J).items()}
        el = cal.get("eligibility") or {}
        cchk: list = []
        where = f"{dsp.stem} calibration"
        for key, v in DECLARED_CALIB.items():
            check(cal.get(key) == v, f"{key} {cal.get(key)!r} (declared {v!r})", cchk, where)
        check(cal.get("dataset_sha256") == ds_sha, "dataset sha256 as sampled", cchk, where)
        n_i = padded(cal.get("n_i") or [], len(prompts), 0)
        check(bool((n_i[train] == SAMPLES).all() and (J[train].sum(1) == SAMPLES).all()),
              f"{SAMPLES} samples on every train rendering ({int((n_i[train] == SAMPLES).sum())}/{len(train)})",
              cchk, where)
        pc = cal.get("per_cond") or {}
        with np.errstate(invalid="ignore"):
            nJ = np.where(J[train].sum(1) > 0, J[train].sum(1), np.nan)  # NaN where never sampled, as in the summary
        same = all(c in pc for c in CONDS) and all(
            np.allclose(padded(pc[c]["a_i"], len(prompts))[train], J[train, 0] / nJ, equal_nan=True) and
            np.allclose(padded(pc[c]["b_i"], len(prompts))[train], shares(J)[c][0][train] / nJ, equal_nan=True)
            for c in CONDS if c in pc)
        check(same, "samples.jsonl reproduces the summary's a_i, b_i", cchk, where)
        check(el.get("verdict") == "main", f"eligibility verdict {el.get('verdict')!r} (declared 'main')", cchk, where)
        admitted = bool(el.get("all_pass"))
        U = train[np.isfinite(s0["rule"][train]) & np.isfinite(s0["exception"][train])]
        miss = {c: int((~np.isfinite(s0[c][train])).sum()) for c in CONDS}
        print(f"calibration {calp} (sha256 {cal_sha[:12]}, seed {cal.get('seed')}): "
              f"{'admitted' if admitted else 'NOT ADMITTED'} ({el.get('decision', 'no verdict')})\n  "
              + "\n  ".join(cchk)
              + f"\ncommon support U: {len(U)}/{len(train)} train renderings (pi0 never passed under rule on "
              f"{miss['rule']}, under exception on {miss['exception']})")
        runs = find_runs(root)
        if not admitted and not a.diagnostic:
            constructions.append(dict(dataset=dsp.stem, admitted=False))
            if runs:
                print(f"  runs present for a construction that was not admitted ({sorted(runs)}); not used")
            continue
        constructions.append(dict(dataset=dsp.stem, admitted=admitted, n_support=len(U)))
        if seeds is None:
            seeds = [s for s, v in runs.items() if set(v) == set(CONDS)]
        extra = sorted(set(runs) - set(seeds))
        if extra:
            print(f"  runs for undeclared seeds {extra}: not used (report them)")
        E, done = [], []
        for s in seeds:
            v = runs.get(s, {})
            where = f"{dsp.stem} seed {s}"
            if set(v) != set(CONDS):
                print(f"  seed {s}: missing {sorted(set(CONDS) - set(v))} run(s)")
                missing.append(f"{where}: {sorted(set(CONDS) - set(v))}")
                continue
            cfg = {c: json.load(open(v[c] / "config.json")) for c in CONDS}
            ev = {c: {e["step"]: e for e in map(json.loads, open(v[c] / "eval.jsonl"))} for c in CONDS}
            cnt = {c: train_counts(v[c], len(prompts)) for c in CONDS}
            if any(S not in ev[c] or S not in cnt[c] for c in CONDS):
                print(f"  seed {s}: no eval at step {S} in both runs")
                missing.append(f"{where}: step {S}")
                continue
            chk: list = []
            for c in CONDS:
                bad = [f"{key} {cfg[c].get(key)!r}" for key, val in DECLARED_RUN.items() if cfg[c].get(key) != val]
                if sorted(cfg[c].get("tie_support") or []) != sorted(CONDS):
                    bad.append(f"tie_support {cfg[c].get('tie_support')}")
                if cfg[c].get("final_eval_samples_train") not in (None, SAMPLES):
                    bad.append(f"final_eval_samples_train {cfg[c].get('final_eval_samples_train')}")
                check(not bad, f"{c}: declared settings" + (f" (differs: {', '.join(bad)})" if bad else ""), chk, where)
            check(all(cfg[c].get("dataset_sha256") == ds_sha and (cfg[c].get("calibration") or {}).get("sha256")
                      == cal_sha for c in CONDS), "dataset and calibration sha256 as recorded by both runs", chk, where)
            diff = sorted(x for x in set(cfg["rule"]) | set(cfg["exception"])
                          if x not in PAIR_KEYS and cfg["rule"].get(x) != cfg["exception"].get(x))
            gc = cfg["rule"].get("git_commit") or "none"
            check(not diff, "configs identical except cond and paths" + (f" (also {diff})" if diff else ""), chk, where)
            check(gc != "none" and not gc.endswith("+dirty"), f"git commit {gc[:12]}{'+dirty' if gc.endswith('+dirty') else ''}",
                  chk, where)
            full = {c: cnt[c][S][train, 0] for c in CONDS}
            check(all((full[c] == SAMPLES).all() for c in CONDS),
                  f"exactly {SAMPLES} samples on every train rendering at step {S} "
                  f"({' / '.join(str(int((full[c] == SAMPLES).sum())) for c in CONDS)} of {len(train)})", chk, where)
            g1 = {}
            for c in CONDS:
                with open(v[c] / "train.jsonl") as fh:
                    first = json.loads(fh.readline())
                g1[c] = [g[:-1] for g in first.get("groups", [])] if first.get("step") == 1 else []
            check(bool(g1["rule"]) and g1["rule"] == g1["exception"],
                  f"step-1 groups identical except rewards ({len(g1['rule'])} groups)", chk, where)
            # Renderings both runs evaluated: every train rendering, except in --diagnostic smoke tests.
            tr = train[(full["rule"] > 0) & (full["exception"] > 0)]
            Up = np.intersect1d(U, tr)
            res = dict(dataset=dsp.stem, seed=s, runs={c: str(v[c]) for c in CONDS}, n_evaluated=len(tr),
                       n_support=len(Up))
            for c in CONDS:
                tie, score, D = drift(cnt[c][S], s0[c], Up)
                n, h, p = cnt[c][S][tr].sum(0)
                rec = ev[c][S]
                res[c] = dict(D=D, tie=tie, score=score, pass_rate=p / n, hack_rate=h / n, gold=rec.get("gold"),
                              kl=rec.get("kl"),
                              excluded_pass_mass=float(cnt[c][S][np.setdiff1d(tr, Up), 2].sum() / p),
                              full_D_range=list(full_range(cnt[c][S], s0[c], tr)),
                              D_step0=drift(cnt[c][0], s0[c], Up)[2] if 0 in cnt[c] else None,
                              logged_D=(rec["hack_share_tie"] - rec["hack_share_score"]
                                        if "hack_share_score" in rec else None),
                              logged_excluded=rec.get("n_excluded"))
            r, e = res["rule"], res["exception"]
            if r["logged_D"] is None or e["logged_D"] is None:
                check(False, "drift logged by grpo.py", chk, where)
            else:
                dd = max(abs(r["logged_D"] - r["D"]), abs(e["logged_D"] - e["D"]))
                n_out = len(train) - len(U)
                check(dd < 1e-9 and r["logged_excluded"] == e["logged_excluded"] == n_out,
                      f"logged drift on U (max |diff| {dd:.1e}; excluded {r['logged_excluded']} / "
                      f"{e['logged_excluded']}, outside U {n_out})", chk, where)
            res["Delta"] = r["D"] - e["D"]
            res["full_Delta_range"] = [r["full_D_range"][0] - e["full_D_range"][1],
                                       r["full_D_range"][1] - e["full_D_range"][0]]
            res["checks"] = chk
            report.append(res)
            E.append({c: cnt[c][S] for c in CONDS})
            done.append(res)
            print(f"  seed {s}: Delta {sgn(res['Delta'])}; D_rule {sgn(r['D'])} (tie {r['tie']:.3f}, score "
                  f"{r['score']:.3f}); D_exception {sgn(e['D'])} (tie {e['tie']:.3f}, score {e['score']:.3f})\n"
                  f"          pass {r['pass_rate']:.3f} / {e['pass_rate']:.3f}, hack among all samples "
                  f"{r['hack_rate']:.3f} / {e['hack_rate']:.3f}, gold {r['gold']:.3f} / {e['gold']:.3f}, eval KL "
                  f"{r['kl']:.3f} / {e['kl']:.3f} (rule / exception)"
                  + (f"; D at step 0 {sgn(r['D_step0'])} / {sgn(e['D_step0'])}" if r["D_step0"] is not None else "")
                  + f"\n          outside U: pass mass {r['excluded_pass_mass']:.4f} / {e['excluded_pass_mass']:.4f}; "
                  f"full-set Delta in {fmt_ci(res['full_Delta_range'])}\n          "
                  + "\n          ".join(chk))
            if a.trajectory:
                traj = []
                for t in sorted(set(cnt["rule"]) & set(cnt["exception"])):
                    dr, de = (drift(cnt[c][t], s0[c], Up)[2] for c in CONDS)
                    traj.append(f"{t}: {sgn(dr)} / {sgn(de)} / {sgn(dr - de)}")
                print("          trajectory (step: D_rule / D_exception / Delta): " + ", ".join(traj))
        if not done:
            continue
        Uc = U[np.all([x[c][U, 0] > 0 for x in E for c in CONDS], 0)]  # = U unless --diagnostic
        lo, hi, unsup = measurement_boot(J, E, Uc, a.boot, rng)
        meas.append((lo[..., 0] - hi[..., 1], hi[..., 0] - lo[..., 1]))
        X = np.zeros((len(qs), len(done), 2, 3))  # question, pair, arm: hacks, passes, passes x s0 over U
        qU = np.array([q_of[prompts[i]["question"]] for i in Uc], int)
        for j, Ej in enumerate(E):
            for m, c in enumerate(CONDS):
                x = Ej[c][Uc]
                np.add.at(X[:, j, m], qU, np.stack([x[:, 1], x[:, 2], x[:, 2] * s0[c][Uc]], 1))
        W = rng.multinomial(len(qs), np.full(len(qs), 1 / len(qs)), size=a.boot).astype(float)
        T = np.einsum("bq,qpaj->bpaj", W, X)
        with np.errstate(invalid="ignore", divide="ignore"):
            Dq = (T[..., 0] - T[..., 2]) / T[..., 1]
        qboot.append(Dq[..., 0] - Dq[..., 1])
        print(f"  95% intervals ({a.boot} replicates; {np.mean(unsup > 0):.1%} of measurement replicates bounded "
              f"a redrawn rendering without calibration success, mean {unsup.mean():.2f} renderings):")
        for j, res in enumerate(done):
            res["Delta_ci95_measurement"] = range_ci(meas[-1][0][:, j], meas[-1][1][:, j])
            res["Delta_ci95_questions"] = ci(qboot[-1][:, j])
            for m, c in enumerate(CONDS):
                res[c]["D_ci95_measurement"] = range_ci(lo[:, j, m], hi[:, j, m])
                res[c]["D_ci95_questions"] = ci(Dq[:, j, m])
            print(f"    seed {res['seed']}: measurement Delta {fmt_ci(res['Delta_ci95_measurement'])}, D_rule "
                  f"{fmt_ci(res['rule']['D_ci95_measurement'])}, D_exception "
                  f"{fmt_ci(res['exception']['D_ci95_measurement'])}; questions (descriptive) Delta "
                  f"{fmt_ci(res['Delta_ci95_questions'])}")

    if invalid and not a.diagnostic:
        status = "invalid for the declared comparison"
    elif missing and not a.diagnostic:
        status = "incomplete"
    elif not report:
        status = "no admitted construction" if not missing else "incomplete"
    else:
        d = np.array([x["Delta"] for x in report])
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        status = ("directionally consistent with the prediction" if pos == len(d) else
                  "directionally opposite to the prediction" if neg == len(d) else "inconclusive")
    if a.diagnostic:
        status = f"DIAGNOSTIC, not the declared comparison: {status}"
    summary = dict(status=status, at_step=S, constructions=constructions, pairs=report, failed_checks=invalid,
                   missing=missing)
    print(f"\nAt step {S}: {status}.")
    for x in invalid:
        print(f"  failed: {x}")
    for x in missing:
        print(f"  missing: {x}")
    if report:
        d = np.array([x["Delta"] for x in report])
        mlo, mhi = (np.concatenate([m[i] for m in meas], 1).mean(1) for i in (0, 1))
        mq = np.concatenate(qboot, 1).mean(1)
        summary.update(mean_Delta=float(d.mean()), mean_Delta_ci95_measurement=range_ci(mlo, mhi),
                       mean_Delta_ci95_questions=ci(mq), n_positive=int((d > 0).sum()), n_negative=int((d < 0).sum()))
        print(f"Delta > 0 in {summary['n_positive']} of {len(d)} pairs, < 0 in {summary['n_negative']} (declared "
              "direction Delta > 0). Per pair: " + ", ".join(
                  f"{x['dataset']} s{x['seed']} {sgn(x['Delta'])} (D {sgn(x['rule']['D'])} / {sgn(x['exception']['D'])})"
                  for x in report)
              + f"\nmean Delta {sgn(d.mean())}, 95% measurement {fmt_ci(summary['mean_Delta_ci95_measurement'])}, "
              f"questions (descriptive) {fmt_ci(summary['mean_Delta_ci95_questions'])}. The training runs are the "
              "replication; neither interval is.")
    out = Path(a.out or Path(a.datasets[0]).parent / "tie_compare.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
