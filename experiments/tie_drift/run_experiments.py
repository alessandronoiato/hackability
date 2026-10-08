#!/usr/bin/env python3
"""Tie-drift experiments E1-E5 (see README.md and draft_tie_drift.tex).

    python experiments/tie_drift/run_experiments.py all --quick
    python experiments/tie_drift/run_experiments.py e3 --adam
    python experiments/tie_drift/run_experiments.py e3 --quick --basis rotated dense
    python experiments/tie_drift/run_experiments.py e3 --quick --opt-sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402

HERE = Path(__file__).resolve().parent
MASSES = (0.15, 0.05, 0.80)
CHECKS: list[dict] = []

COLORS = {
    "tilt": "C0", "chi2": "C1", "bon_uniform": "C2", "bon_likelihood": "C3",
    "pg": "C4", "npg": "C5", "grpo": "C6", "grpo_adam": "C7",
}
LABELS = {
    "tilt": "KL tilt", "chi2": r"$\chi^2$ optimum", "bon_uniform": "BoN, uniform ties",
    "bon_likelihood": "BoN, likelihood ties", "pg": "PG (exact flow)", "npg": "natural PG",
    "grpo": "GRPO (SGD)", "grpo_adam": "GRPO (Adam)",
}


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def check(name: str, passed: bool, detail: str) -> None:
    CHECKS.append(dict(name=name, passed=bool(passed), detail=detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def info(name: str, detail: str, **fields) -> None:
    CHECKS.append(dict(name=name, passed=None, detail=detail, **fields))
    print(f"  [INFO] {name}: {detail}")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, default=_jsonable, indent=1)


def savefig(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.png/.pdf")


def plot_path(ax, path, key: str, method: str, ls: str = "-") -> None:
    color, label = COLORS.get(method, None), LABELS.get(method, method)
    if isinstance(path, list):
        top = max(float(np.nanmax(t["kl"])) for t in path)
        grid = np.linspace(0.0, top, 60)
        vals = np.array([sim.on_kl_grid(t, key, grid) for t in path])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(vals, 0), np.nanstd(vals, 0)
        ax.plot(grid, mu, color=color, ls=ls, label=f"{label} (n={len(path)})")
        ax.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.2)
    else:
        ax.plot(path["kl"], path[key], color=color, ls=ls, label=label)


def settings(args) -> dict:
    q = args.quick
    s = dict(
        seeds=3 if q else 10,
        grpo_steps=1500 if q else 4000,
        pg_steps=3000 if q else 12000,
        e1_N=(32, 16) if q else (64, 32),
        e2_N=64 if q else 128,
        e2_ratios=6 if q else 11,
        e2_masses=[MASSES] if q else [MASSES, (0.05, 0.15, 0.80), (0.30, 0.10, 0.60)],
        e2_extras=not q,
        e2_steps=3000 if q else 6000,
        e3_N=(64, 32),
        e3opt_Ks=[1, 4, 8, 16, 64] if q else None,
        e4_bins=10 if q else 14,
        e5_N=(16, 8),
        e5_betas=5 if q else 9,
        e5_trace_steps=8000 if q else 30000,
    )
    if args.seeds is not None:
        s["seeds"] = args.seeds
    return s


def bon_grid(n_max: int = 1000, k: int = 30) -> np.ndarray:
    return np.unique(np.round(np.logspace(0, np.log10(n_max), k)).astype(int)).astype(float)


def t2_summary(tag: str, results: list[dict[str, float]]) -> None:
    chain = max(r["chain_err"] for r in results)
    decomp = max(r["decomp_err"] for r in results)
    viol = int(sum(r["bound_violations"] for r in results))
    vec = max(r["vec_vs_loop"] for r in results)
    check(f"{tag}: T2 KL chain rule KL = KL_score + KL_tie", chain < 1e-10,
          f"max |err| = {chain:.2e} over {len(results)} policies")
    check(f"{tag}: T2 gold decomposition dJ = C + D", decomp < 1e-10, f"max |err| = {decomp:.2e}")
    check(f"{tag}: T2 drift bound |D_x| <= sum_t p osc phi(KL_t)", viol == 0, f"{viol} violating prompts")
    check(f"{tag}: vectorized vs loop within-tie KL", vec < 1e-10, f"max |diff| = {vec:.2e}")


# --------------------------------------------------------------------------
# E1: tie preservation
# --------------------------------------------------------------------------

def e1(args, S: dict, out: Path) -> None:
    N, m = S["e1_N"]
    base = dict(n_prompts=N, n_loophole=m, masses=MASSES, n_fail=2)
    iso = sim.shared_feature_problem(**base, variant="isolated")
    models = {
        "isolated": iso,
        "exposed": sim.shared_feature_problem(**base, variant="exposed"),
        "tabular": sim.tabular_problem(iso),
    }
    titles = {
        "isolated": "log-linear, coherent hack, isolated",
        "exposed": "log-linear, coherent hack, exposed",
        "tabular": "tabular softmax",
    }
    grids = {
        "tilt": np.concatenate([[0.0], np.logspace(-2, 1.5, 30)]),
        "chi2": np.logspace(2, -2.5, 30),
        "bon_uniform": bon_grid(),
        "bon_likelihood": bon_grid(),
    }
    kl_max = 2.0
    results, t2_all = {}, []
    for name, prob in models.items():
        print(f" model: {titles[name]}")
        cell = sim.cells(prob)
        paths, pols = {}, []
        for meth, grid in grids.items():
            paths[meth] = sim.score_based_path(prob, meth, grid, cell, keep_policies=True)
            pols += paths[meth].pop("_policies")[::5]
        pg = sim.run_flow(prob, step=0.02, kl_max=kl_max, max_steps=S["pg_steps"],
                          record_every=10, keep_policies=True, cell=cell)
        npg = sim.run_flow(prob, natural=True, step=0.02, kl_max=kl_max,
                           max_steps=min(S["pg_steps"], 2000), record_every=10,
                           keep_policies=True, cell=cell)
        grpo = [sim.run_grpo(prob, np.random.default_rng(1000 + s), steps=S["grpo_steps"],
                             kl_max=kl_max, record_every=10, keep_policies=(s == 0), cell=cell)
                for s in range(S["seeds"])]
        paths["pg"], paths["npg"] = pg["traj"], npg["traj"]
        paths["grpo"] = [g["traj"] for g in grpo]
        pols += pg["policies"][::10] + npg["policies"][::10] + grpo[0]["policies"][::10]
        t2_all += [sim.t2_check(prob, lp) for lp in pols]

        sb = max(float(paths[k]["kl_tie_max"].max()) for k in ("tilt", "chi2", "bon_uniform"))
        check(f"E1/{name}: tilt, chi2, uniform-tie BoN preserve ties (T1)", sb < 1e-10,
              f"max within-tie KL = {sb:.2e}")
        lik = float(paths["bon_likelihood"]["kl_tie_max"].max())
        check(f"E1/{name}: likelihood-tie BoN is not score-based", lik > 1e-4,
              f"max within-tie KL = {lik:.3e}")
        resid = sim.realizability_residual(prob)
        npg_tie = float(paths["npg"]["kl_tie_max"].max())
        if resid < 1e-8:
            check(f"E1/{name}: NPG has zero drift (realizable, Prop. NPG(b))", npg_tie < 1e-8,
                  f"max within-tie KL = {npg_tie:.2e}, realizability residual = {resid:.1e}")
        else:
            check(f"E1/{name}: NPG drifts (non-realizable, Prop. NPG(d))", npg_tie > 1e-6,
                  f"max within-tie KL = {npg_tie:.2e}, realizability residual = {resid:.2f}")
        h = paths["pg"]["hack_share"]
        if name == "tabular":
            pred, what = np.sign(MASSES[1] - MASSES[0]), "sign(b - a) (Prop. tabular)"
            obs = np.sign(h[-1] - h[0])
        elif name == "isolated":
            R = float(prob.info["R_form"][0])
            pred, what = np.sign(R - 1.0), f"sign(R - 1), R = {R:.2f} (Thm. threshold)"
            obs = np.sign(h[-1] - h[0])
        else:
            d0 = float(prob.info["drift0_form"][0])
            pred, what = np.sign(d0), f"sign of initial drift {d0:+.4f} (Prop. initial)"
            obs = np.sign(h[1] - h[0])
        check(f"E1/{name}: PG hack-share direction", pred == obs,
              f"predicted {what}; observed hack share {h[0]:.3f} -> {h[-1]:.3f}")
        results[name] = paths

    t2_summary("E1", t2_all)

    cols = [("hack_share", "hack share among passing (loophole prompts)"),
            ("kl_tie", "within-tie KL"), ("gold", "true reward"), ("proxy", "proxy reward")]
    fig, axes = plt.subplots(len(models), len(cols), figsize=(19, 4.2 * len(models)))
    for r, (name, paths) in enumerate(results.items()):
        for c, (key, ylabel) in enumerate(cols):
            ax = axes[r, c]
            for meth in ("tilt", "chi2", "bon_uniform"):
                plot_path(ax, paths[meth], key, meth)
            plot_path(ax, paths["bon_likelihood"], key, "bon_likelihood", ls="--")
            for meth in ("pg", "npg", "grpo"):
                plot_path(ax, paths[meth], key, meth)
            if key == "hack_share":
                ax.axhline(paths["tilt"]["hack_share0"][0], color="k", lw=0.8, ls=":")
            ax.set_xlabel(r"KL$(\pi\|\pi_0)$")
            ax.set_ylabel(ylabel)
            ax.set_title(titles[name], fontsize=10)
    axes[0, 0].legend(fontsize=7)
    savefig(fig, out, "e1_tie_preservation")
    save_json(results, out / "e1_tie_preservation.json")


# --------------------------------------------------------------------------
# E2: coherence threshold
# --------------------------------------------------------------------------

def e2(args, S: dict, out: Path) -> None:
    N = S["e2_N"]
    ms = [2 ** k for k in range(int(np.log2(N)) + 1)]
    ratios = np.logspace(-1.5, 1.0, S["e2_ratios"])

    def one(variant, masses, m, ratio, gamma=0.0, feature_mode="orthogonal", seed=0) -> dict:
        prob = sim.shared_feature_problem(
            n_prompts=N, n_loophole=m, masses=masses, s_h=1.0, s_u=float(np.sqrt(ratio)),
            gamma=gamma, variant=variant, feature_mode=feature_mode, d_random=4 * N, seed=seed)
        grad, _, _, _ = sim.policy_gradient(prob, np.zeros(prob.dim))
        dz = prob.psi @ grad
        res = sim.run_flow(prob, step=0.05, max_steps=S["e2_steps"], stop_fail_mass=1e-3,
                           record_every=0)
        tr = res["traj"]
        return dict(variant=variant, masses=list(masses), m=m, ratio=float(ratio), gamma=gamma,
                    feature_mode=feature_mode, seed=seed,
                    R=float(prob.info["R_form"][0]), drift_pred=float(prob.info["drift0_form"][0]),
                    drift_obs=float(dz[0, sim.HACK] - dz[0, sim.HONEST]),
                    h0=float(tr["hack_share"][0]), h_end=float(tr["hack_share"][-1]),
                    fail_end=float(tr["fail_loophole"][-1]), steps=int(res["steps"]))

    rows = []
    for variant in ("isolated", "exposed"):
        for masses in S["e2_masses"]:
            for m in ms:
                for ratio in ratios:
                    rows.append(one(variant, masses, m, ratio))
    if S["e2_extras"]:
        for variant in ("isolated", "exposed"):
            for gamma in (0.1, 0.3):
                for m in ms:
                    for ratio in ratios:
                        rows.append(one(variant, MASSES, m, ratio, gamma=gamma))
        for seed in range(3):
            for m in ms:
                for ratio in ratios:
                    rows.append(one("isolated", MASSES, m, ratio, feature_mode="random", seed=seed))

    ortho = [r for r in rows if r["feature_mode"] == "orthogonal"]
    err = max(abs(r["drift_obs"] - r["drift_pred"]) for r in ortho)
    check("E2: initial drift formula (Prop. initial), both variants, all gamma", err < 1e-10,
          f"max |observed - predicted| = {err:.2e} over {len(ortho)} configs")

    def agreement(sel, pred_key):
        sel = [r for r in sel if abs(r["h_end"] - r["h0"]) > 1e-9]
        if pred_key == "R":
            sel = [r for r in sel if abs(np.log(r["R"])) > 0.05]
            ok = [np.sign(r["R"] - 1) == np.sign(r["h_end"] - r["h0"]) for r in sel]
        else:
            ok = [np.sign(r["drift_pred"]) == np.sign(r["h_end"] - r["h0"]) for r in sel]
        return (float(np.mean(ok)) if ok else float("nan")), len(ok)

    iso0 = [r for r in ortho if r["variant"] == "isolated" and r["gamma"] == 0.0]
    acc, n = agreement(iso0, "R")
    check("E2: isolated variant, asymptotic winner = sign(R - 1) (Thm. threshold)", acc == 1.0,
          f"agreement {acc:.3f} over {n} configs")
    exp0 = [r for r in ortho if r["variant"] == "exposed" and r["gamma"] == 0.0]
    acc, n = agreement(exp0, "drift")
    info("E2: exposed variant, end state vs initial-drift sign (Conjecture)",
         f"agreement {acc:.3f} over {n} configs")
    for gamma in (0.1, 0.3):
        for variant in ("isolated", "exposed"):
            sel = [r for r in ortho if r["variant"] == variant and r["gamma"] == gamma]
            if sel:
                acc, n = agreement(sel, "drift")
                info(f"E2: {variant}, gamma={gamma}, end state vs initial-drift sign",
                     f"agreement {acc:.3f} over {n} configs")
    rnd = [r for r in rows if r["feature_mode"] == "random"]
    if rnd:
        acc, n = agreement(rnd, "R")
        info("E2: isolated, random near-orthogonal features, end state vs sign(R - 1)",
             f"agreement {acc:.3f} over {n} configs")

    a, b, c = MASSES
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lm = np.log2(ms)
    lr = np.log10(ratios)
    hstep = (lr[1] - lr[0]) / 2
    m_cont = np.logspace(0, np.log2(N), 200, base=2)
    for ax, variant in zip(axes[:2], ("isolated", "exposed")):
        grid = np.full((len(ratios), len(ms)), np.nan)
        for r in ortho:
            if r["variant"] == variant and r["gamma"] == 0.0 and tuple(r["masses"]) == MASSES:
                grid[list(ratios).index(r["ratio"]), ms.index(r["m"])] = r["h_end"] - r["h0"]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1,
                       extent=[lm[0] - 0.5, lm[-1] + 0.5, lr[0] - hstep, lr[-1] + hstep])
        if variant == "isolated":
            ax.plot(np.log2(m_cont), np.log10(a / (b * m_cont)), "k--", label="theory: R = 1")
        else:
            gap = m_cont * c - (N - m_cont) * a
            ok = gap > 0
            ax.plot(np.log2(m_cont[ok]), np.log10(a * c / (b * gap[ok])), "k--",
                    label="initial drift = 0")
        ax.set_ylim(lr[0] - hstep, lr[-1] + hstep)
        ax.set_xlabel(r"$\log_2 m$ (loophole prompts sharing the form)")
        ax.set_ylabel(r"$\log_{10}(s_u^2/s_h^2)$")
        ax.set_title(f"{variant}: final hack share - initial (PG flow)", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        fig.colorbar(im, ax=ax)
    ax = axes[2]
    for masses in S["e2_masses"]:
        sel = [r for r in iso0 if tuple(r["masses"]) == tuple(masses)]
        ax.scatter([np.log(r["R"]) for r in sel], [r["h_end"] - r["h0"] for r in sel], s=14,
                   label=f"isolated, (a,b,c)={masses}")
    if rnd:
        ax.scatter([np.log(r["R"]) for r in rnd], [r["h_end"] - r["h0"] for r in rnd], s=10,
                   marker="x", label="isolated, random features")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\log R$,  $R = m s_u^2 b / (s_h^2 a)$")
    ax.set_ylabel("final hack share - initial")
    ax.set_title("collapse onto the predicted step at R = 1", fontsize=10)
    ax.legend(fontsize=7)
    savefig(fig, out, "e2_threshold")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for gamma in (0.0, 0.1, 0.3):
        sel = [r for r in ortho if r["variant"] == "exposed" and r["gamma"] == gamma]
        if sel:
            ax.scatter([N * r["drift_pred"] for r in sel], [r["h_end"] - r["h0"] for r in sel],
                       s=12, label=f"exposed, gamma={gamma}")
    sel = [r for r in ortho if r["variant"] == "isolated" and r["gamma"] > 0]
    if sel:
        ax.scatter([N * r["drift_pred"] for r in sel], [r["h_end"] - r["h0"] for r in sel],
                   s=12, marker="^", label="isolated, gamma>0")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$N\times$ predicted initial drift")
    ax.set_ylabel("final hack share - initial")
    ax.set_title("does the initial drift sign predict the end state?", fontsize=10)
    ax.legend(fontsize=7)
    savefig(fig, out, "e2_initial_vs_final")
    save_json(rows, out / "e2_threshold.json")


# --------------------------------------------------------------------------
# E3: surface-form randomization
# --------------------------------------------------------------------------

def e3(args, S: dict, out: Path) -> None:
    N, m = S["e3_N"]
    a, b, c = MASSES
    Ks = [2 ** k for k in range(int(np.log2(N)) + 1)]
    kl_budget, kl_max = 1.0, 1.6
    rows, trajs = [], {}
    bon_ref, bon_diff = None, 0.0
    for variant in ("isolated", "exposed"):
        for K in Ks:
            prob = sim.shared_feature_problem(n_prompts=N, n_loophole=m, masses=MASSES,
                                              n_forms=K, variant=variant)
            cell = sim.cells(prob)
            lp = sim.bon_logp(prob, 64)
            bon_ref = lp if bon_ref is None else bon_ref
            bon_diff = max(bon_diff, float(np.abs(lp - bon_ref).max()))
            bon = sim.score_based_path(prob, "bon_uniform", bon_grid(), cell)
            pg = sim.run_flow(prob, step=0.02, kl_max=kl_max, max_steps=S["pg_steps"],
                              record_every=10, cell=cell)["traj"]
            grpo = [sim.run_grpo(prob, np.random.default_rng(2000 + s), steps=S["grpo_steps"],
                                 kl_max=kl_max, record_every=10, cell=cell)["traj"]
                    for s in range(S["seeds"])]
            adam = ([sim.run_grpo(prob, np.random.default_rng(3000 + s), steps=S["grpo_steps"],
                                  kl_max=kl_max, record_every=10, adam=True, cell=cell)["traj"]
                     for s in range(S["seeds"])] if args.adam else [])
            used = np.unique(prob.info["form"][prob.loophole])
            row = dict(variant=variant, K=K,
                       R=float(prob.info["R_form"][used].min()),
                       R_max=float(prob.info["R_form"][used].max()),
                       drift0=float(prob.info["drift0_form"][used].min()),
                       drift0_max=float(prob.info["drift0_form"][used].max()),
                       h0=float(pg["hack_share0"][0]))
            for key in ("hack_share", "gold"):
                row[f"bon_{key}"] = sim.value_at_kl(bon, key, kl_budget)
                row[f"pg_{key}"] = sim.value_at_kl(pg, key, kl_budget)
                for tag, runs in (("grpo", grpo), ("adam", adam)):
                    vals = np.array([sim.value_at_kl(t, key, kl_budget) for t in runs])
                    if vals.size:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            row[f"{tag}_{key}"] = float(np.nanmean(vals))
                            row[f"{tag}_{key}_std"] = float(np.nanstd(vals))
                        row[f"{tag}_reached"] = int(np.isfinite(vals).sum())
            rows.append(row)
            trajs[f"{variant}/K={K}"] = dict(bon=bon, pg=pg, grpo=grpo, adam=adam)
            print(f"  {variant:8s} K={K:3d}  R={row['R']:.2f}  hack share @KL={kl_budget}: "
                  f"BoN {row['bon_hack_share']:.3f}  PG {row['pg_hack_share']:.3f}  "
                  f"GRPO {row.get('grpo_hack_share', float('nan')):.3f}")

    check("E3: BoN policy identical for every K and both variants (T1)", bon_diff < 1e-12,
          f"max |log pi_BoN(K) - log pi_BoN(K=1)| at n=64 = {bon_diff:.1e}")
    iso = [r for r in rows if r["variant"] == "isolated"]
    ok = [np.sign(r["pg_hack_share"] - r["h0"]) == np.sign(r["R"] - 1) for r in iso
          if np.isfinite(r["pg_hack_share"])]
    check("E3: isolated PG hack share at KL budget moves in direction sign(R - 1)",
          len(ok) == len(iso) and all(ok), f"{sum(ok)}/{len(iso)} values of K agree")
    for variant in ("isolated", "exposed"):
        sel = [r for r in rows if r["variant"] == variant]
        info(f"E3: {variant}, hack share @KL={kl_budget}, K=1 vs K={N}",
             f"PG {sel[0]['pg_hack_share']:.3f} vs {sel[-1]['pg_hack_share']:.3f}; "
             f"GRPO {sel[0].get('grpo_hack_share', float('nan')):.3f} vs "
             f"{sel[-1].get('grpo_hack_share', float('nan')):.3f}; "
             f"BoN {sel[0]['bon_hack_share']:.3f} vs {sel[-1]['bon_hack_share']:.3f}")

    k_iso = m * b / a
    k_exp = b * (m * c - (N - m) * a) / (a * c)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, variant in enumerate(("isolated", "exposed")):
        sel = [r for r in rows if r["variant"] == variant]
        x = np.log2([r["K"] for r in sel])
        for r_, (key, ylabel) in enumerate((("hack_share", "hack share among passing"),
                                            ("gold", "true reward"))):
            ax = axes[r_, col]
            ax.plot(x, [r[f"bon_{key}"] for r in sel], "o-", color=COLORS["bon_uniform"],
                    label=LABELS["bon_uniform"])
            ax.plot(x, [r[f"pg_{key}"] for r in sel], "s-", color=COLORS["pg"], label=LABELS["pg"])
            for tag, meth in (("grpo", "grpo"), ("adam", "grpo_adam")):
                if f"{tag}_{key}" in sel[0]:
                    ax.errorbar(x, [r[f"{tag}_{key}"] for r in sel],
                                yerr=[r[f"{tag}_{key}_std"] for r in sel], fmt="^-",
                                color=COLORS[meth], capsize=3, label=LABELS[meth])
            kstar = k_iso if variant == "isolated" else k_exp
            if kstar > 0:
                ax.axvline(np.log2(kstar), color="k", ls="--", lw=0.8,
                           label=r"$K^\star$" if variant == "isolated" else "initial drift = 0")
            ax.set_xlabel(r"$\log_2 K$ (number of surface forms)")
            ax.set_ylabel(f"{ylabel} at KL = {kl_budget}")
            ax.set_title(f"{variant}: N={N}, m={m}, (a,b,c)={MASSES}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    savefig(fig, out, "e3_surface_form")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for r_, variant in enumerate(("isolated", "exposed")):
        for col, K in enumerate((1, N)):
            ax = axes[r_, col]
            tr = trajs[f"{variant}/K={K}"]
            plot_path(ax, tr["bon"], "hack_share", "bon_uniform")
            plot_path(ax, tr["pg"], "hack_share", "pg")
            plot_path(ax, tr["grpo"], "hack_share", "grpo")
            if tr["adam"]:
                plot_path(ax, tr["adam"], "hack_share", "grpo_adam")
            ax.axhline(b / (a + b), color="k", lw=0.8, ls=":")
            ax.set_xlabel(r"KL$(\pi\|\pi_0)$")
            ax.set_ylabel("hack share among passing")
            ax.set_title(f"{variant}, K={K} ({'coherent' if K == 1 else 'randomized'})",
                         fontsize=10)
    axes[0, 0].legend(fontsize=8)
    savefig(fig, out, "e3_trajectories")
    save_json(dict(rows=rows, trajectories=trajs), out / "e3_surface_form.json")
    if args.opt_sweep or args.basis is not None:
        e3_optimizers(args, S, out)


# --------------------------------------------------------------------------
# E3 (opt-in): optimizer x parameter basis (draft: "Preconditioned and adaptive optimizers")
# --------------------------------------------------------------------------

DET_ARMS = {"pg": "PG flow (exact grad.)", "signflow": "sign flow (exact grad.)"}


def opt_arms(sweep: bool) -> list[dict]:
    arms = [dict(tag="sgd", label="GRPO-SGD", optimizer="sgd"),
            dict(tag="adam_lr0.01_b0.999", label="GRPO-Adam lr=0.01 b2=0.999", optimizer="adam",
                 lr=0.01, b2=0.999),
            dict(tag="sign_lr0.01", label="GRPO-signSGD lr=0.01", optimizer="sign", lr=0.01)]
    if sweep:
        for lr in (0.003, 0.01, 0.03):
            for b2 in (0.99, 0.999):
                if (lr, b2) != (0.01, 0.999):
                    arms.append(dict(tag=f"adam_lr{lr:g}_b{b2:g}", label=f"GRPO-Adam lr={lr:g} b2={b2:g}",
                                     optimizer="adam", lr=lr, b2=b2))
        arms.append(dict(tag="rmsprop_lr0.01_b0.99", label="GRPO-RMSprop lr=0.01 b2=0.99",
                         optimizer="rmsprop", lr=0.01, b2=0.99))
        arms.append(dict(tag="adagrad_lr0.1", label="GRPO-Adagrad lr=0.1", optimizer="adagrad", lr=0.1))
    return arms


def opt_bases(args) -> list[str]:
    bases = ["axis"]
    for basis in (args.basis if args.basis is not None else ["rotated", "dense"]):
        if basis not in bases:
            bases.append(basis)
    return bases


def _maxdiff(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
    return float(np.where(np.isnan(diff), np.inf, diff).max())


def e3_optimizers(args, S: dict, out: Path) -> None:
    """Hack share at the KL budget vs K for several optimizers and parameter bases.

    Learning rates of the adaptive arms are in axis-equivalent units: in a
    non-axis basis they are multiplied by the ratio of the first sign-step
    log-prob move in the axis basis to that in the basis, so all bases take
    comparable numbers of steps (in a random basis a sign step moves logits
    ~sqrt(2d/pi) times further). SGD needs no calibration.
    """
    N, m = S["e3_N"]
    a, b, c = MASSES
    h0 = b / (a + b)
    Ks = sorted(S["e3opt_Ks"] or [2 ** k for k in range(int(np.log2(N)) + 1)])
    if Ks[0] != 1 or Ks[-1] != N:
        raise ValueError("the K grid must contain 1 and N")
    bases = opt_bases(args)
    arms = opt_arms(args.opt_sweep)
    seeds = S["seeds"]
    kl_budget, kl_max = 1.0, 1.6
    sf_step, G = 0.01, 8
    print(f"\n  --- optimizer x basis: bases {bases}, K in {Ks}, {len(arms)} GRPO arms x {seeds} seeds ---")

    rho_h, rho_k = sim.signsgd_occupancy(a, b, G)
    for K in sorted({1, 8, N}):
        prob = sim.shared_feature_problem(n_prompts=N, n_loophole=m, masses=MASSES, n_forms=K,
                                          variant="isolated")
        lp = sim.policy_logp(prob, np.zeros(prob.dim))
        forms = np.unique(prob.info["form"][prob.loophole])
        hon, hk = 1 + np.nonzero(prob.loophole)[0], 1 + N + forms
        rng = np.random.default_rng(6000 + K)
        n_draws = 2000
        pos_h = pos_k = neg = 0
        for _ in range(n_draws):
            g = sim.grpo_gradient(prob, lp, rng, group=G)
            tol = 1e-12 * float(np.abs(g).max())
            pos_h += int((g[hon] > tol).sum())
            pos_k += int((g[hk] > tol).sum())
            neg += int((g[hon] < -tol).sum() + (g[hk] < -tol).sum())
        pred_k = 1.0 - (1.0 - rho_k) ** prob.info["m_g"][forms]
        f_h, f_k = pos_h / (n_draws * hon.size), pos_k / (n_draws * hk.size)
        se_h = np.sqrt(rho_h * (1 - rho_h) / (n_draws * hon.size))
        se_k = np.sqrt(float(np.mean(pred_k * (1 - pred_k))) / (n_draws * hk.size))
        ok = (neg == 0 and abs(f_h - rho_h) < 4 * se_h + 2e-3
              and abs(f_k - float(pred_k.mean())) < 4 * se_k + 2e-3)
        check(f"E3opt: GRPO gradient >= 0 on honest/form coordinates, P(> 0) = occupancy "
              f"(Prop. signSGD; isolated, axis, theta=0, K={K})", ok,
              f"honest {f_h:.4f} vs rho_h {rho_h:.4f}; form {f_k:.4f} vs 1-(1-rho_k)^m_g "
              f"{float(pred_k.mean()):.4f}; negative entries {neg}; {n_draws} draws")
    drift = {K: 1.0 - (1.0 - rho_k) ** int(np.bincount(np.arange(m) % K).max()) - rho_h for K in Ks}
    info("E3opt: Prop. signSGD expected initial drift / lr (isolated, axis, s_u = s_h; > 0 favours hack)",
         "; ".join(f"K={K}: {v:+.3f}" for K, v in drift.items())
         + f"  [rho_h = {rho_h:.3f}, rho_k = {rho_k:.3f}; SGD analogue: R_g - 1 with R_g = m_g b/a]",
         initial_drift_signsgd={str(K): v for K, v in drift.items()})

    table: dict[tuple, dict] = {}
    rows: list[dict] = []
    rot_pg, rot_sgd, kern_err, dense_vs_axis = [], [], [], []
    kern_dev = None
    sf_err = {"isolated": 0.0, "exposed": 0.0}

    def put(variant, K, basis, arm, label, vals, det):
        vals = np.asarray(vals, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
        row = dict(variant=variant, K=K, basis=basis, arm=arm, label=label, deterministic=det,
                   values=vals.tolist(), mean=mu, std=sd, reached=int(np.isfinite(vals).sum()), n=vals.size)
        rows.append(row)
        table.setdefault((variant, basis, arm), {})[K] = row

    for variant in ("isolated", "exposed"):
        for K in Ks:
            t0 = time.time()
            base = sim.shared_feature_problem(n_prompts=N, n_loophole=m, masses=MASSES, n_forms=K,
                                              variant=variant)
            cell = sim.cells(base)
            ref_size = sim.first_step_size(base)
            pg_val, sgd_vals = {}, {}
            for basis in bases:
                lin = "dense" if basis == "factored" else basis
                hidden = (args.factor_width or base.dim) if basis == "factored" else 0
                probs = [sim.reparametrize(base, lin, seed=100 + s, dense_mult=args.dense_mult)
                         for s in range(seeds)]
                if basis != "factored":
                    pg = sim.run_flow(probs[0], step=0.02, kl_max=kl_max, max_steps=S["pg_steps"],
                                      record_every=10, cell=cell)["traj"]
                    pg_val[basis] = sim.value_at_kl(pg, "hack_share", kl_budget)
                    put(variant, K, basis, "pg", DET_ARMS["pg"], [pg_val[basis]], True)
                    if basis == "dense":
                        Bm = probs[0].info["B"]
                        kern = Bm.T @ Bm
                        pk = sim.run_flow(base, step=0.02, kl_max=kl_max, max_steps=S["pg_steps"],
                                          record_every=10, cell=cell, precond=kern)["traj"]
                        kern_err.append(_maxdiff([pg_val[basis]], [sim.value_at_kl(pk, "hack_share", kl_budget)]))
                        if kern_dev is None:
                            kern_dev = float(np.linalg.norm(kern - np.eye(kern.shape[0]), 2))
                    sfv = []
                    for s in range(1 if basis == "axis" else seeds):
                        res = sim.run_flow(probs[s], sign=True, step=sf_step, kl_max=kl_max,
                                           max_steps=S["pg_steps"], record_every=10, cell=cell,
                                           keep_policies=(basis == "axis"))
                        sfv.append(sim.value_at_kl(res["traj"], "hack_share", kl_budget))
                        if basis == "axis":
                            for lp in res["policies"]:
                                obs, pred = sim.signflow_prediction(base, lp)
                                sf_err[variant] = max(sf_err[variant], float(np.abs(obs - pred).max()))
                    put(variant, K, basis, "signflow", DET_ARMS["signflow"], sfv, basis == "axis")
                for arm in arms:
                    steps = int(np.ceil(2 * S["grpo_steps"] * max(1.0, 0.01 / arm.get("lr", 0.01))))
                    vals = []
                    for s in range(seeds):
                        kw = dict(optimizer=arm["optimizer"], hidden=hidden, factor_init=args.factor_init,
                                  factor_seed=s)
                        if arm["optimizer"] != "sgd":
                            size = sim.first_step_size(probs[s], hidden=hidden, factor_init=args.factor_init,
                                                       factor_seed=s)
                            kw.update(opt_lr=arm["lr"] * ref_size / max(size, 1e-300),
                                      adam_b2=arm.get("b2", 0.999))
                        res = sim.run_grpo(probs[s], np.random.default_rng(4000 + s), steps=steps,
                                           kl_max=kl_max, record_every=5, cell=cell, **kw)
                        vals.append(sim.value_at_kl(res["traj"], "hack_share", kl_budget))
                    put(variant, K, basis, arm["tag"], arm["label"], vals, False)
                    if arm["tag"] == "sgd":
                        sgd_vals[basis] = vals
            if "rotated" in bases:
                rot_pg.append(_maxdiff([pg_val["rotated"]], [pg_val["axis"]]))
                rot_sgd.append(_maxdiff(sgd_vals["rotated"], sgd_vals["axis"]))
            if "dense" in bases:
                dense_vs_axis.append(_maxdiff([pg_val["dense"]], [pg_val["axis"]]))
            line = "  ".join(f"{basis}: SGD {table[(variant, basis, 'sgd')][K]['mean']:.3f} "
                             f"Adam {table[(variant, basis, 'adam_lr0.01_b0.999')][K]['mean']:.3f} "
                             f"sign {table[(variant, basis, 'sign_lr0.01')][K]['mean']:.3f}"
                             for basis in bases)
            print(f"  {variant:8s} K={K:3d}  {line}  ({time.time() - t0:.1f} s)")

    if "rotated" in bases:
        check("E3opt: exact PG flow invariant under a random rotation Q of the coordinates",
              max(rot_pg) < 1e-6,
              f"max |hack share(rotated) - hack share(axis)| at KL={kl_budget} = {max(rot_pg):.1e} over "
              f"{len(rot_pg)} (variant, K); expected: equal up to round-off, since gradient steps "
              f"commute with orthogonal Q (kernel Q^T Q = I)")
        check("E3opt: GRPO-SGD invariant under rotation (same sampling seeds)", max(rot_sgd) < 1e-6,
              f"max |diff| over seeds and (variant, K) = {max(rot_sgd):.1e}; expected: equal up to "
              f"round-off (a seed-noise-sized mismatch with the PG check passing would be a "
              f"floating-point flip in a sampling comparison, not a bug)")
    if "dense" in bases:
        check("E3opt: dense-basis PG flow = axis PG flow with kernel P = E^T E",
              max(kern_err) < 1e-6,
              f"max |diff| = {max(kern_err):.1e}; expected: equal up to round-off (gradient steps on "
              f"w with theta = E^T w are steps on theta with kernel E^T E)")
        info("E3opt: dense vs axis PG flow (kernel effect, not an error)",
             f"max |hack share(dense) - hack share(axis)| at KL={kl_budget} = {max(dense_vs_axis):.3f}; "
             f"||E^T E - I||_op = {kern_dev:.2f} (Marchenko-Pastur edge ~ 2 sqrt(d/p) + d/p; "
             f"raise --dense-mult to shrink it)",
             dense_vs_axis=max(dense_vs_axis), kernel_dev=kern_dev)
    for variant in ("isolated", "exposed"):
        tol = 1e-9 if variant == "isolated" else 4 * sf_step + 1e-9
        what = ("L = 0: hack share stays b/(a+b) for every K" if variant == "isolated"
                else "L = -2 (eta - log(m_g c/(m'_g a)))_+")
        check(f"E3opt/{variant}: exact sign flow in the axis basis follows Prop. signflow ({what})",
              sf_err[variant] < tol,
              f"max |L_obs - L_pred| over checkpoints and loophole prompts = {sf_err[variant]:.1e} "
              f"(tolerance {tol:.1e} = discretization bound)")

    arm_order = list(DET_ARMS) + [arm["tag"] for arm in arms]
    labels = {**DET_ARMS, **{arm["tag"]: arm["label"] for arm in arms}}
    summary = []
    for variant in ("isolated", "exposed"):
        for basis in bases:
            for arm in arm_order:
                byK = table.get((variant, basis, arm))
                if not byK:
                    continue
                r1, rN = byK[1], byK[N]
                gap = r1["mean"] - rN["mean"]
                sig = float(np.hypot(r1["std"], rN["std"]))
                det = r1["deterministic"] and rN["deterministic"]
                big = bool(np.isfinite(gap) and gap > 2 * sig)
                ent = dict(variant=variant, basis=basis, arm=arm, h_K1=r1["mean"], h_K1_std=r1["std"],
                           h_KN=rN["mean"], h_KN_std=rN["std"], gap=gap, gap_std=sig, gap_gt_2std=big,
                           deterministic=det, reached_K1=f"{r1['reached']}/{r1['n']}",
                           reached_KN=f"{rN['reached']}/{rN['n']}")
                summary.append(ent)
                info(f"E3opt/{variant}/{basis}/{arm}: coherence gap h(K=1) - h(K={N}) @KL={kl_budget}",
                     f"{r1['mean']:.3f}+-{r1['std']:.3f} vs {rN['mean']:.3f}+-{rN['std']:.3f}; "
                     f"gap {gap:+.3f}; gap > 2 std: {'YES' if big else 'no'}"
                     + (" (deterministic)" if det else f" (std = seed std of the difference, "
                                                     f"reached {ent['reached_K1']}, {ent['reached_KN']})"),
                     **ent)
    print(f"\n  coherence gap h(K=1) - h(K={N}) at KL={kl_budget} (BoN: 0 for every basis; * = gap > 2 std)")
    print(f"  {'variant':9s}{'arm':30s}" + "".join(f"{basis:>14s}" for basis in bases))
    for variant in ("isolated", "exposed"):
        for arm in arm_order:
            cells_ = []
            for basis in bases:
                e = next((x for x in summary if (x["variant"], x["basis"], x["arm"]) == (variant, basis, arm)),
                         None)
                cells_.append(f"{'-':>14s}" if e is None else f"{e['gap']:+12.3f}{'*' if e['gap_gt_2std'] else ' '} ")
            print(f"  {variant:9s}{labels[arm][:29]:30s}" + "".join(cells_))

    fig, axes = plt.subplots(2, len(bases), figsize=(5.2 * len(bases), 9), squeeze=False)
    x = np.log2(Ks)
    n_arm = len(arm_order)
    for r_, variant in enumerate(("isolated", "exposed")):
        for c_, basis in enumerate(bases):
            ax = axes[r_, c_]
            for j, arm in enumerate(arm_order):
                byK = table.get((variant, basis, arm))
                if not byK:
                    continue
                if arm == "pg":
                    style = dict(color="k", marker="s", ls="-")
                elif arm == "signflow":
                    style = dict(color="k", marker="D", ls="--", mfc="none")
                else:
                    style = dict(color=f"C{(j - len(DET_ARMS)) % 10}", marker="^", ls="-")
                jit = 0.3 * (j / max(1, n_arm - 1) - 0.5)
                ax.errorbar(x + jit, [byK[K]["mean"] for K in Ks], yerr=[byK[K]["std"] for K in Ks],
                            capsize=2, ms=4, lw=1, label=labels[arm], **style)
            ax.axhline(h0, color="k", lw=0.8, ls=":", label=r"BoN = $\pi_0$ share")
            if variant == "isolated":
                ax.axvline(np.log2(m * b / a), color="0.6", ls="--", lw=0.8, label=r"SGD $K^\star$")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel(r"$\log_2 K$ (number of surface forms)")
            ax.set_ylabel(f"hack share among passing at KL = {kl_budget}")
            ax.set_title(f"{variant}, basis = {basis}", fontsize=10)
    axes[0, 0].legend(fontsize=6)
    savefig(fig, out, "e3_optimizers")
    save_json(dict(rows=rows, summary=summary,
                   settings=dict(bases=bases, Ks=Ks, arms=arms, seeds=seeds, dense_mult=args.dense_mult,
                                 factor_init=args.factor_init, factor_width=args.factor_width,
                                 kl_budget=kl_budget, signflow_step=sf_step)),
              out / "e3_optimizers.json")


# --------------------------------------------------------------------------
# E4: T2 bound tightness and binning
# --------------------------------------------------------------------------

def e4(args, S: dict, out: Path) -> None:
    base = dict(n_prompts=32, n_loophole=16, masses=MASSES, n_fail=2)
    iso = sim.shared_feature_problem(**base, variant="isolated")
    configs = {
        "isolated K=1": iso,
        "exposed K=1": sim.shared_feature_problem(**base, variant="exposed"),
        "tabular": sim.tabular_problem(iso),
        "isolated, random features": sim.shared_feature_problem(
            **base, variant="isolated", feature_mode="random", d_random=128, seed=1),
        "isolated, hack gold = -1": sim.shared_feature_problem(**base, variant="isolated",
                                                               hack_gold=-1.0),
    }
    pts, t2_all = {}, []
    for name, prob in configs.items():
        pg = sim.run_flow(prob, step=0.02, kl_max=2.0, max_steps=S["pg_steps"], record_every=5,
                          keep_policies=True)
        gr = sim.run_grpo(prob, np.random.default_rng(7), steps=S["grpo_steps"], kl_max=2.0,
                          record_every=10, keep_policies=True)
        t2_all += [sim.t2_check(prob, lp) for lp in pg["policies"][::20] + gr["policies"][::20]]
        pts[name] = {k: np.concatenate([pg["traj"][k], gr["traj"][k]])
                     for k in ("drift", "bound_fine", "bound_jensen", "bound_pinsker", "kl_tie",
                               "hack_share0")}
    t2_summary("E4", t2_all)
    allp = {k: np.concatenate([p[k] for p in pts.values()]) for k in next(iter(pts.values()))}
    absd = np.abs(allp["drift"])
    v_fine = int((absd > allp["bound_fine"] + 1e-12).sum())
    v_jen = int((absd > allp["bound_jensen"] + 1e-12).sum())
    v_pin = int((absd > allp["bound_pinsker"] + 1e-12).sum())
    check("E4: |D| <= refined bound on all checkpoints", v_fine == 0,
          f"{v_fine} violations / {absd.size}")
    check("E4: |D| <= osc * phi(mean KL_tie) on all checkpoints", v_jen == 0, f"{v_jen} violations")
    check("E4: |D| <= osc * sqrt(KL_tie / 2) on all checkpoints", v_pin == 0, f"{v_pin} violations")
    mask = allp["bound_pinsker"] > 1e-6
    ratio = float((absd[mask] / allp["bound_pinsker"][mask]).max()) if mask.any() else float("nan")
    h0 = MASSES[1] / (MASSES[0] + MASSES[1])
    info("E4: tightness", f"max |D| / Pinsker bound = {ratio:.3f}; small-deviation limit for a "
                          f"pure hack/honest split 2 sqrt(h0(1-h0)) = {2 * np.sqrt(h0 * (1 - h0)):.3f}")

    prob = sim.shared_feature_problem(n_prompts=32, n_loophole=16, masses=MASSES, n_fail=3,
                                      proxy_noise=0.2, variant="isolated", seed=3)
    deltas = 2.0 ** -np.arange(S["e4_bins"])
    pols = {
        "tilt tau=2": (sim.tilt_logp(prob, 2.0), 2.0),
        "tilt tau=8": (sim.tilt_logp(prob, 8.0), 8.0),
        "BoN n=16": (sim.bon_logp(prob, 16), None),
        "PG (KL=1)": (sim.run_flow(prob, step=0.02, kl_max=1.0, max_steps=S["pg_steps"],
                                   record_every=0)["logp"], None),
        "GRPO (KL=1)": (sim.run_grpo(prob, np.random.default_rng(11), steps=S["grpo_steps"],
                                     kl_max=1.0, record_every=10)["logp"], None),
    }
    binning = {}
    for name, (lp, tau) in pols.items():
        recs = [sim.diagnostics(prob, lp, sim.cells(prob, float(dl))) for dl in deltas]
        binning[name] = dict(delta=deltas, kl_tie=np.array([r["kl_tie"] for r in recs]),
                             drift=np.array([r["drift"] for r in recs]),
                             calib=np.array([r["calib"] for r in recs]),
                             kl=recs[0]["kl"])
        inc = float(np.diff(binning[name]["kl_tie"]).max())
        check(f"E4: within-bin KL nonincreasing under dyadic refinement [{name}]", inc < 1e-12,
              f"max increase = {inc:.1e}")
        if tau is not None:
            excess = float((binning[name]["kl_tie"] - tau ** 2 * deltas ** 2 / 8).max())
            check(f"E4: tilt within-bin KL <= tau^2 delta^2 / 8 [{name}]", excess < 1e-12,
                  f"max excess = {excess:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for name, p in pts.items():
        mk = p["bound_pinsker"] > 1e-9
        ax.scatter(p["bound_pinsker"][mk], np.abs(p["drift"][mk]) + 1e-18, s=6, label=name)
    lim = [1e-6, 1.0]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1.0)
    ax.set_xlabel(r"$\bar o\,\sqrt{\mathrm{KL}_{\rm tie}/2}$")
    ax.set_ylabel(r"$|\mathcal{D}|$ (gold gap to score-based projection)")
    ax.set_title("T2 drift bound (PG and GRPO checkpoints)", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1]
    for name, bres in binning.items():
        ax.plot(bres["delta"], np.maximum(bres["kl_tie"], 1e-18), "o-", ms=3, label=name)
    for tau in (2.0, 8.0):
        ax.plot(deltas, tau ** 2 * deltas ** 2 / 8, "--", lw=0.8, label=rf"$\tau^2\delta^2/8$, $\tau$={tau:g}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 10)
    ax.set_xlabel(r"bin width $\delta$ (proxy noise scale 0.2)")
    ax.set_ylabel("within-bin KL")
    ax.set_title("continuous proxy: within-bin KL vs resolution", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[2]
    for name, bres in binning.items():
        ax.plot(bres["delta"], bres["drift"], "o-", ms=3, label=f"{name}: drift")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"bin width $\delta$")
    ax.set_ylabel(r"drift term $\mathcal{D}^{(\delta)}$")
    ax.set_title("calibration/drift split depends on resolution", fontsize=10)
    ax.legend(fontsize=7)
    savefig(fig, out, "e4_bound_and_binning")
    save_json(dict(points=pts, binning=binning), out / "e4_bound_and_binning.json")


# --------------------------------------------------------------------------
# E5: KL-regularized stationary drift
# --------------------------------------------------------------------------

def e5(args, S: dict, out: Path) -> None:
    N, m = S["e5_N"]
    base = dict(n_prompts=N, n_loophole=m, masses=MASSES, n_fail=1)
    iso = sim.shared_feature_problem(**base, variant="isolated")
    models = {
        "tabular": sim.tabular_problem(iso),
        "isolated K=1": iso,
        "exposed K=1": sim.shared_feature_problem(**base, variant="exposed"),
        f"exposed K={N} (randomized)": sim.shared_feature_problem(**base, variant="exposed",
                                                                  n_forms=N),
    }
    betas = np.logspace(-1, 1, S["e5_betas"])
    rows = []
    for name, prob in models.items():
        resid = sim.realizability_residual(prob)
        for beta in betas:
            st = sim.stationary_point(prob, float(beta))
            d = sim.diagnostics(prob, st["logp"])
            sb = sim.score_based_restricted_optimum(prob, float(beta))
            rows.append(dict(model=name, beta=float(beta), residual=resid, kl_tie=d["kl_tie"],
                             hack_share=d["hack_share"], hack_share0=d["hack_share0"],
                             grad_norm=st["grad_norm"], steps=st["steps"],
                             J=sim.objective(prob, st["logp"], float(beta)), J_sb=sb["J"],
                             grad_sb=sb["grad_norm"], dim_sb=sb["dim_sb"]))
        sel = [r for r in rows if r["model"] == name]
        kt = [r["kl_tie"] for r in sel]
        gs = [r["grad_sb"] for r in sel]
        if resid < 1e-8:
            check(f"E5/{name}: realizable => stationary within-tie KL = 0 (Thm. stationary(b))",
                  max(kt) < 1e-9 and max(gs) < 1e-6,
                  f"residual {resid:.1e}; max KL_tie {max(kt):.1e}; max |grad J(theta_SB)| {max(gs):.1e}")
        else:
            check(f"E5/{name}: grad J(theta_SB) != 0 => stationary within-tie KL > 0 (Thm. stationary(c,e))",
                  min(kt) > 1e-6 and min(gs) > 1e-6 and all(r["J"] > r["J_sb"] for r in sel),
                  f"residual {resid:.2f}; min KL_tie {min(kt):.2e}; min |grad J(theta_SB)| {min(gs):.2e}")
    gmax = max(r["grad_norm"] for r in rows)
    check("E5: stationary-point solver converged", gmax < 1e-7, f"max |grad J_beta| = {gmax:.1e}")

    beta_tr = 0.5
    traces = {}
    for name in ("isolated K=1", "exposed K=1"):
        prob = models[name]
        steps = S["e5_trace_steps"]
        res = sim.run_flow(prob, beta=beta_tr, step=0.02, eta_max=2.0, grad_tol=1e-11,
                           max_steps=steps, record_every=max(1, steps // 400))
        ref = sim.diagnostics(prob, sim.stationary_point(prob, beta_tr)["logp"])
        tr = res["traj"]
        traces[name] = tr
        gap = abs(tr["kl_tie"][-1] - ref["kl_tie"])
        tol = 1e-6 + 1e-2 * ref["kl_tie"]
        if res["grad_norm"] < 1e-7:
            check(f"E5/{name}: vanilla PG (beta={beta_tr}) ends at the NPG stationary point",
                  gap < tol, f"KL_tie PG {tr['kl_tie'][-1]:.3e} vs NPG {ref['kl_tie']:.3e}")
        else:
            info(f"E5/{name}: vanilla PG not converged in {steps} steps",
                 f"|grad| = {res['grad_norm']:.1e}; KL_tie PG {tr['kl_tie'][-1]:.3e} vs "
                 f"stationary {ref['kl_tie']:.3e}")
    tr = traces["isolated K=1"]
    check("E5/isolated K=1: drift is transient (peak > 1e-4, final < 1e-6)",
          tr["kl_tie"].max() > 1e-4 and tr["kl_tie"][-1] < 1e-6,
          f"peak KL_tie {tr['kl_tie'].max():.2e}, final {tr['kl_tie'][-1]:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for name in models:
        sel = [r for r in rows if r["model"] == name]
        ax.plot([r["beta"] for r in sel], [max(r["kl_tie"], 1e-18) for r in sel], "o-", label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("within-tie KL at the stationary point")
    ax.set_title("KL-regularized optimum: persistent drift iff it leaves ker M", fontsize=10)
    ax.legend(fontsize=8)
    for ax, key, ylabel in ((axes[1], "kl_tie", "within-tie KL"),
                            (axes[2], "hack_share", "hack share among passing")):
        for name, tr in traces.items():
            ax.plot(np.maximum(tr["step"], 1), tr[key], label=name)
        ax.set_xscale("log")
        ax.set_xlabel("PG step")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"vanilla PG with KL penalty $\beta$={beta_tr}", fontsize=10)
        ax.legend(fontsize=8)
    axes[2].axhline(MASSES[1] / (MASSES[0] + MASSES[1]), color="k", lw=0.8, ls=":")
    savefig(fig, out, "e5_stationary")
    save_json(dict(rows=rows, traces=traces), out / "e5_stationary.json")


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tie-drift simulations (E1-E5).")
    ap.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--quick", action="store_true", help="small grids, few seeds (< 5 min total)")
    ap.add_argument("--seeds", type=int, default=None, help="override number of GRPO seeds")
    ap.add_argument("--adam", action="store_true", help="E3: also run GRPO with Adam")
    ap.add_argument("--opt-sweep", action="store_true",
                    help="E3: optimizer x basis study with the full optimizer sweep (Adam lr x beta2, "
                         "signSGD, RMSprop, Adagrad); bases default to axis rotated dense")
    ap.add_argument("--basis", nargs="+", choices=["axis", "rotated", "dense", "factored"], default=None,
                    help="E3: parameter bases for the optimizer study (axis is always included). "
                         "Without --opt-sweep only GRPO-SGD, GRPO-Adam (lr 0.01, beta2 0.999) and "
                         "signSGD run. 'factored' = dense basis + two-layer theta = U v (experimental)")
    ap.add_argument("--dense-mult", type=float, default=4.0, help="dense basis width p = dense_mult * d")
    ap.add_argument("--factor-init", type=float, default=0.1, help="factored basis: scale of U0")
    ap.add_argument("--factor-width", type=int, default=None,
                    help="factored basis: width of v (default: d of the axis model)")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    out = args.out / ("quick" if args.quick else "full")
    out.mkdir(parents=True, exist_ok=True)
    S = settings(args)
    todo = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    t_all = time.time()
    for name in todo:
        print(f"\n=== {name.upper()} ===")
        t0 = time.time()
        EXPERIMENTS[name](args, S, out)
        print(f"  ({time.time() - t0:.1f} s)")

    failed = [c for c in CHECKS if c["passed"] is False]
    passed = [c for c in CHECKS if c["passed"] is True]
    print(f"\n=== SUMMARY: {len(passed)} passed, {len(failed)} failed, "
          f"{len(CHECKS) - len(passed) - len(failed)} info  ({time.time() - t_all:.1f} s) ===")
    for c in failed:
        print(f"  FAIL  {c['name']}: {c['detail']}")
    save_json(CHECKS, out / f"checks_{'_'.join(todo)}.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
