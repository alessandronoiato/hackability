"""
RLHF Overoptimisation Simulation
=================================
Reproduces the Gao et al. "goldilocks curve" and shows that the unhackable
proxy (Algorithm 3 from the thesis) eliminates reward model overoptimisation.

Setup:  K-arm bandit, true reward theta*(i) = i/K.
        A "trap arm" at index TRAP_ARM gets systematically inflated observations
        (simulating a biased reward model), so the naive proxy incorrectly ranks
        it above the true best arm.  The unhackable proxy, by grouping arms
        whose confidence intervals overlap, prevents the optimiser from
        exploiting this estimation artefact.

Figure 1 — goldilocks curve  (KL budget  vs  reward, fixed n_data = K)
Figure 2 — hackability rate  (n_data  vs  P[best proxy arm ≠ best true arm])
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── parameters ────────────────────────────────────────────────────────────────
K           = 50          # number of arms / response types
SIGMA_OBS   = 0.3         # observation noise std
BETA_CI     = 2.0         # CI half-width multiplier  β(δ_ci)
N_SEEDS     = 20          # Monte-Carlo seeds
KL_STEPS    = 40          # points on the KL-budget axis
KL_MAX      = 6.0         # maximum KL budget

# trap arm: a BAD arm whose observations are systematically inflated,
# simulating a biased reward model.
TRAP_ARM    = K // 8          # arm index 6, sits in the "bad" half
TRAP_BIAS   = 1.5             # systematic upward bias for trap-arm observations

# N_DATA for Figure 1: one observation per arm (Maximum Variance Reduction
# with flat GP prior is equivalent to round-robin for independent arms)
N_DATA_FIG1 = K

# N_DATA sweep for Figure 2: focus on the practically relevant sparse-data
# regime (1–5 observations per arm), which mirrors real RLHF where reward
# models are trained on O(10^3) comparisons over O(10^3) response types.
N_DATA_FIG2 = np.arange(1, 6) * K    # K, 2K, 3K, 4K, 5K
KL_FIG2 = 2.0   # moderate KL budget representative of practical RLHF

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── true rewards (deterministic, bimodal) ─────────────────────────────────────
# Bad  arms  0 .. K//2-1  : true reward ∈ [0.00, 0.40]
# Good arms  K//2 .. K-1  : true reward ∈ [0.60, 1.00]
# This clear gap means overoptimisation is visible: the naive proxy,
# tricked by the inflated trap arm (a bad arm), drives the policy into
# the low-reward region at large KL budget.
_bad  = np.linspace(0.00, 0.40, K // 2)
_good = np.linspace(0.60, 1.00, K - K // 2)
THETA_STAR = np.concatenate([_bad, _good])

# ── data collection ───────────────────────────────────────────────────────────

def observe(arm: int, rng: np.random.Generator, with_bias: bool = True) -> float:
    """Return one noisy observation of arm `arm`."""
    bias = TRAP_BIAS if (with_bias and arm == TRAP_ARM) else 0.0
    return THETA_STAR[arm] + bias + rng.normal(0, SIGMA_OBS)


def collect_data(n_data: int, rng: np.random.Generator,
                 with_bias: bool = True):
    """
    Round-robin sampling: arm i gets floor(n_data/K) observations, and the
    first (n_data % K) arms get one extra.  This is the Maximum-Variance-
    Reduction schedule for a flat GP prior over independent arms.
    Returns obs_counts[K], obs_sums[K].
    """
    base, extra = divmod(n_data, K)
    obs_counts = np.full(K, base, dtype=int)
    obs_counts[:extra] += 1

    obs_sums = np.zeros(K)
    for arm in range(K):
        for _ in range(obs_counts[arm]):
            obs_sums[arm] += observe(arm, rng, with_bias=with_bias)

    return obs_counts, obs_sums


# ── GP posterior (conjugate Gaussian) ────────────────────────────────────────

PRIOR_VAR = 1.0   # flat / uninformative prior


def gp_posterior(obs_counts: np.ndarray, obs_sums: np.ndarray):
    """
    Conjugate Gaussian posterior.
    Prior: N(0, PRIOR_VAR).  Likelihood: each obs ~ N(theta_i, SIGMA_OBS^2).
    With n observations summing to s:
        post_var = 1 / (1/prior_var + n/sigma_obs^2)
        post_mean = post_var * s / sigma_obs^2
    """
    mu    = np.zeros(K)
    sigma = np.zeros(K)
    for i in range(K):
        n = int(obs_counts[i])
        if n == 0:
            mu[i], sigma[i] = 0.0, np.sqrt(PRIOR_VAR)
        else:
            post_var = 1.0 / (1.0 / PRIOR_VAR + n / SIGMA_OBS**2)
            mu[i]    = post_var * obs_sums[i] / SIGMA_OBS**2
            sigma[i] = np.sqrt(post_var)
    return mu, sigma


# ── proxy reward functions ────────────────────────────────────────────────────

def naive_proxy(mu: np.ndarray) -> np.ndarray:
    """Plain GP posterior mean (standard reward model)."""
    return mu.copy()


def unhackable_proxy(mu: np.ndarray, sigma: np.ndarray,
                     beta: float = BETA_CI) -> np.ndarray:
    """
    Algorithm 3 (thesis):
      CI_n(i)  = [mu_i − β·σ_i,  mu_i + β·σ_i]
      M_n(i)   = { j : CI_n(i) ∩ CI_n(j) ≠ ∅ }
      R̃_n(i)  = max_{j ∈ M_n(i)} mu_j
    """
    lo = mu - beta * sigma
    hi = mu + beta * sigma
    proxy = np.empty(K)
    for i in range(K):
        overlaps  = (lo <= hi[i]) & (lo[i] <= hi)   # boolean mask
        proxy[i]  = mu[overlaps].max()
    return proxy


# ── KL-constrained optimisation ───────────────────────────────────────────────

PI0 = np.ones(K) / K   # uniform reference policy  π₀


def kl_div(pi: np.ndarray) -> float:
    """KL(π ‖ π₀)."""
    mask = pi > 0.0
    return float(np.sum(pi[mask] * np.log(pi[mask] / PI0[mask])))


def optimal_policy(proxy_rewards: np.ndarray, kl_budget: float) -> np.ndarray:
    """
    Solves  argmax_π  E_π[r̃]  s.t.  KL(π ‖ π₀) ≤ kl_budget.
    Solution: π(i) ∝ π₀(i) · exp(r̃(i) / τ), where τ is chosen by binary
    search to saturate the KL constraint.
    """
    if kl_budget <= 1e-9:
        return PI0.copy()

    def policy_at(tau: float) -> np.ndarray:
        logp  = np.log(PI0) + proxy_rewards / tau
        logp -= logp.max()
        pi    = np.exp(logp)
        pi   /= pi.sum()
        return pi

    # KL is maximised at τ→0 (deterministic on argmax).  Check feasibility.
    tau_lo, tau_hi = 1e-6, 1e4

    def f(tau: float) -> float:
        return kl_div(policy_at(tau)) - kl_budget

    if f(tau_lo) < 0:          # budget exceeds achievable KL: go deterministic
        return policy_at(tau_lo)
    if f(tau_hi) > 0:          # budget < 0 case covered above
        return PI0.copy()

    tau_star = brentq(f, tau_lo, tau_hi, xtol=1e-9)
    return policy_at(tau_star)


# ── single-seed simulation (Figure 1) ────────────────────────────────────────

def run_seed_fig1(seed: int):
    rng = np.random.default_rng(seed)
    obs_counts, obs_sums = collect_data(N_DATA_FIG1, rng, with_bias=True)
    mu, sigma            = gp_posterior(obs_counts, obs_sums)

    r_naive  = naive_proxy(mu)
    r_unhack = unhackable_proxy(mu, sigma)

    kl_range = np.linspace(0.0, KL_MAX, KL_STEPS)
    out = {k: np.zeros(KL_STEPS) for k in
           ["naive_true", "naive_proxy", "unhack_true", "unhack_proxy"]}

    for ki, delta in enumerate(kl_range):
        pi_n = optimal_policy(r_naive,  delta)
        pi_u = optimal_policy(r_unhack, delta)
        out["naive_true"][ki]   = pi_n @ THETA_STAR
        out["naive_proxy"][ki]  = pi_n @ r_naive
        out["unhack_true"][ki]  = pi_u @ THETA_STAR
        out["unhack_proxy"][ki] = pi_u @ r_unhack

    return out


# ── Figure 2: true reward vs n_data at fixed KL budget ───────────────────────

def run_seed_fig2(seed: int, n_data: int):
    """
    Returns (true_reward_naive, true_reward_unhack) at KL budget KL_FIG2.
    Uses the same trap-arm bias as Figure 1 so that overoptimisation is
    visible: as n_data grows, the naive proxy becomes more confident in the
    biased trap estimate and concentrates more aggressively on it.
    """
    rng = np.random.default_rng(seed + 7777)
    obs_counts, obs_sums = collect_data(n_data, rng, with_bias=True)
    mu, sigma = gp_posterior(obs_counts, obs_sums)

    pi_n = optimal_policy(naive_proxy(mu),        KL_FIG2)
    pi_u = optimal_policy(unhackable_proxy(mu, sigma), KL_FIG2)
    return float(pi_n @ THETA_STAR), float(pi_u @ THETA_STAR)


def hackability_argmax_set(obs_counts, obs_sums, use_naive: bool) -> bool:
    """
    True if the true-best arm (arm K-1) is NOT reachable: its true reward
    is strictly beaten by the proxy's argmax arm(s).  Uses argmax *set*
    so ties (all arms sharing the same unhackable proxy value) count as
    the best arm being reachable.
    """
    mu, sigma = gp_posterior(obs_counts, obs_sums)
    proxy = naive_proxy(mu) if use_naive else unhackable_proxy(mu, sigma)
    best_true = K - 1
    tol = 1e-9
    argmax_arms = set(np.where(proxy >= proxy.max() - tol)[0])
    return best_true not in argmax_arms


# ── plotting helpers ──────────────────────────────────────────────────────────

COLORS = {
    "naive_true":   "#d62728",   # red
    "naive_proxy":  "#ff7f0e",   # orange
    "unhack_true":  "#1f77b4",   # blue
    "unhack_proxy": "#2ca02c",   # green
}


def shade_band(ax, x, m, s, color, label, ls="-", lw=1.8):
    ax.plot(x, m, ls=ls, color=color, lw=lw, label=label)
    ax.fill_between(x, m - s, m + s, alpha=0.18, color=color)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"RLHF overoptimisation simulation  (K={K}, "
          f"σ_obs={SIGMA_OBS}, β={BETA_CI}, seeds={N_SEEDS})")

    # ── Figure 1: goldilocks curve ────────────────────────────────────────
    print("Running Figure 1 (goldilocks curve)…")
    all_res = [run_seed_fig1(s) for s in range(N_SEEDS)]

    def agg(key):
        arr = np.stack([r[key] for r in all_res])
        return arr.mean(0), arr.std(0)

    kl_range = np.linspace(0.0, KL_MAX, KL_STEPS)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # Plot true rewards only.  Both proxy rewards converge to the same maximum
    # at large δ (the inflated posterior mean of the bias arm, shared by the
    # entire equivalence class), which would falsely suggest the two proxies
    # are equivalent.  The true rewards diverge, which is the relevant signal.
    shade_band(ax, kl_range, *agg("naive_true"),  COLORS["naive_true"],
               r"True reward — naive proxy", ls="-")
    shade_band(ax, kl_range, *agg("unhack_true"), COLORS["unhack_true"],
               r"True reward — unhackable proxy", ls="-")

    baseline = float(PI0 @ THETA_STAR)
    ax.axhline(baseline, color="grey", ls="--", lw=1.1,
               label=fr"Reference baseline ($\mathbb{{E}}_{{\pi_0}}[\theta^*]={baseline:.2f}$)")
    ax.axhline(THETA_STAR[TRAP_ARM], color="#8c564b", ls=":", lw=1.1,
               label=fr"Bias-arm true reward ($\theta^*_{{b}}={THETA_STAR[TRAP_ARM]:.2f}$)")

    ax.set_xlabel(r"KL budget $\delta$", fontsize=12)
    ax.set_ylabel(r"Expected true reward $\mathbb{E}_{\pi^*_\delta}[\theta^*]$", fontsize=12)
    ax.set_title(r"Reward model overoptimisation: true reward under naive vs.\ unhackable proxy",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "rlhf_overopt_curve.pdf")
    fig.savefig(p1, bbox_inches="tight")
    print(f"  Saved {p1}")
    plt.close(fig)

    # ── Figure 2: true reward vs n_data at fixed KL budget ───────────────
    print(f"Running Figure 2 (true reward vs n_data, KL={KL_FIG2})…")
    naive_tr_m, naive_tr_s  = [], []
    unhack_tr_m, unhack_tr_s = [], []

    for nd in N_DATA_FIG2:
        naive_vals, unhack_vals = [], []
        for s in range(N_SEEDS):
            tn, tu = run_seed_fig2(s, nd)
            naive_vals.append(tn)
            unhack_vals.append(tu)
        nm, ns = np.mean(naive_vals), np.std(naive_vals)
        um, us = np.mean(unhack_vals), np.std(unhack_vals)
        naive_tr_m.append(nm);  naive_tr_s.append(ns)
        unhack_tr_m.append(um); unhack_tr_s.append(us)
        spa = nd // K
        print(f"  n_data={nd:4d}  (~{spa:2d} obs/arm)  "
              f"naive_true={nm:.3f}±{ns:.3f}  unhack_true={um:.3f}±{us:.3f}")

    naive_tr_m  = np.array(naive_tr_m)
    naive_tr_s  = np.array(naive_tr_s)
    unhack_tr_m = np.array(unhack_tr_m)
    unhack_tr_s = np.array(unhack_tr_s)
    spa_axis = N_DATA_FIG2 // K

    baseline_true = float(PI0 @ THETA_STAR)   # uniform policy true reward

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    shade_band(ax2, spa_axis, naive_tr_m,  naive_tr_s,
               COLORS["naive_true"],  "True reward — naive proxy",  ls="-")
    shade_band(ax2, spa_axis, unhack_tr_m, unhack_tr_s,
               COLORS["unhack_true"], "True reward — unhackable proxy", ls="-")
    ax2.axhline(baseline_true, color="grey", ls="--", lw=1.1,
                label=fr"Reference baseline ({baseline_true:.2f})")

    # shade the practical RLHF regime (n ≤ 3 obs/arm)
    ax2.axvspan(0.5, 3.5, alpha=0.07, color="steelblue",
                label="Practical regime ($n \\leq 3$)")

    ax2.set_xlabel(r"Observations per arm $n$", fontsize=12)
    ax2.set_ylabel(fr"Expected true reward at $\delta={KL_FIG2}$", fontsize=11)
    ax2.set_title(r"Protection in the sparse-data regime ($\delta=" +
                  str(KL_FIG2) + r"$)", fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.28)
    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, "rlhf_overopt_hackability.pdf")
    fig2.savefig(p2, bbox_inches="tight")
    print(f"  Saved {p2}")
    plt.close(fig2)

    print("Done.")


if __name__ == "__main__":
    main()
