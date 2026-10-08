"""
Blood Glucose Management MDP Simulation
========================================
Demonstrates:
 1. LP feasibility / diversity condition (rank(S) vs number of policies)
 2. Three-way proxy comparison: naive, LP (Algorithm 5), Q-proxy (Algorithm algoQ)
 3. Connection to Pan et al. 2022: more capable agents hack naive proxy more

Run:
    python3 experiments/glucose_mdp.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MDP DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

N_STATES  = 20          # glucose bins 0..19
N_ACTIONS = 5           # insulin dose 0..4
H         = 10          # episode horizon
SIGMA_DYN = 0.5         # transition noise std
SIGMA_OBS = 0.3         # reward observation noise std

def true_reward(g: int) -> float:
    if   g <= 4:  return -2.0
    elif g <= 7:  return -1.0
    elif g <= 12: return  1.0
    elif g <= 16: return -0.5
    else:         return -2.0

TRUE_R = np.array([true_reward(g) for g in range(N_STATES)])

def action_effect(a: int) -> float:
    return 2.0 * a

def step_env(g: int, a: int, rng: np.random.Generator) -> int:
    meal  = rng.choice([0, 3])
    g_next = g - action_effect(a) + meal + rng.normal(0, SIGMA_DYN)
    return int(np.clip(round(g_next), 0, N_STATES - 1))

def rollout(policy, n_steps: int, rng: np.random.Generator, g0: int = 10):
    """Returns list of (state, action, observed_reward)."""
    trajectory = []
    g = g0
    for _ in range(n_steps):
        a = policy(g)
        r_obs = TRUE_R[g] + rng.normal(0, SIGMA_OBS)
        trajectory.append((g, a, r_obs))
        g = step_env(g, a, rng)
    return trajectory

# ─────────────────────────────────────────────────────────────────────────────
# 2.  POLICY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_targeting_policy(g_target: int, noise_std: float = 0.0,
                          rng: np.random.Generator | None = None):
    """Deterministic (or slightly noisy) policy targeting g_target."""
    def policy(g: int) -> int:
        a = int(np.clip(round((g - g_target) / 2.0), 0, N_ACTIONS - 1))
        if noise_std > 0 and rng is not None:
            a = int(np.clip(a + round(rng.normal(0, noise_std)), 0, N_ACTIONS - 1))
        return a
    return policy

# Five diverse policies — spread across the glucose range
POLICY_TARGETS_DIV = [4, 7, 10, 13, 17]
POLICIES_DIVERSE   = [make_targeting_policy(t) for t in POLICY_TARGETS_DIV]
POLICY_NAMES_DIV   = [f"target={t}" for t in POLICY_TARGETS_DIV]

# Two nearly-identical policies for the non-diverse case
# Both target g=10 with tiny perturbation; their state visitation vectors
# will be nearly collinear (rank ≈ 1) even though value estimates differ.
_rng_pol = np.random.default_rng(77)
POLICY_A  = make_targeting_policy(10, noise_std=0.05, rng=_rng_pol)
POLICY_B  = make_targeting_policy(10, noise_std=0.05, rng=np.random.default_rng(88))
POLICIES_NARROW = [POLICY_A, POLICY_B]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA COLLECTION & REWARD ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

N_TRAJ = 200  # episodes per policy

def collect_data(policies, n_traj: int, rng: np.random.Generator):
    """Returns (all_trajs_per_policy, r_obs_per_state)."""
    all_trajs: list[list] = []
    r_obs: dict[int, list[float]] = defaultdict(list)
    for pol in policies:
        trajs = []
        for _ in range(n_traj):
            g0   = int(rng.integers(6, 15))
            traj = rollout(pol, H, rng, g0)
            trajs.append(traj)
            for (g, _, r) in traj:
                r_obs[g].append(r)
        all_trajs.append(trajs)
    return all_trajs, r_obs

def estimate_reward(r_obs: dict[int, list[float]],
                    bias_states: range | None = None,
                    bias: float = 0.0):
    """
    Per-state MLE reward estimate and uncertainty.
    bias_states / bias: add systematic bias to specific states (naive proxy hack).
    """
    r_hat = np.zeros(N_STATES)
    r_std = np.ones(N_STATES) * 0.5        # default when unvisited
    for g in range(N_STATES):
        obs = r_obs.get(g, [])
        n   = len(obs)
        if n >= 2:
            r_hat[g] = float(np.mean(obs))
            r_std[g] = float(np.std(obs, ddof=1) / np.sqrt(n))
        elif n == 1:
            r_hat[g] = float(obs[0])
    if bias_states is not None:
        for g in bias_states:
            r_hat[g] += bias
    return r_hat, r_std

def state_visitation(trajs_for_policy) -> np.ndarray:
    counts = np.zeros(N_STATES)
    total  = 0
    for traj in trajs_for_policy:
        for (g, _, _) in traj:
            counts[g] += 1
            total += 1
    return counts / max(total, 1)

def value_estimate(trajs_for_policy, r_hat: np.ndarray):
    """V̂^π = mean over episodes of sum_{h} r_hat(g_h)."""
    vals = [sum(r_hat[g] for (g, _, _) in traj) for traj in trajs_for_policy]
    mu   = float(np.mean(vals))
    se   = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mu, se

def compute_ci_half(trajs_for_policy, r_std: np.ndarray, beta: float = 0.5):
    """CI half-width = β * mean_episode(Σ_h σ_n(g_h))."""
    widths = [beta * sum(r_std[g] for (g, _, _) in traj)
              for traj in trajs_for_policy]
    return float(np.mean(widths))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  EQUIVALENCE CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def build_equivalence_classes(v_hats, ci_halfs):
    """
    Merge policies whose confidence intervals overlap into equivalence classes.
    Uses connected-components on the overlap graph.
    Returns list of lists of policy indices.
    """
    n = len(v_hats)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            lo = max(v_hats[i] - ci_halfs[i], v_hats[j] - ci_halfs[j])
            hi = min(v_hats[i] + ci_halfs[i], v_hats[j] + ci_halfs[j])
            if lo <= hi:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LP / CONSTRAINED LS  (Algorithm 5)
# ─────────────────────────────────────────────────────────────────────────────

def lp_proxy(S_full: np.ndarray, eq_classes: list[list[int]],
             v_hats: np.ndarray, ci_halfs: np.ndarray,
             r_hat: np.ndarray):
    """
    Constrained least-squares for θ̃:

        min  ||θ̃ - r̂||²
        s.t. S_cls[i] · θ̃ ∈ [m_i, M_i]  for each class i

    When rank(S_cls) = l the system is feasible (exact equality solution exists
    via the closed-form KKT; we then project to the box [m_i, M_i]).
    Returns (theta_tilde, feasible: bool).
    """
    l = len(eq_classes)

    # Per-class averaged state visitation and CI bounds
    S_cls = np.zeros((l, N_STATES))
    m_arr = np.zeros(l)
    M_arr = np.zeros(l)

    for i, cls in enumerate(eq_classes):
        S_cls[i] = np.mean(S_full[cls], axis=0)
        # CI intersection for the class
        lb = max(v_hats[j] - ci_halfs[j] for j in cls)
        ub = min(v_hats[j] + ci_halfs[j] for j in cls)
        if lb > ub + 1e-8:
            return r_hat.copy(), False     # empty intersection
        m_arr[i] = lb
        M_arr[i] = ub

    # ── rank check ──────────────────────────────────────────────────────────
    rk = int(np.linalg.matrix_rank(S_cls, tol=1e-5))
    if rk < l:
        return r_hat.copy(), False

    # ── closed-form equality-constrained LS ─────────────────────────────────
    # min ||θ - r̂||² s.t. S_cls θ = c (c = midpoint of each class interval)
    # KKT solution: θ = r̂ - S_cls^T (S_cls S_cls^T)^{-1} (S_cls r̂ - c)
    c      = 0.5 * (m_arr + M_arr)
    M_mat  = S_cls @ S_cls.T          # (l, l), full rank since rk = l
    try:
        lam    = np.linalg.solve(M_mat, S_cls @ r_hat - c)
    except np.linalg.LinAlgError:
        return r_hat.copy(), False
    theta  = r_hat - S_cls.T @ lam

    # Verify constraints are approximately satisfied
    residuals = S_cls @ theta - c
    if np.max(np.abs(residuals)) > 1.0:   # sanity check
        return r_hat.copy(), False

    return theta, True

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Q-PROXY FALLBACK  (Algorithm algoQ)
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_model(all_trajs) -> np.ndarray:
    """Empirical transition matrix T[s, a, s'] from rollout data."""
    counts = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for trajs in all_trajs:
        for traj in trajs:
            for t in range(len(traj) - 1):
                g, a, _ = traj[t]
                g_next  = traj[t + 1][0]
                counts[g, a, g_next] += 1
    T = counts / (counts.sum(axis=2, keepdims=True) + 1e-8)
    return T

def backward_induction(r_hat: np.ndarray, T: np.ndarray, gamma: float = 0.95):
    """Finite-horizon Q via dynamic programming (H sweeps)."""
    V = r_hat.copy()
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(H):
        for g in range(N_STATES):
            for a in range(N_ACTIONS):
                Q[g, a] = r_hat[g] + gamma * (T[g, a] @ V)
        V = Q.max(axis=1)
    return Q

def q_proxy(Q_hat: np.ndarray, r_std: np.ndarray, beta: float = 0.5):
    """
    For each (s,a) set Q̃(s,a) = max_{a' in M(s,a)} Q̂(s,a'),
    where M(s,a) = {a' : Q-CIs of a and a' overlap at state s}.
    CI half-width at state s = β * σ_n(s).
    """
    Q_tilde = Q_hat.copy()
    for g in range(N_STATES):
        ci = beta * r_std[g]
        M_map = []
        for a in range(N_ACTIONS):
            group = [a2 for a2 in range(N_ACTIONS)
                     if max(Q_hat[g, a] - ci, Q_hat[g, a2] - ci) <=
                        min(Q_hat[g, a] + ci, Q_hat[g, a2] + ci)]
            M_map.append(group)
        for a in range(N_ACTIONS):
            Q_tilde[g, a] = max(Q_hat[g, ap] for ap in M_map[a])
    return Q_tilde

# ─────────────────────────────────────────────────────────────────────────────
# 7.  DOWNSTREAM Q-LEARNING AGENT
# ─────────────────────────────────────────────────────────────────────────────

def q_learning(proxy_r_fn, n_steps: int,
               alpha: float = 0.15, gamma: float = 0.95,
               eps_start: float = 1.0, eps_end: float = 0.05,
               seed: int = 0) -> np.ndarray:
    """Tabular Q-learning with ε-greedy exploration, decaying ε."""
    rng = np.random.default_rng(seed)
    Q   = np.zeros((N_STATES, N_ACTIONS))
    g   = int(rng.integers(6, 15))
    eps = eps_start
    decay = (eps_start - eps_end) / max(n_steps, 1)
    for _ in range(n_steps):
        if rng.random() < eps:
            a = int(rng.integers(0, N_ACTIONS))
        else:
            a = int(np.argmax(Q[g]))
        r_p    = proxy_r_fn(g, a)
        g_next = step_env(g, a, rng)
        Q[g, a] += alpha * (r_p + gamma * Q[g_next].max() - Q[g, a])
        g    = g_next
        eps  = max(eps_end, eps - decay)
    return Q

def evaluate_policy(Q: np.ndarray, n_episodes: int = 500, seed: int = 99):
    """Evaluate greedy policy under the true reward function."""
    rng = np.random.default_rng(seed)
    episode_rewards = []
    fracs_target    = []
    for _ in range(n_episodes):
        g        = int(rng.integers(6, 15))
        ep_r     = 0.0
        in_range = 0
        for _ in range(H):
            a  = int(np.argmax(Q[g]))
            ep_r += TRUE_R[g]
            if 8 <= g <= 12:
                in_range += 1
            g  = step_env(g, a, rng)
        episode_rewards.append(ep_r)
        fracs_target.append(in_range / H)
    return float(np.mean(episode_rewards)), float(np.mean(fracs_target))

# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPERIMENT A — DIVERSITY / RANK PLOT
# ─────────────────────────────────────────────────────────────────────────────

def experiment_diversity():
    print("\n[Experiment A] Rank vs. number of training policies …")
    rng = np.random.default_rng(0)
    all_trajs_full, r_obs_full = collect_data(POLICIES_DIVERSE, N_TRAJ, rng)
    r_hat, r_std = estimate_reward(r_obs_full)

    ranks, n_classes_list, feasible_list = [], [], []

    for n_pol in range(1, len(POLICIES_DIVERSE) + 1):
        trajs_sub = all_trajs_full[:n_pol]
        S         = np.array([state_visitation(t) for t in trajs_sub])
        v_hats    = np.array([value_estimate(t, r_hat)[0]  for t in trajs_sub])
        ci_halfs  = np.array([compute_ci_half(t, r_std)    for t in trajs_sub])
        classes   = build_equivalence_classes(v_hats, ci_halfs)
        l         = len(classes)
        S_cls     = np.array([S[c].mean(axis=0) for c in classes])
        rk        = int(np.linalg.matrix_rank(S_cls, tol=1e-5))
        _, feas   = lp_proxy(S, classes, v_hats, ci_halfs, r_hat)
        ranks.append(rk)
        n_classes_list.append(l)
        feasible_list.append(feas)
        print(f"  n_pol={n_pol}: rank={rk}, l={l}, LP feasible={feas}")

    # ── plot ──────────────────────────────────────────────────────────────────
    xs   = list(range(1, len(POLICIES_DIVERSE) + 1))
    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(xs, ranks,           "o-", color="#2166ac", linewidth=2.5,
            markersize=8, label=r"rank$(S)$  [actual]")
    ax.plot(xs, n_classes_list,  "s--", color="#d6604d", linewidth=2.0,
            markersize=8, label=r"$\ell$  (equivalence classes)")

    # shade feasible / infeasible regions
    for i, (rk, l, feas) in enumerate(zip(ranks, n_classes_list, feasible_list)):
        col = "#1a9641" if feas else "#d73027"
        ax.scatter([xs[i]], [rk], s=120, color=col, zorder=5,
                   marker="*" if feas else "x",
                   linewidths=2)

    from matplotlib.lines import Line2D
    legend_extra = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a9641",
               markersize=10, label="LP feasible"),
        Line2D([0], [0], marker="x", color="#d73027",
               markersize=9, markeredgewidth=2, label="LP infeasible"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labels + ["LP feasible", "LP infeasible"],
              fontsize=9, loc="upper left")

    ax.set_xlabel("Number of training policies", fontsize=12)
    ax.set_ylabel("Rank / number of equivalence classes", fontsize=12)
    ax.set_title("LP feasibility and the diversity condition", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$n={x}$\n(target={t})" for x, t in
                         zip(xs, POLICY_TARGETS_DIV)], fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "glucose_diversity.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved {path}")
    return ranks, n_classes_list, feasible_list

# ─────────────────────────────────────────────────────────────────────────────
# 9.  EXPERIMENT B — STATE VISITATION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _add_glucose_zones(ax):
    zones = [
        (0, 4,   "#d73027", 0.10),
        (5, 7,   "#fdae61", 0.10),
        (8, 12,  "#1a9641", 0.10),
        (13, 16, "#fdae61", 0.10),
        (17, 19, "#d73027", 0.10),
    ]
    for lo, hi, col, alpha in zones:
        ax.axvspan(lo - 0.5, hi + 0.5, color=col, alpha=alpha, zorder=0)

def experiment_state_visitation():
    print("\n[Experiment B] State visitation distributions …")
    rng = np.random.default_rng(1)
    all_trajs_div, _ = collect_data(POLICIES_DIVERSE, N_TRAJ, rng)
    all_trajs_nar, _ = collect_data(POLICIES_NARROW,  N_TRAJ, rng)

    sv_div = np.array([state_visitation(t) for t in all_trajs_div])
    sv_nar = np.array([state_visitation(t) for t in all_trajs_nar])

    # show cosine similarity of the two narrow-policy vectors
    cos_sim = (sv_nar[0] @ sv_nar[1] /
               (np.linalg.norm(sv_nar[0]) * np.linalg.norm(sv_nar[1]) + 1e-12))
    rank_nar = int(np.linalg.matrix_rank(sv_nar, tol=1e-5))
    print(f"  Narrow policies — cosine similarity: {cos_sim:.4f}, rank: {rank_nar}")

    colors_div = plt.cm.Blues(np.linspace(0.35, 0.90, len(POLICIES_DIVERSE)))
    colors_nar = ["#e08214", "#b35806"]

    xs    = np.arange(N_STATES)
    width = 0.8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # diverse
    bottom = np.zeros(N_STATES)
    for i, (sv, col) in enumerate(zip(sv_div, colors_div)):
        ax1.bar(xs, sv, width, bottom=bottom, color=col,
                label=f"$g_{{\\mathrm{{tgt}}}}={POLICY_TARGETS_DIV[i]}$")
        bottom += sv
    ax1.set_title("Diverse policies (5 targets)", fontsize=11)
    ax1.set_xlabel("Glucose state $g$", fontsize=10)
    ax1.set_ylabel("Stacked state visitation", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_xticks(xs[::2])
    _add_glucose_zones(ax1)

    # narrow
    bottom = np.zeros(N_STATES)
    for i, (sv, col) in enumerate(zip(sv_nar, colors_nar)):
        ax2.bar(xs, sv, width, bottom=bottom, color=col,
                label=f"Policy {chr(65+i)} (target$\\approx$10)")
        bottom += sv
    ax2.set_title(f"Non-diverse policies (2 targets $\\approx$ 10)\n"
                  f"cosine sim = {cos_sim:.3f}, rank = {rank_nar}",
                  fontsize=11)
    ax2.set_xlabel("Glucose state $g$", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xticks(xs[::2])
    _add_glucose_zones(ax2)

    # zone legend
    from matplotlib.patches import Patch
    zone_handles = [
        Patch(facecolor="#d73027", alpha=0.3, label="Dangerous"),
        Patch(facecolor="#fdae61", alpha=0.3, label="Sub-optimal"),
        Patch(facecolor="#1a9641", alpha=0.3, label="Target range"),
    ]
    ax2.legend(handles=ax2.get_legend_handles_labels()[0] + zone_handles,
               labels=ax2.get_legend_handles_labels()[1] +
                      ["Dangerous", "Sub-optimal", "Target range"],
               fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "glucose_state_visitation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
    return cos_sim, rank_nar

# ─────────────────────────────────────────────────────────────────────────────
# 10.  EXPERIMENT C — THREE-WAY REWARD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def experiment_reward_comparison():
    print("\n[Experiment C] Three-way proxy comparison …")
    rng_data = np.random.default_rng(2)

    # Collect data using the diverse policy set
    all_trajs, r_obs = collect_data(POLICIES_DIVERSE, N_TRAJ, rng_data)

    # ── NAIVE proxy: GP posterior mean + systematic bias on "high" states ──
    # Bias = +3.0 on states 13-16 makes naive r̂(13-16) ≈ 2.5,
    # clearly dominating the true target range (r̂ ≈ 1.0), so a capable
    # agent learns to target g≈14-15 (hacking) instead of g≈10 (optimal).
    BIAS_STATES = range(13, 17)
    BIAS_VALUE  = 3.0
    r_hat_naive, _ = estimate_reward(r_obs, bias_states=BIAS_STATES,
                                      bias=BIAS_VALUE)

    # ── LP proxy (Algorithm 5) ────────────────────────────────────────────────
    r_hat_lp, r_std = estimate_reward(r_obs)
    S         = np.array([state_visitation(t) for t in all_trajs])
    v_hats    = np.array([value_estimate(t, r_hat_lp)[0] for t in all_trajs])
    ci_halfs  = np.array([compute_ci_half(t, r_std)      for t in all_trajs])
    classes   = build_equivalence_classes(v_hats, ci_halfs)
    theta_lp, lp_ok = lp_proxy(S, classes, v_hats, ci_halfs, r_hat_lp)
    print(f"  LP feasible (diverse, 5 policies): {lp_ok}")

    # If LP succeeded, show failure with non-diverse set
    rng_nar   = np.random.default_rng(3)
    trajs_nar, r_obs_nar = collect_data(POLICIES_NARROW, N_TRAJ, rng_nar)
    r_hat_nar, r_std_nar = estimate_reward(r_obs_nar)
    S_nar     = np.array([state_visitation(t) for t in trajs_nar])
    v_hat_nar = np.array([value_estimate(t, r_hat_nar)[0] for t in trajs_nar])
    ci_nar    = np.array([compute_ci_half(t, r_std_nar)   for t in trajs_nar])
    cls_nar   = build_equivalence_classes(v_hat_nar, ci_nar)
    _, lp_nar_ok = lp_proxy(S_nar, cls_nar, v_hat_nar, ci_nar, r_hat_nar)
    rk_nar    = int(np.linalg.matrix_rank(S_nar, tol=1e-5))
    print(f"  Non-diverse: rank(S)={rk_nar}, "
          f"l={len(cls_nar)}, LP feasible={lp_nar_ok}")

    # ── Q-proxy (Algorithm algoQ) ─────────────────────────────────────────────
    T      = build_transition_model(all_trajs)
    Q_hat  = backward_induction(r_hat_lp, T)
    Q_til  = q_proxy(Q_hat, r_std)

    # ── downstream training at increasing capability ───────────────────────
    step_checkpoints = [200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]

    proxy_fns = {
        "Naive proxy (biased GP mean)":
            lambda g, a, _r=r_hat_naive: float(_r[g]),
        "LP proxy (Algorithm 5)":
            lambda g, a, _t=theta_lp: float(_t[g]),
        "Q-proxy (Algorithm algoQ)":
            lambda g, a, _q=Q_til: float(_q[g, a]),
    }

    results_reward = {k: [] for k in proxy_fns}
    results_frac   = {k: [] for k in proxy_fns}

    N_SEEDS = 5   # average over seeds to smooth learning curves

    for label, pfn in proxy_fns.items():
        print(f"  Training '{label}' …")
        for n_steps in step_checkpoints:
            tr_runs, ft_runs = [], []
            for seed_offset in range(N_SEEDS):
                Q_agent = q_learning(pfn, n_steps, seed=n_steps + seed_offset * 31337)
                tr, ft  = evaluate_policy(Q_agent, seed=n_steps + seed_offset * 999)
                tr_runs.append(tr)
                ft_runs.append(ft)
            results_reward[label].append(float(np.mean(tr_runs)))
            results_frac[label].append(float(np.mean(ft_runs)))
            print(f"    n={n_steps:>7d}: true_r={results_reward[label][-1]:+.2f}, "
                  f"frac_tgt={results_frac[label][-1]:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    styles = {
        "Naive proxy (biased GP mean)":
            ("#d73027", "o-",  "Naive proxy (biased GP mean)"),
        "LP proxy (Algorithm 5)":
            ("#1a9641", "s-",  "LP proxy (Algorithm 5, diverse data)"),
        "Q-proxy (Algorithm algoQ)":
            ("#2166ac", "^-",  "Q-proxy (Algorithm algoQ)"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    optimal_r = TRUE_R[10] * H   # approx optimal episode reward

    for label, (col, marker, lname) in styles.items():
        ax1.plot(step_checkpoints, results_reward[label], marker,
                 color=col, label=lname, linewidth=2.2, markersize=7)
        ax2.plot(step_checkpoints, results_frac[label], marker,
                 color=col, label=lname, linewidth=2.2, markersize=7)

    ax1.axhline(optimal_r, color="gray", linestyle=":", linewidth=1.5,
                label=f"Near-optimal ({optimal_r:.0f})")
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xlabel("Training steps (agent capability)", fontsize=11)
    ax1.set_ylabel(f"Avg. true reward / episode (H={H})", fontsize=11)
    ax1.set_title("True reward vs. agent capability\n"
                  "(Pan et al. 2022 phenomenon)", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    # annotate hacking region
    ax1.annotate("Reward hacking:\nnaive proxy agent\npursues biased states",
                 xy=(50000, results_reward["Naive proxy (biased GP mean)"][-2]),
                 xytext=(8000, -7),
                 fontsize=7.5, color="#d73027",
                 arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2))

    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5,
                label="Perfect glucose control")
    ax2.set_xscale("log")
    ax2.set_xlabel("Training steps (agent capability)", fontsize=11)
    ax2.set_ylabel("Fraction of steps in target range [8,12]", fontsize=11)
    ax2.set_title("Glucose control quality", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "glucose_reward_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved {path}")
    return results_reward, results_frac, step_checkpoints, lp_ok

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Blood Glucose MDP — Reward Hacking Simulation")
    print("=" * 60)

    ranks, n_cls, feasible = experiment_diversity()
    cos_sim, rank_nar      = experiment_state_visitation()
    results_r, results_f, steps, lp_ok = experiment_reward_comparison()

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary table")
    print("=" * 60)
    print(f"{'Proxy':<35} {'Final true reward':>18} {'Frac in target':>15}")
    print("-" * 70)
    for label in results_r:
        fr = results_r[label][-1]
        ft = results_f[label][-1]
        print(f"  {label:<33} {fr:>18.2f} {ft:>15.3f}")
    print()
    print(f"  Diverse (5 pol.)  rank(S)={ranks[-1]}, l={n_cls[-1]}, LP feasible={lp_ok}")
    print(f"  Non-div (2 pol.)  rank(S)={rank_nar},  cosine sim={cos_sim:.3f}")
    print(f"\nAll plots saved to: {OUT_DIR}")
