"""Core model, optimizers and diagnostics for the tie-drift simulations.

Notation follows draft_tie_drift.tex: per prompt i the responses are
honest (index 0), hack (index 1) and n_fail failing responses; pi0 masses
(a, b, c); proxy = verifier pass; log-linear policy
pi_theta(y|i) ∝ pi0(y|i) exp(theta . psi(i, y)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HONEST, HACK, FAIL = 0, 1, 2


@dataclass
class Problem:
    logp0: np.ndarray      # (N, M) reference log-probabilities
    proxy: np.ndarray      # (N, M) proxy reward r~
    gold: np.ndarray       # (N, M) true reward r
    kind: np.ndarray       # (N, M) HONEST / HACK / FAIL
    passing: np.ndarray    # (N, M) bool, noise-free verifier outcome
    loophole: np.ndarray   # (N,) bool, prompts where the hack passes
    psi: np.ndarray        # (N, M, d) features
    info: dict = field(default_factory=dict)

    @property
    def n_prompts(self) -> int:
        return self.logp0.shape[0]

    @property
    def n_responses(self) -> int:
        return self.logp0.shape[1]

    @property
    def dim(self) -> int:
        return self.psi.shape[2]

    def with_features(self, psi: np.ndarray, **info) -> "Problem":
        return Problem(self.logp0, self.proxy, self.gold, self.kind, self.passing,
                       self.loophole, psi, {**self.info, **info})


# --------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------

def shared_feature_problem(
    n_prompts: int = 32,
    n_loophole: int = 16,
    masses: tuple[float, float, float] = (0.15, 0.05, 0.80),
    s_h: float = 1.0,
    s_u: float = 1.0,
    gamma: float = 0.0,
    n_forms: int = 1,
    variant: str = "isolated",
    n_fail: int = 1,
    hack_gold: float = 0.0,
    feature_mode: str = "orthogonal",
    d_random: int = 256,
    proxy_noise: float = 0.0,
    seed: int = 0,
) -> Problem:
    """Shared-feature model of Section 4.4.

    Loophole prompts are i < n_loophole; surface form of prompt i is i mod n_forms.
    variant="isolated": the hack feature appears only on loophole prompts.
    variant="exposed": the same surface form is also tried (and fails) elsewhere.
    proxy_noise > 0 adds U[0, proxy_noise) to every proxy value (continuous proxy,
    no exact ties); keep proxy_noise < 1 so that floor(proxy) is the verifier.
    """
    if variant not in ("isolated", "exposed"):
        raise ValueError(f"unknown variant {variant!r}")
    if feature_mode not in ("orthogonal", "random"):
        raise ValueError(f"unknown feature_mode {feature_mode!r}")
    a, b, c = masses
    if min(masses) <= 0 or abs(a + b + c - 1.0) > 1e-9:
        raise ValueError("masses must be positive and sum to 1")
    N, m, K = n_prompts, n_loophole, n_forms
    if not (0 <= m <= N and 1 <= K <= N and 0 <= gamma < 1 and n_fail >= 1):
        raise ValueError("invalid sizes")
    rng = np.random.default_rng(seed)

    M = 2 + n_fail
    p0 = np.concatenate([[a, b], np.full(n_fail, c / n_fail)])
    logp0 = np.tile(np.log(p0), (N, 1))
    loophole = np.arange(N) < m
    kind = np.tile(np.array([HONEST, HACK] + [FAIL] * n_fail), (N, 1))
    passing = (kind == HONEST) | ((kind == HACK) & loophole[:, None])
    proxy = passing.astype(float)
    if proxy_noise > 0:
        proxy = proxy + proxy_noise * rng.random((N, M))
    gold = np.where(kind == HONEST, 1.0, np.where(kind == HACK, hack_gold, 0.0))
    form = np.arange(N) % K

    if feature_mode == "orthogonal":
        d = 1 + N + K
        basis = np.eye(d)
    else:
        d = d_random
        basis = rng.standard_normal((1 + N + K, d))
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    v, e, u = basis[0], basis[1:1 + N], basis[1 + N:1 + N + K]
    phi = s_h * (np.sqrt(gamma) * v[None, :] + np.sqrt(1.0 - gamma) * e)

    psi = np.zeros((N, M, d))
    psi[:, HONEST] = phi
    has_form = loophole | (variant == "exposed")
    psi[has_form, HACK] = s_u * u[form[has_form]]

    m_g = np.bincount(form[loophole], minlength=K).astype(float)
    if variant == "exposed":
        m_prime = np.bincount(form[~loophole], minlength=K).astype(float)
    else:
        m_prime = np.zeros(K)
    R_form = m_g * s_u ** 2 * b / (s_h ** 2 * a)
    honest_term = s_h ** 2 * a * (c + gamma * ((m - 1) * c + (N - m) * (b + c)))
    drift0_form = (s_u ** 2 * b * (m_g * c - m_prime * a) - honest_term) / N

    info = dict(
        model="shared", N=N, m=m, K=K, a=a, b=b, c=c, s_h=s_h, s_u=s_u, gamma=gamma,
        variant=variant, feature_mode=feature_mode, form=form, m_g=m_g, m_prime=m_prime,
        R_form=R_form, drift0_form=drift0_form, hack_gold=hack_gold, proxy_noise=proxy_noise,
    )
    return Problem(logp0, proxy, gold, kind, passing, loophole, psi, info)


def tabular_problem(prob: Problem) -> Problem:
    """Same pi0 / proxy / gold, one free logit per (prompt, response)."""
    N, M = prob.logp0.shape
    psi = np.eye(N * M).reshape(N, M, N * M)
    return prob.with_features(psi, model="tabular")


# --------------------------------------------------------------------------
# Parameter bases (draft: "Preconditioned and adaptive optimizers"):
# same policy family, different trained coordinates
# --------------------------------------------------------------------------

BASES = ("axis", "rotated", "dense")


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal matrix (QR of a Gaussian matrix with the sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))[None, :]


def reparametrize(prob: Problem, basis: str, *, seed: int = 0, dense_mult: float = 4.0) -> Problem:
    """Express the same log-linear family in other trained coordinates w.

    Logits z = psi . theta with theta = B^T w, i.e. features psi_w(x, y) = B psi(x, y).
    basis="axis": B = I (the original coordinates, one per feature in the
    shared-feature model); "rotated": B = Q Haar-orthogonal (d x d);
    "dense": B = E in R^{p x d}, p = ceil(dense_mult * d), i.i.d. N(0, 1/p).
    Gradient steps on w are gradient steps on theta with kernel B^T B (I for
    rotated, E^T E for dense); coordinate-wise optimizers (Adam, sign descent)
    do not commute with B. info["B"] holds B (None for axis).
    """
    if basis == "axis":
        return prob.with_features(prob.psi, basis="axis", B=None)
    rng = np.random.default_rng(seed)
    d = prob.dim
    if basis == "rotated":
        B = random_orthogonal(d, rng)
    elif basis == "dense":
        p = int(np.ceil(dense_mult * d))
        B = rng.standard_normal((p, d)) / np.sqrt(p)
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return prob.with_features(prob.psi @ B.T, basis=basis, B=B)


def factored_init(d: int, hidden: int, init: float, seed: int) -> np.ndarray:
    """U0 of the two-layer factorization theta = U v (v0 = 0, so theta0 = 0 exactly)."""
    return init * np.random.default_rng(seed).standard_normal((d, hidden)) / np.sqrt(hidden)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

def log_softmax(z: np.ndarray) -> np.ndarray:
    s = z - z.max(axis=-1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))


def policy_logp(prob: Problem, theta: np.ndarray) -> np.ndarray:
    return log_softmax(prob.logp0 + prob.psi @ theta)


def tilt_logp(prob: Problem, tau: float) -> np.ndarray:
    return log_softmax(prob.logp0 + tau * prob.proxy)


def chi2_logp(prob: Problem, beta: float, iters: int = 200) -> np.ndarray:
    """argmax E r~ - beta * chi2(pi || pi0): pi/pi0 = max(0, 1 + (r~ - lambda)/(2 beta))."""
    p0 = np.exp(prob.logp0)
    r = prob.proxy
    lo, hi = r.min(axis=1), r.max(axis=1)
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        mass = (p0 * np.maximum(0.0, 1.0 + (r - lam[:, None]) / (2 * beta))).sum(axis=1)
        too_much = mass > 1.0
        lo = np.where(too_much, lam, lo)
        hi = np.where(too_much, hi, lam)
    lam = 0.5 * (lo + hi)
    p = p0 * np.maximum(0.0, 1.0 + (r - lam[:, None]) / (2 * beta))
    p /= p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(p)


def _log_pow_diff(hi: np.ndarray, lo: np.ndarray, n: float) -> np.ndarray:
    """log(hi^n - lo^n) for 0 <= lo < hi."""
    lo = np.clip(lo, 0.0, None)
    with np.errstate(divide="ignore"):
        log_hi, log_lo = np.log(hi), np.log(lo)
        return n * log_hi + np.log(-np.expm1(n * (log_lo - log_hi)))


def bon_logp(prob: Problem, n: float, tiebreak: str = "uniform") -> np.ndarray:
    """Exact best-of-n output distribution.

    tiebreak="uniform": uniform among tied maximizers (score-based).
    tiebreak="likelihood": prefer the tied response with the largest pi0 mass
    (content-dependent, not score-based).
    """
    p0 = np.exp(prob.logp0)
    r = prob.proxy
    le = r[:, None, :] <= r[:, :, None]
    lt = r[:, None, :] < r[:, :, None]
    F_le = (le * p0[:, None, :]).sum(-1)
    F_lt = (lt * p0[:, None, :]).sum(-1)
    if tiebreak == "uniform":
        logmass = _log_pow_diff(F_le, F_lt, n) + prob.logp0 - np.log(F_le - F_lt)
    elif tiebreak == "likelihood":
        M = r.shape[1]
        key = prob.logp0 - 1e-9 * np.arange(M)[None, :]
        tied = r[:, None, :] == r[:, :, None]
        higher = tied & (key[:, None, :] > key[:, :, None])
        H = (higher * p0[:, None, :]).sum(-1)
        upper = F_le - H
        logmass = _log_pow_diff(upper, upper - p0, n)
    else:
        raise ValueError(f"unknown tiebreak {tiebreak!r}")
    return log_softmax(logmass)


SCORE_BASED = {
    "tilt": tilt_logp,
    "chi2": chi2_logp,
    "bon_uniform": lambda prob, n: bon_logp(prob, n, "uniform"),
    "bon_likelihood": lambda prob, n: bon_logp(prob, n, "likelihood"),
}


# --------------------------------------------------------------------------
# Diagnostics (Theorem 2)
# --------------------------------------------------------------------------

def phi_bound(k: np.ndarray | float) -> np.ndarray:
    k = np.maximum(k, 0.0)
    return np.minimum(np.sqrt(k / 2.0), np.sqrt(-np.expm1(-k)))


def cells(prob: Problem, bin_width: float | None = None) -> np.ndarray:
    """(N, M, M) indicator that two responses of a prompt share a proxy cell."""
    key = prob.proxy if bin_width is None else np.floor(prob.proxy / bin_width)
    return (key[:, :, None] == key[:, None, :]).astype(float)


def diagnostics(prob: Problem, logp: np.ndarray, cell: np.ndarray | None = None) -> dict[str, float]:
    if cell is None:
        cell = cells(prob)
    p = np.exp(logp)
    p0 = np.exp(prob.logp0)
    pos = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(pos, logp - prob.logp0, 0.0)
        pT = np.einsum("njk,nk->nj", cell, p)
        p0T = np.einsum("njk,nk->nj", cell, p0)
        lrT = np.where(pos, np.log(pT) - np.log(p0T), 0.0)
        g = np.einsum("njk,nk->nj", cell, p0 * prob.gold) / p0T
        within = p * (lr - lrT)
        kl_cell = np.where(pos, np.einsum("njk,nk->nj", cell, within) / pT, 0.0)
    kl = (p * lr).sum(1)
    kl_score = (p * lrT).sum(1)
    kl_tie = kl - kl_score
    calib = ((p - p0) * g).sum(1)
    drift = (p * (prob.gold - g)).sum(1)
    dgold = ((p - p0) * prob.gold).sum(1)
    in_cell = cell > 0
    osc = (np.where(in_cell, prob.gold[:, None, :], -np.inf).max(-1)
           - np.where(in_cell, prob.gold[:, None, :], np.inf).min(-1))
    bound_fine = np.where(pos, p * osc * phi_bound(kl_cell), 0.0).sum(1)
    osc_max = float(osc.max())

    out = dict(
        kl=kl.mean(), kl_score=kl_score.mean(), kl_tie=kl_tie.mean(), kl_tie_max=kl_tie.max(),
        proxy=(p * prob.proxy).sum(1).mean(), gold=(p * prob.gold).sum(1).mean(),
        pass_rate=(p * prob.passing).sum(1).mean(),
        dgold=dgold.mean(), calib=calib.mean(), drift=drift.mean(),
        bound_fine=bound_fine.mean(),
        bound_jensen=osc_max * float(phi_bound(kl_tie.mean())),
        bound_pinsker=osc_max * float(np.sqrt(max(kl_tie.mean(), 0.0) / 2.0)),
    )
    lp = prob.loophole
    if lp.any():
        hon = (p * (prob.kind == HONEST)).sum(1)[lp]
        hk = (p * (prob.kind == HACK)).sum(1)[lp]
        hon0 = (p0 * (prob.kind == HONEST)).sum(1)[lp]
        hk0 = (p0 * (prob.kind == HACK)).sum(1)[lp]
        out.update(
            hack_share=(hk / (hon + hk)).mean(), hack_share0=(hk0 / (hon0 + hk0)).mean(),
            gold_loophole=(p * prob.gold).sum(1)[lp].mean(),
            fail_loophole=(p * (prob.kind == FAIL)).sum(1)[lp].mean(),
        )
    return {k: float(v) for k, v in out.items()}


def t2_check(prob: Problem, logp: np.ndarray, bin_width: float | None = None) -> dict[str, float]:
    """Independent per-prompt, per-level recomputation of the Theorem 2 identities."""
    key = prob.proxy if bin_width is None else np.floor(prob.proxy / bin_width)
    P, P0 = np.exp(logp), np.exp(prob.logp0)
    chain_err = decomp_err = 0.0
    violations = 0
    kl_tie_loop = []
    for n in range(prob.n_prompts):
        p, p0, r = P[n], P0[n], prob.gold[n]
        nz = p > 0
        kl_total = float((p[nz] * np.log(p[nz] / p0[nz])).sum())
        _, inv = np.unique(key[n], return_inverse=True)
        kls = klt = C = D = bound = 0.0
        for t in range(inv.max() + 1):
            idx = inv == t
            pt, p0t = p[idx].sum(), p0[idx].sum()
            gt = (p0[idx] * r[idx]).sum() / p0t
            C += (pt - p0t) * gt
            if pt > 0:
                q, q0 = p[idx] / pt, p0[idx] / p0t
                qz = q > 0
                kl_t = float((q[qz] * np.log(q[qz] / q0[qz])).sum())
                kls += pt * np.log(pt / p0t)
                klt += pt * kl_t
                D += pt * ((q * r[idx]).sum() - gt)
                bound += pt * (r[idx].max() - r[idx].min()) * float(phi_bound(kl_t))
        dJ = float(((p - p0) * r).sum())
        chain_err = max(chain_err, abs(kl_total - kls - klt))
        decomp_err = max(decomp_err, abs(dJ - C - D))
        violations += int(abs(D) > bound + 1e-12)
        kl_tie_loop.append(klt)
    vec = diagnostics(prob, logp, cells(prob, bin_width))
    return dict(chain_err=chain_err, decomp_err=decomp_err, bound_violations=float(violations),
                vec_vs_loop=abs(vec["kl_tie"] - float(np.mean(kl_tie_loop))))


def stack(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in records]) for k in records[0]}


def value_at_kl(traj: dict[str, np.ndarray], key: str, target: float) -> float:
    """Linear interpolation at the first crossing of KL = target (nan if never reached)."""
    kl, vals = traj["kl"], traj[key]
    hit = np.nonzero(kl >= target)[0]
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0 or kl[i] == kl[i - 1]:
        return float(vals[i])
    w = (target - kl[i - 1]) / (kl[i] - kl[i - 1])
    return float(vals[i - 1] + w * (vals[i] - vals[i - 1]))


def on_kl_grid(traj: dict[str, np.ndarray], key: str, grid: np.ndarray) -> np.ndarray:
    return np.array([value_at_kl(traj, key, k) for k in grid])


def score_based_path(prob: Problem, method: str, grid: np.ndarray,
                     cell: np.ndarray | None = None, keep_policies: bool = False) -> dict:
    fn = SCORE_BASED[method]
    recs, pols = [], []
    for s in grid:
        logp = fn(prob, float(s))
        rec = diagnostics(prob, logp, cell)
        rec["pressure"] = float(s)
        recs.append(rec)
        if keep_policies:
            pols.append(logp)
    out = stack(recs)
    if keep_policies:
        out["_policies"] = pols
    return out


# --------------------------------------------------------------------------
# Policy-gradient optimizers (Section 4)
# --------------------------------------------------------------------------

def objective(prob: Problem, logp: np.ndarray, beta: float) -> float:
    p = np.exp(logp)
    val = p * prob.proxy
    if beta:
        val = val - beta * p * (logp - prob.logp0)
    return float(val.sum(1).mean())


def policy_gradient(prob: Problem, theta: np.ndarray, beta: float = 0.0):
    """Exact gradient of J_beta; returns (grad, logp, p, centred advantage)."""
    logp = policy_logp(prob, theta)
    p = np.exp(logp)
    adv = prob.proxy - beta * (logp - prob.logp0) if beta else prob.proxy.copy()
    adv = adv - (p * adv).sum(1, keepdims=True)
    grad = np.einsum("nm,nmd->d", p * adv, prob.psi) / prob.n_prompts
    return grad, logp, p, adv


def coord_sign(g: np.ndarray, rel: float = 1e-12) -> np.ndarray:
    """sign(g), with entries below rel * max|g| set to 0.

    The GRPO estimator's centring term sum_j A_j is zero only up to round-off,
    so coordinates of undrawn responses carry ~1e-17 noise that a bare sign()
    would turn into unit steps.
    """
    scale = float(np.abs(g).max()) if g.size else 0.0
    return np.where(np.abs(g) > rel * scale, np.sign(g), 0.0)


def first_step_size(prob: Problem, hidden: int = 0, factor_init: float = 0.1,
                    factor_seed: int = 0) -> float:
    """max |change of log pi| of a unit-rate coordinate-wise normalized step from theta = 0.

    That step is sign(exact gradient), which is also the first bias-corrected
    Adam / RMSprop / Adagrad step up to eps. With hidden > 0 (theta = U v,
    v0 = 0) only v moves at first, so the step is U0 sign(U0^T grad).
    """
    g, _, p, _ = policy_gradient(prob, np.zeros(prob.dim))
    if hidden:
        U0 = factored_init(prob.dim, hidden, factor_init, factor_seed)
        dtheta = U0 @ coord_sign(U0.T @ g)
    else:
        dtheta = coord_sign(g)
    dz = prob.psi @ dtheta
    return float(np.abs(dz - (p * dz).sum(1, keepdims=True)).max())


def signsgd_occupancy(pi_h: float, pi_k: float, group: int) -> tuple[float, float]:
    """(rho_h, rho_k) of Prop. signSGD on a loophole prompt.

    rho_y = P(the group contains y and at least one failing response), the event
    on which the GRPO coefficient of y is > 0 when honest and hack both pass.
    """
    pp = pi_h + pi_k
    rho_h = 1.0 - (1.0 - pi_h) ** group - (pp ** group - pi_k ** group)
    rho_k = 1.0 - (1.0 - pi_k) ** group - (pp ** group - pi_h ** group)
    return float(rho_h), float(rho_k)


def signflow_prediction(prob: Problem, logp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed and predicted L(i) = log(pi(k|i)/pi(h|i)) - log(b/a), i in S, under exact
    sign flow in the axis-aligned shared-feature model (Prop. signflow).

    With eta(i) the honest logit, isolated: L = (s_u/s_h - 1) eta; exposed (s_u = s_h):
    L = -2 (eta - l*_g)_+ with l*_g = max(0, log(m_g c / (m'_g a))) (l* = inf if m'_g = 0).
    """
    inf = prob.info
    if inf.get("model") != "shared" or inf.get("gamma", 0.0) != 0.0 or inf.get("basis", "axis") != "axis":
        raise ValueError("needs the axis-basis shared-feature model with gamma = 0")
    if inf["variant"] == "exposed" and inf["s_u"] != inf["s_h"]:
        raise ValueError("exposed prediction needs s_u = s_h")
    a, c = inf["a"], inf["c"]
    S = np.nonzero(prob.loophole)[0]
    fail = int(np.nonzero(prob.kind[0] == FAIL)[0][0])
    ell = logp - prob.logp0
    eta = ell[S, HONEST] - ell[S, fail]
    obs = ell[S, HACK] - ell[S, HONEST]
    g = inf["form"][S]
    mg, mp = inf["m_g"][g], inf["m_prime"][g]
    with np.errstate(divide="ignore"):
        lstar = np.where(mp > 0, np.maximum(np.log(mg * c) - np.log(np.maximum(mp, 1e-300) * a), 0.0),
                         np.inf)
    pred = (inf["s_u"] / inf["s_h"] - 1.0) * eta - 2.0 * np.maximum(eta - lstar, 0.0)
    return obs, pred


def natural_direction(prob: Problem, p: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """F^+ grad J as the min-norm solution of min_w E_x E_pi (A - w.e)^2."""
    N, M, d = prob.psi.shape
    e = prob.psi - np.einsum("nm,nmd->nd", p, prob.psi)[:, None, :]
    wts = np.sqrt(p / N)
    X = (wts[..., None] * e).reshape(N * M, d)
    y = (wts * adv).reshape(N * M)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _fail_mass_loophole(prob: Problem, p: np.ndarray) -> float:
    return float((p * (prob.kind == FAIL)).sum(1)[prob.loophole].mean())


def run_flow(
    prob: Problem,
    *,
    beta: float = 0.0,
    natural: bool = False,
    sign: bool = False,
    precond: np.ndarray | None = None,
    step: float = 0.02,
    eta_max: float = np.inf,
    max_steps: int = 5000,
    kl_max: float = np.inf,
    grad_tol: float = 0.0,
    stop_fail_mass: float | None = None,
    record_every: int = 10,
    keep_policies: bool = False,
    cell: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
) -> dict:
    """Euler discretization of (natural) policy-gradient flow on J_beta.

    The step is normalized so that no log-probability moves by more than `step`
    (capped by eta_max). For beta = 0 this traces the same path in policy space
    as the gradient flow, only reparametrized in time. For beta > 0 a
    backtracking safeguard keeps J_beta nondecreasing.
    sign=True: exact-gradient sign descent, direction sign(grad J) in the
    trained coordinates (natural takes precedence). precond: fixed (d, d)
    matrix P, direction P @ direction (kernel P in theta space).
    """
    cell = cells(prob) if cell is None else cell
    theta = np.zeros(prob.dim) if theta0 is None else theta0.astype(float).copy()
    records: list[dict[str, float]] = []
    policies: list[np.ndarray] = []
    k = 0
    grad_norm = np.inf
    while True:
        grad, logp, p, adv = policy_gradient(prob, theta, beta)
        grad_norm = float(np.abs(grad).max()) if grad.size else 0.0
        if k == 0 or (record_every and k % record_every == 0):
            rec = diagnostics(prob, logp, cell)
            rec["step"] = float(k)
            records.append(rec)
            if keep_policies:
                policies.append(logp)
        kl_now = float((p * (logp - prob.logp0)).sum(1).mean())
        done = (k >= max_steps or kl_now >= kl_max or grad_norm <= grad_tol
                or (stop_fail_mass is not None and _fail_mass_loophole(prob, p) < stop_fail_mass))
        if done:
            break
        if natural:
            direction = natural_direction(prob, p, adv)
        elif sign:
            direction = coord_sign(grad)
        else:
            direction = grad
        if precond is not None:
            direction = precond @ direction
        dz = prob.psi @ direction
        dlogp = dz - (p * dz).sum(1, keepdims=True)
        size = float(np.abs(dlogp).max())
        if size <= 1e-300:
            break
        eta = min(eta_max, step / size)
        if beta > 0:
            j_now = objective(prob, logp, beta)
            while True:
                cand = theta + eta * direction
                if objective(prob, policy_logp(prob, cand), beta) >= j_now - 1e-14 or eta < 1e-14:
                    break
                eta *= 0.5
                eta_max = eta
            theta = cand
        else:
            theta = theta + eta * direction
        k += 1
    if records[-1]["step"] != float(k):
        rec = diagnostics(prob, logp, cell)
        rec["step"] = float(k)
        records.append(rec)
        if keep_policies:
            policies.append(logp)
    out = dict(traj=stack(records), theta=theta, logp=logp, steps=k, grad_norm=grad_norm)
    if keep_policies:
        out["policies"] = policies
    return out


def grpo_gradient(
    prob: Problem,
    logp: np.ndarray,
    rng: np.random.Generator,
    *,
    group: int = 8,
    batch: int | None = None,
    beta: float = 0.0,
    std_norm: bool = True,
) -> np.ndarray:
    """One GRPO gradient estimate at the policy logp (group-normalized advantages).

    The KL penalty, if any, uses the exact gradient on the sampled prompts.
    """
    N, M, _ = prob.psi.shape
    B = N if batch is None else batch
    p = np.exp(logp)
    idx = np.arange(N) if batch is None else rng.choice(N, size=B, replace=False)
    P = p[idx]
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0
    draws = (rng.random((B, group))[:, :, None] > cdf[:, None, :]).sum(-1)
    R = prob.proxy[idx[:, None], draws]
    A = R - R.mean(1, keepdims=True)
    if std_norm:
        A = A / (R.std(1, keepdims=True) + 1e-6)
    coef = np.zeros((B, M))
    np.add.at(coef, (np.repeat(np.arange(B), group), draws.ravel()), A.ravel())
    coef -= A.sum(1, keepdims=True) * P
    grad = np.einsum("bm,bmd->d", coef, prob.psi[idx]) / (B * group)
    if beta > 0:
        ell = logp[idx] - prob.logp0[idx]
        ell_c = ell - (P * ell).sum(1, keepdims=True)
        grad -= beta * np.einsum("bm,bmd->d", P * ell_c, prob.psi[idx]) / B
    return grad


OPTIMIZERS = ("sgd", "adam", "rmsprop", "adagrad", "sign")


def run_grpo(
    prob: Problem,
    rng: np.random.Generator,
    *,
    steps: int = 1500,
    group: int = 8,
    batch: int | None = None,
    beta: float = 0.0,
    std_norm: bool = True,
    lr_logit: float = 0.02,
    adam: bool = False,
    adam_lr: float = 0.01,
    optimizer: str | None = None,
    opt_lr: float | None = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    hidden: int = 0,
    factor_init: float = 0.1,
    factor_seed: int = 0,
    kl_max: float = np.inf,
    record_every: int = 20,
    keep_policies: bool = False,
    cell: np.ndarray | None = None,
) -> dict:
    """GRPO-style sampled policy gradient with group-normalized advantages.

    SGD learning rate is set so that the exact initial gradient step would move
    the largest log-probability by lr_logit. The KL penalty, if any, uses the
    exact gradient on the sampled prompts.

    optimizer: "sgd" | "adam" | "rmsprop" | "adagrad" | "sign"; default "adam"
    if adam else "sgd". Adaptive optimizers use step size opt_lr (default
    adam_lr). "adam" uses (adam_b1, adam_b2); "rmsprop" is bias-corrected Adam
    with beta1 = 0 and beta2 = adam_b2; "adagrad" divides by the root of the
    summed squared gradients; "sign" steps opt_lr * sign(grad).
    hidden > 0: theta = U v with U in R^{d x hidden}, U0 = factor_init *
    N(0, 1/hidden), v0 = 0 (two-layer factorization, all of (U, v) trained).
    With hidden > 0 "sgd" is step-normalized at every step (max log-prob move
    lr_logit), since the kernel U U^T + |v|^2 I grows during training.
    """
    opt = optimizer or ("adam" if adam else "sgd")
    if opt not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {opt!r}")
    cell = cells(prob) if cell is None else cell
    N, M, d = prob.psi.shape
    theta = np.zeros(d)
    g0, _, p_init, _ = policy_gradient(prob, theta, beta)
    dz0 = prob.psi @ g0
    size0 = float(np.abs(dz0 - (p_init * dz0).sum(1, keepdims=True)).max())
    lr = lr_logit / max(size0, 1e-12)
    step_lr = adam_lr if opt_lr is None else opt_lr
    if hidden:
        U0 = factored_init(d, hidden, factor_init, factor_seed)
        w = np.concatenate([U0.ravel(), np.zeros(hidden)])

        def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return x[:d * hidden].reshape(d, hidden), x[d * hidden:]
    else:
        w = theta
    m1 = np.zeros(w.size)
    m2 = np.zeros(w.size)
    b1, b2, eps = adam_b1, adam_b2, 1e-8
    records: list[dict[str, float]] = []
    policies: list[np.ndarray] = []
    t = 0
    while True:
        logp = policy_logp(prob, theta)
        p = np.exp(logp)
        if t == 0 or (record_every and t % record_every == 0) or t == steps:
            rec = diagnostics(prob, logp, cell)
            rec["step"] = float(t)
            records.append(rec)
            if keep_policies:
                policies.append(logp)
            if t == steps or rec["kl"] >= kl_max:
                break
        grad = grpo_gradient(prob, logp, rng, group=group, batch=batch, beta=beta, std_norm=std_norm)
        if hidden:
            U, v = split(w)
            grad = np.concatenate([np.outer(grad, v).ravel(), U.T @ grad])
        if opt == "sgd":
            if hidden:
                gU, gv = split(grad)
                dz = prob.psi @ (gU @ v + U @ gv)
                size = float(np.abs(dz - (p * dz).sum(1, keepdims=True)).max())
                w = w + (lr_logit / max(size, 1e-12)) * grad
            else:
                w = w + lr * grad
        elif opt == "adam":
            m1 = b1 * m1 + (1 - b1) * grad
            m2 = b2 * m2 + (1 - b2) * grad ** 2
            mh = m1 / (1 - b1 ** (t + 1))
            vh = m2 / (1 - b2 ** (t + 1))
            w = w + step_lr * mh / (np.sqrt(vh) + eps)
        elif opt == "rmsprop":
            m2 = b2 * m2 + (1 - b2) * grad ** 2
            w = w + step_lr * grad / (np.sqrt(m2 / (1 - b2 ** (t + 1))) + eps)
        elif opt == "adagrad":
            m2 = m2 + grad ** 2
            w = w + step_lr * grad / (np.sqrt(m2) + eps)
        else:
            w = w + step_lr * coord_sign(grad)
        if hidden:
            U, v = split(w)
            theta = U @ v
        else:
            theta = w
        t += 1
        if not np.isfinite(theta).all():
            break
    out = dict(traj=stack(records), theta=theta, logp=logp, steps=t, lr=lr, optimizer=opt,
               opt_lr=step_lr)
    if keep_policies:
        out["policies"] = policies
    return out


# --------------------------------------------------------------------------
# Section 5: stationary points and the tie-preserving subspace
# --------------------------------------------------------------------------

def tie_constraint_matrix(prob: Problem, cell: np.ndarray | None = None) -> np.ndarray:
    """Rows psi(x,y) - psi(x,y') over tied pairs (matrix M of Theorem 4)."""
    cell = cells(prob) if cell is None else cell
    iu, ju = np.triu_indices(prob.n_responses, 1)
    tied = cell[:, iu, ju] > 0
    diffs = prob.psi[:, iu, :] - prob.psi[:, ju, :]
    return diffs[tied]


def null_space(A: np.ndarray, d: int, tol: float = 1e-10) -> np.ndarray:
    if A.shape[0] == 0:
        return np.eye(d)
    _, s, vt = np.linalg.svd(A, full_matrices=True)
    rank = int((s > tol * max(float(s.max()), 1.0)).sum())
    return vt[rank:].T


def realizability_residual(prob: Problem) -> float:
    """Relative residual of r~(x,.) - c(x) in span psi(x,.) (0 iff the KL tilt is in the class)."""
    X = (prob.psi - prob.psi.mean(1, keepdims=True)).reshape(-1, prob.dim)
    y = (prob.proxy - prob.proxy.mean(1, keepdims=True)).reshape(-1)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.linalg.norm(X @ w - y) / max(np.linalg.norm(y), 1e-300))


def stationary_point(prob: Problem, beta: float, max_steps: int = 3000, grad_tol: float = 1e-10,
                     theta0: np.ndarray | None = None) -> dict:
    """Stationary point of J_beta via damped natural gradient with backtracking.

    Vanilla and natural PG have the same stationary points (grad J lies in range F).
    """
    return run_flow(prob, beta=beta, natural=True, step=0.5, eta_max=1.0 / beta,
                    max_steps=max_steps, grad_tol=grad_tol, record_every=0, theta0=theta0)


def score_based_restricted_optimum(prob: Problem, beta: float, max_steps: int = 3000) -> dict:
    """theta_SB = argmax over ker M, and the full gradient there (Theorem 4(c))."""
    B = null_space(tie_constraint_matrix(prob), prob.dim)
    if B.shape[1] == 0:
        theta_sb = np.zeros(prob.dim)
    else:
        sub = prob.with_features(prob.psi @ B)
        theta_sb = B @ stationary_point(sub, beta, max_steps=max_steps)["theta"]
    grad, logp, _, _ = policy_gradient(prob, theta_sb, beta)
    return dict(theta=theta_sb, grad_norm=float(np.abs(grad).max()), logp=logp,
                dim_sb=int(B.shape[1]), J=objective(prob, logp, beta))
