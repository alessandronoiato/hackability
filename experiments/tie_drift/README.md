# Tie drift: CPU simulations

Companion code for `draft_tie_drift.tex` (repo root). Pure numpy + matplotlib.

## Run (from the repo root)

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r experiments/tie_drift/requirements.txt

# everything, small grids (target: < 5 min on a laptop)
python experiments/tie_drift/run_experiments.py all --quick

# everything, full grids and 10 GRPO seeds (target: < 1 h; expected ~10-20 min)
python experiments/tie_drift/run_experiments.py all

# single experiments; E3 optionally with an Adam-GRPO arm
python experiments/tie_drift/run_experiments.py e3 --quick --adam
python experiments/tie_drift/run_experiments.py e2 --seeds 5
```

The optimizer x basis study (below) is opt-in and is not part of `all --quick`.

Outputs go to `experiments/tie_drift/results/{quick,full}/`: one PNG + PDF + JSON
per experiment and `checks_<experiments>.json`. Every run prints `[PASS]/[FAIL]`
lines for the exact theoretical predictions and `[INFO]` lines for
quantities that are measured but not predicted (conjectures). The process exits
with code 1 if any check fails, and the summary lists the failures.

## Model

`N` prompts; on each prompt one honest response (passes, gold 1), one hack
(passes only on the `m` loophole prompts, gold 0 or -1), and failing responses;
`pi0 = (a, b, c) = (0.15, 0.05, 0.80)` on every prompt. Log-linear policy
`pi(y|i) ∝ pi0(y|i) exp(theta . psi(i,y))` with prompt-specific honest
directions `phi_i` and hack directions `u_g` shared by all prompts with surface
form `g = i mod K`. `K = 1` is the coherent hack, `K = N` the randomized
surface form; `pi0` and the verifier are identical for every `K`.
Variants: `isolated` (the hack's form appears only where it works) and
`exposed` (the same form is tried and fails on the other prompts). `tabular`
uses one free logit per (prompt, response).

Optimizers: KL tilt and chi^2 optimum (exact), best-of-n with uniform and with
likelihood tie-breaking (exact formulas, no sampling), exact policy-gradient
flow (Euler steps normalized so no log-probability moves more than 0.02 per
step; for beta = 0 this is the gradient-flow path in policy space), natural PG
(min-norm Fisher solve), GRPO (G = 8 samples per prompt, group-normalized
advantages, SGD by default, optional Adam, optional exact KL penalty).

## What each experiment tests

| exp | file | prediction tested |
|---|---|---|
| E1 | `e1_tie_preservation.png` | T1: tilt / chi^2 / uniform-tie BoN keep the hack share among passing outputs at `b/(a+b) = 0.25` and within-tie KL at 0 (machine precision). Likelihood-tie BoN and PG/GRPO drift. NPG has zero drift in the isolated and tabular models (realizable proxy) and drifts in the exposed model. PG: hack share up in the coherent model (`R > 1`), down in tabular (`b < a`). T2 identities checked on ~300 policies. |
| E2 | `e2_threshold.png`, `e2_initial_vs_final.png` | Theorem (coherence threshold): in the isolated model the PG end state is hack iff `R = m s_u^2 b / (s_h^2 a) > 1`. Heatmap boundary should sit on the dashed line, and the right panel should be a clean step at `log R = 0`. The initial-drift formula is checked exactly in both variants and for honest coherence `gamma > 0`. For the exposed variant / `gamma > 0` the agreement between initial-drift sign and end state is reported as INFO (conjecture, not claimed). |
| E3 | `e3_surface_form.png`, `e3_trajectories.png` | The decisive intervention. Hack share at KL = 1 vs number of surface forms `K`: BoN is flat at 0.25 for every K (asserted bit-identical); PG/GRPO are well above 0.25 at K = 1 and below 0.25 for K > K* = m b / a ≈ 10.7 (isolated; exposed boundary ≈ 8.7). Bottom row: true reward. |
| E4 | `e4_bound_and_binning.png` | T2 bound `|D| <= osc * phi(KL_tie)` on all PG/GRPO checkpoints (asserted, three versions of the bound); tightness ratio printed. Continuous proxy (noise 0.2): within-bin KL is nonincreasing under dyadic refinement; for the tilt it is below `tau^2 delta^2 / 8`; PG/GRPO keep a large within-bin KL at the verifier resolution. |
| E5 | `e5_stationary.png` | KL-regularized stationary points: within-tie KL is 0 when the proxy is realizable (tabular, isolated, exposed with K = N) and strictly positive for exposed K = 1, where `grad J(theta_SB) != 0` (Theorem 4(c,e)). Middle/right: vanilla PG with beta = 0.5: transient drift that vanishes (isolated) vs persistent drift (exposed). |

## E3 optimizer x basis study (does coherence survive Adam?)

GRPO-Adam in the default E3 largely erases the K = 1 vs K = N gap. In E3 every
feature is one parameter coordinate (`theta[1+i]` = honest logit of prompt i,
`theta[1+N+g]` = logit of form g), so a per-coordinate normalizer (Adam, sign
descent) sees the coherence advantage only as a magnitude and removes it. The
study reruns E3 in other parameter bases of the same policy family
(`sim.reparametrize`; logits `psi . B^T w`, `w` trained):

* `axis`: `B = I` (as before);
* `rotated`: `B = Q`, Haar-random orthogonal. SGD is exactly invariant;
* `dense`: `B = E` in `R^{p x d}`, `p = dense_mult * d` (default 4), i.i.d.
  `N(0, 1/p)`. SGD is PG with kernel `E^T E`, which is `I` only in expectation;
* `factored` (experimental, only if requested): dense basis plus a two-layer
  `theta_w = U v` (`U0 ~ factor_init * N(0, 1/h)`, `v0 = 0`), all trained. Its
  "SGD" arm is step-normalized at every step. No exact checks.

Arms per (variant, basis, K): exact PG flow, exact sign flow (both
deterministic), and GRPO with SGD, Adam (lr 0.01, beta2 0.999) and signSGD
(lr 0.01). `--opt-sweep` adds Adam lr {0.003, 0.01, 0.03} x beta2 {0.99, 0.999},
RMSprop (bias-corrected, beta1 = 0, beta2 = 0.99) and Adagrad (lr 0.1).
Adaptive learning rates are in axis-equivalent units: in a non-axis basis they
are rescaled so that the first sign step moves log-probs as far as in the axis
basis. In a random basis a sign step otherwise moves logits about
sqrt(2d/pi) times further.
Rotated/dense bases draw a fresh `Q`/`E` per seed (seed 100 + s), so the std
includes basis disorder; sampling seeds (4000 + s) are shared across bases.

```bash
# quick, default arms, bases axis/rotated/dense (estimate: 3-6 min including the main E3)
python experiments/tie_drift/run_experiments.py e3 --quick --basis rotated dense

# quick, full optimizer sweep (estimate: 8-15 min)
python experiments/tie_drift/run_experiments.py e3 --quick --opt-sweep

# full: 10 seeds, K in {1,...,64} (estimate: 40-90 min)
python experiments/tie_drift/run_experiments.py e3 --opt-sweep

# variants
python experiments/tie_drift/run_experiments.py e3 --quick --basis rotated dense --dense-mult 16
python experiments/tie_drift/run_experiments.py e3 --quick --basis dense factored
```

Runtimes are estimates, not measurements (the code was written without being
run). Each (variant, K) line prints its wall time. Outputs:
`e3_optimizers.png/.pdf` (hack share at KL = 1 vs log2 K, one panel per
(variant, basis), one curve per optimizer, BoN = pi0 share dotted),
`e3_optimizers.json` (per-seed values and summary), and entries in
`checks_e3.json`.

Exact checks (`[PASS]/[FAIL]`; a FAIL is a bug or a wrong proof):

* `E3opt: exact PG flow invariant under a random rotation` and `GRPO-SGD
  invariant under rotation (same sampling seeds)`: tolerance 1e-6.
* `dense-basis PG flow = axis PG flow with kernel P = E^T E`: tolerance 1e-6.
  The dense-vs-axis gap itself is printed as `[INFO]` together with
  `||E^T E - I||_op`. It is the kernel effect; increase `--dense-mult` to shrink it.
* `exact sign flow in the axis basis follows Prop. signflow`: isolated, hack
  share exactly 0.25 for every K (tolerance 1e-9); exposed, the closed form
  `L = -2 (eta - log(m_g c / (m'_g a)))_+` (tolerance 4 x step = 0.04, the
  discretization bound).
* `GRPO gradient >= 0 on honest/form coordinates, P(> 0) = occupancy (Prop.
  signSGD)`: Monte Carlo at theta = 0 (2000 draws, 4 standard errors + 0.002).

Summary lines (`[INFO] E3opt/<variant>/<basis>/<arm>: coherence gap ...`, also
in `checks_e3.json` with fields `h_K1, h_KN, gap, gap_std, gap_gt_2std`, plus
a compact table at the end): the hack share at KL = 1 for K = 1 and K = 64,
the gap, and whether gap > 2 x (seed std of the difference).

What the outcome means:

| observation | reading |
|---|---|
| axis: Adam/signSGD gap small or negative, exact sign flow flat at 0.25 (isolated) and about 0.02 for K <= 32 (exposed) | reproduces the earlier run; the deterministic part is Prop. signflow, the remaining isolated K dependence is gradient occupancy/SNR (Prop. signSGD, Remark adam-axis) |
| rotated and dense: Adam and sign gaps come back close to the SGD gap (> 2 std) | the axis-basis Adam result is an alignment artifact; the direction is alive. The LLM claim becomes "coherence wins unless hack features are coordinate-aligned", and the LLM experiment must report the optimizer |
| rotated and dense: the gap comes back but much smaller than SGD | partial survival; quantify it, and use `--dense-mult 16` to separate kernel effects from optimizer effects |
| Adam and sign erase the gap in every basis | the coherence mechanism does not survive adaptive optimizers. Reframe (e.g. around occupancy/SNR, or SGD-like optimizers only) or drop the direction |
| axis Adam gap changes strongly with lr or beta2 (`--opt-sweep`) | contradicts the SNR heuristic (it predicts weak dependence at matched KL) |
| axis signSGD gap smaller than Adam's | contradicts the occupancy heuristic (it predicts a larger gap for signSGD) |

## Reading the output

* Every `[PASS]` line is an exact prediction of a proven statement. A `[FAIL]`
  is either a bug in `sim.py` or an error in the corresponding proof: report it
  before trusting anything else. Tolerances are 1e-10 for identities and
  1e-12 for bounds.
* `[INFO]` lines are the open questions. The most informative ones:
  * E2 exposed agreement: if it is near 1, the conjecture of Section 4.4 of the
    draft (initial sign predicts the end state) is plausible; if it is far from 1, only the
    isolated-model theorem stands and the "exposed" regime needs its own
    analysis.
  * E3 GRPO K = 1 vs K = N: the effect size of the intervention under
    sampling noise.
  * E3 with `--adam`: whether the coherence advantage survives Adam (in the
    axis basis it largely does not). The `--basis` / `--opt-sweep` study
    says whether that is an alignment artifact.

## Kill criteria for the research direction

Simulation cannot confirm the LLM claim; it can kill it cheaply.

1. Any `[FAIL]` that survives a code review: the corresponding theorem is wrong.
2. E3 GRPO (SGD): if the K = 1 and K = N hack shares at KL = 1 differ by less
   than 2 seed standard deviations, the mechanism is too fragile to sampling
   noise to be visible in RLVR.
3. E3 optimizer study: if Adam (and sign descent) erase the K = 1 vs K = N gap
   in the rotated and dense bases too, not only in the axis basis, the NTK/SGD
   analysis is the wrong model of practice; the LLM experiment must then be
   framed around Adam (or the direction abandoned). The axis-basis result
   alone (already observed) is not decisive.
4. E2 with `gamma > 0` (full mode): if modest honest coherence (gamma = 0.1)
   already makes honest win for all realistic `m`, the regime where coherent hacks
   win is narrow; the LLM experiment must measure gradient coherence before
   claiming the mechanism.

The first LLM-scale test is in `draft_tie_drift.tex`, Section 6: a planted
loophole in an RLVR task, surface form randomized across prompts at fixed
per-prompt pass rate, BoN vs GRPO.

## Files

* `sim.py`: problems, parameter bases (`reparametrize`), exact score-based
  policies, diagnostics (KL chain rule, calibration/drift split, bounds), exact
  PG / NPG / sign flow (optional fixed kernel), GRPO with SGD / Adam / RMSprop /
  Adagrad / signSGD (optional two-layer factorization), stationary-point and
  tie-preserving-subspace solvers.
* `run_experiments.py`: CLI, experiments E1-E5, checks, plots.
