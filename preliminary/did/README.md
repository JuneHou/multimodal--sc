# Post-period holdout — equal-weight donor mean vs estimated-weight SC (post-hoc)

2026-08-13. **POST-HOC, NOT pre-registered** — recorded per the prereg's
deviation rule. Goal: mimic the collaborator's satellite comparison —
equal-weight donor counterfactual vs solver-estimated weights — on data where
the counterfactual can be scored against an **observed truth**.

## Why clinical notes can run this and satellite cannot

The collaborator's pipeline (Satellite
`data/scripts/09_test_counterfactual_average.ipynb`) is a 2×2 DiD:
counterfactual = equal-weight (1/10) mean of 10 matched control sites, effect
= treated change minus mean control change. It contains **no accuracy metric
and no holdout** — parallel trends is assumed via site matching, never
tested. (`Satellite/data/test/` is a one-site pipeline smoke test, not an
evaluation fold.) The counterfactual is unobservable at treated sites, and
her data has a single pre-period, so the assumption cannot be arbitrated
there.

n2c2 is the complement: **no treatment event** (so no effect is estimated
here — this experiment checks the estimator, not an effect), but every
patient's next visit is observed. Each patient is a direct test of "did the
counterfactual construction predict reality?" — 287 tests at the primary
cell vs her 10-site placebo.

## Design

Primary cell: qwen3 embeddings, T0=1 — fit on visits t=0,1, forecast the
**observed** t=2 embedding (n=287; asserted identical to the H2 cohort).
Robustness: T0=0 (n=288, single pre-period — the satellite-like cell), T0=2
(n=274); biolord arm. Donor schema: cosine on the flattened pre-period block
(H2 convention), self excluded. Same solver as the whole battery
(`solve_simplex_qp`, SLSQP).

| estimator | donors | weights | isolates |
|---|---|---|---|
| `sc` | 50 nearest | solver-estimated ŵ | the full SC pipeline |
| `equal10` | 10 nearest | 1/10 fixed | **the collaborator's estimator verbatim** |
| `equal50` | 50 nearest | 1/50 fixed | equal weights without tight matching |
| `lvcf` | — | — | carry last visit forward (do-nothing floor) |
| `sc_random` | 50 random | solver-estimated | donor *selection* vs donor *weighting* |

Scoring: per-coordinate RMSE (and cosine) of forecast vs the observed next
visit, same space for all arms; **paired** per-patient contrasts with
percentile-bootstrap CIs (B=2000, seed 0) and win fractions. Post-hoc → no
supported/refuted verdicts, effect sizes and CIs only.

## Files

- `run_postperiod_holdout.py` — the experiment (CPU-only, ~17 s).
- `results/postperiod_holdout.parquet` — one row per (arm, T0, patient,
  estimator).
- `results/postperiod_holdout_summary.json` — per-cell estimator means,
  paired contrasts, Spearman(pre-fit, forecast).
- `results/` is gitignored repo-wide (n2c2 DUA derivative); outputs carry
  embeddings/statistics only, never note text.

## Results (run of 2026-08-13; all solver fits converged; hand spot-check passed)

Primary cell (qwen3, T0=1, n=287), mean forecast RMSE:

| estimator | mean RMSE | vs `sc` (paired diff, CI) | win frac |
|---|---|---|---|
| `equal50` | **0.02453** | +0.00027 [+0.00018, +0.00036] | equal50 wins 62% |
| `sc` | 0.02480 | — | — |
| `equal10` | 0.02506 | −0.00025 [−0.00034, −0.00016] | sc wins 65% |
| `sc_random` | 0.02536 | −0.00056 [−0.00068, −0.00045] | sc wins 71% |
| `lvcf` | 0.02876 | −0.00395 [−0.00450, −0.00341] | sc wins 80% |

Same ordering in every cell and both encoders: `equal50` < `sc` <
`equal10` < `sc_random` < `lvcf`, all paired CIs excluding zero — with one
exception that matters most:

- **At T0=0 (single pre-period), sc vs equal10 is a statistical tie** in
  both encoders (qwen3 diff −0.00010, CI [−0.00020, +0.00000], win 56%;
  biolord CI spans zero, win 48%). The SC gain over the matched-set mean
  appears only with ≥2 pre-periods (T0=1: win 65%; T0=2: win 77%).
- `sc` > `sc_random` everywhere (win 59–73%): donor **selection** carries
  value independent of weighting.
- Everything beats `lvcf` (win 79–93%): the forecasts are not just patient
  identity.
- Spearman(pre-fit RMSE, forecast RMSE) = +0.20…+0.35 (p ≤ 1e-3) in every
  cell: better pre-period fit does predict a better counterfactual, weakly.
- Effect sizes are small in absolute terms (~1–2% of the RMSE scale);
  the paired design is what makes them resolvable.

## Reading

1. **The collaborator's estimator is not wrong — it is dominated by more
   shrinkage, not by cleverness.** The plain 50-donor mean beats both her
   10-donor mean and estimated-weight SC in every cell. With a noisy
   one-step-ahead outcome, variance reduction from broad averaging beats the
   bias it introduces (classic bias–variance; SC's effective donor count is
   ~10–15, i.e. less shrinkage than 1/50).
2. **Weight estimation buys a real but small gain over the matched-set mean
   — and only with ≥2 pre-periods.** At a single pre-period (the satellite's
   actual setting) SC and equal-10 are indistinguishable on predictive
   accuracy. Any case for SC over her method on Helene cannot rest on
   forecast accuracy at T0=1-composite; it rests on donor *selection*
   (sc > sc_random), weight interpretability/sparsity, and the placebo
   machinery.
3. **Pre-fit quality is a usable but weak proxy** (ρ ≈ 0.2–0.35) for
   counterfactual quality — relevant because pre-fit RMSE is the only
   in-sample diagnostic available on the satellite side.

Caveat: "closer to reality" here means closer in embedding geometry, which
H2 showed diverges from clinical-concept structure. The estimator comparison
is fair (all arms scored in the same space), but the result speaks to
predictive geometry, not clinical fidelity. All patients are untreated;
nothing here estimates a treatment effect.

## Deviation record

Post-hoc extension, not in `latent_prereg_2026-08-11.md`. No generation, no
GPU. Design fixed before seeing any output: m=50 pool, k=10 equal arm
(collaborator's matched-set size), T0 grid {0,1,2} with T0=1 primary,
B=2000/seed 0, paired contrasts.

A previous post-hoc experiment in this folder (decode-from-estimated-weights,
2026-08-12) was removed 2026-08-13 as off-goal (planted weights +
generated-text channel; it tested whether output plausibility can price
weight quality — it cannot). Its **450 LLM completions still count** against
the prereg's ~1,500-generation cap: running total remains **1,450**.
