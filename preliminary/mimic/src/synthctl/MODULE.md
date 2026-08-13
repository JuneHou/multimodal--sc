# Module: synthctl

## Purpose

Phase 2 of the latent synthetic control study, and the headline experiment.
MIMIC patients are units and visits are time. For a target patient, fit convex
weights over a donor pool of other patients so the weighted donors reproduce the
target's **pre-period** visits, then use those weights to **forecast the
held-out next visit**:

```
min_{μ,w}  || X_i[0:T0] − μ·1 − Σ_j w_j X_j[0:T0] ||² + P(w)
                                       s.t.  w_j ≥ 0,  Σ_j w_j = 1
forecast:  X̂_i[T0+1] = μ + Σ_j w_j X_j[T0+1]
```

with `μ = 0` and `P ≡ 0` in the default (classical Abadie) configuration, and
two optional extensions from Wang, Xing & Ye (arXiv 2510.26053): an intercept
`μ⁰` (their Eq. 3, after Doudchenko & Imbens 2016) and an L-infinity /
composite max-norm penalty `P(w)` (their Eq. 5).

The question this module answers is not "is the estimator elegant" but "does it
beat the obvious thing". In EHR the obvious thing is last-value-carried-forward
— the patient looks about the same next visit, and usually does. This module
builds the estimator, six baselines (including a supervised ridge regression, so
"beats copying and averaging" and "beats fitting a model" are separate
questions), a placebo, and one evaluation applied identically to all of them.

Methods are ranked by **code-set F1 at matched cardinality**: every method
decodes to the same per-patient budget `K_i`, so the comparison is not decided
by whether a forecast is hard or soft. Brier/MSE, RMSE and cosine are reported
in full as the secondary, probabilistic view. See `evaluate.py`'s header for why
the ranking metric was changed.

**No results are quoted in this file.** Every number previously stated here came
from a run whose outputs have since been purged unreviewed, and unverified
figures must not survive a purge as prose. "Validation performed" below records
only what has actually been executed against the current code.

## Structure

```
src/synthctl/
├── __init__.py      docstring only; submodules imported explicitly
├── fit.py           Panel, solve_simplex_qp, solve_linf, fit_target, fit_all
├── baselines.py     lvcf, nn1, topk_mean, cohort_mean, ridge (+ table labels)
├── evaluate.py      code-set F1 (primary) / Brier / RMSE / cosine / AUROC / boots
├── figures.py       dependency-free SVG (matplotlib is not installed here)
└── MODULE.md        this file

scripts/
├── run_phase2_fit.py       CLI runner; sweeps scaling × T0 × m
├── validate_phase2.py      correctness gate on the estimator (reads MIMIC states)
└── smoke_phase2_fixes.py   synthetic-only unit tests for everything added in the
                            2026-08-01 methodology pass
```

Call path: `run_phase2_fit.py::main` → `run_config` → `state.load_states` →
`fit.Panel.build` (→ `.rownorm_l2()` for the bag ablation) →
`baselines.all_simple_forecasts` → `fit.fit_all` (cached under `results/.cache/`)
→ `evaluate.evaluate_config` → `evaluate.format_table` → `summarise` →
parquet/JSON/SVG.

Public API:

```python
# fit.py
Panel.build(ss: StateSet, T0: int) -> Panel        # donor pool, materialised once
Panel.rownorm_l2() -> Panel                        # per-visit L2 ablation
Panel.target_last_state(i) / donor_index(i) / distances(i, donors)
solve_simplex_qp(A, x, w0=None, maxiter=500, ftol=1e-10,
                 fit_intercept=False) -> SolveResult
solve_linf(A, x, lam, alpha=1.0, w0=None, maxiter=500, ftol=1e-10,
           fit_intercept=False) -> SolveResult
select_donors(panel, i, m, rng=None) -> (rows, selection)
fit_target(panel, i, m=50, rng=None, maxiter=500, ftol=1e-10,
           fit_intercept=False, penalty=None, lam=0.0, alpha=1.0) -> TargetFit
fit_all(panel, m=50, seed=0, random_donors=False, n_jobs=1, ...,
        fit_intercept=False, penalty=None, lam=0.0, alpha=1.0) -> List[TargetFit]
convergence_report(fits) -> dict
# SolveResult: w, sse, status, message, nit, nfev, simplex_violation, mu
# TargetFit  : ... , selection, mu, penalty

# baselines.py
lvcf(panel, i) / nn1(panel, i) / topk_mean(panel, i, k=10) / cohort_mean(panel, i)
ridge(panel, i, alpha=1.0) -> (d,)         # supervised comparator, LOO-patient
ridge_all(panel, alpha=1.0) -> (n, d)      # same, one shared Gram
all_simple_forecasts(panel, k=10, ridge_alpha=1.0) -> {name: (n, d)}
SIMPLE_BASELINES, LABELS, DEFAULT_K, DEFAULT_RIDGE_ALPHA

# evaluate.py
rmse_rows(F, Y) -> (n,)          cosine_rows(F, Y) -> (n,)
boot_mean(v, ...) -> dict        paired_boot(delta, ...) -> dict
topk_sets(F, K) -> [idx]         codeset_metrics(F, Ybin, K) -> dict   # PRIMARY
cv_auroc_oof(X, y, ...) -> dict  pre_post_correlation(pre, post) -> dict
evaluate_config(forecasts, Y, ...) -> dict     # + primary_metric, f1_vs, brier
format_table(result, title, ...) -> str        # ranked by code F1

# figures.py
scatter_sc_vs_lvcf(x, y, path, title, ...) -> Path
histogram(v, path, title, ..., bins=30) -> Path
sc_trajectory(target, synthetic, path, title, T0, ...) -> Path
placebo_distribution(placebo, observed, path, title, ...,
                     lower_is_better=True) -> Path
```

## Hyperparameters

Values **as used**, i.e. what the code does if you run it with no flags. No
end-to-end MIMIC run has been executed against the current code (see "Validation
performed"), so nothing here is back-filled from results.

| name | value | set where | why this value |
|---|---|---|---|
| solver | `scipy.optimize.minimize(method='SLSQP')` | `fit.solve_simplex_qp`, `fit.solve_linf` | Enforces `w ≥ 0` and `Σw = 1` **exactly** rather than by penalty. `cvxpy` is deliberately absent from this project and was not added. The paper's own choice is an interior-point method; SLSQP on the epigraph reformulation reaches the same optimum (verified against a direct non-smooth solve to 1e-7, `smoke_phase2_fixes.py`). |
| `fit_intercept` | **`False`** (default, and what the runner passes) | `solve_simplex_qp`, `solve_linf`, `fit_target`, `fit_all` | `μ⁰` of Wang/Xing/Ye Eq. 3. Implemented and unit-tested, but **off by default**, so the estimator is bit-for-bit the classical one unless asked. Turning it on is a modelling change (the forecast becomes `μ + Σ w_j X_j[T0+1]`, their Eq. 6) and it has not been run end-to-end, so it is not silently made the default. |
| intercept formulation | `μ` **profiled out analytically**; `w` still the only decision variable | `fit._gram`, `fit._mu_of` | `μ` is unconstrained in sign, so `∂f/∂μ = 0` gives `μ*(w) = (1'x − w'(A1))/D` in closed form. Substituting it back yields *the same Gram form on mean-centred data*: `G − aa'/D = ÃÃ'`, `b − s_x a/D = Ãx̃`. So the augmented normal equations are solved exactly, at the cost of one rank-1 downdate of `AA'` up front and **nothing per iteration** — the O(m²) objective/gradient cost is preserved, which is what makes `m=all` affordable. Verified equal to a genuinely joint `(μ, w)` optimisation to 1e-8 on the objective and 1e-5 on `μ`. |
| intercept shape | one **scalar** over the whole flattened pre-period block | `solve_simplex_qp` | Not per-visit, not per-code. The motivating quantity is a patient who is uniformly "more coded" than another, which is a single level shift across all `(T0+1)·d` coordinates. It is also what makes `mu: float` rather than a vector, and it keeps the profiling closed-form. |
| `penalty` | **`None`** (default) | `fit_target`, `fit_all` | `None` = the classical simplex objective. `"linf"` selects `solve_linf`. The runner does not pass it; the L-inf variant exists and is tested but has not been run over the cohort. |
| `lam` (L-inf strength) | **`0.0`** as shipped; no value is in use | `solve_linf`, `fit_target` | On the scale of the **sum-of-squares** residual — this project does not carry the paper's `1/2` factor in their Eq. 4, so a `λ` taken from the paper must be halved to mean the same thing here. `lam=0` is delegated to `solve_simplex_qp` (the epigraph variable would otherwise be unconstrained above and SLSQP would wander a flat direction). **No λ has been selected**: choosing one requires either a cross-validation design or a target sparsity level, and both are modelling decisions that need a run this task did not perform. Monotonicity of `max_j w_j` in `λ` was verified on synthetic data, so the knob does what it says. |
| `alpha` (L-inf composite) | **`1.0`** = pure `λ‖ω‖_∞` (their Eq. 5, line 1) | `solve_linf`, `fit_target` | `alpha=1.0` is a *sentinel* meaning "drop the L1 term", not `α=1` substituted into the composite. `0 < α < 1` gives `λ(α‖ω‖₁ + (1−α)‖ω‖_∞)`. **Negative result, measured:** under the simplex `w ≥ 0, Σw = 1` we have `‖w‖₁ ≡ 1`, so the L1 half is a constant and `α` is **not identified** — `(λ, α)` returns bit-identical weights to `(λ(1−α), 1.0)`. The paper's setting is unconstrained `ω`, where the two norms genuinely differ. Verified to `max|Δw| = 0`. |
| L-inf formulation | epigraph: minimise `f(w) + λ_eff·s` s.t. `s ≥ w_j ∀j`, `s ≥ 0` | `solve_linf` | The paper notes closed forms are unavailable for these penalties, so no Lasso/ridge path is reused. The epigraph is **exactly equivalent, not a relaxation**: `λ_eff > 0` drives `s` down until `s = max_j w_j = ‖w‖_∞`. Because `w ≥ 0` on the simplex, `|w_j| ≤ s` collapses to the linear `w_j ≤ s`, leaving a smooth QP in `m+1` variables with `m+1` linear constraints and analytic Jacobians — exactly SLSQP's problem class. Cost: objective/gradient stay O(m²), but the `m` epigraph inequalities make each iteration dearer than the plain simplex solve. |
| `sse` under a penalty | the **unpenalised** residual `‖x − μ1 − A'w‖²` | `solve_linf` | So `pre_rmse` stays comparable across solver variants. The penalty enters the optimisation only, never the reported fit quality. |
| objective form | Gram: `w'Gw − 2w'b + x'x`, `G = AA'`, `b = Ax` | `fit.solve_simplex_qp` | Exact algebraic rewrite of `‖x − A'w‖²`, not an approximation. Makes each objective/gradient evaluation O(m²) instead of O(m·(T0+1)·d). This is the difference between a 10 s and a multi-minute `m=all` solve; it is what makes the m-sensitivity sweep affordable at all. |
| gradients | analytic, for both objective and the equality constraint | `fit.solve_simplex_qp` | Without them SLSQP finite-differences m+1 times per iteration. |
| warm start | uniform `w = 1/m` | `fit.solve_simplex_qp` | The fit starts at the unweighted donor mean and has to *earn* any sparsity it reports, so the sparsity numbers are not an artifact of the initialisation. |
| `SLSQP_MAXITER` | `500` | `fit.py` | Observed `nit` is 12–20 across every configuration; the cap never bound. |
| `SLSQP_FTOL` | `1e-10` | `fit.py` | Tight relative to the residuals actually seen (post-period squared errors are O(1e−1) for MedCPT, O(1e1) for raw bag). |
| `WEIGHT_EPS` | `1e-4` | `fit.py` | Threshold for the "count of donors actually used". One part in 10⁴ of the simplex. |
| `SIMPLEX_TOL` | `1e-6` | `fit.py` | Feasibility tolerance for the audit. Max observed violation was ~1e−15, i.e. float round-off. |
| `m` (pre-filter) | grid `{25, 50, 100, all}`; **primary 50** | `--m`, `DEFAULT_M` | Solve over the `m` nearest donors by pre-period Euclidean distance. Reported at every grid point (see "m-sensitivity"). `all` = 748 donors at T0=1, 347 at T0=2. |
| pre-filter distance | Euclidean on the flattened pre-period block, `np.argsort(kind="stable")` | `fit.select_donors` | The same geometry the objective minimises. **Stable** sort is load-bearing, not cosmetic: Phase 1 measured ~100 distinct distance values across 748 donors for the bag encoder, so ties are pervasive and an unstable sort would make "nearest donor" a function of BLAS ordering. Stable sort breaks ties by canonical numeric `subject_id`. |
| `T0` | `1` primary; `2` robustness; `0` shifted-`T0` placebo | `--t0` | Cohorts 749 / 348 / 1796 (all `≥ T0+2` visits). |
| encoders × scaling | `bag/raw`, `bag/rowL2` | `CONFIGS` | **MedCPT was dropped from the grid on 2026-08-01.** The state representation is now only the 585-dim binary bag-of-codes, raw and row-L2. A 768-dim dense embedding has no interpretable vocabulary, so it cannot carry the primary metric (code-set F1 needs codes) and every conclusion on it would have rested on the secondary proper-scoring numbers alone. `data/derived/states_medcpt.npz` and the Phase 1 code that built it are **untouched** and `load_states("medcpt")` still works; no configuration points at them. `--encoder medcpt` was removed from the CLI choices rather than left to select zero configurations silently. `scale="none"` throughout (Phase 1 proved `global` is ordering-preserving and the default objective is unregularised, so it would change nothing — note that ceases to hold if `lam > 0` is ever used, since a penalty fixes a scale). `rowL2` is per-**visit** L2, implemented here because `state.py` deliberately does not implement it and must not be modified. |
| `k` (top-k mean baseline) | `10` | `--k`, `baselines.DEFAULT_K` | Fixed a priori. It is the k Phase 1 already reported its neighbour-overlap diagnostics at, so it is the number that was going to be quoted whatever the answer turned out to be. **No baseline was tuned, in either direction.** |
| ridge `alpha` | `1.0` | `--ridge-alpha`, `baselines.DEFAULT_RIDGE_ALPHA` | L2 penalty of the `ridge` baseline. **Fixed a priori and not tuned** — no inner CV, no sweep, the same rule as every other baseline in the file. It is sklearn's own `Ridge` default, so it is the number a reader would assume if none were stated, and choosing it needs no access to the outcome. This makes ridge a *floor*, not a strong opponent: if SC cannot beat an untuned ridge, tuning ridge would only widen the gap. Also matched to `1.0` because the design is wide (`D = (T0+1)·d = 1170` at T0=1 vs 748 donors), so some regularisation is structurally required — `alpha=0` is not solvable in the primal there. |
| ridge fitting protocol | dual/kernel form, leave-one-**patient**-out, intercept fitted | `baselines.ridge`, `ridge_all` | Fit on `panel.donor_index(i)` — the same donor pool the SC fit sees, so neither method trains on its own target. Dual form because `D > n`; the `n×n` system is both smaller and better conditioned than the `D×D` one. An intercept is fitted (both sides column-centred) so it is a fair opponent for an SC *with* an intercept. `ridge_all` shares one Gram across targets and is asserted equal to the per-target path to 1e-9; both agree with `sklearn.linear_model.Ridge` to ~1e-15. |
| primary metric | **code-set F1 at matched cardinality** | `evaluate.evaluate_config`, `format_table` | Changed from post-period RMSE on 2026-08-01. On a binary target MSE is a **proper scoring rule**, so a calibrated soft forecast beats a hard 0/1 one by construction, independently of which codes it names. SC emits a convex combination (soft); LVCF emits 0/1 (hard). Ranking them by RMSE measures the *form* of the prediction as much as its accuracy — demonstrated in `smoke_phase2_fixes.py`, where a constant `0.25` forecast that names nothing beats a hard forecast that is right 75% of the time on MSE and loses to it by 0.23 on F1. Matched-cardinality decoding removes the asymmetry: both sides emit `K_i` codes. |
| fallback ranking | post-period RMSE, **announced in the table header** | `format_table` | Only reachable for a configuration with no interpretable vocabulary. Unreachable from the default grid now that MedCPT is gone; kept so the function cannot silently mis-rank. |
| `nn1` rule | `MAX_K = 1` | `baselines.nn1` | Literally what the source repo `moa-clinical-rag` did. Kept as-is so the table shows what the convex combination buys over the published 1-sparse special case. |
| `K` (code-set cardinality budget) | `|{codes active in the target's own visit at T0}|`, per patient | `evaluate.codeset_metrics` | See the `CARDINALITY RULE` docstring. Available at forecast time, patient-specific, and makes LVCF decode to exactly itself. A fixed global K and a fixed probability threshold were both considered and rejected. |
| `B` (bootstrap) | `2000` | `--boot-B`, `evaluate.BOOT_B` | The vendored default, carried over to `stats.boot_continuous`. **All 2000 are now kept.** |
| bootstrap resampler | `stats.boot_continuous` | `evaluate.boot_mean`, `paired_boot` | **Changed 2026-08-01.** `stats.boot` is written for a binary-label statistic and discards any resample that is single-class in `y`; a per-patient RMSE / F1 / paired difference has no `y`. The previous code fabricated an alternating 0/1 dummy (`_dummy_labels`) purely to satisfy that guard, which was meaningless for the statistic and made the effective resample count a silent function of `n`. `boot_continuous` draws resamples identically (`default_rng(seed)`, `rng.integers(0, n, n)`, same order, same percentile call) but has no guard, so it is **bit-identical to the old path wherever the old path discarded nothing** and correct where it did not. `stats.boot` is unchanged and still used for the AUROC CI, which really is a binary-label statistic. |
| bootstrap seed | `0` | `--boot-seed` | Vendored default; every CI in the report is reproducible. |
| bootstrap unit | **the patient** | `evaluate.paired_boot` | Each patient appears exactly once in the difference vector, so i.i.d. resampling is correct here — unlike the multi-visit case `stats.boot`'s docstring warns about. |
| `alpha` | `0.05` | `evaluate.BOOT_ALPHA` | 95% percentile interval. Not BCa; anti-conservative for small n (matters most for the T0=2 block, n=348). |
| AUROC protocol | `StratifiedKFold(5, shuffle=True, random_state=0)` + `LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')` | `evaluate.cv_auroc_oof` | Byte-identical settings to the vendored `features.cv_auroc`, which is *also* called directly on the bag code sets and cross-checked. The OOF variant exists so dense MedCPT forecasts can be scored at all and so a bootstrap CI can be formed. |
| `--seed` | `0` | runner | The only modelling randomness in the module is the donor draw for the `sc_random` placebo. Seeded per-target as `default_rng([seed, i])`, so a target's random donor set does not depend on how work was chunked across processes: results are identical at any `--jobs`. |
| `--jobs` | `24` | runner | Targets are independent. Results are re-sorted into panel order regardless of completion order, so `--jobs` never changes the output. |
| BLAS threads | `1` per process, set before `import numpy` | `run_phase2_fit.py` header | **Not a style choice.** Parallelism is over targets; left unpinned, each of 48 workers also spawned a full BLAS pool and the 128-core box hit load average 284 with the `m=all` fits making no measurable progress. Single-threaded per worker is both faster and neighbourly. |
| device | CPU only | — | No GPU is touched anywhere in this phase. |

## Inputs

| path | schema | provenance |
|---|---|---|
| `data/derived/states_bag.npz` | `X` float32 `(9582, 585)` 0/1 · `subject_id` · `t` · `row` · `vocab` `(585,)` · `meta_json` | Phase 1, via `load_states("bag")`. Column names are always read from the file's own `vocab` array. **The only state input this module now reads.** |
| ~~`data/derived/states_medcpt.npz`~~ | `X` float32 `(9582, 768)` unit-norm rows · … | Phase 1. **No longer an input.** The file is deliberately left in place and `load_states("medcpt")` still works, but no configuration in `run_phase2_fit.py` reads it. (`scripts/validate_phase2.py` still loads it for its solver checks — see "Validation performed".) |
| `data/derived/timeline_mimic3_mortality.parquet` | `subject_id`, `t`, `label` (only these three columns are read) | Phase 0, **`rederived` cohort mode** — 9,582 rows, 6,112 subjects, 749 with ≥3 visits, 348 with ≥4. Supplies the mortality label at `T0+1` for the outcome AUROC. |
| `src/vendored/stats.py` | `boot`, **`boot_continuous`**, `auc_mw`, `star` | Every confidence interval in this module. `boot_continuous` was added by this phase for the per-patient continuous statistics; `boot` is still used for the AUROC. See `src/vendored/MODULE.md`. |
| `src/vendored/features.py` | `cv_auroc` | The code-set outcome AUROC. **Patched by this phase** — see `src/vendored/MODULE.md`, "Patches applied after vendoring". |

No MIMIC data is read directly and nothing is copied into the tree beyond
`data/derived/` (read-only here) and `results/` / `figures/` (both gitignored).

## Outputs

All paths relative to `preliminary/mimic/`.

| path | schema | who consumes it |
|---|---|---|
| `results/phase2_targets.parquet` | one row per (encoder, scale, rownorm, T0, m, method, subject_id): `post_rmse`, `cosine`, `code_f1`, `K_budget`, and for SC rows also `pre_rmse`, `eff_donors_ipr`, `eff_donors_count`, `max_weight`, `solver_status`, `solver_nit`, `simplex_violation`. `ridge` now appears as one of the `method` values. | any per-patient re-analysis; the figures |
| `results/phase2_weights.parquet` | long format, primary `m` only: (config, T0, m, method, `subject_id`, `donor_subject_id`, `w`) for every `w > 1e-4` | donor-attribution / interpretability follow-ups |
| `results/phase2_forecasts_<key>_m50.npz` | `subject_id`, `truth` float32 `(n, d)`, `fc_<method>` float32 `(n, d)` per method | anyone wanting a different metric without refitting |
| `results/phase2_aggregate.json` | `{meta, configs[]}`; each config carries `primary_metric` / `primary_metric_note` / `secondary_metrics` plus every method's bootstrapped **F1 (primary)**, `f1_vs_ref`, `f1_vs` (all references), Brier, RMSE, cosine, AUROC, all pairwise paired comparisons, sparsity, pre-vs-post correlation and the convergence report. `meta` now also carries `ridge_alpha` and `primary_metric`. | the memo; the audit trail |
| `results/phase2_headline.txt` | every printed table plus the headline / shrinkage / decomposition / m-sensitivity / robustness / pre-period / convergence blocks, and the run metadata. The headline block is now led by code F1 with RMSE alongside. | the readable artifact |
| `results/.cache/fits_*.npz` | cached per-configuration fits (`donor_rows`, `w`, `mu`, diagnostics) | resumability only; safe to delete. **`mu` is new**; caches written before it are read back as `mu=0`, which is what they were. |
| `figures/phase2_sc_vs_lvcf_*.svg` | per-patient SC-vs-LVCF scatter with the y=x line | the memo |
| `figures/phase2_weight_sparsity_*.svg` | distribution of `1/Σw²` | the memo |
| `figures/phase2_pre_rmse_*.svg` | distribution of pre-period RMSE (diagnostic) | the memo |
| `figures/phase2_trajectory_*.svg` | **NEW.** The canonical SC plot: the target's and the synthetic control's trajectories over `t`, with a dashed rule at the pre/post divide. The scalar plotted is **total code mass `Σ_k X[t,k]`** — deliberately a *linear* functional, so the synthetic line really is the same convex combination of the donors' summaries; a norm or cosine would not have that property and the plot would be misleading. The target is the **median-SC-RMSE** patient, picked by a stable argsort so it is neither a cherry-pick nor run-dependent. | the memo — this is how a reader judges whether the pre-period fit is real |
| `figures/phase2_placebo_*.svg` | **NEW.** The placebo/permutation distribution: the same estimator's post-period RMSE on every *other* patient, with the plotted target's value marked and an empirical rank p-value at the `(1+k)/(1+n)` convention. | the memo — a single fit statistic has no scale without it |

## How to run

```bash
/data/wang/junh/envs/medrag/bin/python \
    /data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic/scripts/run_phase2_fit.py --jobs 24
```

Runs the full default sweep (2 bag configs × T0 ∈ {1,2,0} × the per-T0 m grid)
and writes everything listed above. CPU only; no GPU, no network, no model load.
Re-running is cheap: every fit is cached under `results/.cache/` keyed by
(panel, m, selection, seed), so a second run only re-scores.

Useful flags:

```bash
  --encoder {bag,all}          --rownorm {raw,rowl2,all}   --scale {none,global,zscore}
  --t0 1 --t0 2                # repeatable
  --m 25 --m all               # repeatable; 'all' = no pre-filter
  --primary-m 50 --primary-t0 1
  --k 10                       # top-k baseline
  --ridge-alpha 1.0            # ridge baseline L2; NOT tuned
  --seed 0 --boot-B 2000 --boot-seed 0
  --jobs N --no-resume --no-auroc --auroc-all-m --no-forecasts
  --out-prefix phase2
```

The default `python` on `PATH` lacks numpy/scipy/pandas/sklearn; the `medrag`
path is required.

The synthetic-only unit tests for everything added in the 2026-08-01 pass:

```bash
/data/wang/junh/envs/medrag/bin/python \
    /data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic/scripts/smoke_phase2_fixes.py
```

~30 s, reads no MIMIC-derived data, writes only to a temp directory.

## Validation performed

**No end-to-end MIMIC run has been executed against the current code.** Read
that first: everything below is either a synthetic unit test or a source-level
check. There are no cohort-level numbers in this file, and the ones that used to
be here were removed because the run that produced them was purged unreviewed.

**`scripts/smoke_phase2_fixes.py` — 75/75 checks pass** (synthetic arrays only,
fixed seeds, temp-directory output). What it actually establishes:

| area | what was checked |
|---|---|
| intercept | Recovers a planted level shift (`μ*=0.73` recovered to 3e-4) and the planted weights. **Agrees with a genuinely joint `(μ, w)` SLSQP optimisation over 8 random restarts** to 1e-8 on the objective and 1e-5 on `μ` — i.e. the profiled Gram form is not an approximation. `sse` is the residual at the fitted `(μ, w)`. `μ` really is free in sign (a −2.5 shift is recovered). Feasibility unchanged (`simplex_violation = 0`). The decision vector is still `m`-dimensional, so the Gram-form efficiency claim is structural, not asserted. |
| L-infinity | `lam=0` reproduces `solve_simplex_qp` exactly (`max\|Δw\| = 0`). **The epigraph reformulation matches a direct non-smooth `‖·‖_∞` optimisation** (40 random restarts) to 1e-7 on the objective and 1e-4 on the weights. Increasing `λ` monotonically lowers `max_j w_j` (0.534 → 0.106 over λ ∈ {0,…,100}) and raises the effective donor count (2.91 → 9.70), i.e. it does densify. `sse` excludes the penalty. Composes with the intercept. Rejects `lam<0`, `alpha≤0`, `alpha>1`. |
| **α is not identified** | `(λ=1, α=0.4)` and `(λ=0.6, α=1.0)` return **bit-identical** weights, confirming the `‖w‖₁ ≡ 1` degeneracy under the simplex. This is a limitation of porting an unconstrained-`ω` penalty onto a simplex-constrained estimator, and it is documented rather than hidden. |
| ridge | `ridge_all` equals the per-target `ridge` to 2.7e-14; both agree with `sklearn.linear_model.Ridge` to ~6e-15, in both the `D<n` and the `D>n` regime. Perturbing a target's own `Y` by 1e4 leaves its own forecast bit-identical, so the leave-one-patient-out claim is tested, not assumed. Registered in `SIMPLE_BASELINES` and `LABELS`; `all_simple_forecasts` emits it at `(n, d)`. |
| bootstrap | `boot_continuous` is bit-identical (1e-12) to a plain unguarded percentile bootstrap; reproducible under a fixed seed and sensitive to a changed one; `alpha=0.10` is narrower than `alpha=0.05`; accepts a custom statistic; raises on empty input. `stats.boot` is confirmed **unchanged** — it still raises on single-class `y` — while `boot_continuous` on the same continuous data does not. `paired_boot` matches the unguarded bootstrap exactly and no longer reports `n_discarded`; `_dummy_labels` and `_count_discards` are gone from the module. |
| primary metric | `evaluate_config` reports `primary_metric == "code_f1"` when a vocabulary is supplied and falls back to `"post_rmse"` (announced in the table header) when it is not. `f1_vs_ref` and `f1_vs` are emitted; Brier equals RMSE² per patient. **The motivating asymmetry is demonstrated numerically**: a constant `0.25` forecast that names nothing beats a hard forecast that is right 75% of the time on RMSE (0.431 vs 0.497) and loses to it by 0.234 on code F1 (0.267 vs 0.501). `format_table` puts code F1 first, ranks by it, and labels the right-hand columns as the secondary probabilistic view. |
| placebo labelling | `select_donors(..., rng=…)` is labelled `"random"` at every `m` including `m ≥ n−1` and `m=None`; the estimator path is still `"all"` there. `fit_target` propagates it. |
| new figures | Both emit well-formed, dark-mode-aware SVG. `sc_trajectory` draws two polylines, the pre/post rule with both sides labelled, and direct labels for each series (no colour-only encoding); it rejects mismatched or empty input and survives a constant trajectory without dividing by zero. `placebo_distribution` marks the observed value, and its rank p-value was checked against a hand-computed `(1+k)/(1+n)`; the `lower_is_better=False` direction was checked separately; an all-non-finite placebo set raises. |
| `features.py` | See the next block. |

**Item 8 — the two `features.py` fixes claimed by a previously halted agent are
both really in the file.** Confirmed at source level (via an AST round-trip that
strips docstrings, because `features.py`'s *docstring* legitimately names the
old `field or []` pattern in its patch note and a plain substring scan
false-positives on it) and behaviourally:

- `_flatten` no longer contains `field or []` in executable code; it has
  `if field is None: field = []`, and the nested-element check is widened to
  `(list, tuple, np.ndarray)`. It returns `["a","b","c"]` for a length-3 object
  ndarray — the exact call that used to raise
  `ValueError: The truth value of an array…` — and handles `None`, an empty
  ndarray, a nested list and a nested object ndarray.
- `design_matrix` builds `kept = sorted(...)`.
- **Cross-process determinism was re-verified by running three separate
  interpreter processes at `PYTHONHASHSEED` = 1, 987654 and 42** and comparing
  `sha1(X.tobytes())` and `sha1(vocab)`. All three agree:
  `X` `4ac83047eeb9b1a97271a626e35d1e8a6db1f5af`, vocab
  `84fd722549e05c5446143c10cf1b351b3019af0d`, shape `(300, 48)`.

  **Caveat, stated because it changes what this proves:** the bags used are
  **synthetic** (300 random code sets over a fixed 48-code vocabulary),
  generated inside the test from a fixed `random.Random(1234)`. The earlier
  verification recorded in `src/vendored/MODULE.md` used the real Phase 0
  parquet and reported different hashes (`18ae4ebc…` / `230eb3ff…`, shape
  `(9582, 585)`). Those were **not** reproduced here, because re-running them
  requires reading MIMIC-derived data, which this task forbids. So what is
  established is that the *mechanism* is hash-seed independent, not that the
  specific MIMIC artifact re-derives to the recorded digest.

**What was NOT run, and what that costs.**

- **Nothing has been run on MIMIC data at all**, by explicit constraint. No
  cohort-level metric, no table, no figure from real states. `results/` and
  `figures/` were not written to.
- **`scripts/validate_phase2.py` was not run.** It reads `load_states(...)`, so
  it is a MIMIC run. Its check 8 was *edited* to match the new bootstrap
  (`n_discarded` no longer exists), and that edit is therefore **unexecuted**.
  It also still exercises `load_states("medcpt")` for its solver checks even
  though MedCPT is no longer configured; that was left alone rather than changed
  blind.
- **The intercept and the L-infinity variant have never been fitted to a real
  patient.** They are off by default, unreferenced by the runner, and validated
  only against synthetic problems and brute-force reference solutions. Nothing
  is known about how large `μ` actually is on this cohort, whether it improves
  the forecast, or what `λ` would be sensible. **No `λ` has been chosen.**
- **The runner's end-to-end path was exercised on a synthetic panel only.** A
  60-patient synthetic bag-like panel was pushed through `Panel` → `fit_all` →
  `all_simple_forecasts` → `evaluate_config` → `format_table` →
  `_emit_figures` → `jsonable` → `summarise`, writing figures to a temp
  directory. That proves the plumbing composes and that the JSON serialises; it
  proves nothing about cost, convergence or results at `n=749`, `d=585`,
  `m=all`.
- **Ridge cost at full scale is estimated, not measured.** `ridge_all` forms one
  `749×749` Gram and then one Cholesky per target; the arithmetic says tens of
  seconds per configuration, but it has only been timed on `n ≤ 60`.
- **The two new figures have never been rendered from real data**, so their
  layout is untested at realistic scale (749 placebo values, a 3-point
  trajectory). They were visually specified, not visually reviewed.
- No test of whether the row-L2 ablation interacts with the intercept (a level
  shift on unit-norm rows means something different from a level shift on raw
  counts).
- The `phase2_headline.txt` / `phase2_aggregate.json` on disk were produced by
  the **previous** code and are not consistent with any of the above.

## Known limitations

1. **`alpha` in `solve_linf` is not identified under the simplex** (`‖w‖₁ ≡ 1`),
   so the composite penalty collapses to a pure max-norm at strength
   `λ(1−α)`. Measured, documented, not worked around. A faithful port of the
   paper's composite penalty would have to drop the simplex constraints, which
   would also mean the forecast is no longer a convex combination of real
   patients — a bigger change than this pass made.
2. **The intercept is a single scalar over the whole flattened pre-period
   block.** It cannot express a level shift that differs between visits or
   between code groups. That is the right object for the motivating story ("this
   patient is uniformly more coded") and it is what makes the closed-form
   profiling possible, but it is a modelling restriction, not a neutral choice.
3. **`λ` has no selected value.** Shipping the solver without a defensible `λ`
   means the L-infinity variant is currently un-runnable as a reported method.
   Choosing one needs a cross-validation design over the pre-period, or a target
   effective-donor count, and either is a run this pass did not perform.
4. **The ridge baseline is a floor, not a strong opponent** — `alpha=1.0`, no
   inner CV, one hyperparameter never tuned. It answers "does SC beat *a*
   supervised regression", not "does SC beat the best supervised regression".
5. **Code-set F1 is only defined for the bag encoder.** Now that MedCPT is out
   of the grid this is not a live limitation, but it is the reason the fallback
   branch in `format_table` exists, and it means any future dense encoder cannot
   be scored on the primary metric at all.
6. **`stats.boot` and `stats.boot_continuous` now coexist** and are easy to
   confuse. The rule is: binary-label statistic (AUROC) → `boot`; continuous
   per-unit statistic → `boot_continuous`. Nothing enforces it.
