# Module: tests

## Purpose

Establish that the Phase 2 synthetic-control estimator is **correct**, using
only data constructed inside the test file whose answers are known analytically
before the code runs.

`src/synthctl` was written by an agent that was halted mid-task and whose output
was never reviewed. This module answers "does the code do what its docstrings
say", separately from and prior to "does the estimator answer the scientific
question". The two are not the same, and this module deliberately establishes
only the first.

**These tests cannot produce a research result, by construction.** No MIMIC
state file, no timeline parquet and no cohort is read; nothing is written
anywhere. Every input comes from a seeded `np.random.Generator` inside the test
file. That is enforced rather than promised: `_install_io_guard()` wraps
`builtins.open`, `np.load` and `pandas.read_parquet` and raises `ForbiddenIO` on
any path ending in `.npz` / `.parquet` / `.csv` / `.feather` or lying under the
derived-data, figures or output directories. A reviewer can confirm the claim by
reading forty lines, and would see a loud exception rather than a silent
finding if it were ever violated.

## Structure

```
preliminary/mimic/tests/
├── test_synthctl.py   the whole suite: helpers, 16 tests, and its own runner
└── MODULE.md          this file
```

`test_synthctl.py` has four parts:

1. **Constants** — every seed and tolerance the suite depends on, at the top of
   the file, nothing buried in a call.
2. **The I/O guard** — `ForbiddenIO`, `_check_path`, `_install_io_guard`, and
   `test_io_guard_blocks_project_paths`, which runs first because if the guard
   does not hold, no other result in the file supports a
   "constructed data only" claim. Forbidden directory names are matched as
   whole path **components**, so a relative path is caught exactly like an
   absolute one.
3. **Constructed-data helpers** —
   `make_panel(...)` builds a `fit.Panel` **directly from arrays**, so the
   solver tests never touch the Phase 1 loader; `make_stateset(...)` fabricates
   an in-memory `StateSet` (binary "code bags" from an RNG, never loaded from
   disk) so the end-to-end test can exercise the real `Panel.build` path.
   `_reference_percentile_boot(...)` is an independent bootstrap used to check
   the module's own.
4. **The tests**, then `main()`.

Feature-detection helpers `_accepts(fn, *names)` and `_lookup(module, *names)`
let a test **skip with an explicit message** when the function it targets does
not exist yet, instead of erroring. That is why the suite could be written
against contracts (intercept, L-inf penalty, ridge baseline, continuous
bootstrap) while `src/` was being changed concurrently by another agent.

Nothing under `src/` is modified by this module. Bugs are reported, not patched.

## Hyperparameters

Values **as used** in the recorded run below, not as intended.

| name | value | where | why this value |
|---|---|---|---|
| `SEED` | `20260801` | master seed | Every RNG in the file is `default_rng(SEED + k)` or `default_rng([SEED, i, j])`. No test draws from global numpy state, so the suite is reproducible bit-for-bit and no test can be made to pass by re-running it. |
| `TOL_WEIGHT` | `1e-6` | weight recovery, exact donor, feasibility | The brief's tolerance. Measured error on the primary construction is 6.7e-15, i.e. nine orders of headroom. |
| `TOL_SIMPLEX` | `1e-6` | every solve | Matches `fit.SIMPLEX_TOL`. Max observed violation 2.4e-11. |
| `TOL_SSE_ZERO` | `1e-12` | exactly-representable targets | ~`TOL_WEIGHT²`; the objective is a squared distance. |
| `TOL_ALGEBRA` | `1e-9` relative | Gram-form identity | Worst observed 3.8e-16. |
| `TOL_INTERCEPT` | `1e-5` | intercept recovery | Looser than `TOL_WEIGHT` because profiling `mu` out is a rank-1 downdate of the Gram matrix; measured error 1.5e-7. |
| `TOL_MONOTONE` | `1e-7` | L-inf sweep | Slack when asserting monotonicity in `lam`; a monotone sequence can still wobble by the solver's own precision. |
| `TOL_PERMUTATION` | `1e-4` | permutation invariance (numeric) | Set to `fit.WEIGHT_EPS`. Past that the reported effective-donor counts would become order-dependent, which would be a defect rather than float noise. Measured drift 6.6e-7. |
| `N_UNITS`, `D_DIM`, `T0_DEFAULT` | `25`, `30`, `1` | generic random panel | Small: these are correctness tests, not benchmarks. `D = (T0+1)·d = 60 > m`, so a random donor pool is in general position and the exact convex representation of a target is **unique** — without that, "recover `w*`" would not be a well-posed assertion. |
| weight-recovery pool | 23 donors, `w* = 0.7/0.2/0.1` at rows 5/11/17 | `test_weight_recovery` | 3 real donors "embedded among ~20 random" per the brief. |
| permutation panel | `n=20, d=16, T0=3` | permutation tests | `T0=3` so there is a real 4-element permutation to apply; at the project's primary `T0=1` a permutation is only a swap. |
| tie panel | 13 units, 8-way exact tie at distance 1 | `test_tie_breaking` | Ties are constructed exactly (unit vectors on distinct axes), not approximately. |
| `N_SMOKE` | `30` | end-to-end | The brief's ~30-patient synthetic cohort: 30 patients × 4 visits × d=24 binary. |
| `BOOT_B_TEST` | `200` | every bootstrap in tests | The module default of 2000 is slower and adds nothing to a coverage check. |
| `BOOT_N` | `300` | synthetic `delta` | Large enough that a 95% percentile interval on a mean is stable. |
| `m` in tests | `None` (full pool), `8`, `10`, `3` | various | `m=None` isolates the solver from the pre-filter; the finite values exercise the selection rule. |
| `n_jobs` checked | `1, 2, 3` | determinism | Both the estimator and the seeded random-donor placebo. |
| runner | self-contained, in `test_synthctl.py` | — | **`pytest` is not installed in this environment** (checked: `import pytest` → `ModuleNotFoundError`), so the file carries its own runner that prints PASS/FAIL/SKIP per test and exits non-zero on failure. It is written to plain-pytest conventions (`test_*` names, bare `assert`), so `pytest tests/test_synthctl.py` works unchanged if pytest is ever added; the only pytest-specific behaviour is that a `Skipped` exception would report as an error rather than a skip. |
| device / network | none | — | CPU only, no model load, no network, no file I/O at all. |

## Inputs

**None.** That is the point of the module.

Every array is generated in-process from a seeded RNG. The only things read from
the filesystem are the Python modules under test:

| import | used for |
|---|---|
| `src.synthctl.fit` | `Panel`, `solve_simplex_qp`, `solve_linf`, `select_donors`, `fit_target`, `fit_all`, `convergence_report`, `WEIGHT_EPS` |
| `src.synthctl.baselines` | `lvcf`, `nn1`, `topk_mean`, `cohort_mean`, `ridge`, `all_simple_forecasts` |
| `src.synthctl.evaluate` | `rmse_rows`, `boot_mean`, `paired_boot`, `codeset_metrics`, `evaluate_config`, `format_table` |
| `src.datasets.state` | `StateSet` (**constructed in memory from fabricated arrays**, never loaded) and `sorted_ids` (the canonical id ordering the tie-break test asserts against) |
| `src.vendored.stats` | `boot_continuous`, probed directly |

## Outputs

**No files.** The suite writes nothing: no artifacts, no cache, no fixtures.

Its only output is stdout — one `PASS` / `FAIL` / `SKIP` line per test, with the
measured quantities printed under each pass (weight error, simplex violation,
recovered intercept, permutation drift, bootstrap interval, …) — and the process
exit code: `0` if nothing failed, `1` otherwise.

## How to run

```bash
/data/wang/junh/envs/medrag/bin/python \
    /data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic/tests/test_synthctl.py
```

`-q` suppresses the per-test measurements and the tracebacks. Whole suite: 1.8 s.
The default `python` on `PATH` lacks numpy/scipy/sklearn; the `medrag` path is
required.

## Validation performed

Recorded run, 2026-08-01, against `src/synthctl` as of that afternoon (i.e.
**after** the intercept, the L-inf variant, the ridge baseline and the continuous
bootstrap landed; every skip-guarded test therefore executed for real).

```
synthctl correctness suite  (seed=20260801, constructed data only)
PASS  test_io_guard_blocks_project_paths
PASS  test_weight_recovery
PASS  test_simplex_feasibility
PASS  test_exact_donor
PASS  test_intercept_recovery
PASS  test_linf_densification
PASS  test_lvcf_identity
PASS  test_nn1_matches_independent_argmin
PASS  test_ridge_baseline
FAIL  test_permutation_invariance_bitwise
PASS  test_permutation_invariance_numeric
PASS  test_gram_form_equivalence
PASS  test_paired_bootstrap
PASS  test_continuous_bootstrap_discards_nothing
PASS  test_determinism
PASS  test_tie_breaking
PASS  test_end_to_end_smoke
16 passed, 1 failed, 0 skipped in 1.7s
```

### What each test establishes — and what it does not

**`test_io_guard_blocks_project_paths`** — seven representative project data and
output paths, relative and absolute, are refused by `open`, and `np.load` is
refused too; an ordinary scratch file still opens. Probed in **read** mode on
non-existent paths, so a guard failure raises `FileNotFoundError` rather than
creating or truncating anything.
*Establishes:* the module's central claim — that these tests cannot read MIMIC
data or write project output — is enforced by code rather than by convention.
*Does not establish:* that the guard covers every conceivable I/O path. It wraps
`builtins.open`, `np.load` and `pandas.read_parquet`; a low-level `os.open`, a
C-extension writing directly, or a subprocess would bypass it. The suite uses
none of those, but the guard is a backstop, not a sandbox.

**`test_weight_recovery`** — the target is an exact convex combination
(0.7/0.2/0.1) of three donors hidden among twenty random ones; the solver
returns those weights to **6.7e-15** and zero elsewhere, at the solver level and
again through `fit_target`, and the forecast equals `Σⱼ wⱼ Yⱼ`.
*Establishes:* `solve_simplex_qp` finds the global minimiser of the stated
objective when that minimiser is unique, and the forecast is assembled from the
weights as documented.
*Does not establish:* that the objective is the right one. This is the sharpest
version of the general caveat — a solver can be perfect and the estimand still
be the wrong thing to estimate. It also says nothing about the case the project
actually runs, where no exact representation exists and the minimiser may be
close to non-unique.

**`test_simplex_feasibility`** — 18 solves over `m ∈ {2,3,5,12,24,40}`, all
converged; min weight exactly `0.0`, max `|Σw − 1| = 2.4e-11`.
*Establishes:* the constraints are enforced, not approximated, and
`SolveResult.simplex_violation` reports honestly.
*Does not establish:* anything at the `m = 748` scale the real run uses, nor
that `status == 0` there.

**`test_exact_donor`** — target is a copy of donor 7: weight 0.99999999999 on
that donor, pre-period RMSE exactly `0.0`.
*Establishes:* the degenerate case is not special-cased wrongly and a perfect
pre-period fit is reachable.
*Does not establish:* that a perfect pre-period fit means anything for the
forecast — see `evaluate.pre_post_correlation`, which exists precisely because
it does not.

**`test_intercept_recovery`** — donors offset by a known `c = 1.7`: `mu` is
recovered to 1.5e-7, the weights match the no-offset solution, `mu` is exactly
`0.0` when `fit_intercept=False`, a spurious intercept is not invented on
un-offset data (1.5e-7), and `TargetFit.forecast == mu + Σⱼ wⱼ Yⱼ` — i.e. the
fitted level shift is actually carried into the forecast rather than fitted and
dropped.
*Does not establish:* that a per-patient level shift is clinically the right
model, or that `mu` as a single scalar over the flattened block (rather than
per-coordinate) is the right parameterisation.

**`test_linf_densification`** — on a pool where the unpenalised fit is provably
1-sparse (one donor reproduces the target exactly), max weight falls and
`1/Σw²` rises **monotonically** over `lam ∈ {0, .001, .01, .05, .2, 1}`; the
unpenalised residual is non-decreasing in `lam` (a penalised solution cannot
beat the unpenalised optimum); at `lam = 1e6` the weights are **exactly uniform**
(max weight `0.05 = 1/m`), which is the analytic `lam → ∞` limit; and
`solve_linf(lam=L, alpha=a)` equals `solve_linf(lam=L(1−a), alpha=1)` to
**0.0**, confirming the module's own claim that `alpha` is not identified on the
simplex.
*Does not establish:* that the penalty densifies *materially* at usable `lam`.
On this construction the effect at `lam ≤ 1` is tiny (max weight 1.000 → 0.993,
IPR 1.00 → 1.01) because the exact-donor residual is steep; the magnitude is
construction-dependent and this test only fixes the direction. **It also
surfaced a solver-health issue, recorded but not asserted:** `solve_linf`
returns SLSQP `status = 8` ("positive directional derivative for linesearch") at
`lam = 100` and `lam = 1000` on this pool, while still returning the correct
point. Statuses at `lam ∈ {0…10}` and `1e6` are `0`.

**`test_lvcf_identity`** — `lvcf(panel, i)` is `np.array_equal` to `X_i[T0]` for
every unit, and is a copy rather than a view into `Panel.P`.
*Establishes:* the baseline that matters is exactly the baseline it claims to be,
with no accidental aliasing that a caller could mutate.

**`test_nn1_matches_independent_argmin`** — `nn1` matches an argmin over
pre-period distance computed here with an explicit loop and `np.sqrt(np.sum(...))`,
not with `Panel.distances`; `topk_mean` likewise.
*Establishes:* the neighbour rule is the documented one, checked against an
independent implementation rather than against itself.

**`test_ridge_baseline`** — donors generated by a known linear map
`Y = X M' + 0.01·noise`: ridge RMSE `0.0385` vs cohort-mean RMSE `0.888`.
*Establishes:* the ridge baseline learns a linear structure that the cohort mean
cannot, so it is a live baseline rather than a decorative one.
*Does not establish:* anything about its behaviour on real states, where no such
map exists.

**`test_permutation_invariance_bitwise` — THE ONE FAILURE.** Synthetic control
is "unchanged by design" under a permutation of the pre-intervention time
indices (Abadie et al.; Rho et al., TASC). Applying one permutation to the
pre-period blocks of *every* unit changes **40 of 40** forecasts. The
magnitudes: `max|Δw| = 6.6e-7`, `max|Δforecast| = 2.1e-6` on a forecast scale of
1.37 (≈1.5e-6 relative), while the **objective value at the two points differs
by only 2.9e-11 relative** and the selected donor sets are unchanged.
*Establishes:* the claim as stated — bit-identity — is **false** for this
implementation. Both returned points are optima of the same objective; the
cause is that SLSQP terminates on an *absolute* `ftol = 1e-10` on the objective,
which for a quadratic pins `w` only to ≈`sqrt(ftol)`, combined with `G = A Aᵀ`
being summed in a different order after the permutation (the Gram matrices
differ by 5e-16 relative). It is a precision statement, not a coding error: no
code path branches on the time order.
*Does not establish:* that anything downstream is wrong — see the next test.

**`test_permutation_invariance_numeric`** — the same experiment at the precision
the solver actually delivers: drift stays below `WEIGHT_EPS = 1e-4`, the two
objective values agree to 2.9e-11 relative, no donor set changes, and **no
reported `eff_donors_count` changes**.
*Establishes:* the practical consequence of the bitwise failure is nil for every
quantity the project reports — RMSE, cosine, code-set F1, effective-donor
counts, and the `w > 1e-4` weight parquet all sit far above the 1e-6 drift.
*Does not establish:* safety for any future claim about individual weights at
the 1e-6 level, or reproducibility across BLAS builds — the same mechanism that
makes a permutation matter would make a different BLAS matter.

**`test_gram_form_equivalence`** — `w'(AA')w − 2w'(Ax) + x'x` equals
`‖x − A'w‖²` to 3.8e-16 relative over four `(m, D)` shapes × five random
simplex points, and `SolveResult.sse` equals the direct residual exactly.
*Establishes:* the O(m²) rewrite that makes the whole phase affordable is the
exact algebraic identity the docstring claims, and the reported `sse` is the
residual at the returned `w` rather than a stale objective value.

**`test_paired_bootstrap`** — a `delta` centred at exactly −0.5: the 95%
interval `[−0.605, −0.391]` covers it and excludes 0 in the winning direction;
after a +5 shift the interval `[+4.39, +4.61]` excludes the old mean and
`beats_ref` flips to `False`; the reported `win_frac` matches an independent
count.
*Establishes:* the CI machinery brackets a known mean and moves with the data.
*Does not establish:* calibrated coverage. This is one draw, not a coverage
simulation; and the interval is a plain percentile bootstrap, not BCa, so it is
anti-conservative for small `n` or a skewed statistic.

**`test_continuous_bootstrap_discards_nothing`** — the CI from `paired_boot` and
`boot_mean` is **bit-identical** to `_reference_percentile_boot`, an independent
percentile bootstrap written in the test file that keeps every one of the `B`
draws. Checked at `n = 300` and at `n = 6`; at `n = 6` the superseded
binary-label resampler would have discarded **7 of 200** draws (0 at `n = 300`).
*Establishes:* the continuous resampler drops nothing, and that this was a real
defect at small `n` rather than a stylistic tidy-up — the `T0 = 2` block and any
subgroup analysis are where it would have bitten.
*Does not establish:* that the bootstrap unit is right. Each patient appearing
once in `delta` is an assumption about the data, and no test on constructed data
can check it.

**`test_determinism`** — `fit_all` twice at `n_jobs=1`, then `n_jobs=2`, then the
seeded placebo at `n_jobs=1` vs `n_jobs=3`: weights, forecasts and donor rows all
`np.array_equal`. Changing the seed *does* change the placebo's donor sets, so
"seeded" is not vacuous.
*Establishes:* results do not depend on how work was chunked across processes,
which is the claim `--jobs 24` rests on.
*Does not establish:* determinism across machines, BLAS builds or scipy
versions — see the permutation finding for why that is not a pedantic caveat.

**`test_tie_breaking`** — an exact 8-way distance tie resolves to panel rows
`[1,2,3]` = subject ids `['1001','1002','1003']`, stably over repeated calls;
the panel's row order is asserted to *be* `sorted_ids` order rather than assumed
to be; `nn1` picks the same first donor; and the random-donor placebo stays
labelled `"random"` when `m` covers the whole pool.
*Establishes:* the stable-sort guarantee holds, and "nearest donor" under ties
is canonical numeric id order rather than BLAS ordering — load-bearing for the
bag encoder, whose distances are integer-valued and heavily tied.
*Does not establish:* that breaking ties by patient id is clinically neutral.
It is arbitrary; it is merely arbitrary *reproducibly*.

**`test_end_to_end_smoke`** — a fabricated 30-patient × 4-visit × 24-code cohort
through `Panel.build` → `rownorm_l2` → `all_simple_forecasts` → `fit_all`
(estimator and placebo) → `evaluate_config` → `format_table` →
`convergence_report`. All seven methods produce finite forecasts; RMSE, cosine,
F1 and Jaccard are finite with the mean inside its own CI; the outcome AUROC is
finite; the vendored `cv_auroc` raised nothing; `Panel.build` is checked not to
accumulate visits (`X_i[T0]` is that visit's own codes); the cardinality rule is
checked to make LVCF decode to exactly itself (F1 = 1.0); no fit failed to
converge and max simplex violation was 2.0e-15.
*Establishes:* the modules **compose**. Phase 1 found they did not
(`features.py` crashed on numpy object arrays), so this is a real assertion.
*Does not establish:* that any number in it is meaningful. The cohort is
fabricated; the metrics are asserted finite and self-consistent, never compared
across methods. No ranking of methods is produced anywhere in this module, and
that is deliberate.

### The general caveat, stated once

Everything above is *internal* correctness: the code computes what the
docstrings say, on data where the right answer is known by construction. None of
it is evidence about MIMIC, about clinical states, or about whether convex
donor-matching over patient-visits is a sound design. A passing weight-recovery
test proves the solver optimises the stated objective; it says nothing about
whether that objective is the right one for the science. Read
`src/synthctl/MODULE.md` for that question.

## Known limitations

1. **The permutation failure is reported, not fixed.** `src/` is owned by
   another agent; this module does not patch it. If it is judged worth fixing,
   the lever is `SLSQP_FTOL` (absolute on an objective whose scale varies by
   encoder) or a polish step, not the test.
2. **Scale.** Every panel here is ≤ 30 units and ≤ 60 pre-period coordinates.
   The real configuration is 749 targets × 748 donors × 1536 coordinates. Solver
   convergence, conditioning and runtime at that scale are **not** covered; the
   only evidence about them is `convergence_report` on the real run.
3. **`m = all` is untested at realistic `m`.** The largest pool solved here is
   40 donors.
4. **No coverage simulation.** The bootstrap tests check that an interval
   brackets a known mean once, not that 95% intervals cover 95% of the time.
   A proper coverage check would need thousands of replicates and would still be
   a check on the percentile method, which is known to be anti-conservative at
   small `n`.
5. **The L-inf magnitude is construction-dependent.** Monotonicity and the
   `lam → ∞` limit are established; "how much denser at a useful `lam`" is not,
   and the `status = 8` at intermediate `lam` is recorded rather than asserted,
   so a future regression there would not fail the suite.
6. **No test of `figures.py`**, of `scripts/run_phase2_fit.py`'s argument
   handling, or of the caching layer.
7. **The end-to-end test asserts finiteness and self-consistency, not
   accuracy.** It would pass on an estimator that was internally coherent and
   completely useless. That is the correct division of labour — accuracy is the
   real run's job — but it means "smoke test passes" must never be quoted as
   evidence that the method works.
8. **The guard had a hole, and it did damage before it was closed.** The first
   version of `_check_path` matched forbidden directories as substrings *with
   surrounding separators*, so it caught `/x/figures/y` but not the relative
   `<output>/y`. A write-mode probe of the guard during development therefore
   went through and **truncated `phase2_headline.txt` to 0 bytes** (2026-08-01
   16:22). That file is a rendering of results held in full by the intact
   `phase2_aggregate.json`, and the fits cache was untouched, so it is
   regenerable by re-running the Phase 2 runner — but it was destroyed, not
   modified, and nothing in this module restored it. `_check_path` now matches
   whole path components, and the guard is probed in read mode only.
9. **Skip-guarded tests depend on names.** `_accepts` / `_lookup` probe for a
   fixed set of parameter and function names. If `src/` renames
   `fit_intercept`, `solve_linf`, `ridge` or `boot_continuous`, the affected
   test will silently start skipping instead of failing. The run recorded above
   has **zero skips**, so every test executed for real; a future run reporting
   skips should be read as "this feature was not tested", not as "this feature
   is fine".
