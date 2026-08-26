# FSC fidelity audit — three places notebooks 07–09 do not follow Okano & Kurisu

Written 2026-08-26, at Jun's request, after an audit of what the FSC build actually does
versus what the authors' code does. These were not disclosed when the results in
notebooks 07/08/09 were first reported. **All three affect the augmented estimator, and
therefore affect the placebo null that was reported as the headline.** The un-augmented
FSC (`FSCM`) results are not affected by any of them.

Reference implementation: `/data/wang/junh/githubs/FSC` (read-only), arXiv:2601.07539.

---

## Issue 1 — λ was never tuned: it sat on an interval boundary in 20 of 28 configurations

### What their code does

`FSCM_aug` / `FSCM_aug_covmat` take λ as an argument. Their scripts select it with
`optimise(obj_func, interval = c(a, b))` over a **linear** interval — `c(0, 1)` in
`service.R` L61, `c(0, 10)` in `mortality.R` L148 — and then hardcode the result
(`# 0.00186 is selected`, `# 5.889182 is selected`).

### What we did, and what went wrong

We carried those two intervals over unchanged. They were chosen for *their* outcome
scales (UN trade-in-services covariances; age-at-death quantile functions in years). Our
outcomes are FSQ latent channel statistics on [0, 1] — Gram entries around 0.3–0.6. There
is no reason the same interval should bracket the optimum, and it does not:

| | count of 28 (file × sensor × method × representation) |
|---|---|
| λ at the **ceiling** of the interval | **13** |
| λ at the **floor** | **7** |
| λ strictly interior | **8** — and only 2 distinct values (0.0288, 0.0449), all Sentinel-1 covmat |

### The diagnostic that should have caught it was broken

`fsc_bridge.R` flagged `lambda_at_bound` as `abs(λ − b) < 1e-8`. R's `optimise` converges
to a tolerance of about `.Machine$double.eps^0.25 ≈ 1.2e-4` *relative*, so a λ of
0.999934 against a ceiling of 1.0 is at the boundary for every practical purpose but is
6.6e-5 away — well outside a 1e-8 test. **Every table printed `frac_at_bound = 0.0`.**
That was an artefact of the tolerance, not a finding, and it was reported as if it were
reassuring.

### Consequence

In `FSCM_aug_covmat`, `mat_1 <- solve(t(r_0dot) %*% r_0dot + lambda * diag(M*T_0))`. As
λ grows, `mat_1 → 0` and the augmented weights collapse onto the FSC weights. So:

- the **13 ceiling** configurations are FSC wearing an augmented label — which is exactly
  why the `fsc` and `afsc` rows were nearly identical. That similarity was reported as a
  result. It was a symptom.
- the **7 floor** configurations are essentially unregularized augmentation, i.e. maximal
  extrapolation.
- only **8** were tuned at all.

Their `placebo` / `placebo_covmat` use the **augmented** estimator, so the placebo null
inherits this. Of the placebo runs actually executed: Gram S1 used an interior λ (0.0449);
Gram S2, quantile S2 and combined S2 used ceiling λ; quantile S1 and combined S1 used
floor λ.

### Fix

1. **Search λ on a log grid**, not their linear `optimise`. A linear interval cannot
   bracket an optimum that may lie anywhere across several orders of magnitude, which is
   the actual failure here. The search calls **their unmodified `cross_val` /
   `cross_val_covmat`** at each grid point and takes the argmin — the CV objective is
   theirs, only the search strategy changes. This is a deliberate, disclosed deviation
   and must stay disclosed: their published λ values were found by `optimise` on their
   own scales, and we cannot reproduce that procedure on ours without it hitting a wall.
2. **Assert the optimum is interior** with a *relative* tolerance (1e-3), and fail loudly
   rather than silently reporting a boundary value.
3. Re-run every augmented result and every placebo with the corrected λ.

---

## Issue 2 — the projection back onto the outcome space was never applied

### What their code does

The augmented estimator can produce weights far outside the simplex (measured on their
own service data: min weight −42.6, 11 of 22 negative). The weighted combination is then
not guaranteed to be a valid object of the outcome space, so both of their scripts project
it back:

- `mortality.R` L163: `ascm_outcomes[[t]] <- modif(vals = ascm_outcomes[[t]], low = 0,
  upp = 110/1000, grids)` — truncate to the admissible range, then monotone-rearrange, so
  the result is genuinely a quantile function.
- `service.R` L105: `nearPD(temp_mat_sym, base.matrix = TRUE)$mat` — project the synthetic
  covariance back onto the PSD cone.

### What we did

Neither. `grep nearPD\|modif fsc_bridge.R` returns only comments. Both functions were
verified to work in the R smoke test and then never wired into the fit path.

- For the **Gram** arm this accidentally matches `service.R`, which computes its
  pre-treatment fit on *unprojected* outcomes (L78–83) and applies `nearPD` only when
  converting to matrices for interpretation. So the reported metric is faithful — but no
  valid PSD synthetic covariance was ever produced, so nothing in that arm can be
  interpreted as a covariance matrix.
- For the **quantile** arm this is a straight infidelity: `mortality.R` projects before
  use, we did not. Our augmented "quantile functions" are not guaranteed monotone, i.e.
  they may not be quantile functions at all.

### Fix

Apply the projection to the augmented synthetic outcome before scoring, and report
projected *and* unprojected side by side so the difference is visible rather than assumed
negligible.

One detail their code does not face: our quantile vector is **5 channels concatenated**.
`modif` must be applied **per channel block**, since rearranging the whole concatenation
would sort values across channels. For the combined arm, `modif` on the quantile block and
`nearPD` on the Gram block.

---

## Issue 3 — the prediction intervals were never implemented

### What their code does

`mortality.R` L278–298 constructs conformal-style bands: compute pre-treatment residuals,
take their quantiles, and form `observed − synthetic ∓ quantile_vec` for the post periods.
This is the inference machinery the FSC paper offers on top of a point estimate.

### What we did

Nothing. This was repeatedly cited in discussion as the main reason to adopt FSC — "their
prediction sets give us the inference our pipeline lacks" — and then not built. The
notebooks report point estimates and a placebo rank test only.

### Fix

Implement it following `mortality.R` L278–298, and report the bands alongside the
post-period gaps.

---

## Also missing: the expanding-window validation scheme

Not one of the three above, but relevant to the same question and worth recording.
Notebooks 02–06 score two schemes: `frozen_joint` (fit P01–P08, score P09+P10 pooled) and
`expanding` (fit P01–P08 → predict P09; **refit** P01–P09 → predict P10, the advisor's
design). Notebooks 07–09 implemented only the first. The rolling scheme requires a second
FSC fit per group at `T_0 = 9` and was simply not built.

---

## Status of the previously reported results

**Unaffected** (no λ, no projection, no augmentation involved):

- `FSCM` runs on our panel and behaves as on their data (Σw = 1, ~3 of 5 donors non-zero).
- The FSC holdout ordering, generation-fill Sentinel-2: chip mean (5) 0.807 < Gram (15)
  0.839 < combined (115) 0.946 < quantile (100) 0.958.
- The quantile grid-resolution gate (FSC only, λ fixed at 0.01).
- Non-interference with notebooks 01–06 (≤1e-16).

## Post-fix status (2026-08-26, later the same day)

Fixes 1 (λ log-grid on their unmodified CV) and 2 (projection wired in, per channel
block) are implemented and the prediction results re-run per site × period under both
schemes (`panel_fsc_prediction.csv`, `panel_fsc_lambda.csv`). Findings of the fixes
themselves:

- **The λ boundary hits were the right answer to a badly-posed search.** Their CV is
  monotonically decreasing in λ over 12 orders of magnitude on most of our
  configurations (interior optimum only for Sentinel-1 chip-mean/Gram, λ ≈ 0.01–0.03).
  The augmentation is inert: afsc ≡ fsc to ≤ 0.02 RMSE everywhere.
- **A fourth issue was found while wiring in fix 2**: `np.triu_indices` (row-major) vs
  R's `upper.tri` (column-major) silently scrambled the Gram vector at the language
  boundary. All L²-based results were unaffected (fixed permutation, applied to both
  sides); only `nearPD` — the one step reading the vector *as* a matrix — was corrupted,
  producing the +0.012–0.016 "projection cost" earlier attributed to `nearPD` itself.
  Fixed in `panel_repr.IU5` (column-major, matching `service.R` L92–95) and gated by a
  Python round-trip identity and a cross-language reconstruction check (max|diff| = 0).
  Post-fix the projection shift is exactly 0 (all synthetic Grams already PSD).
- Fix 3 (prediction intervals) remains open.

Two further errata, caught by Jun 2026-08-26 (both corrected in the report, the
notebooks' markdown, and `panel_fsc.py`):

- **The stated binomial ladder was shifted one step too generously.** Explanatory text
  claimed "4 of 10 rank-1 sites → p ≈ 0.016, 5 of 10 → 0.002". The exact
  Binomial(10, 1/6) upper tail is **4/10 → 0.0697, 5/10 → 0.0155, 6/10 → 0.0024** — so
  reaching the neighbourhood of 0.016 needs five rank-1 sites, not four. The executed
  notebooks always computed `binom.sf` exactly (the reported p = 0.0697 for 4/10 was
  correct); only the prose ladder was wrong.
- **The Sentinel-1 cross-cache comparison is not perfectly paired.** The chip-mean
  cache lacks site 07's complete P01–P10 S1 group, so its FSC arms average 9 groups
  against genfill's 10 — and within the chip-mean S1 table the 980-d models still use
  all 10 (they tolerate incomplete windows; FSC needs a rectangular panel). Restricting
  genfill S1 to the same 9 groups shifts means by −0.08 to +0.004 and changes no
  conclusion; the caveat now sits in the report's results section. Sentinel-2 is fully
  paired.

**Withdrawn pending the fixes:**

- Every `afsc` (augmented) number in notebooks 07–09.
- "The B-spline path is slightly worse than the no-basis path everywhere" — both ran at
  ceiling λ (9.999944 of 10; 0.9999 of 1), so that compared two differently mis-tuned
  estimators.
- "The project's first properly-inferred null" — the placebo is a real test and none of
  the 20 cells reached significance, but it ran on an untuned augmented estimator, so it
  should be described as a placebo test with an untuned penalty, not as a tuned
  augmented-FSC null.
