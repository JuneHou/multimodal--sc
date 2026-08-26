# ts_SCM_ASCM — SCM / ASCM validation of the tokenizer pipeline on the biweekly panel

**What this folder is** (2026-08-19, Jun-directed): the embedding-space arm of the panel
validation the collaborator ran in feature space (`data/scripts/14_test_scm_validation.ipynb`,
`15_test_augmented_synthetic_control.ipynb`). Two experiments — **SCM** and **ridge ASCM**,
their exact algorithms — with donors selected by our embedding method (5 nearest neighbors
in TerraMind tokenizer latent space, from every control that has panel imagery) and
weights fit on the 980-d quantized latents; then the synthetic counterfactuals are
**decoded and compared with the observed treated images** at held-out pre-hurricane
periods, under both the collaborator's frozen split (fit P01–P08 → score P09+P10) and
the advisor's expanding window (fit P01–P08 → predict P09; fit P01–P09 → predict P10).
Separate folder from `embed_DiD/` because the data is different: the 10-treated ×
50-control × 20-period biweekly panel, not the 26-site before/after chips.

## Limitation: kNN cannot search the 260-control snapshot pool

`embed_DiD` ran kNN over **all 260** covariate-matched controls because those sites had
before/after snapshot chips. This panel experiment cannot repeat that search.

kNN distance is ‖encode(treated image) − encode(control image)‖, averaged over shared
P01–P08 periods. A control with no image has no latent and cannot enter the neighbor
list. The biweekly files on disk are only the 60-site sample
(`data/daily_datasets/selected_site_sample.csv`): 10 treated (`0001`–`0010`) plus their
**top-5** matched controls (ranks 1–5) = **50**. Of `finals/site_matching_table.csv`'s
260 controls (26 treated × 10 ranks), **210 have no P01–P20 GeoTIFF** — ranks 6–10 of
these 10 treated sites, and all controls of treatments `0011`–`0026`. They were never
encoded (`data/embeddings_tok_panel/latents_biweekly.npz` contains 60 sites).

So “kNN over all 50” means: search **every control that exists in this dataset**. That
is already unrestricted given the imagery. It is *wider* than notebook 14 (which used
only each treated site’s own 5 covariate matches and did not share the pool). The “5”
in “5 nearest neighbors” is how many of those 50 are kept for the simplex, not the
search set.

Searching the snapshot-era 260 would require downloading, compositing, and encoding
those 210 missing sites. It is a data gap, not a filter in `knn_donors()`. S1 neighbor
distances being almost flat (12.66–13.28) is consistent with “nobody in this 50 is
close in 980-d”; whether the other 210 would help is untestable until they have panel
images.

## Pipeline

| notebook | role | writes |
|---|---|---|
| `01_encode_biweekly_panel` | encode all 2,322 biweekly images (channel-first tifs transposed; determinism + orientation gates) | `../../data/embeddings_tok_panel/latents_biweekly.npz`, `manifest.json`, `panel_latent_index.csv` |
| `02_scm_latent_validation` | Experiment 1: kNN donors, simplex SCM (SLSQP, Σw=1, w≥0, joint over 980 dims), both validation schemes, P11–P20 effects, quality-40 sensitivity arm | `panel_knn_donors.csv`, `panel_scm_weights.csv`, `panel_scm_validation.csv`, `panel_scm_effects.csv`, 2 figures |
| `03_ascm_latent_validation` | Experiment 2: ridge ASCM (Ben-Michael et al. Eq. 18, closed form, λ by leave-one-period-out CV) | `panel_ascm_weights.csv`, `panel_ascm_lambda_cv.csv`, `panel_ascm_validation.csv`, `panel_scm_vs_ascm.csv`, `panel_ascm_effects.csv`, 2 figures |
| `04_decode_holdout_groundtruth` | decode synthetic counterfactuals (P09–P20) + reconstruction floor; three-space comparison vs observed truth; feature-unit effect trajectories | `panel_decode_validation.csv`, `panel_effect_features.csv`, 4+ figures |
| `05_generation_cloudfill_panel` | TerraMind-large generation cloud fill (S1→S2 full replacement, 791 chips; S2→S1 for 35 whole-missing S1) + full pipeline rerun through the modules; comparison vs the chip-mean run | `panel_genfill_*.csv` (provenance, donors, SCM/ASCM, decode, effects, vs_chipmean), 8 figures |
| `06_ridge_estimator` | Experiment 3: the unconstrained-ridge SC estimator of Wang et al. (arXiv:2509.18156, text embeddings) run through this folder's holdout, 4 λ/intercept variants × both latent inputs, scored against the w=0 null; three-way SCM/ASCM/ridge comparison | `panel_ridge_*.csv` and `panel_genfill_ridge_*.csv` (weights, lambda_cv, validation, effects), `panel_[genfill_]three_way_comparison.csv`, 2 figures |
| `07_fsc_quantile` | Experiment 4a: Okano–Kurisu **functional SC** on the per-channel token *distribution* (quantile functions, 2-Wasserstein — their `mortality.R` case). Both augmented paths (B-spline and no-basis), grid-resolution gate, placebo | `panel_[genfill_]fsc_quantile_*.csv`, 1 figure |
| `08_fsc_gram` | Experiment 4b: FSC on the 5×5 channel **Gram matrix** (15 upper-triangle entries — their `covvec`/`service.R` case exactly, SPSD/Frobenius). Fullest placebo coverage (2 caches × 2 sensors × 4 post periods) | `panel_[genfill_]fsc_gram_*.csv`, `panel_fsc_gram_placebo_summary.csv`, 1 figure |
| `09_fsc_combined` | Experiment 4c: FSC on the **direct sum** quantile ⊕ Gram (our extension, not in their code) | `panel_[genfill_]fsc_combined_*.csv`, 1 figure |

**Reusable modules** (extracted 2026-08-20, parity-proven against the executed 02/03 to
≤1e-12): `panel_lib.py` (index/chips/Panel/kNN/flags + tokenizer & generation machinery +
decode-vs-truth + figures), `panel_scm.py` (`scm_fit`, `run_scm`), `panel_ascm.py`
(`ascm_fit`, λ CV, `run_ascm`), `panel_ridge.py` (`ridge_fit`, `null_rmse`, `run_ridge`,
`compare_all`), and for notebooks 07–09 `panel_repr.py` (pooled representations +
their `func_vals_list` adapter), `panel_fsc.py` (`RBridge`, `run_arm`, `run_placebo`,
`placebo_ranks`) and `fsc_bridge.R`. A new experiment only changes the latent dict fed to
`Panel`; notebook 05's first cells re-assert parity on every run, and notebook 06
re-asserts that SCM/ASCM still reproduce their committed CSVs (≤1e-15) with donors
unchanged.

## Functional SC (notebooks 07–09): their R code is the estimator

Notebooks 07–09 do **not** reimplement functional SC. They call Okano & Kurisu's reference
R implementation (`/data/wang/junh/githubs/FSC`, arXiv:2601.07539) **unmodified**, through
`fsc_bridge.R` — a thin CLI that `source()`s their `main_functions.R` by absolute path,
rebuilds their `func_vals_list` from a binary block written by `panel_fsc.py`, calls
`FSCM` / `FSCM_aug` / `FSCM_aug_covmat` / `cross_val*` / `placebo*`, and writes CSV back.
Their repo is read-only; nothing there is ever edited.

*R environment* — `/data/wang/junh/envs/rsynth` (R 4.3.3). Their code needs packages that
were missing; installed 2026-08-25 with the env's own toolchain on PATH (the conda
`x86_64-conda-linux-gnu-gfortran` is not on PATH by default, which is why source builds
fail without it): `quadprog` 1.5-8, `cubicBsplines` 1.0.0, `Rearrangement` 2.1,
`Matrix` 1.6-5, plus the `lattice`/`survival`/`MASS`/`SparseM`/`MatrixModels`/`quantreg`
chain. Current CRAN `Matrix` and `MASS` require R ≥ 4.4, so archive builds were used
(`Matrix_1.6-5`, `MASS_7.3-60.0.1`). Verified by running their unmodified `service.R` and
`mortality.R` estimation paths on their own published data.

*Two properties of their code worth knowing before reading results.* (1) Their augmented
weights **sum to 1 even though nothing constrains them to** — it falls out of the centring
step (`main_functions.R` L272-281), which zeroes the row sums of `r_0dot`; this differs
from `panel_ascm`, where Σw=1 is imposed by KKT. (2) The augmented estimator costs
O((M·T₀)²) — measured `cross_val_covmat`: 6.5 s at M=45, 33 s at M=100, 218 s at M=500 —
while plain `FSCM` is free at any M. Their own scripts therefore select λ **once** and
hardcode it (`service.R` L56-61, `mortality.R` L143-148); notebooks 07–09 mirror that with
`lambda_="cv_once"`, and pin the same λ inside their `placebo` by passing a 1e-9-half-width
interval to its `optimise` call.

**`null_rmse` — always report it.** `panel_ridge.null_rmse()` gives the RMSPE of the
w = 0 predictor, i.e. the pooled P01–P08 mean. Pooled over all 60 sites that is 1 by
construction of the scaler, but **per site it scatters around 1**, and 37 of the 60
per-site nulls are already below 1. So `flag_level` (pass iff RMSPE < 1) is leaky: an
estimator can pass by predicting nothing. Quote `gain_over_null` next to it — this
applies to the SCM/ASCM tables in notebooks 02, 03 and 05 as much as to notebook 06.

Fixed knobs: TerraMind v1 tokenizers (frozen), latent (5,14,14)=980-d, chip-mean NaN fill,
bilinear 101→224, decode timesteps 50 / seed 0; z-scoring per sensor × dim pooled over all
60 sites, P01–P08 only, ddof=1; a period enters a fit only if treated AND all 5 donors have
latents; flags = notebook-14 ratio (≤1.5/≤2.0) AND whether holdout RMSPE beats the mean predictor (`flag_level` pass if RMSPE < 1). Run order 01→02→03→04, then 05 standalone, then 06 standalone (CPU only — no GPU,
nothing re-encoded; it reads both latent caches and refits weights)
(it re-derives everything through the modules; `satellite` env, one GPU). Notebook 05
adds the generation knobs: `terramind_v1_large_generate`, `standardize=True`,
`timesteps=10`, full-image replacement, seeds on `random`+torch.

## Results (executed 2026-08-19, all gates passed)

1. **Latent-space SCM has no held-out predictive power.** Holdout RMSPE sits at the mean
   baseline (≈ 1) for every site and sensor (S1 1.03–1.09; S2 0.89–1.44), and the training
   RMSPE is the same — the weighted average of 5 donor latents cannot beat the overall
   site mean even in-sample. Every ratio flag is "good" (nothing overfits because nothing
   fits); every `flag_level` FAILs. The 980 latent dimensions mostly encode site-specific spatial layout
   that no other site shares. The quality-40 filter does not change this.
2. **ASCM is inert**: 72/80 λ at a grid boundary, no negative weights (main w8),
   validation identical to SCM to the third decimal, ≤0.7% of latent values leave [0,1].
   Ridge extrapolation has nothing to leverage when the donor span is this far from the
   target — the stronger version of the collaborator's own feature-space ASCM result.
3. **After decoding, the S2 vegetation indices are decoder-limited, not donor-limited**:
   synthetic NDVI |error| 0.17 vs reconstruction floor 0.15 (NDWI 0.145 vs 0.137) — a
   perfect latent would barely improve them. S2 reflectance bands: synthetic 1.1–3.6×
   floor. S1 chip means: synthetic *beats* the per-image floor (VV 0.39 dB vs 0.87) —
   chip-mean levels are easy for an averaged latent; this says nothing about spatial
   fidelity. Per-pixel imagery is far from usable in either case (VV floor 2.4 dB RMSE).
4. **Feature-unit effects (P11–P20, decoded)**: mean NDVI −0.037 (SD 0.157), NDWI −0.041,
   B8 −0.040 — the vegetation-loss direction; S1 VH +0.26 dB (0.91). Per-site effects are
   below the P09/P10 noise level; only cross-site mean directions carry information.
   In latent units, S2's mean gap rises 0.89 → 1.16 pre→post (+31%); S1 does not move.
5. **Verdict vs the feature-based pipeline**: the collaborator's notebook-14 validation
   (0.18–0.78 SD against its own scaler) beats the mean baseline that this latent-space
   pipeline fails. Fitting weights on 11 chip-mean features works; fitting them on raw
   spatial latents does not. Candidate redesigns (decisions to make, not defaults): fit
   weights on decoded features; pool the latent spatially before weighting; keep more
   than 5 neighbors from the 50; or collect panel imagery for the other 210 of the 260
   snapshot-era controls and re-run kNN. The last of these is new data, not a code change.
6. **Generation cloud fill (notebook 05, executed 2026-08-20) raises the S2 ceiling but
   does not rescue the latent pipeline.** Replacing cloudy S2 chips with TerraMind-large
   S1→S2 generations (IBM recipe verbatim, full replacement; all S2 images incl. treated
   P09/P10 before encoding only; ground truth always raw observed pixels): S2 holdout
   RMSPE 1.18 → 1.05 mean in standardized units — each run z-scored by its own pooled
   pre-period SD (expanding P09 0.97, first cells below the mean
   baseline), `flag_level` passes 4/30 → 10/30, and the collaborator's problem sites
   improve most (site 4: 1.38 → 0.73; site 5: 1.19 → 0.98) — supporting the
   image-quality confound suspected in the report. **Fixed-scaler caveat:** re-scored on
   the ORIGINAL run's scaler the absolute embedding error is nearly unchanged
   (P09 1.107 → 1.092; P10 1.244 → 1.225): the standardized-units improvement is mostly a larger
   cross-site latent spread (0.226 vs 0.206 median dim-SD; control diversity UP, no
   homogenization), i.e. generation fixes donor selection and latent signal-to-noise,
   not donor span. Clean inputs alone do not create predictive power. Two diagnostics:
   the chip-mean run's S2 kNN donors overlap the genfill run's by only 0.6/5 (S1 4.7/5)
   — chip-mean S2 distances were largely comparing cloud masks, not land; and arm B
   (S2→S1 for 35 whole-missing acquisitions) restored every fit to 8/8 training periods.
   Decoded S2 indices improve to their generation-inclusive floor (NDVI |err| 0.135 vs
   0.174) while B8/B11 worsen — NIR/SWIR is what radar-conditioned generation invents.
   The decoded NDVI effect grows to −0.137 (≈ the validation error): hypothesis-
   generating only, since the counterfactual side is generation-derived and the
   feature-based arm sees no effect above noise.
7. **The unconstrained-ridge estimator (notebook 06) beats SCM/ASCM only by refusing to
   use the donors — and reveals that the simplex was actively hurting.** Wang et al.'s
   text-SC estimator (unconstrained ridge, λ = 1 fixed, no Σw = 1) scores holdout RMSPE
   0.988 (S1) and 0.994 (S2, per-period-normalized variant) against SCM/ASCM's 1.061 and
   1.050, and wins 58/60 cells. But `sum_weights` collapses to 0.085–0.169 (S1) and
   0.50–0.75 (S2), CV picks λ medians of 631/1,395 on a grid topping out at 1e4, and the
   **w = 0 null for S1 is 0.991** — so the true gain is 0.002–0.003 SD. On S1 ridge has
   no predictive power; it simply shrinks to the pooled mean, which SCM cannot do.
   **SCM/ASCM at 1.06 are worse than predicting nothing**: Σw = 1 forces commitment to a
   donor mix worse than the null, which sharpens result 1 above. S2 shows a little real
   signal but unsystematically — mean gain +0.027, median −0.000, five sites up (04:
   0.990 → 0.803) and five down. The paper's two undetermined details are immaterial
   here: literal λ = 1 ≡ CV on S1 (0.9884 vs 0.9883, as predicted from the scaling), and
   the unpenalized-intercept reading of its `j=0` term changes nothing (0.9883 either
   way), so per-unit demeaning in the Tian–Lee–Panchenko sense does not rescue the fit.
   Both latent inputs agree. Verdict: the estimator is not why the text pipeline works —
   what that setting has is a similarity-trained embedding plus a cosine ≥ 0.8 donor
   gate, and ridge's own shrinkage confirms no member of our 50-control pool is that
   close. Not in this build: Abadie in-space placebo tests, which would turn
   `gain_over_null` into a permutation p-value for every estimator.
8. **Functional SC on pooled representations (notebooks 07–09) gives the project's first
   properly-inferred null — and no pooled representation beats five numbers.** Running
   Okano & Kurisu's unmodified R on the matched-donor design (each treated site + its own
   5 covariate-matched controls, N=6), FSC holdout error relative to predicting the
   treated site's *own* P01–P08 mean (`ratio_own_mean`), generation-fill / Sentinel-2:
   **chip mean (5 dims) 0.807**, Gram (15) 0.839, quantile (100) 0.958, direct sum (115)
   0.946 — i.e. the 5-number channel mean is the best of them, and the direct sum is worse
   than either of its parts. Sentinel-1 is bimodal: means above 1, but 4–5 of 10 sites
   below it, with site 01 alone (ratio ≈ 2.0–2.6) driving the average. Adding outcome
   dimensions hurts because the same 4 free weights must satisfy more constraints without
   gaining flexibility — Tian–Lee–Panchenko's O(1/√(KT₀)) needs shared factor loadings,
   which a quantile coordinate and a covariance entry do not have.
   **Placebo: across 20 cells (caches × sensors × post periods × arms) nothing reaches
   significance** — strongest is p = 0.070 (Gram, generation-fill S1, P11); the rest
   0.18–1.00, median ranks 2–4.5 against a null median of 3.5.
   Two method findings: their **B-spline path is slightly worse than the no-basis path
   everywhere** (e.g. S2 0.9697 vs 0.9549) — the FSQ lattice makes each quantile function
   a step function with ≤8 jumps, and a 50-knot spline smooths across real jumps, so use
   `FSCM_aug_covmat` on quantile vectors for this encoder; and the reduced quantile grid
   is validated against their full 100-point grid end-to-end (differences ≤0.03).
   **Scope of the null:** this holdout tests prediction of the *next pooled level*, where
   §7 of `Docs/8-27-update.md` already measured that 5 numbers recover chip-level NDVI at
   R² 0.917. The quantile/Gram advantage measured there was in *within-chip dispersion*
   (NDVI sd 0.540 → 0.828), which this holdout does not isolate — so this is not evidence
   that the distributional information is absent.

## Provenance / gates

Data read-only from `../../data/biweekly_datasets/` (paths constructed from metadata,
never from the inventory's absolute path strings) and `../../data/daily_datasets/
selected_site_sample.csv`. Gates in code: encode determinism (exact), chip orientation +
unit ranges, path-vs-inventory match (2,322), SLSQP convergence + Σw=1 on every fit,
ASCM Σw=1 ≤1e-9 and train-MSE ≤ SCM, row-count asserts on every CSV. The biweekly tifs
are channel-first (C,101,101) unlike `finals/` chips (101,101,C) — every reader here
transposes and asserts; anything consuming these files should do the same.
