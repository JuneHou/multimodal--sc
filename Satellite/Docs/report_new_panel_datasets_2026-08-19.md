# Report: the new time-series satellite datasets and what they change for us

**Date:** 2026-08-19
**Inputs reviewed:** `Satellite/data/{daily,weekly,biweekly}_datasets/`, `Satellite/data/quality_comparison/`,
scripts `Satellite/data/scripts/10–13_*.ipynb`, the collaborator's two emails, and the advisor's
evaluation proposal. Every number below is read from the named file on this disk; claims
that exist only in the emails are marked *(reported — not verifiable locally)*.

---

## 1. Summary

The collaborator turned the study from a **two-snapshot** design (one before / one after
median image) into a **20-period panel**: 10 pre-hurricane and 10 post-hurricane biweekly
composites (May 10 2024 – Feb 13 2025, boundary Sept 27 2024) for 10 treatment sites and
50 counterfactual sites. This is the missing ingredient for a real evaluation: for the
first time, donor quality and pipeline quality can be scored against **actually observed
images** (held-out pre-hurricane periods) instead of only against another pipeline's
output. The biweekly version is complete on disk and is the one to use. The SCM/ASCM
validation notebooks (14/15) arrived on 2026-08-19 and are analyzed in §4: S1 validates
well (one degenerate site), S2 is much harder, the email's "sites 4 and 5" failures are
S2-only, and much of the apparent donor-quality problem may actually be *image*-quality
(no cloud filtering) — their output files (including the never-inspected post-hurricane
effects) still need to be shared. The advisor's expanding-window scheme (fit P01–P08 →
predict P09; fit P01–P09 → predict P10) is a stricter refinement of the validation
notebook 14 ran, and both apply directly to our tokenizer pipeline.

---

## 2. What was added

### 2.1 The sample (`data/daily_datasets/selected_site_sample.csv`, 60 rows)

- **10 treatment sites** = `treatment_0001…0010`, the *first 10* of our existing 26
  (selected in `scripts/10_download_daily_satellite_data.ipynb` as the head of
  `treatment_sites.geojson`).
- **50 counterfactual sites** = for each treatment, the **top-5 by `control_rank` from the
  SAME covariate matching table we already use** (`data/finals/site_matching_table.csv`,
  NLCD class + elevation + slope + distance). So the donor pool is a subset of the existing
  260 — no new matching was done. New site-id convention: `counterfactual_{treatment}_{rank}`.

### 2.2 Three temporal resolutions (same 11 bands, same 101×101 px / 10 m / EPSG:32617 chips)

| dataset | unit | files | status |
|---|---|---|---|
| `daily_datasets/` (606 MB) | one file per acquisition date | 5,485 tif | complete for the 60-site sample |
| `weekly_datasets/` (366 MB) | 7-day pixel-wise nanmedian | 3,374 tif | **incomplete** (see §6) |
| `biweekly_datasets/` (285 MB) | 14-day pixel-wise nanmedian | 2,322 tif | **complete**: 2,322 created + 78 site-periods with zero acquisitions = 2,400 (`biweekly_build_action_summary.csv`) |

Folder layout everywhere: `{sensor}/{treatment|counterfactual}/{before|after}/`, with the
site, period id, and date window encoded in the filename
(e.g. `treatment_0001_after_P10_2025-01-31_2025-02-13_biweekly_sentinel2.tif`).

### 2.3 The panel clock (`biweekly_datasets/biweekly_period_definitions.csv`)

The study window is the same for all three datasets — 140 days before (2024-05-10 to
2024-09-26) and 140 days after (2024-09-27, the Helene reference date, to 2025-02-13) —
but the "10 before + 10 after" structure belongs to the **biweekly** version only:

- **biweekly**: 10 + 10 periods of 14 days (`before_P01…P10`, `after_P01…P10`). In the
  second email's notation, before_P01–P10 = P01–P10 and after_P01–P10 = P11–P20.
- **weekly**: 20 + 20 periods of 7 days (`before_W01…W20`, `after_W01…W20`;
  `weekly_datasets/weekly_period_definitions.csv`, 40 rows).
- **daily**: no fixed periods at all — one file per actual acquisition date. A site is
  imaged on only ~11–13% of calendar days, i.e. roughly 16–18 images per site per half
  (`daily_datasets/daily_dataset_summary.csv`), at irregular dates.

### 2.4 How the data was made (scripts 10–12)

- **Notebook 10 (download):** Google Earth Engine. S2 = `COPERNICUS/S2_SR_HARMONIZED`,
  scenes with ≤80% cloud, SCL mask removing cloud shadow / medium & high-probability cloud /
  cirrus / snow; bands B2,B3,B4,B8,B11,B12 (×0.0001) plus NDVI and NDWI computed per pixel.
  S1 = `COPERNICUS/S1_GRD`, IW mode, ascending, VV/VH plus VV−VH. "Daily" = one mosaic per
  acquisition date, no compositing.
- **Notebooks 11/12 (weekly/biweekly):** pixel-wise `np.nanmedian` across the daily scenes
  falling in each window; a period with zero acquisitions gets no file.
- **Valid pixel** = at least one band is finite at that location; the fraction of valid
  pixels is the quality score used throughout the metadata CSVs.

### 2.5 Image quality on disk (verified numbers)

From `biweekly_datasets/biweekly_quality_summary.csv`:

| sensor / group | mean valid % (before → after) | periods with data |
|---|---|---|
| S1 treatment | 99.9 → 99.9 | 100% → 94% |
| S1 counterfactual | 99.9 → 99.9 | 97.6% → 93% |
| S2 treatment | 57.8 → 56.8 | 100% → 96% |
| S2 counterfactual | 68.7 → 59.8 | 100% → 95.8% |

These match the first email's 58/57 and 69/60. Nanmedian compositing buys S2 an extra
**+11.6 to +19.9 percentage points** of valid pixels vs the average source scene
(`mean_improvement_vs_mean_source`), and ~0 for S1 (already clean).

Two DIFFERENT kinds of "missing" must not be confused:

1. **Missing pixels inside an image** ("valid pixel fraction"). A pixel is valid if at
   least one band has a usable value there. S2 pixels go missing because the SCL cloud
   mask deletes cloud shadow, medium/high-probability cloud, cirrus, and snow (plus
   partial-swath edges); a masked pixel is NaN. The biweekly nanmedian recovers a pixel
   only if it was clean in at least one acquisition during the 14 days — a pixel cloudy
   in every acquisition of the window stays NaN. That is why S2 biweekly images still
   average only ~57–69% valid. S1 is radar and sees through clouds, so its images are
   ~99.9% valid.
2. **Missing periods** (a site-period with zero acquisitions → no image file at all).
   This is S1's weak spot (no acquisition scheduled in some 14-day windows) and affects
   S2 mildly after the hurricane. Per site, out of 10 expected periods per half
   (`biweekly_datasets/biweekly_site_dimension.csv`, all 60 sites):

   | sensor / half | sites with 10/10 | 9/10 | 8/10 | 6/10 |
   |---|---|---|---|---|
   | S2 before | 60 | — | — | — |
   | S2 after | 35 | 25 | — | — |
   | S1 before | 57 | — | — | 3 |
   | S1 after | 38 | 3 | 19 | — |

   So the pre-period is complete for S2 at every site, but a fifth of S1 site-halves
   have 8 of 10 after-periods, and three sites have only 6 of 10 S1 before-periods —
   any pre-period validation split must handle these gaps explicitly.

Per `daily_datasets/daily_dataset_summary.csv`, a site gets an image on only ~11–13% of
calendar days (both sensors), and daily S2 scenes average 41–52% valid pixels (median as
low as 28% for treatment/before). The email's "about 25%" is consistent with the median
before-period figure rather than the mean. Either way the conclusion is the same: daily S2
is too sparse for encode/decode; **biweekly is the right resolution for our pipeline**,
exactly as the collaborator suggests.

### 2.6 Rich metadata worth knowing about

`biweekly_datasets/biweekly_image_quality.csv` (2,400 rows = 60 sites × 20 periods × 2
sensors) has per-image everything: per-band valid fractions, acquisition dates,
source-scene provenance, quality labels. `daily_datasets/daily_calendar_index.csv` is a
dense 16,800-row day×site grid with days-relative-to-Helene. These can drive any
missing-data handling we design without touching the imagery. (Caveat: their
`file_path`/`folder` columns store the collaborator's absolute Dropbox paths — use only
the tabular columns, or map filenames, never those path strings.)

---

## 3. What changed vs our current progress

| | current work (notebooks 01–04) | new panel |
|---|---|---|
| treated sites | 26 | 10 (the first 10 of the same 26) |
| donor pool | 260 controls (10 per treated) | 50 controls (top-5 per treated, same matching table) |
| time | 2 snapshots (before/after median) | 20 biweekly periods (10 pre + 10 post) |
| counterfactual weighting | equal-weight mean of matched/kNN donors (DiD) | supports estimated SC weights fit on pre-periods |
| evaluation | pipeline-vs-pipeline only (notebook 04: tokenizer vs feature-based, no ground truth) | held-out pre-periods = **observed ground truth** |
| S2 missing pixels | mild (median composites over long windows) | severe: ~40% invalid per biweekly image on average |

What carries over unchanged: the 11 bands, chip geometry (101×101, 10 m, EPSG:32617 —
our 224-resize encode prep applies as-is), the covariate matching table, and the whole
encode/decode stack. What does **not** carry over: results are not directly comparable
across the two designs (different site count, different period definition), and the
missing-pixel problem is now first-order instead of negligible.

**kNN cannot search the 260-control snapshot pool on this panel.** `embed_DiD` could,
because all 260 had before/after chips. Distance needs `encode(control image)`; of the
matching table's 260 controls, **210 have no P01–P20 GeoTIFF** (ranks 6–10 of treatments
`0001`–`0010`, and every control of `0011`–`0026`). The tokenizer panel latents therefore
cover 60 sites, and latent-space kNN in `notebooks/ts_SCM_ASCM/` searches those 50
controls — every site that exists in `biweekly_datasets/` — not a subset of a larger
encoded pool. Repeating the 260-site search would be new data collection. (Within the
50, that arm already *shares* the pool across treated sites, unlike notebook 14's
per-site 5 covariate matches.)

---

## 4. The SCM/ASCM validation — notebooks 14 and 15 (received 2026-08-19, analyzed)

Both notebooks are now in `data/scripts/` with fully stored outputs, so everything below
is read from the code and its embedded results, not from the email. Their **output CSV/
figure files** still are not local (all paths point to the collaborator's Dropbox
`test/scm_validation/` and `test/ascm_validation/` folders), but the headline numbers are
recoverable from the notebooks themselves.

To be clear about which script does what: notebooks 10–12 build the data, and notebook 13
only **compares image quality** across the daily/weekly/biweekly versions — it contains no
causal analysis. The validation move lives entirely in notebooks 14 and 15.

**Design at a glance — time split and donor pool:**

| | periods | role |
|---|---|---|
| fit ("training") | before_P01–P08 (2024-05-10 … 2024-09-12) | donor weights estimated here, then frozen |
| validation | before_P09–P10 (2024-09-13 … 2024-09-26) | still pre-hurricane → predictions compared with **actually observed** treated images |
| effects | after_P01–P10 = "P11–P20" (2024-09-27 … 2025-02-13) | frozen weights give the counterfactual; gap = hurricane effect |

**Donor pool:** each treated site uses **only its own 5 covariate-matched counterfactuals**
(the top-5 by `control_rank` from `finals/site_matching_table.csv` — NLCD land-cover class
+ elevation + slope + distance matching). The 50-control pool is never shared across
treated sites, and no donors are selected from the data itself.

### 4.1 What notebook 14 actually does (verified)

- **Feature panel built inline** from the biweekly tifs (driven by
  `biweekly_image_quality.csv`, 2,322 usable images): per site × sensor × period,
  nan-aware chip mean per band (masked read → NaN → mean over finite pixels). All 11
  bands, both sensors. So the outcome is exactly the chip-mean-feature convention we
  already use — good news for comparability.
- **Donor pool = only the treated site's own 5 matched counterfactuals** (sorted by
  `control_rank`); the 50-control pool is never shared across sites.
- **SCM — a fit, not model training.** No model is trained in the machine-learning sense;
  "training periods" only means "the periods the weights were fitted on". Per treated
  site × sensor, the entire "model" is 5 numbers, found by a small constrained
  least-squares optimization:
  - **find** w₁…w₅ (one weight per matched counterfactual site),
  - **subject to** each wⱼ ≥ 0 and w₁+…+w₅ = 1 (a weighted average — no extrapolation),
  - **minimizing** the mean squared difference between the treated site's z-scored band
    means and the weighted mix of the donors' band means, over periods P01–P08 × all
    bands of that sensor (fit jointly: 5 parameters against 24 data points for S1,
    up to 64 for S2).

  Features are z-scored by the pooled all-site pre-period mean/SD; the optimizer is scipy
  SLSQP (no intercept). Once solved, "prediction" is just the same weighted average of the
  donors in a new period. Weights frozen; P09–P10 predicted for validation; P11–P20 gaps
  computed as effects and written out — but **the effects are never printed, plotted, or
  examined** anywhere in the notebook.
- **How the evaluation works.** RMSE always compares the synthetic prediction with the
  **actually observed** treated values: error = (observed − weighted donor mix), in
  z-scored units, so RMSE is in "training-SD units" — 1.0 = the typical cross-site spread
  of a band in the pre-period, and values well below 1 mean the synthetic control is
  genuinely informative. Per site × sensor, errors are pooled across bands: the
  *training RMSE* covers P01–P08 (in-sample — the weights were chosen to minimize exactly
  these errors, so it is optimistically small by construction) and the *validation RMSE*
  covers P09–P10 with frozen weights (the honest out-of-sample number). A site is flagged
  by the **ratio** validation/training RMSE (≤1.5 good, ≤2 caution, else poor): the ratio
  uses each site's own training fit as its personal baseline, so it asks "did the
  pre-period agreement generalize, or was it overfitting?" — see §4.4 for what the ratio
  cannot catch.

### 4.2 Notebook 14 results (from stored outputs)

Validation RMSE (P09–P10, in training SDs), per site:

| site | S1 | S2 | | site | S1 | S2 |
|---|---|---|---|---|---|---|
| 0001 | 0.44 | 0.66 | | 0006 | 0.43 | 0.52 |
| 0002 | 0.54 | 0.53 | | 0007 | **0.72 (poor)** | 0.45 |
| 0003 | 0.12 | 0.42 | | 0008 | 0.19 | 0.52 |
| 0004 | 0.14 | **1.68 (poor)** | | 0009 | 0.18 | 0.42 |
| 0005 | 0.19 | **1.17** | | 0010 | 0.07 | 0.29 |

- **S1 validates well** (sensor-mean per-band RMSE 0.18–0.42 SD) — with one failure the
  email did not mention: **site 7**, whose S1 fit trains on only 4 of 8 periods (missing
  acquisitions) and collapses to a degenerate single-donor solution (weight 1.0), ratio 2.2.
- **S2 is clearly harder** (sensor-mean per-band RMSE 0.43–0.78 SD). The email's "sites 4
  and 5 do poorly" is an **S2-only** statement — on S1 those two sites are among the best.
  And note a flag quirk: site 5's S2 fit is bad in training AND validation (1.27 → 1.17 SD),
  so its ratio looks fine and it is flagged "good"; the ratio-based flag misses
  level failures.

### 4.3 What notebook 15 adds (verified)

Ridge-augmented SCM (own implementation of Ben-Michael, Feller & Rothstein 2021, closed
form, negative weights allowed, no intercept), with the ridge penalty chosen by
leave-one-pre-period-out cross-validation on P01–P08 only. Same split, same joint-band
fitting. Two important prints: **9 of 20 fits pick the maximum penalty** (= no
augmentation; ASCM identical to SCM there), and standardization differs from notebook 14
(per-cluster donor SDs instead of pooled), so **RMSE numbers are not comparable across
the two notebooks**.

SCM vs ASCM on the held-out P09–P10 (notebook 15's own scale):

- **S1: ASCM helps where it acts** — site 1 improves 0.53 → 0.30 SD, site 2 1.00 → 0.90,
  site 6 0.53 → 0.49; all three S1 bands improve on average.
- **S2: a wash or worse** — 7 of 8 bands get worse on average; the single biggest change
  is site 5 **degrading** 1.11 → 1.70 SD, driven by heavy extrapolation (total negative
  weight 1.35). Matches the email's caution exactly, with the mechanism visible: when
  the donors can't span the treated site, ridge extrapolation overfits the pre-period
  and fails out of sample.

### 4.4 Caveats that matter before building on these notebooks

1. **No image-quality filtering.** Partially cloudy S2 composites enter at full weight;
   of the 480 S2 pre-period site-images, only ~half are "excellent" and 67 are fully
   unusable. The two failing S2 sites are exactly where validation-period images are
   thin (site 4's donor at 12% valid pixels in P09; site 5's treated image at 39%).
   Much of the "donor quality" signal may actually be *image* quality — filtering on
   `valid_pixel_fraction` should be tested before concluding donors are bad.
2. **The ratio-based flag is not sufficient on its own.**
   - It only catches *degradation* (overfitting), not *level* failures: a fit that is
     equally bad in training and validation gets ratio ≈ 1 and a "good" flag. Site 5's S2
     is the live example — training RMSE 1.27, validation 1.17, ratio 0.92, flagged
     "good" even though both errors exceed one full SD. A complete evaluation needs both
     checks: ratio ≈ 1 (no overfitting) **and** holdout RMSPE well below the mean
     baseline of 1 (the fit is actually informative). After P01–P08 z-scoring, RMSPE of
     the pooled mean is 1 by construction.
   - The ratio is a noisy statistic: its numerator comes from only 2 held-out periods, so
     a lucky/unlucky pair moves it a lot (site 10's S1 ratio is 0.19; site 7's 2.22 rests
     partly on a tiny 4-period training fit). The advisor's expanding-window scheme gives
     somewhat better-founded validation points, but the 2-period limit is inherent to
     having 10 pre-periods.
3. **Missing periods silently shrink the training set** — some S2 fits use as few as 4 of
   8 pre-periods (32 observations for 5 weights); S1 site 7's failure is this.
4. `VV_minus_VH` is a linear combination of VV and VH but enters the joint objective as a
   third block, effectively double-weighting the radar ratio; it is consistently the
   worst-fit S1 band.
5. Post-hurricane effect estimates (P11–P20) exist only in the unshipped output CSVs and
   were never inspected in either notebook; also periods P12 and P15 are missing for a
   third of sites (no acquisitions), which nothing in the notebooks flags.

### 4.5 The proposal for our pipeline (second email)

Select donors / estimate weights from P01–P08 **in embedding space**, freeze, construct
and decode the synthetic images for P09–P10, and compare decoded vs observed treated
images. This tests the whole encode → weight → decode chain against ground truth — a much
stronger evaluation than notebook 04's comparison against the feature-based pipeline. And
since notebook 14's outcome is the same nan-aware chip-mean feature we use, the two
pipelines can now be scored on the *same* held-out truth with the *same* metric.

### 4.6 The P11–P20 effects — pushed 2026-08-20, now inspected

The collaborator's GitHub push (commit `9bbc182`, merged) delivered the previously
unshipped output folders at the repo root: `test/scm_validation/` and
`test/ascm_validation/`, including `05_post_hurricane_effects.csv` (SCM, native + her
standardized units; effect = actual − synthetic) and `08_post_hurricane_effects.csv`
(ASCM vs SCM standardized gaps). This closes §4.4's item 5. What the effects say:

- **No band shows a mean post-hurricane effect above the held-out noise.** Mean effect
  across sites × periods, in her scaler's SD units, divided by the same pipeline's own
  P09/P10 validation RMSE, lands between 0.03 and 0.25 for every band. In native units:
  NDVI −0.002 (SD 0.053), NDWI −0.008, B8 −0.006 reflectance; S1 VH +0.035 dB
  (SD 0.53), VV +0.022 dB.
- **No consistent direction either.** Per-site mean NDVI effects span −0.058
  (treatment_0002) to +0.058 (treatment_0001); per-period means drift from slightly
  positive (P11–P14) to slightly negative (P17–P18) with nothing resembling a step at
  the hurricane boundary.
- **Coverage is thin where it matters:** only 66 of 100 S2 site-periods are estimable
  (two post periods drop out at all 10 sites, more partially — her rule discards a
  period when the treated site or any donor is missing); S1 has 91/100. And her own
  validation flags (sites 4/5 fail S2) mean a fifth of the S2 effect rows sit on fits
  that were never trustworthy.
- **ASCM tracks SCM on average** (mean |gap difference| 0.17 SD) **but individual
  site-periods diverge by up to 5.8 SD** — the extrapolation instability of §4.3
  showing up directly in the effect estimates.
- **Cross-check against our decoded effects (ts_SCM_ASCM notebook 04):** our means were
  NDVI −0.037, NDWI −0.041, B8 −0.040, VH +0.26 dB — a mild vegetation-loss direction,
  also below its own noise. The feature-based arm does not corroborate that direction:
  its means are ~10× smaller and mixed-sign. The joint, honest reading is that
  **neither pipeline detects a post-hurricane effect above its validation noise on this
  panel**, and any cross-site mean direction quoted from either arm alone should be
  treated as unconfirmed.

---

## 5. The advisor's evaluation proposal

> P01–P08: estimate weights, fix, predict P09.
> P01–P09: estimate weights, fix, predict P10.

This is an **expanding-window, one-step-ahead** validation, versus the collaborator's
single fit on P01–P08 predicting P09–P10 jointly. The two are compatible — same idea,
different splits — but the advisor's version is the stricter and cleaner one:

- each prediction uses **all** information available up to that point (the way the
  post-period estimator will actually be used);
- it yields two separately-fitted checks instead of one fit scored twice, so a lucky/
  unlucky P01–P08 fit doesn't decide both validation points;
- it is the standard backdating/placebo-in-time pattern from the SC literature.

The cost is trivial here (one extra fit). Recommendation: run **both** — the collaborator's
frozen-weight version for direct comparability with the reported notebook-14 numbers, and the
advisor's expanding version as the headline validation.

---

## 6. Gaps and missing files

**Resolved 2026-08-19 afternoon:** notebooks 14 and 15 arrived in `data/scripts/` with
their stored outputs, and §4 above is now based on them directly. (An earlier sync had
refreshed `data/test/treatment_0001/` with the notebook-09 counterfactual-average TEST
outputs instead — values byte-identical to what notebook 04's gate already verified, plus
an accidental nested duplicate folder.) The feature-table question is also resolved: both
notebooks build their feature panel inline from the biweekly tifs.

**Resolved 2026-08-20 morning (her GitHub push, commit `9bbc182`, merged):** both
remaining items arrived at the repo root —
- the notebook-14/15 **output folders** `test/scm_validation/` and
  `test/ascm_validation/` (weights, per-period predictions, the 81-point penalty curves,
  and the **post-hurricane effect CSVs**, analyzed in §4.6), plus `test/treatment_0001/`
  rasters; and
- the **updated README** as Quarto source + rendered HTML
  (`README_Hurricane_Helene_Satellite_Dataset.qmd`/`.html`, 4 tabs: Version 1,
  Version 2, SCM Validation, ASCM Validation).

Her push also made the repo-root `scripts/` the canonical copy of notebooks 00–15
(the duplicate tracked copies under `Satellite/data/scripts/` were untracked in favor of
hers on Jun's instruction; the local Dropbox copies stay on disk unchanged — note her
pushed 11/12/13 differ from the older Dropbox export).

**Incomplete deliverables (facts on disk; the imagery is used as-is per our data rules):**
- `weekly_datasets/` is an aborted build: notebook 11's stored log stops at combination
  73/120; only 2 of its 9 metadata CSVs exist (`weekly_period_definitions.csv`,
  `weekly_sample_validation.csv` — and the latter reports the intended roster, not what's
  on disk); 4 of the 50 controls (`counterfactual_0010_02…05`) have zero weekly files.
  Since the biweekly dataset is the recommended one and is complete, this only matters if
  we ever want weekly resolution — then the fix is the collaborator's to make (or, for the *missing
  metadata CSVs only*, a csv-only regeneration could be arranged with approval).
- `quality_comparison/` covers just two sites (`treatment_0001`, `counterfactual_0001_01`) —
  a pilot, though notebook 13 is written for the full 60-site sample. Do not quote
  `03_overall_dataset_comparison.csv` as a dataset-wide statement.
- `daily_datasets/` also holds leftover files from an earlier wider pass (26 treatments,
  85 controls). Always select through `selected_site_sample.csv`; never glob the folders.

**None of this blocks the main path:** the biweekly imagery + metadata, which is what both
the feature-based validation design and ours need, is complete.

---

## 7. Does this solve the evaluation problem?

**Solved:** the part that was fundamentally missing. Notebook 04 could only measure how
much two pipelines disagree — it could not say which is right, because nothing observable
played the role of truth. With 10 pre-hurricane periods, held-out pre-period prediction
scores any donor rule (covariate matching, 768-d kNN, tokenizer-latent kNN) and any
weighting (equal, SCM, ASCM) against **images that were actually observed**. The reported
site-4/site-5 result is exactly the kind of finding this unlocks. It also gives 10 post
periods for effect trajectories and much stronger placebo calibration.

**Still not solved:**
- post-hurricane counterfactual truth (P11–P20 remains unobservable by construction —
  validation quality is evidence, not proof, that post-period counterfactuals are right);
- the decoder reconstruction floor: if decoded images sit far from observed images even
  for a *perfect* latent, validation RMSE conflates donor error with decode error. The
  clean control: encode each observed treated P09/P10 image and decode it straight back —
  that reconstruction error is the floor no donor method can beat, and validation scores
  should be read relative to it.

**New problems the panel introduces:**
- **Missing pixels are now first-order for S2 (~40% invalid on average).** The two
  pipelines currently treat them differently: the feature-based route ignores invalid
  pixels (nan-aware statistics), while our encode prep fills NaN before the tokenizer.
  At 40% invalid, the filled content materially shapes the latent. This must be settled
  as an explicit design decision before any encoding of panel data.
- **S1 has missing periods** (whole site-periods with no acquisition) — panel methods need
  a stance on gaps in time.
- **Smaller, different sample:** 10 treated / 50 controls with panel imagery (not 26 /
  260). Notebook 14 used only each site's own 5 covariate matches; the tokenizer arm
  (`ts_SCM_ASCM`) already searches all 50 then keeps 5 neighbors. The remaining gap vs
  `embed_DiD`'s 260-control kNN is missing P01–P20 chips for 210 sites, not a search
  filter. Whether those 210 would yield closer latent neighbors is untestable until they
  are downloaded and encoded.

---

## 8. Proposed next steps (each needs approval before work starts)

1. **Request from the collaborator:** the notebook-14/15 output folders (above all the
   post-hurricane effect CSVs, which exist nowhere else) and the updated README.
   (Optionally mention the weekly build looks unfinished, in case weekly resolution is
   meant to be usable.)
2. **Build the biweekly feature panel** (site × sensor × period chip-mean features from
   `biweekly_datasets/`, nan-aware — now verifiably the SAME convention notebook 14 uses,
   so it doubles as a reproduction of notebook 14's `01_extracted_features` panel) —
   the shared input for reproducing the notebook-14 validation and for scoring our
   pipeline in feature units.
3. **Design the pipeline holdout validation** as a written plan before any code: donors +
   weights from pre-periods in tokenizer-latent space; predict P09/P10 under both splits
   (frozen P01–P08 and the advisor's expanding window); score decoded-vs-observed in image
   space, feature space, and latent space; include the encode→decode reconstruction floor
   as the reference line; placebo runs on controls.
4. **Test the image-quality explanation of the S2 failures** (§4.4): rerun the notebook-14
   validation with a `valid_pixel_fraction` threshold (e.g. ≥0.4 / ≥0.6) and see whether
   sites 4/5 recover — this separates "bad donors" from "bad images" and directly informs
   whether our pipeline needs a quality filter too.
5. **Open design decisions to settle first:**
   - NaN handling for encoding at ~40% invalid S2 pixels (current chip-mean fill vs
     alternatives) — scientific implications: what the latent represents;
   - image-quality filtering: whether both pipelines should exclude or down-weight
     site-periods below a valid-pixel threshold (notebooks 14/15 currently use none);
   - per-sensor vs joint donor selection (still open from the 26-site work);
   - donor-pool scope: notebook 14's 5 matched donors corner easily. The tokenizer
     arm already searches all 50 panel controls. Repeating `embed_DiD`'s 260-control
     kNN is blocked until ranks 6–10 and treatments 0011–0026 have P01–P20 imagery;
     keeping more than 5 neighbors from the existing 50 is a separate, cheaper knob;
   - whether the hold on estimated-weight SC is lifted now that the advisor has replied,
     and which weighting (equal-weight kNN mean vs SCM vs ASCM) the validation should cover.

---

## Provenance

Site sample: `data/daily_datasets/selected_site_sample.csv`; matching:
`data/finals/site_matching_table.csv`; periods:
`data/biweekly_datasets/biweekly_period_definitions.csv`; biweekly quality/coverage:
`biweekly_quality_summary.csv`, `biweekly_build_action_summary.csv`,
`biweekly_image_quality.csv`; daily stats: `data/daily_datasets/daily_dataset_summary.csv`,
`daily_download_action_summary.csv`; weekly state: directory listing +
`weekly_sample_validation.csv` vs on-disk files + notebook 11's stored run log;
quality-comparison scope: `data/quality_comparison/02_site_dimension.csv` site_id column;
collection/masking details: `data/scripts/10_download_daily_satellite_data.ipynb`;
compositing: `data/scripts/11/12_*.ipynb`; SCM/ASCM methods and all §4 numbers: the code
and stored outputs of `data/scripts/14_test_scm_validation.ipynb` (site summary, per-band
RMSE tables) and `data/scripts/15_test_augmented_synthetic_control.ipynb` (penalty
selection, SCM-vs-ASCM comparison, weight diagnostics). §4.6 numbers: the repo-root
`test/scm_validation/05_post_hurricane_effects.csv`,
`08_feature_validation_rmse.csv` (standardized units), and
`test/ascm_validation/08_post_hurricane_effects.csv` from her push `9bbc182`.
Email-only claims are marked as reported.
