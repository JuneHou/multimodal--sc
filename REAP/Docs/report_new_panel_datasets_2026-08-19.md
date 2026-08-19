# Report: the new time-series satellite datasets and what they change for us

**Date:** 2026-08-19
**Inputs reviewed:** `REAP/data/{daily,weekly,biweekly}_datasets/`, `REAP/data/quality_comparison/`,
scripts `REAP/data/scripts/10–13_*.ipynb`, the collaborator's two emails, and the advisor's
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
output. The biweekly version is complete on disk and is the one to use. The collaborator's
SCM/ASCM validation notebooks (14/15) and their results are described in the second email but are
not in our checkout — they need to be requested. The advisor's expanding-window scheme
(fit P01–P08 → predict P09; fit P01–P09 → predict P10) is a stricter refinement of the
validation the collaborator already ran, and both apply directly to our tokenizer pipeline.

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

20 contiguous 14-day periods: `before_P01` (2024-05-10) … `before_P10` (ends 2024-09-26),
`after_P01` (starts 2024-09-27, the Helene reference date) … `after_P10` (ends 2025-02-13).
In the second email's notation, before_P01–P10 = P01–P10 and after_P01–P10 = P11–P20.

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
(`mean_improvement_vs_mean_source`), and ~0 for S1 (already clean). Note S1's caveat is
the opposite of S2's: pixels are always valid, but **6–35 of 500 site-periods have no S1
acquisition at all** — the panel has holes in time, not in space.

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

---

## 4. The collaborator's SCM/ASCM validation (second email)

*(reported — not verifiable locally: `scripts/14_test_scm_validation.ipynb`,
`scripts/15_test_augmented_synthetic_control.ipynb`, and their entire output folder —
weights, feature-level RMSEs, validation plots, post-hurricane effects — are not in this
checkout.)*

The workflow: estimate SC weights on P01–P08; freeze them; predict the treated sites in
P09–P10 (still pre-hurricane, so predictions can be compared with actual untreated
observations); then apply the frozen weights to P11–P20 for effect estimation. Reported
findings: S1 validates well (RMSE generally below one training SD); S2 is harder but most
average validation RMSEs still below one SD; **treatment sites 4 and 5 validate poorly**
(their 5-donor pools may not reproduce their pre-trends); ASCM helps some S1 cases but is
not consistently better than SCM for S2 — the email reads this as donor quality being the issue.

The same email proposes this design for our pipeline: select donors / estimate weights from
P01–P08 **in embedding space**, freeze, construct and decode the synthetic images for
P09–P10, and compare the decoded images directly with the observed treated images. This
tests the whole encode → weight → decode chain against ground truth — a much stronger
evaluation than notebook 04's comparison against the feature-based pipeline.

Note this validation must run on a per-site-per-period **feature table** (chip means per
band, presumably), and that table was not shipped either — scripts 10–13 produce imagery
and quality metadata only.

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

**Must come from the collaborator (cannot be derived locally):**
- `scripts/14_test_scm_validation.ipynb`, `scripts/15_test_augmented_synthetic_control.ipynb`,
  and their full output folder (weights, feature-level RMSE, validation plots, effects).
  Without these, none of §4's numbers can be checked or built on.
- The per-site-per-period feature table those notebooks consume (or its generating script).
- The updated README the first email mentions — the local
  `data/README_Hurricane_Helene_Satellite_Dataset.md` is dated Aug 11 and does not mention
  any of the new datasets.

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
- **Smaller, different sample:** 10 treated / 5 donors each. Only 5 matched donors per
  site is thin for SCM-style weighting (the reported sites-4/5 result already hints at this); the
  50-control pool could be shared across treated sites, but that changes the design and
  is a decision, not a default.

---

## 8. Proposed next steps (each needs approval before work starts)

1. **Request from the collaborator:** notebooks 14/15 + their complete output folder, the
   feature-table script/CSV they consume, and the updated README. (Optionally mention the
   weekly build looks unfinished, in case weekly resolution is meant to be usable.)
2. **Build the biweekly feature panel** (site × sensor × period chip-mean features from
   `biweekly_datasets/`, nan-aware, matching the collaborator's extraction convention) —
   the shared input for reproducing the notebook-14 validation and for scoring our pipeline in feature
   units.
3. **Design the pipeline holdout validation** as a written plan before any code: donors +
   weights from pre-periods in tokenizer-latent space; predict P09/P10 under both splits
   (frozen P01–P08 and the advisor's expanding window); score decoded-vs-observed in image
   space, feature space, and latent space; include the encode→decode reconstruction floor
   as the reference line; placebo runs on controls.
4. **Open design decisions to settle first:**
   - NaN handling for encoding at ~40% invalid S2 pixels (current chip-mean fill vs
     alternatives) — scientific implications: what the latent represents;
   - per-sensor vs joint donor selection (still open from the 26-site work);
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
compositing: `data/scripts/11/12_*.ipynb`. Email-only claims are marked as reported.
