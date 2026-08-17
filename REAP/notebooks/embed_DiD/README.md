# embed_DiD — the collaborator's equal-weight DiD with embedding inputs
### Report, 2026-08-14

**Question.** Does replacing the 11 hand-extracted band features with learned image
embeddings change the counterfactual analysis — and is the change an improvement?
Design principle throughout: the estimator is the **collaborator's own notebook code**
(`data/scripts/09_test_counterfactual_average.ipynb`), modified only by logged,
declaration-level edits, so that each experiment manipulates exactly one variable.

**Encoders** (survey: `../../Docs/encoder_selection_2026-08-14.md`): S1 → CROMA-base
`SAR_GAP` (reflect-pad 101→104, dB, control-pool mean±2σ normalization); S2 →
TerraMind-v1-base S2L2A 6-band subset (×10,000 to DN, TerraMind's own v1
standardization applied manually, bilinear 101→224); both 768-d, no fusion — every
site keeps a separate S1 and S2 vector. All constants in
`../../data/embeddings/encoding_manifest.json`.

## Folder layout (reorganized 2026-08-15)

The **main folder is the kNN-donor experiment** — the fully embedding-based pipeline
(donor selection AND estimation in embedding space). The **`same_repr/` subfolder is the
ablation arm**: same embedding representation, but the feature-based covariate-matched donors held
fixed. It exists to isolate variables — comparing it to the feature-based pipeline isolates the
*representation* (donors identical); comparing the main experiment to it isolates the
*donor-selection rule* (representation identical). Shared infrastructure, the two
preparation notebooks, and the **tokenizer-space arm** (`04_tokenizer_space_pipeline` — a
separate, self-contained coherent experiment in the decodable tokenizer latent, replacing
the former decode-quality notebook on 2026-08-16) live in the main folder.

## The pipeline (all notebooks executed, zero errors)

Notebooks are numbered in OUR workflow order = execution order (renamed 2026-08-15).
Only `02_DiD_estimator_feature09` carries a `_featureNN` suffix: it is the one notebook that IS
the collaborator's code (the feature-based 09). Everything else is ours.

| notebook | role | what it does | provenance |
|---|---|---|---|
| `00_encode_chips` | preparation | 1,144 chips → 768-band 1×1-px GeoTIFFs mirroring `finals/` | ours |
| `00_solver_verification_vs_R` | preparation | cross-validates our SC solver against R `Synth` (for the future estimated-weight phase) | ours; R env `rsynth` |
| `01_knn_donor_matching` | kNN experiment | donor selection: 10 nearest controls in before-embedding space | ours; keeps only the feature-based 10-donor ranked design + output schema (so 02 consumes it unchanged) |
| `02_DiD_estimator_feature09` | shared | the equal-weight DiD estimator (both experiments execute it) | the feature-based 09, five declaration edits logged in its header |
| `03_DiD_all_pairs` | kNN experiment | 26 sites × 2 sensors, kNN donors; ends with the head-to-head rank comparison vs the original setup | ours (executes the feature-based code per site) |
| `same_repr/03b_run_featureDonors_and_compare` | ablation arm | 26 sites, the feature-based covariate donors | ours (same; moved + re-run 2026-08-15) |
| `04_tokenizer_space_pipeline` | tokenizer-space arm | self-contained coherent experiment: encode → per-sensor kNN → equal-weight DiD → decode the estimator's own counterfactuals, all in ONE latent space | ours; TerraMind tokenizers (replaced the former decode-quality notebook 2026-08-16) |

**Fidelity gates.** The feature-based code + our environment + the ORIGINAL chips reproduces the feature-based
shipped `test/treatment_0001` summary to max |diff| 1.9e-6 (float32 noise). On 1×1
embedding rasters, the feature-based pixel-wise mean equals the direct mean of the 10 embedding
vectors to 1.0e-6 — the feature-based code computes exactly the equal-weight embedding average.

**Solver verification (nb 00, `solver_verification_vs_R.csv`).** Our simplex-QP solver
(`solve_simplex_qp`, scipy SLSQP — our own code, used for all estimated-weight SC) is
cross-validated against the official R `Synth` package (CRAN 1.1.10, `kernlab::ipop`
engine) on 31 problems: planted mixtures, all 26 real 11-band problems, three real
768-d embedding problems. With V fixed uniform so both solve the identical W-problem:
weights agree everywhere to ≤ 2e-3 once ipop is run at tightened precision, and the
objective at our weights is never worse than at the official weights on any of the 111
comparisons. At Synth's *shipped default* precision the official package under-converges
(29 disagreements, its SSE worse by a median 20% — a known issue that motivated the
MSCMT R package), and at tight precision ipop crashes on 13/62 arms; our pipeline never
uses ipop, so neither failure mode affects our results.

## Results

**1. Equal weights waste most of the embedding-space match (`same_repr/03b`).** Pre-period
imbalance ‖e(treated) − mean e(donors)‖ (S1: 5.1, S2: 2.9) is as large as the measured
effect norms (4.5 / 2.6). The band-space analog motivated estimated weights
(`../02_did_vs_synthetic_control.ipynb`, 26/26 pre-fit wins); the same headroom exists
in embedding space.

**2. Embedding-similarity donors tighten the counterfactual at every site (nb 03).**
kNN donors (Euclidean on before-embeddings, per sensor, pool = all 260 controls, with
replacement) overlap the feature-based covariate-matched 10 by only ~1/10 and share the treated
site's NLCD class just 60% of the time — yet cut pre-imbalance by a median 33% (S1) /
38% (S2), improving all 26 sites (Wilcoxon p ≈ 3e-8). Effect norms shrink accordingly
(S1 4.47→3.33, S2 2.58→2.07): part of the covariate-arm "effect" was donor mismatch.

**3. The kNN experiment and the original setup rank sites completely differently (nb 03,
`rank_comparison_original_vs_knn.png`).** Per-site effect-magnitude ranks — the collaborator's
original setup (11 band features + feature-based donors) vs our kNN embedding experiment —
agree at only Spearman ρ = **0.04** (S1, p = 0.84) / **0.08** (S2, p = 0.71): near-zero
agreement on which sites were hit hardest. Changing the analysis changes the answer, so
neither ranking can validate the other by itself; the planned placebo-rank calibration is
the adjudicator (which analysis separates real treated sites from pseudo-treated controls?).

**4. Tokenizer-space arm: a fully coherent encode → match → estimate → decode experiment
(nb 04, replaced 2026-08-16, Jun-directed).** The earlier decode evaluation was rejected
on coherence grounds: it validated latent averaging in a space the estimator never used
(the estimator's pooled 768-d vectors have no decoder). Nb 04 is now a self-contained
pipeline in the **quantized TerraMind tokenizer latent** (5×14×14 = 980 per sensor;
encoder and decoder are the two halves of one autoencoder): per-sensor kNN donors
(nb 01's rule), equal-weight DiD (nb 02's verified algebra), then decoding of the
estimator's **actual** counterfactual latents — encode space = estimate space = decode
space. **Comparison side corrected 2026-08-17 (Jun-directed):** the decoded
counterfactual is compared against **the collaborator's actual notebook-09
counterfactual** — her covariate-matched donors, her image-level procedure (bilinear
resampling to the treatment chip's shape, masked→NaN, pixel-wise nanmean), rebuilt for
all 26 sites and **gated in code** against her shipped `test/treatment_0001` outputs
(max|diff| 1.9e-6; her notebook is a single-site test, so treatment_0001 is the one
site with a shipped file to check against — the gate verifies the implementation all 26
sites go through). This is a **total-pipeline** comparison (donors, representation, and
route all differ at once), not a decode-fidelity isolate. Presentation (Jun-directed,
2026-08-17): the state comparison is the full **26-site × 11-band disagreement grid**,
visualized as **heatmaps** (bias with diverging colors centered at 0; RMSE sequential;
per-sensor color scales) plus a band-metrics figure (Spearman ρ as bars; SD(tokenizer)
vs SD(feature) as grouped bars — ordering and scale; labels use "tokenizer" vs
"feature-based" for the two pipelines throughout); the effect comparison is one
**per-site effect magnitude** (z-scored L2 norm over the sensor's bands, her effects as
reference — notebook 03's convention) compared across the 26 sites per sensor as a
decomposition: ρ = ordering, bias = level, SD vs SD = scale, RMSE = total. Verdict:
**the two pipelines give substantially different answers.** The bias heatmap's S2 half
is single-color columns — constant offsets of ~6–12 signal-SDs (B2/B3/B4 ≈ +0.05
reflectance, NDVI ≈ −0.19, NDWI ≈ +0.19: the decoder floor on top of the donor
difference) — while the S1 half flips sign site by site (VV −0.24…+1.38 dB,
site-specific donor-driven disagreement). State ordering is moderate at best (S1 VV ρ
+0.51, VH +0.46; S2 −0.06…+0.41), and **SD(tokenizer) < SD(feature) on every band** — the
decoded counterfactuals carry roughly half of her cross-site spread on S1 (VV 0.28 vs
0.54 dB) and a quarter to a third on S2's informative bands: the decoder compresses
real site-to-site variation. Effect magnitudes: ordering ρ +0.27 (S1) / −0.13 (S2);
level bias +0.94 / +24.5 (on S2 the per-band offsets survive differencing and
accumulate in the norm, so the magnitude mostly measures the decoder offset); scale
SD(tokenizer) 1.12 vs 0.83 and 2.74 vs 1.24; total RMSE 1.42 / 24.7 — both larger than
the feature-based spread itself. Also notable: the three donor sources largely disagree, and
the tokenizer latent is the outlier (all pairwise overlaps in
`tok_donor_overlap_pairs.csv`; donors are whole sites, out of 10): tokenizer ∩ 768-d
mean 0.38/10 (S1) and 0.31/10 (S2), tokenizer ∩ covariate 0.42 and 0.38 — most treated
sites share zero donors — while 768-d ∩ covariate agree more at 1.35 and 1.00 (max
5/10). S2's tokenizer lists concentrate on 55 distinct donors (max reuse 19), and the
two learned arms' effect-norm rankings agree at |ρ| ≤ 0.24 — representation choice
changes both donors and rankings, reinforcing result 3's need for placebo-rank
adjudication.
Equal weights again leave pre-imbalance ≈ effect norm (9.2/8.7 S1, 7.1/6.9 S2) — the
same estimated-weights headroom as everywhere else. Caveats: the decoder's
reconstruction floor is folded into all decode numbers (per-chip floor pass still
deferred); the S2 tokenizer runs on 6 of its 12 expected bands (rest at pretraining
mean); donor selection per sensor (joint selection is an open design discussion).

## What this means for the synthetic-control goal

SC promises: weights that reproduce the treated site's pre-state, so the post-period
gap is causal. Mapping the results onto its four requirements:

1. **Headroom for estimated weights: confirmed in embedding space** (result 1) — the
   equal-weight pre-imbalance is the quantity a simplex QP on before-embeddings would
   minimize. Estimated weights on embeddings are the pending centerpiece.
2. **Composability — the field's biggest objection to latent SC (LEPA; AlphaEarth
   geometry) — the tokenizer arm decodes the estimator's own counterfactual** (result
   4): encode, match, estimate, and decode all live in one autoencoder latent, so the
   counterfactual can always be inspected as an image. The same-donor decode-fidelity
   check was replaced on 2026-08-17 (Jun-directed) by the pipeline-vs-pipeline
   comparison against the collaborator's notebook-09 counterfactual; what is on record
   now: the two pipelines differ substantially (S1 state disagreement is site-specific,
   VV bias −0.24…+1.38 dB, state ρ ≈ +0.5; S2 carries a constant decoder-floor offset
   of ~6–12 signal-SDs per band; effect-magnitude rank agreement only ρ +0.27 S1 /
   −0.13 S2) — a measurement of pipeline dependence, not a validation of either side.
   For the 768-d arm the composability argument remains indirect (linear probe
   R² 0.80–1.00 + linearity of averaging) — its space has no decoder. S1 is also the
   causally clean sensor (the USGS inventory is S2-NDVI-derived), so the flagship arm
   remains: **S1, latent space, estimated weights**, with the tokenizer latent as the
   representation where the counterfactual can always be inspected as an image.
3. **Donor-pool construction is a first-class lever** (result 2): retrieval in
   embedding space beats covariate matching before any weighting. SC proper should
   drop the fixed-10 structure and run the simplex over all 260 controls — selection
   and weighting in one optimization (the simplex induces sparsity; band-space SC
   found ~1–3 effective donors).
4. **Effects are not representation-invariant** (result 3), so latent SC without an
   adjudication layer is unfalsifiable. The adjudicators are placebo-rank calibration
   (which arm separates real treated sites from pseudo-treated controls?) and the
   planted-mixture battery (which representation recovers known weights and planted
   effects?) — the H2 analog from the clinical preliminary, now with empirical urgency.

Remaining structural gap: with one before-composite we match pre-*states*, not
pre-*trajectories*; the monthly 2023–24 panel re-export is what turns this into
textbook SC.

**Assembled next experiment** (each element justified by a result above): S1
embeddings, donor pool = all 260 controls, simplex QP on before-embeddings (reuse
`preliminary/mimic`'s `solve_simplex_qp`), effects read in embedding space AND through
the decoder, calibrated by placebo ranks against the band-space arm.

## Artifacts

`same_repr/embed_did_effects.csv`, `same_repr/embed_did_site_norms.csv`,
`same_repr/embed_vs_band_did.csv` (covariate ablation arm); `embed_knn_effects.csv`,
`embed_knn_site_norms.csv` (kNN experiment, main folder);
`rank_comparison_original_vs_knn.png` (nb 03); `tok_knn_site_matching.csv`,
`tok_donor_overlap_pairs.csv`, `tok_effects_site_norms.csv`, `tok_decode_features.csv`,
`tok_state_bias_grid.csv`, `tok_state_rmse_grid.csv`, `tok_effect_magnitudes.csv`,
`tok_state_bias_heatmap.png`, `tok_state_rmse_heatmap.png`,
`tok_state_band_metrics.png`,
`tok_decode_effect_comparison_<site>.png` (tokenizer-space arm, nb 04);
embeddings + manifests under `../../data/embeddings*/` and `../../data/embeddings_tok/`
(gitignored). Environment: conda env `satellite` (`/data/wang/junh/envs/satellite`).
