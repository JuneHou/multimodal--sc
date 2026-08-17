# Choosing an image encoder for embedding-space synthetic control
### Hurricane Helene satellite phase — encoder survey and recommendation, 2026-08-14

**Purpose.** Replace the 11 chip-mean band features in `02_did_vs_synthetic_control.ipynb`
with learned image embeddings, keeping the same estimator (simplex QP on standardized
before-features, 26 treated × 10 donors). This memo answers: *which encoder preserves the
most usable information for that estimator?* It synthesizes three parallel literature
surveys (optical encoders; SAR/multimodal encoders; causal-inference precedents and
evaluation methodology), run 2026-08-14. Jun runs the experiment; the collaborator is
separately trying the spectral (band-mean) features.

**Companion documents:** `tutorial_satellite_causal_2026-08-12.md` (design background),
`notebooks/01_dataset_analysis.ipynb` (dataset facts cited below),
`notebooks/02_did_vs_synthetic_control.ipynb` (the band-space experiment this extends).

---

## 0. TL;DR — the recommendation

Run **four encoders + two baselines**, all under ONE fixed preprocessing convention,
and let a validation gate (§4) pick the winner:

| Track | 1st choice | 2nd choice | Why |
|---|---|---|---|
| S2 optical (6 bands) | **MOSAIKS/RCF, empirical mode** | **CROMA-base** (optical branch) | RCF: convex combinations are *exact by construction*; CROMA: only FM taking 101×101 near-natively |
| S1 SAR (VV/VH) — the non-circular outcome | **CROMA-base `modality='SAR'`** | **SoftCon ViT-B/14** or **Galileo** | CROMA: chip-size + product match; SoftCon: best frozen S1 numbers anywhere |
| Joint S1+S2 | **TerraMind-base `['S1GRD','S2L2A']`** | **Galileo-base** | TerraMind: only model with an S1GRD-vs-S1RTC switch AND an S2 band-subset API |
| Baselines (mandatory) | current 11 band means; pooled backscatter stats (VV/VH mean, std, ratio, percentiles) | plain **DINOv2** | If a 6-number handcrafted vector matches the FM, that IS the finding |

**Do not use:** DOFA for S1 (pretrained on raw linear-amplitude DN in **vh,vv** order — a
different physical quantity from our dB chips, plus a placeholder wavelength); AlphaEarth /
Google Satellite Embedding (annual cadence straddles Helene, and its unit-sphere geometry
is *measured* to break vector averaging); TESSERA (terrain-flattened RTC + annual — double
distribution mismatch); Presto as primary (per-pixel, no spatial context — keep only as a
256-d pooled baseline); SpectralGPT (GPL-3.0); SkySense (non-commercial license);
SeCo-style seasonal-contrast models (trained to be *invariant* to exactly the seasonal-window
change our design measures).

**The one finding that reframes the whole question (§2):** the literature has validated
that frozen EO embeddings are linearly *decodable* (ridge on frozen features recovers
elevation/temperature at R² ≈ 0.97), but the only direct measurements of linear
*composability* — whether Σwᵢ·e(xᵢ) means anything — are negative for deep ViT embeddings
(Prithvi latent interpolation MRR < 0.2; AlphaEarth "weighted combinations are unreliable").
Our estimator needs composability, not just decodability. MOSAIKS-style random convolutional
features are the exception: because they are a ReLU followed by a spatial **mean**, an
area-weighted mixture of land covers has, *by algebra*, the convex combination of the pure
covers' feature vectors. That's why the "naive random encoder" is not the straw man here —
it is the only encoder whose geometry provably matches the estimator, and the planted-mixture
battery (§4, test 2) is the empirical test that decides whether any deep encoder matches it.

---

## 1. Good news first: our data is in-distribution

- **The S1 chips need no apology.** GEE's `COPERNICUS/S1_GRD` (thermal-noise removed,
  calibrated σ⁰ in dB, geometric terrain correction only, **no radiometric flattening**)
  is exactly what SSL4EO-S12 was built from — and SSL4EO-S12 is the pretraining substrate
  for CROMA, SoftCon, DeCUR, FG-MAE, and TerraMind's `S1GRD` branch. Galileo went further:
  its S1 corpus is literally `COPERNICUS/S1_GRD` + `IW` + `["VV","VH"]` + **median
  composite** — our pipeline verbatim, with published normalization stats
  (VV −11.73 ± 4.89 dB, VH −18.86 ± 5.73 dB) that apply to our chips directly.
  The models that WOULD mismatch are the RTC-trained ones (TESSERA, TerraMind's separate
  `S1RTC` branch) — avoid those branches, not the GRD ones.
- **Our 6 S2 bands are a first-class citizen.** B2,B3,B4,B8,B11,B12 is literally Prithvi's
  HLS band set (but Prithvi is 30 m and optical-only — resolution mismatch, skip), exactly
  three complete Galileo channel groups (`S2_RGB` + `S2_NIR_10m` + `S2_SWIR`, with the
  red-edge/B8A groups legitimately maskable), and TerraMind's own documented subset example
  (`['BLUE','GREEN','RED','NIR_NARROW','SWIR_1','SWIR_2']`).
- **Chip geometry:** 101×101 is prime, so ViT patch grids never divide it. Resolutions:
  CROMA pad→104 (patch 8, zero resampling — the cleanest); Galileo crop→96 (its native
  pretraining tile); AnySat crop→100; 224-fixed models (SoftCon, TerraMind, DOFA,
  Panopticon) need a 2.2× upsample. RCF and MMEarth (convolutional) take 101×101 as-is —
  the Microsoft Planetary Computer MOSAIKS tutorial uses *"~101×101 pixel"* S2 chips,
  literally our geometry.

## 2. The decodability-vs-composability gap (why encoder choice is a geometry question)

What the field measures ("information preserved") and what our estimator needs are two
different properties:

**Linear decodability — well established.** Frozen-embedding + ridge/linear-probe evidence:
26 environmental variables reconstructed from AlphaEarth embeddings, 12 at R² > 0.90,
elevation/temperature ≈ 0.97 (arXiv:2602.10354); TESSERA canopy height R² 0.66 where
competitors were < 0.05; NeuCo-Bench formalizes exactly this protocol (frozen fixed-size
embeddings, task-agnostic linear probes, R²).

**Linear composability — the property our simplex needs — is mostly unvalidated, and the
direct evidence is negative for deep models:**
- **LEPA (arXiv:2603.07246):** interpolating Prithvi-EO-2.0 patch embeddings gives MRR
  < 0.2 (their learned-equivariant fix reaches 0.8) — "the embedding manifold is highly
  non-convex."
- **AlphaEarth geometry (arXiv:2604.18715):** effective dimensionality ~13 of 64; tangent
  spaces rotate > 60° at 84% of locations; verdict: "vector averaging and weighted
  combinations are unreliable" — while *retrieval* stays coherent. (Also: AlphaEarth
  vectors are unit-normalized, so convex combinations leave the sphere by construction.)
- **Two mitigating facts:** (a) our combination is *local* — 10 donors that already share
  NLCD class, elevation, slope, and ≤100 km — so we need only local flatness, which is the
  regime where even AlphaEarth's geometry behaves (that's why retrieval works); (b) NeuCo-Bench
  found post-encoding aggregation (averaging embeddings) *beat* pre-encoding aggregation
  (averaging inputs) for seasonal views — averaging embeddings is not doomed, it is
  encoder-dependent.
- **MOSAIKS/RCF is exact:** feature = spatial mean of ReLU activations ⇒ a chip that is an
  area mixture of covers has exactly the area-weighted convex combination of the covers'
  features. Composability holds by construction, and the same linearity gives a free
  by-product: the fitted weight vector applied to *unpooled* activations yields a per-pixel
  treated-minus-synthetic gap map (their "label super-resolution" — >30% of within-image
  variance explained at 32×32 sub-grids from chip-level labels only). A scar-localization
  map for free.
- **Theory anchor:** Okano & Kurisu (arXiv:2601.07539) formalize SC for metric-space-valued
  outcomes via isometric Hilbert embeddings — the estimator is licensed exactly when the
  embedding is near-isometric. Nobody has tested which EO encoder satisfies that. Our
  planted-mixture battery is that test, and it is unclaimed ground.

**Also load-bearing:** benchmark rankings are unreliable (same checkpoint, same benchmark,
linear-probe accuracy reported anywhere from 33.0 to 89.6 across papers — arXiv:2605.12678),
and preprocessing choices move frozen-probe results by 25+ points (Corley et al.,
arXiv:2305.13456 — resize + channel standardization took an ImageNet ResNet from 0.66 to
0.91). So: pick candidates by *structural fit*, compare them *on our chips*, under *one*
preprocessing convention.

## 3. The candidates

### 3a. MOSAIKS / RCF (empirical mode) — optical 1st choice
Random 3×3×C convolutional filters **sampled from our own chips** (not Gaussian), ZCA-whitened,
ReLU(±) then global average pool; K = 2048–8192 features; downstream = ridge/QP.
`torchgeo.models.RCF(in_channels=6, mode='empirical', dataset=...)` — note torchgeo's
default `mode='gaussian'` and `bias=-1.0` differ from the paper (empirical patches, bias +1).
BSD-3/MIT implementations; 101×101 native. Evidence it is not a straw man: top-2 on 4/5
multispectral datasets in Corley et al. (BigEarthNet mAP 76.29 vs SSL4EO-MoCo 70.17,
ImageNet 65.88); in Google's own AlphaEarth evaluation MOSAIKS **beat SatCLIP, Prithvi, and
Clay** (ASTER emissivity R² 0.69 vs AEF's 0.72; supervised change detection 72.0% vs AEF's
78.4%). Weakness: bag-of-textures (no spatial arrangement in the pooled vector — recover it
via unpooled maps); weaker under geographic shift; weak on RGB object semantics (irrelevant
for us). ZCA must be fit on OUR 6-band patches.

### 3b. CROMA-base — S1 1st choice, optical/joint runner-up
Radar ViT + optical ViT + cross-attention fusion; contrastive (radar↔optical) **+ MAE**;
pretrained on SSL4EO-S12 (our exact S1 product); 2D-ALiBi ⇒ `image_resolution` any multiple
of 8 (pad 101→104, zero resampling — unique among FMs); `PretrainedCROMA(size='base',
modality='SAR')` returns a pooled `SAR_GAP` vector; MIT, weights on HF (`antofuller/CROMA`).
Evidence: best GFM on PANGAEA Sen1Floods11 (90.89 mIoU) and MADOS (67.55); SAR-only linear
probe 77.9 mAP BigEarthNet-SAR / 87.5 EuroSAT-SAR; best FM in the 10%-label regime.
Caveats: S2 branch wants **12-band L1C** — our 6-band L2A compromises the *joint* path
(zero-fill + reflectance mismatch), so use CROMA for S1-only and (with the channel-slice
caveat flagged as exploratory) optical-only; if the collaborator can re-export 12-band L1C
chips, CROMA becomes joint 1st choice.

### 3c. SoftCon ViT-B/14 — strongest frozen S1 numbers
Multi-label soft contrastive on SSL4EO-S12, DINOv2 init; dedicated 2-channel SAR checkpoints
(`B2_vitb14_softcon.pth`, 768-d; Apache-2.0). The one table comparing all S1 encoders under
a single protocol puts it first: **82.5 mAP** BigEarthNet-SAR-10%, **89.1** EuroSAT-SAR
(vs CROMA 77.9/87.5, SSL4EO-MoCo 76.6/82.4, FG-MAE 71.7/80.7, SSL4EO-MAE 69.8/79.3).
Costs: 224×224/patch-14 ⇒ 2.2× upsample; normalization constants not published (derive from
our donor pool, once, and freeze); contrastive objective ⇒ run it, but watch whether it
washes out disturbance signal (test 2/3 below will show it).

### 3d. Galileo (base + nano) — tightest pipeline match
Multimodal ViT, channel-group tokens, dual global/local contrastive losses (the *local* loss
on shallow projections is the only explicit low-level objective in this field — designed for
small-object signal). Its S1 pretraining recipe is our pipeline verbatim (GEE S1_GRD, IW,
VV/VH, median composite, dB stats published); our 6 S2 bands = 3 complete channel groups,
red-edge/B8A groups maskable (edit `s_t_m` yourself — documented mask semantics, not a
worked example). T=1 legal, all aux modalities optional, month defaults matter (fix to
composite-window midpoint for every site). Crop 101→96 (its native tile). MIT;
`nasaharvest/galileo`; `single_file_galileo.py` is dependency-free. **Nano's 128-d output
is a real advantage for a 10-donor fit** (see §5 dimensionality). Frozen S1-only
Sen1Floods11 linear probe 79.4 mIoU — best published. Caveats: pretraining spatial crops
were 4–12 px (96×96 is an extrapolation); their own ablation says removing S1 from
pretraining barely hurt — S1 was supplementary for them.

### 3e. TerraMind-base — joint 1st choice
Dual-scale (pixel+token) multimodal encoder; the only model that (i) separates `S1GRD` from
`S1RTC` and lets us pick the branch matching our data, (ii) takes `S2L2A` directly with a
**documented band-subset API** (their example subset ≈ our band set), (iii) applies
pretraining standardization automatically. Best PANGAEA average of any GFM (59.57 vs
from-scratch UNet 57.58 — the only GFM above the UNet). Apache-2.0, `ibm-esa-geospatial`.
Costs: 224/16 resample; its own ablation: S1-only costs ~9 mIoU vs S2-only on Sen1Floods11.
Bonus: its generative decoder can decode an interpolated embedding back to imagery — a
qualitative check on whether the donor convex hull is well-behaved.

### 3f. Worth one run each, not primary
- **AnySat** (`s1-asc`, crop 100, patch_size 40 m, `output='tile'`): owns the only published
  frozen S1-only *change-detection* number (BraDD-S1TS 78.9 mIoU linear probe). S2 path
  needs 10 bands — S1-only for us. Needs a nominal day-of-year: fix to composite midpoint.
- **Copernicus-FM**: best absolute frozen S1 classification (EuroSAT-S1 87.2, BEN-S1 83.3)
  — but on Flood-S1, the S1 change task, a random-init supervised net beats every FM
  (78.3 vs 77.7), and there is no published S1-only usage example. Also: it conditions on
  lon/lat/time metadata — zero or fix it, or donors get different conditioning than targets.
- **MMEarth (ConvNeXt-V2 MAE, atto)**: the only convolutional FM — native 101×101, fine
  spatial detail; license is dual MIT/CC-BY-NC (fine for research).
- **DINOv2 control**: on the one SAR retrieval benchmark, generic DINOv2/v3 sit within ~8
  points of the best EO specialist. If DINOv2 ties the EO models on our gate, EO pretraining
  isn't buying anything here.

## 4. The validation gate (run before believing any encoder)

Order matters; each test is cheap on 286 chips. This is the satellite port of the n2c2
planted-mixture battery, plus field-standard probes.

0. **Freeze preprocessing first.** One resize rule, one per-channel standardization
   (constants computed from the 260-donor pool), identical across all encoders and both
   periods. Log it. (Corley: this choice is worth 25+ points — bigger than encoder gaps.)
1. **Linear decodability probe.** Frozen embedding → ridge → predict the 11 chip-mean band
   values + slope + elevation, spatial-block CV (not random splits). *Acceptance bar: beat
   the raw 11-band vector at predicting held-out continuous variables.* An encoder that
   can't reconstruct chip-mean NDVI/VV has discarded the signal the current design runs on.
2. **Planted-mixture recovery — the gate.** Build synthetic chips as known convex mixtures
   of real control chips (spatial mosaics at known area fractions; also pixel blends),
   embed, recover simplex weights with our `solve_simplex_qp`, score weight-recovery error
   AND ‖Σwᵢe(xᵢ) − e(mix)‖. RCF should pass by algebra (it's the positive control);
   Prithvi-class models are expected to fail (LEPA). Whatever passes proceeds; whatever
   fails is demoted to matching-only use. **This test is also the publishable contribution
   — nobody has run it on EO encoders.**
3. **Pre-period placebo fit in both spaces.** Fit weights on before-embeddings; report
   pre-fit residual in embedding space AND decoded back to band space via the probe from
   test 1. Tiny embedding residual + poor band residual = fitting unphysical directions
   (representation-induced confounding, the RICB failure mode).
4. **Retrieval fallback.** Run k-NN matching in embedding space alongside simplex SC. If
   they disagree materially, trust matching and report the disagreement (retrieval is the
   operation the geometry papers say survives even when arithmetic doesn't).
5. **SAR-channel probe.** Does the joint/optical encoder predict chip-mean VV/VH as well as
   optical quantities? An effectively-optical-only encoder is disqualifying here, since S1
   is the non-circular outcome (the USGS inventory is an S2-NDVI-change product).
6. **Concatenation arm.** Fused embeddings beat the best single model on 4/6 tasks in the
   one complementarity study — pre-register a concat arm so "no single winner" is a result,
   not a dead end.

## 5. Design rules that matter more than encoder choice

- **Dimensionality vs donor budget.** 10 donors span a 9-dim simplex; fitting a 768-d
  target with 10 weights is heavily overdetermined and the residual will be dominated by
  high-variance nuisance directions. Prefer low-dim outputs (Galileo-nano 128, Presto-pooled
  256, SoftCon ViT-S 384) or PCA all embeddings to ~10–20 dims (fit PCA on donors only)
  before the QP. First-order decision, not a detail.
- **Don't pool to one vector per chip (for FMs).** Keep the token grid (`optical_encodings`
  / `SAR_encodings` for CROMA, per-patch averages for Galileo) and fit SC per token or on
  scar-scale sub-windows; PANGAEA's small-object task shows pooled GFM features losing to a
  UNet by 19 mIoU. This is our §11 dilution finding in encoder form: a scar at ~4% area
  moves a chip-mean feature by ~4% of the texture contrast.
- **Pool intermediate layers, not just the last.** Two independent studies: mid-depth ViT
  blocks carry more low-level information than the final layer (reconstruction models hold
  low-level info late; contrastive models don't). Extract blocks ~8/10/12 and let test 3
  pick.
- **Match donors on radar geometry.** No RTC ⇒ embedding distance partly encodes
  slope/aspect vs look direction. We already know 21/26 donor groups are orbit-homogeneous
  (01 §8); restrict or covariate-adjust the 5 mixed groups.
- **Fix nominal dates/months once** (Galileo month, AnySat DOY): composite-window midpoint,
  identical for all sites — otherwise it's an uncontrolled per-unit knob.
- **Keep roles straight (Rambachan–Singh–Viviano).** The landslide causes the image, so
  image-derived quantities are post-outcome variables; using embeddings as the
  *matching/weight-fitting* space (fit on BEFORE chips only) is the safe role. Any
  embedding-derived quantity on the left-hand side needs the RSV estimator / multiple
  imputation / PPI-style corrections. Our current design (weights on before-features,
  effects read on physical bands) already respects this — keep it.
- **Humility check from the nearest published attempt:** unsupervised landslide change
  detection on frozen SSL4EO-DINO embeddings scored F1 = 31.7% — better than naive band
  differencing (19.3%), *worse than hand-designed spectral indices*. The realistic outcome
  space includes "indices win," and the baselines in §0 make that a reportable result.

## 6. Where this contributes (paper positioning)

Verified gaps as of 2026-08-14: no published synthetic control with simplex weights on EO
foundation-model embeddings (Okano–Kurisu supply the metric-space theory; no EO
application); no benchmark evaluates raw frozen-embedding *distance/composition* (all
attach probes or decoders); no direct measurement of whether contrastive/seasonal-invariant
objectives destroy disturbance signal in Sentinel-1; no SAR foundation-model embedding
applied to landslide detection; no published S1-based Helene landslide mapping. The
planted-mixture gate (test 2) + band-vs-embedding SC comparison (H2 analog) sit exactly in
these gaps.

## 7. Key sources

Composability/geometry: LEPA arXiv:2603.07246 · AlphaEarth geometry arXiv:2604.18715 ·
Okano–Kurisu arXiv:2601.07539 · Earth-embeddings review arXiv:2608.03410.
Evaluation methodology: PANGAEA arXiv:2412.04204 · NeuCo-Bench arXiv:2510.17914 ·
Corley preprocessing arXiv:2305.13456 · "No One Knows the SOTA" arXiv:2605.12678.
Encoders: MOSAIKS Rolf et al. Nat.Comm. 12:4392 (algorithm in NBER w28045 §S.3.3) ·
CROMA arXiv:2311.00566 · SoftCon arXiv:2405.20462 · Galileo arXiv:2502.09356 ·
TerraMind arXiv:2504.11171 · AnySat arXiv:2412.14123 · Copernicus-FM arXiv:2503.11849 ·
DOFA arXiv:2403.15356 (S1-disqualified: Satlas linear-DN vh/vv) · AlphaEarth arXiv:2507.22291.
Causal-with-satellites: Rambachan–Singh–Viviano arXiv:2411.10959 · Proctor–Carleton–Sum
NBER w30861 · Ratledge et al. Nature 611 (2022) · Jerzak et al. arXiv:2301.12985.
Landslide+SAR: frozen-embedding landslide CD DOI:10.1080/17538947.2025.2547292 ·
global S1 landslide DL GMD 19:167 (2026) · S1 backscatter conceptual model RemoteSens.
17(19):3313.

*Caveats: numbers above are as reported by the surveyed papers; the GFM literature has
documented reproducibility spreads (§2), so treat rankings as soft and our own gate (§4) as
the decision procedure. Coverage of SatMAE/Scale-MAE/SeCo/SatCLIP details is thinner than
the rest (one survey agent was cut short); none of the four was on track to be recommended —
SatMAE/Scale-MAE rank mid-pack on PANGAEA, SeCo's seasonal invariance is contraindicated,
SatCLIP is a location encoder.*
