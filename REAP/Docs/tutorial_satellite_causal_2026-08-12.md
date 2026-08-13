# Causal inference on satellite images — the Helene pilot, two estimators, and the proposal — a tutorial

2026-08-12. Written for Jun, using the REAP Hurricane Helene dataset and the
CALS ORI proposal as the running examples. Companion to
[tutorial_synthetic_control_2026-08-11.md](../../preliminary/Docs/tutorial_synthetic_control_2026-08-11.md)
(the solver, the factor model, and the n2c2 weight-recovery battery live
there; this document does not repeat them, it builds on them). Executed
companions in `REAP/notebooks/`: `01_dataset_analysis.ipynb` (descriptive
dataset characterization) and `02_did_vs_synthetic_control.ipynb` (the
DiD-vs-SC estimator comparison and placebo tests). Reading list
with specific sections at the end. Every cited paper was verified to exist at
a real URL; two items that could not be fully verified are marked as such.

## 0. Terminology — the remote-sensing vocabulary used below

Written for a causal-inference reader, not a remote-sensing one.

- **Sentinel-2 (S2)** — an ESA satellite pair carrying an *optical* sensor: a
  passive camera measuring reflected sunlight in several wavelength channels,
  10 m pixels, ~5-day revisit. Blind through clouds and at night.
- **Sentinel-1 (S1)** — ESA satellites carrying *SAR* (synthetic-aperture
  radar): an *active* sensor that transmits microwave pulses and measures the
  echo. Works through clouds and darkness — which is why it matters for a
  hurricane, when the optical view is cloud-blocked for days.
- **Band** — one measured channel per pixel. An S2 pixel here carries 8
  numbers: **B2/B3/B4** = blue/green/red visible reflectance, **B8** =
  near-infrared (NIR; healthy leaves reflect it strongly), **B11/B12** =
  shortwave infrared (SWIR; sensitive to moisture and bare soil), plus two
  derived indices. The "B" numbers are just ESA's channel numbering.
- **Reflectance** — fraction of incoming sunlight reflected, scaled ~0–1.
- **Backscatter (VV, VH; in dB)** — radar echo strength. VV = transmit
  vertically polarized, receive vertical (surface roughness, moisture);
  VH = transmit vertical, receive horizontal ("cross-pol") — energy only
  rotates polarization by bouncing around inside 3-D structure, so VH tracks
  *volume scattering* from canopy or debris. Logarithmic dB scale; values
  like −10 (VV) and −16 (VH) are typical here.
- **NDVI** = (NIR − Red)/(NIR + Red) ∈ [−1, 1] — the standard vegetation-health
  index (dense forest ≈ 0.8 in this data; bare soil near 0). **NDWI** =
  (Green − NIR)/(Green + NIR) — surface water / wetness.
- **Chip** — a small image cropped around a site: here 101×101 pixels at 10 m
  = a 1.01 km square.
- **Median composite** — not a single-day photo: every satellite pass inside
  the date window is stacked, cloudy/shadowed pixels are masked out, and each
  pixel takes the median of what remains. One cleaned image per period.
- **NLCD** — the USGS National Land Cover Database: every cell classified as
  deciduous forest, pasture/hay, developed, etc.
- **DEM / elevation / slope** — digital elevation model; elevation in meters
  and terrain slope in degrees at the site center, used for matching.
- **USGS landslide inventory** — expert-mapped *point locations* of
  landslides triggered by Helene. This is the treatment-assignment record.
- **Treated vs control, before vs after — two different axes.** *Treated
  site* = a 1 km² window centered on a mapped Helene landslide (26 of them).
  *Control (counterfactual) site* = a similar window with no mapped landslide
  within 5 km (260 of them). Treated/control says **where**; before/after
  says **when**. Every site — treated and control alike — has both a before
  image (2024-08-15→09-23) and an after image (2024-10-01→10-31); the
  "treatment event" (the storm, 09-26/27) sits between the two periods.
  A control site is *not* "a site before the disaster": it lived through the
  same storm, just without a mapped landslide.

## 1. The estimand — same potential outcomes, new outcome type

Nothing about the causal question changes when the outcome becomes an image.
For site *i* at time *t* there are two potential outcomes (Rubin 1974; Imbens
& Rubin 2015): the factual `Y_it^F`, what we observe, and the counterfactual
`Y_it^0`, what the site would have looked like had the shock not occurred.
The causal effect is their difference:

    τ_it = Y_it^F − Y_it^0

This is exactly the proposal's estimation framework (§6 of the narrative),
and the proposal's central argument is about *timing*: official outcome data
— yields, farm income, employment — arrive months or quarters after a
disaster, but Sentinel-2 revisits every ~5 days at 10 m. If `Y` is
image-derived, the causal question becomes answerable while response
decisions are still being made. The methodological price: `Y_it^0` is now a
counterfactual *image-derived condition*, not a scalar, and no one observes
it. The "not a scalar" contrast is with *classical* SC (where Y is GDP or
cigarette sales), not with our clinical preliminary — structurally the two
phases are the same latent-SC problem: a high-dimensional artifact per
unit-time (a clinical note there, an image chip here), an encoder that maps
it to a vector (text embedding there, image encoding there), and SC run on
the encoded representation. Only the source modality changed. Every method
below is a strategy for filling the counterfactual in.

The pilot instantiation:

| Framework object | Helene dataset realization |
|---|---|
| unit *i* | a 1 km × 1 km site (101×101 px @ 10 m), 26 treated + 260 controls |
| shock | Hurricane Helene, landfall 2024-09-26, Appalachian impact 09-27 |
| pre-period | median composite 2024-08-15 → 09-23 |
| post-period | median composite 2024-10-01 → 10-31 |
| `Y_it` | 8 Sentinel-2 bands + 3 Sentinel-1 bands per pixel, or features thereof |
| treatment assignment | site appears in the USGS preliminary landslide inventory |
| donor pool for site *i* | its 10 matched counterfactual sites (or all 260) |

## 2. Two currencies of comparison — matching, then equal weights vs estimated weights

The collaborator's design already contains a causal method; it is worth
naming its parts precisely, because our contribution is a replacement for
exactly one of them.

**Part 1 — matching (design stage).** Each treated site got 10 controls with
the same NLCD land-cover class, similar elevation and slope, ≥ 5 km from any
mapped landslide, drawn from 74,998 candidates, no control reused. This is
classical matched-design causal inference: make treated and control units
comparable on observables *before* looking at outcomes. It worked, as far as
it goes — all 260 matches are "strict" quality, mean elevation gap 15.5 m,
mean slope gap 0.83°.

**Part 2 — the estimator (analysis stage).** The `test/` folder implements

    adjusted effect = (Y_treat,after − Y_treat,before)
                    − mean_j (Y_control_j,after − Y_control_j,before)

which is difference-in-differences on the matched set: a 2×2 DiD where the
control quantity is the *equal-weight average* of the 10 matched controls.
Its identifying assumption is **parallel trends**: absent the hurricane, the
treated site's change would have equaled the average matched control's
change. Matching makes that assumption more plausible (similar terrain and
land cover should trend similarly through a season), and the subtraction
removes both site-level level differences and the region-wide seasonal
change (August greenness → October senescence) that contaminates any naive
before/after difference.

**Synthetic control changes only the weights.** Instead of `w_j = 1/10`
imposed by fiat, estimate them:

    min_w  || Z_treat,before − Σ_j w_j Z_control_j,before ||²
    s.t.   w_j ≥ 0,  Σ_j w_j = 1

    counterfactual change:  Σ_j w_j (Y_j,after − Y_j,before)
    effect:                 ΔY_treat − Σ_j w_j ΔY_j

where `Z` is whatever representation of the pre-period we fit in (band
features, pixels, or embeddings — §3–4). The equal-weight DiD is the single
point `w = (0.1, …, 0.1)` of the simplex; SC searches the whole simplex for
the point that best reproduces the treated site's pre-state. Everything from
the preliminary tutorial's §1–3 applies verbatim — same QP, same simplex
geometry, same solver (`solve_simplex_qp`), same no-extrapolation and
sparsity properties.

**Why estimating weights should matter here** — the matching table says the
10 controls are not interchangeable: within a group, distances run 5–51 km,
elevation gaps reach 129 m, and the collaborator's own `matching_score`
varies severalfold between rank 1 and rank 10. The mean gives the worst
match the same say as the best. SC down-weights it, and reports doing so —
a fitted result reads like "this site's counterfactual is 0.6 × control 3 +
0.4 × control 7", where 0.6 and 0.4 are the *solver's estimated output*, not
a fixed choice (the only fixed weights in this project are the collaborator's
1/10; in the n2c2 validation we additionally *planted* weights as ground
truth to test whether the solver recovers them — that is where "planted 0.9,
recovered 0.884" comes from). The estimated weights are the auditable causal
claim, which is what the proposal means by outputs "auditable by Extension
agents" (Module 3).

**Why both get reported anyway.** DiD's assumption is weaker in one
direction — it tolerates a constant level gap between treated and synthetic
unit — and with a single pre-period SC cannot demonstrate pre-trend fit over
time. The honest reading: DiD-on-matches is the transparent baseline the
proposal promises ("feature-space synthetic-control models" as the
"transparent baseline", Module 2), SC is the refinement, and agreement
between them is itself evidence. Disagreement localizes exactly which
controls (or which assumption) drive the estimate.

## 3. One pre-period is not fatal — the image is the pre-trajectory

Classical SC fits weights on T0 pre-treatment *time points*: each pre-period
is one equation ("the weighted donors must reproduce the treated unit's
outcome in year t"), and T0 equations pin down the m unknown weights. We
have T0 = 1: one before-composite. If the outcome were a single number per
period (say, mean NDVI), that would be 1 equation in 10 unknowns —
infinitely many weight vectors fit perfectly and the fit means nothing.

But the before-observation is not one number. The chip is 101 pixels × 101
pixels, and each pixel carries 8 S2 band values — 101 × 101 × 8 = 81,608
numbers (plus 3 more bands per pixel from S1; ~112k in total). Requiring the
weighted donors to reproduce *every* one of those numbers is ~112k equations
in 10 unknowns — heavily over-determined, the opposite of hopeless. Even
retreating to summary features (a mean per band, quantiles, changed-pixel
fractions — a few dozen numbers) still leaves more equations than unknowns.
The role that "a long pre-period" plays in Abadie's bias bound — enough
constraints that only a truly comparable donor combination can satisfy them
all — is played here by the *dimensionality of the single pre-image*.

This has an exact precedent, and it is the same one the n2c2 battery's
prior-art sweep surfaced: **sparse hyperspectral unmixing** (Iordache,
Bioucas-Dias & Plaza, IEEE TGRS 2011). There, a pixel's spectrum is modeled
as a nonnegative, sum-to-one mixture of library spectra, and validity is
established by *planting known abundances and recovering them* — the field's
standard protocol and precisely our H1 design. The satellite move brings the
project to the literature where its core mathematics already lives.

The caveat carried over from the preliminary (Ferman, JASA 2021):
reconstructing the pre-image and identifying the weights are different
events. A good pre-fit with non-unique weights is possible; on n2c2 the
LP certificate distinguished them. The same check — is the recovered `w`
unique on the simplex? — ports to image features unchanged and should be run
before any real-data weight is interpreted.

## 4. The latent move — why not pixels, and what must be re-established

The collaborator's composites are the pixel-space answer, and their own
comparison figure is the argument against stopping there. The treatment chip
is a photograph of one real place, so its structures are crisp — you can see
the roads, field boundaries, and buildings. The "10-control mean" image is
built by averaging pixel (1,1) across ten *different places*, then pixel
(1,2) across those ten places, and so on. A road crossing control A at some
pixel does not line up with anything in control B; averaging misaligned
structures cancels them, leaving smooth green mush (see
`01_dataset_analysis.ipynb` §4, bottom row — the blur is obvious). Pixel (row, col) at ten different locations is
not the same physical quantity — the collaborator's own notebook warns that
alignment is "relative position in the chip," not geography. Averaging
destroys exactly the spatial structure a landslide signature lives in, and
the resulting "counterfactual image" is not a plausible picture of any
place. That is the pixel-space dead end the latent move exists to escape.

Three spaces to run SC in, in increasing order of ambition:

1. **Band-feature space** (now): per-site summaries — mean NDVI, mean VV,
   fraction of pixels past a change threshold. Transparent, interpretable,
   the proposal's primary deliverable. Loses within-chip spatial pattern.
2. **Pixel space** (the collaborator's test): keeps resolution, but the
   mixing assumption (site ≈ convex combination of other sites, pixelwise)
   is geometrically wrong across locations.
3. **Embedding space** (the exploratory track): encode each chip with a
   remote-sensing foundation model; mix there; interpret weights there;
   decode or translate back to indicators. The proposal names this
   explicitly — "exploratorily in latent or image space via foundation-model
   embeddings" (Objective 3).

The n2c2 preliminary tells us exactly which parts of option 3 come for free
and which do not:

| carries over unchanged | must be re-established on images |
|---|---|
| the QP solver, Gram form, simplex constraints | whether image-embedding geometry supports convex mixing at all (H1 rerun: plant mixtures of control-chip embeddings, recover) |
| LP uniqueness certificates | retrieval/donor selection — on n2c2 it was the *only* error source (2.7/5 donors found); here the matched 10 are given, but the 74,998-candidate pool is the real donor universe |
| the planted-mixture validation protocol | representation dependence (H2 analog: do band-feature weights and embedding weights agree? On n2c2, two representations of the same notes disagreed at chance — assume nothing) |
| the falsifiability discipline (a battery that cannot fail is not a test) | decoding (H3/H4 analog: generating the counterfactual *image* — the collaborator's mean image is the pixel-space decoder, and it blurs; a learned decoder is future work) |

One image-specific risk the proposal's own reference list flags: **shortcut
learning** (Geirhos et al., Nature Machine Intelligence 2020). An encoder may
represent chips by features that predict well but are causally irrelevant
(sensor artifacts, chip border effects, road density). If SC weights are fit
in such an embedding, "similar" means similar shortcuts. This is the image
version of the H2 finding, and the planted-mixture battery plus
representation-comparison is our instrument for catching it.

## 5. The proposal, concept by concept

The seed proposal ("Causal AI for Real-Time Agricultural Shock Monitoring,"
PI Le Wang; $20K CALS ORI + $10K Sanghani; 8/2026–6/2027) commits to a
**five-stage pipeline**. Mapping each stage onto this dataset:

1. **Shock detection & classification** — identify timing, location, type,
   severity. Helene pilot: done externally by USGS (the landslide inventory
   *is* the detection product) and NOAA (best track, wind swath — both in
   `data/boundaries/`). The pilot therefore tests stages 2–4, not stage 1.
2. **Factual prediction (Module 1)** — observed-state estimates from
   real-time data. The proposal's indicator list: NDVI, EVI, NBR, NDMI,
   land-surface temperature, soil-moisture proxies, fractional cover, flood
   exposure, crop stress, land-cover measures, plus learned embeddings.
   The dataset delivers NDVI and NDWI precomputed; NBR-like information via
   B11/B12 (SWIR); moisture/flood/structure via S1 VV/VH. EVI and LST would
   need re-export — noted as a gap, not a blocker.
3. **Counterfactual prediction (Module 2, "the causal core")** — the
   proposal is explicit that it applies the *logic* of synthetic control
   ("comparable unaffected areas contain information about what would have
   happened") rather than only the fixed convex-weights construction, and
   promises (a) feature-space SC as the transparent baseline, (b) learned
   counterfactual-imputation models with multimodal covariate adjustment
   (the Chernozhukov et al. 2018 DML reference), (c) exploratory latent-space
   imputation via foundation models. §2–4 above are exactly (a) and (c);
   the matched-control design is the donor-pool discipline both need.
4. **Causal impact estimation** — `τ̂ = Y^F − Ŷ^0`, validated by pre-shock
   prediction accuracy, placebo tests, leave-one-region-out cross-validation,
   sensitivity analysis, and comparison against later-released administrative
   outcomes (Objective 4). The 260 controls make placebo tests natural:
   pretend a control is treated, estimate its "effect," and the treated
   sites' effects should be extreme in that null distribution (Abadie's
   permutation inference; Liu, Wang & Xu 2024 for the modern diagnostics).
5. **Decision support (Module 3)** — image-derived quantities translated to
   indicators, maps, rankings, uncertainty, alerts, plain-language summaries
   for Extension agents, emergency managers, lenders, and policymakers. For
   the pilot this is the per-site effect table and map; the proposal's
   county-level economic linkage (Jean et al. 2016; Burke & Lobell 2017;
   Donaldson & Storeygard 2016) is future scope.

Methodological anchors named in the proposal and where they enter: Abadie
et al. 2010 (the SC estimator, §2), Ben-Michael et al. 2021 (augmented SC —
bias-correct SC with an outcome model when pre-fit is imperfect; natural
here because one composite pre-period makes perfect fit unlikely), Athey
et al. 2021 (matrix completion — treat all sites × periods as a matrix with
the treated-post block missing; with only two periods it reduces to close
kin of DiD, but it is the frame that scales when more dates are added),
Chernozhukov et al. 2018 (DML for covariate adjustment on elevation, slope,
land cover), Rubin 1974 / Imbens & Rubin 2015 (potential outcomes), Athey &
Imbens 2017 (the state of applied policy evaluation). The remote-sensing
anchors: Drusch et al. 2012 (Sentinel-2 mission), Tucker 1979 / Huete et al.
2002 / Didan et al. 2015 (vegetation indices), Reichstein et al. 2019 (deep
learning for Earth system science), Geirhos et al. 2020 (shortcut learning,
§4), Jean et al. 2016 / Burke & Lobell 2017 (satellite-to-economic-outcome
translation).

Timeline fit: the dataset as shared already satisfies the Months 1–3
milestones (pilot selected, treated and comparison areas defined, indicators
constructible) and the data half of Months 3–7 (feature-space
counterfactual-imputation models); the Months 6–10 exploratory latent-space
work is where the n2c2 battery port lands.

## 6. What this dataset can and cannot support

**Can, today:** matched-pair DiD on any band feature (all 26 treatments ×
11 bands); feature-space SC with per-treatment donor pools of 10 (or the
full 260, or the 74,998-candidate pool in `sites/`); placebo inference from
260 controls; the planted-mixture validity battery on chip embeddings; S1/S2
cross-sensor agreement checks (radar sees through clouds; optical is the
inventory's own source — see the circularity note below).

**Cannot, yet:**

- **Temporal pre-trends.** One before-composite means parallel-trends cannot
  be tested against history. Fixable: the sites are fixed polygons and the
  Earth Engine notebooks are in `data/scripts/`; re-exporting monthly
  composites for 2023–2024 would turn every design above into a true panel.
  This is the single highest-value data extension.
- **Treatment intensity.** The inventory is preliminary *points*, no
  footprints, sizes, or severity — effects are per-site, not per-unit-area.
  Sutton & Stanley (2026) already document this limitation in print (no
  failure timing either); cite them rather than rediscovering it.
- **Interference/SUTVA.** Controls are ≥ 5 km from *mapped* landslides but
  shared the storm: rainfall, wind, and flooding hit them too. So `τ̂`
  estimates the effect of *landslide occurrence*, not of *the hurricane* —
  the control experience nets out the storm's diffuse effects. The README
  says this; the tutorial's job is to say it in estimand language.
- **A circularity to keep in view.** The USGS inventory was built largely
  from Sentinel-2 NDVI change over nearly the same windows this dataset
  uses (pre 8/26 & 9/22, post 10/2–10/12; Schaefer et al. 2025). Treatment
  status was thus *assigned by looking at the outcome variable*. An
  estimated "effect of a mapped landslide on NDVI change" is partly true by
  construction. Three mitigations: report S1 effects (radar was not used to
  build the inventory — an independent outcome); report non-NDVI optical
  bands; and frame NDVI results as *quantifying* the disturbance the
  inventory flagged, not as discovering it.
- **Note also** the email/README discrepancy (10 km vs 5 km exclusion — the
  pipeline implements 5 km: `06_create_sites.ipynb` §5/§11,
  `LANDSLIDE_EXCLUSION_METERS = 5_000`; the email misstates it) and the
  missing `treatment_0026`: IDs were assigned to all 27 Virginia landslides
  before matching, and §16–17 of the same notebook drop any treatment
  without exactly 10 controls — its own output records "treatment_0026
  obtained only 4 of 10 required controls." A documented rule, not an
  accident; worth confirming with the collaborator only that the email's
  10 km figure should read 5 km.

## 7. Related work — the intersection, ranked by closeness

Selection rule, per Jun: a paper counts only if it *both* works on
disaster/landslide satellite data *and* estimates a causal/counterfactual
quantity — detection and mapping papers are context, not related work.
Headline from three verified searches (2026-08-12): **the exact
intersection "Hurricane Helene + causal inference" is empty.** The USGS
inventory (Burgi et al. 2025, DOI 10.5066/P14CHGKS) has ~3 citing papers,
all detection, physical characterization, or hazard-model validation. No
paper applies SC, DiD, BACI, or any counterfactual estimator to Helene, with
or without imagery. The gap this project occupies is real and three layers
deep: no causal work on Helene; no chip-level SC on landslides for *any*
event; and the nearest SC-on-imagery work is wildfire, where scars are
large and spectrally unambiguous — Helene scars are small, steep, linear,
and cloud-afflicted, which is precisely the argument for the S1+S2
multimodal design.

Nearest neighbors, closest first (all verified at real URLs):

1. **Serra-Burriel, Delicado, Prata & Cucchietti 2021, "Estimating
   heterogeneous wildfire effects using synthetic controls and satellite
   remote sensing," *Remote Sensing of Environment* 265:112649
   (arXiv:2012.05140).** Generalized SC on NDVI/NBR/NDMI time series for
   >1,000-acre California wildfires, 1996–2016: for each burned unit, impute
   the no-fire index trajectory from a donor pool selected on *pre-event
   spectral similarity* rather than geographic proximity; effect = observed
   − counterfactual; GSC beats naive nearby-region comparison on pre-period
   predictive accuracy. **Why it matters to us:** this is our design with
   the disaster swapped — same outcome family, same donor logic, same
   estimand. The single prior-art citation if only one is allowed, and the
   template for the pre-fit diagnostic we should mirror.
2. **Fick, Nauman, Brungard & Duniway 2021, "Evaluating natural experiments
   in ecology: using synthetic controls in assessments of remotely sensed
   land treatments," *Ecological Applications* 31(3):e02264.** SC adapted to
   site/pixel-level RS vegetation time series for unplanned, unreplicated
   landscape events; simulations + a brush-clearing case study show recovery
   of effects under climate confounding and sensor noise, accuracy governed
   by donor count and quality. **Why:** the operational manual for our
   setting — same unit (sites, not jurisdictions), same trigger structure
   (dated unplanned event); their donor-count sensitivity results directly
   inform whether 10 matched controls suffice or the 74,998-candidate pool
   should be opened.
3. **Barton-Henry & Wenz 2022, "Nighttime light data reveal lack of full
   recovery after hurricanes in Southern US," *Environmental Research
   Letters* 17(11).** DiD across seven Cat-4/5 US landfalls: treated =
   FEMA-aid counties, controls = unaffected same-state counties, outcome =
   VIIRS nighttime lights; 2–14% depression three years out. **Why:** the
   cleanest published "hurricane + satellite outcome + textbook causal
   estimator," and its county-level coarseness is exactly the gap our
   site-level chips close.
4. **Zheng et al. 2025, "Nighttime lights reveal substantial spatial
   heterogeneity and inequality in post-hurricane recovery," *Remote Sensing
   of Environment* 319:114645.** Per-built-up-pixel counterfactual
   business-as-usual NTL trajectories for ten US hurricanes via Bayesian
   change detection; recovery = time to rebound to BAU. (Author list beyond
   the lead not fully verified — publisher blocked the fetch.) **Why:** the
   granularity precedent — counterfactuals per pixel; but its counterfactual
   is extrapolated own-history, not a donor combination, so it cannot absorb
   region-wide shocks the way SC/DiD do. Useful as the contrast that
   motivates donors.
5. **Xu, Dimasaka, Wald & Noh 2022, "Seismic multi-hazard and impact
   estimation via causal inference from satellite imagery," *Nature
   Communications* 13:7793.** Causal Bayesian network over ground shaking →
   landslide/liquefaction → building damage, inferred variationally from
   SAR damage-proxy maps (ALOS-2, Sentinel-1, COSMO-SkyMed) for four
   earthquakes; hurricane extension published as a SIGSPATIAL 2023 workshop
   paper (DOI 10.1145/3615884.3629422). **Why:** the most-cited
   "causal + satellite + landslides" paper, and a *different problem* — they
   disentangle which hazard caused an observed pixel change (structural
   model); we estimate how a site differs from its no-disaster counterfactual
   (design-based). One sentence drawing that line belongs in any related-work
   section, because reviewers will surface this paper.
6. **Coffman & Noy 2012, "Hurricane Iniki: measuring the long-term economic
   impact of a natural disaster using synthetic control," *Environment and
   Development Economics* 17(2):187–205.** Kauai vs a synthetic Kauai from
   unaffected Hawaiian islands; population ~12% below counterfactual after
   18 years. No imagery. **Why:** establishes "SC is the right tool for a
   hurricane" in economics; we cite it for identification and Serra-Burriel
   for implementation.
7. **Andam, Ferraro, Pfaff, Sanchez-Azofeifa & Robalino 2008, "Measuring the
   effectiveness of protected area networks in reducing deforestation,"
   *PNAS* 105(42):16089–94.** The canonical matched-pixel impact evaluation:
   matching on terrain/land-cover observables, explicit spillover
   measurement, Rosenbaum sensitivity analysis; matched estimates ~10× smaller
   than naive ones. **Why:** the direct ancestor of the collaborator's
   matched design; its three moves (match on terrain, state a separation
   rule and test spillover, report hidden-bias sensitivity) all transfer.
8. **Sills et al. 2015, "Estimating the impacts of local policy innovation:
   the synthetic control method applied to tropical deforestation," *PLOS
   ONE* 10(7):e0132590.** First prominent Abadie-style SC on a
   satellite-derived outcome (Paragominas deforestation vs synthetic
   municipality). **Why:** the precedent for SC-on-a-satellite-outcome, and
   its differences (one treated unit, long annual pre-period, scalar
   outcome) delineate what our many-treated, short-pre, image-valued setting
   adds.
9. **Rambachan, Singh & Viviano 2024, "Program evaluation with remotely
   sensed outcomes," arXiv:2411.10959.** Identification when the image is a
   *post-outcome* proxy (the outcome causes the image); shows naive
   predict-then-regress is biased; provides diagnostics, including an
   overidentification-style test across image representations. **Why:**
   landslide damage causes the Sentinel signature, so this is our setting's
   theory paper — and their representation-consistency diagnostic is the
   published version of our H2: if the effect changes when the embedding
   changes, that is a specification alarm, not a footnote.
10. **Jerzak, Johansson & Daoud 2022/2023 — "Estimating causal effects under
    image confounding bias…" (arXiv:2206.06410) and "Image-based treatment
    effect heterogeneity" (CLeaR 2023, PMLR 213:531–552; `causalimages` R
    package).** Identification when confounders live in imagery; deep model
    for image-driven effect heterogeneity, applied to development RCTs.
    **Why:** the theoretical license for using pre-event chips (and their
    embeddings) as the conditioning/matching object, plus the reviewer
    vocabulary for image-chips-as-causal-data.
11. **Takahata, Suetsugu, Fukaya & Shirota 2024, "Bayesian state-space
    synthetic control … for forest carbon credits," *Environmental Data
    Science* (DOI 10.1017/eds.2024.5).** SC weighting + state-space temporal
    model + regularized horseshoe prior for small donor pools, with
    calibrated intervals. **Why:** the inference machinery matched to our
    weakness — short pre-period, many candidate donors, need for honest
    uncertainty on the SC gap.

Context shelf (not intersection; cite only to justify data choices): the
Helene record — Burgi et al. 2025 (the inventory, USGS data release,
DOI 10.5066/P14CHGKS), Schaefer et al. 2025 (*GSA Today* — the inventory
paper; S2 was the sole complete pre/post source, windows nearly identical to
ours), Scheip et al. 2026 (*ESPL*, post-Helene lidar), Sutton & Stanley 2026
(*Weather and Forecasting*, LHASA precipitation-forecast validation against
the inventory — also the printed source for the inventory's limitations);
the rapid-mapping canon — Mondini et al. 2019 (*Remote Sensing*, S1
amplitude detects ~84% of rapid landslides but rarely delineates them),
Handwerger et al. 2022 (*NHESS*, S1 backscatter-change heatmaps in GEE — the
method behind NASA's operational Helene proxy maps, which were never
validated in print: an open door), Milledge et al. 2022 (*NHESS*, ALDI —
optical/Landsat NDVI differencing), Burrows et al. 2020 (*NHESS*, InSAR
coherence methods); measurement caveats — Proctor, Carleton & Sum 2023/2026
(NBER w30861 / *JAERE*, satellite retrieval error biases regression
estimates; multiple imputation fixes it); foundation models for the
embedding arm — Prithvi-EO-2.0 (arXiv:2412.02732, HLS-trained, temporal
embeddings, but 30 m and no SAR), SatMAE (NeurIPS 2022, band-group +
temporal masking), MOSAIKS (Rolf et al., *Nature Communications* 2021 —
cheap fixed random features; the natural second representation for the
Rambachan-style consistency check), Clay v1.5 (open weights, the only one
pretrained on S1 *and* S2 — but no peer-reviewed paper: cite as software and
validate ourselves). One naming trap: **CICRL-FLM** (*GIScience & Remote
Sensing* 2025, DOI 10.1080/15481603.2025.2598078) sounds like a scoop —
"counterfactual… causal… landslide mapping" — but appears to be
counterfactual representation learning inside a segmentation network, i.e.
mapping accuracy, not effect estimation. Unverified beyond title/DOI
(publisher blocked); check before citing, and expect to distinguish from it.

## 8. Reading list — in order, with the specific sections

Assumes the preliminary tutorial's list (Abadie JEL 2021 survey; Cunningham
Mixtape ch. 10; ADH JASA 2010 §2; Abadie & L'Hour JASA 2021; Ferman & Pinto;
Amjad/Shah/Shen JMLR 2018; Athey et al. JASA 2021; Doudchenko & Imbens 2016;
Boyd & Vandenberghe §4.4/§5.5.3; Iordache et al. TGRS 2011) is already in
progress. New, satellite-specific:

1. **Serra-Burriel et al. 2021** (arXiv:2012.05140) — read in full; it is
   short. Note the donor-selection-by-spectral-similarity section and the
   pre-period fit diagnostics — both to be mirrored.
2. **Fick et al. 2021, *Ecological Applications*** — §Methods for the SC
   adaptation to RS time series; the simulation section for donor-count
   sensitivity. The operational template.
3. **Ben-Michael, Feller & Rothstein 2021, "The augmented synthetic control
   method," JASA 116(536):1789–1803** — §2–3. The proposal cites it; it is
   the fix for imperfect pre-fit, which one composite pre-period guarantees.
4. **Rambachan et al. 2024** (arXiv:2411.10959) — §1–3 and the diagnostics
   section. The post-outcome-proxy framing is our exact setting.
5. **Andam et al. 2008, PNAS** — skim for the three design moves (matching,
   spillover, sensitivity). Then read the collaborator's
   `site_matching_table.csv` columns again; the correspondence is 1:1.
6. **Schaefer et al. 2025, GSA Today** — the inventory's construction,
   which is our treatment-assignment mechanism. Read with §6's circularity
   note in mind.
7. **Mondini et al. 2019 + Handwerger et al. 2022** — enough of each to
   know what S1 backscatter can and cannot see in landslide terrain (VV/VH
   both rise and fall depending on land cover; detection ≠ delineation).
8. **Liu, Wang & Xu 2024, AJPS** — the diagnostics umbrella (placebo,
   equivalence tests) for the causal phase, as in the preliminary list.
9. **Geirhos et al. 2020, *Nature Machine Intelligence*** — before trusting
   any foundation-model embedding: the shortcut-learning failure mode the
   H2-analog test must be designed to catch.
10. **Chernozhukov et al. 2018, *Econometrics Journal* (DML)** — §1–2 only,
    for the covariate-adjustment layer the proposal's Module 2 promises.

Suggested order for a two-week pass: 1 → 2 → 5 → 3 → 4 → 6 → 7 → the rest
as needed. After 1–4, re-read §5 of this document and every proposal
sentence should map to a concrete estimator and a concrete file in
`REAP/data/`.
