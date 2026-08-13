# Findings — clinical direction potential

## Requirements
- Start from the 2026-08-12 discussion: MIMIC-CXR does not fix the ~3-visit problem.
- Explore clinical potential of latent synthetic control without reopening the MIMIC family as the Helene twin.
- Be claim-honest: geometry ≠ longitudinal counterfactual ≠ virtual control arm.

## Locked constraints (do not reopen)
- Jun 2026-08-11: MIMIC family excluded on all time axes (admissions, notes, imaging).
- n2c2 2018 Track 1 is the latent-recovery corpus in hand (288 patients, mean 4.39 notes, median 3.9 years).
- CheXpert Plus is the designated later multimodal / medical-image corpus (mean 2.9 studies).
- Advisor selected the satellite (REAP / Helene) track as the next phase; clinical preliminary is concluded as a methods battery.
- NACC-UDS / I-CONECT remains the treated-panel option; I-CONECT has a data-transfer fee up to $2,500 and is a PI request.

## Research Findings

### Panel depth already measured
- MIMIC-III mortality timeline (this project's rebuild): 6,112 patients, 9,582 admissions; 749 with ≥3 visits (T0=1); 348 with ≥4 (T0=2).
- n2c2: 287/288 (99.7%) have ≥3 notes; 274/288 (95.1%) have ≥4; mean 4.39 notes; 1,264 records.
- MIMIC-CXR: ~65k patients, ~228k studies, mean ~3.5 studies/patient, typically inside one ICU stay.
- CheXpert Plus: 64,725 patients, 187,711 studies, mean 2.9.
- NCT00174655: 977 patients, mean 9.31 visits — deep but sticky code bags; LVCF F1 0.774 beat SC; treatment near-degenerate; no real prose in the demo release.
- NACC-UDS: >54k participants, up to 20 annual visits; Wu et al. already ran matching-based SC on cognitive scores. I-CONECT is the treated arm.

### Earned clinical results (n2c2 battery, 2026-08-11)
- H1 dominant-weight: planted 0.9 recovered as 0.884 (error 0.016). Oracle/m=all exact + LP-unique.
- H1 diffuse: mean L1 0.452, retrieval-limited (2.66/5 true donors in m=50).
- H2 refuted: bag-vs-latent weight cosine inside permutation null IQR. Representation is load-bearing.
- H3 confounded (decoder copies); H4 refuted (decoder switches). Stand-in decoder, not the latent.
- Digit-string NCT run was ruled invalid; it remains the falsifiability proof.

### What MIMIC would actually buy
- Films + reports, mortality bit (prevalence 0.12 at next visit), scale at T0=1.
- Not a long pre-path. Not clean Y(0) donors (other ICU patients). Intersection of ≥3 admissions AND linked films is smaller than 749.
- Legal as an image-embedding H1 analog; weak as a clinical SC case study next to Helene.

### Competitive gap (dataset-agnostic)
- No published per-patient synthetic control from clinical text.
- SyncTwin (NeurIPS 2021): closest EHR SC relative; structured-only (CPRD).
- DT-GPT: trajectory forecast, no donor/convex structure; serializes structured fields into templates, does not consume narratives.
- Unlearn / PROCOVA: twin as ANCOVA covariate inside a still-randomized trial — not a virtual control arm.
- Wu et al. 2025: NACC matching SC on cognitive scores; not latent/text/image.
- Planted-mixture weight recovery in learned embeddings as an SC validity check: no direct precedent (nearest: hyperspectral unmixing).

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Four claim rungs, not one "clinical case study" | Mixing geometry, phenotype forecast, treated counterfactual, and virtual-control claims is how this project overclaims |
| n2c2 owns C0–C1 | In hand, deep notes, ADVANCED-CAD; no treatment, no images |
| CheXpert Plus owns C2 | Medical image to match Sentinel; thin time, no treatment |
| NACC/I-CONECT owns C3 | Only treated clinical panel with a long pre-path |
| MIMIC owns none of C1–C3 as the Helene twin | Thin ICU panel; donors are not clean Y(0) |
| C4 (registration-grade virtual control) is out of scope | PROCOVA lesson: simulation-as-control is not the design |

## Claim ladder
| Rung | Claim | Needs | Corpus | Status |
|------|-------|-------|--------|--------|
| C0 | Embeddings are a legal SC input (weights identifiable) | planted mixtures | n2c2 | earned |
| C1 | Latent SC forecasts a held-out phenotype / next note | panel + label, placebo | n2c2 | doable, not run |
| C2 | Image embeddings are a legal SC input | films + encoder | CheXpert Plus | designated, not run |
| C3 | Longitudinal clinical counterfactual under a real treatment | long pre-path + untreated donors + known T0 | NACC / I-CONECT | not in hand |
| C4 | Virtual control arm for a pivotal trial | randomization or regulator design | none of ours | out of scope |

## Three-track clinical program
1. **Methods (n2c2).** Paper 1 = battery already run. Optional upgrade = C1 placebo on ADVANCED-CAD / next note. Paper 2 = learned retrieval + latent + decoder, evaluated by the same battery.
2. **Image geometry (CheXpert Plus).** H1 analog on CXR embeddings. Transfer test for the satellite encoder story. Not a disaster-style causal case.
3. **Treated panel (NACC / I-CONECT).** The only Helene-shaped clinical study. Budget/time decision against REAP, not a dataset swap.

## Ranked next moves (clinical, assuming REAP stays primary)
1. n2c2 C1 placebo-Y (fit on notes t=0,1; score held-out note or ADVANCED-CAD) — upgrades paper 1 without new data.
2. Write paper 1 from the existing archive.
3. Learned retrieval (paper 2) — the actual method contribution; also serves REAP donor quality.
4. CheXpert H1 analog when an image encoder is in the satellite pipeline.
5. NACC request only if the advisor wants a treated clinical case alongside Helene.
6. Do not: MIMIC-CXR as Helene twin; n2c2+CXR join; virtual-control-arm claims.

## 2026-08-13 reframing — Ŷ(0) at a decision, not prediction
User rejected C1 as the clinical direction. The target is: at time t, what would this patient’s trajectory be if we withhold treatment — REAP’s Y(0).

Implications:
- n2c2, CheXpert, MIMIC-CXR, and NCT00174655 (comparator-only, 96% same regimen) cannot host this. NCT already demonstrated the failure mode: SC collapsed to next-visit forecasting and lost to LVCF.
- Placebo on untreated patients remains validation of the Y(0) machine, not the product.
- Decision points: eligibility (matching only); initiation T0 (Helene twin); intensification (new T0 or sequential methods); retrospective impact (REAP product); next-visit forecast (out).
- Choice (start vs withhold) needs Ŷ(0) and Ŷ(1). Classical SC supplies Ŷ(0). Stay at one T0 to stay parallel to REAP.
- Clinical Helene twin = I-CONECT treated arm + NACC untreated donors (Wu et al. already did score-space SC). Increment = learned latent, not a prediction head.
- Two-arm PDS trial is an optional randomized calibration path; NCT00003299 has both arms but AE tables only.

## 2026-08-13 lock — two equal case studies (same pipeline, same model)
User: a clinical **trial** experiment runs **in parallel** with Helene, same importance, same five-stage pipeline, same model, both supporting the foundation-model work. Forget n2c2 for this pair.

Shared objects (from CALS_ORI_Proposal_CausalAI.pdf):
1. Detect T0 → 2. factual Y^F from imagery → 3. Ŷ(0) via simplex baseline then learned/latent imputation → 4. τ = Y^F − Ŷ(0) → 5. decision support.

| | Helene / REAP | Clinical trial |
|---|---|---|
| Role | No-RCT deployment | RCT calibration |
| Treatment | Shock hits this unit | Tested medicine vs not |
| Donors after T0 | Unhit plots still imaged | Control arm still imaged, no tested medicine |
| Y | Sentinel chip → indices + embedding | MRI → volume / embedding |
| Proof average not invented | Placebo + later USDA | Mean τ_i ≈ RCT ATE |

**Host that can actually share the image model:** I-SPY 2 (TCIA). Serial DCE-MRI at trial T0–T3; experimental vs standard paclitaxel→AC; public codebook is arm/HR/HER2/pCR/FTV, **not notes**. Estimand is withhold the **add-on**, not withhold chemo.

**Cannot be the equal case:** n2c2 (no image, no switch); CheXpert (no arms); PRO-ACT/PDS tables (arms, no images); NACC/I-CONECT (scores; Wu already); MIMIC (locked).

**Same model caveat:** one SC/imputation head and validation suite; domain vision encoder or adapter is allowed. A single pretrained net on Sentinel and breast MRI is not implied.

## 2026-08-13 — I-SPY 2 validation for the Helene-parallel paper

**Verdict: yes, with named caveats.** It is the only public two-arm serial-imaging trial that can run the CALS five-stage pipeline beside Helene. Occupied literature is pCR *prediction* from MRI (Li et al. 2020) and Bayesian adaptive allocation using FTV. Unoccupied: per-patient simplex / latent \(\hat Y_i(0)\) from control-arm image trajectories.

### What is actually public
- TCIA ISPY2 DOI 10.7937/TCIA.D8Z0-9T85, CC BY 4.0 (switched from BY-NC 2022-07-08). Free TCIA account; no I-SPY DAPC for this dump.
- Imaging Cohort 1: **985** patients (719 ISPY2 + 266 ACRIN-6698). Download the combined manifest or the 719 set is not comparable to published trial results.
- ~95% of DCE studies pass FTV QC; 719-pt collection: 2688 studies ≈ **3.7 MRI visits / patient** (schema is 4: T0–T3).
- Clinical CSV ~57 kB: demographics, HR, HER2, pCR, arm — not notes.
- Multi-feature xlsx **384 patients / 130 kB**: FTV, longest diameter, sphericity, contralateral BPE at T0, T1, T2, T3 (Li et al., *npj Breast Cancer* 2020, doi 10.1038/s41523-020-00203-7). This is the I-SPY analog of Helene `site_band_features.csv`.
- Full DICOMs 1.6 TB (719) or 2.4 TB (985), plus derived PE/SER maps and FTV masks. BreastDCEDL_ISPY2 (54 GB NIfTI) is **pretreatment T0 only** — usable for encoder H1, **not** for post-drug \(\hat Y(0)\).
- GEO I-SPY2-990 (GSE194040): 987 pts, **210 control**; experimental e.g. pembro 69, VC 71, neratinib 114. Pembrolizumab published ATE (Nanda et al. JAMA Oncol 2020): pCR 44% vs 17% (ERBB2−), 60% vs 22% (TNBC). I-SPY uses **contemporary controls** (all controls until that arm closes), not only concurrent.

### Identification (maps onto Helene tutorial §1–3, §6)
| Object | Helene (in hand) | I-SPY 2 |
|---|---|---|
| Unit | 1 km chip; 26 treated + 260 controls | Patient; ~210 control, experimental per arm ~50–130 |
| Treatment | USGS-mapped landslide | Tested add-on vs standard paclitaxel→AC |
| \(\hat Y(0)\) | Path if landslide missed this plot | Path if add-on withheld (still on chemo) |
| Pre | One median composite (2024-08-15→09-23) | One pre-drug MRI (trial T0) |
| Post | One composite (Oct 2024) | T1 (~3 wk), T2 (inter-regimen), T3 (pre-surgery) |
| Donor after T0 | Unhit plots, same storm | Control arm, no tested medicine |
| High-dim pre identifies \(w\) | Chip ≈ 112k numbers, 10 donors | DCE volume / 4 features / embedding vs ~210 controls |
| Matching analog of NLCD | Land cover, elevation, slope | HR, HER2, MammaPrint signature |
| Circularity | Inventory built from S2 NDVI | Platform adapts on FTV; Li predicts pCR from FTV | Report non-NDVI / S1; report non-FTV (sphericity, BPE, embeddings) |
| SUTVA | Controls share the storm | Controls share NAC; \(\tau\) is add-on, not “no cancer treatment” |
| Adaptive randomization | n/a | Assignment probability depends on signature and accumulating FTV — stratify donors on HR/HER2/MP; do not pool all 9 arms vs all controls as if 1:1 RCT |

One pre-image is the **same** Helene design (tutorial §3), not a new weakness. I-SPY is **stronger on post-period**: three post visits vs Helene’s one. Helene is stronger as the no-RCT product. Complementary n: Helene few-treated; I-SPY many-treated generalized SC (Serra-Burriel 2021 analog).

### What must not be claimed
- Notes / VLM: public I-SPY has none.
- Replacing the trial ATE: mean \(\tau_i\) should *sit near* published pCR/FTV contrasts; the paper’s object is unit-level \(\hat Y_i(0)\) and \(w_i\).
- BreastDCEDL as the serial panel.
- Uniform control mean as the method (that is the RCT).

### Paper use (recommended)
1. **Week 1, feature space:** 384-pt xlsx + 985-pt CSV. Clone Helene notebook 02: DiD-on-matches vs simplex SC on (FTV, LD, sphericity, BPE); donors = control arm, stratified by HR/HER2; placebo among controls; one graduated arm (pembrolizumab) as the known-ATE check.
2. **Shared method figure:** same five stages, two columns.
3. **Latent track (equal to Helene embeddings):** encode T0 DCE (cropped ipsilateral series already in TCIA); plant mixtures of control embeddings (H1); fit \(w\) on T0 embedding; carry to T1–T3. Domain encoder (MRI), shared QP head.
4. **Do not download 2.4 TB until (1) says the panel and arm labels support SC.** Derived FTV masks / PE maps are the next MRI increment, not raw bilateral stacks.

## 2026-08-13 — I-CONECT + NACC access (verified)
- **Possible:** Wu et al., Alz & Dem 2025 (doi 10.1002/alz.70460) already ran this pairing. Code: github.com/chao-yi-wu/synthetic_control. Files named: investigator_nacc64.csv, iconect_long.csv, iconect_wide.csv.
- **NACC:** Free worldwide (naccdata.org). 15 min request + DUA. Ack 3 business days; data 48 h. 56k+ participants, 217k+ visits, **mean 3.7** (range 1–20). Wu freeze: 50,259; 5,132 I-CONECT-eligible. Must distinguish hypothesis from Wu. Manuscripts need NACC admin review before journal submission. UDSv3 retired 2026-03-01.
- **I-CONECT:** Not deposited. Live page i-conect.org/request-data → Layton ADRC form + consult. Contact on Wu README: hdodge@mgh.harvard.edu. Lab form read 2026-08-10: cognitive/clinical transfer **up to $2,500**; audio/video up to $3,000 + PHI DUA 30–90 days. Public page currently lists no price. Skip media for this estimand.
- **Trial size:** Target 320; 186 randomized. Wu clean samples: MoCA treated n=12 (MCI, in-person throughout); CFA treated n=31 (normal cognition). Linear interpolation of NACC annual visits to 6-month endpoints is a known artifact.
- **Scientific occupancy:** Dataset pairing is taken. Increment must be the estimator (simplex latent SC), not the join.
