# Task Plan: Clinical direction potential

## Goal
Run two equal case studies on the CALS ORI five-stage pipeline and the same causal model (image encoder + simplex baseline + latent Ŷ(0)): Helene / REAP and a two-arm imaging trial (I-SPY 2).

## Next Step
Download TCIA I-SPY2 clinical CSV (985) + multi-feature xlsx (384); clone Helene `02_did_vs_synthetic_control.ipynb` onto FTV/LD/sphericity/BPE with control-arm donors stratified by HR/HER2. Do not pull 2.4 TB DICOMs until that panel is verified.

## Current Phase
Phase 7 — I-SPY 2 validated; feature-space parallel experiment

## Phases

### Phase 1: Requirements & Discovery
- [x] Restore the 2026-08-12 MIMIC-CXR discussion as the starting constraint
- [x] Read n2c2, trial-dataset, causal-direction, final-report, and REAP tutorial docs
- [x] Document locked rulings and earned results in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Separate claim types (geometry vs placebo-Y vs treated counterfactual vs virtual control)
- [x] Assign corpora to claims (n2c2, CheXpert Plus, NACC/I-CONECT; MIMIC out)
- [x] Rank next experiments by ROI against REAP as primary
- **Status:** complete

### Phase 3: Implementation
- [x] Write findings.md with the claim ladder and corpus map
- [x] Produce a canvas of the clinical program
- [x] Reframe canvas around Ŷ(0) at a treatment decision (user rejected prediction)
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Canvas uses only in-hand numbers (no invented CheXpert tails or NACC means)
- [x] Chat answer does not reopen the MIMIC family as a Helene twin
- [x] C1 next-note / ADVANCED-CAD forecast demoted; not the clinical product
- **Status:** complete

### Phase 5: Delivery
- [x] User can open the canvas beside chat
- [x] User-facing estimand answer for the Ŷ(0)-at-decision reframing
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Do not switch the clinical case study to MIMIC | Same thin panel already measured (749 at T0=1, 348 at T0=2); CXR does not lengthen it; Jun 2026-08-11 ruling still stands |
| Keep n2c2 for notes + labels | Only in-hand corpus with near-universal ≥3 notes over ~4 years and a usable phenotype (ADVANCED-CAD) |
| CheXpert Plus is image-geometry only | Mean 2.9 studies; no treatment; legal H1 analog, not a Helene twin |
| Treated clinical panel remains NACC / I-CONECT | Only path with a real T0 and untreated donors; access cost is the blocker |
| Clinical is a parallel methods program, not a REAP replacement | Advisor already selected satellite; clinical value is validation + an unoccupied text-SC niche |
| Clinical product is Ŷ(0) at a treatment decision, not prediction | User 2026-08-13: same as REAP non-disaster counterfactual; n2c2/NCT/CheXpert cannot host it |
| I-CONECT + NACC is the clinical Helene twin | Only in-scope corpus with a real intervention, known T0, untreated donors, and a long pre-path |
| Two equal case studies, same pipeline, same model | User 2026-08-13: Helene + clinical trial in parallel, both support the image foundation-model work; n2c2 out of this pair |
| Clinical trial host for the image pipeline is I-SPY 2 | Only public two-arm trial with serial images; control arm donates post-T0 Y(0) for the tested add-on |
| Same model = shared stages + SC head, not one backbone | Sentinel and breast MRI need domain encoders; PRO-ACT/PDS tables cannot run this pipeline |
| I-SPY 2 is validated as the Helene-parallel paper case | Serial DCE T0–T3, control arm Y(0) for the add-on, 384-pt feature table = Helene band-features; Li 2020 occupies pCR prediction not SC |
| Start with 384-pt xlsx, not 2.4 TB | Same as Helene notebook 02 (features first, embeddings later); BreastDCEDL is T0-only |
| Analyze one graduated arm vs contemporary controls | Adaptive platform: do not treat 9 arms vs 210 controls as a simple RCT; pembrolizumab has a published pCR ATE |

## Errors Encountered
| Error | Resolution |
|-------|------------|
