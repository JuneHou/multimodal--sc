# Progress Log

## Session: 2026-08-13

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-08-13
- Actions taken:
  - Read the 2026-08-12 MIMIC-CXR discussion from the prior transcript
  - Read FINAL_REPORT, n2c2 one-pager, trial dataset decision, causal-direction notes, REAP satellite tutorial, literature outline
- Files created/modified:
  - none yet

### Phase 2: Planning & Structure
- **Status:** complete
- Actions taken:
  - Split clinical potential into claim rungs C0–C4
  - Assigned corpora; kept MIMIC out of the Helene-twin slot
  - Ranked next experiments against REAP-as-primary

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Wrote findings.md and first canvas (claim ladder / three tracks)
  - User reframed: not prediction; Ŷ(0) at a treatment decision
  - Rewrote canvas around estimand, decision points, and hosts of untreated donors

### Phase 6: Two equal case studies
- **Status:** in progress
- Actions taken:
  - User locked Helene + clinical trial as equal, same pipeline, same model, supporting the image FM; n2c2 out of the pair
  - Mapped CALS five-stage pipeline onto I-SPY 2 (only public two-arm serial-imaging trial)
  - Canvas: `two-case-studies-same-pipeline.canvas.tsx`
  - Updated findings.md and task_plan.md with the lock

### Phase 7: I-SPY 2 validation and paper plan
- **Status:** complete (validation); experiment not started
- Actions taken:
  - Verified TCIA access, 985/384/210-control numbers, T0–T3 schema, Li 2020 occupancy, pembrolizumab ATE, circularity parallel to Helene NDVI
  - Planned week-1 clone of Helene notebook 02 on the 384-pt multi-feature table; latent MRI after
  - Wrote canvas `two-case-studies-same-pipeline.canvas.tsx`; findings addendum on I-SPY validation
