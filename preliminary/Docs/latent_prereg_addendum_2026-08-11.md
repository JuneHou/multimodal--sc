# Pre-registration addendum — dataset swap to n2c2 (2026-08-11)

Amends [latent_prereg_2026-08-11.md](latent_prereg_2026-08-11.md), written
before the L0–L2 rerun on the new dataset. The original prereg's run on
NCT00174655 was **ruled invalid by Jun** (inputs were opaque digit strings,
violating the textual-input requirement); this addendum changes the dataset
and the affected operational details. **Hypotheses H1–H4, all thresholds,
seeds, solver, bootstrap settings, and interpretation rules are unchanged.**

## Changed: data and setup

- **Dataset**: n2c2 2018 Track 1 / 2014 i2b2 longitudinal corpus — 288
  patients, 1,264 records, verified real prose (`latent/results/
  n2c2_verify.json`). Held by Jun under their existing n2c2 registration;
  gitignored at `latent/data/n2c2/`; no redistribution.
- **Artifact**: the record text itself — no rendering step (L0 becomes
  parsing). Loader rules fixed by verification: split on `^Record date:` at
  line start; **sort records by parsed date** (train 112, 344 are
  non-chronological in file order); t = 0-based index after sorting.
- **Encoders unchanged** (Qwen3-Embedding-8B → MRL-768 primary + 4096-d
  stored; BioLORD-2023). Operational change: records reach 2,985 tokens, so
  **BioLORD requires chunk+mean-pool above 512 tokens** (Phase-1 pattern);
  Qwen3's 32k window needs none. Concept-order canonicalization does not
  apply (natural prose is embedded as-is).
- **Corpus scale**: 1,264 record latents (was 9,092). Consequences:
  `m=all` = 1,263 donors is now **feasible and replaces the m=500 fallback**
  from the invalid run; semi-synthetic N stays 500/cell.
- **Severity labels for H4**: `death` does not exist here. Primary severity
  signal = **ADVANCED-CAD (met 170 / not-met 118, balanced)**; secondary =
  MI-6MOS (26/262). KETO-1YR is degenerate (1/287) and excluded. The
  test-split ADVANCED-CAD keeps the organizers' uncorrected annotations for
  files 140/156/205/266/277 — noted, not repaired.
- **H2 bag space**: n2c2 ships no structured codes. The observable space is
  **UMLS concept bags** extracted per record with QuickUMLS over Jun's UTS
  Metathesaurus download (pending), filtered to disorder/drug/procedure/
  finding semantic types, min-df ≥ 5, deterministic index. **Pre-registered
  caveat: no negation handling in v1** — negated concepts enter the bag;
  if H2 fails, a NegEx-filtered v2 is the one permitted follow-up before
  interpreting. H2 cannot run until the UMLS download lands; H1 (the
  load-bearing claim) does not depend on it.
- **Splits**: experiments use all 288 patients (the train/test split serves
  the criteria-classification task, not ours; no model is being trained).

## Unchanged (restated for clarity)

H1 semi-synthetic cells (K ∈ {2,5,10}; Dirichlet + spiky-0.9; σ ∈ {0,
residual-matched}; pools now {oracle-K, m=50, m=all=1,263}); H1 thresholds
and the oracle hard gate; H2 permutation-null criteria; H3/H4 gates (score()
validation first; generation caps); encoder-selection rule; the rule that
H1 failing in both arms demotes H2–H4 to exploratory.
