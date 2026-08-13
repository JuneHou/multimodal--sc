# FINAL REPORT — Latent recovery for synthetic control on clinical text

2026-08-11. **This is both the final report of the preliminary study and the
meeting report** — a single self-contained document. Every number here is
reproduced from artifacts on disk (`latent/results/`); the other Docs files
are appendices, indexed at the end.

> **WRAP-UP (2026-08-11):** the clinical preliminary is concluded. The
> advisor selected the satellite dataset; the next phase moves to the repo
> top level (reserved for it since the project's first day). Cleanup:
> re-downloadable/regenerable artifacts removed (~500 MB — invalid-run
> embeddings, QuickUMLS venv, reference clones, PDS/NACC placeholders);
> retained: the n2c2 data (Jun's DUA copy — irreplaceable while DBMI
> registration is frozen), all results/figures/code, and this Docs set —
> everything needed to write paper 1 from the archive.

## TL;DR

We tested whether synthetic control's causal weights survive a learned
embedding of clinical text. Pre-registered, four hypotheses, real data
(n2c2, 288 patients, 1,264 clinical narratives). Result: **the weights are
identifiable in the text-latent space — a planted 0.9 mixing weight is
recovered as 0.884, and with the right donors present, recovery is exact
with an LP-certified unique solution.** Every observed failure localizes to
an off-the-shelf component: donor *retrieval* (finds only 2.7 of 5 true
donors), *representation choice* (concept-code space and text space
disagree about patient similarity at chance level), and the *stand-in
decoder* (copies rather than blends). A prior-art sweep found no direct
precedent for the validation method itself. Together: the method's core is
sound and the open problems are exactly the components the proposed
contribution would learn.

## 1. Research question

Advisor's direction: *"we are taking a specific causal method [synthetic
control] to help us build a causal model [the donor weights], which can
then help us develop or impute the counterfactual."* The weights ARE the
causal model — "this patient ≈ 0.9 × patient A + 0.1 × patient B" — and
the counterfactual is imputed by applying that recipe to donors' later
records. Jun's operational question (the advisor's 0.9 example): **if the
true mixing weight is 0.9, does the method recover ≈ 0.9 — in a latent
space learned from clinical text?** If yes, fitted weights on real patients
carry interpretation and causal weight; if no, they are decoration on a
black-box predictor.

## 2. Design — the pre-registered battery

Hypotheses, thresholds, seeds, and interpretation rules were frozen in
writing before any run (prereg + addendum, see index). The four tests:

- **H1 — weight identifiability (the core).** Manufacture pseudo-patients
  as known convex mixtures of real records' embeddings (weights planted by
  us — a diffuse random pattern and a dominant 0.9/0.1 pattern), hand the solver only the
  blended vector and a candidate pool, score recovery of both the donor
  identities and the weights. Cells: K ∈ {2,5,10} donors × 2 weight shapes
  × 2 noise levels × 3 pool types (oracle / retrieval-50 / all-1,263) × 2
  encoders, 500 replicates each, seed 0. Planting is the only way to have
  ground truth — real patients' true weights are unobservable.
- **H2 — cross-representation consistency.** Real patients: fit weights on
  the same 50 donors twice — in UMLS-concept-bag space and in text-latent
  space — and test agreement against a permutation null.
- **H3 — generative sensibility.** Decode mixtures to text; require normal
  perplexity, cycle-consistency (re-embedded text lands near the mixed
  vector), and similar-mixtures → similar-texts.
- **H4 — semantic interpolation.** Blend a severe patient (ADVANCED-CAD
  met) with a similar non-severe one at α ∈ {0…1}; require decoded severity
  monotone in α.

Anti-self-deception machinery: hard oracle gate (pool = exactly the true
donors must recover near-exactly, else stop); permutation nulls; a
random-bags negative control; LLM scorer validated against an independent
implementation before any perplexity was quoted; every deviation recorded.

**Why exact recovery here is not "leakage":** there is no learned model and
no train/test split — the solver is a fixed optimization; a perfect fit
*exists* by construction, and the scientific content is *uniqueness*
(certified by an independent LP solver) and *falsifiability* (the identical
test failed on a bad representation — see §4). This is inverse-problem
methodology (compressed sensing, matrix completion, hyperspectral
unmixing), where noiseless exact recovery is the expected behavior in the
identifiable regime and the science lives at the regime's boundary.

## 3. Theory → code

| theory | code | file |
|---|---|---|
| SC estimator: min ‖x − Σw_j z_j‖² s.t. w ≥ 0, Σw = 1 | Gram-form QP, SLSQP, analytic gradients; KKT ⇒ global optimum | `mimic/src/synthctl/fit.py::solve_simplex_qp` (built and unit-tested in the MIMIC phase; reused unchanged) |
| latent state of the artifact | Qwen3-Embedding-8B → 768-d MRL (primary; picked over BioLORD by the data, against our on-record prediction) | `latent/src/encode.py`, `latent/scripts/run_n2c2_embed.py` |
| the panel | n2c2 records date-sorted per patient, t = 0,1,2,… | `latent/src/n2c2_corpus.py` |
| planted-mixture battery | replicate construction, pools, metrics | `latent/scripts/run_n2c2_weight_recovery.py` |
| observable space (H2) | QuickUMLS concept bags (Jun's prebuilt index), NegEx v2 | `latent/src/n2c2_concepts.py`, `latent/scripts/run_n2c2_h2.py` |
| decoder stand-in (H3/H4) | prompt-conditioned Qwen2.5-7B, temp 0; perplexity via the vendored `score()` (validated first — gate PASSED, off-by-one absent) | `latent/src/genmix.py`, `latent/scripts/run_l3_l4_*.py` |
| uniqueness certificates | per-coordinate LP ranges + random-restart probe | `latent/results/uniqueness_check.json` |

## 4. What we ran (timeline, one day: 2026-08-11)

1. **Dataset selection** (after Jun excluded MIMIC entirely): candidates
   verified with real text excerpts; **n2c2 2018 Track 1** (= the 2014 i2b2
   longitudinal corpus) chosen — 288 patients, mean 4.4 records over ~4
   years each, real prose, already held under Jun's registration.
2. **The invalid first run — kept as a lesson.** The trial-data attempt
   embedded WHO/COSTART *numeric code strings* by accident; H1 refuted with
   pathological geometry (cosine concentration 0.99). Jun ruled it invalid
   (violated the textual-input design). It now serves as the battery's
   falsifiability proof: same test, bad representation, clean failure.
3. **L0–L2 on real prose**: parse (asserted against independent
   verification) → embed both arms → full H1 grid. Mid-run, Jun trimmed
   the m=all cells to diagnostic size (they are a control, not the method)
   — the N=50 subset reproduces the discarded N=500 values to 3 decimals.
4. **H2** v1, then the pre-registered NegEx v2 follow-up (negation filter
   removed 21.4% of concept mass; verdict unchanged).
5. **L3/L4** with the score() gate (passed; validated against an
   independent implementation, max diff 0.017 nats).
6. **Uniqueness check** after the prior-art sweep raised the
   solution-polytope risk.
Compute: everything on the lab box (CPU + single A40s); ~1,000 LLM
completions; the one waste was ~2.4 h of oversized control cells.

## 5. Results

**H1 — the core result.** Decision cell (K=5, m=50, σ=0, qwen3-768):

| metric | value | pre-registered bar |
|---|---|---|
| dominant 0.9: recovered ŵ_max | **0.884** (err 0.016 [CI .0155–.0166]) | support ≤ 0.10 ✔ (6× inside) |
| dominant-weight: top-1 donor found | **100%** | ≥ 90% ✔ |
| diffuse: mean L1(ŵ,w*) | 0.452 | support ≤ 0.30 ✘ (refute > 0.25 dominant-weight err / < 60% top-1: far away) |
| formal verdict | **"neither"** — dominant-weight supported, diffuse missed | |

**Where the diffuse error comes from — certified, not conjectured:**
- Oracle pool (true donors handed over): L1 ≈ **1.5e-05**. Gate passed.
- m=all pool (true donors guaranteed present among 1,263): recovery
  **exact to solver precision (mean L1 ≤ 2.6e-05, 0.0000 at 4 dp)** — and
  the uniqueness check defeated the polytope risk: the unconstrained
  solution set is 494-dimensional, but nonnegativity collapses it to a
  single LP-certified point = w* (max off-support mass ≤ 1.7e-12; all 11
  warm starts converge; target sits on the hull face spanned by its true
  donors).
- m=50 retrieval pool: only **2.66 of 5** true donors present (all 5
  present in just 1.0% of diffuse replicates; 0% in the dominant-weight family).
  The m=50 QP is strictly convex — its solution is *provably unique but
  uniquely wrong* when donors are missing.
- **⇒ uniqueness is solved by pool size; correctness is retrieval's job.**
  All residual H1 error is retrieval. (Digit-string run for contrast:
  0.88/5 donors found, oracle gate failed, space concentration 0.99 vs
  0.71 now.)

**H2 — refuted, robustly.** Bag-vs-latent weight cosine 0.425 (v1) → 0.422
after negation cleanup, inside the permutation null's IQR; top-donor
agreement 1.3× null vs the 3× bar. The two representations of the same
notes tell *different* mixing stories. Per the pre-registered
interpretation rule: a finding about representation choice, not method
failure — representation is load-bearing for causal interpretation.

**H3 — passed its thresholds but confounded; do not quote as support.**
Perplexity ratio 0.983, cycle rank 2/1,265 — but the copy diagnostic shows
the prompt-conditioned decoder copies one donor nearly verbatim (median 84%
8-gram overlap), which trivially passes both metrics. **H4 — refuted**
(severity not monotone in α; the decoder *switches* records at α≈0.75
instead of interpolating). Both indict the stand-in decoder, not the
latent: H1 already showed the mixing structure is present.

**Novelty (verified sweep):** no direct precedent for planted-mixture
weight recovery in learned embeddings as an SC validity check. Nearest:
sparse hyperspectral unmixing (same math, satellite imagery — our target
domain), one 2026 Bayesian-SC preprint (outcome panels), Arora et al. TACL
2018 (our algebra as a theorem, run forward only).

## 6. Interpretation — four arrows, one contribution

1. H1-diffuse: **retrieval** fails to surface true donors → learn retrieval.
2. H2: **fixed representations** disagree → learn the latent.
3. H3/H4: the **stand-in decoder** retrieves instead of decoding → learn
   the decoder.
4. H1-core + uniqueness: the **causal geometry itself is sound** — so
   learning those three components is warranted, not wishful.

Proposed contribution (paper 2): retrieval, latent, and decoder learned as
one system, evaluated by this battery. The battery itself, with the
certificates and the falsifiability story, is paper 1 (workshop-ready;
main-track with a second corpus + encoder sweep + a link from battery
scores to real-patient placebo performance).

## 7. Limitations (state these before anyone else does)

Single dataset (288 patients) and one encoder pair; the "0.884" is
semi-synthetic — real patients have no ground-truth weights (that is *why*
planting is the methodology); the generation stages used a stand-in
decoder; our on-record prediction (BioLORD > Qwen3 for recovery) was wrong;
encoder pretraining could in principle include i2b2-derived text (affects
embedding quality at most — cannot manufacture the mixing geometry we
impose ourselves); H2's concept extraction has known noise (extraction
F1, negation handled only in v2); n2c2's cohort is diabetic patients from
one health system.

## 8. For the meeting — suggested flow

1. **The question** (advisor's own sentence + the 0.9 example): are SC's
   causal weights real in a latent space of clinical text?
2. **The dataset story** (one-pager doc): the problem was never "clinical
   data too short" — ICU/trial panels are the wrong slice of care;
   longitudinal outpatient narratives are the right one; n2c2 in hand.
3. **The design in one slide**: plant the recipe, hide it, score recovery;
   controls: oracle gate, nulls, falsified-once-on-bad-input.
4. **The result in one slide**: 0.9 → 0.884; exact + LP-unique with donors
   present; failure = retrieval (2.7/5). H2: representations disagree ⇒
   representation is a causal modeling choice.
5. **The ask**: this motivates the learned system (paper 2) and a
   paper-1 write-up of the battery; decisions needed — I-CONECT's $2,500,
   PDS/NACC continuation, and the paper-1 target venue.
Likely questions with answers in this doc: "is 100% suspicious?" (§2
leakage note + §5 uniqueness), "why Qwen2.5-7B?" (measurement instrument,
DUA-local, validated infra), "is this novel?" (§5 novelty), "single
dataset?" (§7 + the cheap-rerun design).

## 9. Docs index — what every file is for

| file | role |
|---|---|
| **FINAL_REPORT_latent_recovery_2026-08-11.md** | **this file — final report = meeting report; start here** |
| n2c2_dataset_background_2026-08-11.md | shareable dataset one-pager (provenance, why not MIMIC, stats) — hand to the advisor/lab |
| experiment_walkthrough_2026-08-11.md | step-by-step design/implementation narrative with worked examples, written for a listener with NO causal background (co-advisor-ready); includes the hard-questions answers |
| tutorial_synthetic_control_2026-08-11.md | Jun's study guide: SC theory, the solver, reading list with sections |
| latent_prereg_2026-08-11.md + latent_prereg_addendum_2026-08-11.md | frozen pre-registration (thresholds/seeds) — cite, don't edit |
| trial_dataset_decision_2026-08-10.md | the full archive: reports 1–14 (searches, decisions, agent results, exclusions) — the audit trail behind every claim here |
| causal_direction_notes_2026-08-10.md | the earlier direction memo (advisor-sentence mapping, paper map A–F, meeting checklist) — historical context for how we got here |

Working artifacts live outside Docs: results in `latent/results/`, data in
`latent/data/` (DUA-restricted, gitignored), observable-phase work in
`observable/`, the MIMIC substrate in `mimic/`.
