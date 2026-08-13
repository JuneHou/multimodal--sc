# Causal counterfactual imputation — reading map and meeting prep

2026-08-10. Context: advisor's reply to the MIMIC-visit-length email — "we are
taking a specific causal method to help us build a causal model, which can then
help us develop or impute the counterfactual." This note fixes the vocabulary,
maps the recommended papers and how they relate, and lists what to bring to the
resumed weekly meeting.

## Vocabulary: simulation framing vs causal framing

Two ways to pose "what happens to this patient under treatment A vs B":

- **Simulation framing** (the three forwarded papers, digital twins, world
  models): build a generative model of trajectories; roll it forward under each
  action; trust the answer to the extent the model is realistic. The target is
  fidelity. Model error becomes answer error, with no design that bounds it.
- **Causal framing** (the advisor's direction): define a potential-outcomes
  estimand — e.g. the trajectory patient *i* would have had under control —
  and **impute** it with an estimator whose validity rests on stated
  identification assumptions (randomization, low-rank factor structure,
  unconfoundedness), not on the model being a good simulator. The target is
  identification.

"Causal framing" does **not** mean synthetic data. A "synthetic patient" in the
synthetic-control sense is a weighted combination of *real* donor patients —
it is the imputed counterfactual, not generated fake data.

The PROCOVA story is the cleanest illustration of why the distinction matters.
Unlearn.ai's digital twin predicts each enrolled patient's untreated
trajectory from baseline. The twin is **not** used as the control arm (that
would be simulation-as-counterfactual: any model bias flows straight into the
effect estimate, which is why regulators reject virtual control arms for
pivotal trials). Instead the twin's prediction enters as **one covariate in an
ANCOVA** on a still-randomized trial. Randomization identifies the effect; the
twin only absorbs outcome variance, so the trial needs fewer patients. If the
twin is garbage the estimate stays unbiased, just less efficient. That
subordination of the predictive model to a causal design is what made it
EMA-qualified and FDA-acceptable ("a special case of ANCOVA"). Lesson: the
mature end of the digital-twin field already learned that simulation alone is
not registration-grade; the causal machinery is what carries the claim.

**"Causal machinery," concretely** — the estimator families with explicit
identification assumptions:

1. Randomization + covariate adjustment (ANCOVA, PROCOVA).
2. **Counterfactual estimators for panel data** — fit structure on pre-period,
   impute post-period Y(0): Abadie synthetic control, interactive fixed
   effects / gsynth / fect, matrix completion, CSC-IPCA, proximal SC,
   synthetic survival control. *This is the family the advisor means and the
   one `preliminary/` already implements.*
3. g-methods for time-varying treatments: g-formula, marginal structural
   models / IPW, g-estimation; deep versions CRN and Causal Transformer.
4. Doubly robust / orthogonal learners (DR-, R-learner, TMLE) for
   heterogeneous effects.

## Paper map

### A. Synthetic control as counterfactual imputation (the home family)

| paper | what it adds | relation to us |
|---|---|---|
| [Liu, Wang & Xu 2024, AJPS](https://onlinelibrary.wiley.com/doi/full/10.1111/ajps.12723) ([fect](https://yiqingxu.org/packages/fect/)) | The umbrella term "counterfactual estimators": SC, IFEct, matrix completion unified as impute-Y(0)-then-difference. Practical diagnostics (placebo, equivalence tests). | The advisor's phrasing ("impute the counterfactual") is this literature's language. Read first; frames everything else. |
| [Synthetic Survival Control, 2025](https://arxiv.org/html/2511.14133) | SC extended to counterfactual *hazard trajectories*, multi-period staggered treatments; validated on multi-country cancer clinical data. | Closest recent "SC on patients" paper. Shows per-patient SC in health outcomes is publishable now. |
| [Counterfactual Forecasting for Panel Data, 2025](https://arxiv.org/html/2511.06189v2) | Low-rank latent-factor imputation that also *forecasts* counterfactuals for not-yet-assigned interventions; healthcare-motivated. | Their "latent factors" are linear. Our question is what happens when the state itself must be learned. |
| [CSC-IPCA, 2024](https://arxiv.org/html/2408.09271v2) | SC where the factor space is estimated by instrumented PCA from high-dimensional covariates. | The *linear* version of "latent synthetic control." The gap we'd fill is the nonlinear/learned-encoder version. |
| [Doubly robust proximal synthetic controls, Biometrics 2024](https://academic.oup.com/biometrics/article/80/2/ujae055/7685803) | Identification theory: when is the SC imputation actually the causal quantity, with proxies for latent confounding. | The assumptions section any "valid latent state" claim must engage with. |

### B. Representation learning for counterfactual trajectories (the neighbor family)

| paper | what it adds | relation to us |
|---|---|---|
| [Causal Transformer, ICML 2022](https://arxiv.org/pdf/2204.07258); [van der Schaar lab line](https://www.vanderschaar-lab.com/causal-effect-inference/) (CRN etc.) | Learn balanced representations of patient history; estimate counterfactual outcomes under treatment sequences from observational data. | Learned states, but no donor/convex structure, no interpretable "which patients compose the counterfactual," and observational-only validation. Our SC angle gives both. |

### C. Digital twins meeting causal designs (the converging field)

| paper | what it adds | relation to us |
|---|---|---|
| [Enhancing RCTs with digital twins, npj Syst Biol 2025](https://www.nature.com/articles/s41540-025-00592-0) | Argues twins-as-external-control "does not control bias"; twins-as-prognostic-covariate (PROCOVA) preserves randomization and is registration-suitable. | The field's own admission that simulation needs causal scaffolding. Positioning citation. |
| [Digital twins in AD trials, 2025](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/trc2.70181) | PROCOVA-style twins in Alzheimer's trials; the CPAD/historical-control data ecosystem. | Points at concrete long-panel trial datasets. |
| [Digital Twins as Synthetic Controls in Single-Arm Trials, 2026](https://arxiv.org/pdf/2605.12832) | The two communities explicitly merging. | Know what is already claimed before the meeting. |

**Relationships in one paragraph.** Family A owns identification but assumes
the state is given and low-dimensional (or linearly reducible, CSC-IPCA).
Family B owns learned states but gives up the donor structure, the
interpretability, and mostly the design-based validation. Family C shows the
applied field converging on "predictive model inside a causal design" and
supplies the datasets and regulatory motivation. The open middle: **learn a
nonlinear per-visit state in which convex donor imputation is valid, and
validate it against randomized ground truth** — which is also exactly the
capability the satellite-imagery project needs.

### D. Prior art on "similar patient" construction in trials (added after Q&A)

Two mature strands already build per-patient comparators on trial data; know
both, adopt the best as baselines rather than rebuilding:

| paper / system | what it does | relation to us |
|---|---|---|
| [DT-GPT, npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-02004-3) | LLM fine-tuned to forecast individual patient trajectories from EHR/trial history; benchmarked on NSCLC, ICU and Alzheimer's; beats prior ML forecasters. | Most recent general-purpose per-patient forecaster. The strongest *predictive* baseline our SC-on-latents must beat on control-arm holdout. |
| Unlearn AD-DTG-3.1 ([AD trials paper](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/trc2.70181), [Phase 2 assessment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11712969/), [medRxiv real-data methodology 2025](https://www.medrxiv.org/content/10.1101/2025.10.28.25338899.full.pdf)) | Disease-specific digital twin generator: expected value + variance of each patient's control outcome at 3-month cadence, trained on historical control arms (CPAD ecosystem). | The regulatory-grade prognostic model. Defines the accuracy bar and the AD data path. |
| External-control-arm methods: [ATT method comparison, 2024](https://arxiv.org/html/2408.07193), [PSM vs G-computation vs DDML evaluation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9795588/), [pitfalls review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539876/) | Population-level adjustment: propensity matching, Mahalanobis, MAIC reweighting, G-computation, doubly debiased ML. Recent comparisons favor G-computation / DDML over plain PSM. | The statistics literature's answer to "similar patients." Population-level ATE, not per-patient trajectories — the gap our per-patient donor imputation sits in. |

Positioning: existing work is either individual-level *prediction* (DT-GPT,
twin generators — no donor structure, no per-patient causal design) or
population-level *adjustment* (matching/weighting for an ATE). Per-patient
convex donor imputation with placebo validation is the open middle.

### E. How trials define and generate *similar patient pairs* (experiment-level)

This is the literature the preliminary similar-patient experiment should adopt
from directly. Four notions of "similar," in increasing relevance to us:

| notion of similar | canonical method | source |
|---|---|---|
| Same covariate values | Mahalanobis distance matching, with calipers; exact/coarsened exact matching | [Zhao et al., PSM with R guide](https://atm.amegroups.org/article/view/61857/html); [Austin balance diagnostics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3472075/) |
| Same probability of treatment / trial inclusion | Propensity score matching; "on-trial score" **optimal pair matching** of each trial patient to an external standard-of-care patient | [Lin et al., on-trial score](https://arxiv.org/pdf/2108.08756); [matched controls for single-arm phase II trials](https://arxiv.org/pdf/2007.15935) |
| Same predicted *untreated outcome* | **Prognostic score matching** (Hansen 2008, Biometrika): two patients are similar iff a model predicts the same Y(0) for them. Digital twins are the modern nonlinear version of this score. | Hansen 2008; PROCOVA line (section C) |
| Nearby in a learned representation | Deep patient-similarity: embed EHR history, match by embedding distance | [Deep Patient, Sci Rep 2016](https://www.nature.com/articles/srep26094); [Suo et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29994534/); [temporal similarity CNN](https://arxiv.org/pdf/1902.03376); [ConvAE stratification, npj 2020](https://www.nature.com/articles/s41746-020-0301-z) |

How this maps onto the preliminary experiment:

- **Matched-pair = our `nn1` baseline.** The entire pair-matching literature is
  the 1-sparse special case of SC. The preliminary experiment's question —
  does a convex combination beat the best single match — is exactly
  "does synthesis beat pair matching," which Phase 2 already operationalises.
- **The three score-based notions are alternative `select_donors` rules.**
  Phase 2 pre-filters donors by Euclidean distance on the pre-period block;
  Mahalanobis, propensity/on-trial score, and prognostic score are drop-in
  alternative donor-selection geometries, each with a citation the trial
  audience already trusts. Prognostic-score selection is the most aligned with
  the causal goal (similar = same expected untreated future).
- **Balance diagnostics are an expected evaluation.** Standardised mean
  differences between target and (weighted) donors is how the trial field
  audits match quality; cheap to add next to pre-period RMSE.
- Deep-similarity work shows learned-embedding matching is accepted practice,
  but none of it validates matches by *forecast* quality — our placebo/holdout
  design is the differentiator even at the preliminary stage.

### F. Dataset decision (2026-08-10) — see the dedicated doc

The starting dataset and reference papers were decided the same day; the full
record (candidate table, per-paper similarity mechanisms, PDS access terms,
dataset schema, agent-report summaries) is in
[trial_dataset_decision_2026-08-10.md](trial_dataset_decision_2026-08-10.md).
Short version: **primary = TWIN (KDD 2023) on NCT00174655 via Project Data
Sphere** (preprocessed data already downloadable; free ~48 h registration for
the full package); **secondary = Wu et al. NACC synthetic-control** (deep
20-annual-visit donor panels; application). One correction to section E's
framing: TWIN ships no pair artifact — its retrieval is batch-local,
visit-level, and recomputed per forward pass — so the adopted donor schema is
a globalized variant of its retrieval (or TWIN-GPT's cosine top-5), extracted
and validated by us; KNNSampler was rejected as too simple (weak donors risk
false-negative conclusions about synthesis).

## The proposed contribution, made concrete

"Learn the latent state in which SC-style counterfactual imputation is valid"
— implementation sketch:

1. **Encoder** φ: per-visit observations (labs, scores, codes, eventually
   text/images) → latent state z_it. Per-visit, never cumulative — Phase 1's
   no-leakage design carries over unchanged.
2. **Differentiable SC layer**: for each target, solve the simplex-constrained
   least squares on pre-period latents (unrolled projected gradient, so
   gradients flow through the solve; our SLSQP solver stays as the frozen
   evaluation-time reference).
3. **Training loss** on control-arm patients only: post-period imputation
   error of the SC forecast in latent (and/or decoded) space, plus
   reconstruction/regularization. This directly optimizes "SC transfers from
   pre to post in this space," which is the property classical SC assumes.
4. **Validity checks as the evaluation**: (i) held-out control-arm patients —
   impute their trajectory, compare to truth (the placebo design Phase 2
   already implements); (ii) treated patients — imputed untreated trajectory
   vs the randomized control arm's distribution (calibration); (iii) ablations
   vs raw-space SC, CSC-IPCA, and CRN/Causal Transformer.
5. **The "action-responsive vs nuisance factors" gap** from the email maps to
   a structured latent: factors whose perturbation changes the imputed effect
   vs factors that only aid reconstruction — an invariance-style objective,
   and the part that is genuinely novel relative to A and B.

Pipeline reuse from `preliminary/`: the whole shape transfers — timeline →
per-visit states → donor pool → convex weights → held-out-visit forecast →
placebo evaluation. What changes on trial data: donors restricted to the
control arm (so the combination estimates the *untreated* trajectory); the
time axis is scheduled visits (aligned across patients — strictly nicer than
MIMIC's irregular admissions); the encoder consumes trial measurements instead
of CCS/ATC bags. Caveat carried over from Phase 1: nearest-neighbor
"similarity" is a smell test, not the estimator — SC fits a convex
combination, and the cardinality-bias lesson says don't judge the state space
by 1-sparse matches.

## For the meeting

1. **Reply with availability first** — the concrete ask in the advisor's email
   is restarting weekly meetings.
2. **Status of the preliminary study** — now two results, both honest:
   (a) MIMIC's panel is too short (~350 patients ≥4 visits) — an argument
   about the panel, not the method; (b) first full run on NCT00174655
   (961 patients at T0=1): SC is the best donor-based forecaster and beats
   the published TWIN method on its own dataset, but **LVCF (copying the
   patient's own last visit) beats everything** — trial med/AE code sets are
   too persistent. Read: cross-patient signal is real; the state
   representation (sticky code bags) is the binding constraint; and in the
   causal design LVCF is not an admissible counterfactual estimator anyway.
   Full table: decision doc, report 6.
3. **A written estimand**: for treated patient i, the counterfactual
   trajectory under control over visits T0+1..T; effect = observed − imputed;
   method validated by control-arm hold-out calibration.
4. **Datasets — state as of 2026-08-11, one per track**:
   - *Latent-recovery track (the research question)*: **n2c2 2018 Track 1 /
     i2b2 2014 longitudinal corpus** — 288 patients, 1,264 real clinical
     narratives, mean 4.4 records over ~4 years each; already in hand under
     Jun's registration. Meeting one-pager:
     [n2c2_dataset_background_2026-08-11.md](n2c2_dataset_background_2026-08-11.md).
   - *Observable event-prediction (completed)*: NCT00174655 via Project Data
     Sphere — LVCF-wins result, archived.
   - *Later multimodal stage*: CheXpert Plus (paired image+report, 64,725
     patients). NACC-UDS remains the deep-panel causal option.
   See [trial_dataset_decision_2026-08-10.md](trial_dataset_decision_2026-08-10.md).
   Jun's actions: UMLS/UTS download (unblocks the cross-space test), PDS
   registration if the trial track continues, NACC-UDS request.
5. **The one-sentence positioning**: not a world model, not a digital twin —
   a learned state representation under which synthetic-control counterfactual
   imputation is valid, validated against randomized ground truth.
6. **Word-by-word mapping of the advisor's sentence to the proposal** — bring
   this as the alignment check:

   > "we are taking **a specific causal method** [= synthetic control]
   > **to help us build a causal model** [= the learned latent state + donor /
   > factor structure over it, i.e. the space in which SC's assumptions hold]
   > **which can then help us develop or impute the counterfactual**
   > [= the SC forecast: the weighted donor combination *is* the imputed
   > untreated trajectory]."

   Caveat to raise explicitly: an alternative reading of "causal method →
   causal model" is causal *discovery* → graph/SCM → counterfactuals computed
   from the graph (echoing the clinical-trial perspective paper). Panel-data
   imputation and structural causal modelling are different methods. Opening
   question for the meeting: "I read the direction as: SC is the causal
   method, the causal model is the learned state and donor structure, and the
   counterfactual is imputed by the SC fit — is that right, or do you mean
   causal discovery over patient characteristics?"
7. **Scope for now**: the immediate experiment is similar-patient synthesis
   on trial data — build states, select donors, fit convex weights, check
   holdout-visit similarity — i.e. the MIMIC preliminary re-run on a panel
   with real longitudinal depth, with section E's matching rules as donor-
   selection baselines. The full learned-latent method (above) is the paper
   that follows if the preliminary works.
