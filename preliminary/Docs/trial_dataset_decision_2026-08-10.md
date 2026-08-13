# Trial dataset decision — NCT00174655 via Project Data Sphere (+ NACC track)

2026-08-10. This document is the archive of everything gathered for the
dataset decision, condensed so no original paper, repo, or agent report needs
re-reading. Decision approved by Jun in the planning session of this date.
Companion note: [causal_direction_notes_2026-08-10.md](causal_direction_notes_2026-08-10.md).

## The decision

Redo the MIMIC similar-patient-synthesis preliminary experiment on clinical
trial data. Selection criteria (Jun's): (1) patient-level data publicly
accessible, application acceptable; (2) GitHub code revealing the reference
paper's exact similar-patient pairs. Two papers survive; both are pursued:

1. **Primary — TWIN (KDD 2023) on NCT00174655** (BIG 02-98 breast cancer,
   Sanofi) via Project Data Sphere; code in PyTrial. Preprocessed data is
   downloadable today with no application; the full package needs a free
   ~48 h PDS registration.
2. **Secondary — NACC synthetic-control paper** (Wu et al., Alzheimer's &
   Dementia 2025; `github.com/chao-yi-wu/synthetic_control`) on NACC-UDS
   (free application; up to 20 annual visits per participant). Deep-panel
   track; treated side (I-CONECT trial) access under verification.

**Design decision (Jun):** the published similar-patient retrieval schema is
the donor-construction step of our pipeline — it builds the donor pool on the
chosen dataset, and our SC synthesis fits over that pool in all subsequent
experiments. KNNSampler is excluded as too simple; the candidates are the
globalized variant of TWIN's retrieval and TWIN-GPT's cosine top-5 (chosen by
donor quality). Full-pool SC without the retrieval pre-filter is kept only as
an ablation. The headline comparison: SC-over-retrieved-donors vs TWIN's own
generated synthetic visits vs simple baselines, all through the same
evaluation (code-set F1 at matched cardinality).

## Candidate table (why each was kept or excluded)

| candidate | dataset (access) | code | pairs in code? | verdict |
|---|---|---|---|---|
| **TWIN, KDD 2023** (Das, Wang, Sun) | NCT00174655, Project Data Sphere (free reg.) | PyTrial `pytrial/tasks/trial_simulation/sequence/twin.py` | mechanism yes, artifact no (see below) | **KEPT — primary** |
| **Wu et al. 2025, A&D** (chao-yi-wu/synthetic_control) | NACC-UDS (free, DUA + short proposal); I-CONECT treated arm = personal PI request | public repo | yes — returns donor NACCIDs + distances per target | **KEPT — secondary** |
| TWIN-GPT, ACM TOMM 2024 | same NCT00174655 | none released | described (cosine top-5) but no code | excluded as source; its retrieval mechanism reimplemented as a candidate schema |
| KNNSampler (PyTrial utility, not a paper) | same | `.../sequence/knn.py` | yes (joblib-recoverable) | excluded per Jun — t-SNE-2D + k=3 KNN, too simple; weak donors risk false-negative results |
| Andrews et al., ADNI digital twins (A&D 2025; ART 2026) | ADNI (application) | none | Gower top-20, described only | excluded — no code |
| Unlearn "Digital Twins as Synthetic Controls in Single-Arm Trials" (arXiv 2605.12832) | proprietary ALS/HD aggregations | none; no data/code availability statements | model predictions, not donor pairs | excluded |
| SyncTwin (NeurIPS 2021, van der Schaar) | CPRD (ISAC application + fee); repo has PKPD simulations only | github.com/vanderschaarlab/SyncTwin-NeurIPS-2021 | weighted donor combination (close to our method!) | excluded on data access; methodologically a close relative to cite |
| TrialSynth (Gao) | PDS trials, but data omitted from repo | github.com/chufangao/TrialSynth ("very rough state") | no matching (VAE+Hawkes) | excluded |
| SynTwin (PSB 2024, EpistasisLab) | SEER (free email reg.) | public | similarity network | excluded — cross-sectional registry, no longitudinal visits |
| PPMI symphony-dt (medRxiv 2026) | PPMI (4,628 participants / 28,185 visits) | gitlab.com/ahmed.hemedan/symphony-dt (Apache 2.0) | no — per-patient forecasts only | excluded — no pairs; deepest code+data panel otherwise |
| DT-GPT (npj Digital Medicine 2025) | NSCLC Flatiron (closed), MIMIC-IV, ADNI | github.com/MendenLab/DT-GPT | no — pure forecaster | not a pair source; remains the strongest predictive baseline for the project |

## Similarity mechanisms across the papers (they are NOT all KNNSampler)

Jun's question "do all papers use the KNN sampler?" — no. Each has its own:

| paper/system | similarity mechanism | pairs extractable? |
|---|---|---|
| TWIN (KDD 2023) | dot product between multi-hot **visit** code vectors; top-4 **including self**; **batch-local** (candidates = the 64 rows of the current minibatch, `shuffle=False`); recomputed every forward pass; never saved | no artifact — but deterministic given config, so a ~20-line instrumentation hook makes the pairs reproducible |
| KNNSampler (PyTrial utility) | t-SNE 2-D embedding of visit table + sklearn NearestNeighbors, k=3, global, Euclidean | yes (joblib dump) — but t-SNE is sklearn-version-dependent |
| TWIN-GPT (TOMM 2024) | cosine similarity, top-5 records per timestep | no code at all |
| Andrews et al. (ADNI) | Gower's distance on baseline covariates, top-20 real participants per subject | no code |
| Wu et al. (NACC) | hard filters (age ± auto-widening 1→10, sex, race, marital status, cognitive status) then cosine / Euclidean / Mahalanobis ranking on z-scored cognitive variables; returns top-n donor NACCIDs + distances (`main.py::multiple_distance()`) | yes |

## Condensed agent report 1 — TWIN code recon (PyTrial)

Source files:
- TWIN: `pytrial/tasks/trial_simulation/sequence/twin.py` — classes `TWIN`
  (line ~219, orchestrator, one `UnimodalTWIN` per non-frozen event type) and
  `UnimodalTWIN` (line ~566, VAE per event type).
- Retrieval sites (identical 8 lines in both): `UnimodalTWIN._train`
  (~line 728) and `_generate_one_loop` (~line 818):
  `n_cross_n = torch.matmul(x, x.T); top_5_index = torch.topk(n_cross_n, 4)`
  then rows `[self, self, nb1, nb2, nb3]` fed to `DotProductAttention`
  (~line 89) → context vector → VAE.
- Row construction: `_translate_sequence_to_df` (~line 792) → one row per
  (patient, visit); `_next_step_df` (~line 694); similarity vector = event
  columns **plus frozen treatment columns** (~lines 698–703, 725).
- KNNSampler: `.../sequence/knn.py` — `TSNE(n_components=2, random_state=0)`
  then `NearestNeighbors`; `_generate` does per-column random substitution
  from the k=3 neighbors; `generate_data` at line ~27 is dead code (wrong
  arity).

The ten red flags recorded for TWIN's retrieval:
1. No pair artifact — recomputed per forward pass, `save_model` keeps only
   state_dicts.
2. Batch-local: candidate pool = the 64 rows of the current minibatch
   (default `batch_size=64`, `shuffle=False`) → pairs depend on dataframe row
   order and batch size.
3. Visit-level, not patient-level — a "pair" may be two visits of the same
   patient.
4. Self included in top-4 (`n_cross_n[i,i] = ||x_i||²` is the row max);
   attention weight concentrates on the self-copy; ~3 genuine neighbors.
5. Unnormalized dot product on multi-hot codes → similarity confounded with
   code count.
6. Baseline covariates (age, tumor size, grade, nodes…) never enter the
   matching — only event codes do.
7. Demo cohort is treatment-degenerate: 937/977 patients on the same
   DOX+CTX+MTX+5FU set; docetaxel absent from the vocabulary (comparator-arm
   release).
8. Frozen treatment columns sit inside the similarity vector — an implicit
   same-treatment constraint.
9. `load_trial_patient_sequence` docstring calls the data "synthetic" — it
   is real Sanofi trial data (provenance traced through
   `process_NCT00174655.ipynb` reading raw SAS files).
10. `_remove_the_last_visit` drops every patient's final visit from training
    and generation; `TWIN(max_visit=13)` default caps depth.

Standalone repos: `github.com/trishad2/TWIN` (author's original; same
retrieval with explicit `k=4`; ships train/test CSVs of the multi-hot visit
table; **no license** → reference only, vendor nothing). `trishad2/SeqTrial`
= successor paper, not TWIN. TWIN-GPT: no code (different group; Yue Wang,
Tianfan Fu et al.). PyTrial license: BSD-2-Clause. Neither repo carries any
data-use statement for the demo data.

## Condensed agent report 2 — Project Data Sphere access + trial structure

- Listing: PDS content 127, UID `Breast_SanofiU_1998_127`, DOI
  10.34949/1z8y-rc53, provider Sanofi, uploaded 2015. "Available for
  Download". Files: redacted protocol PDF, sample CRF,
  `XRP6976_RP56976_PR_315_data_definition.xls`, data zip. Live page:
  data.projectdatasphere.org/projectdatasphere/html/content/127 (JS-only;
  readable via Wayback snapshots).
- **Comparator-arm only: 994 of 2,887 patients.** Docetaxel arms withheld.
  "Non standard raw data" (sponsor-native SAS tables, not SDTM).
- Access: free registration → click-through Data User Agreement → ~48 h
  approval → direct download (Sanofi-provided ⇒ no extra NCI gate). Free SAS
  cloud tools included.
- DUA (Cancer Research Platform Agreement, May 2020): purpose must be
  "in the field of cancer research" (methods/AI work on the oncology data is
  standard practice on the platform); no re-identification; **no
  redistribution — every user registers individually** (each lab member);
  publications must carry the acknowledgment: *"This [publication] is based
  on research using information obtained from www.projectdatasphere.org,
  which is maintained by Project Data Sphere. Neither Project Data Sphere
  nor the owner(s) of any information from the web site have contributed to,
  approved or are in any way responsible for the contents of this
  [publication]."*; no patents on procedures derived from the data; IRB is
  the user's responsibility (de-identified secondary use is typically
  exempt).
- Trial (BIG 02-98; Francis et al., JNCI 2008): adjuvant chemo in 3-weekly /
  4-weekly cycles, 7–8 cycles over ~24–30 weeks; clinical/hematologic/
  biochemical assessment before each cycle; 10-year protocol follow-up
  (long-term DFS/OS exists as time-to-event outcomes, not as repeated
  measures).
- Depth as used by ML papers: TWIN/SeqTrial 971 patients / 8,292 visits /
  max 14; TrialSynth 953 patients / 7.35 events per subject; PyTrial demo
  zip: 977 patients / 9,092 visit rows, mean 9.31, mode 8, max 17.
- Known discrepancy to resolve after download: PyTrial's paper Table 1
  labels the 971-patient sequential dataset NCT00694382 while its text,
  SeqTrial, and TWIN-GPT all say NCT00174655 — looks like a table typo;
  verify which zip TWIN preprocessed.
- Deeper PDS options under the same registration: NCT00522834 melanoma
  (310 pts, 3,578 visits, **max 37/patient**), NCT00981058 squamous NSCLC
  (548 pts, 4,000 visits, max 26), NCT00694382 SAVE-ONCO (**SDTM**, 1,604
  pts, richest event vocabulary), NCT00003299 SCLC (**both arms**, 587 pts —
  but AE tables only). Prostate Cancer DREAM Challenge (Synapse syn2813558):
  4 PDS trials, ~2,070 pts, curated baseline + raw longitudinal labs/vitals/
  lesions/meds tables, public code from the challenge.

## Condensed agent report 3 — alternatives sweep (what else was checked)

Checked and rejected with reasons in the candidate table above: TWIN-GPT,
Andrews/ADNI Gower twins, Unlearn single-arm DT paper, SyncTwin/CPRD,
TrialSynth, SynTwin/SEER, PPMI symphony-dt, DT-GPT, plus: Seibold et al. ALS
forest-kernel weights (PRO-ACT, code `HeidiSeibold/personalised_medicine` —
outcome is a single slope, panel collapses), FedECA (IPTW, proprietary data),
SPRINT repos (prediction only), Enroll-HD hdml (prognosis only), NIDDK
DPP/Look AHEAD (no matching+code combo found), Unlearn AD-DTG-3.1
(EMA-qualified prognostic twin, closed). Bottom line as reported: nothing
beats the TWIN/PDS + NACC pair of tracks on the two criteria.

## Condensed agent report 4 — NACC / I-CONECT access (verified 2026-08-10)

**NACC-UDS: easy.** Free to researchers worldwide. Process: check NACC
publications for overlap → submit the REDCap request form
(nacc.redcap.rit.uw.edu) → accept the electronic DUA. The proposal is
literally "four or five sentences" (aims, hypotheses, methods, main
variables). DUA is signed by the individual requester — **no faculty-PI
signature requirement is documented** (cross-institution collaborators each
sign). No IRB letter required. Acknowledgment within 3 business days; data
delivered **within 48 h** of approval; download links expire ~2 weeks.
Delivery: long-format CSV, one row per participant-visit; >54k participants,
up to 20 annual visits, >200k assessments. Required acknowledgment text (U24
AG072122 + ADRC grant list); **no co-authorship required**, but manuscripts
must pass NACC's brief administrative review *before* journal submission.
Caveats: UDSv3 was retired 2026-03-01 and UDSv4 rows now appear in
deliveries — the Wu et al. code filters `FORMVER == 3` and silently discards
v4; neuropath values repeat across visit rows (double-counting trap); NACC
is a case series, not a population sample.

**I-CONECT: not free, not deposited.** In no NIA repository (NIAGADS, Aging
ResearchBiobank, AD Workbench, ICPSR all empty). Official route: the request
page at i-conect.org/request-data → MGH REDCap form (Interdisciplinary Brain
Center, contact CYOUNG@MGH.HARVARD.EDU; alternative: PI Hiroko Dodge,
hdodge@mgh.harvard.edu). **Data storage/transfer fee up to $2,500 per
request** (audio/video: up to $3,000 plus a PHI DUA taking 30–90 days).
The cognitive/clinical outcome data used by Wu et al. (clinical diagnosis,
neuropsychological testing, phone check-ins, NIH Toolbox) is requestable
separately from audio/video, so the PHI DUA delay need not apply — but the
base fee stands. Trainee requests are contemplated (form lists "PhD
Candidate" as a position title); a Zoom consultation follows submission.

**Wu et al. code on NACC alone: feasible, ~1 day of surgery.** The matching
machinery is I-CONECT-agnostic: targets and donors are split only by a
`real` flag, the loop keys on `NACCID`, and the feature vector is 100% NACC
variables. Matching = hard pre-filter (exact SEX, RACE, MARISTAT, NORMCOG;
age ± tolerance auto-widening 1→10 y) then cosine/Euclidean/Mahalanobis on
z-scored cognitive scores (Craft Story recall ×3, category fluency animals +
vegetables, number span fwd/back, verbal naming, MoCA, education, GDS).
Required edits: `process_nacc.py` hard-fails without `Data/iconect_long.csv`
(delete the concat; feed the NACC subset straight to `missing_code()`);
`main.py` reads `Data/iconect_wide.csv` (stub out); z-scores are computed on
the combined frame (NACC-only z-scores are cleaner but not on the published
scale); `util.nacc_six()` linearly interpolates 6/12-month scores between
NACC annual visits (applies to both sides in a placebo — known artifact);
hardcoded relative paths throughout; deps = pandas/numpy/scipy/sklearn (no
requirements.txt). Paper's frame of reference: NACC freeze 64 (2024-03-28),
50,259 participants, 5,132 meeting I-CONECT eligibility (age ≥74,
`CDRGLOB < 1`, `NACCGDS ≤ 7`, valid MoCA + animals).

## Dataset facts (demo release, verified locally on 2026-08-10)

Local copy: `/data/wang/junh/datasets/PDS/NCT00174655/seq_patient/`
(zip from `storage.googleapis.com/pytrial/seq_patient_nct00174655.zip`,
140,988 bytes, byte-identical to the copy in trishad2/TWIN).

- `visit.pkl` — plain nested list, `[patient][visit] -> [treatment_codes,
  medication_codes, adverse_event_codes, ae_serious_codes, visit_stage_int,
  timestamp_float]` (integer codes indexed against voc.pkl).
- `voc.pkl` — dict of `Voc` objects (class defined in the processing
  notebook: `word2idx` starting `{'[PAD]':0}`, `idx2word`); pickle carries a
  `__main__` class reference, needs a small shim to load. Sizes (incl. PAD):
  treatment 5 (DOX, CTX, MTX, 5FU), medication 101 (WHO drug codes, top-100),
  adverse_event 278 (COSTART), ae_serious 77, visit_stage 3
  (treatment/followup).
- `feature.csv` — 977 rows; columns: RUSUBJID, performance status, height,
  weight, death, AGE, RACE, SEX, SURGERY, num relapse, primary tumor, tumor
  size, multifocal tumor, num positive axillary lymph nodes, tumor location,
  histopathologic grade, histopathologic type. Loader uses `num relapse` and
  `death` as labels.
- `timestamp.pkl` / `visit_stage.pkl` — parallel per-visit lists (days since
  first visit; treatment vs follow-up stage).
- Cohort: 977 patients, 9,092 visits, mean 9.31/patient, mode 8, max 17.
  Mean codes per visit: treatment 1.82, medication 2.48, AE 3.83.
- Caveats: comparator arms only; near-degenerate treatment variation (96%
  same regimen); raw SAS files not included (come with the PDS package).

## Condensed agent report 5 — local verification + TWIN run (executed 2026-08-10/11)

All artifacts under `preliminary/results/trial_nct00174655/` (gitignored);
scratch scripts in `preliminary/scratch_trial/`; metadata in
`run_manifest.json`.

**Go/no-go: GO.** 977 patients, 9,092 visits, mean **9.31**/patient (median
9, max 17; histogram mode 8). Eligibility (≥ T0+2 visits): T0=1 → **961**,
T0=3 → 951, T0=5 → **936**, T0=7 → 571. Against MIMIC (T0=1 → 749, T0=2 →
348): this panel supports pre-periods 3–5 visits deep with ~950 patients —
the capability MIMIC lacked.

**TWIN ran end-to-end** (CPU, 113 s training, 20 epochs, medication + AE
VAEs, treatment frozen): model + generated records saved
(`twin_model/`, `twin_generated.pkl` — 977 patients × real visit counts).
**Pair dump deterministic**: two independent runs byte-identical
(`twin_pairs.parquet`, 64,920 rows); generation-site pairs identical to
training-site pairs. Global retrieval variants saved
(`retrieval_global_dot.parquet`, `retrieval_global_cosine.parquet`, top-10,
self-excluded, same 8,115-row universe → row-for-row comparable).

**TWIN retrieval quality — Jun's donor-quality concern confirmed with
numbers:**
- **71.2% (medication) / 56.9% (AE) of TWIN's top-4 donors are the same
  patient as the target** (batch-local retrieval over patient-ordered,
  unshuffled rows).
- Only **21.6%** of TWIN's non-self pairs appear anywhere in the global
  dot-product top-10 — the shipped retrieval is essentially not
  cohort-level similarity.
- Generation fidelity is poor at shipped defaults: collapsed-decoder
  signature (modal outputs are contiguous high-index code runs; 170 distinct
  medication code-sets generated vs 2,010 real).
- PyTrial bug: `TWIN._build_model` never forwards `learning_rate` — training
  ran at UnimodalTWIN's default 1e-3 while the config records 5e-5.

**Data quirks for the loader** (all verified): `voc.pkl` is a *dill*
by-value pickle — `dill.load` works, no shim; `visit.pkl[i][j]` is 6 fields
`[treatment, medication, AE, ae_serious, visit_stage_int, timestamp_days]`;
**use fields 4–5, not `timestamp.pkl`/`visit_stage.pkl`**, which are longer
than `visit.pkl` for 594/977 patients and cannot be aligned positionally;
the `visit_stage` integers (0/1) do not match `voc['visit_stage']` indexing;
`feature.csv` `weight` is a string column (contains `">= 125"`); minor
missingness in AGE/SURGERY/tumor columns.

**Methodological caution for the headline table:** TWIN's generated visit at
T0+1 is a *reconstruction conditioned on the patient's own real record*
(digital-twin fidelity), not a forecast — it sees the ground truth. The
comparison must either use TWIN's next-step prediction path or label the
TWIN row as reconstruction (upper bound), not forecast.

## Condensed agent report 6 — first full experiment on NCT00174655 (run 2026-08-10/11)

Runner: `scripts/run_trial_fit.py` (new; existing MIMIC modules untouched and
reused by import). States: 377-dim per-visit binary bag (100 `m:` medication
+ 277 `a:` AE codes; treatment excluded as the intervention variable). All
five sanity gates passed (bit-identical no-leakage rebuild; panel alignment;
nn1 = brute force 961/961; LVCF self-decode F1=1.0; donor-schema equivalence
check). Artifacts: `results/trial_nct00174655/trial_headline.txt`,
`trial_aggregate.json`, `trial_targets.parquet`, figures under
`figures/trial_nct00174655/`. Wall clock 191 min at jobs=24 (m=all dominates,
~1 h/config; the m∈{25,50} grid alone is ~6 min). Numbers below are as
reported by the run's aggregate JSON (agent-executed; audit trail on disk).

**Headline (T0=1, m=50, euclid schema, n=961), code-set F1 [95% CI]:**

| method | F1 | vs LVCF |
|---|---|---|
| LVCF (copy own visit at T0) | **0.774** [0.766, 0.783] | — |
| ridge (α=1, LOO) | 0.741 | −0.034* |
| **SC (simplex QP)** | 0.661 | −0.114* |
| top-10 donor mean | 0.629 | −0.146* |
| SC on random donors (placebo) | 0.569 | −0.206* |
| 1-NN donor | 0.531 | −0.244* |
| twin_forecast (TWIN's own prediction head, fair) | 0.444 | −0.331* |
| cohort mean | 0.443 | −0.332* |
| twin_recon (sees truth; decoder-collapsed) | 0.096 | −0.679* |

**Findings:**
1. **LVCF wins outright at every T0, m, and schema.** Trial CRF med/AE code
   sets are extremely persistent: a patient's own consecutive-visit F1 is
   0.774. No donor-based method (nor ridge) beats copying the patient's own
   last visit on this state representation.
2. **Matching and synthesis do carry real signal**: SC beats cohort mean by
   +0.218*, the random-donor placebo by +0.092*, top-10 mean by +0.032*, and
   1-NN by a wide margin. The convex combination is the best *donor-based*
   forecaster — the hypothesis "synthesis > matching" holds; "synthesis >
   self-persistence" does not.
3. **Both TWIN comparators lose badly.** TWIN's genuine next-step prediction
   head scores 0.444 — statistically indistinguishable from the cohort mean —
   despite an in-batch-neighbor advantage; its reconstruction mode is
   decoder-collapsed (0.096 while seeing the truth).
4. **The TWIN-lift donor schemas beat Phase 2's Euclidean rule everywhere**:
   cosine > dot > euclid at all T0 (+0.014–0.018 F1, more effective donors,
   better pre-fit); donor sets genuinely differ (euclid∩dot overlap 0.25).
   Vindicates adopting the published retrieval geometry — though no schema
   closes the LVCF gap.
5. **Deeper pre-periods do not help SC** (Δ vs LVCF: −0.114 / −0.108 /
   −0.173 at T0=1/3/5); T0 variation tracks cohort/visit-phase drift, both
   methods moving together. Pre-fit does transfer (pre-vs-post RMSE r=0.74),
   but extra pre-period constraints buy nothing.
6. **Solver caveat**: m=all degrades at deep T0 — SLSQP "positive directional
   derivative" non-convergence for 1.5% / 26% / 59% of targets at T0=1/3/5
   (feasible but not certified optimal). m∈{25,50}: zero failures.
7. Protocol notes: simple baselines use the full pool (existing
   `baselines.py` untouched), so they are constant across the m/schema grid;
   placebo at primary config only; no AUROC analogue (only a patient-level
   death flag exists); two K=0 patients scored by the existing empty-set
   convention; box was heavily contended during the m=all timings.

**Interpretation for the causal direction (not in the run report):** the
LVCF result is a fact about *forecasting persistent code sets*, not a defeat
of the causal design. In counterfactual imputation the donor pool is the
*other arm* and "copy your own last visit" is not an admissible counterfactual
estimator — it cannot respond to intervention. What the preliminary
establishes: donor synthesis extracts real cross-patient signal (beats every
non-self reference, including the published method on its own dataset), and
the state representation is the binding constraint — per-cycle labs/vitals
from the full PDS package (unused by any prior paper) are the natural
upgrade, since sticky code bags favor persistence by construction.

## Condensed agent report 7 — latent-recovery L0–L2 results (run 2026-08-11)

> **STATUS: RULED INVALID by Jun, 2026-08-11.** The rendered inputs were
> opaque digit strings (WHO drug / COSTART codes), not text — this violates
> the experiment design, which requires textual input. The H1/H2 verdicts
> below are void; the battery re-runs once a code→term mapping exists. The
> report is retained as the audit trail and for the mechanism/infrastructure
> findings (solver diagnostic, encoder pipeline, cost measurements), which
> survive.

Pre-registered in [latent_prereg_2026-08-11.md](latent_prereg_2026-08-11.md);
outputs under `latent/{data,results}/`; code in `latent/{src,scripts}/`.
Repo was reorganized the same day into `mimic/` / `observable/` / `latent/`
(see MOVE_NOTES.md in each).

**Verdicts: H1 (weight identifiability) REFUTED in both encoder arms. H2
(cross-space consistency) neither supported nor refuted.** Per the prereg's
own interpretation rule, L3/L4 are now exploratory only and were not run.

Key numbers (decision cell K=5, m=50, σ=0):
- qwen3-768: mean L1(ŵ,w*) 1.651; dominant-weight case recovers ŵ_max ≈ 0.589 when
  truth is 0.9 (|error| 0.311, threshold 0.25). biolord: L1 1.838, ŵ_max ≈
  0.531. Noise level irrelevant (σ=0 vs matched changes L1 < 0.01).
- **Mechanism, measured: failure is at retrieval, not the solver.** The
  m=50 pool built around the mixed point contains on average < 1 of the 5
  true donors (0% of replicates contain all 5); a convex mixture of K
  visits is closer to dozens of unrelated visits than to its own
  constituents. The space is ultra-concentrated: mean top-3 cosine 0.991
  (qwen3) / 0.996 (biolord). m=500 barely helps (1.76/5, L1 1.449).
- Oracle gate technically failed (mean L1 up to 0.048 vs 0.01 threshold)
  but diagnostics bound it as an SLSQP early-stopping artifact on
  near-singular Grams: warm-starting at w* returns w* exactly (L1=0), so
  w* is the argmin. Two orders below the m=50 misses; cannot explain the
  refutation.
- H2: weight cosine bag-vs-latent 0.471 (qwen3) / 0.384 (biolord) — above
  the permutation null's IQR but below its p95; top-1 donor agreement 1.6×
  null (3× required). Latent pre-period fits are near-perfect (RMSE 0.0048
  vs 0.074 in bag space) — degenerate, consistent with the concentration
  finding.
- Encoder ranking (for whatever survives): qwen3-768 wins both H1 primary
  and H2 tiebreak. MRL-768 vs native 4096: 90.7% top-1 agreement.
  Retrieval smell test was genuinely good — top-3 donor Jaccard 0.786
  (med, qwen3) vs 0.056 random — retrieval quality and mixture
  identifiability dissociate.

**The reframing discovery: the rendered "texts" are opaque digit strings.**
`voc.idx2word` for medications holds raw WHO drug codes (`00955301001`) and
for AEs raw COSTART codes (`00038`) — the demo release ships no code→term
dictionary (only the 4 treatment names are English). Both encoders embedded
digit strings; all retrieval structure is lexical overlap, not clinical
semantics. **So H1's refutation is a verdict on this dataset's opaque-code
renderings, not yet on clinical text.** The extreme cosine concentration is
partly this artifact (digit strings all look alike to a text encoder).

Deviations recorded in the summary JSONs: m=all replaced by m=500@N=50
(measured cubic solver scaling: m=9,092 ≈ 3.5 h/solve — infeasible);
residual-matched σ operationalized as median real-fit residual /√d; design
2 run despite the gate (justified by the bounded diagnostic); H1 criteria
cell-reading noted; summary JSONs regenerated after an encoding fix.

**Resolution (Jun, 2026-08-11): halt this track; redo from dataset
selection.** The latent experiment requires real textual input; the
NCT00174655 demo release cannot provide it (the preprocessing notebook was
inspected: it read only the `C_CTXWHO` / `C_AECOS` *code* columns from the
raw SAS tables; whether the raw tables also carry name/verbatim columns is
unknowable without the PDS data dictionary). A fresh dataset search for
longitudinal per-timepoint clinical narratives is underway; new selection
criteria: (1) real prose per patient per timepoint, verified by looking at
actual samples before selection; (2) multiple timepoints per patient;
(3) accessible (public/application); (4) bonus: paired structured data for
the cross-space test, and/or images (satellite endgame). **Standing ruling
(Jun, 2026-08-11): the MIMIC family is excluded entirely — all variants,
all time axes (admissions, notes, imaging studies). Do not re-propose.** The mechanism finding from the invalid run (convex
mixtures of concentrated unit-norm embeddings are hard to invert —
argument for a learned latent) survives as motivation.

## Condensed agent report 8 — prior-work map, dataset-agnostic slice (2026-08-11)

From a residual search thread; its MIMIC-specific content is NOT used
(standing exclusion) — only the positioning facts that hold for any dataset:

- **SyncTwin (NeurIPS 2021, van der Schaar lab) is the single closest prior
  work to per-patient synthetic control from EHR — and it is structured-only**
  (CPRD labs/vitals, 125,784 individuals; ~15 of 24,557 controls contribute
  per twin; pre-treatment fit gate with δ=0.12 → 75% acceptance). Encodes
  temporal covariates to a latent, matches in representation space, twin =
  weighted average of contributors. No text anywhere. Code:
  ZhaozhiQIAN/SyncTwin-NeurIPS-2021; no arXiv, NeurIPS proceedings only.
- Method lineage to cite: Athey et al. matrix completion (JASA 2021); Robust
  SC (JMLR 2018); Synthetic Interventions (arXiv 2006.07691); causal matrix
  completion / synthetic NN (arXiv 2109.15154). Applied: SECRETS (SI within
  real RCTs, 25–76% sample-size reduction); SCouT (transformer donor
  combination); Carrigan 2020 (Flatiron EHR control arms, r=0.86 vs RCT).
- **DT-GPT caveat confirmed at source**: it serializes *structured*
  variables into text templates — it does not consume clinical narratives.
  The "text-based digital twin" niche is genuinely unoccupied.
- **The gap, quotable**: no published work builds a per-patient synthetic
  control from clinical text; latent-trajectory modeling (neural-ODE/state-
  space: ICE-NODE, COPER, SETOR) runs on codes/vitals while text-sequence
  work uses attention pooling with no dynamics; and longitudinal text/image
  histories are barely exploited (a 2025 survey: 20 of 21 longitudinal
  report-generation methods use exactly one prior study).
- Two negative results worth citing when arguing for a *learned* latent:
  CliniBench found BM25 beats dense sentence-transformer retrievers for
  similar-patient retrieval; Nützel et al. (MICCAI-W 2025) found UMLS
  concept sets + Tversky beat CLIP/CXR-BERT embeddings for report
  retrieval. (Both consistent with our invalidated run's mechanism finding
  that off-the-shelf embedding geometry does not support the task.)
- Sourcing caveat from the agent itself: some 2026 arXiv IDs were not
  independently re-fetched — spot-check before citing.

## Condensed agent report 9 — longitudinal clinical-text dataset shortlist (2026-08-11)

Redone search, MIMIC excluded by standing ruling; every shortlisted
candidate has a verified real-text excerpt. Full details and URLs in the
agent transcript; the decision facts:

| candidate | text depth / patient | text verified | paired data | access | catch |
|---|---|---|---|---|---|
| **MedAlign** (Stanford) | 276 patients, 46,252 notes, **median 101 notes/patient** (median 64k tokens/patient) | real Stanford notes (paper's illustrative snippet is synthetic; corpus is real) | **3.6M OMOP structured events** — ideal for the cross-space test | Redivis application, CITI + DUA, **7–10 business days**, no fee found | licensed as **evaluation-only benchmark** ("cannot be used for model training") — must read license before committing; no images; only 276 patients |
| **CheXpert Plus** (Stanford AIMI) | 64,725 patients, 187,711 report+image studies, mean 2.9 studies/patient; **≥5-study tail size unverified** (no published distribution) | yes — README `section_findings` prose incl. explicit Comparison sections (text encodes change over time) | 8 demographic fields + DICOM metadata; **paired images** (satellite-endgame alignment) | **self-serve download**, Stanford research-use agreement, free, immediate | per-patient depth thin on average; must measure the deep tail ourselves after download |
| **n2c2 2014 longitudinal** (i2b2/UTHealth) | 296 patients, 1,304 records, **2–5 records/patient**, ~617 tokens/record | yes — two verified prose excerpts (letter + discharge registers) | none (structured data explicitly excluded) | **FROZEN** — DBMI portal: "Temporarily Unavailable", registration closed, no timeline | best design fit (CAD-progression strata, two annotation layers) but unobtainable today; only path is emailing DBMI |

Ruled out with reasons verified: EHRSHOT (structured-only — exactly the
digit-string failure mode), All of Us (no free-text notes available),
INSPECT (impression-only text, ~1 CT per patient), Synthea (no narrative
text). Synthetic last-resort tier noted (Asclepius — not longitudinal).
Not completed: the oncology/EHR track (GENIE, SEER pathology, CPRD 403,
OpenSAFELY, non-US corpora) and PadChest/Open-i/TCIA/BIMCV depth
verification — can be resumed if none of the three suit.

**Addendum (2026-08-11): Jun likely already holds the n2c2 corpus.** The
notebook `moa-clinical-rag/data/n2c2_cohort.ipynb` processed n2c2 **2018
Track 1** — which reuses the 2014 longitudinal corpus — and its outputs
confirm the real thing: per-patient XML with a *sequence* of timestamped
records ("Number of notes in record: 6", each starting `Record date:`),
genuine prose, 13 eligibility labels. It read **both** `train/` (202
patients) **and** the gold-standard `test/` (86) — the full 288-patient
release, **not just the test split**. The files are not on the lab machine;
the notebook's paths are Jun's Google Drive
(`MyDrive/NEU/SummerResearch/2018_cohort_selection/`). If that folder still
exists, the DBMI registration freeze is moot. Action: Jun retrieves the
folder to `/data/wang/junh/datasets/n2c2_2018_track1/` (chmod 700); then we
verify the records-per-patient distribution locally. Two flags: (a) the
original n2c2 DUA (signed when the data was obtained) should be checked for
current-use terms — no redistribution either way; (b) n2c2 has **no paired
structured data**, so the cross-space weight test would need pseudo-codes
extracted from the text (e.g. UMLS concepts) rather than native codes.

## Condensed agent report 10 — n2c2 verified in hand; DATASET DECIDED (2026-08-11)

**Decision: the latent-recovery track runs on n2c2 2018 Track 1 (= the 2014
i2b2 longitudinal heart-disease corpus), which Jun already holds under
their existing n2c2 registration.** Uploaded to
`preliminary/latent/data/n2c2/` (gitignored, verified). Beats MedAlign
(deeper per-patient text but 7–10-day application + evaluation-only
license, no severity labels) and CheXpert Plus (scale + images but mean 2.9
studies/patient with unverified tail) on the criteria that matter for this
experiment: verified rich prose, universal ≥3-timepoint coverage, severity
labels, zero access friction. CheXpert Plus remains the designated
dataset for the later multimodal stage.

Verified corpus facts (full details `latent/results/n2c2_verify.json`):
- **288 patients (202 train + 86 test), 1,264 records, 781k tokens.**
  Records/patient: mean 4.39 (1×2, 13×3, 147×4, 127×5) → **≥3: 287/288
  (99.7%), ≥4: 274/288 (95.1%)**. Tokens/record: mean 618, median 528, max
  2,985. Date spans: median 3.9 years per patient (surrogate-shifted
  years 2059–2173).
- Registers: 881 clinic/consult notes, 251 discharge/admission, 132
  letters. Real prose with preserved typos.
- Integrity: 0 malformed, 0 empty, 0 duplicate ids or TEXT hashes within
  or across splits; all 288 carry exactly the 13 criteria tags.
- Criteria prevalence highlights: ADVANCED-CAD 170/118 (balanced — best
  severity label), MAJOR-DIABETES 156/132, MI-6MOS 26/262;
  **KETO-1YR 1/287 — unusable**. Test-split ADVANCED-CAD carries the
  organizers' deliberately-uncorrected annotations for files 140, 156,
  205, 266, 277.
- **Two loader rules discovered**: records must be sorted by parsed date,
  not file order (train 112 and 344 are non-chronological in-file);
  `^Record date:` at line start is the exact record separator (1,264/1,264
  boundary match) — asterisk rules are NOT reliable (table formatting
  inside records).
- Official eval script present (Apache 2.0, stdlib-only, scores the 13
  criteria from TAGS; ignores TEXT).
- Remaining gap: no native structured data → H2 cross-space test uses
  UMLS-extracted concept bags (QuickUMLS over MRCONSO/MRSTY once Jun's
  UTS download lands; negation caveat pre-registered in the addendum).

## Condensed agent report 11 — n2c2 L0–L2 results on real prose (run 2026-08-11)

Pre-registered (prereg + addendum); outputs `latent/results/n2c2_*`;
per-cell table in `n2c2_weight_recovery_summary.json`. H2 awaits Jun's UMLS
download. All parse asserts passed against the independent verification.

**Headline verdicts:**
- **Oracle gate PASSED** both arms (mean L1 ≈ 1.5e-05 vs 0.01 threshold) —
  the failure mode of the digit-string run is gone.
- **m=all control: EXACT recovery** (L1 = 0.0000, dominant-weight error 0.0000, both
  arms, σ=0). With retrieval removed, convex weights are perfectly
  identifiable in the latent space of real prose. **The geometry supports
  weight identification; the digit-string result was the artifact, as
  suspected.**
- **Decision cell (K=5, m=50, σ=0): formally "neither"** — between the
  pre-registered thresholds. The dominant-weight case (the advisor's 0.9 example)
  is emphatically positive: recovered ŵ_max = 0.884 vs truth 0.9 (error
  0.016 [CI 0.0155–0.0166], 6× inside the 0.10 support bound), top-1 donor
  100%. The miss is the diffuse-weights case: mean L1 0.452 (qwen3) vs
  the 0.30 support bound, top-1 95.6%.
- **The residual error is attributable to retrieval, not geometry or
  solver**: the m=50 pool contains 2.66/5 true donors (qwen3; 3× better
  than the digit-string run's 0.88/5, but still incomplete), while
  oracle/m=all cells recover exactly. Better donor retrieval → better
  recovery; consistent with Jun's donor-quality thesis and with retrieval
  as the method's pressure point.
- **Encoder selection: Qwen3-768 wins** (diffuse-family L1 0.452 vs BioLORD
  0.732; also wins the weak smell-test queries). The on-record prediction
  that mean-pooled BioLORD would win recovery was WRONG — recorded.
- **Space health on real prose**: mean top-3 cosine 0.710/0.737 (vs
  0.991/0.996 on digit strings — concentration collapsed); smell test
  clinically coherent (renal-cell-carcinoma query → renal-mass donors;
  CAD query → vascular-disease donors). MRL 768 vs 4096: pairwise-cosine
  r = 0.962, top-1 agreement 0.647.
- Noise sensitivity: residual-matched σ (≈0.019) roughly +0.2 L1 across
  cells; dominant-weight error rises to ~0.06 — still well inside support territory.
- K scaling: L1 grows with K (0.11 → 0.88 for qwen3 diffuse weights at m=50) —
  diffuse many-donor mixtures are the hard regime; 2-donor and dominant-
  donor mixtures are essentially solved.

Deviations recorded: m=all trimmed mid-run to K=5/N=50 (Jun: it is a
diagnostic control, not the method; superseded N=500 partials logged, match
to 3 decimals); H2 deferred (UMLS); σ definition inherited; asterisk
banner lines stripped from record text; oracle-gate diagnostic unnecessary.
Wall clock: parse 0.6 s, embed 574 s, L2 final 790 s (+~2.4 h discarded
pre-trim m=all compute).

**Prereg consequence**: H1 is not refuted in either arm → L3/L4 proceed as
pre-registered (gated on the score() validation), with qwen3-768 as the
selected encoder. Interpretation for the meeting: latent recovery *works*
on real clinical text — exactly identifiable given the right donors —
and the open problem is donor retrieval for diffuse mixtures, which is the
method's own next contribution (learned retrieval / learned latent).

## Condensed agent report 12 — H2 cross-space consistency: REFUTED both arms (2026-08-11)

Outputs `latent/results/n2c2_cross_space*`, `n2c2_concept_extraction.json`;
bags `latent/data/n2c2_concept_bags.parquet`. Extraction: QuickUMLS 1.4.2 in
a dedicated venv (`/data/wang/junh/envs/quickumls`; medrag untouched) over
Jun's pre-built index (UMLS 2024AA, unqlite; used unchanged after a probe).
1,264 records → mean 59 concepts/record, vocab 1,915 CUIs after min-df ≥ 5;
extraction eyeballs correctly; known artifacts: "his"→histidine false
positive (near-constant column, left in per prereg), no negation handling
(pre-registered v1 caveat). Leakage check bit-identical.

**H2 verdict: refuted in both arms — and the refutation SURVIVES the
pre-registered NegEx v2 follow-up, making the finding final.** At T0=1
(287 targets, identical 50-donor pools in all spaces): mean weight-cosine
bag-vs-latent 0.425 → 0.422 v2 (qwen3) / 0.390 → 0.378 v2 (biolord) —
inside the permutation null's IQR both times; top-1 donor agreement 1.3× /
0.9× the null in v2 vs the 3× bar. The v2 filter was real and worked:
negspacy `en_clinical` over the v1 spans removed 21.4% of concept mass
(top negated: edema, dyspnea, chest pain, fever — exactly
review-of-systems denials), plus the histidine artifact; vocabulary
1,915 → 1,678; leakage checks bit-identical in both versions. Removing a
fifth of the (noisy) concept mass moved the agreement metric by −0.004 —
negation noise is NOT the explanation. Fits healthy everywhere (0
non-converged; no degeneracy; random-bags negative control behaves).

**Final reading per the prereg's interpretation rules: H1 held (latent
mixing identifiable, exactly so with donors present) and H2 failed
robustly — the prose-embedding space and the extracted-concept space tell
genuinely DIFFERENT mixing stories about the same patients on the same
donors. Representation choice is load-bearing for causal weight
interpretation. This is the empirical motivation for learning the latent
state rather than adopting either representation off the shelf.**

## Condensed agent report 13 — L3/L4 generation stages (run 2026-08-11)

Outputs `latent/results/{score_gate_report, l3_summary, l4_summary,
copy_diagnostic, l3_code_recovery}.json` + parquets. 1,000 completions
(under the 1,500 cap); Qwen2.5-7B temp 0 on one A40.

- **score() gate PASSED** — the vendored, never-executed
  `QwenIntegrator.score()` is correct: NLL ordering fluent 1.50 < shuffled
  5.81 < random 14.31 nats/token; byte-deterministic; cross-checked against
  an independent transformers implementation on a second GPU (max diff
  0.017 nats, with misaligned variants 3–13 nats away — the off-by-one risk
  is affirmatively absent). Operational caveat FOR vendored/MODULE.md
  (mimic/ not modified per constraints, recorded here instead):
  `prompt_logprobs=0` materializes full-prompt logits, which OOMs beside
  the default `gpu_memory_utilization=0.85` — score at ≤0.5 or in a
  separate engine process.
- **H3 formally SUPPORTED (ppl ratio 0.983 ≤ 1.5; median cycle rank 2 of
  1,265; ρ=0.28, p≈1e-90) — but read with the copy diagnostic: the pass is
  CONFOUNDED.** The retrieval-conditioned decoder mostly *copies one donor
  record* (median 84% 8-gram overlap with a donor; 0.99 donors contribute
  >20%; the copied donor is the top-weight donor only 56% of the time) —
  and copying a real record trivially passes perplexity + cycle-rank. The
  pre-registered metrics cannot distinguish generation from retrieval; do
  NOT quote "H3 supported" as evidence that mixtures decode to sensible
  synthetic records.
- **H4 REFUTED** (mean per-pair Spearman(α, severity) 0.143 logistic /
  0.101 LLM-judge, < 0.3 on both; scorer AUROC 0.683). Mechanism visible
  in the copy fractions: the decoder *switches* records around α≈0.75
  instead of interpolating (at α=0.5 it still emits mostly the not-met
  record). Group means do move the right way; per-pair monotonicity fails.
  Perplexity flat across α (≤2.7% deviation) — mixtures stay "sensible."
- Code recovery (descriptive): generated records' concept-F1 vs expected
  mixture set 0.42; vs top-donor bag 0.56 — same copying signature.
- Deviations recorded: donor prompts truncated at 2,048 tokens; 84% of
  completions hit the 800-token cap; repetition_penalty 1.0 (greedy);
  copy diagnostic added post hoc; metric (d) computed (H2's extraction
  landed mid-run); checkpointing added after an OOM cost ~25 min.

**Reading: L3/L4 indict the STAND-IN DECODER, not the latent space.** H1
already established the mixing structure is identifiable in the latent;
L3/L4 show a prompt-conditioned LLM cannot *decode* mixtures — it
retrieves. Generation-side conclusions require the trained decoder that is
the proposed contribution. Full-battery scoreboard: H1 dominant-weight supported /
diffuse retrieval-limited; H2 refuted (representation divergence, robust
to negation cleanup); H3 confounded-pass; H4 refuted (decoder switches).
Every negative localizes to an off-the-shelf component — retrieval, the
fixed encoder pair, the stand-in decoder — which is the empirical case for
learning them.

## Condensed agent report 14 — prior art on planted-mixture weight recovery (2026-08-11)

**Verdict: no direct precedent.** Nobody has planted convex-mixture weights
on real units' *learned embedding* vectors and scored a simplex solver on
recovering both weights and donor identities as a synthetic-control
validity check. The literature splits on two axes with no paper in both
cells: planted weight+identity recovery exists only in RAW observation
spaces; learned-latent convex structure exists only with LEARNED
dictionaries (no donor pool, so identity can't be posed).

Three nearest neighbors, ranked (full citations in the agent transcript):
1. **Sparse hyperspectral unmixing** (Iordache/Bioucas-Dias/Plaza, IEEE
   TGRS 2011; survey+package: HySUPP, TGRS 2024) — same donor-pool +
   constrained-solver + planted-abundance protocol, per pixel; differs
   only in that the space is physical spectra. **Satellite imagery** — the
   project's target domain has a native tradition of exactly this
   validation, without causal semantics. Adopt their metric vocabulary
   (aRMSE / SAD / SRE). Deep-AE unmixing keeps mixing in raw space
   (decoder = endmember matrix) — the learned-mixing cell stays open.
2. **Bayesian donor selection in SC** (Lee/Lim/Kim/Wang, arXiv 2607.08142,
   Jul 2026, unrefereed) — the only paper scoring both weight error AND
   donor-identity TPR/TNR against a planted simplex w*; outcome panels,
   no embedding.
3. **Arora et al., TACL 2018** (word-sense linear structure) — our
   generative assumption as a theorem in a learned text space, with a
   planted pseudoword-mixture experiment run FORWARD only (never inverts
   to recover the weights).

Contrast citations: mixture-proportion estimation (Blanchard/Scott line)
is a false friend — one scalar from two distributions vs our K-vector from
one vector; nearer single-sample analogue is RNA-seq deconvolution
(CIBERSORT, Nat Methods 2015; reference-missing-cell-type failures =
our true-donor-absent condition). InstaHide attacks (Carlini et al., ICLR
2021; Chen et al., ICLR 2021) recover identities/content from convex
mixtures but λ is handed over or assumed uniform. Deep archetypal analysis
(MIDAA, Genome Biology 2025; AAnet; Keller et al.) plants and scores
simplex weights in learned latents but over LEARNED archetypes, not real
donors. Mixup never inverts. SC-on-embeddings exists once (arXiv
2502.13322, Community Notes on Gemini embeddings) with no planted weights.
Identifiability theory to cite: Fu/Gillis sufficiently-scattered condition
(arXiv 2007.11446); Canen & Song simplex-weight inference (arXiv
2501.15692).

**Two design warnings from the econometrics line — both now resolved:**
1. **Abadie & L'Hour polytope risk — CHECKED AND DEFEATED**
   (`latent/results/uniqueness_check.json`, 2026-08-11). The risk premise
   is literally realized: at m=all=1,263 in d=768, the *affine* exact-fit
   set is 494-dimensional and the Gram is rank-deficient (no
   strict-convexity argument available). But nonnegativity + sum-to-one
   collapse it to a **single point equal to w***, LP-certified for all 20
   probed pseudo-targets: max weight assignable off the true 5-donor
   support ≤ 1.7e-12; all 25,260 per-coordinate LP ranges < 1e-9; z* sits
   on the hull *boundary*, in the relative interior of the 4-dim face
   spanned by its 5 true donors. All 11 warm starts (uniform + 10 random)
   converge to w* (max pairwise L1 5.9e-5 = ftol floor). **The m=all exact
   recovery is identification, not solver bias** — the
   sufficiently-scattered/sparse-nonnegative-recovery resolution, verified
   not assumed. Bonus separation: the m=50 QP is strictly convex (unique
   trivially) yet its unique solution is NOT w* (mean L1 0.295) — so m=50
   error is purely retrieval, m=all correctness is purely geometry; the
   two error sources are now cleanly disjoint.
2. **Ferman (JASA 2021)**: reconstructing the target vector and recovering
   w are different events — report reconstruction error and weight error
   separately (our tables already do; the distinction is now stated).

## Storage layout (as approved)

Raw data outside the repo, personal access only; repo holds code, docs, and
gitignored derived artifacts (covered by the existing `.gitignore`):

- `/data/wang/junh/datasets/PDS/NCT00174655/` — zip + `seq_patient/` +
  `pds_full/` (after registration). `chmod 700`.
- `/data/wang/junh/datasets/NACC/{UDS,iconect}/` — when granted. `chmod 700`.
- `/data/wang/junh/githubs/{PyTrial,TWIN,synthetic_control_nacc}` — reference
  clones (only PyTrial has a license — BSD-2; vendor nothing from TWIN).
- `preliminary/data/derived/trial_nct00174655/`, `results/trial_nct00174655/`,
  `figures/trial_nct00174655/`, `data/derived/nacc/` — all gitignored.
- New keys `pds_root`, `nacc_root` go in `config/paths.yaml` (gitignored).

## Actions that are Jun's to take — **CLOSED 2026-08-11, all moot**

The preliminary wrapped and the advisor moved the project to the satellite
dataset: PDS registration, the NACC-UDS request, and the I-CONECT fee
decision are all no longer needed. Kept below for the record only.

## (superseded) Actions that were Jun's to take

1. **Register on Project Data Sphere** (free, ~48 h): needed for the full
   994-patient package, the data dictionary, and formally for continued use
   of the demo data under the DUA.
2. **Submit the NACC-UDS data request** — 15 minutes: REDCap form at
   nacc.redcap.rit.uw.edu + electronic DUA (individual signature; no faculty
   sign-off documented); proposal is 4–5 sentences; data arrives within 48 h
   of approval. Before submitting, check the NACC publications list for
   overlap with Wu et al.'s hypothesis (required by their process — ours
   differs: synthesis vs matching, placebo design).
3. **I-CONECT is a budget decision for the advisor meeting, not an email**:
   up to **$2,500 data transfer fee** via the MGH REDCap form (cognitive/
   clinical data only; skip audio/video and its extra fee + 30–90 day PHI
   DUA). The NACC-only placebo experiment needs no I-CONECT and can start as
   soon as NACC delivers.
