# Module: datasets

## Purpose

Phase 0 of the latent synthetic control study: rebuild the corrupted time axis of
the KARE/PyHealth-derived MIMIC-III mortality cohort so that patient trajectories
are actually chronological. Everything downstream — fitting donor weights on
early visits and forecasting later ones — is meaningless on a scrambled axis, so
this module is a blocking prerequisite with a hard validation gate.

The defect: PyHealth's `MIMIC3Dataset.parse_basic_info` sorts each patient's
admissions by `HADM_ID`, a random surrogate key with no temporal meaning
(`third_party/kare/ehr_prepare/utils.py`). KARE's
`mortality_prediction_mimic3_fn` then enumerates that list positionally, so
`visit 0 → visit N` in `pateint_mimic3_mortality.json` is a random permutation of
the true admission sequence. Compounding this,
`third_party/kare/ehr_prepare/sample_prepare.py:34-41` overwrites `visit_id` with
the visit index, severing the link back to `HADM_ID`. This module recovers that
link from raw MIMIC-III, proves the recovery against the published cohort, and
re-emits every trajectory on an `ADMITTIME` axis with recomputed labels.

## Structure

```
src/datasets/
├── __init__.py            (created by the scaffolding agent; untouched)
├── rebuild_timeline.py    all Phase 0 logic
└── MODULE.md              this file

scripts/
└── run_phase0_rebuild.py  thin CLI runner; exits non-zero if the gate fails
```

Call path: `run_phase0_rebuild.py::main` → `rebuild_timeline.run()` →
`ensure_resources` → `load_admissions` / `build_codes_by_hadm` →
`recover_visit_index_to_hadm` → `check_hand_verified_example` →
`derive_name_map` → `validate` → **gate** → `rebuild` → `quantify`.

`run()` returns a `Phase0Result` dataclass; the runner serialises it to
`data/derived/phase0_rebuild_report.json`.

Public API:

```python
run(mimic3_root=MIMIC3_ROOT, cohort_json=COHORT_JSON,
    resource_dir=DEFAULT_RESOURCE_DIR, output_path=DEFAULT_OUTPUT,
    kare_pickle=KARE_SAMPLE_PKL, cohort_mode=DEFAULT_COHORT_MODE,
    write=True) -> Phase0Result

ensure_resources(resource_dir=DEFAULT_RESOURCE_DIR) -> Path
load_resource_names(resource_dir) -> Dict[str, Dict[str, str]]
load_crossmaps(resource_dir) -> Dict[str, Dict[str, List[str]]]
load_admissions(mimic3_root=MIMIC3_ROOT) -> pd.DataFrame
build_codes_by_hadm(crossmaps, mimic3_root=MIMIC3_ROOT)
    -> Dict[str, Dict[str, List[str]]]
recover_visit_index_to_hadm(admissions, codes_by_hadm) -> Dict[str, List[str]]
rederive_visit_set(admissions, codes_by_hadm) -> Dict[str, List[str]]
load_cohort(cohort_json=COHORT_JSON) -> Dict[str, dict]
derive_name_map(cohort, recovered, codes_by_hadm, resource_names)
    -> Tuple[Dict[str, Dict[str, str]], Dict[str, dict]]
validate(cohort, recovered, codes_by_hadm, name_map,
         kare_pickle=KARE_SAMPLE_PKL, max_report=25) -> dict
check_hand_verified_example(recovered, admissions, subject="1006",
                            expected=("108462","147743","189081")) -> dict
rebuild(recovered, admissions, codes_by_hadm, name_map,
        cohort_mode=DEFAULT_COHORT_MODE) -> Tuple[pd.DataFrame, dict]
quantify(cohort, recovered, admissions, rebuilt, codes_by_hadm=None) -> dict
standardize_icd9cm(code) -> str
standardize_icd9proc(code) -> str
```

## Hyperparameters

| name | value | set where | why this value |
|---|---|---|---|
| `MATCH_RATE_GATE` | `0.99` | `rebuild_timeline.py` module constant | Specified by the Phase 0 brief. Below this the trajectory framing is abandoned rather than patched. **Actual achieved: 1.0000.** |
| mapping source | PyHealth `medcode` resource CSVs, fetched from `https://storage.googleapis.com/pyhealth/resource/` | `PYHEALTH_RESOURCE_URL`, `RESOURCE_FILES` | PyHealth is **not installed** in either `medrag` or `causal`, and `~/.cache/pyhealth/medcode` is empty. The bucket PyHealth itself downloads from is reachable, so the same six CSVs (`CCSCM`, `CCSPROC`, `ATC`, `ICD9CM_to_CCSCM`, `ICD9PROC_to_CCSPROC`, `NDC_to_ATC`) are fetched directly and cached in `data/resources/`. This is the authoritative source, not a substitute. |
| visit sort key (as built) | `HADM_ID` ascending, numeric | `recover_visit_index_to_hadm` | Reproduces PyHealth's `sort_values(["SUBJECT_ID","HADM_ID"])`. Cast to `int64` because MIMIC's `HADM_ID` is numeric and lexical string sort would differ. This is the **buggy** order, replicated deliberately so indices line up with the published keys. |
| visit sort key (as emitted) | `ADMITTIME` ascending, ties broken by `int(hadm_id)` | `rebuild` | The correction. Tie-break is arbitrary but deterministic; no ties were observed to matter. |
| drop rule (recovery) | drop the **last** hadm-sorted admission | `recover_visit_index_to_hadm` | KARE's loop is `for i in range(len(patient) - 1)`, confirmed at `utils.py:100`. |
| drop rule (emission) | drop the **chronologically last** admission | `rebuild` | Forced, not chosen: the label is the next admission's expire flag, and a patient's final admission has no next admission, so the label has no referent. This is the same rule as KARE's, applied on the correct axis. |
| code filter | `len(conditions) * len(procedures) * len(drugs) != 0`, evaluated on **post-mapping** code lists | `recover_visit_index_to_hadm` | Verbatim from `utils.py:115`. Evaluated post-mapping (not on raw row counts) because `get_code_list` returns mapped codes, so an admission whose every ICD-9 code fails to cross-walk counts as empty. |
| ATC level | 3, i.e. first 4 characters | `ATC_LEVEL`, `ATC_LEVEL_PREFIX_LEN` | KARE passes `{"NDC": ("ATC", {"target_kwargs": {"level": 3}})}` (`ehr_data_prepare.py:45`). |
| ATC name-table filter | rows with `level == "3.0"` | `ATC_LEVEL_ROW_VALUE` | Verbatim from `ehr_data_prepare.load_mappings()` (`ehr_data_prepare.py:33`). |
| crosswalk arity | **one-to-many** | `_read_crossmap` | `NDC_to_ATC.csv` has 1,143,020 rows over 791,874 distinct NDCs; 151,528 NDCs map to more than one ATC and PyHealth expands each event into one per target. Collapsing to a dict silently drops codes — this was the single largest bug found during development (it dropped the L2 match rate to 11.7%). ICD-9 crosswalks are genuinely 1-to-1. |
| ICD-9 standardisation | dot inserted after 3 chars (4 for `E`-codes) for CM; after 2 chars for PROC | `standardize_icd9cm`, `standardize_icd9proc` | MIMIC stores undotted (`40301`), the crosswalks are dotted (`403.01`). Mirrors `pyhealth.medcode.InnerMap.standardize`. |
| code dedup | first-occurrence order preserved, per admission per table | `build_codes_by_hadm` | PyHealth's `Visit.get_code_list(remove_duplicate=True)` default. Validation compares sets, so order is cosmetic; it is preserved anyway so the parquet's list columns are stable. |
| name reconciliation | constraint propagation, resource CSV as tie-break | `derive_name_map` | The cohort was built against an **older** snapshot of `CCSCM.csv` / `CCSPROC.csv` than the bucket now serves. Rather than hand-guess an alias table, the true `code → name` is solved for and then *asserted* to be injective and to reproduce every visit exactly. See "Validation performed". |
| `cohort_mode` | **`"rederived"` (default)** | `DEFAULT_COHORT_MODE`, `--cohort-mode` | Re-runs the visit selection on the corrected `ADMITTIME` axis. **This is what the parquet on disk was actually built with, and what Phase 1 and Phase 2 consume**; the default was `"kare_recovered"` until 2026-08-01, which meant a no-flag re-run did not reproduce the artifact and would have silently handed downstream modules a different, roughly half-size cohort. `"kare_recovered"` — re-sort exactly the visits KARE selected — is the literal Phase 0 spec and is still reachable via `--cohort-mode`, but it roughly halves the cohort (see Known limitations #1) because KARE's arbitrary drop and the chronological drop compound. Both are measured and reported; only the selected one is written. |
| `max_report` | `25` | `validate` | How many mismatches to retain for diagnosis. Cosmetic — zero were produced. |

## Inputs

| path | schema | provenance |
|---|---|---|
| `/data/wang/junh/datasets/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv` | reads `SUBJECT_ID`, `HADM_ID`, `ADMITTIME`, `HOSPITAL_EXPIRE_FLAG` | MIMIC-III v1.4, PhysioNet credentialed. Read-only, never copied. |
| `.../DIAGNOSES_ICD.csv` | reads `HADM_ID`, `ICD9_CODE` | same |
| `.../PROCEDURES_ICD.csv` | reads `HADM_ID`, `ICD9_CODE` | same |
| `.../PRESCRIPTIONS.csv` | reads `HADM_ID`, `NDC` (770 MB; only 2 columns loaded) | same |
| `/data/wang/junh/githubs/moa-clinical-rag/data/ehr_data/pateint_mimic3_mortality.json` | dict keyed `{subject_id}_{visit_index}`; each value `{"label": "0"/"1", "visit j": {conditions, procedures, drugs}}` as lowercased CCS/ATC English names | KARE + PyHealth output. 46 MB, 9,717 keys, 6,186 subjects. The cohort this module must reproduce. Byte-identical (md5 `bf39be6e…`) to the copy at `/data/wang/junh/datasets/KARE/ehr_data/`. |
| `/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality.pkl` | PyHealth `SampleDataset`: `samples[i] = {visit_id, patient_id, conditions, procedures, drugs, label}` plus `patient_to_index` | **Optional cross-check only.** Written by `ehr_data_prepare.py` *before* `sample_prepare.py` clobbered `visit_id`, so it still carries true `hadm_id`s. Unpickled without PyHealth via class stubs. If absent, checks L1/L2 are skipped with a warning and only L3 runs. |
| `preliminary/mimic/data/resources/*.csv` | six PyHealth medcode CSVs | Auto-downloaded on first run from the PyHealth GCS bucket; cached thereafter. Public code vocabularies, not PhysioNet data. |

## Outputs

| path | schema | who consumes it |
|---|---|---|
| `preliminary/mimic/data/derived/timeline_mimic3_mortality.parquet` | `subject_id: str`, `t: int32` (0-based chronological index, contiguous), `hadm_id: str` (unique), `admittime: datetime64[ns]`, `conditions/procedures/drugs: list<str>` (English CCS/ATC names, matching the source cohort's vocabulary exactly), `label: int8` | Phase 1+ synthetic control: donor-weight fitting on early `t`, forecasting later `t`. Note pandas materialises the list columns as `numpy.ndarray` of `object` on read, not Python lists. |
| `preliminary/mimic/data/derived/phase0_rebuild_report.json` | `{passed, validation{l1,l2,l3,rebuild_meta}, spot_check, name_drift, damage{kendall_tau, ordering, cohort_*, monotonicity_*}}` | The project memo; the ordering-damage figures are a reportable finding about KARE-derived data. |

Written parquet (default `rederived` mode — **this is the file on disk and the
one Phase 1/2 read**): **9,582 rows, 6,112 subjects, 749 patients with ≥3
visits, 348 with ≥4**, label prevalence 0.1205. Confirmed against
`phase0_rebuild_report.json → validation.rebuild_meta.cohort_mode = "rederived"`
and against the Phase 1 section below, which has always quoted these figures.

*Corrected 2026-08-01.* This paragraph previously documented the
`kare_recovered` output (6,249 rows / 3,987 subjects / label prevalence
0.1083), which is **not** the file that exists, so the Phase 0 and Phase 1
sections of this one file contradicted each other. For reference, the
`--cohort-mode kare_recovered` figures are 6,249 rows, 3,987 subjects, 0.45 MB,
label prevalence 0.1083 (677 positives), `t` ranging 0–26, and **470 / 218**
patients with ≥3 / ≥4 visits.

## How to run

```bash
/data/wang/junh/envs/medrag/bin/python \
    /data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic/scripts/run_phase0_rebuild.py
```

Runs in ~16 s (first run adds ~30 s to download resources). Exits 0 on gate pass,
1 on failure. Useful flags:

```bash
  --cohort-mode kare_recovered   # the literal Phase 0 spec; HALVES the cohort
  --no-write                # run everything, write no parquet
  --output PATH  --report PATH  --cohort-json PATH  --mimic3-root PATH
```

The default `python` on `PATH` lacks pandas; the `medrag` env path is required.

## Validation performed

The gate is three layers, ordered from most independent to most direct. **All
three returned 100.0000%.**

| layer | what it checks | result |
|---|---|---|
| L1 | Recovered `hadm_id` sequence per subject vs. the `visit_id` sequence in PyHealth's own pre-clobber pickle. Uses no names, no cohort JSON. | **6,186 / 6,186 subjects = 100.0000%** (9,717 / 9,717 visits) |
| L2 | Regenerated CCS/ATC **code** sets per `hadm_id` vs. the code sets PyHealth recorded. Uses no names, no ordering. | **9,717 / 9,717 admissions = 100.0000%** |
| L3 (the gate) | Regenerated **name** sets vs. the published `visit N` content, over every `(subject_id, visit_index, j)` entry. | **18,249 / 18,249 entries = 100.0000%**, threshold 99% |

L3 covers 18,249 entries rather than 9,717 because cohort key `{subject}_{i}`
carries `visit 0 … visit i`; every nested visit is checked, not just the last.

Hand-verified spot check (subject `1006`) reproduced exactly: visit 0 = `108462`,
visit 1 = `147743`, visit 2 = `189081`. The module additionally confirms
automatically that `108462` is the subject's **chronologically last** admission
(`2159-08-20`, `HOSPITAL_EXPIRE_FLAG=1` — the death admission) while visit 1
`147743` is the earliest (`2158-10-16`). The dropped fourth admission `199286`
(`2159-07-22`) is the one the original label was read from, as expected.

**Name-map derivation and why it is not circular.** The cohort's English names
come from an older snapshot of the CCS resource CSVs than the bucket now serves.
Drift measured: conditions 11 / 266 codes, procedures 177 / 195 codes, drugs
0 / 192 codes. Two mechanisms — stray quote characters from CSV re-quoting
(`spinal fusion"`, `"delirium`) and CCS categories that later gained a qualifier
(`pneumonia` → `pneumonia (except that caused by tuberculosis or sexually
transmitted disease)`). The true map is solved for by intersecting, per code, the
name sets of every visit containing it, then eliminating to a fixed point, with
the resource name breaking residual ties. The solver is then *asserted* to
produce an injective map that reproduces all 9,717 visits' name sets exactly —
it does, for all three sections, with zero unresolved codes. Because this
derivation uses the pickle's own `hadm_id`s, it cannot manufacture a passing L1
or L2; those two layers are name-free and independently confirm the ordering.

**Negative results and things that were wrong before they were right.**

- The first implementation loaded `NDC_to_ATC.csv` as a 1-to-1 dict. This
  silently dropped every secondary ATC code and produced L2 = **11.70%** and
  L3 = **1.37%**. The failure signature was diagnostic: `missing` was always
  non-empty and `extra` always empty, i.e. a strict subset. Fixed by loading the
  crosswalk one-to-many.
- The first monotonicity check flagged 2 patients as violations. On inspection
  both had label sequence `[1,1]`, not `[1,0]` — no death is followed by a
  survivor. These are subjects `11438` and `23523`, each of whom has **two
  consecutive admissions carrying `HOSPITAL_EXPIRE_FLAG=1`** three and thirty-six
  days apart, a MIMIC-III source anomaly unrelated to ordering. The check now
  tests for a `1` followed anywhere by a `0`, and the anomaly is counted and
  reported separately.
- **The brief's figure of 295 non-monotone sequences could not be reproduced.**
  Counting patients whose label sequence contains a death followed by a survivor
  in `pateint_mimic3_mortality.json` gives **198**, stable across three counting
  conventions (patients-with-any-violation, violating-positions,
  and immediately-followed-by-zero all give 198). For reference,
  `pateint_mimic3_readmission.json` gives 1,270 and both MIMIC-IV files give 0.
  The 198 figure is what this module reports as the pre-fix baseline; the
  discrepancy with 295 is unexplained and worth a second look by whoever
  produced it, though it does not affect the rebuild.

**What was NOT checked:**

- Nothing was validated against MIMIC-IV; this module is MIMIC-III only, and the
  `10004401` MIMIC-IV example from the brief was not re-derived.
- `ADMITTIME` is taken at face value. MIMIC-III date-shifts each patient into the
  future by a consistent per-patient offset, which preserves within-patient
  ordering and intervals (all this module needs) but makes absolute dates and
  cross-patient calendar alignment meaningless.
- The label uses `HOSPITAL_EXPIRE_FLAG` only. `DEATHTIME`, `PATIENTS.DOD`, and
  out-of-hospital deaths are ignored — as in the original task. A patient who
  dies after discharge is labelled 0 throughout.
- Age ≥ 18 is **not** enforced. KARE left this as a `TODO` and it was not added,
  so neonatal and paediatric admissions remain in the cohort.
- The readmission variant of the cohort (`pateint_mimic3_readmission.json`) was
  not rebuilt, though it has the identical defect and 1,270 non-monotone patients.
- No check that the emitted parquet is stable across pandas/pyarrow versions.
- L1/L2 depend on a pickle outside the project tree that could disappear; if it
  does, only L3 runs and the validation is materially weaker.

## Known limitations

1. **`--cohort-mode kare_recovered` roughly halves the cohort — which is why it
   is no longer the default.** Two drop rules compound: KARE already removed one
   arbitrary (hadm-last) admission per patient, and the corrected axis removes
   the chronologically last one. For ~56% of subjects these are different
   admissions, costing 3,468 of 9,717 visits and emptying 2,199 subjects.
   Result: **≥3 visits 752 → 470, ≥4 visits 348 → 218.** The default
   `rederived` re-runs the selection on the corrected axis, adding back the
   admissions KARE dropped only because they sorted last by surrogate key, and
   gives **≥3 = 749, ≥4 = 348** (9,582 visits, 6,112 subjects) — statistically
   equivalent to the pre-fix cohort and strictly more correct. This *was* left
   as a decision for the caller, with `kare_recovered` as the default on the
   grounds that it is the literal Phase 0 spec. That was resolved on 2026-08-01
   in favour of `rederived`, because the caller had already made the choice —
   the parquet on disk, and therefore every Phase 1 and Phase 2 number, is
   `rederived` — and a default that silently emits a different cohort from the
   one the pipeline runs on is a trap, not a neutral choice.
2. **The label reaches outside the cohort.** Following the original semantics,
   the label at `t` is the expire flag of the next admission in the patient's
   *full* admission list, which may be an admission excluded by the code filter
   and therefore absent from the parquet. Consecutive rows are consequently not
   necessarily consecutive admissions, and `admittime` gaps should be used rather
   than assuming `t` is evenly spaced.
3. **Kendall's tau is undefined for single-visit patients** (4,350 of 6,186).
   Aggregate tau is computed over the 1,836 multi-visit patients only; the
   headline "19.85% of patients reordered" figure over all 6,186 is diluted by
   patients who structurally cannot reorder — among multi-visit patients it is
   **66.88%**.
4. The name map is derived per run from the cohort JSON. If the cohort file is
   replaced, the map silently re-derives against the new content; it is not
   pinned to a checksum.
5. `build_codes_by_hadm` builds Python dicts over all 4.15 M prescription rows
   and peaks at a few GB of RAM. Fine on this machine, not streaming-safe.
6. Resource CSVs are fetched over the network on first run with no checksum
   pinning. If the PyHealth bucket revises them again, the drift-resolution step
   absorbs the change silently — which is the intended robustness, but it means
   a future run could produce different names without announcing it. The
   `name_drift` block of the JSON report is the audit trail.

---

# Module: state

## Purpose

Phase 1 of the latent synthetic control study: turn each patient-visit from the
Phase 0 timeline into a numeric **state vector**, and expose the donor-pool
interface that Phase 2's synthetic-control estimator fits against. Two encoders
are produced in parallel — a binary bag-of-codes and a MedCPT dense embedding —
so Phase 2 can be run under both and the representation's contribution isolated.

The defining constraint is that states are **per-visit, never cumulative**. The
upstream repo this project inherited from stored nested prefixes (cohort key
`{subject}_{k}` carried visits `0..k`), so a "state at `t`" silently contained
every earlier visit. Under that encoding a pre-period fit is trivially good,
because the donor and target share the same accumulated history by construction.
Phase 0's parquet is already per-visit; this module consumes it row-wise, adds
no accumulation, and proves the absence of accumulation by rebuilding the `t=0`
states from a dataframe in which no later visit exists.

## Structure

```
src/datasets/
├── state.py                 all Phase 1 logic
└── MODULE.md                this file (second section)

scripts/
└── run_phase1_states.py     CLI runner; builds, validates, writes JSON report
```

Call path: `run_phase1_states.py::main` → `pick_device` →
`state.build_bag_states` → `state.save_states` → `state.validate_no_leakage` →
`state.build_medcpt_states` (→ vendored `embedding.embed`) → `state.save_states`
→ `state.build_medcpt_truncated` (measurement only) → `state.load_states` →
donor-interface exercise → `state.distance_profile` / `neighbour_overlap` /
`nearest_donors` → gate → `phase1_states_report.json`.

Public API:

```python
# --- building -----------------------------------------------------------
render_visit_text(conditions, procedures, drugs) -> str
chunk_visit_texts(conditions, procedures, drugs, tok, maxlen=512,
                  length_cache=None) -> List[str]
build_bag_states(df, min_df=5) -> (X, vocab: List[str], index: pd.DataFrame)
build_medcpt_states(df, cache_path, model_name=MEDCPT_MODEL, maxlen=512,
                    batch=32, device=None, verbose=True) -> (X, index, stats)
build_medcpt_truncated(df, cache_path, ...) -> X          # measurement only
save_states(encoder, X, index, derived_dir=DEFAULT_DERIVED, vocab=None,
            meta=None) -> (npz_path, index_parquet_path)

# --- consuming (this is what Phase 2 codes against) ---------------------
load_states(encoder, derived_dir=DEFAULT_DERIVED, scale="none") -> StateSet

class StateSet:
    encoder: str; X: np.ndarray; index: pd.DataFrame
    vocab: Optional[List[str]]; meta: dict; scale: str
    d -> int                                   # state dimensionality
    subjects -> List[str]                      # canonical id order
    n_visits(subject_id) -> int
    visit_counts() -> pd.Series
    row(subject_id, t) -> int
    state(subject_id, t) -> np.ndarray                    # (d,)
    block(subject_id, t_lo=0, t_hi=None) -> np.ndarray    # (t_hi-t_lo+1, d)
    pre_period(subject_id, T0) -> np.ndarray              # (T0+1, d)
    eligible_subjects(T0) -> List[str]                    # >= T0+2 visits
    donor_pool(T0, exclude=None) -> (np.ndarray, List[str])
                                        # ((n_donors, T0+2, d), ids)
    donor_matrix(T0, exclude=None) -> (D_pre, D_next, ids)
                                        # ((n,(T0+1)*d), (n,d), ids)

# --- validation ---------------------------------------------------------
validate_no_leakage(df, X_bag, vocab, index) -> dict
sparsity(X) -> dict
nearest_donors(ss, target, T0, k=3) -> List[(subject_id, distance)]
neighbour_overlap(ss_a, ss_b, T0, targets, k=10) -> dict
distance_profile(ss, T0, targets, k=10) -> dict
sorted_ids(ids) -> List[str]
```

## Hyperparameters

| name | value | set where | why this value |
|---|---|---|---|
| `DEFAULT_MIN_DF` | `5` | `state.py` constant, `--min-df` | The vendored `design_matrix` default, made explicit. 647 distinct codes exist in the 9,582 visits; 585 survive `min_df=5`, i.e. **62 codes dropped**. Measured effect on the geometry: top-10 donor overlap between `min_df=1` and `min_df=5` is **0.99** (min 0.90) over 40 targets, so the knob is nearly immaterial. Alternative `min_df=1` (keep all 647) is defensible and costs almost nothing; 5 is kept for consistency with the vendored baseline. |
| bag encoding | binary 0/1 presence | vendored `design_matrix` | Codes are CCS/ATC category presences, not counts; the Phase 0 parquet deduplicates within a visit anyway. |
| section prefixes | `c:` / `p:` / `d:` | vendored `_bag` | Keeps a condition and a drug of the same English name distinct. |
| bag order passed to `design_matrix` | **sorted lists**, not sets | `build_bag_states` | Not cosmetic. `design_matrix` builds its vocabulary from `Counter.items()`, i.e. first-seen order; iterating a `set` of `str` makes that order depend on `PYTHONHASHSEED`, so the saved column order (and therefore the matrix) would differ between processes. A deduplicated sorted list gives `Counter` identical document-frequency counts while pinning the order. Verified: identical sha1 of `X` and of the vocab across two separate interpreter runs. |
| `MEDCPT_MODEL` | `ncbi/MedCPT-Article-Encoder` | `state.py` constant | Specified by the brief; 768-d, CLS pooling, L2-normalised. Article encoder (not Query) because a visit is a document, not a query. **No substitution was made.** |
| MedCPT pooling / normalisation | CLS at position 0, L2-normalise, eps `1e-12` | vendored `embedding.embed` | Unchanged from the vendored implementation. |
| `MEDCPT_MAXLEN` | `512` | `state.py` constant | Hard ceiling, not a choice: this checkpoint's `max_position_embeddings` is 512. |
| `MEDCPT_BATCH` | `32` | `state.py` constant | Vendored default; the whole job is ~10.5k short texts. |
| chunking + mean-pooling | on, greedy pack in section order | `chunk_visit_texts` | **The brief's premise that the texts are short enough for 512 not to bind is wrong** — 927 of 9,582 visits (9.7%) exceed 512 tokens, max 977. See "Validation performed". Codes are packed greedily, in section order, into chunks that fit; each chunk is embedded and the chunk vectors are mean-pooled and re-L2-normalised. A one-chunk visit (90.3% of them) yields a bit-identical vector to plain embedding, so this is a strict superset of the naive path. Alternative considered and rejected: plain truncation, which would always drop the tail of the **drug** list (drugs render last), i.e. systematically blind the state of exactly the poly-pharmacy patients. |
| chunk token budget | `512 - 2 - Σ(len("<Header>:") + 1)` | `chunk_visit_texts` | Reserves `[CLS]`/`[SEP]` plus one header and terminator per section. Conservative; the realised max chunk length was **509 ≤ 512**, asserted at build time, not assumed. |
| text rendering | `Conditions: a; b. Procedures: c. Medications: d.` | `render_visit_text`, `SECTIONS` | The brief's format. Empty sections render as `"<Header>: none."` rather than being dropped, so the string shape is constant. |
| `<n>` tag stripping | on, **embedding text only** | `_NTAG` in `state.py` | The ATC name table carries XML-ish numeric markup (`vitamin b<n>12</n> and folic acid`) in 3,407 of 9,582 visits. Stripped for the rendered text because it tokenises as junk (`<`, `n`, `>` are separate word-pieces); the **bag-of-codes vocabulary keeps the raw string verbatim**, because there the string is an identifier, not prose. Alternative: leave it in — costs ~5 tokens/visit of noise and changes nothing structurally. |
| `scale` (state scaling) | `"none"` (default) | `load_states(scale=...)`, `SCALINGS` | See "Scaling" below. Options `none` / `global` / `zscore`. |
| `STATE_DTYPE` | `float32` | `state.py` constant | Brief's contract. The bag matrix is 0/1 so float32 is exact; MedCPT's cache is already float32. |
| id ordering | numeric ascending (`int(subject_id)`), non-numeric after, lexical | `_id_key`, `sorted_ids` | MIMIC subject ids are numeric strings, where `"1004" < "10004"` numerically but not lexically. Every `(n_donors, ...)` array is ordered by this, so it is pinned rather than left to `str` sort. Asserted in the runner. |
| row order of the saved matrix | canonical id order, then `t` | `run_phase1_states.py` | Not load-bearing (`StateSet` re-sorts by `t` per subject) but makes the artifact byte-reproducible. |
| `T0` (primary) | `1` | `--t0` | Brief's primary config: fit on `t=0,1`, forecast `t=2`. |
| donor eligibility | `>= T0+2` visits | `eligible_subjects` | A unit needs `t=0..T0` to fit on plus a state at `t=T0+1` — the donor's value to forecast *from*, or the target's held-out truth. At `T0=1` that is the 749-patient `>=3`-visit cohort. |
| `--seed` | `0` | runner | Seeds only the sampling of targets for the neighbour/smell diagnostics. No modelling randomness exists in this module; both encoders are deterministic. |
| device selection | auto: emptiest GPU if `>= 8000 MiB` free, else CPU | `pick_device`, `--device` | GPUs on this box are contended. Selection is done by setting `CUDA_VISIBLE_DEVICES`, because the vendored `embed` only understands the literal string `"cuda"` (see "Known limitations" #3). **As run: physical `cuda:4`, 25,487 MiB free.** CPU is a supported fallback and would take a few minutes for this workload. |
| embedding cache | `data/derived/.emb_cache/medcpt_visits.pkl` | runner | Vendored `embed`'s content-addressed cache, keyed by sha1 of the chunk text. Derived from MIMIC, therefore gitignored like all of `data/`. |

## Inputs

| path | schema | provenance |
|---|---|---|
| `preliminary/mimic/data/derived/timeline_mimic3_mortality.parquet` | `subject_id: str`, `t: int32`, `hadm_id: str`, `admittime: datetime64[ns]`, `conditions/procedures/drugs: list<str>`, `label: int8` | Phase 0 of this module (section 1 above), `rederived` cohort mode. 9,582 rows, 6,112 subjects, 749 with `>=3` visits, 348 with `>=4`, label prevalence 0.1205. Read-only. |
| `ncbi/MedCPT-Article-Encoder` | HF BERT encoder, 109.5M params, hidden 768, `max_position_embeddings` 512 | **Already present in the local HF cache** at `$HF_HOME=/data/wang/junh/.cache/huggingface/models--ncbi--MedCPT-Article-Encoder`, snapshot `d05a736da4bb84ee4057b7f7999485be6ed85465`, 419 MB. Loaded with `HF_HUB_OFFLINE=1`; **no download was needed or attempted**. |

**Deliberately not used:** `moa-clinical-rag/data/base_context_qwen/patient_embeddings_*.pkl`. They are 4096-d of unverified provenance — the builder calls an OpenAI-style API while the directory is named `_qwen`, and the associated paper claims MedCPT at 768-d. This module re-embeds from scratch.

## Outputs

All under `preliminary/mimic/data/derived/`. **This is the contract Phase 2 / Agent D codes against.**

| path | schema | who consumes it |
|---|---|---|
| `states_bag.npz` | `X` float32 `(9582, 585)` 0/1 · `subject_id` `<U` `(9582,)` · `t` int32 `(9582,)` · `row` int32 `(9582,)` · `vocab` `<U` `(585,)` column names in matrix-column order · `meta_json` 0-d object holding a JSON string | Phase 2 SC estimator, via `load_states("bag")` |
| `states_bag_index.parquet` | `subject_id: str`, `t: int32`, `row: int32` | convenience for pandas consumers; identical content to the npz index arrays |
| `states_medcpt.npz` | `X` float32 `(9582, 768)`, every row unit-norm · `subject_id` · `t` · `row` · `meta_json` (**no `vocab` key**) | Phase 2 SC estimator, via `load_states("medcpt")` |
| `states_medcpt_index.parquet` | as above | as above |
| `phase1_states_report.json` | `{config, cohort, bag, leakage, medcpt, medcpt_truncation, donor_pool, scaling, cross_encoder_top10_overlap, nn_smell_test, checks, passed}` | the memo; the audit trail for every number quoted below |

Row alignment: `X[row]` is the state of `(subject_id[row], t[row])`; `row` is
`arange(n)`, so the index arrays are positional. `meta_json` decodes to
`{encoder, shape, dtype, source, scaling_params{row_norm_mean, row_norm_rms,
col_mean[d], col_std[d]}, ...}` plus encoder-specific keys (`min_df`, `binary`,
`prefixed_sections` for bag; `model`, `pooling`, `l2_normalised`,
`chunked_mean_pooled`, `maxlen`, `token_stats` for medcpt).

Agent D should not need to touch the npz directly:

```python
from src.datasets.state import load_states
ss = load_states("medcpt")                      # or "bag"
Xi = ss.pre_period(target_id, T0=1)             # (2, 768)
D, ids = ss.donor_pool(T0=1, exclude=target_id) # (748, 3, 768), len(ids)==748
                                                # D[j, 0:2] pre-period, D[j, 2] forecast-from
D_pre, D_next, ids = ss.donor_matrix(T0=1, exclude=target_id)  # (748,1536),(748,768)
y_true = ss.state(target_id, 2)                 # held-out truth
```

## Scaling

Bag-of-codes rows are binary with mean L2 norm **6.518**; MedCPT rows are
L2-normalised with norm exactly **1.0**. The SC objective is a squared distance,
so this is not a free choice — but it matters less than it looks, and the parts
where it matters are separable.

**Default: `scale="none"` — matrices are stored and returned as built.** The two
matrices are consumed *separately* by Phase 2 (never concatenated into one state
vector), so no cross-encoder commensurability is required at fit time. Within an
encoder, a global positive scalar cannot change the argmin of the unregularised
SC problem nor any distance ordering. That was verified, not assumed:
`none` vs `global` top-10 donor overlap is **1.00** for MedCPT, and the two
distance vectors agree to a max relative difference of **2.4e-07** (float32 eps).

Alternatives, both implemented and reachable via `load_states(scale=...)`:

- **`"global"`** — divide by the build-time RMS row norm, so both encoders have
  mean row norm ≈ 1. Provably ordering-preserving; its purpose is to put bag and
  MedCPT *reported RMSEs* on one scale, and to let a ridge/entropy penalty λ
  tuned on one encoder transfer to the other. **Recommended for Phase 2 as soon
  as a regularised objective or a cross-encoder RMSE comparison enters**, since
  it is free of geometric consequence.
- **`"zscore"`** — per-column standardisation using build-time column moments.
  This **does** change the answer: top-10 overlap against `none` is **0.315**
  for bag and 0.940 for MedCPT. Not the default for two reasons. (i) On a binary
  matrix it is an IDF-like reweighting — a code with document frequency 5/9582
  gets ~44x the weight of a common one, so a single idiosyncratic code can
  dominate a squared-distance fit. That may well be *desirable* and is worth an
  ablation, but it is a modelling decision that belongs to Phase 2, not a
  representation default. (ii) The moments are estimated over all 9,582 visits
  including post-period ones, a transductive dependence best avoided in a
  forecasting design.
- **Not implemented:** TF-IDF weighting, per-column L2 (unit-norm columns), and
  row-wise L2 on the bag matrix (which would convert its Euclidean geometry to a
  cosine one and directly address the cardinality bias reported below). The last
  of these is the most promising and is a one-line addition if Phase 2 wants it.

## How to run

```bash
HF_HUB_OFFLINE=1 /data/wang/junh/envs/medrag/bin/python \
    /data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic/scripts/run_phase1_states.py
```

~4 minutes end-to-end on one A40 (including the extra truncation-probe embedding
pass); re-runs are near-instant because `embed` is content-addressed. Exits 0 on
gate pass, 1 on failure. `HF_HUB_OFFLINE=1` is optional but makes it explicit
that no download occurs. The default `python` on `PATH` lacks torch/pandas.

Useful flags:

```bash
  --device cpu               # do not touch the GPUs at all
  --device cuda              # force GPU even if contended
  --skip-medcpt              # bag-of-codes only, ~15 s, no model load
  --skip-truncation-probe    # skip the second embedding pass
  --min-df N  --t0 K  --n-smell N  --seed S  --report PATH
```

## Validation performed

**Gate: all 8 checks PASS.**
`{cohort_749_348, no_cumulative_leakage, donor_pool_matches_cohort,
target_excluded_from_donors, donor_ids_ordered, medcpt_dim_768,
all_chunks_within_maxlen, medcpt_rows_unit_norm}`

**MedCPT availability.** Checked *before* building anything. Found already
cached under `$HF_HOME`; loaded successfully with `HF_HUB_OFFLINE=1`, confirming
hidden size 768, `model_type` bert, 109.48M params. No download attempted, no
substitute encoder used.

**Cohort counts reproduce exactly.** 9,582 visits, 6,112 subjects, **749**
subjects with `>=3` visits, **348** with `>=4`, label prevalence 0.12054. The
donor pool built through the public API independently returns **749** eligible
subjects at `T0=1`, matching the `>=3` count.

**No cumulative leakage — the central check.** All 6,112 `t=0` states were
rebuilt from a dataframe containing *only* the `t=0` rows, against the same
vocabulary, so no later visit of any patient was visible. The result is
**bit-identical** to the corresponding rows of the full build, including for the
749 patients who do have later visits. Corroborating: mean active codes by `t`
within the `>=3`-visit cohort is 44.1 / 44.7 / 47.1 / 49.7 / 49.7 / 46.9 for
`t=0..5` — flat-to-mildly-increasing, as a per-visit encoding should be. Under
the inherited nested-prefix encoding `t=2` would be roughly 3x `t=0`.

**Dimensionality and sparsity.**

| encoder | shape | frac nonzero | mean active/row | mean row L2 |
|---|---|---|---|---|
| bag | `(9582, 585)` | 0.0760 | 44.48 | 6.518 |
| medcpt | `(9582, 768)` | ~1.0 (dense) | 768.0 | 1.000 |

**The 512-token ceiling binds — the brief's assumption was wrong.** Measured
rather than assumed, over the unchunked renderings: mean 294.6 tokens, p50 264,
p95 595, **max 977**, and **927 of 9,582 visits (9.67%) exceed 512**. Driven by
a mean of 29.2 drugs/visit with names like *"iv solutions used in parenteral
administration of fluids, electrolytes and nutrients"*. Chunking produced 10,525
chunks (10,488 unique) with max 2 chunks per visit; the realised max chunk
length is **509 <= 512**, asserted at build time.

**What truncation would have cost.** Measured directly by building the naive
truncated states as well and comparing: cosine(chunked, truncated) has mean
0.9951, **min 0.9271**, with 943 rows below 0.99 and **463 rows below 0.95**.
So for ~5% of visits the naive path would have moved the state materially, and
always in the same direction — losing the drug tail.

**Build determinism.** `build_bag_states` was run in two separate interpreter
processes; sha1 of `X` and of the vocabulary string matched exactly, confirming
the `PYTHONHASHSEED` hazard in `design_matrix` is neutralised.

**Donor-interface contract.** At `T0=1`: `donor_pool` returns `(748, 3, 585)`
with 748 ids, target excluded, ids in canonical order; `donor_matrix` returns
`(748, 1170)` and `(748, 585)`; `pre_period` returns `(2, 585)`. Row alignment
spot-checked: `D[j, T0+1] == ss.state(ids[j], T0+1)`. The npz round-trip was
asserted equal to the in-memory matrix.

**Scaling sensitivity.** Reported in "Scaling" above. One subtlety worth
recording: `none` vs `global` top-10 overlap for the **bag** encoder is 0.97,
not 1.00, which initially looked like a contradiction. It is not — it is
entirely **ties**. Binary bag-of-codes gives integer-valued squared distances, so
across 748 donors there are only ~100 distinct distance values, and **55% of
targets have an exact tie at the k=10 boundary**; a float32 rescale reorders
tied donors arbitrarily. The distance vectors themselves agree to 2.4e-07.
MedCPT has 734 distinct values across 748 donors and 0% boundary ties, hence its
1.00.

**Nearest-neighbour smell test — honest read: MedCPT looks good, bag-of-codes
is mixed-to-poor.** Top-3 donors printed for 5 random `>=3`-visit targets under
both encoders (full text in `phase1_states_report.json → nn_smell_test`).

- *MedCPT* is consistently clinically coherent. A sepsis / shock / respiratory-
  failure / ARF target drew donors who are themselves sepsis + respiratory-
  failure + ARF patients across both pre-period visits; a cardiac target
  (CAD, atherosclerosis, lipid disorders) drew valve-disorder / CHF / CAD
  donors; an aspiration-pneumonitis + epilepsy + delirium target drew
  aspiration-pneumonitis + respiratory-failure + sepsis donors. Nothing in the
  sample looked absurd.
- *Bag-of-codes* produces some genuinely good matches — target 27574 (sepsis,
  shock, respiratory failure, ARF) matched donor 4787 with almost the same
  condition set; target 14467 matched donor 4916 sharing diverticulitis +
  peritonitis. But it also produces clear failures, and they share a mechanism:
  **patients with few codes are near everyone**. Euclidean distance on binary
  indicators is `sqrt(|A Δ B|)`, so a low-cardinality bag has a small symmetric
  difference with any bag and floats to the top of the ranking regardless of
  content. Target 14520 (CHF, CKD, pneumonia, hypertension) got 3rd-nearest
  donor 3497 — a brain-cancer / epilepsy / asthma patient with 3 conditions.
  Target 10675 (liver disease, hepatitis, ARF) got nearest donor 71108 — 3
  conditions, clinically unrelated. Compounded by the tie structure above.
- The two encoders substantially disagree: **cross-encoder top-10 donor overlap
  is 0.17** (min 0.0) over 40 targets. They are not interchangeable
  representations, which is the point of building both.
- Caveat on how much this proves: this is a *1-sparse* probe (best single
  donor). SC fits a convex combination, so a representation can rank single
  donors poorly and still support a good synthetic control. Treat it as a smell
  test, not as evidence about Phase 2's outcome.

**What was NOT checked.**

- **No downstream predictive validation whatsoever.** Nothing here shows either
  state space actually supports forecasting `t=2`; that is Phase 2's job. No
  AUROC, no reconstruction error, no baseline comparison was run.
- The MedCPT embeddings were **not** compared against any external reference
  implementation of MedCPT, nor against the 4096-d `patient_embeddings_*.pkl`
  this module deliberately refuses to use. Correctness rests on the vendored
  `embed` being faithful (audited by Agent A, not re-audited here).
- Mean-pooling of chunk vectors is asserted to be a strict superset of the naive
  path only for **single-chunk** visits. For the 927 multi-chunk visits, no
  evidence is offered that mean-pooling is the *best* aggregation — only that it
  is better than discarding the tail. Max-pooling, first-chunk, and
  length-weighted means were not tried.
- The `zscore` and `global` scalings were exercised through `load_states` and
  their neighbour effects measured, but **neither was run through an actual SC
  fit**; the claim that `global` is inconsequential rests on the
  ordering-preservation argument plus the distance-agreement measurement, not on
  end-to-end results.
- `T0=2` and higher were not exercised beyond the API accepting them; only
  `T0=1` was validated end-to-end.
- Sparse storage was not used for the bag matrix (7.6% dense, 22 MB as float32,
  561 KB compressed) — fine at this scale, not checked at larger ones.
- No check that the npz artifacts are stable across numpy versions.

## Known limitations

1. **Two defects in Agent A's vendored `src/vendored/features.py`. Reported, not
   patched.**
   (a) `_flatten` does `for v in field or []`. Phase 0's parquet materialises
   its list columns as `numpy.ndarray` of object (Phase 0's own MODULE.md says
   so), and `bool(ndarray)` raises `ValueError: The truth value of an array with
   more than one element is ambiguous`. So `features._bag` **cannot consume the
   Phase 0 parquet as-is** — the two modules in this repo do not compose. Worked
   around by `state._codes()`, which coerces each cell to `list[str]` before the
   vendored call. A one-line fix upstream would be `if field is None: field = []`.
   (b) `design_matrix` builds its vocabulary from `Counter.items()` over
   `set`-valued bags, so the column order depends on `PYTHONHASHSEED` and is not
   reproducible across processes. Worked around by passing sorted lists (see
   Hyperparameters); the underlying fragility remains for any other caller.
2. **One row of Phase 0's parquet violates Phase 0's documented code filter.
   Reported, not patched.** `rebuild_timeline` documents a
   `len(conditions)*len(procedures)*len(drugs) != 0` filter, but subject
   `11723` at `t=1` (`hadm_id` 160890) has **zero procedures**. It is the only
   such row in 9,582, and that subject has 2 visits so is **not in the
   `>=3` cohort** — Phase 2 is unaffected. This module counts it rather than
   crashing (`medcpt.n_visits_with_empty_section` in the report) and renders it
   as `"Procedures: none."`. Worth a look from Agent B, since it suggests the
   filter is applied on a different code list than the one emitted in
   `rederived` mode.
3. **`embedding.embed` only understands the literal device string `"cuda"`.**
   Passing `"cuda:4"` takes the CPU branch for the model but moves the batch to
   `cuda:4`, producing a device mismatch at the forward pass. Device selection
   therefore has to go through `CUDA_VISIBLE_DEVICES`, which `pick_device` sets
   before torch is imported. Not a bug in the vendored file's own terms, but a
   sharp edge for any future caller on a multi-GPU box.
4. **The bag-of-codes geometry has a cardinality bias.** Euclidean distance on
   binary indicators makes patients with few codes near-neighbours of everyone,
   which visibly corrupts the top-3 rankings (see the smell test). Row-wise L2
   normalisation (cosine) or IDF weighting would address it; neither is the
   default, both are one-line changes. Phase 2 should ablate this before
   concluding anything about bag-of-codes as a representation.
5. **Bag distances are heavily tied** — ~100 distinct values across 748 donors,
   55% of targets tied at the k=10 boundary. Any "nearest donor" statement under
   the bag encoder is therefore weakly identified, and results that depend on
   tie-break order should be treated as noise.
6. **The vocabulary and the `zscore` moments are fit on all 9,582 visits**,
   including the post-period visits that Phase 2 holds out. The vocabulary is
   unsupervised (no labels touch it) so this is mild, but it is a transductive
   dependence, and a strict evaluation would fit both on pre-period rows only.
7. **States carry no time information.** `admittime` gaps, visit number, and the
   `label` column are all excluded from the state vector; a state is a pure
   function of that visit's codes. Phase 0 warns that consecutive `t` are not
   necessarily consecutive admissions and that gaps are irregular, so Phase 2
   should not assume `t` is evenly spaced.
8. `build_bag_states` and `build_medcpt_states` iterate the dataframe row-wise
   with `iterrows` and hold the full dense matrices in memory. Fine for 9,582
   visits; not streaming-safe.
