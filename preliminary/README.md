# Latent Synthetic Control on Clinical Text — preliminary study

Synthetic control where the units are patients, time is the sequence of visits,
structured codes are the covariates, and the high-dimensional artifact carried
alongside each visit is the thing we ultimately want to control on. The headline
experiment fits convex donor weights on a unit's pre-period visits and forecasts
the held-out next visit, measured against conventional baselines.

This is a **preliminary study** for a larger satellite-imagery project. The
repository top level is reserved for that future work; everything here lives
under `preliminary/`.

## Layout

Three studies, side by side. Each owns its own code, outputs and notes;
`config/` and `Docs/` are shared at the root.

```
preliminary/
├── README.md               this file
├── config/                 shared
│   ├── paths.example.yaml  committable template
│   └── paths.yaml          local, gitignored
├── Docs/                   shared decision records and memos
│
├── mimic/                  the original MIMIC study, phases 0-2,
│   │                       AND the shared substrate the others import
│   ├── src/
│   │   ├── vendored/       self-contained copies from moa-clinical-rag
│   │   ├── datasets/       MIMIC -> patient/visit panels
│   │   └── synthctl/       donor weights, forecasting, baselines
│   ├── scripts/            runnable entry points
│   ├── tests/              estimator correctness suite
│   ├── data/               gitignored
│   ├── results/            gitignored
│   ├── figures/            gitignored
│   └── MOVE_NOTES.md
│
├── observable/             clinical-trial observable event-prediction
│   │                       experiment (NCT00174655), with TWIN as comparator
│   ├── src/                trial state + donor-selection schemas
│   ├── scripts/            the trial runner and the TWIN forecast extractor
│   ├── scratch_trial/      frozen record of the runs that produced results/
│   ├── results/            gitignored
│   ├── figures/            gitignored
│   └── MOVE_NOTES.md
│
└── latent/                 latent-recovery preliminary — in progress
    └── src/ scripts/ data/ results/ figures/
```

### `mimic/` — the MIMIC study and the shared substrate

Units are MIMIC patients; conditions / procedures / drugs are the covariates and
clinical notes the artifact. Phase 0 rebuilds the visit timeline, Phase 1 builds
per-visit states under two encoders (bag-of-codes and MedCPT), Phase 2 fits and
scores the synthetic control.

`mimic/src/` is also the code the other two studies import — `src.datasets.state`,
`src.synthctl.*`, `src.vendored.*` — by putting `preliminary/mimic/` on
`sys.path`. The dependency runs one way only: `mimic/` imports nothing from its
siblings.

`mimic/src/vendored/` holds embedding, feature/baseline, statistics and LLM
infrastructure copied — not imported — from
`/data/wang/junh/githubs/moa-clinical-rag`. That repo is under active edits and
some of its modules do not import cleanly, so we keep frozen copies. Every
vendored file's header records its exact source path, line range, and each
change made. **Read `mimic/src/vendored/MODULE.md`** for the API, every
hyperparameter as actually used, what was validated, and — importantly — what
was not.

### `observable/` — the clinical-trial port

The same Phase-2 estimator, unchanged, applied to a real randomised trial
(NCT00174655: 977 breast-cancer patients, 9,092 visits, from PyTrial's
`seq_patient` release). The state is the observable event record — medications
and adverse events, 377 binary dims per visit — and treatment is deliberately
excluded, because it is the intervention the design holds fixed. TWIN is the
comparator, both as a fidelity ceiling and, via its prediction head, as a fair
forecaster.

New here: the trial state loader and the `dot` / `cosine` / `euclid`
donor-selection schemas that lift TWIN's retrieval rule to the patient level.
The estimator itself is imported from `mimic/`, not reimplemented. See
`observable/MOVE_NOTES.md` for the import layout.

### `latent/` — in progress

Skeleton only.

## Setup

Use the existing environment; the default `python` on PATH has none of the
dependencies (vllm 0.11.0, torch 2.8.0, transformers 4.57.1, faiss 1.7.4,
sklearn 1.7.2, numpy 1.26.4):

```bash
PY=/data/wang/junh/envs/medrag/bin/python

cp preliminary/config/paths.example.yaml preliminary/config/paths.yaml
# edit paths.yaml if your MIMIC roots differ

$PY preliminary/mimic/scripts/smoke_vendored.py     # expect 4/4 PASS, exit 0
$PY preliminary/mimic/tests/test_synthctl.py        # 16 pass, 1 documented fail
```

The smoke test needs no network, no GPU and no model download.

`config/paths.yaml` names the `mimic/` derived, results and figures directories.
`observable/` and `latent/` manage their own output directories and do not read
those keys.

Note `cvxpy` is not installed and is not to be added as a dependency; the convex
donor-weight step must be solved another way (projected gradient, or scipy).

## Data handling

All MIMIC-III / MIMIC-IV / MIMIC-IV-Note data is PhysioNet credentialed-access.
**Nothing derived from it may reach a commit.** The root `.gitignore` excludes
`data/`, `results/`, `figures/`, `config/paths.yaml`, `*.parquet`, `*.pkl` and
`.emb_cache/`. Those patterns are unanchored, so they cover the per-study
directories (`mimic/data/`, `mimic/results/`, `observable/results/`,
`observable/figures/`, `latent/…`) exactly as they covered the flat layout.
Embedding caches count as derived data — they are keyed by note text and are
just as restricted as the raw tables. `config/paths.example.yaml` contains no
data and is committable.

There is **no git repository here yet** by design. The `.gitignore` is written
in advance so that whenever one is initialised, the exclusions are already in
place and no restricted file is ever staged.

## Status

`mimic/`: phases 0-2 complete, vendoring smoke-tested, estimator correctness
suite in `mimic/tests/`.

`observable/`: the NCT00174655 run is complete; results and figures under
`observable/`.

`latent/`: in progress.

The feasibility memo — whether latent synthetic control on clinical text is
worth carrying to the satellite-imagery setting — will land in `Docs/`.
