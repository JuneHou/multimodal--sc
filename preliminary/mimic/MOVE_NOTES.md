# MOVE_NOTES — `preliminary/mimic/`

**Date: 2026-08-11.** `preliminary/` was split three ways: `mimic/` (this
directory, the original MIMIC study plus the shared code substrate),
`observable/` (the NCT00174655 clinical-trial experiment) and `latent/` (empty
skeleton, in progress). `config/`, `Docs/` and `README.md` stayed at the
`preliminary/` root.

**Every result under this directory predates the move.** Nothing was re-run,
re-fit or regenerated. The only changes are file locations and the path strings
listed below.

## What moved here

| was | is now |
|---|---|
| `preliminary/src/` | `preliminary/mimic/src/` |
| `preliminary/scripts/` | `preliminary/mimic/scripts/` |
| `preliminary/tests/` | `preliminary/mimic/tests/` |
| `preliminary/data/` | `preliminary/mimic/data/` |
| `preliminary/results/` | `preliminary/mimic/results/` (was empty) |
| `preliminary/figures/` | `preliminary/mimic/figures/` (was empty) |

`data/` carries the whole MIMIC-derived set unchanged: the Phase 0 timeline
parquet, the Phase 1 `states_bag` / `states_medcpt` npz + index parquets, the
two phase reports, `data/resources/` (six public PyHealth medcode CSVs) and
`data/derived/.emb_cache/`.

`results/` and `figures/` held nothing but `trial_nct00174655/`, which is trial
work and went to `observable/`; no MIMIC-era output existed to move. The two
directories were carried over empty so the Phase 2 runner's output paths still
resolve.

## What left this tree

Three things that used to live under `src/` and `scripts/` are trial work and
now live in `observable/` — see `../observable/MOVE_NOTES.md`:

* `src/datasets/trial_nct00174655.py` → `observable/src/trial_nct00174655.py`
* `src/synthctl_trial/` → `observable/src/synthctl_trial/`
* `scripts/run_trial_fit.py`, `scripts/run_trial_twin_forecast.py` →
  `observable/scripts/`

`mimic/src/` is unchanged by their departure: nothing in `src/datasets/`,
`src/synthctl/` or `src/vendored/` ever imported them. The dependency runs one
way only — `observable/` imports `mimic/src`, never the reverse.

## Files edited, and the exact edit

### Code

| file | edit |
|---|---|
| `scripts/validate_phase2.py` | `ROOT` was a hardcoded absolute `.../preliminary`; now `.../preliminary/mimic`. This was the **only** absolute-path constant in the tree. Invocation line in the docstring: `scripts/` → `mimic/scripts/`. |
| `scripts/run_phase0_rebuild.py` | docstring invocation path only: `preliminary/scripts/` → `preliminary/mimic/scripts/`. |
| `scripts/run_phase1_states.py` | same docstring-only edit. |
| `scripts/smoke_vendored.py` | same docstring-only edit. |
| `tests/test_synthctl.py` | same docstring-only edit (`preliminary/tests/` → `preliminary/mimic/tests/`). |

Nothing else in `src/` or `scripts/` needed touching. Every other root is
`__file__`-relative — `Path(__file__).resolve().parents[N]` — and `data/` moved
alongside the code, so all of these still resolve, verified rather than assumed:

* `src/datasets/state.py`: `DEFAULT_DERIVED` / `DEFAULT_TIMELINE` →
  `mimic/data/derived/…`, both exist.
* `src/datasets/rebuild_timeline.py`: `DEFAULT_RESOURCE_DIR` /
  `DEFAULT_OUTPUT` → `mimic/data/{resources,derived}/…`, both exist.
  (`KARE_SAMPLE_PKL` is an absolute path into `/data/wang/junh/datasets/`,
  outside the project, and is untouched.)
* `scripts/run_phase2_fit.py`: `DERIVED` / `RESULTS` / `FIGURES` / `CACHE` →
  `mimic/…`.
* `scripts/smoke_phase2_fixes.py`, `src/synthctl/{fit,evaluate}.py`,
  `src/datasets/state.py` sys.path inserts → `mimic/`, which is now the
  directory holding the `src` package.

### Docs (mechanical path substitution, no content rewritten)

`src/datasets/MODULE.md`, `src/synthctl/MODULE.md`, `src/vendored/MODULE.md`,
`tests/MODULE.md`: `preliminary/src` → `preliminary/mimic/src`,
`preliminary/scripts` → `preliminary/mimic/scripts`, `preliminary/tests` →
`preliminary/mimic/tests`, `preliminary/data` → `preliminary/mimic/data`. One
extra line in `src/synthctl/MODULE.md`: "All paths relative to `preliminary/`"
→ "relative to `preliminary/mimic/`".

### Config (at the `preliminary/` root, shared)

`config/paths.yaml` and `config/paths.example.yaml`: `derived_dir`,
`results_dir`, `figures_dir` repointed to the `preliminary/mimic/` locations,
with a comment recording that `observable/` and `latent/` manage their own
output directories and do not read these keys. `mimic3_root` / `mimic4_root` /
`mimic4_note_root` / `source_repo` are unchanged.

## Housekeeping

`__pycache__/` directories were deleted before the move so no stale bytecode
carried a pre-move path. They regenerate on the next run.

The root `.gitignore` patterns (`data/`, `results/`, `figures/`, `.emb_cache/`,
`*.parquet`, `*.pkl`, `**/config/paths.yaml`) are unanchored, so they match at
the new depth without change. Confirmed with `git check-ignore -v` on
`mimic/data/derived/states_bag.npz`, `mimic/data/resources/ATC.csv`,
`mimic/data/derived/.emb_cache/medcpt_visits.pkl`, `mimic/results`,
`mimic/figures`.

## Verification run after the move

* `mimic/scripts/smoke_vendored.py` → `4/4 modules PASS`, exit 0.
* `mimic/tests/test_synthctl.py` → `16 passed, 1 failed, 0 skipped`, the
  outcome `tests/MODULE.md` records as expected. The one failure is
  `test_permutation_invariance_bitwise` — SLSQP's absolute `ftol` pinning `w`
  only to ~`sqrt(ftol)` — which is a documented numerical property of the
  solver, is unrelated to file layout, and failed identically before the move.
  The I/O guard still fires: it matches forbidden path *components*
  (`derived`, `figures`, `results`) and file suffixes, not absolute prefixes,
  so the extra `mimic/` level does not affect it, and `test_io_guard_blocks_
  project_paths` passes from the new location.
