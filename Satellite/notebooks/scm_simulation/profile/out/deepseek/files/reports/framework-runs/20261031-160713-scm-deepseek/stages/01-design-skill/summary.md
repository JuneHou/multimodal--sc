# design_skill — Synthetic Control ATT Starter

## Task

Estimate the effect of a treatment on one treated unit from panel data
(`*_panel.csv` with `unit_id,period,treatment_group,Z1,Z2,Y_observed`) using the
synthetic control method of Abadie, Diamond & Hainmueller, writing
`answer.json` = `{counterfactual, att, rationale}`.

## Outcome

Built `skills/statistical-and-machine-learning-foundations-for-science/synthetic-control-att-starter`.

- `scripts/estimate_synthetic_control.R` — Synth `dataprep()`+`synth()` estimator;
  projects the donor-weighted `Y0plot` to the treated periods, computes ATT.
- `scripts/inspect_panel.py` — panel validation with blocking issues/warnings.
- `scripts/make_example_panel.py` — deterministic panel with a known effect.
- `scripts/validate_answer.py` — contract + ATT-consistency validation.
- `examples/tobacco_panel.csv`, `assets/toy_answer.json`, `assets/toy_summary.json`.
- `assets/robustness_sweep.sbatch` — Slurm job-array robustness ladder.

## Verified results

- `Rscript ... --panel examples/tobacco_panel.csv --treated-periods 1989..2000`
  → ATT ≈ -7.45 (true effect -8), pre-treatment MSPE < 1.
- `python3 scripts/validate_answer.py ...` → `"status":"ok"`, empty problems.
- `make smoke-synthetic-control` → `{"status":"ok","method":"synth","att":-7.4455}`.
- Skill tests: 6 passed. Repo smoke: 2 passed.
- `python3 scripts/validate_repository.py` → 377 resources, 268 skills.
- `python3 scripts/audit_skill_suite.py` → 268/268 smoke-covered.

## Pre-existing (unrelated) integration failures

- `test_audit_skill_suite_reports_full_coverage` and
  `test_stage_model_routing_uses_expected_models_and_efforts` fail identically on
  the unmodified checkout.
- `test_skill_review_scripts` benchmark cases error on missing Slurm envs
  (`slurm/envs/*/bin/python` not present in this sandbox).
