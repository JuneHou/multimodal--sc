---
name: synth-single-treated-unit-scm
description: Estimate single-treated-unit synthetic-control counterfactuals from a local panel CSV with Synth and write answer.json for the requested treated periods.
---

## Purpose
Estimate the untreated counterfactual path for one treated unit from panel data with the Abadie, Diamond, and Hainmueller synthetic control method. This skill is designed for tasks that already have one local `*_panel.csv` and need an `answer.json` containing period-level counterfactuals plus the average treatment effect on the treated.

## When to use / When not to use
Use it when:
- There is exactly one treated unit, identified as `treated`.
- You have a balanced panel with one row per `unit_id` and `period`.
- The donor pool contains at least two untreated units.
- The task names the treated periods that need counterfactual estimates.

Do not use it when:
- Treatment timing differs across units or there are multiple treated units.
- You need staggered-adoption estimators, interactive fixed effects, or inference over many placebo reassignments.
- The panel is unbalanced, has missing treated-period outcomes, or does not have the required `Z1`, `Z2`, and `Y_observed` columns.
- The task needs current external data, package downloads, or online reference lookup. This skill is offline once `Synth` is installed.

## Environment Setup
Minimum runtime:
- `R >= 4.3.3`

Required packages:
```bash
Rscript -e 'install.packages(c("Synth", "jsonlite"), repos = "https://cloud.r-project.org")'
```

`Synth 1.1-10` was used for the verified repository run. `Synth` pulls `kernlab`, `optimx`, and `rgenoud` as dependencies. If your R installation is older or pinned to a restrictive CRAN snapshot, install into a fresh library first:
```bash
mkdir -p r-lib
Rscript -e 'install.packages(c("Synth", "jsonlite"), lib = "r-lib", repos = "https://cloud.r-project.org")'
```

Pre-flight check:
```bash
Rscript -e 'library(Synth); library(jsonlite); cat(as.character(getRversion()), "\n"); cat(as.character(packageVersion("Synth")), "\n"); cat(as.character(packageVersion("jsonlite")), "\n")'
```

Offline fallback when package installation is unavailable:
```bash
R CMD INSTALL <local-synth-source>
Rscript -e 'install.packages("<local-jsonlite-source>", repos = NULL, type = "source")'
```
Use that route only when you already have local source bundles or a local checkout of the packages. The skill itself does not fetch anything from the network.

## Inputs
- `*_panel.csv`
  Type: CSV file, required.
  Required columns: `unit_id`, `period`, `treatment_group`, `Z1`, `Z2`, `Y_observed`.
  Example value: `toy_tobacco_panel.csv`.
  Notes: one row per unit-period; `treated` must appear in `unit_id`; the panel must be balanced across units and periods.

- `treated periods`
  Type: comma-separated list passed to `--treated-periods` or repeated `--treated-period`.
  Required: yes.
  Example value: `2006,2007,2008`.
  Notes: the skill treats every period earlier than the earliest requested treated period as pre-treatment.

- `--panel`
  Type: file path, optional.
  Example value: `./toy_tobacco_panel.csv`.
  Notes: if omitted, the script auto-detects exactly one `*_panel.csv` in the current working directory.

- `--out`
  Type: file path, optional.
  Default: `answer.json`.
  Notes: must be writable.

- `--diagnostics-out`
  Type: file path, optional.
  Example value: `scm_diagnostics.json`.
  Notes: writes donor weights, pre-fit diagnostics, and observed-versus-counterfactual values for validation.

- `--treated-unit-id`
  Type: string, optional.
  Default: `treated`.
  Example value: `treated`.

## Outputs
- `answer.json`
  Format: JSON object.
  Key fields:
  - `counterfactual`: object mapping each requested treated period to the estimated untreated outcome for `treated`.
  - `att`: numeric average treatment effect over the requested periods, computed as observed minus counterfactual.
  - `rationale`: short text summary of the specification and fit quality.
  Shape guidance: one top-level object with exactly those three keys.

- Optional diagnostics JSON from `--diagnostics-out`
  Format: JSON object.
  Key fields:
  - `panel_file`, `treated_unit_id`, `donor_units`
  - `pre_periods`, `target_periods`
  - `donor_weights`, `predictor_weights`
  - `pre_rmspe`, `att`
  - `comparison_rows`: per-period observed and counterfactual values
  Shape guidance: tens of fields at most; safe to inspect directly with `jq`.

## Procedure
1. Run a pre-flight package check before touching the panel.
   Command:
   ```bash
   Rscript -e 'library(Synth); library(jsonlite); cat(as.character(packageVersion("Synth")), "\n")'
   ```
   Output: prints the installed `Synth` version.
   Validate: the command exits `0` and prints a version string such as `1.1.10`.
   Decision: if it fails with `there is no package called`, fix the environment first; do not continue.

2. Inspect the panel structure and confirm which periods will be used as pre-treatment.
   Run this from the skill directory when using the bundled example, or from the task workspace when relying on auto-detection:
   ```bash
   Rscript scripts/run_synth_single_treated_unit_scm.R \
     --panel examples/toy_tobacco_panel.csv \
     --treated-periods 2006,2007,2008 \
     --describe-panel
   ```
   Output: JSON summary on stdout with keys including `panel_file`, `unit_count`, `period_count`, `donor_count`, `available_periods`, and `inferred_pre_periods`.
   Validate: `balanced_panel` must be `true`, `donor_count` must be at least `2`, and `inferred_pre_periods` must contain the intended untreated history.
   Decision: if the pre-period set is wrong, change the requested treated periods; if the panel is unbalanced or missing columns, repair the CSV before fitting SCM.

3. Run the actual synthetic-control estimation and write the task contract output.
   Example with the bundled dataset:
   ```bash
   Rscript scripts/run_synth_single_treated_unit_scm.R \
     --panel examples/toy_tobacco_panel.csv \
     --treated-periods 2006,2007,2008 \
     --out answer.json \
     --diagnostics-out scm_diagnostics.json
   ```
   Output:
   - `answer.json` with the requested counterfactuals and ATT.
   - `scm_diagnostics.json` with donor weights and pre-fit diagnostics.
   Validate:
   - `jq '.counterfactual, .att' answer.json`
   - `jq '.pre_rmspe, .donor_weights[0:3]' scm_diagnostics.json`
   Decision: if `pre_rmspe` is large relative to the scale of `Y_observed`, treat the estimate as weak and say so in downstream interpretation; otherwise continue.

4. Inspect the fit before reporting the answer.
   Commands:
   ```bash
   jq '.comparison_rows[] | select(.period == "2005" or .period == "2006" or .period == "2007" or .period == "2008")' scm_diagnostics.json
   jq '.donor_weights' scm_diagnostics.json
   ```
   Output: period-level observed/counterfactual rows and the donor-weight table.
   Validate: the last pre-treatment row should show a small treated-minus-synthetic gap; donor weights must be non-negative and sum to approximately `1`.
   Decision: if one donor dominates and pre-fit remains poor, document that the donor pool is weak; if weights and fit look sensible, use `answer.json` as the final deliverable.

5. Real task workflow for an arbitrary workspace that already contains one `*_panel.csv`.
   Command:
   ```bash
   Rscript <skill_dir>/scripts/run_synth_single_treated_unit_scm.R \
     --treated-periods 1989,1990,1991 \
     --out answer.json \
     --diagnostics-out scm_diagnostics.json
   ```
   Output: `answer.json` in the workspace plus optional diagnostics.
   Validate: `python3 -m json.tool answer.json >/dev/null` should exit `0`.
   Decision: if auto-detection fails because there are zero or multiple `*_panel.csv` files, rerun with an explicit `--panel <path>`.

6. Deterministic CI or smoke-test path.
   Command:
   ```bash
   Rscript scripts/run_synth_single_treated_unit_scm.R \
     --panel examples/toy_tobacco_panel.csv \
     --treated-periods 2006,2007,2008 \
     --out /tmp/scm-answer.json \
     --diagnostics-out /tmp/scm-diagnostics.json
   ```
   Output: deterministic JSON files suitable for assertions in unit tests.
   Validate: the example ATT should be negative and the pre-treatment RMSPE should stay close to zero.
   Decision: if the example output changes materially, investigate package-version drift or a regression in the script.

## Troubleshooting / Error Handling
- Symptom: `Expected exactly one '*_panel.csv' file in the current directory`.
  Cause: auto-detection found none or more than one candidate file.
  Remediation:
  1. Run `ls *_panel.csv`.
  2. Pick the intended panel.
  3. Rerun with `--panel <that-file>`.
  Alternate path: move into a clean workspace containing only the target panel file.

- Symptom: `Panel must be balanced`.
  Cause: at least one unit is missing one or more periods, or there are duplicate `unit_id`/`period` rows.
  Remediation:
  1. Run the script with `--describe-panel`.
  2. Check `duplicate_unit_period_rows` and `available_periods`.
  3. Deduplicate or fill the missing rows before rerunning.
  Alternate path: if the task allows, rebuild the panel upstream and export a balanced CSV.

- Symptom: `Need at least 3 pre-treatment periods` or a similar pre-period error.
  Cause: the earliest requested treated period leaves too little untreated history.
  Remediation:
  1. Confirm the treatment start date.
  2. Request later treated periods, or provide a longer historical panel.
  3. Rerun `--describe-panel` to verify the inferred pre-period set.
  Alternate path: if only one or two pre-periods exist, route to a simpler descriptive comparison rather than SCM.

- Symptom: `there is no package called 'Synth'`.
  Cause: package missing from the active R library.
  Remediation:
  1. Install with `Rscript -e 'install.packages(c("Synth", "jsonlite"), repos = "https://cloud.r-project.org")'`.
  2. If internet install is blocked, install from a local source checkout with `R CMD INSTALL <local-synth-source>`.
  3. Re-run the pre-flight check.
  Alternate path: stop and report the environment issue explicitly; do not silently substitute a different estimator.

- Symptom: `dataprep` or `synth` errors about no variation or malformed numeric inputs.
  Cause: one or more predictors are constant across all donors, not numeric, or the outcome history gives no usable contrast.
  Remediation:
  1. Inspect the panel columns with:
     ```bash
     python3 - <<'PY'
     import csv
     from collections import defaultdict
     panel_path = "<panel.csv>"
     values = defaultdict(set)
     with open(panel_path, newline="") as handle:
         for row in csv.DictReader(handle):
             for column in ("Z1", "Z2", "Y_observed"):
                 values[column].add(row[column])
     for column in ("Z1", "Z2", "Y_observed"):
         ordered = sorted(values[column])
         print(column, "unique_count=", len(ordered), "sample=", ordered[:10])
     PY
     ```
  2. Fix parsing issues such as quoted numbers or missing values.
  3. If the donor pool truly has no variation, expand the donor set upstream.
  Alternate path: if only `Z1`/`Z2` are degenerate but the pre-treatment outcomes vary, revise the panel generation process before using SCM.

## Validation
Successful end-to-end runs should satisfy all of the following:
- `answer.json` exists and parses as JSON.
- `answer.json` has exactly `counterfactual`, `att`, and `rationale`.
- Every requested treated period appears as a key under `counterfactual`.
- `scm_diagnostics.json` shows donor weights summing to approximately `1`.
- `scm_diagnostics.json` reports a small `pre_rmspe` relative to the outcome scale.

For the bundled example:
- Requested periods: `2006`, `2007`, `2008`
- Expected direction: negative ATT because observed smoking is below the synthetic control after treatment
- Expected fit pattern: the pre-treatment residuals stay near zero while post-treatment gaps become increasingly negative

## Related Skills
- `dowhy-average-treatment-effect-starter` for backdoor-style ATE estimation in observational tabular data.
- `pymc-bayesian-linear-regression-starter` when the task needs a probabilistic regression baseline rather than donor-pool matching.
