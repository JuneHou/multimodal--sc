# Synthetic Control ATT Starter (Abadie-Diamond-Hainmueller)

## Purpose

Estimate the counterfactual outcome of a single treated unit in each treated
period and the average treatment effect on the treated (ATT) from a panel of
that unit plus untreated donor units, using the synthetic control method of
Abadie, Diamond and Hainmueller (2010) via the `Synth` R package. The skill
turns a `*_panel.csv` file into the required `answer.json`
(`counterfactual`, `att`, `rationale`).

## When to use

- You have panel data with one treated unit, several untreated donor units, and
  a clear pre-treatment window over which the donors can be weighted to
  reproduce the treated unit.
- The intervention is a discrete event affecting a single unit (a state, region,
  firm, or country), and you want a transparent, donor-weighted counterfactual.
- The task names the treated periods explicitly (or they are inferable from a
  `treatment_group` contrast).

## When not to use

- Many treated units with a common intervention time: use a staggered/multi-unit
  difference-in-differences or synthetic DiD design instead of a single-unit SCM.
- You only need an average effect over a randomized/observational cohort with a
  single cross-section: use `dowhy-average-treatment-effect-starter`.
- You have fewer than two donor units, or no pre-treatment periods to fit — the
  estimator will refuse to run.
- The panel has time-varying covariates that also change because of treatment;
  those must be excluded or handled explicitly before fitting.

## Environment Setup

Required runtime and packages:

- R >= 4.3 (validated on R 4.3.3).
- `Synth` (validated on 1.1-10) and `jsonlite` (validated on 2.0.0).
- Python >= 3.10 for the helper scripts (standard library only).

Install the R packages (run once):

```bash
Rscript -e 'install.packages(c("Synth","jsonlite"), repos="https://cloud.r-project.org")'
```

If you must pin the exact paper software, install the bundled source copy:

```bash
R CMD INSTALL code/Synth        # the Synth 1.1-10 source tree shipped in this repo
```

Pre-flight check (all three must print a version and the two `TRUE`s):

```bash
Rscript --version
Rscript -e 'cat(as.character(packageVersion("Synth")), as.character(packageVersion("jsonlite")),
  requireNamespace("Synth", quietly=TRUE), requireNamespace("jsonlite", quietly=TRUE), sep="\n")'
python3 --version
```

Environment variables: none required. If `Rscript` is not on `PATH`, prepend the
directory that contains it, e.g. `export PATH="$R_HOME/bin:$PATH"` (or your
conda env's `bin` directory), then call
`Rscript scripts/estimate_synthetic_control.R ...`.

Offline fallback: `Synth`'s only hard runtime dependencies are `kernlab`,
`optimx`, and `rgenoud`. If CRAN is unreachable, install the vendored
`code/Synth` source tree plus those three packages from any local mirror, or use
the `--allow-fallback` donor-mean path (documented under Troubleshooting) to
produce a transparent, if less faithful, counterfactual.

## Inputs

| Name | Type / format | Required | Example |
| --- | --- | --- | --- |
| panel file | CSV named `*_panel.csv` | yes | `tobacco_panel.csv` |
| `unit_id` column | string | yes | `treated`, `donor03` |
| `period` column | numeric | yes | `1989` |
| `treatment_group` column | string (`treated`/`donor`) | yes | `donor` |
| `Z1` column | numeric predictor | yes | `1.899` |
| `Z2` column | numeric predictor | yes | `16.672` |
| `Y_observed` column | numeric outcome | yes | `39.7956` |
| treated periods | comma list, `--treated-periods` | no | `1989,1990,1991` |
| treated unit id | string, `--treated-unit` | no (default `treated`) | `treated` |
| pre-treatment window | `--pre-period-start`/`--pre-period-end` | no | `1980`/`1988` |

The panel must be one row per unit and period with at least two donor units and
at least two pre-treatment periods. Periods missing for any unit are dropped
during fit so the Synth matrices stay aligned.

## Outputs

- `answer.json` (path set by `--out`, default `./answer.json`):
  ```json
  {
    "counterfactual": {"1989": 29.235, "1990": 31.4303},
    "att": -7.4455,
    "rationale": "Synthetic control ... ATT = mean(observed - synthetic) ..."
  }
  ```
  - `counterfactual`: one finite number per requested treated period (keys are
    the period labels as strings).
  - `att`: finite number, the mean of `observed - counterfactual` over the
    requested periods.
  - `rationale`: non-empty free text describing the fit and donor weights.
- summary diagnostics (path set by `--summary-out`, optional): donor weights
  (`solution_w`), predictor weights (`solution_v`), `pre_treatment_mspe`,
  per-period `observed`/`counterfactual`/`gaps`.

## Procedure

Run all commands from your working directory (the one holding the panel file).
Paths are relative and portable; do not hardcode repository or home paths.

1. Locate the panel and confirm it parses. If the directory holds exactly one
   `*_panel.csv`, the estimator auto-discovers it; otherwise pass `--panel`.

   ```bash
   ls *_panel.csv
   python3 scripts/inspect_panel.py --panel tobacco_panel.csv --treated-unit treated
   ```

   Produces a JSON report with `unit_count`, `donor_count`, `treated_unit`,
   `common_period_count`, `blocking_issues` and `warnings`.
   Decision: if `status` is `error` or `blocking_issues` is non-empty, fix the
   panel (missing columns, fewer than two donors, non-numeric outcome) before
   proceeding. Warnings about unbalanced periods are acceptable.

2. Decide the treated periods and pre-treatment window. If the treated unit rows
   carry `treatment_group == "treated"` for exactly the post-intervention
   periods, the estimator infers them; otherwise pass `--treated-periods`.

   ```bash
   # Option A: let the estimator infer a genuine treated-period contrast
   Rscript scripts/estimate_synthetic_control.R --panel tobacco_panel.csv --out answer.json
   # Option B: name the treated periods explicitly (required when treatment_group
   # is a unit-level label with no within-period contrast)
   Rscript scripts/estimate_synthetic_control.R \
     --panel tobacco_panel.csv \
     --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000 \
     --out answer.json --summary-out sc_summary.json
   ```

   The estimator fits `dataprep()` + `synth()`, projects the donor-weighted
   `Y0plot` onto the treated periods, and writes `answer.json` plus stdout JSON
   `{"status":"ok","method":"synth","att":...}`.
   Decision: if stdout shows `"method":"donor-mean-fallback"`, the Synth fit
   failed and a transparent donor mean was used — inspect `stderr` and
   `sc_summary.json.synth_error` before trusting the number.

3. Inspect the fit quality in the summary. A pre-treatment MSPE near zero and
   donor weights concentrated on a few units indicate a faithful synthetic
   control.

   ```bash
   python3 - <<'PY'
   import json
   d = json.load(open("sc_summary.json"))
   print("MSPE:", d["pre_treatment_mspe"], "ATT:", d["att"])
   print("top donors:", sorted(d["solution_w"].items(), key=lambda kv: -kv[1])[:5])
   PY
   ```

   Decision: if MSPE is large (comparable to the outcome variance) or no donor
   weight exceeds ~0.05, the donor pool may not span the treated unit. Add
   `--lag-periods` (e.g. `--lag-periods 1987,1988`) to include lagged outcomes as
   predictors, tighten the pre-treatment window, or revisit the predictor set.

4. Validate the answer against the contract, including the ATT consistency check
   against the panel.

   ```bash
   python3 scripts/validate_answer.py \
     --answer answer.json --panel tobacco_panel.csv \
     --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000
   ```

   Prints `{"status":"ok", "att":..., "computed_att":..., "problems":[]}`.
   Decision: if `status` is `error`, read `problems` and repair `answer.json`
   (missing periods, non-numeric counterfactual) or rerun the estimator.

5. (Optional, recommended for selectivity) Run the robustness ladder and confirm
   the ATT is stable:

   ```bash
   Rscript scripts/estimate_synthetic_control.R --panel tobacco_panel.csv \
     --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000 \
     --lag-periods 1987,1988 --out answer_lag.json --summary-out sc_summary_lag.json
   Rscript scripts/estimate_synthetic_control.R --panel tobacco_panel.csv \
     --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000 \
     --genoud --out answer_genoud.json
   ```

   Decision: if the ATT from the default, lag-augmented, and `--genoud` runs
   agree within a few percent, report the default run. If they diverge wildly,
   the fit is fragile — prefer the specs with the lowest pre-treatment MSPE and
   state the sensitivity in `rationale`.

### Realistic usage example

Recover the effect of a tobacco-control-style intervention on a single unit:

```bash
mkdir -p run && cd run
# Build a deterministic panel whose true effect is -8 and regenerate it locally
python3 ../scripts/make_example_panel.py --out tobacco_panel.csv --first-treated-period 1989
python3 ../scripts/inspect_panel.py --panel tobacco_panel.csv
Rscript ../scripts/estimate_synthetic_control.R \
  --panel tobacco_panel.csv \
  --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000 \
  --out answer.json --summary-out sc_summary.json
python3 ../scripts/validate_answer.py --answer answer.json --panel tobacco_panel.csv \
  --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000
python3 -c "import json; print('recovered ATT:', json.load(open('answer.json'))['att'])"
```

Expected: a pre-treatment MSPE well under 1.0, an ATT near the injected `-8`
(e.g. about `-7.45`), and `validate_answer.py` reporting `"status":"ok"`.

### Deterministic CI / smoke path

For a fast, dependency-light check that does not require a real panel, run the
bundled fixture and the validator:

```bash
Rscript scripts/estimate_synthetic_control.R \
  --panel examples/tobacco_panel.csv \
  --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000 \
  --out answer.json
python3 scripts/validate_answer.py --answer answer.json --panel examples/tobacco_panel.csv \
  --treated-periods 1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000
```

This is deterministic because the fixture and seed are fixed.

## Troubleshooting

1. **`Rscript: command not found` / `Synth` missing.**
   Symptom: `estimate_synthetic_control.R` exits non-zero with
   `This script requires the 'Synth' R package`. Remediation: install the R
   runtime and run the Environment Setup install command; verify with the
   pre-flight check. Fallback: run the estimator with `--allow-fallback` to emit
   a donor-mean counterfactual (flagged in `rationale`), or fit the vendored
   source tree with `R CMD INSTALL code/Synth`.

2. **`Panel is missing required columns ...` or non-numeric outcome.**
   Symptom: exit code 1 from `inspect_panel.py` with `blocking_issues`. Fix the
   CSV header/values so it has `unit_id,period,treatment_group,Z1,Z2,Y_observed`
   with numeric `period`/`Y_observed`. Re-run `inspect_panel.py` until
   `blocking_issues` is empty.

3. **`Could not infer treated periods` or `treated periods not present`.**
   Symptom: the estimator cannot find a within-period `treatment_group`
   contrast, or `--treated-periods` names periods the treated unit lacks.
   Remediation: pass `--treated-periods` with the exact treated periods, and
   confirm the periods exist in the panel via `inspect_panel.py`
   (`period_min`/`period_max`).

4. **Poor pre-treatment fit / large MSPE.**
   Symptom: `sc_summary.json.pre_treatment_mspe` is large relative to the
   outcome variance, or donor weights are diffuse. Remediation: add
   `--lag-periods` for pre-period outcomes, narrow `--pre-period-start`/
   `--pre-period-end`, adjust `--predictors`, or enable `--genoud` for a global
   pre-search. If the fit stays poor, the donor pool cannot span the treated
   unit — document this rather than trusting the ATT.

5. **`Synth fitting failed: ...` (e.g. `At least one predictor in X0 has no
   variation across control units`).**
   Symptom: Synth stops because a predictor is constant across donors.
   Remediation: drop that predictor with `--predictors Z1` (keep at least one
   varying predictor), or use `--allow-fallback`. Confirm the choice is recorded
   in `rationale`.

## Validation

- `python3 scripts/validate_answer.py --answer answer.json --panel <panel> \
  --treated-periods <list>` prints `"status":"ok"` with empty `problems` and a
  `computed_att` matching `att` (within `5e-3`, since the estimator rounds to 4
  decimals).
- `answer.json` contains exactly the requested treated-period keys, a finite
  `att`, and a non-empty `rationale`.
- On the bundled fixture the ATT is approximately `-7.45` (true effect `-8`)
  with pre-treatment MSPE below `1.0`.
- Skill-local tests: `python3 -m unittest discover -s tests -p 'test_*.py'` from
  the skill directory.

## Related Skills

- `dowhy-average-treatment-effect-starter` — cohort average treatment effects
  via DoWhy identification/refutation when you have many units and no single
  treated case.
- `pymc-bayesian-linear-regression-starter` — Bayesian regression alternatives
  for panel/outcome modeling.
- `arviz-posterior-diagnostics-starter` — posterior diagnostics when combining
  SCM with a Bayesian fit.
