---
name: shc-provided
description: Use this skill to forecast one land parcel's satellite band values for the next month from its own nine-month history with the synthetic historical control method, choosing the method's settings and its methodological choices (what is matched, how the weights are constrained, how bands are weighted, how blocks are selected, how the prediction is formed) and running the bundled fitter. It is for a one-month-ahead forecast of a single parcel from its own past when the fitting code is provided, not for forecasts that use other parcels.
---

## Purpose

Forecast the mean value of every band of one land parcel for P10 from the parcel's own months P01 to P09. The parcel's past is its comparison. Blocks of its own earlier months are weighted to reproduce its most recent months, and what followed those blocks, weighted the same way, becomes the forecast. The fitter in `scripts/` carries out the method. Every setting and every methodological choice of the estimator is yours.

## When to use

- One parcel's monthly band values are to be forecast one month ahead from its own past.
- The fitting code in `scripts/` is to be used and the design of the estimator is open.

## When not to use

- The estimator is to be written as a program of your own with no fitting code provided. The skill `shc-written` covers that case. It is not present in this working directory.
- Other parcels are to serve as comparisons. This method uses only the parcel's own past.

## Environment Setup

- Python 3.11 with numpy 2.4.6, scipy 1.17.1 and pandas 3.0.5. These are the versions the fitter was checked with.
- Nothing can be installed. The working directory has no network and nothing outside it is visible.
- No environment variable or configuration file is needed.
- Pre-flight check, run from the skill directory.

```bash
python -c "import numpy, scipy, pandas; print(numpy.__version__, scipy.__version__, pandas.__version__)"
python scripts/fit.py --help
```

The first command prints `2.4.6 1.17.1 3.0.5`. The second prints the fitter's arguments and exits with status 0.

## Inputs

| name | type and format | required | example |
|---|---|---|---|
| `data.csv` | CSV with one row per band per month and columns `band`, `period_id` (P01 to P09 in time order), `month` (YYYY-MM) and `mean_value` | required, already in the working directory | `examples/data.csv` |
| `spec.json` | JSON object holding every field below, with no defaults | required, written by you | `examples/spec.json` (the synthetic example only) |

The fit sees P01 to P09 only. The values of P10 are withheld. No script can read them and no field may refer to P10.

`spec.json` holds two kinds of field. The settings are numbers. The choices are each one of a few options. A file with a field missing, a field misspelled, or a value outside its range is refused with the field named. Weights fitted on the standardized prepared series keep the observed scale only when they sum to one, so `nonneg` and `unconstrained` need `raw` or `readd_seasonal`. The fitter refuses a specification that breaks a rule and names the fields, so read the rules before writing the file.

### Settings

| field | type and range | meaning |
|---|---|---|
| `m` | integer >= 1 | length in months of the matching window and of every historical block |
| `n` | integer >= 1 | how many months after each historical block that block is taken to predict; the first of them is used |
| `kernel_bandwidth` | number > 0 | Gaussian bandwidth, in months, of the local-linear smoother that estimates each band's latent trajectory |
| `zeta` | number >= 0 | in stepwise selection a further block is kept only if it lowers the matching MSE by more than this |
| `beta_trend_degree` | 0, 1 or 2 | degree of the time trend partialled out when the annual sine/cosine seasonal terms are estimated |

### Choices

| field | options | meaning |
|---|---|---|
| `weight_constraint` | `simplex`, `nonneg`, `unconstrained` | simplex = weights >= 0 summing to 1; nonneg = weights >= 0, any sum; unconstrained = any real weights |
| `intercept` | `none`, `fitted` | none = no constant term; fitted = a constant is fitted alongside the block weights and added to the prediction |
| `band_weighting` | `equal`, `inverse_variance`, `subset` | equal = every band's rows enter the joint fit with weight 1; inverse_variance = each band's rows scaled by 1/SD of that band over P01-P09 in the fitting space; subset = fit on the bands listed in band_subset only (all bands are still predicted) |
| `fit_space` | `latent_std`, `raw` | latent_std = weights fitted on each band's smoothed, deseasonalized, standardized trajectory; raw = weights fitted on the observed band values |
| `readd_seasonal` | `true`, `false` | false = predict as the weighted combination of the observed values that followed each kept block; true = predict in the latent space, undo the standardization, and add the target month's seasonal term |
| `selection` | `greedy_forward`, `exhaustive` | greedy_forward = stepwise: best single block first, then add one at a time while the MSE falls by more than zeta; exhaustive = every non-empty subset of the blocks, lowest MSE wins (zeta unused) |
| `band_subset` | list of band names | list of band names, required if and only if band_weighting is subset |

### Rules that tie the fields together

- m + n must be at most 9, so that at least one historical block exists (blocks = 9 - m - n + 1).
- readd_seasonal = true requires fit_space = latent_std (there is no seasonal term to add back in raw space).
- intercept = fitted requires fit_space = raw or readd_seasonal = true (a constant fitted in standardized latent units has no meaning added to an observed-value prediction).
- weight_constraint = nonneg or unconstrained requires fit_space = raw or readd_seasonal = true (weights fitted in standardized latent space keep the observed values' scale only when they sum to one).
- band_subset is given if and only if band_weighting = subset, and lists band names of this sensor without repeats.
- No field may refer to P10. The fit sees P01-P09 only.

## Outputs

| output | format | key fields | size |
|---|---|---|---|
| standard output of `scripts/fit.py` | JSON object | `m`, `n`, `matching_window`, `blocks_available`, `blocks_kept` (each with `block`, `matching_months`, `forward_month`, `weight`), `intercept` (only when fitted), `matching_mse`, `prediction` (`period_id`, `month`, `values` with one number per band), all numbers to six decimals | about 30 lines |
| file given with `--out` | the same JSON object | as above | about 30 lines |
| `.fit_log.jsonl` | one JSON line per run of `scripts/fit.py` | `spec`, `donor_ids`, `weights`, `fit_mse`, `predictions` at full precision | one line per run |
| `answer.json` | JSON object | `prediction` with a number for every band under the band names `data.csv` uses, `rationale` with why this design in a few sentences | one object |

`matching_mse` is the mean squared error of the weighted blocks against the matching window, over the rows of every band in the fit and in the space the fit was made in. It is computed inside P01 to P09 and is not the error on P10.

## Procedure

Steps 1 and 2 check the environment and the fitter on the synthetic example in `examples/`. Steps 3 to 8 forecast the parcel in `data.csv`. Run every command from the skill directory.

1. Run the pre-flight check of Environment Setup. If a version differs or the help text does not appear, report it in the rationale of `answer.json` and continue.

2. Run the fitter on the synthetic example.

   ```bash
   python scripts/fit.py --data examples/data.csv --spec examples/spec.json --log example_log.jsonl
   ```

   It prints a JSON summary. `blocks_kept` should list block 2 with weight 0.7 and block 4 with weight 0.3, and `prediction.values` should equal `examples/expected_output.json` within 1e-6. If they agree, the fitter works and you continue. If they differ, continue and report the difference in the rationale of `answer.json`. `examples/README.md` explains how the example was built.

3. Look at the parcel.

   ```bash
   python scripts/show_data.py
   ```

   It prints nine lines, P01 to P09 with every band, and names the month to predict. Check that there are nine months and one value per band in each. If the script refuses `data.csv`, its message names the problem (see Troubleshooting) and the parcel cannot be fitted.

4. Read the method and decide the design.

   ```bash
   cat references/method.md
   ```

   It describes the construction and the alternatives at each step. Decide from the parcel's data and the method which space to match, how to constrain the weights, whether to fit a constant, how to weight the bands, how to select blocks, how to form the prediction, and the five settings. Check every rule under Inputs against your decision.

5. Write `spec.json` with every field. Every value in angle brackets is yours. Add `"band_subset": [...]` only with `"band_weighting": "subset"`.

   ```bash
   cat > spec.json <<'EOF'
   {"m": <int>, "n": <int>, "kernel_bandwidth": <number>, "zeta": <number>, "beta_trend_degree": <0, 1 or 2>,
    "weight_constraint": "<simplex|nonneg|unconstrained>", "intercept": "<none|fitted>",
    "band_weighting": "<equal|inverse_variance|subset>", "fit_space": "<latent_std|raw>",
    "readd_seasonal": <true|false>, "selection": "<greedy_forward|exhaustive>"}
   EOF
   python -m json.tool spec.json
   ```

   The second command prints the file if it is valid JSON. If it reports an error, correct the file before step 6.

6. Run the fitter on the parcel.

   ```bash
   python scripts/fit.py --spec spec.json
   ```

   It prints the JSON summary and appends one line to `.fit_log.jsonl`. A run succeeded when the exit status is 0 and `prediction.values` holds a number for every band. If the fitter refuses the specification, its message names every field at fault. Correct those fields and run again.

7. Decide whether to revise. The matching error is in-sample. A design that fits P01 to P09 better does not necessarily predict P10 better. If you judge another design right, write it to `spec.json` and repeat step 6. Every run is recorded.

8. Write `answer.json`. The prediction you submit is your answer. If it differs from the fitter's, say why in the rationale.

   ```bash
   cat > answer.json <<'EOF'
   {"prediction": {"<band>": <value>, ...}, "rationale": "<why this design, in a few sentences>"}
   EOF
   ```

   The task ends when a valid `answer.json` exists.

## Troubleshooting

| symptom | cause and remedy | fallback |
|---|---|---|
| `fit.py: error: the following arguments are required: --spec` with exit status 2 | the specification was not passed by name. Run `python scripts/fit.py --spec spec.json`. | none needed |
| `fit: spec.json not found; write it first` or `fit: spec.json is not valid JSON: ...` with exit status 1 | the file is missing or malformed. Write it as in step 5 and check it with `python -m json.tool spec.json`. | none needed |
| `fit: refused: ...` with exit status 1, for instance `fit: refused: missing required field(s): selection` or `fit: refused: fields 'm','n': m + n = 10 exceeds 9; no historical block would exist` | a field is missing, misspelled or out of range, or the fields break a rule that ties them together. One message lists every violation and names the fields. Correct each named field so that the tables and every rule under Inputs hold. | none needed |
| `fit: the fit could not run: SLSQP failed: ...` with exit status 1 | the weight optimizer did not converge for this design. Change the weight constraint, the intercept, the band weighting or the selection and run again. | if no design runs, submit the forecast you can justify from the data and state in the rationale that the fitter failed |
| `answer.json: prediction must give a number for every band of this parcel, under the band names the parcel's data uses` after your command | `answer.json` lacks a band or names one differently. Use exactly the band names in `data.csv`. | none needed |

## Validation

- Step 2 reproduces `examples/expected_output.json`. Blocks 2 and 4 are kept with weights 0.7 and 0.3 within 1e-6.
- The last run of step 6 exits with status 0, and `prediction.values` holds one number for each band that `data.csv` holds.
- `answer.json` is valid JSON with a number for every band under the names in `data.csv` and a rationale. This command prints its band names and the start of its rationale.

```bash
python -c "import json; a = json.load(open('answer.json')); print(sorted(a['prediction']), a['rationale'][:80])"
```

## Related Skills

- `shc-written` makes the same forecast with the estimator written as your own program and no fitting code provided. It is not present in this working directory.
