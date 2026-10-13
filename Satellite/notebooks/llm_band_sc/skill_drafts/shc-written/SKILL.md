---
name: shc-written
description: Use this skill to forecast one land parcel's satellite band values for the next month from its own nine-month history by implementing the synthetic historical control method yourself in Python and running it. It is for a one-month-ahead forecast of a single parcel from its own past when no fitting code is provided, not for forecasts that use other parcels.
---

## Purpose

Forecast the mean value of every band of one land parcel for P10 from the parcel's own months P01 to P09 with a program you write. The parcel's past is its comparison. Blocks of its own earlier months are weighted to reproduce its most recent months, and what followed those blocks, weighted the same way, becomes the forecast. `references/method.md` describes the construction. Every setting and every choice in it is yours, made inside your program.

## When to use

- One parcel's monthly band values are to be forecast one month ahead from its own past.
- No fitting code is provided and the estimator is to be implemented as your own program.

## When not to use

- A fitter for the method is provided and only its settings and choices are open. The skill `shc-provided` covers that case. It is not present in this working directory.
- Other parcels are to serve as comparisons. This method uses only the parcel's own past.

## Environment Setup

- Python 3.11 with numpy 2.4.6, scipy 1.17.1 and pandas 3.0.5, and the Python standard library. Nothing else is available.
- Nothing can be installed. The working directory has no network and nothing outside it is visible.
- No environment variable or configuration file is needed.
- Pre-flight check, run from the skill directory.

```bash
python -c "import numpy, scipy, pandas; print(numpy.__version__, scipy.__version__, pandas.__version__)"
python scripts/show_data.py --help
```

The first command prints `2.4.6 1.17.1 3.0.5`. The second prints the arguments of the data viewer and exits with status 0.

## Inputs

| name | type and format | required | example |
|---|---|---|---|
| `data.csv` | CSV with one row per band per month and columns `band`, `period_id` (P01 to P09 in time order), `month` (YYYY-MM) and `mean_value` | required, already in the working directory | `examples/data.csv` |

The program sees P01 to P09 only. The values of P10 are withheld and no file holds them.

## Outputs

| output | format | key fields | size |
|---|---|---|---|
| `predict.py` | Python program run as `python predict.py`, taking `--data PATH` with `data.csv` as its default | prints one line per band, the band name as written in `data.csv`, a space, and the predicted mean value for P10 | one line per band |
| `answer.json` | JSON object | `prediction` with a number for every band under the band names `data.csv` uses, `rationale` with what the program does and why in a few sentences | one object |

## Procedure

Steps 1 to 3 look at the data and the method. Steps 4 and 5 write the program and check it on the synthetic example in `examples/`. Steps 6 to 8 forecast the parcel in `data.csv`. Run every command from the skill directory.

1. Run the pre-flight check of Environment Setup. If a version differs or the help text does not appear, report it in the rationale of `answer.json` and continue.

2. Look at the parcel.

   ```bash
   python scripts/show_data.py
   ```

   It prints nine lines, P01 to P09 with every band, and names the month to predict. The same values are in `data.csv`. Check that there are nine months and one value per band in each. If the script refuses `data.csv`, its message names the problem (see Troubleshooting) and the parcel cannot be forecast.

3. Read the method and decide the design.

   ```bash
   cat references/method.md
   ```

   Decide how to build the blocks, what to match, how to fit and constrain the weights, how to select blocks, and how to form the prediction.

4. Write `predict.py`. The lines below read the data into a table with one row per month and one column per band. The rest of the program is yours.

   ```bash
   cat > predict.py <<'EOF'
   import argparse
   import pandas as pd

   ap = argparse.ArgumentParser(description="Forecast P10 of one parcel with the synthetic historical control.")
   ap.add_argument("--data", default="data.csv", help="path of the parcel's data file (default data.csv)")
   d = pd.read_csv(ap.parse_args().data)
   bands = list(dict.fromkeys(d["band"]))
   table = d.pivot(index="period_id", columns="band", values="mean_value").loc[sorted(set(d["period_id"])), bands]
   # table: rows P01 to P09 in time order, one column per band
   # ... your implementation of the method in references/method.md ...
   # for band in bands: print(band, value)
   EOF
   ```

5. Check the program on the synthetic example, if it can run under the design described in `examples/README.md`.

   ```bash
   python predict.py --data examples/data.csv
   ```

   Under that design a correct program prints the values in `examples/expected_output.json` within 1e-6. If it prints other values under that design, the block construction or the weighting is wrong. Inspect the blocks and the weights your program forms and correct it. Under any other design the values differ from the file.

6. Run the program on the parcel.

   ```bash
   python predict.py
   ```

   A run succeeded when the exit status is 0 and there is one line for each band in `data.csv` with a finite number. If it fails, fix what fails and run again. Every run is recorded.

7. Decide whether to revise. A design that fits P01 to P09 better does not necessarily predict P10 better. If you judge another design right, change the program and repeat step 6.

8. Write `answer.json`.

   ```bash
   cat > answer.json <<'EOF'
   {"prediction": {"<band>": <value>, ...}, "rationale": "<what the program does and why>"}
   EOF
   ```

   The task ends when a valid `answer.json` exists.

## Troubleshooting

| symptom | cause and remedy | fallback |
|---|---|---|
| `ModuleNotFoundError: No module named '...'` | only numpy, scipy, pandas and the standard library exist here. Rewrite the program with those. | none needed |
| `KeyError` naming a band or a column | the program uses a name that `data.csv` does not hold. Read the band names from the `band` column and the columns from the file's header. | none needed |
| an optimizer that reports `success = False`, or weights that do not meet the constraint you chose | the weight problem did not converge. Check the optimizer's result before using its weights, and change the starting point, the tolerance or the solver. | submit the forecast you can justify from the data and state in the rationale that the fit failed |
| `[killed: exceeded 120 s]` after the output | every command is stopped after 120 seconds. Reduce the work, for instance the number of block subsets tried. | none needed |
| `answer.json: prediction must give a number for every band of this parcel, under the band names the parcel's data uses` after your command | `answer.json` lacks a band or names one differently. Use exactly the band names in `data.csv`. | none needed |

## Validation

- Under the design of `examples/README.md` the program reproduces `examples/expected_output.json` within 1e-6.
- `python predict.py` exits with status 0 and prints one line with a finite number for each band that `data.csv` holds.
- `answer.json` is valid JSON with a number for every band under the names in `data.csv` and a rationale. This command prints its band names and the start of its rationale.

```bash
python -c "import json; a = json.load(open('answer.json')); print(sorted(a['prediction']), a['rationale'][:80])"
```

## Related Skills

- `shc-provided` makes the same forecast with a bundled fitter whose settings and choices are open. It is not present in this working directory.
