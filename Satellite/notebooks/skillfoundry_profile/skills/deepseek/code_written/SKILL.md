# Synthetic Historical Control Band Forecast

## Purpose

Forecast every satellite band of a single land parcel at the **next time point**
after its observed history, using the Synthetic Historical Control (SHC) method of
Chen, Yang & Yang (*Synthetic Historical Control for Policy Evaluation*, SSRN 4995085).

The method is a time-series analogue of the synthetic control (SC) method: because a
single parcel has no cross-sectional untreated units, its **own distant history** is used
as the donor pool. A "treated block" is the most recent `m` observations; each "historical
block" is an older window that mirrors the treated block's pre/post split. The treated
pre-segment is matched with a convex (simplex) combination of historical pre-segments,
and the same weights are applied to the historical "pseudo-post" segments to build the
counterfactual. The primary use case: an agent sitting in a directory with `data.csv`
must write `answer.json` shaped as
`{"prediction": {"<band>": <number>, ...}, "rationale": "<text>"}` with **no network access**.

## When to use / When not to use

Use this skill when all of these hold:
- You have **one** unit (parcel / region / site) with its own band history, one value per band per time point.
- There are **no valid cross-sectional controls** (every unit is treated, or only one unit is available).
- The latent band pattern is seasonal / cyclic / recurring (not a pure monotone growth trend).
- The task asks for the single next time point (`n = 1`); forecasting several steps is possible but higher-bias.

Do **not** use this skill when:
- Multiple never-treated units exist → use a panel synthetic control or difference-in-differences skill instead.
- You only need a persistence / last-value baseline → that is a one-line fallback, not this method.
- The series is dominated by an unbounded growth trend with no recurring structure → detrend or difference first (the script does this with `--detrend auto|linear`), otherwise SHC matching is invalid.
- You need probabilistic intervals → this skill returns point forecasts only; route to an uncertainty-quantification skill for intervals.

## Environment Setup

**Primary (recommended, fully offline):** Python **≥ 3.9** (validated on 3.11) with the
standard library only. No third-party packages are required; `scripts/` import only
`argparse`, `csv`, `json`, `math`, `statistics`, `pathlib`, and `sys`.

Pre-flight check:

```bash
python3 -c "import csv, json, math, statistics; print('stdlib ok', __import__('sys').version.split()[0])"
python3 scripts/run_shc_band_forecast.py --help >/dev/null && echo "shc script ok"
```

Both commands must print `ok` lines and exit `0`.

**Optional scientific upgrade (needs install, still offline at run time):** if you want the
kernel smoothing and the simplex QP solved with compiled linear algebra, install NumPy/SciPy
and point `PYTHONPATH` at them. This is *not required* and the script does not import them.

```bash
# pickle-free, user-level install (choose one)
python3 -m pip install --user "numpy>=1.24" "scipy>=1.10"
# or, when the environment is read-only, install into a local prefix:
python3 -m pip install --target ./vendor "numpy>=1.24" "scipy>=1.10"
```

When a fully isolated environment is needed (for example a node without system Python):

```bash
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install --upgrade pip
# stdlib-only run needs nothing else; add numpy/scipy only for the optional upgrade
```

**External CLI tools:** none. The skill has no shell-tool dependency.

**Environment variables / config:** none required. All paths are passed as CLI arguments;
there are no hardcoded paths, so the skill runs from any working directory.

**Slurm exercise path (optional):** the deterministic exercise template is
`slurm/jobs/shc_band_forecast_cpu.sbatch` (repo-relative). It runs the toy example and the
validator and writes JSON to `slurm/reports/`. Submit it from the repository root with
`sbatch slurm/jobs/shc_band_forecast_cpu.sbatch`. This path is only for cluster validation;
the standalone runbook below does not need Slurm.

## Inputs

| Name | Type / format | Required | Example |
|---|---|---|---|
| `data.csv` | CSV, one parcel's band history | yes | see below |
| `--input` | path override for the CSV (default `data.csv`) | optional | `--input data.csv` |
| `--out` | path for `answer.json` (default `answer.json`) | optional | `--out answer.json` |
| `--summary-out` | path for diagnostics JSON | optional | `--summary-out summary.json` |
| `--bands` | comma-separated band subset | optional | `--bands ndvi,nir,red,swir` |
| `--pre-min` / `--pre-max` | bounds on pre-duration `m` | optional | `--pre-min 4 --pre-max 24` |
| `--donor-mode` | `all-in` (default), `stepwise`, `single` | optional | `--donor-mode all-in` |
| `--detrend` | `auto` (default), `linear`, `none` | optional | `--detrend auto` |
| `--lags` | lagged self-controls in the semi-parametric regression | optional | `--lags 1` |

**Accepted CSV layouts (auto-detected):**

1. **Long** (canonical): columns whose names contain `band`/`variable`/`channel`,
   `time`/`date`/`t`, and `value`/`reflectance`/`y`.
   ```
   band,time,value
   ndvi,0,0.551
   ndvi,1,0.560
   nir,0,0.318
   ```
2. **Wide, rows = time points** and remaining columns are bands:
   ```
   time,ndvi,nir,red,swir
   0,0.551,0.318,0.121,0.209
   1,0.560,0.322,0.124,0.212
   ```
3. **Wide, rows = bands** and columns are time points (first column named `band`):
   ```
   band,t0,t1,t2
   ndvi,0.551,0.560,0.571
   nir,0.318,0.322,0.330
   ```

Timestamps may be numeric or strings (`2015-01`, `2020Q1`); non-numeric labels are
ordered by first appearance. Missing cells (`NA`, `NaN`, `""`, `.`) are dropped.

## Outputs

| File | Format | Key fields | Size / shape |
|---|---|---|---|
| `answer.json` | JSON object | `prediction` (object: band → float), `rationale` (non-empty string) | one key per band; a few hundred bytes |
| `summary.json` (optional) | JSON diagnostics | `n_bands`, `bands`, `predictions`, `donor_mode`, `detrend`, `lags`, `band_diagnostics` (per-band `m`, `bandwidth`, `matching_mse`, `n_donors`, `top_donor_weights`, `status`) | 1–3 KB per band |
| stdout | one-line JSON | `{"status": "ok", "n_bands": N, "predictions": {...}}` | parseable by pipelines |

Ground-truth or reference values are **not** part of the task input; the skill predicts
the unseen next time point directly.

## Procedure

Work from the skill directory (the folder containing this `SKILL.md`). All commands below
are relative and portable; substitute your own paths as needed.

### Step 0 — Pre-flight

```bash
python3 scripts/run_shc_band_forecast.py --help
```

Expected: usage text listing every flag, exit `0`. If this fails, see Troubleshooting.

### Step 1 — Inspect the input contract

```bash
head -3 data.csv
python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("data.csv")))
print("rows:", len(rows), "columns:", list(rows[0].keys()))
PY
```

Decision checkpoint: confirm you can see a band column, a time column, and a numeric value
column (any of the aliases above). If the header is unrecognizable, either rename columns
or convert to the long layout in Step 1b.

### Step 1b (fallback) — Normalize a non-standard CSV

```bash
python3 - <<'PY'
import csv
# rename to the canonical long layout: band,time,value
with open("data.csv") as fin, open("data_long.csv", "w", newline="") as fout:
    r = csv.DictReader(fin)
    w = csv.writer(fout); w.writerow(["band", "time", "value"])
    for row in r:
        w.writerow([row["<bandcol>"], row["<timecol>"], row["<valuecol>"]])
print("wrote data_long.csv")
PY
```

### Step 2 — Run the forecast

```bash
python3 scripts/run_shc_band_forecast.py \
  --input data.csv \
  --out answer.json \
  --summary-out summary.json
```

Expected stdout (one line):

```json
{"status": "ok", "n_bands": 4, "predictions": {"ndvi": 0.664606, "nir": 0.402777, "red": 0.168775, "swir": 0.274985}}
```

Artifacts produced: `answer.json` (the deliverable) and `summary.json` (diagnostics).

Decision checkpoint: `status == "ok"` and `n_bands` equals the number of band keys in
`data.csv`. If a band is missing, re-run with `--bands <the missing band>` on its own and
read the stderr warning to see whether its history is too short.

### Step 3 — Validate the deliverable

```bash
python3 scripts/validate_answer.py --answer answer.json
```

Expected: `{"status": "valid", ...}`, exit `0`. This enforces the exact
`{"prediction": {...}, "rationale": "..."}` contract (top-level keys, numeric finite
values, non-empty rationale). Fix and re-run if it reports errors.

### Step 4 — Check the matching quality (scientific sanity, not just schema)

```bash
python3 - <<'PY'
import json
diag = json.load(open("summary.json"))
for band, d in diag["band_diagnostics"].items():
    print(band, "m=", d.get("m"), "mse=", round(d.get("matching_mse", -1), 5),
          "n_donors=", d.get("n_donors"), "status=", d.get("status"))
PY
```

Decision checkpoint (apply the SHC matching criterion of the paper, Section 2.4):
- `matching_mse ≤ 0.05` (standardized units) for most bands → the pre-segment is well
  reproduced; the forecast is credible. Proceed.
- `0.05 < matching_mse ≤ 0.2` → weak match. Try `--detrend linear` (or `--detrend none`
  if you already detrended) and `--donor-mode stepwise`, then re-check. Report the
  residual match quality in the `rationale`.
- `status != "ok"` or a `fallback` note → the series is too short or constant; use the
  persistence baseline explicitly and say so in the rationale (`--bands` on a healthy band
  is preferable if only some bands are degenerate).

### Step 5 — Optional: tune the latent smoother

```bash
python3 scripts/run_shc_band_forecast.py --input data.csv --out answer.json \
  --pre-min 6 --pre-max 36 --mse-tol 0.05 --donor-mode all-in --detrend auto
```

- `--pre-min`/`--pre-max` bound the candidate pre-duration `m`. The script picks the
  largest `m` whose matching MSE is within `--mse-tol`, balancing the paper's two
  theoretical criteria (larger `m` reduces smoothness bias; smaller `m` makes the matching
  condition easier to satisfy).
- `--donor-mode stepwise` gives a parsimonious donor set (easier to interpret in the
  rationale); `--donor-mode single` reduces to the single best historical control.
- `--lags 1` adds a lagged self-control to the semi-parametric regression if the band has
  serial correlation.

### Realistic worked example

`examples/toy_parcel_data.csv` is a 60-step, 4-band parcel with a 12-step seasonal cycle
plus a mild trend and noise; `examples/toy_parcel_truth.json` holds the held-out next step.

```bash
python3 scripts/run_shc_band_forecast.py \
  --input examples/toy_parcel_data.csv \
  --out scratch_shc_answer.json --summary-out scratch_shc_summary.json
python3 scripts/validate_answer.py \
  --answer scratch_shc_answer.json \
  --expected-bands ndvi,nir,red,swir \
  --truth examples/toy_parcel_truth.json
```

Observed result: `accuracy.mean_abs_error ≈ 0.009` across bands (per-band errors reported by the
validator). This is the scientifically meaningful end-to-end check — it exercises parsing,
detrending, kernel smoothing, simplex matching, forecasting, and contract validation.

### Deterministic CI / smoke path (separate from the real run)

The smoke path must not depend on network or on the specific values of a real dataset:

```bash
mkdir -p scratch_ci
python3 scripts/run_shc_band_forecast.py \
  --input examples/toy_parcel_data.csv --out scratch_ci/shc_ci_answer.json
python3 scripts/validate_answer.py --answer scratch_ci/shc_ci_answer.json
```

The repository also wires this into `make smoke-shc-band-forecast` and the smoke test
`tests/smoke/test_shc_band_forecast.py`.

## Troubleshooting / Error Handling

| # | Symptom | Cause | Remediation |
|---|---|---|---|
| 1 | `error: failed to read data.csv: [Errno 2] No such file or directory` (exit `2`) | Running from the wrong directory or wrong `--input`. | `ls` the working directory; pass the absolute or correct relative path with `--input`. The agent task places `data.csv` in the working directory, so `--input data.csv` normally suffices. |
| 2 | `error: ... has no data rows` / `error: no usable band series found` (exit `2`) | Empty file, header-only file, or band/value columns not recognized. | Confirm the header with `head -1 data.csv`; rename columns to `band,time,value` (Step 1b), or pass a long-format file. |
| 3 | A band is silently absent from `prediction`, stderr shows `band 'X' has fewer than 2 points; skipped`. | That band's history is too short or all-missing. | Provide more history for that band; if impossible, forecast it with a documented baseline and state the fallback in the rationale. Validate the final band set with `--expected-bands`. |
| 4 | `matching_mse` is large (> 0.2) for all bands, forecasts drift to the mean. | A monotone growth trend violates the recurrence assumption, or the history is too short. | Re-run with `--detrend linear` (or `auto`); if the series was already detrended, use `--detrend none`. If history is short, lower `--pre-min`. Increase `--pre-max` only if additional blocks exist. |
| 5 | `validate_answer.py` reports `prediction ... must be a JSON number` or `band set mismatch`. | A band value was written as a string, or the wrong band set was produced. | Re-run the forecast; if writing manually, emit JSON numbers (no quotes) and JSON `null` never. Use `--expected-bands` to assert the required keys explicitly. |
| 6 | `ModuleNotFoundError: No module named 'numpy'` (only if you enabled the optional upgrade). | Optional compiled dependency absent. | The core script never needs NumPy. Remove the optional import path or run the stdlib script directly; the offline fallback is the pure-Python solver, which is the default. |
| 7 | Run is slow on very long histories (minutes). | The simplex solver and bandwidth LOOCV scale with series length and `--pre-max`. | Cap the work: `--max-blocks 120` (default), a smaller `--pre-max`, or a coarser `--bandwidth-grid`. These change speed, not the interface. |

Scripts fail loudly: they catch `OSError`/`ValueError`/`json.JSONDecodeError`, print a
message to **stderr**, and exit with a non-zero code (`2` read/parse, `3` no prediction,
`4` write). No silent failures.

## Validation

A successful end-to-end run satisfies **all** of the following:

1. `python3 scripts/run_shc_band_forecast.py --input data.csv --out answer.json` exits `0`
   and prints `{"status": "ok", ...}`.
2. `answer.json` parses and has exactly two top-level keys: `prediction` (object of
   band → finite number) and `rationale` (non-empty string).
3. `python3 scripts/validate_answer.py --answer answer.json --expected-bands <bands>` exits `0`.
4. Every expected band appears in `prediction` (compare against `data.csv` band values).
5. On the bundled example, `validate_answer.py --truth examples/toy_parcel_truth.json`
   reports `accuracy.mean_abs_error < 0.05` (reference run ≈ `0.009`).
6. `python3 -m unittest discover -s skills/statistical-and-machine-learning-foundations-for-science/synthetic-historical-control-band-forecast/tests -p 'test_*.py'` passes.

Repository-level checks: `make smoke-shc-band-forecast`, and
`python3 -m unittest tests.smoke.test_shc_band_forecast`.

## Related Skills

- `dowhy-average-treatment-effect-starter` — identification, estimation, and refutation for cross-sectional causal effects; the conceptual sibling when *cross-sectional* controls exist.
- `pymc-bayesian-linear-regression-starter` — probabilistic regression when you need intervals rather than point forecasts.
- `arviz-posterior-diagnostics-starter` — posterior/uncertainty diagnostics for a Bayesian upgrade path.
- `scipy-statistical-testing-starter` — significance testing you can layer on top of the forecast residuals.
- `dlinear`/`physics-informed-forecasting-starter` (earth domain) — alternative time-series forecasters when recurrence-based matching is inappropriate.
