---
name: synthetic-historical-control-satellite-forecast
description: Forecast the next time point for each parcel-level satellite band from a local long-format `data.csv` using the Synthetic Historical Control method with lagged multiband controls, historical-block matching, and an `answer.json` output contract. Do not use it when you need spatial neighbors, multiple parcels, or cloud APIs.
---

## Purpose
Forecast every band's next value for a single land parcel from its own historical satellite band series, then write `answer.json` as `{"prediction": {"<band>": <number>, ...}, "rationale": "<text>"}`.

This skill operationalizes the Synthetic Historical Control (SHC) idea from Chen, Yang, and Yang for the one-step-ahead offline setting: residualize each target band on lagged multiband controls, smooth the latent path, match the latest pre-forecast block against historical blocks from the same parcel, and transfer the matched blocks' next-step residuals into the forecast.

## When to use / When not to use
Use it when:
- You have exactly one parcel's history in a local `data.csv`.
- The file is long format: one row per `band`, `time`, and `value`.
- You need an offline forecast with no network access.
- The series has enough history to form multiple historical donor blocks, typically at least 6-8 time points.

Do not use it when:
- You need cross-parcel pooling, regional neighbors, or spatial interpolation.
- You need cloud-hosted catalog search, raster ingestion, or image preprocessing first.
- The file is not a complete band-by-time panel.
- You need a full causal effect study around a known intervention date rather than a next-step forecast.

Route adjacent tasks to:
- `rasterio-windowed-raster-preprocessing-starter` for raster extraction and preprocessing.
- `xarray-climate-cube-starter` for multidimensional environmental cube construction.
- `dowhy-average-treatment-effect-starter` for graph-based causal effect estimation with explicit treatments and controls.

## Environment Setup
Minimum runtime:
- Python `>=3.10`

Core packages:
- None beyond the Python standard library for the bundled runner.

Optional convenience packages:
```bash
pip install numpy==2.1.3 pandas==2.2.3
```
These are optional only. The bundled script does not require them.

Environment variables and config:
- None required.

Pre-flight check:
```bash
python3 --version
python3 scripts/run_shc_satellite_forecast.py --help
python3 scripts/validate_answer.py --help
```

Offline fallback:
- If `pip install` is blocked or the machine has no network access, use the bundled pure-Python scripts exactly as shipped.
- If you also need to inspect the paper locally, use a local PDF utility rather than online search:
```bash
which pdftotext
pdftotext <paper.pdf> - | rg "smoothness|matching condition|stepwise-matching|differencing"
```

## Inputs
- `data.csv`
  - Type/format: CSV, long format.
  - Required: yes.
  - Required columns: `band`, `time`, `value`. The runner also accepts common aliases such as `band_name`, `timestamp`, and `measurement`.
  - Example row: `NDVI,2025-11,0.39`
  - Shape guidance: every band must appear exactly once at every time point.
- `--mode`
  - Type/format: one of `auto`, `level`, `difference`.
  - Required: no, default `auto`.
  - Example value: `auto`
- `--details-out`
  - Type/format: JSON path.
  - Required: no.
  - Example value: `diagnostics.json`

## Outputs
- `answer.json`
  - Format: JSON object.
  - Key fields:
    - `prediction`: object keyed by band name with numeric forecasts.
    - `rationale`: text summary of the SHC fit, donor windows, and fallback decisions.
  - Shape guidance: one numeric prediction per distinct band in `data.csv`.
- Optional diagnostics JSON from `--details-out`
  - Format: JSON object.
  - Key fields:
    - `band_count`
    - `time_count`
    - `per_band[]`
      - `method`
      - `transform`
      - `pre_length`
      - `donor_count`
      - `donor_end_positions`
      - `donor_weights`
      - `pre_rmse`
      - `normalized_pre_rmse`

## Procedure
### Real run
1. Confirm the input is a complete single-parcel panel.
   Command:
   ```bash
   python3 - <<'PY'
   import csv
   from collections import defaultdict
   rows = list(csv.DictReader(open("data.csv", encoding="utf-8")))
   bands = sorted({row["band"] for row in rows})
   times = []
   seen = set()
   for row in rows:
       if row["time"] not in seen:
           seen.add(row["time"])
           times.append(row["time"])
   counts = defaultdict(int)
   for row in rows:
       counts[row["band"]] += 1
   print({"rows": len(rows), "bands": bands, "time_count": len(times), "per_band_counts": dict(counts)})
   PY
   ```
   Artifact: no file, but the printed dictionary should show the same count for every band.
   Inspect/validate: if one band count differs from the others, stop and repair `data.csv`; the runner expects a rectangular band-by-time panel.
   Decision logic: if `time_count < 6`, do not trust SHC matching; either gather more history or use a simpler baseline outside this skill.

2. Run the SHC forecast and save both the contract output and the diagnostics.
   Command:
   ```bash
   python3 scripts/run_shc_satellite_forecast.py \
     --input data.csv \
     --out answer.json \
     --details-out diagnostics.json \
     --mode auto \
     --min-pre-length 4 \
     --max-pre-length 10 \
     --max-donors 4
   ```
   Artifacts:
   - `answer.json`
   - `diagnostics.json`
   Key fields produced:
   - `answer.json.prediction.<band>`
   - `answer.json.rationale`
   - `diagnostics.json.per_band[].normalized_pre_rmse`
   - `diagnostics.json.per_band[].donor_end_positions`
   Inspect/validate:
   ```bash
   python3 - <<'PY'
   import json
   answer = json.load(open("answer.json", encoding="utf-8"))
   diagnostics = json.load(open("diagnostics.json", encoding="utf-8"))
   print({"prediction_bands": sorted(answer["prediction"]), "diagnostic_bands": [row["band"] for row in diagnostics["per_band"]]})
   print(diagnostics["per_band"])
   PY
   ```
   Decision logic:
   - If `method` is `synthetic_historical_control` and `normalized_pre_rmse <= 0.35`, keep the forecast.
   - If the script switched to `ridge_ols_fallback`, read the rationale and consider forcing `--mode difference` on the rerun when raw-level matching was unstable.
   - If donor windows are concentrated in a single old regime that you know is scientifically irrelevant, shorten `--max-pre-length` or rerun after trimming the history to a more homogeneous season or sensor era.

3. Validate the answer contract before handing the result to downstream automation.
   Command:
   ```bash
   python3 scripts/validate_answer.py \
     --input data.csv \
     --answer answer.json \
     --out validation.json
   ```
   Artifact: `validation.json`
   Key fields:
   - `valid`
   - `missing_bands`
   - `extra_bands`
   - `non_numeric_bands`
   Inspect/validate:
   ```bash
   cat validation.json
   ```
   Decision logic:
   - If `valid` is `true`, the output matches the required contract.
   - If `missing_bands` is non-empty, rerun the forecast after fixing band labels or duplicate rows in `data.csv`.
   - If `non_numeric_bands` is non-empty, inspect `answer.json` for serialization or failure-handling issues and rerun.

4. Sanity-check the prediction against the latest observed step.
   Command:
   ```bash
   python3 - <<'PY'
   import csv, json
   rows = list(csv.DictReader(open("data.csv", encoding="utf-8")))
   latest = {}
   for row in rows:
       latest[row["band"]] = float(row["value"])
   answer = json.load(open("answer.json", encoding="utf-8"))["prediction"]
   print({band: {"last": latest[band], "predicted": answer[band], "delta": round(answer[band] - latest[band], 6)} for band in sorted(answer)})
   PY
   ```
   Artifact: no file, but the printed dictionary exposes jump sizes by band.
   Inspect/validate: compare the deltas to known seasonal or phenological behavior.
   Decision logic:
   - If the deltas align with the parcel's expected next-step cycle, keep the run.
   - If one band jumps much more than the others and also has weak pre-fit, rerun with `--mode difference` or cut obvious regime breaks from the history.

### Deterministic smoke path
5. Run the bundled toy example from the skill directory.
   Command:
   ```bash
   python3 scripts/run_shc_satellite_forecast.py \
     --input examples/toy_satellite_history.csv \
     --out scratch-answer.json \
     --details-out scratch-details.json
   python3 scripts/validate_answer.py \
     --input examples/toy_satellite_history.csv \
     --answer scratch-answer.json \
     --out scratch-validation.json
   ```
   Artifacts:
   - `scratch-answer.json`
   - `scratch-details.json`
   - `scratch-validation.json`
   Inspect/validate:
   ```bash
   python3 - <<'PY'
   import json
   print(json.load(open("scratch-answer.json", encoding="utf-8")))
   print(json.load(open("scratch-validation.json", encoding="utf-8")))
   PY
   ```
   Decision logic:
   - `scratch-validation.json.valid` should be `true`.
   - The toy example should forecast all three bands and provide non-empty rationale text.

### Non-trivial usage example
6. Use the toy file as a seasonal remote-sensing benchmark with drift.
   Command:
   ```bash
   python3 scripts/run_shc_satellite_forecast.py \
     --input examples/toy_satellite_history.csv \
     --out example-answer.json \
     --details-out example-details.json \
     --mode auto
   ```
   Scientifically meaningful interpretation:
   - `NDVI` and `NIR` should forecast the next green-up reset after three seasonal cycles.
   - `SWIR` should move in the opposite direction, consistent with a drier-to-wetter transition signal.
   Inspect/validate:
   ```bash
   python3 - <<'PY'
   import json
   details = json.load(open("example-details.json", encoding="utf-8"))
   for row in details["per_band"]:
       print(row["band"], row["prediction"], row["transform"], row["normalized_pre_rmse"])
   PY
   ```
   Decision logic:
   - Prefer a run where `transform` is `difference` or `level` based on the lower `normalized_pre_rmse`, not personal preference.
   - If all bands fall back to `ridge_ols_fallback`, the history is probably too short or too non-recurrent for SHC and should be treated as a baseline-only forecast.

## Troubleshooting / Error Handling
1. Symptom: `synthetic historical control forecast failed: Could not find a band column...`
   Remediation:
   - Rename the CSV columns to `band`, `time`, and `value`, or use supported aliases.
   - Re-run the panel check from Procedure step 1.
   Alternate path:
   - If you cannot change upstream field names, create a temporary normalized copy:
   ```bash
   python3 - <<'PY'
   import csv
   with open("data.csv", encoding="utf-8") as src, open("normalized.csv", "w", encoding="utf-8", newline="") as dst:
       rows = list(csv.DictReader(src))
       writer = csv.DictWriter(dst, fieldnames=["band", "time", "value"])
       writer.writeheader()
       for row in rows:
           writer.writerow({"band": row["band_name"], "time": row["timestamp"], "value": row["measurement"]})
   PY
   ```

2. Symptom: `Band 'X' is missing ... time points` or `Duplicate observation for band=..., time=...`
   Remediation:
   - Remove duplicate rows.
   - Fill or drop incomplete time points so every band has exactly one row at each time.
   Alternate path:
   - If only one or two time points are missing, repair the source table first; this skill intentionally refuses silent imputation.

3. Symptom: `History is too short...` or `Need at least two historical blocks...`
   Remediation:
   - Gather more history.
   - Lower `--max-pre-length` so the treated block can be matched against more historical donors.
   Alternate path:
   - If you must produce a quick baseline on short history, rerun with the same command and accept the `ridge_ols_fallback` path, but document that SHC matching was not feasible.

4. Symptom: forecasts exist but `normalized_pre_rmse` is consistently high, or the rationale says `ridge_ols_fallback`
   Remediation:
   - Rerun with `--mode difference` because the paper notes that monotone raw-level trends should be differenced or detrended before matching.
   - Shorten `--max-pre-length` if the regime changed recently.
   Alternate path:
   - Use only the OLS baseline as a diagnostic, not as a strong SHC claim, when no recurrent historical pattern exists.

5. Symptom: `validate_answer.py` reports `missing_bands`, `extra_bands`, or `non_numeric_bands`
   Remediation:
   - Open `answer.json` and confirm it is exactly one JSON object with the two top-level keys `prediction` and `rationale`.
   - Ensure each prediction value is numeric and every input band appears once.
   Alternate path:
   - Regenerate `answer.json` directly with the main script instead of hand-editing the file.

## Validation
- Contract check:
  ```bash
  python3 scripts/validate_answer.py --input data.csv --answer answer.json
  ```
- Expected end-to-end success criteria:
  - `answer.json` parses as JSON.
  - `prediction` has one key per band in the input.
  - Every prediction value is numeric.
  - `rationale` is non-empty and names the SHC fit behavior.
  - `diagnostics.json.per_band[].normalized_pre_rmse` is available for every band.
- Unit tests:
  ```bash
  python3 -m unittest discover -s tests -p 'test_*.py'
  ```

## Related Skills
- `rasterio-windowed-raster-preprocessing-starter`
- `xarray-climate-cube-starter`
- `dowhy-average-treatment-effect-starter`
