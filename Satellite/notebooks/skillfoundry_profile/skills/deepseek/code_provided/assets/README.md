# Assets

Reference outputs from the deterministic example run. They are committed so reviewers can
diff a fresh run against a known-good artifact without any network access.

- `example_answer.json`: the `answer.json` produced from `examples/toy_parcel_data.csv`.
- `example_summary.json`: the diagnostics summary (`--summary-out`) from the same run.
- `example_truth.json`: the held-out next-step values for the same example, used only by the
  validator to compute `mean_abs_error`.

Regenerate all three with:

```bash
python3 scripts/run_shc_band_forecast.py \
  --input examples/toy_parcel_data.csv \
  --out assets/example_answer.json --summary-out assets/example_summary.json
```

Reference `mean_abs_error` on this example is ~`0.009`.
