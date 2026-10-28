# Example Data

`toy_parcel_data.csv` is a 60-step, 4-band history for a single land parcel, laid out in the
canonical **long** format (`band,time,value`):

- 4 bands: `ndvi`, `nir`, `red`, `swir`
- a 12-step seasonal cycle plus a mild linear trend and small noise
- the last observed time point is `t = 58`; the held-out target is `t = 59`

`toy_parcel_truth.json` holds the held-out next-step values so the validator can report a
`mean_abs_error`:

```json
{"ndvi": 0.694, "nir": 0.404, "red": 0.174, "swir": 0.274}
```

Run the example from the skill directory:

```bash
python3 scripts/run_shc_band_forecast.py \
  --input examples/toy_parcel_data.csv \
  --out scratch_shc_answer.json --summary-out scratch_shc_summary.json
python3 scripts/validate_answer.py \
  --answer scratch_shc_answer.json \
  --expected-bands ndvi,nir,red,swir \
  --truth examples/toy_parcel_truth.json
```
