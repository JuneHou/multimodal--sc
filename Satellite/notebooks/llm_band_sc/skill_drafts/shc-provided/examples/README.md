# Synthetic example

`data.csv` is a synthetic Sentinel-1 parcel built so that its block weights are known. The values of P01 to P06 were set by hand. In every band each of the last three months, P07 to P09, equals 0.7 times the matching month of block 2 plus 0.3 times the matching month of block 4. With three matching months and one forward month there are six blocks. Block 2 matches P02 to P04 and its forward month is P05. Block 4 matches P04 to P06 and its forward month is P07. In every month VV-VH equals VV minus VH.

`spec.json` is the specification under which the fitter recovers those weights. It matches the observed values, where the planted relation holds exactly. The bandwidth and the trend degree are required fields that take no part in a fit on observed values.

`expected_output.json` is what the fitter returns for this file. It keeps blocks 2 and 4 with weights 0.7 and 0.3 and predicts P10 as 0.7 times P05 plus 0.3 times P07 in each band, which gives VV −10.022, VH −15.903 and VV-VH 5.881. A run agrees with it when every weight and every predicted value is within 1e-6 of these numbers.

Run it from the skill directory.

```bash
python scripts/fit.py --data examples/data.csv --spec examples/spec.json --log example_log.jsonl
```

The specification for the parcel in `data.csv` of the working directory is chosen from that parcel's own data.
