# Synthetic example

`data.csv` is a synthetic Sentinel-1 parcel built so that its block weights are known. The values of P01 to P06 were set by hand. In every band each of the last three months, P07 to P09, equals 0.7 times the matching month of block 2 plus 0.3 times the matching month of block 4. With three matching months and one forward month there are six blocks. Block 2 matches P02 to P04 and its forward month is P05. Block 4 matches P04 to P06 and its forward month is P07. In every month VV-VH equals VV minus VH.

A program recovers those weights when it builds blocks of three matching months and one forward month, matches the observed values of all bands jointly, keeps the weights non-negative with a sum of one, and adds blocks stepwise while the matching error falls by more than 1e-6. It then predicts P10 as 0.7 times P05 plus 0.3 times P07 in each band, which gives VV −10.022, VH −15.903 and VV-VH 5.881. `expected_output.json` holds these numbers. A run agrees with it when every predicted value is within 1e-6 of them.

Run a program on it from the skill directory.

```bash
python predict.py --data examples/data.csv
```

The design of the program for the parcel in `data.csv` of the working directory is chosen from that parcel's own data.
