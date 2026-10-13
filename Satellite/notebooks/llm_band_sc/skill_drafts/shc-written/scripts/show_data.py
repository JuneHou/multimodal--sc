#!/usr/bin/env python3
"""Print one land parcel's nine monthly band values and name the month to predict.

Run from the skill directory:
    python scripts/show_data.py
    python scripts/show_data.py --data examples/data.csv
    python scripts/show_data.py --data data.csv --out data_summary.json

The data file has one row per band per month, columns band, period_id (P01 to P09), month (YYYY-MM) and
mean_value. A file that holds any other period, misses a month or a band, or has a non-numeric value is refused
with a message on stderr and exit status 1.
"""
import argparse
import json
import sys
from pathlib import Path

BANDS = {"sentinel1": ["VV", "VH", "VV-VH"], "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]}
SENSOR_NAME = {"sentinel1": "Sentinel-1 radar", "sentinel2": "Sentinel-2 optical"}
DECIMALS = {"sentinel1": 2, "sentinel2": 3}
PERIODS = [f"P{k:02d}" for k in range(1, 10)]
COLUMNS = ["band", "period_id", "month", "mean_value"]


class DataError(ValueError):
    pass


def next_month(month):
    y, m = map(int, month.split("-"))
    return f"{y + (m == 12):04d}-{m % 12 + 1:02d}"


def read_parcel(path):
    """Read and check one parcel's file. Returns (sensor, bands, rows, target) where rows is a list of
    (period_id, month, {band: value}) in time order and target is (period_id, month) of the month to predict."""
    import numpy as np
    import pandas as pd
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{path} not found")
    try:
        d = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"{path} could not be read as CSV: {e}")
    if d.empty:
        raise DataError(f"{path} has no rows")
    missing = [c for c in COLUMNS if c not in d.columns]
    if missing:
        raise DataError(f"{path} lacks column(s) {missing}; expected {COLUMNS}")
    periods = sorted(set(d["period_id"].astype(str)))
    if periods != PERIODS:
        extra = [p for p in periods if p not in PERIODS]
        raise DataError(f"{path} must hold exactly the nine periods P01-P09"
                        + (f"; it also holds {extra}" if extra else f"; it holds {periods}"))
    sensor = next((s for s, b in BANDS.items() if set(d["band"].astype(str)) == set(b)), None)
    if sensor is None:
        raise DataError(f"{path} bands {sorted(set(d['band'].astype(str)))} are not the bands of one sensor {BANDS}")
    if d.duplicated(["band", "period_id"]).any() or len(d) != 9 * len(BANDS[sensor]):
        raise DataError(f"{path} must hold one row per band per period ({9 * len(BANDS[sensor])} rows)")
    values = pd.to_numeric(d["mean_value"], errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise DataError(f"{path} has a missing or non-numeric mean_value")
    d = d.assign(mean_value=values)
    rows = []
    for p in PERIODS:
        g = d[d["period_id"] == p]
        months = sorted(set(g["month"].astype(str)))
        if len(months) != 1:
            raise DataError(f"{path} period {p} has month(s) {months}; expected one YYYY-MM month")
        rows.append((p, months[0], {b: float(g.loc[g["band"] == b, "mean_value"].iloc[0]) for b in BANDS[sensor]}))
    return sensor, BANDS[sensor], rows, ("P10", next_month(rows[-1][1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data.csv"), metavar="PATH",
                    help="path (string) of the parcel's data file; default: data.csv in the current directory")
    ap.add_argument("--out", type=Path, default=None, metavar="PATH",
                    help="path (string) of a JSON file to write the same data to; default: none written")
    a = ap.parse_args(argv)
    try:
        sensor, bands, rows, target = read_parcel(a.data)
    except DataError as e:
        sys.exit(f"show_data: {e}")
    d = DECIMALS[sensor]
    print(f"{SENSOR_NAME[sensor]}: mean band values of one land parcel, {len(rows)} consecutive calendar months. "
          f"Bands: {', '.join(bands)}.")
    for p, month, vals in rows:
        print(f"{p} = {month}: " + ", ".join(f"{b} {vals[b]:.{d}f}" for b in bands))
    print(f"The month to predict is {target[0]} = {target[1]}. Its values are withheld.")
    if a.out is not None:
        summary = {"data": str(a.data), "sensor": sensor, "bands": bands,
                   "months": [{"period_id": p, "month": m, "values": v} for p, m, v in rows],
                   "target": {"period_id": target[0], "month": target[1]}}
        try:
            a.out.write_text(json.dumps(summary, indent=2) + "\n")
        except OSError as e:
            sys.exit(f"show_data: could not write {a.out}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
