"""Show the panel in the working directory: the units, the periods and one variable by period and unit.

    python scripts/show_data.py            Y_observed
    python scripts/show_data.py Z1         another column (Z1, Z2 or Y_observed)
"""
import glob
import sys

import pandas as pd

files = glob.glob("*_panel.csv")
if len(files) != 1:
    raise SystemExit(f"expected one *_panel.csv file in the working directory, found {files}")
d = pd.read_csv(files[0])
var = sys.argv[1] if len(sys.argv) > 1 else "Y_observed"
if var not in ("Z1", "Z2", "Y_observed"):
    raise SystemExit("the variable must be Z1, Z2 or Y_observed")
units = ["treated"] + sorted(u for u in d.unit_id.unique() if u != "treated")
print(f"file {files[0]}: {len(d)} rows, columns {list(d.columns)}")
print(f"units: {len(units)} ({units[0]}, then {units[1]} to {units[-1]}); periods {d.period.min()} to {d.period.max()}")
print(f"treatment_group is 1 for {sorted(d.loc[d.treatment_group == 1, 'unit_id'].unique())} and 0 for the others")
print(f"\n{var} by period (rows) and unit (columns):")
wide = d.pivot(index="period", columns="unit_id", values=var)[units]
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(wide.round(3).to_string())
