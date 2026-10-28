"""sf_same_cells — every row of the monthly table beside her SHC and the P07 to P09 mean on the same cells.

    python sf_same_cells.py

Her SHC side is built only from the numbers her notebook printed
(`Satellite/data/scripts/scripts/validation/shc/10_shc_monthly_multiple_outcomes.ipynb`): the joint standardized P10
error of each site that her cell 26 prints to six decimals. The script first checks that those 27 values give every
summary she printed (cell 28 mean_validation_nmse 0.655934 and median_validation_nmse 0.415021 at m = 4, cell 30
mean_standardized_P10_sq_error 0.492461 and 0.786712), then takes, for each row, the mean and median of her values over
the row's processed cells (her cell 28 summaries) and the mean over the processed cells of each sensor (her cell 30
summary, which equals the mean of the cells' joint values because every cell of a sensor has the same bands). The row's
own numbers and the P07 to P09 mean's numbers on the same cells come from results/cells_monthly.csv (sf_table.py).
Writes into results/
    same_cells_monthly.csv          per row: processed cells, the row's summaries, her SHC's and the P07 to P09 mean's
                                    summaries over the same cells
    numbers_same_cells_monthly.csv  one number-registry row per number of the file above, written at generation
"""
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OUT = HERE / "results"
NB10 = "Satellite/data/scripts/scripts/validation/shc/10_shc_monthly_multiple_outcomes.ipynb"
SENSORS = [("sentinel1", "Sentinel-1"), ("sentinel2", "Sentinel-2")]
HER_PRINTED = {"mean": "0.655934", "median": "0.415021", "sentinel1": "0.492461", "sentinel2": "0.786712"}


def f6(x):
    return f"{x:.6f}"


def her_site_values():
    """{(sensor, site_id): value} from the text her cell 26 printed."""
    nb = json.loads((ROOT / NB10).read_text())
    text = "".join("".join(o.get("text", "")) for o in nb["cells"][26]["outputs"])
    out, sensor = {}, None
    for line in text.splitlines():
        m = re.match(r"(sentinel\d): \d+ sites", line.strip())
        if m:
            sensor = m.group(1)
            continue
        m = re.match(r"OK (treatment_\d+): joint standardized P10 error=([\d.]+)", line.strip())
        if m:
            out[(sensor, m.group(1))] = float(m.group(2))
    return out


def summaries(values):
    """values: {(sensor, site_id): joint}. Her cell 28 mean and median, her cell 30 mean per sensor."""
    s = pd.Series(values)
    out = {"mean": f6(s.mean()), "median": f6(s.median())}
    for k, _ in SENSORS:
        v = [x for (sen, _), x in values.items() if sen == k]
        out[k] = f6(sum(v) / len(v)) if v else "–"
    return out


def main():
    her = her_site_values()
    assert len(her) == 27 and summaries(her) == HER_PRINTED, summaries(her)
    C = pd.read_csv(OUT / "cells_monthly.csv")
    cells = C.drop_duplicates(["model", "skill", "sensor", "site_id"])
    p0709 = cells[cells.skill == "P07 to P09 mean"].set_index(["sensor", "site_id"])["joint_validation_nmse"]
    rows, reg, today = [], [], dt.date.today().isoformat()
    src = "Satellite/notebooks/skillfoundry_profile/results/cells_monthly.csv"
    for (model, skill), g in cells.groupby(["model", "skill"], sort=False):
        p = g[g.status == "processed"]
        keys = list(zip(p.sensor, p.site_id))
        row = {"model": model, "skill": skill, "cells": len(keys),
               **{f"cells_{k}": int((p.sensor == k).sum()) for k, _ in SENSORS}}
        if keys:
            own = summaries(dict(zip(keys, p.joint_validation_nmse)))
            hs = summaries({k: her[k] for k in keys})
            rule = summaries({k: p0709[k] for k in keys})
        else:
            own = hs = rule = {k: "–" for k in HER_PRINTED}
        for label, d in (("row", own), ("her", hs), ("p0709", rule)):
            row.update({f"{label}_{k}": v for k, v in d.items()})
        rows.append(row)
        what = f"{model}, {skill}, over its {len(keys)} processed monthly cells"
        key = f"same_m_{model}_{skill.replace(' ', '_')}"
        reg.append((f"{key}_cells", f"{len(keys)}, {row['cells_sentinel1']}, {row['cells_sentinel2']}",
                    f"processed cells, both sensors, Sentinel-1, Sentinel-2, {what}", src,
                    "sf_same_cells.py, cells with status processed", today))
        reg.append((f"{key}_her", ", ".join(hs[k] for k in HER_PRINTED),
                    f"her SHC over the same cells: mean and median joint value (her cell 28), mean per sensor, Sentinel-1 "
                    f"then Sentinel-2 (her cell 30), {what}", f"{NB10}; {src}",
                    "sf_same_cells.py, from the per-site values her cell 26 printed", today))
        reg.append((f"{key}_p0709", ", ".join(rule[k] for k in HER_PRINTED),
                    f"P07 to P09 mean over the same cells: mean and median joint value, mean per sensor, Sentinel-1 then "
                    f"Sentinel-2, {what}", src, "sf_same_cells.py, as her cells 28 and 30", today))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "same_cells_monthly.csv", index=False)
    pd.DataFrame(reg, columns=["name", "value", "what it counts", "source file", "how it is computed", "date read"]
                 ).to_csv(OUT / "numbers_same_cells_monthly.csv", index=False)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
