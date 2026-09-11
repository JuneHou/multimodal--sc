"""lb_features — (1) band means of the monthly composites for the 27 target cells and their 5 donors
(P01-P10), her extraction (float64, non-finite -> NaN, nanmean per band over the whole chip); (2) her notebook-26 stored results parsed from the notebook's outputs.
    python lb_features.py
"""
import json
import re
import time

import numpy as np
import pandas as pd
import tifffile

import lb_lib as lb
import panel_lib as pl
import panel_monthly as pm


def band_means(tif, sensor):
    """Her notebook-26 extraction: read as float64, non-finite -> NaN, nanmean per band over the whole chip
    (channel-first (C, 101, 101) composites; band order as in panel_lib.BANDS)."""
    a = tifffile.imread(tif).astype("float64")
    a[~np.isfinite(a)] = np.nan
    assert a.ndim == 3 and a.shape[0] == len(lb.BANDS[sensor]), (tif, a.shape)
    return {f"mean_{b}": (float(np.nanmean(a[i])) if np.isfinite(a[i]).any() else np.nan)
            for i, b in enumerate(lb.BANDS[sensor])}


def extract():
    roster = pm.load_roster()
    idx = pd.read_csv(pm.INDEX_CSV)
    key = {(r.site_id, r.sensor, int(r.seq)): r for r in idx.itertuples()}
    rows, t0 = [], time.time()
    for sensor in lb.SENSORS:
        targets = sorted(idx[(idx.sensor == sensor) & (idx.group == "treatment")].site_id.unique())
        for t in targets:
            donors = roster[(roster.matched_treatment_site_id == t) & (roster.group == "counterfactual")
                            & (roster.control_rank.between(1, lb.N_DONORS))].sort_values("control_rank")
            assert len(donors) == lb.N_DONORS, (t, len(donors))
            sites = [(t, "target", 0)] + [(d.site_id, "donor", int(d.control_rank)) for d in donors.itertuples()]
            for site, role, rank in sites:
                for q in range(1, lb.TARGET + 1):
                    r = key.get((site, sensor, q))
                    usable = bool(r is not None and r.usable)
                    row = {"sensor": sensor, "site_id": site, "role": role, "target_site": t, "control_rank": rank,
                           "seq": q, "period_id": f"P{q:02d}", "month": lb.CALENDAR[q], "usable": usable}
                    if usable:
                        row.update(band_means(r.tif, sensor))
                    rows.append(row)
        print(f"  {sensor}: {len(targets)} targets x {1 + lb.N_DONORS} sites x {lb.TARGET} months ({time.time() - t0:.0f}s)", flush=True)
    cols = ["sensor", "site_id", "role", "target_site", "control_rank", "seq", "period_id", "month", "usable"] + \
           [f"mean_{b}" for s in lb.SENSORS for b in lb.BANDS[s]]
    df = pd.DataFrame(rows).reindex(columns=cols)
    df.to_csv(lb.FEATURES_CSV, index=False)
    print(f"{len(df)} rows -> {lb.FEATURES_CSV.name}; unusable rows: {int((~df.usable).sum())}")
    return df


def her_nb26():
    nb = json.load(open(lb.NB26))
    t = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "text" in o:
                t.append("".join(o["text"]))
            elif "data" in o and "text/plain" in o["data"]:
                t.append("".join(o["data"]["text/plain"]))
    t = "\n".join(t)
    # per-site held-out P10 values: 'sentinel1: 12 sites' then '  OK treatment_0015: joint standardized P10 error=1.181648'
    sites, sensor = [], None
    for line in t[t.index("Held-out validation function defined."):].splitlines():
        m = re.match(r"^(sentinel[12]): \d+ sites", line)
        if m:
            sensor = m.group(1); continue
        m = re.match(r"^\s+OK (treatment_\d{4}): joint standardized P10 error=(-?\d+\.\d+)", line)
        if m and sensor:
            sites.append({"sensor": sensor, "site_id": m.group(1), "her_joint_validation_nmse": float(m.group(2))})
        if line.startswith("=== Held-out P10 validation by sensor"):
            break
    S = pd.DataFrame(sites).drop_duplicates(["sensor", "site_id"])
    S.to_csv(lb.HER26_SITES, index=False)
    # sensor table and sensor x band table
    tables = []
    i = t.index("=== Held-out P10 validation by sensor ===")
    for line in t[i:i + 3000].splitlines():
        m = re.match(r"^\d+\s+(sentinel[12])\s+(-?\d+\.\d+)\s*$", line)
        if m:
            tables.append({"kind": "sensor", "sensor": m.group(1), "band": "", "sq_error": np.nan, "standardized_sq_error": float(m.group(2))})
        m = re.match(r"^\d+\s+(sentinel[12])\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", line)
        if m:
            tables.append({"kind": "sensor_band", "sensor": m.group(1), "band": m.group(2), "sq_error": float(m.group(3)), "standardized_sq_error": float(m.group(4))})
        if line.startswith("Saved 01_feature_panel"):
            break
    T = pd.DataFrame(tables); T.to_csv(lb.HER26_TABLES, index=False)
    # printed feature-panel rows (head/tail of her 01_feature_panel): for the extraction gate
    feats = []
    for m in re.finditer(r"^\s*\d+\s+(treatment_\d{4})\s+(P\d{2})\s+(\d{4}-\d{2})\s+(sentinel[12])\s+(\S+)\s+(-?\d+\.\d+)", t, re.M):
        feats.append({"site_id": m.group(1), "period_id": m.group(2), "month": m.group(3), "sensor": m.group(4),
                      "band": m.group(5), "her_mean_value": float(m.group(6))})
    F = pd.DataFrame(feats).drop_duplicates(); F.to_csv(lb.HER26_FEATURE_ROWS, index=False)
    print(f"her nb26: {len(S)} per-site values, {len(T)} table rows, {len(F)} printed feature rows")
    return S, T, F


if __name__ == "__main__":
    extract()
    her_nb26()
