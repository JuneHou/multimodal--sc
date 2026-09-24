"""lb_guard — the withheld month. The model may know that P10 exists, that it is the target and which calendar
month it is (every shipped prompt says so). It never sees a P10 band value, and nothing may fit on P10.

    training_panel()                 the ONLY panel loader the arms may call: her long schema, P01-P09, no P10 row
    cell_panel(panel, sensor, site)  one cell's nine-period slice, for a spec fit or a sandbox run
    canary_panel()                   the full panel with every P10 value replaced by a sentinel, and the sentinels
    p10_value_strings(sensor, site)  the formatted P10 values as a prompt would print them (gate G9 asserts absence)
    assert_no_sentinel(blob, sentinels, where)
"""
import numpy as np
import pandas as pd

import lb_lib as lb
import lb_her_panel

N_TRAIN = 9
SENTINEL_BASE = -987654.321


def training_panel(features_csv=lb.FEATURES_CSV):
    p = lb_her_panel.her_feature_panel(features_csv, max_seq=N_TRAIN)
    assert int(p.period_num.max()) == N_TRAIN and "P10" not in set(p.period_id), "training panel leaked P10"
    n_expected = 12 * N_TRAIN * 3 + 15 * N_TRAIN * 8                          # S1: 12 sites x 3 bands; S2: 15 x 8
    assert len(p) == n_expected, (len(p), n_expected)
    return p


def cell_panel(panel, sensor, site):
    c = panel[(panel.sensor == sensor) & (panel.site_id == site)].reset_index(drop=True)
    assert len(c) == N_TRAIN * len(lb.BANDS[sensor]) and "P10" not in set(c.period_id), (sensor, site, len(c))
    return c


def canary_panel(features_csv=lb.FEATURES_CSV):
    """Every P10 band value replaced by a distinct sentinel; anything downstream that shows a sentinel read P10."""
    d = pd.read_csv(features_csv)
    d = d[d.role == "target"].copy()
    sentinels = {}
    for i, b in enumerate(lb.BANDS["sentinel1"] + lb.BANDS["sentinel2"]):
        col = f"mean_{b}"
        mask = (d.seq == lb.TARGET) & d[col].notna()
        sentinels[b] = SENTINEL_BASE + i
        d.loc[mask, col] = sentinels[b]
    return lb_her_panel.panel_from_wide(d), sentinels


def p10_value_strings(sensor, site, features_csv=lb.FEATURES_CSV):
    d = pd.read_csv(features_csv)
    row = d[(d.role == "target") & (d.sensor == sensor) & (d.site_id == site) & (d.seq == lb.TARGET)].iloc[0]
    dec = lb.DECIMALS[sensor]
    return [f"{float(row[f'mean_{b}']):.{dec}f}" for b in lb.BANDS[sensor]]


def assert_no_sentinel(blob, sentinels, where):
    text = blob if isinstance(blob, str) else repr(blob)
    for b, s in sentinels.items():
        for form in (f"{s:.3f}", f"{s:.2f}", f"{s:.1f}", repr(s), str(int(s))):
            assert form not in text, f"sentinel for {b} ({form}) found in {where}"
