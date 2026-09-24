"""lb_her_panel — build the collaborator's `feature_panel` (long form) from our lb_features_monthly.csv (wide form).

Her cells 6-7 read TIFFs into one row per site x period x sensor x band with a `mean_value`, then derive
period_num, month_date, calendar_month_num, x_sin_month, x_cos_month and sort. We do not have her TIFF
tree; our per-band means are already gated against her printed values (lb_tests gate 2). So the I/O half
of cells 6-7 is replaced by a melt of our CSV, and the derived-column half is HER statements, executed
verbatim from lb_her_nb10.CELL7_DERIVED_SRC in a namespace where `feature_panel` is bound.

    her_feature_panel(role="target", max_seq=None)   -> her long panel, all periods present in the CSV
    her_feature_panel(max_seq=9)                      -> the same with P10 absent (the arms' training panel)
    panel_from_wide(df)                               -> the same from any wide frame (sensor, site_id, period_id, month, mean_<band>)
"""
import numpy as np
import pandas as pd

import lb_lib as lb
import lb_her_nb10 as her

REQUIRED = ["site_id", "sensor", "band", "period_id", "period_num", "month", "calendar_month_num",
            "mean_value", "x_sin_month", "x_cos_month"]      # what prepare_band_series / get_observed_value read


def panel_from_wide(d):
    parts = []
    for sensor in her.BANDS:                       # her BANDS, her band order
        s = d[d.sensor == sensor]
        if s.empty:
            continue
        for band in her.BANDS[sensor]:
            parts.append(pd.DataFrame({"site_id": s.site_id.values, "period_id": s.period_id.values,
                                       "month": s.month.values, "sensor": sensor, "band": band,
                                       "mean_value": s[f"mean_{band}"].astype(float).values}))
    ns = {"feature_panel": pd.concat(parts, ignore_index=True), "np": np, "pd": pd}
    exec(her.CELL7_DERIVED_SRC, ns)                # her cell-7 statements, unchanged
    panel = ns["feature_panel"]
    missing = [c for c in REQUIRED if c not in panel.columns]
    assert not missing, f"panel lacks {missing}"
    return panel


def her_feature_panel(features_csv=lb.FEATURES_CSV, role="target", max_seq=None):
    d = pd.read_csv(features_csv)
    d = d[d.role == role]
    if max_seq is not None:
        d = d[d.seq <= max_seq]
    return panel_from_wide(d)


if __name__ == "__main__":
    p = her_feature_panel()
    print(p.shape, list(p.columns))
    print(p.head(12).to_string())
