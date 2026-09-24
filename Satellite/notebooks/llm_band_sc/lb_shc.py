"""lb_shc — the collaborator's held-out SHC validation, run through HER code on our monthly band means.

Since 2026-09-23 this is a thin adapter: the arithmetic is lb_her_nb10 (her functions, extracted byte for
byte from data/shc/10_shc_monthly_multiple_outcomes.ipynb by lb_extract_her.py) driven by
lb_her_run.heldout_cell. This module only keeps the public surface that lb_score, lb_baselines and
lb_tests consume. The pre-2026-09-23 condensed re-implementation is frozen in lb_shc_condensed.py; the
drift report (python lb_her_run.py --drift) showed the two agree on all 27 cells: joint error and every
prediction differ by exactly 0, latent_mse by at most 7.6e-15 (pandas.Series.std vs np.std, last ulp),
selected blocks identical.

    run_all(features_csv)        -> DataFrame: sensor, site_id, joint_validation_nmse, selected_blocks, weights,
                                    latent_mse, pred_<band>   (same columns as before)
    heldout_shc(series, sensor)  -> {rows, joint, selected_blocks, weights, latent_mse, predictions}
                                    series = {seq: {mean_<band>: value}} for P01-P10 of one site

Gate 3b (lb_tests): her 27 stored per-site values to 1e-5 (they are printed to 6 decimals; observed 4.2e-07).
Gate G14 (lb_spec_tests): run_all is bit-identical to lb_shc_parity.csv, the reference frozen from her code.
"""
import numpy as np
import pandas as pd

import lb_lib as lb
import lb_her_nb10 as her
import lb_her_panel
import lb_her_run

# Her settings, exposed under the names this module used to define. Values are read from her extracted code.
M, N_VAL = her.M, 1
KERNEL_BANDWIDTH, ZETA, SD_TOL = her.KERNEL_BANDWIDTH, her.ZETA, her.SD_TOL
OPT_TOL, OPT_MAXITER, BETA_TREND_DEGREE = her.OPT_TOL, her.OPT_MAXITER, her.BETA_TREND_DEGREE
TRAIN = [int(p[1:]) for p in her.VALIDATION_TRAIN_PERIODS]     # P01-P09
TARGET = int(her.VALIDATION_PERIOD[1:])                        # 10


def _public(r):
    rows = pd.DataFrame({"band": [lb.disp(b) for b in r["results"]["band"]],
                         "actual_P10": r["results"]["actual_P10"].to_numpy(float),
                         "synthetic_P10": r["results"]["synthetic_P10"].to_numpy(float),
                         "standardized_sq_error": r["results"]["standardized_sq_error"].to_numpy(float)})
    return {"rows": rows, "joint": float(r["joint_validation_nmse"]), "selected_blocks": list(r["donor_ids"]),
            "weights": np.asarray(r["weights"], float), "latent_mse": float(r["latent_mse"]),
            "predictions": dict(zip(rows["band"], rows["synthetic_P10"]))}


def heldout_shc(series, sensor):
    wide = pd.DataFrame([{"sensor": sensor, "site_id": "site", "period_id": f"P{q:02d}", "month": lb.CALENDAR[q], **v}
                         for q, v in sorted(series.items())])
    return _public(lb_her_run.heldout_cell(lb_her_panel.panel_from_wide(wide), sensor, "site"))


def run_all(features_csv=lb.FEATURES_CSV):
    df, _ = lb_her_run.run_all(features_csv)
    cols = lb_her_run.PUBLIC_COLUMNS + [c for c in df.columns if c.startswith("pred_")]
    return df[cols].reset_index(drop=True)


if __name__ == "__main__":
    res = run_all()
    lb_her_run.compare_to_her_stored(res)
    df = pd.read_csv(lb.FEATURES_CSV)
    df = df[df.role == "target"]
    worst = 0.0
    for _, row in res.iterrows():                  # the series path must give the panel path's numbers
        h = heldout_shc(lb_her_run._series(df, row.site_id, row.sensor), row.sensor)
        worst = max(worst, abs(h["joint"] - row.joint_validation_nmse))
        assert h["selected_blocks"] == [int(x) for x in row.selected_blocks.split()], row.site_id
    print(f"heldout_shc (series path) vs run_all (panel path): max |d joint| = {worst:.1e} over {len(res)} cells")
