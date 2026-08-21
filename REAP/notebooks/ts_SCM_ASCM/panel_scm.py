"""panel_scm — Experiment 1: SCM weight fitting + validation on a latent panel.

The collaborator's notebook-14 optimization, verbatim (see notebook 02 of this folder):
**find** w₁…w_J (one per donor), **subject to** wⱼ ≥ 0 and Σw = 1, **minimizing** the
mean squared difference between the treated site's z-scored latent and the weighted donor
mix over fit-window periods × all latent dimensions, jointly, per site × sensor
(SLSQP, bounds (0,1), init 1/J, ftol 1e-12, maxiter 5000).

Validation schemes: `frozen_joint` (notebook-14: w8 fit on P01–P08 scored on P09+P10
pooled) and `expanding` (advisor: w8 → P09; w9 fit on P01–P09 → P10). Metric is
holdout RMSPE in P01–P08 SD units (same formula as ADH RMSPE on z-scored outcomes).
Flags: notebook-14 ratio (≤1.5/≤2.0) AND whether holdout RMSPE beats the mean
predictor (`flag_level` pass if RMSPE < 1).
A period enters a fit/metric only if the treated site AND all J donors have latents.

`run_scm(panel, donors)` returns the weights / validation / effects DataFrames with the
exact schemas of `panel_scm_weights.csv`, `panel_scm_validation.csv`,
`panel_scm_effects.csv` — parity with the executed notebook 02 is asserted in
notebook 05 before any new experiment runs.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panel_lib import SENSORS, TRAIN8, TRAIN9, POST, J, flags


def scm_fit(y, X):
    """y (T*D,), X (T*D, J) z-scored, no NaN. Returns simplex weights (J,)."""
    Jn = X.shape[1]

    def obj(w):
        r = y - X @ w
        return float(np.mean(r * r))

    res = minimize(obj, np.full(Jn, 1.0 / Jn), method="SLSQP",
                   bounds=[(0.0, 1.0)] * Jn,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 5000, "ftol": 1e-12})
    assert res.success, res.message
    w = res.x
    assert abs(w.sum() - 1.0) < 1e-6
    return w


def fit_weights(panel, donors, arms=("main",)):
    """SCM fits for w8 (P01–P08) and w9 (P01–P09) per arm × sensor × treated site.
    Returns (weights DataFrame, W: {(arm, sensor, tid, window): (w, donors, seqs)})."""
    w_rows, W = [], {}
    for arm in arms:
        for sensor in SENSORS:
            for tid in panel.treatments:
                dl = donors[arm][(sensor, tid)]
                if dl is None:
                    continue
                for wname, window in (("P01-08", TRAIN8), ("P01-09", TRAIN9)):
                    seqs = panel.usable(tid, sensor, window, dl, arm)
                    if arm == "main":
                        assert len(seqs) >= 2, (arm, sensor, tid, wname, seqs)
                    if len(seqs) < 2:
                        W[(arm, sensor, tid, wname)] = None
                        continue
                    y, X = panel.design(tid, sensor, seqs, dl)
                    w = scm_fit(y, X)
                    W[(arm, sensor, tid, wname)] = (w, dl, seqs)
                    for c, wi in zip(dl, w):
                        w_rows.append({"arm": arm, "sensor": sensor,
                                       "treatment_site_id": tid,
                                       "fit_window": wname, "donor": c,
                                       "weight": float(wi),
                                       "largest_donor_weight": float(w.max()),
                                       "effective_donors":
                                           float(1.0 / np.sum(w ** 2)),
                                       "n_train_periods": len(seqs)})
    wdf = pd.DataFrame(w_rows)
    if "main" in arms:
        assert len(wdf.query("arm == 'main'")) == 2 * len(panel.treatments) * 2 * J
    return wdf, W


def validate(panel, W, arms=("main",)):
    """Both validation schemes for every fitted site × sensor. Returns DataFrame."""
    val_rows = []
    for arm in arms:
        for sensor in SENSORS:
            for tid in panel.treatments:
                f8 = W.get((arm, sensor, tid, "P01-08"))
                f9 = W.get((arm, sensor, tid, "P01-09"))
                if f8 is None or f9 is None:
                    continue
                w8, dl, seqs8 = f8
                w9, _, seqs9 = f9
                tr8 = panel.rmse_of(w8, tid, sensor, seqs8, dl)
                tr9 = panel.rmse_of(w9, tid, sensor, seqs9, dl)
                vs = panel.usable(tid, sensor, [9, 10], dl, arm)
                entries = [("frozen_joint", "pooled", w8, tr8, vs, len(seqs8)),
                           ("expanding", "P09", w8, tr8,
                            panel.usable(tid, sensor, [9], dl, arm), len(seqs8)),
                           ("expanding", "P10", w9, tr9,
                            panel.usable(tid, sensor, [10], dl, arm), len(seqs9))]
                for scheme, ep, w, tr, seqs, ntr in entries:
                    if not seqs:
                        val_rows.append({"arm": arm, "sensor": sensor,
                                         "treatment_site_id": tid, "scheme": scheme,
                                         "eval_period": ep, "train_rmse": tr,
                                         "valid_rmse": np.nan, "ratio": np.nan,
                                         "flag_ratio": "no_data",
                                         "flag_level": "no_data",
                                         "n_train_periods": ntr,
                                         "n_valid_periods": 0})
                        continue
                    v = panel.rmse_of(w, tid, sensor, seqs, dl)
                    ratio, fr, fl = flags(tr, v)
                    val_rows.append({"arm": arm, "sensor": sensor,
                                     "treatment_site_id": tid, "scheme": scheme,
                                     "eval_period": ep, "train_rmse": tr,
                                     "valid_rmse": v, "ratio": ratio,
                                     "flag_ratio": fr, "flag_level": fl,
                                     "n_train_periods": ntr,
                                     "n_valid_periods": len(seqs)})
    val = pd.DataFrame(val_rows)
    if "main" in arms:
        assert len(val.query("arm == 'main'")) == 2 * len(panel.treatments) * 3
    return val


def effects(panel, W, arm="main"):
    """P11–P20 latent-gap effects with the frozen w8 (computed AND inspectable).
    Returns (effects DataFrame, traj: {(sensor, tid): {seq: gap}} incl. pre-periods)."""
    eff_rows, traj = [], {}
    for sensor in SENSORS:
        for tid in panel.treatments:
            w8, dl, _ = W[(arm, sensor, tid, "P01-08")]
            series = {}
            for q in range(1, 21):
                have_t = panel.ok(tid, sensor, q, arm)
                have_d = all(panel.ok(c, sensor, q, arm) for c in dl)
                g = np.nan
                if have_t and have_d:
                    zt = panel.Z(tid, sensor, q)
                    zs = np.tensordot(w8, np.stack(
                        [panel.Z(c, sensor, q) for c in dl]), axes=1)
                    g = float(np.sqrt(np.mean((zt - zs) ** 2)))
                series[q] = g
                if q >= 11:
                    eff_rows.append({"sensor": sensor, "treatment_site_id": tid,
                                     "period": f"P{q}", "latent_gap_norm": g,
                                     "treated_latent_present": have_t,
                                     "donors_present": have_d})
            traj[(sensor, tid)] = series
    eff = pd.DataFrame(eff_rows)
    assert len(eff) == 2 * len(panel.treatments) * len(POST)
    return eff, traj


def run_scm(panel, donors, arms=("main",)):
    """Full Experiment-1 pass: fits → validation → effects (main arm).
    Returns dict(weights=..., validation=..., effects=..., W=..., traj=...)."""
    wdf, W = fit_weights(panel, donors, arms)
    val = validate(panel, W, arms)
    eff, traj = effects(panel, W, arm="main")
    return {"weights": wdf, "validation": val, "effects": eff, "W": W, "traj": traj}
