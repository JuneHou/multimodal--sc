"""panel_ascm — Experiment 2: ridge-augmented SCM (ASCM) on a latent panel.

The collaborator's notebook-15 method, verbatim (see notebook 03 of this folder):
ridge-augmented SCM (Ben-Michael, Feller & Rothstein 2021, Eq. 18), closed form,

  min_w (1/2λ)‖x − Xw‖² + (1/2)‖w − w_SCM‖²  s.t. Σw = 1,

anchored on the SCM weights from `panel_scm.run_scm`. Negative weights allowed, no
intercept. λ per site × sensor × fit-window by leave-one-fit-period-out CV over the fit
window (grid logspace(−4,4,81); the SCM anchor refit once per fold, not per λ).

Honesty diagnostics carried into the outputs: λ at a grid boundary; weight distance from
SCM; negative-weight counts; and the latent-range check (fraction of the raw weighted
latent outside the tokenizer's trained [0,1]).

`run_ascm(panel, donors, scm)` returns DataFrames with the exact schemas of
`panel_ascm_weights.csv`, `panel_ascm_lambda_cv.csv`, `panel_ascm_validation.csv`,
`panel_scm_vs_ascm.csv`, `panel_ascm_effects.csv` — parity with the executed notebook 03
is asserted in notebook 05 before any new experiment runs.
"""
import numpy as np
import pandas as pd

from panel_lib import SENSORS, TRAIN8, TRAIN9, POST, J, LAMBDAS, flags
from panel_scm import scm_fit


def ascm_fit(y, X, w_scm, lam):
    """closed-form KKT solve of (1/2lam)||y-Xw||^2 + (1/2)||w-w_scm||^2, sum w = 1."""
    Jn = X.shape[1]
    H = np.eye(Jn) + X.T @ X / lam
    b = w_scm + X.T @ y / lam
    A = np.zeros((Jn + 1, Jn + 1))
    A[:Jn, :Jn] = H
    A[:Jn, Jn] = 1.0
    A[Jn, :Jn] = 1.0
    rhs = np.concatenate([b, [1.0]])
    w = np.linalg.solve(A, rhs)[:Jn]
    assert abs(w.sum() - 1.0) < 1e-9
    return w


def latent_range(panel, w, sensor, dl, seq):
    """weighted RAW latent (decode space): min, max, frac outside [0,1]."""
    v = np.tensordot(w, np.stack([panel.L(c, sensor, seq) for c in dl]), axes=1)
    return float(v.min()), float(v.max()), float(((v < 0) | (v > 1)).mean())


def fit_weights(panel, donors, scm_W, arms=("main",)):
    """λ CV + final closed-form ASCM fits anchored on the SCM weights.
    Returns (weights df, λ-CV df, WA: {(arm, sensor, tid, win): (w, dl, seqs, λ)})."""
    cv_rows, w_rows, WA = [], [], {}
    for arm in arms:
        for sensor in SENSORS:
            for tid in panel.treatments:
                dl = donors[arm].get((sensor, tid))
                if dl is None:
                    continue    # not estimable under the quality-40 filter
                for wname, window in (("P01-08", TRAIN8), ("P01-09", TRAIN9)):
                    seqs = panel.usable(tid, sensor, window, dl, arm)
                    if len(seqs) < 2:
                        continue
                    cv_sse = np.zeros(len(LAMBDAS))
                    for hold in seqs:
                        rest = [q for q in seqs if q != hold]
                        if len(rest) < 2:
                            continue
                        y_r, X_r = panel.design(tid, sensor, rest, dl)
                        w_scm_f = scm_fit(y_r, X_r)             # once per fold
                        y_h, X_h = panel.design(tid, sensor, [hold], dl)
                        for li, lam in enumerate(LAMBDAS):
                            w = ascm_fit(y_r, X_r, w_scm_f, lam)
                            r = y_h - X_h @ w
                            cv_sse[li] += float(r @ r)
                    best = int(np.argmin(cv_sse))
                    lam = float(LAMBDAS[best])
                    for li, lam_i in enumerate(LAMBDAS):
                        cv_rows.append({"arm": arm, "sensor": sensor,
                                        "treatment_site_id": tid,
                                        "fit_window": wname,
                                        "lambda": float(lam_i),
                                        "cv_sse": float(cv_sse[li]),
                                        "chosen": li == best})
                    # final fit on the full window, anchored on the SCM weights
                    y, X = panel.design(tid, sensor, seqs, dl)
                    fit8 = scm_W[(arm, sensor, tid, wname)]
                    w0 = fit8[0]
                    w = ascm_fit(y, X, w0, lam)
                    mse_a = float(np.mean((y - X @ w) ** 2))
                    mse_s = float(np.mean((y - X @ w0) ** 2))
                    assert mse_a <= mse_s + 1e-12, (arm, sensor, tid, wname)
                    WA[(arm, sensor, tid, wname)] = (w, dl, seqs, lam)
                    neg = w[w < 0]
                    for c, wi in zip(dl, w):
                        w_rows.append({"arm": arm, "sensor": sensor,
                                       "treatment_site_id": tid,
                                       "fit_window": wname, "donor": c,
                                       "weight": float(wi), "lambda": lam,
                                       "lambda_at_boundary":
                                           best in (0, len(LAMBDAS) - 1),
                                       "weight_distance_from_scm":
                                           float(np.linalg.norm(w - w0)),
                                       "n_negative_weights": int((w < 0).sum()),
                                       "sum_negative_weight": float(-neg.sum()),
                                       "n_train_periods": len(seqs)})
    wa = pd.DataFrame(w_rows)
    if "main" in arms:
        assert len(wa.query("arm == 'main'")) == 2 * len(panel.treatments) * 2 * J
    return wa, pd.DataFrame(cv_rows), WA


def validate(panel, WA, arms=("main",)):
    """Both schemes + the latent-range diagnostic. Returns DataFrame."""
    val_rows = []
    for arm in arms:
        for sensor in SENSORS:
            for tid in panel.treatments:
                f8 = WA.get((arm, sensor, tid, "P01-08"))
                f9 = WA.get((arm, sensor, tid, "P01-09"))
                if f8 is None or f9 is None:
                    continue
                w8, dl, seqs8, lam8 = f8
                w9, _, seqs9, lam9 = f9
                tr8 = panel.rmse_of(w8, tid, sensor, seqs8, dl)
                tr9 = panel.rmse_of(w9, tid, sensor, seqs9, dl)
                entries = [("frozen_joint", "pooled", w8, tr8,
                            panel.usable(tid, sensor, [9, 10], dl, arm), len(seqs8)),
                           ("expanding", "P09", w8, tr8,
                            panel.usable(tid, sensor, [9], dl, arm), len(seqs8)),
                           ("expanding", "P10", w9, tr9,
                            panel.usable(tid, sensor, [10], dl, arm), len(seqs9))]
                for scheme, ep, w, tr, seqs, ntr in entries:
                    row = {"arm": arm, "sensor": sensor,
                           "treatment_site_id": tid, "scheme": scheme,
                           "eval_period": ep, "train_rmse": tr,
                           "n_train_periods": ntr, "n_valid_periods": len(seqs)}
                    if not seqs:
                        row.update({"valid_rmse": np.nan, "ratio": np.nan,
                                    "flag_ratio": "no_data",
                                    "flag_level": "no_data",
                                    "latent_frac_outside_01": np.nan})
                    else:
                        v = panel.rmse_of(w, tid, sensor, seqs, dl)
                        ratio, fr, fl = flags(tr, v)
                        fo = float(np.mean([latent_range(panel, w, sensor, dl, q)[2]
                                            for q in seqs]))
                        row.update({"valid_rmse": v, "ratio": ratio,
                                    "flag_ratio": fr, "flag_level": fl,
                                    "latent_frac_outside_01": fo})
                    val_rows.append(row)
    vala = pd.DataFrame(val_rows)
    if "main" in arms:
        assert len(vala.query("arm == 'main'")) == 2 * len(panel.treatments) * 3
    return vala


def compare_with_scm(scm_val, ascm_val):
    """Side-by-side SCM vs ASCM validation → the panel_scm_vs_ascm.csv schema."""
    cmp_ = scm_val.merge(
        ascm_val, on=["arm", "sensor", "treatment_site_id", "scheme", "eval_period"],
        suffixes=("_scm", "_ascm"))
    cmp_["improvement"] = cmp_["valid_rmse_scm"] - cmp_["valid_rmse_ascm"]
    cmp_["ascm_better"] = cmp_["improvement"] > 0
    keep = ["arm", "sensor", "treatment_site_id", "scheme", "eval_period",
            "valid_rmse_scm", "valid_rmse_ascm", "improvement", "ascm_better",
            "latent_frac_outside_01"]
    return cmp_, cmp_[keep]


def effects(panel, WA, arm="main"):
    """P11–P20 latent-gap effects with the frozen ASCM w8 + latent-range diagnostic."""
    eff_rows = []
    for sensor in SENSORS:
        for tid in panel.treatments:
            w8, dl, _, _ = WA[(arm, sensor, tid, "P01-08")]
            for q in POST:
                have_t = panel.ok(tid, sensor, q, arm)
                have_d = all(panel.ok(c, sensor, q, arm) for c in dl)
                g = fo = np.nan
                if have_t and have_d:
                    zt = panel.Z(tid, sensor, q)
                    zs = np.tensordot(w8, np.stack(
                        [panel.Z(c, sensor, q) for c in dl]), axes=1)
                    g = float(np.sqrt(np.mean((zt - zs) ** 2)))
                    fo = latent_range(panel, w8, sensor, dl, q)[2]
                eff_rows.append({"sensor": sensor, "treatment_site_id": tid,
                                 "period": f"P{q}", "latent_gap_norm": g,
                                 "latent_frac_outside_01": fo,
                                 "treated_latent_present": have_t,
                                 "donors_present": have_d})
    effa = pd.DataFrame(eff_rows)
    assert len(effa) == 2 * len(panel.treatments) * len(POST)
    return effa


def run_ascm(panel, donors, scm, arms=("main",)):
    """Full Experiment-2 pass: λ CV + fits → validation → SCM comparison → effects.
    `scm` = the dict returned by panel_scm.run_scm (anchor weights + validation).
    Returns dict(weights=..., lambda_cv=..., validation=..., vs_scm=...,
    vs_scm_full=..., effects=..., WA=...)."""
    wa, cv, WA = fit_weights(panel, donors, scm["W"], arms)
    vala = validate(panel, WA, arms)
    cmp_full, cmp_keep = compare_with_scm(scm["validation"], vala)
    effa = effects(panel, WA, arm="main")
    return {"weights": wa, "lambda_cv": cv, "validation": vala,
            "vs_scm": cmp_keep, "vs_scm_full": cmp_full, "effects": effa, "WA": WA}
