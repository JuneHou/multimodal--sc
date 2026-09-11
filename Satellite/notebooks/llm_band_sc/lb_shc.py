"""lb_shc — the collaborator's held-out SHC validation (data/shc/10_shc_monthly_multiple_outcomes.ipynb),
run by us on our monthly band means so that the SHC row sits on the same footing as the model rows.
Copied step by step from her code (cells 4, 11, 13, 15, 17, 19, 25):
  allowed periods P01-P09; per band: beta_sin, beta_cos by OLS after partialling out [1, centred t]
  (BETA_TREND_DEGREE = 1); residualized y = y - X_season @ beta; l_hat = local-linear Gaussian kernel
  smoother, bandwidth 2.0, evaluated at the same t; l_hat standardized per band (mean, SD ddof 1);
  validation blocks from P01-P09 with m = 4, n = 1 (D1 pre P01-P04 -> P05 ... D5 pre P05-P08 -> P09),
  treated pre block P06-P09; bands stacked; stepwise donor selection (first donor = smallest trial MSE,
  then add while improvement > ZETA = 1e-4), SLSQP simplex weights (ftol 1e-10, maxiter 5000, tiny
  weights zeroed and renormalised); prediction of P10 = weights @ the selected blocks' observed raw value
  at their forward month; metric = (actual - synthetic)^2 / nanvar(P01-P09, ddof 1), mean over bands.
Gate (lb_tests): reproduces her 27 stored per-site values.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import lb_lib as lb

M, N_VAL = 4, 1
KERNEL_BANDWIDTH = 2.0
ZETA = 1e-4
SD_TOL = 1e-8
OPT_TOL, OPT_MAXITER = 1e-10, 5000
BETA_TREND_DEGREE = 1
TRAIN = list(range(1, 10))            # P01-P09
TARGET = 10


def gaussian_kernel(u):
    return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2)


def local_linear_predict(t_train, y_train, t_eval, bandwidth):
    t_train = np.asarray(t_train, float); y_train = np.asarray(y_train, float)
    t_eval = np.atleast_1d(np.asarray(t_eval, float))
    out = np.full(len(t_eval), np.nan)
    good = np.isfinite(t_train) & np.isfinite(y_train)
    t_tr, y_tr = t_train[good], y_train[good]
    if len(t_tr) < 2:
        return out
    for i, t0 in enumerate(t_eval):
        w = gaussian_kernel((t_tr - t0) / bandwidth)
        X_loc = np.column_stack([np.ones(len(t_tr)), t_tr - t0])
        XtW = X_loc.T * w
        out[i] = (np.linalg.pinv(XtW @ X_loc) @ (XtW @ y_tr))[0]
    return out


def _trend_basis(t, degree=1):
    t = np.asarray(t, float); tc = t - np.mean(t)
    return np.column_stack([np.ones(len(t))] + [tc ** d for d in range(1, degree + 1)])


def latent_std(y, t, months):
    """Her estimate_seasonal_beta_and_latent + per-band standardization. y: raw band means at t."""
    theta = 2.0 * np.pi * np.asarray(months, float) / 12.0
    X_season = np.column_stack([np.sin(theta), np.cos(theta)])
    T_basis = _trend_basis(t, BETA_TREND_DEGREE)
    M_T = np.eye(len(t)) - T_basis @ np.linalg.pinv(T_basis.T @ T_basis) @ T_basis.T
    beta = np.linalg.pinv((M_T @ X_season).T @ (M_T @ X_season)) @ ((M_T @ X_season).T @ (M_T @ y))
    resid = y - X_season @ beta
    l_hat = local_linear_predict(t, resid, t, KERNEL_BANDWIDTH)
    if not np.isfinite(l_hat).all():
        raise ValueError("latent estimate contains NaN")
    l_mean = float(np.mean(l_hat)); l_sd = float(np.std(l_hat, ddof=1))
    if not np.isfinite(l_sd) or l_sd < SD_TOL:
        l_sd = 1.0
    return (l_hat - l_mean) / l_sd, beta


def blocks(pre_periods, m, n):
    T0 = len(pre_periods); N = T0 - m - n + 1
    assert N >= 1
    return [{"donor_id": j + 1, "pre_periods": pre_periods[j:j + m], "post_periods": pre_periods[j + m:j + m + n]}
            for j in range(N)]


def solve_convex_weights(target, donor_matrix, cols):
    X = donor_matrix[:, cols]; J = X.shape[1]
    if J == 1:
        return np.array([1.0]), float(np.mean((target - X[:, 0]) ** 2))
    obj = lambda w: float(np.mean((target - X @ w) ** 2))
    res = minimize(obj, np.full(J, 1.0 / J), method="SLSQP", bounds=[(0.0, 1.0)] * J,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                   options={"ftol": OPT_TOL, "maxiter": OPT_MAXITER})
    if not res.success:
        raise RuntimeError(res.message)
    w = np.asarray(res.x, float); w[np.abs(w) < 1e-12] = 0.0; w = w / w.sum()
    return w, obj(w)


def stepwise(target, donor_matrix):
    n = donor_matrix.shape[1]; selected, remaining, current = [], list(range(n)), np.inf
    while remaining:
        trials = []
        for c in remaining:
            w, mse = solve_convex_weights(target, donor_matrix, selected + [c])
            trials.append((c, mse, np.inf if not np.isfinite(current) else current - mse))
        c, mse, imp = min(trials, key=lambda x: x[1])
        if len(selected) == 0 or imp > ZETA:
            selected.append(c); remaining.remove(c); current = mse
        else:
            break
    w, mse = solve_convex_weights(target, donor_matrix, selected)
    return selected, w, mse


def heldout_shc(series, sensor):
    """series: {seq: {mean_<band>: value}} for P01-P10. Returns per-band rows and the joint value."""
    bands = lb.BANDS[sensor]
    t = np.array(TRAIN, float); months = [int(lb.CALENDAR[q][5:7]) for q in TRAIN]
    vb = blocks([f"P{q:02d}" for q in TRAIN], M, N_VAL)
    treated_pre = [f"P{q:02d}" for q in TRAIN[-M:]]
    z_parts, Z_parts, y_by_band = [], [], {}
    for b in bands:
        y = np.array([series[q][f"mean_{b}"] for q in TRAIN], float)
        if not np.isfinite(y).all():
            raise ValueError(f"{b}: missing pre-period value")
        z, _ = latent_std(y, t, months)
        zmap = {f"P{q:02d}": z[i] for i, q in enumerate(TRAIN)}
        z_parts.append(np.array([zmap[p] for p in treated_pre]))
        Z_parts.append(np.column_stack([[zmap[p] for p in blk["pre_periods"]] for blk in vb]))
        y_by_band[b] = y
    target = np.concatenate(z_parts); donor_matrix = np.vstack(Z_parts)
    selected, w, mse = stepwise(target, donor_matrix)
    rows = []
    for b in bands:
        actual = float(series[TARGET][f"mean_{b}"])
        donor_values = np.array([series[int(vb[c]["post_periods"][0][1:])][f"mean_{b}"] for c in selected], float)
        synthetic = float(donor_values @ w)
        train_var = float(np.nanvar(y_by_band[b], ddof=1))
        sq = (actual - synthetic) ** 2
        rows.append({"band": lb.disp(b), "actual_P10": actual, "synthetic_P10": synthetic,
                     "standardized_sq_error": sq / train_var if train_var >= SD_TOL else np.nan})
    df = pd.DataFrame(rows)
    return {"rows": df, "joint": float(df.standardized_sq_error.mean()), "selected_blocks": [vb[c]["donor_id"] for c in selected],
            "weights": w, "latent_mse": mse, "predictions": {r["band"]: r["synthetic_P10"] for r in rows}}


def run_all(features_csv=lb.FEATURES_CSV):
    F = pd.read_csv(features_csv)
    out = []
    for sensor in lb.SENSORS:
        for site in sorted(F[(F.sensor == sensor) & (F.role == "target")].site_id.unique()):
            d = F[(F.site_id == site) & (F.sensor == sensor)]
            series = {int(r.seq): {c: getattr(r, c) for c in F.columns if c.startswith("mean_")} for r in d.itertuples()}
            r = heldout_shc(series, sensor)
            out.append({"sensor": sensor, "site_id": site, "joint_validation_nmse": r["joint"],
                        "selected_blocks": " ".join(map(str, r["selected_blocks"])), "weights": " ".join(f"{v:.4f}" for v in r["weights"]),
                        "latent_mse": r["latent_mse"], **{f"pred_{b}": v for b, v in r["predictions"].items()}})
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = run_all()
    her = pd.read_csv(lb.HER26_SITES)
    m = df.merge(her, on=["sensor", "site_id"])
    m["diff"] = (m.joint_validation_nmse - m.her_joint_validation_nmse).abs()
    pd.set_option("display.width", 200)
    print(m[["sensor", "site_id", "selected_blocks", "weights", "joint_validation_nmse", "her_joint_validation_nmse", "diff"]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("max |diff| vs her stored values:", f"{m['diff'].max():.2e}")
