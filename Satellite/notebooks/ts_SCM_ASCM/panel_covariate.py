"""panel_covariate — a learned covariate mapping g inside the latent SHC (plan 2026-09-10).

Model: y_{i,t} = g_t(x_i) + l_{i,t} + delta_{i,t} d_{i,t} + eps_{i,t}, y = the 980-d latent of site i at period t,
x_i = (land-cover class, elevation, slope) from the matching table. g replaces the paper's covariate term
x'beta only (slide 6, steps 1-3); the kernel stage, blocks, stepwise selection, weights, counterfactual and
effect of panel_shc.py are untouched and run on the residual u = y - g.

Fit (per sensor, per fill, per period t): multi-output ridge y_{i,t} = a_t + B_t x_i + e over the UNTREATED
rows at t -- every usable site for t <= T0, control sites only for t >= P11 -- with the intercept unpenalised.
x = one-hot NLCD class (reference class 41 = deciduous forest, the majority) + standardized elevation and
slope (standardized over the 286 sites once). The residual used for site i is the leave-one-site-out one,
u_{i,t} = y_{i,t} - g_t^{(-i)}(x_i), so no site's own latent enters its own covariate stage at any period.
The ridge penalty is chosen once per sensor x fill by the mean leave-one-site-out squared error over
P01-P10 on a log grid and reported.

Outputs: ResidualPanel (drop-in for pm.MonthlyPanel: series() returns u; observed() returns y; g() the fitted
covariate part), and panel_covariate.csv (description of g: per period the cross-site R^2 per channel, the
coefficients per feature and channel at P09-P12, the chosen penalty).
"""
import numpy as np
import pandas as pd

import panel_monthly as pm

REF_CLASS = 41
LAMBDA_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
LEVERAGE_TOL = 1e-6                   # a row with hat value 1 is the only usable member of its class at that period
NCH, NPARCEL = 5, 196
DESCRIBE_PERIODS = (9, 10, 11, 12)


# ----------------------------------------------------------------- descriptors
def descriptors(match_csv=pm.MATCH_CSV):
    """One row per site: nlcd_class, elevation_m, slope_deg (treated from the treatment_* columns, controls
    from the counterfactual_* columns; every match is 'strict', so the pair's class is the control's class)."""
    m = pd.read_csv(match_csv)
    t = m.drop_duplicates("treatment_site_id")[["treatment_site_id", "nlcd_class", "treatment_elevation_m", "treatment_slope_deg"]]
    t.columns = ["site_id", "nlcd_class", "elevation_m", "slope_deg"]
    c = m[["counterfactual_site_id", "nlcd_class", "counterfactual_elevation_m", "counterfactual_slope_deg"]]
    c.columns = t.columns
    d = pd.concat([t, c], ignore_index=True)
    assert d.site_id.is_unique
    return d.set_index("site_id")


def design_matrix(desc):
    """(sites x features) float64: one-hot classes except REF_CLASS, then z-scored elevation and slope.
    Returns (X, feature names, site order)."""
    classes = sorted(c for c in desc.nlcd_class.unique() if c != REF_CLASS)
    cols, names = [], []
    for c in classes:
        cols.append((desc.nlcd_class == c).astype(float).values); names.append(f"class_{c}")
    for v in ("elevation_m", "slope_deg"):
        x = desc[v].values.astype(float)
        cols.append((x - x.mean()) / x.std(ddof=1)); names.append(v)
    return np.column_stack(cols), names, list(desc.index)


# ----------------------------------------------------------------- ridge with unpenalised intercept
def ridge_fit(X, Y, lam):
    """Y (n x D) on [1, X] (n x (1+p)); penalty lam on the p slopes only. Returns (coef (1+p) x D, hat diag (n,))."""
    n = X.shape[0]
    A = np.column_stack([np.ones(n), X])
    D = np.eye(A.shape[1]); D[0, 0] = 0.0
    G = A.T @ A + lam * D
    Ginv = np.linalg.pinv(G)
    coef = Ginv @ (A.T @ Y)
    hdiag = np.einsum("ij,jk,ik->i", A, Ginv, A)
    return coef, hdiag


def loo_residuals(X, Y, lam):
    """Leave-one-site-out residuals of the ridge fit (exact for penalised least squares with fixed lam):
    e_i^(-i) = e_i / (1 - h_ii). A row with h_ii = 1 (the only usable site of its class at that period) has no
    closed form; it is refitted without the row, where its class column is all zero, so its prediction uses the
    intercept, elevation and slope only. Returns (residuals n x D, in-sample fit n x D, coef, n_refit)."""
    coef, h = ridge_fit(X, Y, lam)
    A = np.column_stack([np.ones(X.shape[0]), X])
    fit = A @ coef
    e = Y - fit
    out = np.empty_like(e); ok = h < 1.0 - LEVERAGE_TOL
    out[ok] = e[ok] / (1.0 - h[ok])[:, None]
    for i in np.where(~ok)[0]:
        c_i, _ = ridge_fit(np.delete(X, i, 0), np.delete(Y, i, 0), lam)
        out[i] = Y[i] - A[i] @ c_i
    return out, fit, coef, int((~ok).sum())


# ----------------------------------------------------------------- the mapping
def fit_g(panel, sensor, lam=None, grid=LAMBDA_GRID, desc=None):
    """Fit g_t on the untreated rows of every period; return dict with
      lam, g: {(site, seq): (D,) leave-one-site-out prediction g_t^(-i)(x_i)} for EVERY usable site-period
      (treated post rows get the prediction of the fit they were excluded from by construction),
      rows: {seq: [sites in the fit]}, coef: {seq: (1+p) x D}, r2: {seq: (NCH,)}, names."""
    desc = descriptors() if desc is None else desc
    X_all, names, order = design_matrix(desc)
    pos = {s: k for k, s in enumerate(order)}
    sites = panel.sites(sensor)
    untreated = lambda s, q: (q <= pm.T0) or (panel.group_of[s] == "counterfactual")
    per_period = {}
    for q in pm.PERIODS:
        fit_sites = [s for s in sites if untreated(s, q) and panel.L(s, sensor, q) is not None]
        Y = np.stack([panel.L(s, sensor, q) for s in fit_sites])
        X = X_all[[pos[s] for s in fit_sites]]
        per_period[q] = (fit_sites, X, Y)
    # penalty by mean leave-one-site-out squared error over P01-P10
    if lam is None:
        score = {}
        for l in grid:
            tot, cnt = 0.0, 0
            for q in pm.PRE:
                fit_sites, X, Y = per_period[q]
                e, _, _, _ = loo_residuals(X, Y, l)
                tot += float(np.sum(e ** 2)); cnt += e.size
            score[l] = tot / cnt if np.isfinite(tot) else np.inf
        lam = min(score, key=score.get)
    else:
        score = None
    g, coef, r2, rows, refit = {}, {}, {}, {}, {}
    for q in pm.PERIODS:
        fit_sites, X, Y = per_period[q]
        e, fit, cf, n_refit = loo_residuals(X, Y, lam)
        rows[q] = list(fit_sites); coef[q] = cf; refit[q] = n_refit
        for k, s in enumerate(fit_sites):
            g[(s, q)] = Y[k] - e[k]                      # leave-one-site-out prediction for a site in the fit
        # sites NOT in the fit (treated at q >= P11): the full fit's prediction (they were never in it)
        for s in sites:
            if (s, q) not in g and panel.L(s, sensor, q) is not None:
                g[(s, q)] = (np.concatenate([[1.0], X_all[pos[s]]]) @ cf)
        ss_res = np.sum(e ** 2, axis=0); ss_tot = np.sum((Y - Y.mean(0)) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2c = 1.0 - ss_res / ss_tot
        r2[q] = np.nanmean(r2c.reshape(NCH, NPARCEL), axis=1)  # leave-one-out R^2 per channel
    flag = "floor" if lam == grid[0] else "ceiling" if lam == grid[-1] else "interior"
    return {"lam": lam, "lam_flag": flag, "lam_scores": score, "g": g, "rows": rows, "coef": coef, "r2": r2, "refit": refit, "names": names}


class ResidualPanel:
    """Wraps a pm.MonthlyPanel: series() returns u = y - g (leave-one-site-out), observed() returns y,
    g() the covariate part; every other attribute is forwarded."""

    def __init__(self, panel, fits):
        self._p = panel; self.fits = fits       # fits: {sensor: fit_g(...) result}

    def __getattr__(self, name):
        return getattr(self._p, name)

    def g_vec(self, site, sensor, seq):
        return self.fits[sensor]["g"].get((site, int(seq)))

    def L(self, site, sensor, seq):
        y = self._p.L(site, sensor, seq)
        if y is None:
            return None
        gv = self.g_vec(site, sensor, seq)
        return y - gv if gv is not None else None

    def series(self, site, sensor, seqs):
        nan = np.full(self.D, np.nan)
        return np.stack([self.L(site, sensor, q) if self.L(site, sensor, q) is not None else nan for q in seqs])

    def observed(self, site, sensor, seqs):
        return self._p.series(site, sensor, seqs)

    def g(self, site, sensor, seqs):
        nan = np.full(self.D, np.nan)
        return np.stack([self.g_vec(site, sensor, q) if self.g_vec(site, sensor, q) is not None else nan for q in seqs])


def describe(fits_by_fill_sensor):
    """panel_covariate.csv rows: per fill x sensor x period the leave-one-out R^2 per channel, the number of
    rows in the fit, the penalty; and at DESCRIBE_PERIODS the per-feature coefficient averaged per channel."""
    out = []
    for (fill, sensor), f in fits_by_fill_sensor.items():
        for q in pm.PERIODS:
            row = {"fill": fill, "sensor": sensor, "seq": q, "period": pm.pid(q), "n_rows": len(f["rows"][q]),
                   "lam": f["lam"], "lam_flag": f["lam_flag"], "n_leverage_one": f["refit"][q], "kind": "r2"}
            row.update({f"r2_ch{c + 1}": float(v) for c, v in enumerate(f["r2"][q])})
            out.append(row)
            if q in DESCRIBE_PERIODS:
                cf = f["coef"][q]
                for k, name in enumerate(["intercept"] + f["names"]):
                    r = {"fill": fill, "sensor": sensor, "seq": q, "period": pm.pid(q), "n_rows": len(f["rows"][q]),
                         "lam": f["lam"], "kind": "coef", "feature": name}
                    r.update({f"coef_ch{c + 1}": float(v) for c, v in enumerate(np.nanmean(cf[k].reshape(NCH, NPARCEL), axis=1))})
                    out.append(r)
    return pd.DataFrame(out)


def build(panel, fill):
    """Fit g for both sensors of one cache; return (ResidualPanel, {(fill, sensor): fit})."""
    fits = {s: fit_g(panel, s) for s in pm.SENSORS}
    return ResidualPanel(panel, fits), {(fill, s): fits[s] for s in pm.SENSORS}
