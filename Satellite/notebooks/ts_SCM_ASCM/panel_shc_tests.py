"""Consistency tests for panel_shc (run: python panel_shc_tests.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import panel_shc as sh

# 1. eq (7) blocks reproduce the slide-13 / paper example: T0 = 10, m = 4, n = 2 -> N = 5
tp, tpost, hist = sh.blocks(10, 4, 2)
assert tp == [7, 8, 9, 10] and tpost == [11, 12] and len(hist) == 5
assert hist[0] == ([5, 6, 7, 8], [9, 10]) and hist[-1] == ([1, 2, 3, 4], [5, 6]), hist
for i, (pre, post) in enumerate(hist, 1):
    z = 2 + i - 1
    assert pre == [q - z for q in tp] and post == [q - z for q in tpost]
_, _, h3 = sh.blocks(10, 4, 3); assert len(h3) == 4
print("blocks: eq (7) reproduced (T0=10, m=4, n=2 -> 5 blocks; n=3 -> 4)")

# 2. smoother limits: small h -> (nearly) the series itself; huge h -> the global OLS line
rng = np.random.default_rng(0); T0, D = 10, 7
Y = rng.normal(size=(T0, D))
Ls = sh.smooth_latent(Y, 0.02)
assert np.abs(Ls - Y).max() < 1e-3, np.abs(Ls - Y).max()
Lb = sh.smooth_latent(Y, 1e4)
t = np.arange(T0) / T0; Z = np.column_stack([np.ones(T0), t])
line = Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
assert np.abs(Lb - line).max() < 1e-6, np.abs(Lb - line).max()
# an exactly linear series is returned exactly at any h (local-linear is exact for lines)
Ylin = np.outer(np.arange(T0), rng.normal(size=D)) + rng.normal(size=D)
for h in (0.05, 0.3, 3.0):
    assert np.abs(sh.smooth_latent(Ylin, h) - Ylin).max() < 1e-9
# a missing period gets zero weight and its own row is still estimated
Ym = Y.copy(); Ym[4] = np.nan
Lm = sh.smooth_latent(Ym, 0.3)
assert np.isfinite(Lm).all() and np.abs(Lm[[0, 9]] - sh.smooth_latent(Y, 0.3)[[0, 9]]).max() > 0
print("smoother: identity at h->0, global line at h->inf, exact on lines, missing rows handled")

# 3. LOO-CV picks a sensible bandwidth on smooth signal + noise, and the curve is recorded
tt = np.arange(T0)
Ysig = np.sin(2 * np.pi * tt / 10)[:, None] * rng.normal(size=(1, D)) + 0.7 * rng.normal(size=(T0, D))
h, flag, rows = sh.choose_bandwidth(Ysig)
assert len(rows) == len(sh.H_GRID) and sum(r["chosen"] for r in rows) == 1 and flag in ("floor", "interior", "ceiling")
cv = np.array([r["cv"] for r in rows]); assert np.isfinite(cv).all() and cv.min() == cv[[r["chosen"] for r in rows]][0]
# nearly noise-free signal -> CV prefers the least smoothing (floor); heavy noise -> more smoothing
h_clean, f_clean, _ = sh.choose_bandwidth(np.sin(2 * np.pi * tt / 10)[:, None] * rng.normal(size=(1, D)) + 0.01 * rng.normal(size=(T0, D)))
assert h >= h_clean, (h, h_clean)
print(f"bandwidth: LOO-CV curve recorded; noisy series h = {h:.3g} ({flag}), near-clean series h = {h_clean:.3g} ({f_clean})")

# 4. weights: simplex, and a planted convex combination is recovered by stepwise selection
tp, tpost, hist = sh.blocks(10, 3, 2)
L = rng.normal(size=(10, D))
w_true = np.array([0.6, 0.4])
for q, (q1, q2) in zip(tp, zip(hist[0][0], hist[3][0])):        # treated pre = 0.6*block1 + 0.4*block4
    L[q - 1] = w_true[0] * L[q1 - 1] + w_true[1] * L[q2 - 1]
y, X = sh._design(L, tp, hist)
sel, w, mse, path, cands = sh.stepwise_donors(y, X, zeta=1e-10)
assert sorted(sel) == [0, 3] and np.abs(np.sort(w) - np.sort(w_true)).max() < 1e-4 and mse < 1e-8, (sel, w, mse)
assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
print(f"stepwise: planted blocks {sorted(c + 1 for c in sel)} recovered, w = {np.round(w, 3)}, mse = {mse:.1e}")

# 5. counterfactual: weights on the donors' forward rows; effect = observed - that
Yfull = np.vstack([L, rng.normal(size=(2, D))])
cf = sh.counterfactual(L, Yfull, w, sel, hist, tpost)
k = 0
manual = w[0] * L[hist[sel[0]][1][k] - 1] + w[1] * L[hist[sel[1]][1][k] - 1]
assert np.abs(cf["l0"][k] - manual).max() < 1e-12 and np.abs(cf["delta"][k] - (Yfull[10] - manual)).max() < 1e-12
print("counterfactual: l0 = sum_j w_j l_hat[hist j, post]; delta = y_post - l0")

# 6. end-to-end on a synthetic series with the post periods present
res = sh.shc_fit(Yfull, T0=10, m=3, n=2)
assert res["ok"] and res["N"] == 6 and abs(sum(res["w"]) - 1) < 1e-9 and res["delta"].shape == (2, D)
print(f"shc_fit: N = {res['N']}, selected {res['selected']}, h = {res['h']:.3g} ({res['h_flag']}), "
      f"n_eff = {res['n_eff']:.2f}, mse_pre = {res['mse_pre']:.2e}")

# ----------------------------------------------------------------- covariate mapping g (plan 2026-09-10)
import pandas as pd
import panel_covariate as pc
import panel_monthly as pm

# 7. planted y = a + B x + noise on 286 synthetic sites: B recovered at zero penalty; LOO residual property
rng7 = np.random.default_rng(7)
Xs = np.column_stack([rng7.integers(0, 2, size=(286, 6)).astype(float), rng7.normal(size=(286, 2))])
Btrue = rng7.normal(size=(9, 40)); Ys = np.column_stack([np.ones(286), Xs]) @ Btrue + 1e-9 * rng7.normal(size=(286, 40))
coef, h = pc.ridge_fit(Xs, Ys, 0.0)
assert np.abs(coef - Btrue).max() < 1e-6, np.abs(coef - Btrue).max()
e_loo, fit, _, _ = pc.loo_residuals(Xs, Ys, 0.3)
i = 5
coef_i, _ = pc.ridge_fit(np.delete(Xs, i, 0), np.delete(Ys, i, 0), 0.3)
pred_i = np.concatenate([[1.0], Xs[i]]) @ coef_i
assert np.abs((Ys[i] - pred_i) - e_loo[i]).max() < 1e-8, "closed-form leave-one-out differs from refitting"
print("covariate 7: planted coefficients recovered (max |dB| < 1e-6); closed-form leave-one-site-out equals a refit without the site")

# 8. with all descriptors zero g is the cross-site mean; untreated residuals sum to zero at every period
coef0, _ = pc.ridge_fit(np.zeros((286, 8)), Ys, 1.0)
assert np.abs(coef0[0] - Ys.mean(0)).max() < 1e-9 and np.abs(coef0[1:]).max() < 1e-12
print("covariate 8: with zero descriptors g_t is the cross-site mean")

# 9-11. on the real panel (chip-mean fill, both sensors): leave-one-site-out holds, treated post rows absent
panel = pm.MonthlyPanel.from_npz(pm.CACHE_CHIPMEAN)
for sensor in pm.SENSORS:
    f = pc.fit_g(panel, sensor)
    treated = panel.treatments(sensor)
    for q in pm.PERIODS:
        if q >= pm.TREAT_START:
            assert not (set(f["rows"][q]) & set(treated)), f"treated site in the fit at P{q:02d}"
        else:
            assert set(f["rows"][q]) <= {s for s in panel.sites(sensor) if panel.L(s, sensor, q) is not None}
    # perturb one site's latent at P05: no other site's g changes; the perturbed site's own g is unchanged too
    s0 = treated[0]; q0 = 5
    lat = dict(panel.flat); key = (s0, sensor, q0)
    lat[key] = lat[key] + 1.0
    p2 = pm.MonthlyPanel.__new__(pm.MonthlyPanel); p2.__dict__.update(panel.__dict__); p2.flat = lat
    f2 = pc.fit_g(p2, sensor, lam=f["lam"])
    others = [s for s in f["rows"][q0] if s != s0]
    d_other = max(np.abs(f["g"][(s, q0)] - f2["g"][(s, q0)]).max() for s in others)
    d_self = np.abs(f["g"][(s0, q0)] - f2["g"][(s0, q0)]).max()
    assert d_self < 1e-9, d_self
    assert d_other > 1e-6, "perturbing a site's latent should move the other sites' fits (it is in their fit)"
    rp = pc.ResidualPanel(panel, {sensor: f})
    u = rp.series(s0, sensor, pm.PERIODS); y = rp.observed(s0, sensor, pm.PERIODS); g = rp.g(s0, sensor, pm.PERIODS)
    assert np.allclose(u + g, y, equal_nan=True)
    print(f"covariate 9-11 ({sensor}): penalty {f['lam']:g} ({f['lam_flag']}), leverage-one rows refitted: {sum(f['refit'].values())}; treated sites absent from every P11+ fit; a site's own g does not "
          f"depend on its own latent (diff {d_self:.1e}); u + g = y")

# 12. regression: --covariate none on a subset reproduces the 4 September fits exactly (written to a scratch dir)
import panel_shc_run as run, tempfile, os
F_old = pd.read_csv("panel_shc_fits.csv")
sub = F_old[(F_old.fill == "chipmean") & (F_old.n == 2) & (F_old.m == 4) & (F_old.group == "treatment")]
rows = []
for sensor in pm.SENSORS:
    for site in panel.treatments(sensor):
        f, e, h, p = run.run_site(panel, site, sensor, "chipmean", 2, 4, keep_curves=False)
        rows.append(f)
F_new = pd.DataFrame(rows)
num = [c for c in F_new.columns if F_new[c].dtype.kind == "f" and c in sub.columns]
merged = sub.merge(F_new, on=["fill", "sensor", "site", "n", "m"], suffixes=("_old", "_new"))
assert len(merged) == len(sub) == len(F_new), (len(merged), len(sub), len(F_new))
worst = max(np.nanmax(np.abs(merged[f"{c}_old"] - merged[f"{c}_new"])) for c in num)
assert worst < 1e-12, worst          # the stored CSV carries 16 significant digits; differences are round-trip only
assert (merged.selected_old == merged.selected_new).all()
print(f"covariate 12: the driver without --covariate reproduces the 4 September fits on {len(merged)} treated cells (max diff {worst:g}; selected blocks identical)")
print("ALL TESTS PASSED")
