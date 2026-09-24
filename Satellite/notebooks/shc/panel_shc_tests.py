"""Consistency tests for panel_shc (run: python panel_shc_tests.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys
_SHARED = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "ts_SCM_ASCM")   # panel_lib, panel_monthly, panel_align live there
if _SHARED not in _sys.path:
    _sys.path.insert(0, _SHARED)
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

# ----------------------------------------------------------------- learned stage 1 (plan 2026-09-11)
import pandas as pd
import panel_generator as pg
import panel_monthly as pm

# 7. a precomputed l_hat equal to the kernel's own output reproduces the kernel fit exactly (stage 2 untouched)
res_k = sh.shc_fit(Yfull, T0=10, m=3, n=2)
res_l = sh.shc_fit(Yfull, T0=10, m=3, n=2, L_hat=res_k["L"])
d7 = max(np.abs(res_k["w"] - res_l["w"]).max(), abs(res_k["mse_pre"] - res_l["mse_pre"]),
         np.abs(res_k["delta"] - res_l["delta"]).max(), np.abs(res_k["l0"] - res_l["l0"]).max())
assert res_k["selected"] == res_l["selected"] and d7 == 0.0 and res_l["h_flag"] == "generator", (d7, res_l["h_flag"])
print("generator 7: shc_fit with L_hat = the kernel's l_hat reproduces the kernel fit exactly (max diff 0)")

# 8. training cells: nothing after the fit's T0, every cell usable, held-out 10 % disjoint from training
panel = pm.MonthlyPanel.from_npz(pm.CACHE_CHIPMEAN)
for sensor in pm.SENSORS:
    for T0 in pg.FITS.values():
        cl = pg.cells(panel, sensor, T0)
        assert all(1 <= q <= T0 for _, q in cl) and all(panel.L(s_, sensor, q) is not None for s_, q in cl)
        tr, ho = pg.split_cells(cl)
        assert not (set(tr) & set(ho)) and len(tr) + len(ho) == len(cl) and len(ho) == int(round(0.1 * len(cl)))
        tr2, ho2 = pg.split_cells(cl)
        assert ho == ho2, "held-out draw is not reproducible"
    print(f"generator 8 ({sensor}): cells stop at the fit's T0, all usable, 10 % held out, disjoint, reproducible")

D = 980
# 9. planted panel y = a_site + b_period + noise, with a and b spatially smooth 5 x 14 x 14 maps (a random 4 x 4
#    grid per channel interpolated to 14 x 14): both arms recover the held-out cells (R^2 > 0.9)
def _smooth_maps(r, n, scale=0.3):
    """(n, D): per item and channel a random 4x4 grid bilinearly interpolated to 14x14."""
    x = np.linspace(0, 3, 14); out = np.empty((n, 5, 14, 14))
    for i in range(n):
        for c in range(5):
            g = r.normal(size=(4, 4)) * scale
            rows = np.stack([np.interp(x, np.arange(4), g[k]) for k in range(4)])          # (4, 14)
            out[i, c] = np.stack([np.interp(x, np.arange(4), rows[:, j]) for j in range(14)], axis=1)
    return out.reshape(n, D)
class _Planted:
    def __init__(self, n_sites=60, T0=10, seed=9):
        r = np.random.default_rng(seed)
        self._sites = [f"site_{i:03d}" for i in range(n_sites)]
        a = _smooth_maps(r, n_sites)[:, None, :]; b = _smooth_maps(r, T0)[None, :, :]
        self.Y = 0.5 + a + b + 0.02 * r.normal(size=(n_sites, T0, D))
    def sites(self, sensor, group=None): return list(self._sites)
    def L(self, site, sensor, q): return self.Y[self._sites.index(site), q - 1]
pl_ = _Planted()
cfg9 = dict(pg.CFG, max_epochs=2000, patience=100)   # test-only budget: the cnn head sits on a plateau for ~250 epochs before learning
for arm in pg.ARMS:
    f9 = pg.train(pl_, "sentinel1", arm, 10, cfg=cfg9, log=None)
    assert f9["desc"]["r2_all"] > 0.9, (arm, f9["desc"]["r2_all"])
    assert f9["L_hat"]["site_000"].shape == (10, D) and f9["desc"]["stop"] in ("patience", "max_epochs")
    print(f"generator 9 ({arm}): planted site + period structure recovered on the held-out cells, R2 = {f9['desc']['r2_all']:.3f}, "
          f"best epoch {f9['desc']['best_epoch']}/{f9['desc']['epochs_run']} on {f9['desc']['device']}")

# 10. regression: the kernel arm reproduces the 4 September fits exactly (a subset, nothing written)
import panel_shc_run as run
F_old = pd.read_csv("panel_shc_fits.csv")
sub = F_old[(F_old.fill == "chipmean") & (F_old.n == 2) & (F_old.m == 4) & (F_old.group == "treatment")]
rows = []
for sensor in pm.SENSORS:
    for site in panel.treatments(sensor):
        f, e, h, p, lam = run.run_site(panel, site, sensor, "chipmean", 2, 4, keep_curves=False)
        assert lam == {}
        rows.append(f)
F_new = pd.DataFrame(rows)
num = [c for c in F_new.columns if F_new[c].dtype.kind == "f" and c in sub.columns]
merged = sub.merge(F_new, on=["fill", "sensor", "site", "n", "m"], suffixes=("_old", "_new"))
assert len(merged) == len(sub) == len(F_new), (len(merged), len(sub), len(F_new))
worst = max(np.nanmax(np.abs(merged[f"{c}_old"] - merged[f"{c}_new"])) for c in num)
assert worst < 1e-12, worst          # the stored CSV carries 16 significant digits; differences are round-trip only
assert (merged.selected_old == merged.selected_new).all()
print(f"generator 10: the kernel arm reproduces the 4 September fits on {len(merged)} treated cells (max diff {worst:g}; selected blocks identical)")

# 11. the cnn head flattens in the cache's (channel, row, col) order, so channel_means() reads the right coordinates
torch = pg._lazy_torch("cpu")
g11 = pg.make_generator(3, 10, "cnn", device="cpu")
with torch.no_grad():
    S11 = torch.tensor([0, 1, 2]); Q11 = torch.tensor([0, 4, 9])
    grid = g11.grid(S11, Q11).numpy(); flat = g11(S11, Q11).numpy()
assert grid.shape == (3, 5, 14, 14) and np.array_equal(flat, grid.reshape(3, -1))
for c in range(5):
    assert np.array_equal(flat[:, c * 196:(c + 1) * 196], grid[:, c].reshape(3, 196))
print("generator 11: cnn output flattened as (channel, row, col), matching the latent cache's ravel")

# 12. lambda stored by a generator run equals observed - (weights @ generator forward rows), recomputed from the stored files
if os.path.exists("panel_shc_lambda_mlp.npz") and os.path.exists("panel_generator_lhat_mlp.npz"):
    lam = np.load("panel_shc_lambda_mlp.npz"); Lh, _ = pg.load_lhat("panel_generator_lhat_mlp.npz")
    Fm = pd.read_csv("panel_shc_fits_mlp.csv"); Fm = Fm[(Fm.fill == "chipmean") & (Fm.ok == True)]
    worst12, n12 = 0.0, 0
    for _, r in Fm.iterrows():
        pre, post, hist = sh.blocks(10, int(r.m), int(r.n))
        sel = [int(v) - 1 for v in str(r.selected).split()]; w = [float(v) for v in str(r.w).split()]
        L = Lh[("chipmean", r.sensor, 10, r.site)]
        for k, q in enumerate(post):
            key = f"chipmean|{r.sensor}|{r.site}|{int(r.n)}|{int(r.m)}|P{q:02d}"
            if key not in lam.files:
                continue
            y = panel.L(r.site, r.sensor, q)
            l0 = sum(w[j] * L[hist[c][1][k] - 1] for j, c in enumerate(sel))
            worst12 = max(worst12, float(np.abs((y - l0) - lam[key]).max())); n12 += 1
    assert n12 > 0 and worst12 < 1e-3, (n12, worst12)     # weights are stored to 4 decimals in the CSV
    print(f"generator 12: lambda = observed - weights @ generator forward rows on {n12} stored vectors (max diff {worst12:.1e}, weights read to 4 decimals)")
else:
    print("generator 12: skipped (no generator run stored yet)")
print("ALL TESTS PASSED")
