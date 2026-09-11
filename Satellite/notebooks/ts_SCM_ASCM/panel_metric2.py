"""Metric 2 (M2) for every by-design arm at P10, under every cloud-fill strategy.

M2 = plain RMSE over the outcome dimensions, native units, NO scaler (Jun, 2026-09-01):

    per site i :  rmse_i = || yhat_i - y_i ||_2 / sqrt(M)
    reported   :  mean over the 10 treated sites, train RMSE alongside

`rel_l2_test` (= rmse_i / rms(y_i), the FNO relative-L2 convention) is written as a
secondary column only: within a representation it ranks estimators identically (measured
2026-09-01 on every block), and its denominator is mostly the FSQ level offset.

Fills (four; generation fill is dropped from the project):
    chipmean  latent set latents_biweekly.npz          both sensors, all representations
    histfill  latent set latents_biweekly_histfill.npz both sensors, all representations
    masked    chip-mean latents pooled over VALID parcels only (parcel_validity.npz,
              PARCEL_THR / MIN_VALID of panel_repr); chips below MIN_VALID fall back to
              all-196 pooling                        S2 only, pooled representations only
    maskdrop  masked, and a chip below MIN_VALID is MISSING: the period is dropped for
              the whole donor group (panel_fsc.run_scheme_pergroup convention); P10 must
              survive                                S2 only, pooled representations only

Estimators by design (each representation runs only what it was created for, plus the
two REFERENCE predictors equal_weight (C3) and own_history_mean (C2) — these estimate no
    weights and are not synthetic controls; `role` column marks them):
    chip_mean            scm (identity-embedding SCM baseline), perdim_scm (TLP eq 5, one
                         fit per channel) and its perdim_scm_demeaned / _std variants (the
                         same level handling as the shared arms below, so sharing is the
                         only difference), scm_demeaned
                         (notebook-16 multi-outcome SC on the 5 channels, within-site demeaning),
                         scm_demeaned_std (the same, PLUS Tian-Lee-Panchenko fn 5 / the
                         collaborator's COVID step 2: each outcome x period cell divided by
                         its cross-DONOR SD, and V = diag(1/M_k) applied explicitly)
    gram/quantile/comb.  afsc = Okano-Kurisu AUGMENTED FSC (their unmodified R code via
                         panel_fsc.RBridge, lambda by THEIR cross_val_covmat on a wide log
                         grid), with scm as the un-augmented reference.  Plain FSCM() IS
                         simplex SCM (gate_fscm below proves it on the stored weights), so
                         the scm row is never called "FSC".
    block245/latent980   scm (same coordinate), scm with a parcel permutation (Exp 1
                         alignment arms; D_min is the reported one)
Band means are the collaborator's experiment and are not run here.

Fit window: P01-P09 (or the group's surviving subset of it), test P10, 10 treated sites,
their 5 covariate-matched donors, SLSQP simplex for every Python fit.

Outputs
    panel_metric2_sites.csv        one row per (fill, sensor, repr, estimator, perm, site)
    panel_metric2_p10.csv          means over sites + C2/C3 counts
    panel_metric2_afsc_lambda.csv  CV curve and chosen lambda per afsc cell
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import panel_lib as pl, panel_repr as pr, panel_align as pa, panel_fsc as pf
pd.set_option("display.width", 240)

TRAIN, TEST = list(range(1, 10)), 10
FILLS = ("chipmean", "histfill", "masked", "maskdrop")
POOLED = ("chip_mean", "gram", "quantile", "combined")
REPRS = POOLED + ("block245", "latent980")
FSC_REPRS = ("gram", "quantile", "combined")
ALIGN_ARMS = ("B", "C", "D_min", "D_full", "E")
REFS = ["equal_weight", "own_history_mean"]
ESTIMATORS = {"chip_mean": REFS + ["scm", "perdim_scm", "perdim_scm_demeaned",
                                   "perdim_scm_demeaned_std", "scm_demeaned", "scm_demeaned_std"],
              "gram": REFS + ["scm", "afsc"],
              "quantile": REFS + ["scm", "afsc"],
              "combined": REFS + ["scm", "afsc"],
              "block245": REFS + ["scm"],
              "latent980": REFS + ["scm"]}
LAM_GRID = np.logspace(-8, 8, 33)      # their CV objective, our search (see panel_fsc)
LAM_FLOOR_LIMIT = 1e-14                # extend downward two decades at a time to here
_z = np.load(pl.TS / "parcel_perms.npz"); PERMS = {tuple(k.split("|")): _z[k] for k in _z.files}


# ---------------------------------------------------------------- metric
def rmse(pred, y):
    m = np.isfinite(pred) & np.isfinite(y)
    return float(np.sqrt(np.mean((pred[m] - y[m]) ** 2))) if m.any() else np.nan


def rms(y, pred=None):
    """RMS of the target over the entries the error is scored on (joint finite mask), so
    rel_l2 = rmse / rms exactly (the notebook-12..16 relative-L2 convention)."""
    m = np.isfinite(y) if pred is None else (np.isfinite(y) & np.isfinite(pred))
    return float(np.sqrt(np.mean(y[m] ** 2))) if m.any() else np.nan


# ---------------------------------------------------------------- features
def features(panel, sensor, name, ALL, fill, validity=None):
    """{(site, seq): vector or None (missing)} for seq 1..10, plus the combined block
    scale (None for other representations). `masked`/`maskdrop` pool over valid parcels."""
    F = {}
    for s in ALL:
        for q in range(1, 11):
            v = panel.L(s, sensor, q)
            if v is None:
                F[(s, q)] = None; continue
            if name == "latent980":
                F[(s, q)] = v; continue
            if name == "block245":
                F[(s, q)] = pa.block_mean(v); continue
            A = pr._A(panel, s, sensor, q)
            if fill in ("masked", "maskdrop"):
                nv = pr.n_valid(validity, panel, s, sensor, q)
                if nv is not None and nv >= pr.MIN_VALID:
                    keep = validity[(s, sensor, panel.pid_of_seq[q])] >= pr.PARCEL_THR
                    F[(s, q)] = pr._repr_masked(A, name, keep)
                else:
                    F[(s, q)] = pr._repr_raw(A, name) if fill == "masked" else None
            else:
                F[(s, q)] = pr._repr_raw(A, name)
    scale = None
    if name == "combined":
        # the representation's definition (panel_repr.build_block): each block scaled to
        # unit mean square ONCE over the whole set, so the map stays a linear isometry
        nq = 5 * len(pr.GRIDS_Q)
        X = np.stack([v for v in F.values() if v is not None])
        scale = (float(np.sqrt((X[:, :nq] ** 2).mean())), float(np.sqrt((X[:, nq:] ** 2).mean())))
        for k, v in F.items():
            if v is not None:
                w = v.copy(); w[:nq] /= scale[0]; w[nq:] /= scale[1]; F[k] = w
    return F, scale


def donor_vec(panel, sensor, name, t, j, q, perm_arm):
    p = PERMS[(perm_arm, sensor, t, j)]
    if (p >= 0).sum() < pa.MIN_MATCHED:
        return "drop"
    if panel.L(j, sensor, q) is None:
        return None
    v = pa.aligned_vec(panel, j, sensor, q, p)
    return pa.block_mean(v) if name == "block245" else v


# ---------------------------------------------------------------- augmented FSC (R)
def afsc_cell(blocks, name, scale, tag, bridge, grid=LAM_GRID):
    """Their FSCM_aug_covmat on every group's rectangular block, lambda chosen ONCE per
    cell by their cross_val_covmat pooled over groups on a log grid; extended downward
    while the optimum sits on the floor. Returns (per-group results, lambda rows)."""
    rk = pf.repr_meta(name, pr.GRIDS_Q)
    if name == "combined":                       # scaled quantile block: admissible range
        rk = dict(rk, low=0.0, upp=1.0 / scale[0])
    grid = np.array(sorted(grid)); lam_rows = []
    while True:
        cv_tot = pd.Series(0.0, index=grid); n_ok = pd.Series(0, index=grid)
        for gi, (t, block, T_0) in enumerate(blocks):
            res = bridge.run(block, "lambda_grid", "covmat", T_0, f"{tag}_g{gi}", lambdas=grid, **rk)
            cv = res["lamgrid"].set_index("lambda")["cv"]
            for lam in grid:
                c = cv.get(lam, np.nan)
                if np.isfinite(c):
                    cv_tot[lam] += c; n_ok[lam] += 1
        valid = n_ok == len(blocks)
        if not valid.any():
            raise RuntimeError(f"{tag}: no lambda evaluable on every group")
        lam = float(cv_tot[valid].idxmin())
        lo, hi = float(grid[valid].min()), float(grid[valid].max())
        at_floor = lam == lo
        at_ceiling = (hi - lam) / hi < pf.INTERIOR_TOL
        if at_floor and lo > LAM_FLOOR_LIMIT:
            grid = np.concatenate([np.array([lo * 1e-2, lo * 1e-1]), grid]); continue
        break
    flag = "floor" if at_floor else "ceiling" if at_ceiling else "interior"
    for l_ in grid:
        lam_rows.append({"lambda": float(l_), "cv_sum": float(cv_tot[l_]) if valid[l_] else np.nan,
                         "n_groups_ok": int(n_ok[l_]), "chosen": bool(l_ == lam), "flag": flag})
    out = []
    for gi, (t, block, T_0) in enumerate(blocks):
        res = bridge.run(block, "fit", "covmat", T_0, f"{tag}_fit_g{gi}", lambda_=lam, **rk)
        w = res["weights"].sort_values("donor"); fit = res["fit"].sort_values("period")
        pre_ok = fit["period"] <= T_0; te = fit["period"] == T_0 + 1
        out.append({"site": t, "lambda": lam, "lambda_flag": flag,
                    "w_afsc": w["weight_afsc"].to_numpy(), "w_fsc": w["weight_fsc"].to_numpy(),
                    "rmse_test": float(fit.loc[te, "rmse_afsc_proj"].iloc[0]),
                    "rmse_test_unproj": float(fit.loc[te, "rmse_afsc"].iloc[0]),
                    "rmse_train": float(np.sqrt(np.mean(fit.loc[pre_ok, "rmse_afsc_proj"] ** 2))),
                    "rmse_test_fscm": float(fit.loc[te, "rmse_fsc"].iloc[0])})
    return out, lam_rows


# ---------------------------------------------------------------- evaluation
def evaluate(panel, sensor, name, fill, F, DON, TREAT, estimator, perm_arm=None, m=3,
             validity=None, bridge=None, lam_log=None, scale=None):
    rows, afsc_blocks, base_rows = [], [], []
    for t in TREAT:
        if perm_arm is None:
            dl = list(DON[t]); g = lambda j, q: F[(j, q)]
        else:
            dl = [j for j in DON[t]
                  if not isinstance(donor_vec(panel, sensor, name, t, j, TEST, perm_arm), str)]
            g = lambda j, q: donor_vec(panel, sensor, name, t, j, q, perm_arm)
        ok = lambda q: F[(t, q)] is not None and all(g(j, q) is not None for j in dl)
        pre = [q for q in TRAIN if ok(q)]
        nv = pr.n_valid(validity, panel, t, sensor, TEST) if validity is not None else None
        row = {"fill": fill, "sensor": sensor, "repr": name,
               "M": len(next(v for v in F.values() if v is not None)),
               "estimator": estimator, "perm": perm_arm or "-",
               # references are NOT synthetic controls: no weights are estimated
               "role": "reference" if estimator in REFS else "estimator", "site": t,
               "n_donors": len(dl), "n_pre": len(pre),
               "n_valid_test": np.nan if nv is None else nv}
        if not ok(TEST) or len(pre) < 2 or len(dl) == 0:
            row.update(rmse_test=np.nan, note="dropped: P10 missing, <2 pre periods, or no donors")
            rows.append(row); continue
        y = F[(t, TEST)]; M_ = len(y); nanv = np.full(M_, np.nan)
        # all train periods, NaN rows where a unit is missing: simplex_scm drops a row
        # with any non-finite entry, i.e. the period is dropped for the whole group
        # (the maskdrop convention); own-history and demeaning means use each unit's
        # own available periods (notebook 12-16 semantics, reproduced by gate (b)).
        Y = np.stack([F[(t, q)] if F[(t, q)] is not None else nanv for q in TRAIN])
        Xd = [np.stack([g(j, q) if g(j, q) is not None else nanv for q in TRAIN]) for j in dl]
        xte = np.column_stack([g(j, TEST) for j in dl])
        pred_eq = np.nanmean(xte, 1); pred_own = np.nanmean(Y, 0)
        w_max = np.nan; train_pred = None
        if estimator == "equal_weight":
            pred = pred_eq; train_pred = np.nanmean(np.stack(Xd), 0)
        elif estimator == "own_history_mean":
            pred = pred_own; train_pred = np.repeat(pred_own[None], len(TRAIN), 0)
        elif estimator.startswith("perdim_scm"):
            # Tian-Lee-Panchenko eq (5): one simplex fit PER OUTCOME k. `_demeaned` applies
            # eq (6)'s per-unit x per-outcome pre-period demeaning to each channel's own
            # fit, and `_std` divides each period cell by its cross-donor SD (fn 5); with a
            # single outcome per fit V = diag(1/M_k) is a scalar and drops out. Same level
            # handling as the shared-weight arms, so the two families differ ONLY in
            # whether the weights are shared.
            dm = "_demeaned" in estimator; std = estimator.endswith("_std")
            mt = np.nanmean(Y, 0); md = np.column_stack([np.nanmean(X, 0) for X in Xd])
            pred = np.full(M_, np.nan); train_pred = np.full_like(Y, np.nan); wmx = 0.0
            n_used = 0
            for k in range(M_):
                Xk = np.column_stack([X[:, k] for X in Xd])          # (T, J)
                yk = Y[:, k]
                if dm:
                    yk = yk - mt[k]; Xk = Xk - md[k]
                yk_f, Xk_f = yk, Xk
                if std:
                    sd = np.std(Xk, axis=1, ddof=1)
                    use = np.isfinite(yk) & np.isfinite(Xk).all(1) & np.isfinite(sd) & (sd > 0)
                    sc = np.where(use, 1.0 / np.where(sd > 0, sd, 1.0), np.nan)
                    yk_f = yk * sc; Xk_f = Xk * sc[:, None]; n_used += int(use.sum())
                wk = pa.simplex_scm(yk_f, Xk_f)
                if dm:
                    pred[k] = mt[k] + (xte[k] - md[k]) @ wk; train_pred[:, k] = Xk @ wk + mt[k]
                else:
                    pred[k] = np.nansum(wk * xte[k]); train_pred[:, k] = Xk @ wk
                wmx = max(wmx, float(np.nanmax(wk)))
            w_max = wmx
            if std:
                row["n_cells_used"] = n_used
        elif estimator == "afsc":
            block = np.zeros((1, 1 + len(dl), len(pre) + 1, M_))
            block[0, 0] = np.vstack([np.stack([F[(t, q)] for q in pre]), y[None]])
            for k, j in enumerate(dl):
                block[0, k + 1] = np.vstack([np.stack([g(j, q) for q in pre]), g(j, TEST)[None]])
            assert np.isfinite(block).all(), (t, "afsc block must be dense")
            afsc_blocks.append((t, block, len(pre))); base_rows.append((row, y, pred_eq, pred_own))
            continue
        else:                       # scm / scm_demeaned / scm_demeaned_std
            dm = estimator.startswith("scm_demeaned")
            std = estimator.endswith("_std")
            mt = np.nanmean(Y, 0); md = np.column_stack([np.nanmean(X, 0) for X in Xd])
            Yf = Y - mt if dm else Y
            Xf = [X - md[:, k] for k, X in enumerate(Xd)] if dm else Xd
            yv = Yf.ravel(); Xv = np.column_stack([X.ravel() for X in Xf])
            if std:
                # Tian-Lee-Panchenko fn 5 / her COVID step 2-3: sigma_kt is the SD ACROSS
                # DONORS at that outcome x period cell (ddof = 1, as R's sd); zero-SD cells
                # are dropped; V = diag(1/M_k) with M_k the usable cells of outcome k.
                sd = np.std(np.stack(Xf), axis=0, ddof=1).ravel()
                use = np.isfinite(yv) & np.isfinite(Xv).all(1) & np.isfinite(sd) & (sd > 0)
                Mk = use.reshape(-1, M_).sum(0).astype(float); Mk[Mk == 0] = 1.0
                v = np.tile(1.0 / Mk, len(TRAIN))       # cell (t, k) -> index t*M + k
                sc = np.where(use, np.sqrt(v) / np.where(sd > 0, sd, 1.0), np.nan)
                yv = yv * sc; Xv = Xv * sc[:, None]
                row["n_cells_used"] = int(use.sum())
            w = pa.simplex_scm(yv, Xv)
            w_max = float(np.nanmax(w))
            pred = (mt + (xte - md) @ w) if dm else xte @ w
            Xs = np.stack(Xd)                                     # (J, T, M)
            train_pred = np.tensordot(w, Xs - md.T[:, None, :], axes=(0, 0)) + mt if dm \
                else np.tensordot(w, Xs, axes=(0, 0))
        if train_pred is not None:
            row["rmse_train"] = rmse(train_pred.ravel(), Y.ravel())
        row.update(rmse_test=rmse(pred, y), rms_y_test=rms(y, pred), w_max=w_max)
        row["rel_l2_test"] = row["rmse_test"] / row["rms_y_test"]
        row["C2"] = bool(row["rmse_test"] < rmse(pred_own, y))
        row["C3"] = bool(row["rmse_test"] < rmse(pred_eq, y))
        rows.append(row)
    if afsc_blocks:
        tag = f"m2_{fill}_{sensor}_{name}"
        t0 = time.time()
        res, lam_rows = afsc_cell(afsc_blocks, name, scale, tag, bridge)
        for r in lam_rows:
            lam_log.append({"fill": fill, "sensor": sensor, "repr": name, **r})
        for (row, y, pred_eq, pred_own), r in zip(base_rows, res):
            assert row["site"] == r["site"]
            row.update(rmse_test=r["rmse_test"], rmse_test_unproj=r["rmse_test_unproj"],
                       rmse_train=r["rmse_train"], rms_y_test=rms(y),
                       w_max=float(np.max(r["w_afsc"])), sum_w=float(np.sum(r["w_afsc"])),
                       n_negative=int((r["w_afsc"] < 0).sum()),
                       lambda_=r["lambda"], lambda_flag=r["lambda_flag"],
                       rmse_test_fscm=r["rmse_test_fscm"])
            row["rel_l2_test"] = row["rmse_test"] / row["rms_y_test"]
            row["C2"] = bool(row["rmse_test"] < rmse(pred_own, y))
            row["C3"] = bool(row["rmse_test"] < rmse(pred_eq, y))
            rows.append(row)
        print(f"    afsc {tag}: lambda={res[0]['lambda']:.3g} ({res[0]['lambda_flag']}) "
              f"{time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- gates
def gate_fscm(panel, tol=2e-4):
    """Their un-augmented FSCM() == our simplex SCM on the same block (stored weights,
    chip-mean fill, P01-P08 window). Establishes that `scm` is what the deck called FSC."""
    worst = 0.0
    for name in FSC_REPRS:
        W = pd.read_csv(pl.TS / f"panel_fsc_{name}_weights.csv").query(
            "label == 'chipmean' and method == 'covmat' and repr == @name")
        for sensor in pl.SENSORS:
            groups, _ = pr.complete_groups(panel, sensor, pr.build_groups(panel))
            periods = pr.complete_periods(panel, sensor, groups)
            block = pr.build_block(panel, sensor, name, groups, periods)[0]
            for gi in range(len(groups)):
                stored = W.query("sensor == @sensor and group == @gi + 1").sort_values("donor")
                stored = stored["weight_fsc"].to_numpy()[:5]
                y = block[gi, 0, :8].ravel()
                X = np.column_stack([block[gi, j, :8].ravel() for j in range(1, 6)])
                worst = max(worst, float(np.abs(stored - pa.simplex_scm(y, X)).max()))
    assert worst <= tol, f"FSCM != simplex SCM: max |dw| = {worst:.2e}"
    print(f"gate_fscm: their FSCM() == simplex SCM on every stored group, max|dw| = {worst:.1e}")


def gate_mo(seed=0, J=5, T=9, K=5):
    """(a) our demeaning is Tian-Lee-Panchenko eq (6): Ydot_itk = Y_itk - mean_pre(Y_i.k);
    (b) with a constant cross-donor SD, `scm_demeaned_std` returns `scm_demeaned`'s
    weights, which isolates standardisation from demeaning AND confirms that V =
    diag(1/M_k) is proportional to V = I whenever every outcome has the same cell count."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(T, K)); Xd = [rng.normal(size=(T, K)) for _ in range(J)]
    mt = Y.mean(0); md = np.column_stack([X.mean(0) for X in Xd])
    Yf = Y - mt; Xf = [X - md[:, k] for k, X in enumerate(Xd)]
    assert np.abs(Yf.mean(0)).max() < 1e-12 and np.abs(Yf - (Y - Y.mean(0))).max() < 1e-12
    w_dm = pa.simplex_scm(Yf.ravel(), np.column_stack([X.ravel() for X in Xf]))
    sd = np.full(T * K, 2.5)                     # constant sigma_kt
    Mk = np.full(K, float(T)); v = np.tile(1.0 / Mk, T)
    sc = np.sqrt(v) / sd
    w_std = pa.simplex_scm(Yf.ravel() * sc,
                           np.column_stack([X.ravel() for X in Xf]) * sc[:, None])
    # tol is SLSQP's, not the estimator's: a constant sigma rescales the objective, and
    # simplex_scm's absolute ftol = 1e-12 then stops at a slightly different iterate
    assert np.abs(w_dm - w_std).max() < 1e-5, np.abs(w_dm - w_std).max()
    print(f"gate_mo: eq (6) demeaning exact; constant-sigma std == plain demeaned "
          f"(max|dw| = {np.abs(w_dm - w_std).max():.1e}), so V = diag(1/M_k) ~ V = I")


def summarise(sites):
    keys = ["fill", "sensor", "repr", "M", "estimator", "role", "perm"]
    S = sites.copy(); ok = S["rmse_test"].notna()
    S["C2n"] = S["C2"].where(ok).fillna(False).astype(bool)
    S["C3n"] = S["C3"].where(ok).fillna(False).astype(bool)
    S["okn"] = ok
    agg = {"rmse_test": ("rmse_test", "mean"), "rmse_train": ("rmse_train", "mean"),
           "rel_l2_test": ("rel_l2_test", "mean"), "n_sites": ("okn", "sum"),
           "C2": ("C2n", "sum"), "C3": ("C3n", "sum")}
    if "lambda_" in S:
        agg["lambda"] = ("lambda_", "first"); agg["lambda_flag"] = ("lambda_flag", "first")
    return S.groupby(keys, sort=False).agg(**agg).reset_index()


# ---------------------------------------------------------------- main
def main(fills=FILLS, reprs=REPRS, sensors=pl.SENSORS, run_afsc=True, out_dir=None, tag="", gates=True):
    """`tag` suffixes every output file (panel_metric2{tag}_*.csv); `gates=False` skips the
    method gates. Defaults reproduce the shipped grid."""
    out_dir = pl.TS if out_dir is None else out_dir
    panels = {"chipmean": pl.Panel.from_npz(pl.LATD / "latents_biweekly.npz"),
              "histfill": pl.Panel.from_npz(pl.LATD / "latents_biweekly_histfill.npz")}
    panels["masked"] = panels["maskdrop"] = panels["chipmean"]
    validity = pr.load_validity()
    if gates:
        gate_fscm(panels["chipmean"]); gate_mo()
    ALL = sorted(panels["chipmean"].roster["site_id"]); TREAT = panels["chipmean"].treatments
    DON = pr.matched_donors(panels["chipmean"])
    bridge = pf.RBridge(); sites, lam_log = [], []
    for fill in fills:
        panel = panels[fill]
        for sensor in sensors:
            if fill in ("masked", "maskdrop") and sensor != "sentinel2":
                continue
            for name in reprs:
                if fill in ("masked", "maskdrop") and name not in POOLED:
                    continue
                F, scale = features(panel, sensor, name, ALL, fill, validity)
                if scale: print(f"  combined block scale ({fill},{sensor}): q {scale[0]:.4f}, gram {scale[1]:.4f}")
                for est in ESTIMATORS[name]:
                    if est == "afsc" and not run_afsc: continue
                    sites += evaluate(panel, sensor, name, fill, F, DON, TREAT, est,
                                      validity=validity, bridge=bridge, lam_log=lam_log, scale=scale)
                if name in ("latent980", "block245"):
                    for arm in ALIGN_ARMS:
                        sites += evaluate(panel, sensor, name, fill, F, DON, TREAT, "scm", perm_arm=arm,
                                          validity=validity)
                print(f"{fill:8s} {sensor:9s} {name:9s} done", flush=True)
    S = pd.DataFrame(sites)
    S.to_csv(out_dir / f"panel_metric2{tag}_sites.csv", index=False)
    P = summarise(S); P.to_csv(out_dir / f"panel_metric2{tag}_p10.csv", index=False)
    if lam_log:
        pd.DataFrame(lam_log).to_csv(out_dir / f"panel_metric2{tag}_afsc_lambda.csv", index=False)
    print(P.to_string(index=False))
    return S, P


if __name__ == "__main__":
    main()
