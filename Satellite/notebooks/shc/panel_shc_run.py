"""Run SHC (panel_shc) on the monthly latent panel: temporal placebo, negative controls,
effect run. Design: memory `project-shc-corrected-design`, plan `quirky-growing-wombat`.

Grid: fill {chipmean, histfill} x sensor {S1, S2} x n {2, 3} x m {2 .. T0 - n}, on every
site the sensor has (treated sites and their matched counterfactuals).

Per site x sensor x fill x (m, n):
  effect run        T0 = 10 (P01-P10 pre), post = P11..P(10+n) — P11 is the treatment period.
                    Stage 1 on P01-P10, blocks, stepwise donors, weights, counterfactual,
                    Delta = Y_post - l_hat0 (and the raw eq-27 variant).
  temporal placebo  T0' = 10 - n, post = P(T0'+1)..P10, all pre-Helene, the identical pipeline
                    on P01..P(T0') only; scored as RMSE over the D coordinates of the observed
                    period against l_hat0 (and against the raw variant), with the site's own
                    P01..P(T0') mean as the reference row. Needs N' = T0' - n - (m-1) >= 1.
  negative controls the effect run on the matched counterfactual sites; per treated site the
                    rank of its |Delta| among its controls -> p (floor 1/(1+n_controls)).

Outputs (notebooks/ts_SCM_ASCM/):
  panel_shc_fits.csv        one row per site x sensor x fill x n x m (weights, donors, h, MSE_pre,
                            placebo errors, effect sizes per post period)
  panel_shc_effects.csv     one row per site x ... x post period: ||Delta||/sqrt(D), per-channel
                            mean Delta, raw variant, validity of the period
  panel_shc_bandwidth.csv   LOO-CV curves (treated sites, effect window)
  panel_shc_path.csv        stepwise selection path and every candidate tried (treated sites)
  panel_shc_summary.csv     means over the treated sites per cell, N_eff averaged site by site,
                            negative-control ranks

    python panel_shc_run.py                      # everything (the 4 September run; outputs unsuffixed)
    python panel_shc_run.py --quick              # treated sites only, one fill (smoke test)
    python panel_shc_run.py --stage1 mlp|cnn     # stage 1 = the learned generator of panel_generator.py (2026-09-11):
                                                 # treated sites only, outputs *_mlp.csv / *_cnn.csv, plus
                                                 # panel_generator_<arm>.csv (description of f), *_curves, *_lhat.npz,
                                                 # panel_shc_lambda_<arm>.npz (lambda = observed - counterfactual, per site and month)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import os as _os, sys as _sys
_SHARED = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "ts_SCM_ASCM")   # panel_lib, panel_monthly, panel_align live there
if _SHARED not in _sys.path:
    _sys.path.insert(0, _SHARED)
import panel_lib as pl, panel_monthly as pm, panel_shc as sh
import panel_generator as pg

FILLS = {"chipmean": pm.CACHE_CHIPMEAN, "histfill": pm.CACHE_HISTFILL}
NS = (2, 3)
T0 = pm.T0
NCH, NPARCEL = 5, 196
P11_NOTE = {"sentinel2": "S2 P11 (Sep 2024) composite sources are 2/7/22 Sep: no post-landfall image",
            "sentinel1": "S1 P11 mixes 22 Sep with 28 Sep acquisitions"}


def rms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan


def channel_means(delta):
    return np.nanmean(np.asarray(delta, float).reshape(NCH, NPARCEL), axis=1)


def run_site(panel, site, sensor, fill, n, m, keep_curves, lhat=None):
    """Effect run + temporal placebo for one site. Returns (fit row, effect rows, h rows, path rows, lambda).
    lhat: None (kernel stage 1) or {T0: (T0, D) array} from the learned generator, one array per fit
    (T0 = 10 effect, 8 validation n = 2, 7 validation n = 3); stage 2 is identical either way.
    lambda: {post period: observed - counterfactual (D,)} for the effect run, empty without lhat."""
    Y = panel.series(site, sensor, pm.PERIODS)            # (21, D): the observed latent
    Yobs = Y
    G = np.zeros_like(Y)                                  # kept so the validation lines read as before
    D = Y.shape[1]
    lam = {}
    base = {"fill": fill, "sensor": sensor, "site": site, "group": panel.group_of[site],
            "matched_treatment": panel.treat_of[site], "n": n, "m": m, "T0": T0}
    fit = dict(base)
    eff_rows, h_rows, path_rows = [], [], []
    # ---- effect run
    r = sh.shc_fit(Y, T0, m, n, L_hat=None if lhat is None else lhat[T0])
    fit.update(h=r["h"], h_flag=r["h_flag"], ok=r["ok"], note=r["note"])
    if keep_curves:
        h_rows = [{**base, **hr} for hr in r["h_rows"]]
    if r["ok"]:
        fit.update(N=r["N"], selected=" ".join(map(str, r["selected"])), n_selected=len(r["selected"]),
                   w=" ".join(f"{v:.4f}" for v in r["w"]), w_max=float(np.max(r["w"])), n_eff=r["n_eff"],
                   mse_pre=r["mse_pre"], passes_nu=r["passes_nu"], nu=r["nu"], zeta=r["zeta"],
                   var_pre=r["var_pre"], single_block=r["single_block"], mse_single=r["mse_single"],
                   mse_allin=r["mse_all"], n_post_observed=r["n_post_observed"],
                   pre_periods=f"P{r['treated_pre'][0]:02d}-P{r['treated_pre'][-1]:02d}",
                   post_periods=f"P{r['treated_post'][0]:02d}-P{r['treated_post'][-1]:02d}")
        if keep_curves:
            path_rows = [{**base, "kind": "path", **p} for p in r["path"]] + \
                        [{**base, "kind": "candidate", **c} for c in r["candidates"]]
        for k, q in enumerate(r["treated_post"]):
            d = r["delta"][k]; obs = np.isfinite(d).all()
            e = {**base, "post_period": f"P{q:02d}", "k": k + 1, "observed": bool(obs),
                 "valid_frac": panel.vfrac.get((site, sensor, q), np.nan),
                 "delta_rms": rms(d), "delta_raw_rms": rms(r["delta_raw"][k]),
                 "y_post_rms": rms(Yobs[q - 1]), "l0_rms": rms(r["l0"][k] + G[q - 1]),   # observed y; counterfactual l0
                 "note": P11_NOTE[sensor] if q == pm.TREAT_START else ""}
            for c, v in enumerate(channel_means(d)):
                e[f"delta_ch{c + 1}"] = float(v)
            eff_rows.append(e)
            fit[f"delta_rms_k{k + 1}"] = e["delta_rms"]
            if lhat is not None and obs:
                lam[q] = np.asarray(d, float)
        fit["delta_rms_mean"] = float(np.nanmean([fit[f"delta_rms_k{k + 1}"] for k in range(n)]))
    # ---- temporal placebo (same m, n; window ends before Helene)
    T0p = T0 - n
    if T0p - n - (m - 1) >= 1:
        rp = sh.shc_fit(Y[:T0], T0p, m, n, L_hat=None if lhat is None else lhat[T0p])
        fit.update(placebo_T0=T0p, placebo_ok=rp["ok"], placebo_h=rp["h"], placebo_h_flag=rp["h_flag"])
        if rp["ok"]:
            fit.update(placebo_selected=" ".join(map(str, rp["selected"])),
                       placebo_w=" ".join(f"{v:.4f}" for v in rp["w"]), placebo_n_eff=rp["n_eff"],
                       placebo_mse_pre=rp["mse_pre"])
            own_mean = np.nanmean(Yobs[:T0p], axis=0)                          # reference: own mean of the observed latent
            errs, errs_raw, errs_ref = [], [], []
            for k, q in enumerate(rp["treated_post"]):
                yq = Yobs[q - 1]                                                # observed latent of the held-out month
                errs.append(rms(yq - (rp["l0"][k] + G[q - 1])))                 # counterfactual l0
                errs_raw.append(rms(yq - (rp["y0_raw"][k] + G[q - 1])))
                errs_ref.append(rms(yq - own_mean))
                fit[f"placebo_rmse_P{q:02d}"] = errs[-1]
            fit.update(placebo_rmse=float(np.nanmean(errs)), placebo_rmse_raw=float(np.nanmean(errs_raw)),
                       placebo_rmse_ownmean=float(np.nanmean(errs_ref)),
                       placebo_beats_ownmean=bool(np.nanmean(errs) < np.nanmean(errs_ref)))
    else:
        fit.update(placebo_T0=T0p, placebo_ok=False)
    return fit, eff_rows, h_rows, path_rows, lam


def summarise(F):
    """Means over treated sites per cell; N_eff averaged site by site; negative-control ranks."""
    out = []
    keys = ["fill", "sensor", "n", "m"]
    T = F[(F.group == "treatment") & (F.ok == True)]
    C = F[(F.group == "counterfactual") & (F.ok == True)]
    for k, g in T.groupby(keys, sort=False):
        row = dict(zip(keys, k))
        row.update(n_sites=len(g), N=int(g.N.iloc[0]), h_median=float(g.h.median()),
                   h_interior=int((g.h_flag == "interior").sum()),
                   n_selected_mean=float(g.n_selected.mean()),
                   most_recent_only=int((g.selected == "1").sum()),      # block 1 = shift n = most recent
                   oldest_only=int((g.selected == str(int(g.N.iloc[0]))).sum()),
                   n_eff_mean=float(g.n_eff.mean()), mse_pre_mean=float(g.mse_pre.mean()),
                   passes_nu=int(g.passes_nu.sum()),
                   delta_rms_mean=float(g.delta_rms_mean.mean()))
        if "placebo_rmse" in g and g.placebo_ok.fillna(False).any():
            gp = g[g.placebo_ok == True]
            row.update(placebo_sites=len(gp), placebo_rmse=float(gp.placebo_rmse.mean()),
                       placebo_rmse_raw=float(gp.placebo_rmse_raw.mean()),
                       placebo_rmse_ownmean=float(gp.placebo_rmse_ownmean.mean()),
                       placebo_beats_ownmean=int(gp.placebo_beats_ownmean.sum()))
        # negative controls: rank of each treated site's |Delta| among its own controls
        ranks, ps = [], []
        for _, t in g.iterrows():
            c = C[(C.fill == t.fill) & (C.sensor == t.sensor) & (C.n == t.n) & (C.m == t.m) &
                  (C.matched_treatment == t.site)]
            if len(c) and np.isfinite(t.delta_rms_mean):
                rank = 1 + int((c.delta_rms_mean >= t.delta_rms_mean).sum())
                ranks.append(rank); ps.append(rank / (1 + len(c)))
        if ranks:
            row.update(nc_sites=len(ranks), nc_rank1=int(sum(r == 1 for r in ranks)),
                       nc_p_mean=float(np.mean(ps)), nc_controls_mean=float(np.mean([1 / p - 1 for p in ps])))
        out.append(row)
    return pd.DataFrame(out)


def main(quick=False, stage1="kernel", out_dir=".", suffix=None):
    """stage1: 'kernel' = the paper's kernel smoother (the 4 September run, outputs unsuffixed, every site);
    'mlp' | 'cnn' = the learned generator of panel_generator.py in that arm (treated sites only, outputs
    suffixed _mlp / _cnn, plus the description of f and the lambda vectors). out_dir lets a test write elsewhere."""
    suffix = {"kernel": "", "mlp": "_mlp", "cnn": "_cnn"}[stage1] if suffix is None else suffix
    out = lambda name: os.path.join(out_dir, f"{name}{suffix}.csv")
    fits, effs, hs, paths = [], [], [], []
    gen_fits, lambdas, s1_series = {}, {}, None
    for fill, cache in FILLS.items():
        if quick and fill != "chipmean":
            continue
        panel = pm.MonthlyPanel.from_npz(cache)
        lhat_of = None
        if stage1 != "kernel":
            # Sentinel-1 latents are identical under both fills (no cloud mask): train once, reuse when verified equal
            cur_s1 = np.stack([panel.series(s, "sentinel1", pm.PERIODS) for s in panel.sites("sentinel1")])
            gen_fits[fill] = {}
            for sensor in pm.SENSORS:
                prev = next(iter(gen_fits.values())) if len(gen_fits) > 1 else None
                if sensor == "sentinel1" and prev is not None and s1_series is not None and \
                        np.array_equal(cur_s1, s1_series, equal_nan=True):
                    gen_fits[fill][sensor] = prev[sensor]
                    print(f"  {fill} sentinel1: latents identical to the first fill, generators reused", flush=True)
                    continue
                gen_fits[fill][sensor] = {T0: pg.train(panel, sensor, stage1, T0, tag=f"{fill} ") for T0 in pg.FITS.values()}
            s1_series = cur_s1
            lhat_of = lambda site, sensor: {T0: f["L_hat"][site] for T0, f in gen_fits[fill][sensor].items()}
        for sensor in pm.SENSORS:
            treated = panel.treatments(sensor)
            sites = list(treated) if (quick or stage1 != "kernel") else \
                list(treated) + [c for t in treated for c in panel.controls_of(t, sensor)]
            t0 = time.time()
            for n in NS:
                for m in range(2, T0 - n + 1):
                    for site in sites:
                        f, e, h, p, lam = run_site(panel, site, sensor, fill, n, m, keep_curves=site in treated,
                                                   lhat=None if lhat_of is None else lhat_of(site, sensor))
                        fits.append(f); effs += e; hs += h; paths += p
                        for q, v in lam.items():
                            lambdas[f"{fill}|{sensor}|{site}|{n}|{m}|P{q:02d}"] = v.astype(np.float32)
                print(f"  {fill} {sensor} n={n}: {len(sites)} sites x m=2..{T0 - n} done ({time.time() - t0:.0f}s)", flush=True)
    F = pd.DataFrame(fits); F.to_csv(out("panel_shc_fits"), index=False)
    pd.DataFrame(effs).to_csv(out("panel_shc_effects"), index=False)
    pd.DataFrame(hs).to_csv(out("panel_shc_bandwidth"), index=False)
    pd.DataFrame(paths).to_csv(out("panel_shc_path"), index=False)
    S = summarise(F); S.to_csv(out("panel_shc_summary"), index=False)
    if stage1 != "kernel":
        pg.describe(gen_fits).to_csv(os.path.join(out_dir, f"panel_generator{suffix}.csv"), index=False)
        pg.curves(gen_fits).to_csv(os.path.join(out_dir, f"panel_generator_curves{suffix}.csv"), index=False)
        pg.save_lhat(gen_fits, os.path.join(out_dir, f"panel_generator_lhat{suffix}.npz"))
        np.savez_compressed(os.path.join(out_dir, f"panel_shc_lambda{suffix}.npz"), **lambdas)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    return F, S


if __name__ == "__main__":
    st = sys.argv[sys.argv.index("--stage1") + 1] if "--stage1" in sys.argv else "kernel"
    main(quick="--quick" in sys.argv, stage1=st)
