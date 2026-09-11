"""Build notebooks/ts_SCM_ASCM/23_shc_covariate.ipynb: the SHC of notebook 22 rerun on y - g, with g the learned
covariate mapping (month + land cover, elevation, slope; panel_covariate.py), beside the 4 September run.
Left column of every table = the existing CSVs (covariate-free), right column = the *_covariate.csv files.
Run once to create the notebook, then execute it with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
C.append(nbf.v4.new_markdown_cell("""# 23 — Synthetic Historical Control on the monthly latent panel with a learned covariate mapping

Plan of 10 September 2026 (`quirky-growing-wombat`). Model y_t = g(x_t) + ℓ_t + δ_t d_t + ε_t: the covariate term
x_t'β of the paper (slide 6, steps 1–3) is replaced by g, a multi-output ridge regression of the 980-d latent on
land-cover class, elevation and slope, fitted **across sites at each period** on untreated rows only (every site at
P01–P10, control sites at P11–P21), leave-one-site-out so no site's own latent enters its own g. Everything after
that — kernel stage, blocks, stepwise selection, convex weights, counterfactual, effect — is `panel_shc.py`
unchanged, run on u = y − g by `panel_shc_run.py --covariate site`. The temporal validation (fit P01–P08,
predict P09–P10) and the effect are scored on the observed latent y against g + ℓ̂⁽⁰⁾, so every number has the
same definition and units as in notebook 22 / report §5–§7.

**Left = covariate-free run of 4 September (existing CSVs, not recomputed); right = the run with g.** S1 before S2.
Every number is read from the CSVs."""))
C.append(nbf.v4.new_code_cell("""import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
def load(suffix):
    F = pd.read_csv(f"panel_shc_fits{suffix}.csv"); E = pd.read_csv(f"panel_shc_effects{suffix}.csv")
    S = pd.read_csv(f"panel_shc_summary{suffix}.csv"); return F, E, S
F0, E0, S0 = load(""); F1, E1, S1 = load("_covariate"); G = pd.read_csv("panel_covariate.csv")
print(len(F0), "fits without g;", len(F1), "fits with g; same cells:", set(map(tuple, F0[["fill","sensor","site","n","m"]].values)) == set(map(tuple, F1[["fill","sensor","site","n","m"]].values)))
KEYS = ["fill", "sensor", "n", "m"]
def side(S_a, S_b, cols, label_a="without g", label_b="with g"):
    a = S_a.set_index(KEYS)[cols].add_suffix(f" | {label_a}"); b = S_b.set_index(KEYS)[cols].add_suffix(f" | {label_b}")
    out = a.join(b, how="outer")
    return out[[c for pair in zip(a.columns, b.columns) for c in pair]].reset_index()"""))
C.append(nbf.v4.new_markdown_cell("## 0. The mapping g: penalty, leave-one-site-out R² per channel and period, coefficients at P09–P12\n\nR² is the share of the cross-site variance of the latent at that period explained by the descriptors, leave-one-site-out, averaged over the 196 parcels of each channel. Coefficients are per feature (reference class 41; elevation and slope z-scored over the 286 sites), averaged over the parcels of each channel."))
C.append(nbf.v4.new_code_cell("""r2 = G[G.kind == "r2"]
print("penalty chosen per fill x sensor (leave-one-site-out squared error over P01-P10):")
print(r2.drop_duplicates(["fill","sensor"])[["fill","sensor","lam","lam_flag"]].to_string(index=False))
for sensor in ("sentinel1", "sentinel2"):
    print(f"\\n=== {sensor}: leave-one-site-out R^2 per channel by period ===")
    print(r2[r2.sensor == sensor][["fill","period","n_rows","n_leverage_one"] + [f"r2_ch{c}" for c in range(1,6)]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
cf = G[G.kind == "coef"]
for sensor in ("sentinel1", "sentinel2"):
    print(f"\\n=== {sensor}, chip-mean fill: coefficients per channel (mean over parcels) ===")
    print(cf[(cf.sensor == sensor) & (cf.fill == "chipmean")][["period","feature"] + [f"coef_ch{c}" for c in range(1,6)]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 1. Stage 1 — bandwidths chosen by leave-one-out CV (treated sites, effect window P01–P10)"))
C.append(nbf.v4.new_code_cell("""def bw(F):
    T = F[F.group == "treatment"]; hh = T[T.m == T.m.min()][["fill","sensor","site","h","h_flag"]].drop_duplicates(["fill","sensor","site"])
    return hh.groupby(["fill","sensor"]).agg(sites=("site","nunique"), h_median=("h","median"), h_min=("h","min"), h_max=("h","max"),
                                            interior=("h_flag", lambda v: int((v=="interior").sum())))
print(bw(F0).join(bw(F1), lsuffix=" | without g", rsuffix=" | with g").to_string())"""))
C.append(nbf.v4.new_markdown_cell("## 2. Temporal validation (fit on P01–P(10−n), predict the last n pre-hurricane months): RMSE over the 980 coordinates between the observed latent and the counterfactual, mean over treated sites\n\nReference = the site's own P01..P(10−n) mean of the observed latent (same in both runs)."))
C.append(nbf.v4.new_code_cell("""cols = ["placebo_rmse", "placebo_rmse_raw", "placebo_rmse_ownmean", "placebo_beats_ownmean", "most_recent_only", "n_eff_mean"]
for sensor in ("sentinel1", "sentinel2"):
    t = side(S0[S0.sensor == sensor], S1[S1.sensor == sensor], cols)
    print(f"\\n=== {sensor} ===")
    print(t.sort_values(["fill","n","m"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 3. Pre-period matching and donor selection (the series matched is y without g, u = y − g with g)"))
C.append(nbf.v4.new_code_cell("""cols = ["N", "mse_pre_mean", "passes_nu", "most_recent_only", "oldest_only", "n_selected_mean", "n_eff_mean"]
for sensor in ("sentinel1", "sentinel2"):
    t = side(S0[S0.sensor == sensor], S1[S1.sensor == sensor], cols)
    print(f"\\n=== {sensor} ===")
    print(t.sort_values(["fill","n","m"]).to_string(index=False, float_format=lambda v: f"{v:.5f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 4. Effect run: ‖δ̂‖/√980 per post period, treated sites against their matched control sites (m = 4)"))
C.append(nbf.v4.new_code_cell("""def eff(E):
    g = E[E.m == 4]
    return g.groupby(["sensor","fill","n","post_period","group"]).agg(sites=("site","nunique"), delta_rms=("delta_rms","mean"), delta_raw=("delta_raw_rms","mean"))
print(eff(E0).join(eff(E1), lsuffix=" | without g", rsuffix=" | with g").round(4).to_string())
cols = ["delta_rms_mean", "nc_sites", "nc_rank1", "nc_p_mean"]
for sensor in ("sentinel1", "sentinel2"):
    t = side(S0[(S0.sensor == sensor) & (S0.m == 4)], S1[(S1.sensor == sensor) & (S1.m == 4)], cols)
    print(f"\\n=== {sensor}, m = 4: treated departure and negative-control rank ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 5. Per treated site, n = 2, m = 4, historical fill: departure at P11 and P12, rank among the eleven, validation error (as report §7)"))
C.append(nbf.v4.new_code_cell("""def site_table(F, fill="histfill", n=2, m=4):
    T = F[(F.group=="treatment")&(F.fill==fill)&(F.n==n)&(F.m==m)&(F.ok==True)]
    Cc = F[(F.group=="counterfactual")&(F.fill==fill)&(F.n==n)&(F.m==m)&(F.ok==True)]
    out = []
    for _, r in T.iterrows():
        c = Cc[Cc.matched_treatment == r.site]
        out.append({"sensor": r.sensor, "site": r.site[-4:], "h": r.h, "P11": r.get("delta_rms_k1", np.nan), "P12": r.get("delta_rms_k2", np.nan),
                    "controls_median": float(c.delta_rms_mean.median()) if len(c) else np.nan,
                    "rank": 1 + int((c.delta_rms_mean >= r.delta_rms_mean).sum()) if len(c) else np.nan,
                    "validation": r.get("placebo_rmse", np.nan), "reference": r.get("placebo_rmse_ownmean", np.nan)})
    return pd.DataFrame(out).set_index(["sensor","site"])
print(site_table(F0).join(site_table(F1), lsuffix=" | without g", rsuffix=" | with g").round(3).to_string())"""))
C.append(nbf.v4.new_markdown_cell("## 6. Per-channel mean δ̂ (treated vs controls, m = 4, n = 2, historical fill)"))
C.append(nbf.v4.new_code_cell("""ch = [f"delta_ch{c}" for c in range(1,6)]
def chan(E):
    g = E[(E.m==4)&(E.n==2)&(E.fill=="histfill")]
    return g.groupby(["sensor","post_period","group"])[ch].mean()
print(chan(E0).join(chan(E1), lsuffix=" | without g", rsuffix=" | with g").round(4).to_string())"""))
nb["cells"] = C
nbf.write(nb, "23_shc_covariate.ipynb")
print("wrote 23_shc_covariate.ipynb")
