"""Build notebooks/ts_SCM_ASCM/24_shc_generator.ipynb: the latent SHC with stage 1 replaced by the learned generator
f(l_i, t) of panel_generator.py (plan 2026-09-11), in two arms (mlp, cnn), beside the kernel-stage run of 4 September.
Every table reads the CSVs: kernel = panel_shc_*.csv (not recomputed), mlp = *_mlp.csv, cnn = *_cnn.csv.
Run once to create the notebook, then execute it with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
C.append(nbf.v4.new_markdown_cell("""# 24 — Synthetic Historical Control on the monthly latent panel with stage 1 learned by a generator

Stage 1 of the paper (slide 6: covariate step and local-linear kernel smoothing of the site's own months, giving
ℓ̂_t for t ≤ T0) is replaced by a neural network f(l_i, t) trained across sites on the pre-period rows: l_i is a
learned 32-value code per site, t enters as a learned 8-value embedding per period, two hidden layers of 256, and
two output heads run as two arms — **mlp** (a fully connected layer to the 980 values) and **cnn** (a convolutional
decoder: 32 × 7 × 7 → transposed convolution to 32 × 14 × 14 → 3 × 3 convolution to 5 × 14 × 14). Loss: mean squared
error + 1e-3·mean‖l_i‖²; Adam 1e-3, weight decay 1e-4 on the weights; batch 64; up to 5000 epochs with the best
epoch kept and a stop after 50 epochs without improvement on 10 % of the site-months held out at random; seed 0.
One generator per fit: T0 = P10 (effect), P08 (validation n = 2), P07 (validation n = 3); a generator never sees a
period after its T0. Stage 2 (blocks, stepwise matching, weights, counterfactual) is `panel_shc.py` unchanged.

**kernel = the 4 September run (existing CSVs, not recomputed); mlp / cnn = the two generator arms.** Treated sites.
"""))
C.append(nbf.v4.new_code_cell("""import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
ARMS = {"kernel": "", "mlp": "_mlp", "cnn": "_cnn"}
F = {a: pd.read_csv(f"panel_shc_fits{s}.csv") for a, s in ARMS.items()}
E = {a: pd.read_csv(f"panel_shc_effects{s}.csv") for a, s in ARMS.items()}
S = {a: pd.read_csv(f"panel_shc_summary{s}.csv") for a, s in ARMS.items()}
G = {a: pd.read_csv(f"panel_generator{s}.csv") for a, s in ARMS.items() if a != "kernel"}
CV = {a: pd.read_csv(f"panel_generator_curves{s}.csv") for a, s in ARMS.items() if a != "kernel"}
for a in ARMS:
    F[a] = F[a][F[a].group == "treatment"]
print({a: len(F[a]) for a in ARMS}, "treated fits per arm")
KEYS = ["fill", "sensor", "n", "m"]
def side(cols, arms=("kernel", "mlp", "cnn"), sel=None):
    parts = []
    for a in arms:
        t = S[a] if sel is None else S[a][sel(S[a])]
        parts.append(t.set_index(KEYS)[cols].add_suffix(f" | {a}"))
    out = parts[0].join(parts[1:], how="outer")
    return out[[f"{c} | {a}" for c in cols for a in arms]].reset_index()"""))
C.append(nbf.v4.new_markdown_cell("## 0. The generator: what it learned (held-out 10 % of the site-months of each fit)\n\nR² is 1 − (held-out squared error) / (held-out variance about the held-out cells' own mean per coordinate), pooled over the coordinates of a channel; `n_filled` counts months whose composite was unusable and which the generator fills (the kernel left them NaN). The cnn head sits on a plateau at the data's mean for 250 to 1000 epochs before it learns; where the 50-epoch stopping rule fired inside that plateau the fit is marked **untrained** (held-out R² below 0.1) and the SHC numbers built on it are those of a generator that predicts each site's mean. The rule was kept as set on 11 September; the untrained fits are reported as such."))
C.append(nbf.v4.new_code_cell("""cols = ["fill", "sensor", "arm", "fit", "n_sites", "n_train", "n_held", "n_filled", "params", "epochs_run", "best_epoch", "stop",
        "train_mse", "held_mse", "r2_all", "status"] + [f"r2_ch{c}" for c in range(1, 6)] + ["code_norm_mean", "device", "seconds"]
Gall = pd.concat(G.values()); Gall["status"] = np.where(Gall.r2_all < 0.1, "UNTRAINED (stopped on the plateau)", "trained")
print(Gall.sort_values(["fill", "sensor", "arm", "T0"], ascending=[True, True, True, False])[cols]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
UNTRAINED = {(r.fill, r.sensor, r.arm, int(r.T0)) for r in Gall.itertuples() if r.r2_all < 0.1}
print("untrained fits:", sorted(UNTRAINED))"""))
C.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey="row")
for r, sensor in enumerate(("sentinel1", "sentinel2")):
    for c, T0 in enumerate((10, 8, 7)):
        ax = axes[r, c]
        for a in ("mlp", "cnn"):
            cv = CV[a][(CV[a].sensor == sensor) & (CV[a].fill == "histfill") & (CV[a].T0 == T0)]
            ax.plot(cv.epoch, cv.held_mse, label=f"{a} held-out"); ax.plot(cv.epoch, cv.train_mse, ":", label=f"{a} training")
        ax.set_title(f"{sensor}, historical fill, T0 = P{T0:02d}"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_xscale("log")
        if r == 0 and c == 0: ax.legend()
plt.tight_layout(); plt.savefig("panel_generator_curves.png", dpi=110); plt.show()"""))
C.append(nbf.v4.new_markdown_cell("## 1. Which block is chosen, and the pre-period matching MSE of eq 31 of the selected block (treated sites, T0 = P10)\n\n`most_recent_only` = treated fits where block 1 is selected alone."))
C.append(nbf.v4.new_code_cell("""cols = ["most_recent_only", "n_selected_mean", "mse_pre_mean", "n_eff_mean"]
for sensor in ("sentinel1", "sentinel2"):
    print(f"\\n=== {sensor} ===")
    print(side(cols, sel=lambda t, s=sensor: t.sensor == s).sort_values(["fill", "n", "m"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 2. Prediction error (temporal validation: fit on P01–P(10 − n), predict the last n pre-hurricane months; RMSE over the 980 coordinates between the observed latent and the synthetic counterfactual, mean over the held-out months and the treated sites)\n\nOwn mean = the site's own P01–P(10 − n) mean used as the prediction (the same in every arm)."))
C.append(nbf.v4.new_code_cell("""cols = ["placebo_rmse", "placebo_rmse_raw", "placebo_beats_ownmean"]
for sensor in ("sentinel1", "sentinel2"):
    t = side(cols, sel=lambda t, s=sensor: t.sensor == s)
    own = S["kernel"][S["kernel"].sensor == sensor].set_index(KEYS)["placebo_rmse_ownmean"]
    t = t.set_index(KEYS).join(own).reset_index()
    t["cnn generator"] = ["UNTRAINED" if (r.fill, sensor, "cnn", 10 - int(r.n)) in UNTRAINED else "trained" for r in t.itertuples()]
    print(f"\\n=== {sensor} ===  (the validation at n uses the generator with T0 = 10 - n)")
    print(t.sort_values(["fill", "n", "m"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 3. Prediction error per held-out month (m = 4, historical fill, mean over treated sites)"))
C.append(nbf.v4.new_code_cell("""rows = []
for a in ARMS:
    T = F[a][(F[a].fill == "histfill") & (F[a].m == 4) & (F[a].placebo_ok == True)]
    for (sensor, n), g in T.groupby(["sensor", "n"]):
        r = {"sensor": sensor, "n": n, "arm": a}
        for q in (8, 9, 10):
            c = f"placebo_rmse_P{q:02d}"
            if c in g: r[f"P{q:02d}"] = g[c].mean()
        r["mean"] = g.placebo_rmse.mean(); r["own mean"] = g.placebo_rmse_ownmean.mean()
        rows.append(r)
print(pd.DataFrame(rows).sort_values(["sensor", "n", "arm"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))"""))
C.append(nbf.v4.new_markdown_cell("## 4. Per treated site, n = 2, m = 4, historical fill: prediction error and own mean, kernel | mlp | cnn"))
C.append(nbf.v4.new_code_cell("""def per_site(a):
    T = F[a][(F[a].fill == "histfill") & (F[a].n == 2) & (F[a].m == 4) & (F[a].placebo_ok == True)]
    return T.set_index(["sensor", "site"])[["placebo_rmse", "placebo_rmse_ownmean", "selected"]].rename(
        columns={"placebo_rmse": f"prediction error | {a}", "placebo_rmse_ownmean": "own mean", "selected": f"selected | {a}"})
t = per_site("kernel").join(per_site("mlp").drop(columns="own mean")).join(per_site("cnn").drop(columns="own mean"))
print(t.round(3).to_string())"""))
C.append(nbf.v4.new_markdown_cell("## 5. λ = observed − counterfactual at the post-hurricane months (treated sites only): ‖λ‖/√980 per month, mean over sites, and per channel (m = 4)"))
C.append(nbf.v4.new_code_cell("""def eff(a):
    g = E[a][(E[a].m == 4) & (E[a].group == "treatment")]
    return g.groupby(["sensor", "fill", "n", "post_period"]).agg(sites=("site", "nunique"), lam_rms=("delta_rms", "mean"), lam_raw=("delta_raw_rms", "mean"))
t = eff("kernel").join(eff("mlp"), rsuffix=" | mlp").join(eff("cnn"), rsuffix=" | cnn")
print(t.round(4).to_string())
ch = [f"delta_ch{c}" for c in range(1, 6)]
rows = []
for a in ARMS:
    g = E[a][(E[a].m == 4) & (E[a].n == 2) & (E[a].fill == "histfill") & (E[a].group == "treatment")]
    for (sensor, q), gg in g.groupby(["sensor", "post_period"]):
        rows.append({"sensor": sensor, "post month": q, "arm": a, **{c: gg[c].mean() for c in ch}})
print("\\nper-channel mean of lambda:")
print(pd.DataFrame(rows).sort_values(["sensor", "post month", "arm"]).round(4).to_string(index=False))"""))
C.append(nbf.v4.new_markdown_cell("## 6. Per treated site, n = 2, m = 4, historical fill: ‖λ‖/√980 at P11 and P12, kernel | mlp | cnn"))
C.append(nbf.v4.new_code_cell("""def lam_site(a):
    T = F[a][(F[a].fill == "histfill") & (F[a].n == 2) & (F[a].m == 4) & (F[a].ok == True)]
    return T.set_index(["sensor", "site"])[["delta_rms_k1", "delta_rms_k2"]].rename(columns={"delta_rms_k1": f"P11 | {a}", "delta_rms_k2": f"P12 | {a}"})
print(lam_site("kernel").join(lam_site("mlp")).join(lam_site("cnn")).round(3).to_string())"""))
nb["cells"] = C
nbf.write(nb, "24_shc_generator.ipynb")
print("wrote 24_shc_generator.ipynb")
