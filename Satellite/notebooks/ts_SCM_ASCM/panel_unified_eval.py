"""One evaluation for every estimator, in every representation, at P10.

Rule (single, applied identically everywhere):
    per site i:   rmse_i = sqrt( mean over dims of (yhat - y)^2 )
    reported   :  mean over the 10 treated sites
Train window P01-P09, test P10, 5 matched donors per treated site.

Two columns per cell:
  raw  -- in the representation's own units. No scaler, so no denominator choices
          and no scaling artifacts. This is the column FSC reported (M3).
  std  -- each dimension divided by its pooled train SD over all 60 sites (M2),
          the scaler MATCHED to the representation being scored.
Comparisons are valid DOWN a column within one representation. Across
representations the outcome differs, so only same-dimension rows compare.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import panel_lib as pl, panel_repr as pr, panel_align as pa
pd.set_option("display.width", 240)

TRAIN, TEST = list(range(1, 10)), 10
_z = np.load("parcel_perms.npz"); PERMS = {tuple(k.split("|")): _z[k] for k in _z.files}
REPRS = ["bands", "chip_mean", "gram", "quantile", "combined", "block245", "latent980"]
_IDX = None


def band_features(panel, sensor, ALL):
    global _IDX
    if _IDX is None: _IDX = pl.build_panel_index()
    nb = len(pl.BANDS[sensor]); F = {}
    for r in _IDX.loc[_IDX.file_exists & (_IDX.sensor == sensor)].itertuples():
        if r.seq > 10: continue
        c = pl.read_chip_biweekly(r.tif, sensor)
        with np.errstate(all="ignore"):
            F[(r.site_id, r.seq)] = np.array([np.nanmean(c[:, :, k]) for k in range(nb)])
    for s_ in ALL:
        for q in range(1, 11): F.setdefault((s_, q), np.full(nb, np.nan))
    return F


def features(panel, sensor, name, ALL):
    if name == "bands":
        return band_features(panel, sensor, ALL)
    F = {}
    for s in ALL:
        for q in range(1, 11):
            v = panel.L(s, sensor, q)
            if v is None:
                F[(s, q)] = None; continue
            if name == "latent980":   F[(s, q)] = v
            elif name == "block245":  F[(s, q)] = pa.block_mean(v)
            else:                     F[(s, q)] = pr._repr_raw(pr._A(panel, s, sensor, q), name)
    M = len(next(v for v in F.values() if v is not None))
    return {k: (np.full(M, np.nan) if v is None else v) for k, v in F.items()}


def aligned(panel, sensor, name, t, j, q):
    p = PERMS[("D_min", sensor, t, j)]
    if (p >= 0).sum() < pa.MIN_MATCHED: return None
    v = pa.aligned_vec(panel, j, sensor, q, p)
    return pa.block_mean(v) if name == "block245" else v


def run(cache, sensor):
    panel = pl.Panel.from_npz(pl.LATD / cache)
    ALL = sorted(panel.roster["site_id"]); TREAT = panel.treatments
    DON = pr.matched_donors(panel)
    rows = []
    for name in REPRS:
        F = features(panel, sensor, name, ALL)
        T = np.stack([F[(s, q)] for s in ALL for q in TRAIN])
        sd = np.nanstd(T, 0, ddof=1); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        for arm in ("equal_weight", "own_history_mean", "scm_crosssec", "scm_crosssec_demeaned",
                    "scm_aligned"):
            if arm == "scm_aligned" and name not in ("latent980", "block245"):
                continue
            raw, std = [], []
            for t in TREAT:
                y = F[(t, TEST)]
                if arm in ("scm_aligned",):
                    dl = [j for j in DON[t] if aligned(panel, sensor, name, t, j, TEST) is not None]
                    g = lambda j, q: aligned(panel, sensor, name, t, j, q)
                else:
                    dl = list(DON[t]); g = lambda j, q: F[(j, q)]
                if arm == "equal_weight":
                    pred = np.nanmean(np.column_stack([g(j, TEST) for j in dl]), 1)
                elif arm == "own_history_mean":
                    pred = np.nanmean(np.stack([F[(t, q)] for q in TRAIN]), 0)
                else:
                    dm = arm.endswith("demeaned")
                    Y = np.stack([F[(t, q)] for q in TRAIN])
                    Xd = [np.stack([g(j, q) for q in TRAIN]) for j in dl]
                    mt = np.nanmean(Y, 0); md = np.column_stack([np.nanmean(X, 0) for X in Xd])
                    Yf = Y - mt if dm else Y
                    Xf = [X - md[:, k] for k, X in enumerate(Xd)] if dm else Xd
                    w = pa.simplex_scm(Yf.ravel(), np.column_stack([X.ravel() for X in Xf]))
                    xte = np.column_stack([g(j, TEST) for j in dl])
                    pred = (mt + (xte - md) @ w) if dm else xte @ w
                raw.append(pa.rmse(y - pred)); std.append(pa.rmse((y - pred) / sd))
            rows.append({"cache": cache, "sensor": sensor, "repr": name, "M": len(sd),
                         "estimator": arm, "raw": float(np.nanmean(raw)),
                         "std": float(np.nanmean(std))})
    return rows


rows = []
for sensor in ("sentinel1", "sentinel2"):
    rows += run("latents_biweekly.npz", sensor)
df = pd.DataFrame(rows)

# --- FSC, same window and test period, from its stored per-site predictions ---
fsc = pd.read_csv("panel_fsc_prediction.csv").query(
    "label == 'chipmean' and fit_window == 'P01-09' and eval_period == 'P10' and estimator == 'fsc'")
fr = (fsc.groupby(["sensor", "repr"]).pred_rmse.mean().reset_index()
      .rename(columns={"repr": "repr_", "pred_rmse": "raw"}))
for r in fr.itertuples():
    m = df[(df.sensor == r.sensor) & (df["repr"] == r.repr_)]
    rows.append({"cache": "latents_biweekly.npz", "sensor": r.sensor, "repr": r.repr_,
                 "M": int(m.M.iloc[0]) if len(m) else np.nan,
                 "estimator": "fsc_okano_kurisu", "raw": float(r.raw), "std": np.nan})
df = pd.DataFrame(rows)
df.to_csv("panel_unified_eval_p10.csv", index=False)

for sensor in ("sentinel1", "sentinel2"):
    print(f"\n================ {sensor} — test P10, train P01-P09 ================")
    for name in REPRS:
        d = df[(df.sensor == sensor) & (df["repr"] == name)]
        if not len(d): continue
        print(f"\n-- {name} (M = {int(d.M.iloc[0])} dims) --")
        print(d[["estimator", "raw", "std"]].sort_values("raw")
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
