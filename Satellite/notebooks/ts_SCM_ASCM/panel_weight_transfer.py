import sys, os
sys.path.insert(0, "/data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/ts_SCM_ASCM")
os.chdir("/data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/ts_SCM_ASCM")
import numpy as np, pandas as pd
import panel_lib as pl, panel_repr as pr, panel_align as pa
pd.set_option("display.width", 240)

panel = pl.Panel.from_npz(pl.LATD / "latents_biweekly.npz")
ALL = sorted(panel.roster["site_id"]); TREAT = panel.treatments; DON = pr.matched_donors(panel)
idx = pl.build_panel_index()
_z = np.load("parcel_perms.npz"); PERMS = {tuple(k.split("|")): _z[k] for k in _z.files}
TRAIN = list(range(1, 10)); TEST = 10
BANDS = {"sentinel2": ["B2","B3","B4","B8","B11","B12","NDVI","NDWI"],
         "sentinel1": ["VV","VH","VV_minus_VH"]}

def band_feats(sensor):
    nb = len(BANDS[sensor]); BF = {}
    for r in idx.loc[idx.file_exists & (idx.sensor == sensor)].itertuples():
        if r.seq > 10: continue
        c = pl.read_chip_biweekly(r.tif, sensor)
        with np.errstate(all="ignore"):
            BF[(r.site_id, r.seq)] = np.array([np.nanmean(c[:,:,k]) for k in range(nb)])
    for s in ALL:
        for q in range(1, 11): BF.setdefault((s,q), np.full(nb, np.nan))
    return BF

def repr_feats(sensor, name):
    F = {}
    for s in ALL:
        for q in range(1, 11):
            v = panel.L(s, sensor, q)
            if v is None:
                F[(s,q)] = None; continue
            F[(s,q)] = v if name == "lat980" else pr._repr_raw(pr._A(panel,s,sensor,q), name)
    return F

def donor_vec(F, sensor, t, j, q, aligned):
    if not aligned: return F[(j,q)]
    p = PERMS[("D_min", sensor, t, j)]
    if (p >= 0).sum() < pa.MIN_MATCHED: return None
    return pa.aligned_vec(panel, j, sensor, q, p)

def weights(F, sensor, t, demean, aligned=False):
    """simplex weights fitted in representation F; returns (donor list, w)."""
    dl, cols = [], []
    for j in DON[t]:
        v = [donor_vec(F, sensor, t, j, q, aligned) for q in TRAIN]
        if any(x is None for x in v): continue
        dl.append(j); cols.append(np.stack(v))
    y = np.stack([F[(t,q)] if F[(t,q)] is not None else np.full_like(cols[0][0], np.nan)
                  for q in TRAIN])
    if demean:
        y = y - np.nanmean(y,0); cols = [X - np.nanmean(X,0) for X in cols]
    return dl, pa.simplex_scm(y.ravel(), np.column_stack([X.ravel() for X in cols]))

def score_on_bands(BF, dl, w, t, demean):
    """apply weights to DONOR BAND MEANS; return raw error vector at P10."""
    yte = BF[(t,TEST)]
    xte = np.column_stack([BF[(j,TEST)] for j in dl])
    if demean:
        mt = np.nanmean(np.stack([BF[(t,q)] for q in TRAIN]),0)
        md = np.column_stack([np.nanmean(np.stack([BF[(j,q)] for q in TRAIN]),0) for j in dl])
        pred = mt + (xte - md) @ w
    else:
        pred = xte @ w
    return yte - pred

rows = []
for sensor in ("sentinel1","sentinel2"):
    BF = band_feats(sensor)
    T = np.stack([BF[(s,q)] for s in ALL for q in TRAIN])
    sd = np.nanstd(T,0,ddof=1); sd[~np.isfinite(sd)|(sd==0)] = 1.0
    REPRS = {"bands (her outcome)": (BF, False),
             "TerraMind 5ch (chip mean)": (repr_feats(sensor,"chip_mean"), False),
             "quantile (distributional, 100-d)": (repr_feats(sensor,"quantile"), False),
             "gram (15-d)": (repr_feats(sensor,"gram"), False),
             "combined quantile+gram (115-d)": (repr_feats(sensor,"combined"), False),
             "latent980": (repr_feats(sensor,"lat980"), False),
             "latent980 aligned": (repr_feats(sensor,"lat980"), True)}
    for rname,(F,al) in REPRS.items():
        for demean in (False, True):
            E = []
            for t in TREAT:
                dl, w = weights(F, sensor, t, demean, al)
                E.append(score_on_bands(BF, dl, w, t, demean))
            E = np.array(E)
            rows.append({"sensor": sensor, "weights_from": rname, "demeaned": demean,
                         "raw_rmse": float(np.sqrt(np.nanmean(E**2))),
                         "std_rmse": float(np.sqrt(np.nanmean((E/sd)**2)))})
    # equal-weight baseline on the same metric
    E = np.array([BF[(t,TEST)] - np.nanmean(np.column_stack([BF[(j,TEST)] for j in DON[t]]),1) for t in TREAT])
    rows.append({"sensor": sensor, "weights_from": "equal weight (baseline)", "demeaned": False,
                 "raw_rmse": float(np.sqrt(np.nanmean(E**2))), "std_rmse": float(np.sqrt(np.nanmean((E/sd)**2)))})

df = pd.DataFrame(rows)
print("\n=== ALL representations scored on the SAME target: donor band means at P10 ===")
for sensor in ("sentinel1","sentinel2"):
    print(f"\n--- {sensor} ({len(BANDS[sensor])} bands) ---")
    print(df[df.sensor==sensor][["weights_from","demeaned","raw_rmse","std_rmse"]]
          .sort_values("raw_rmse").to_string(index=False))
df.to_csv("panel_weight_transfer_p10.csv", index=False)
