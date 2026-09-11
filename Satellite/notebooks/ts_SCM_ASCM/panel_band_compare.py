import sys, os
sys.path.insert(0, "/data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/ts_SCM_ASCM")
os.chdir("/data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/ts_SCM_ASCM")
import numpy as np, pandas as pd
import panel_lib as pl, panel_repr as pr, panel_align as pa
pd.set_option("display.width", 250)

BANDS = ["B2","B3","B4","B8","B11","B12","NDVI","NDWI"]
panel = pl.Panel.from_npz(pl.LATD / "latents_biweekly.npz")
ALL = sorted(panel.roster["site_id"]); TREAT = panel.treatments; DON = pr.matched_donors(panel)
idx = pl.build_panel_index()

# ---- band means, exactly her definition: spatial mean over finite pixels, per band ----
BF = {}
rows = idx.loc[idx.file_exists & (idx.sensor == "sentinel2")]
for r in rows.itertuples():
    if r.seq > 10: continue
    chip = pl.read_chip_biweekly(r.tif, "sentinel2")          # (101,101,8)
    with np.errstate(all="ignore"):
        BF[(r.site_id, r.seq)] = np.array([np.nanmean(chip[:,:,k]) for k in range(8)])
for s in ALL:
    for q in range(1, 11):
        BF.setdefault((s, q), np.full(8, np.nan))

print("sanity, pooled mean per band over 60 sites P01-P10:")
M = np.stack([BF[(s,q)] for s in ALL for q in range(1,11)])
for k,b in enumerate(BANDS):
    print(f"  {b:5s} mean {np.nanmean(M[:,k]):8.4f}  sd {np.nanstd(M[:,k]):7.4f}")

def scaler(train_p):
    T = np.stack([BF[(s,q)] for s in ALL for q in train_p])
    mu = np.nanmean(T,0); sd = np.nanstd(T,0,ddof=1)
    sd[~np.isfinite(sd)|(sd==0)] = 1.0
    return mu, sd

def fit(train_p, joint, demean, test_q=10):
    """returns (raw errors (10,8), sd used (8,))"""
    mu, sd = scaler(train_p)
    E = []
    for t in TREAT:
        dl = DON[t]
        Y  = np.stack([BF[(t,q)] for q in train_p])                    # (T,8)
        Xd = [np.stack([BF[(d,q)] for q in train_p]) for d in dl]
        yte = BF[(t,test_q)]; xte = np.column_stack([BF[(d,test_q)] for d in dl])  # (8,J)
        mt = np.nanmean(Y,0); md = np.column_stack([np.nanmean(X,0) for X in Xd])
        Yf  = Y - mt if demean else Y
        Xf  = [X - md[:,j] for j,X in enumerate(Xd)] if demean else Xd
        if joint:
            w = pa.simplex_scm(Yf.ravel(), np.column_stack([X.ravel() for X in Xf]))
            W = np.tile(w,(8,1)).T
        else:
            W = np.column_stack([pa.simplex_scm(Yf[:,k], np.column_stack([X[:,k] for X in Xf]))
                                 for k in range(8)])
        pred = (mt if demean else 0) + np.array(
            [np.nansum(W[:,k]*(xte[k]-(md[k] if demean else 0))) for k in range(8)])
        E.append(yte - pred)
    return np.array(E), sd

def baseline(kind, train_p, test_q=10):
    mu, sd = scaler(train_p); E = []
    for t in TREAT:
        yte = BF[(t,test_q)]
        if kind == "equal":
            pred = np.nanmean(np.column_stack([BF[(d,test_q)] for d in DON[t]]),1)
        else:
            pred = np.nanmean(np.stack([BF[(t,q)] for q in train_p]),0)
        E.append(yte - pred)
    return np.array(E), sd

def rep(E, sd):
    raw = float(np.sqrt(np.nanmean(E**2)))
    std = float(np.sqrt(np.nanmean((E/sd)**2)))
    return raw, std

P19 = list(range(1,10))
arms = [
 ("baseline: equal-weight donor avg",      baseline("equal", P19)),
 ("baseline: own-history mean (P01-P09)",  baseline("own",   P19)),
 ("standard SCM, per band, T0=9",          fit(P19, False, False)),
 ("multi-outcome SC, joint, T0=9",         fit(P19, True,  False)),
 ("standard SCM per band + demeaning",     fit(P19, False, True)),
 ("multi-outcome SC joint + demeaning (her Exp 2)", fit(P19, True, True)),
 ("multi-outcome SC joint, T0=1 P09 only (her Exp 1)", fit([9], True, False)),
]
print("\n=== Sentinel-2, 8 band means, test P10, 10 treated sites ===")
print(f"{'arm':52s} {'RAW rmse':>10s} {'STD rmse':>10s}")
for name,(E,sd) in arms:
    r,s = rep(E,sd); print(f"{name:52s} {r:10.4f} {s:10.3f}")

E,sd = arms[5][1]
print("\nper-band, her Exp-2 analog (joint + demeaning), test P10:")
print(f"{'band':6s} {'RAW':>9s} {'pooled SD':>10s} {'STD':>8s}")
for k,b in enumerate(BANDS):
    print(f"{b:6s} {np.sqrt(np.nanmean(E[:,k]**2)):9.4f} {sd[k]:10.4f} "
          f"{np.sqrt(np.nanmean((E[:,k]/sd[k])**2)):8.3f}")
print(f"\nmean pooled SD across 8 bands = {sd.mean():.4f}  "
      f"(raw->standardized divides by roughly this)")
