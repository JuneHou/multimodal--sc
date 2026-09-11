import sys, os
sys.path.insert(0, "/data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/ts_SCM_ASCM")
os.chdir("/data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/ts_SCM_ASCM")
import numpy as np, pandas as pd
import panel_lib as pl, panel_repr as pr, panel_align as pa
pd.set_option("display.width", 200)

_z = np.load("parcel_perms.npz"); PERMS = {tuple(k.split("|")): _z[k] for k in _z.files}
TRAIN, TEST = list(range(1, 10)), 10
CACHES = {"chipmean": "latents_biweekly.npz", "histfill": "latents_biweekly_histfill.npz"}

def run(cache_name, sensor):
    panel = pl.Panel.from_npz(pl.LATD / CACHES[cache_name])
    ALL = sorted(panel.roster["site_id"]); TREAT = panel.treatments; DON = pr.matched_donors(panel)

    def vec(s, q, perm=None, block=False):
        if perm is None:
            v = panel.L(s, sensor, q); v = np.full(980, np.nan) if v is None else v
        else:
            v = pa.aligned_vec(panel, s, sensor, q, perm)
        return pa.block_mean(v) if block else v

    def scaler(block):
        T = np.stack([vec(s, q, block=block) for s in ALL for q in TRAIN
                      if panel.L(s, sensor, q) is not None])
        n = 49 if block else 196
        V = T.reshape(len(T), 5, n).transpose(1, 0, 2).reshape(5, -1)
        mu = np.repeat(np.nanmean(V, 1), n); sd = np.repeat(np.nanstd(V, 1, ddof=1), n)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        return lambda v: (v - mu) / sd

    out = []
    for block in (False, True):
        z = scaler(block)
        for arm in ("A_samecoord", "D_min_aligned", "H0_own_mean"):
            errs = []
            for t in TREAT:
                yte = z(vec(t, TEST, block=block))
                if arm in ("A_samecoord", "D_min_aligned"):
                    dl, P = [], {}
                    for j in DON[t]:
                        if arm == "A_samecoord": P[j] = None; dl.append(j)
                        else:
                            p = PERMS[("D_min", sensor, t, j)]
                            if (p >= 0).sum() >= pa.MIN_MATCHED: P[j] = p; dl.append(j)
                    ytr = np.concatenate([z(vec(t, q, block=block)) for q in TRAIN])
                    Xtr = np.column_stack([np.concatenate(
                        [z(vec(j, q, P[j], block)) for q in TRAIN]) for j in dl])
                    w = pa.simplex_scm(ytr, Xtr)
                    Xte = np.column_stack([z(vec(j, TEST, P[j], block)) for j in dl])
                    pred = Xte @ w
                else:
                    pred = np.nanmean(np.stack([z(vec(t, q, block=block)) for q in TRAIN]), 0)
                errs.append(pa.rmse(yte - pred))
            out.append({"cache": cache_name, "sensor": sensor,
                        "outcome": "245-d (2x2 block)" if block else "980-d",
                        "arm": arm, "test_rmse": float(np.nanmean(errs))})
    return out

rows = []
for sensor in ("sentinel1", "sentinel2"):
    rows += run("chipmean", sensor)
rows += run("histfill", "sentinel2")
df = pd.DataFrame(rows)
piv = df.pivot_table(index=["cache","sensor","arm"], columns="outcome", values="test_rmse")
piv["ratio_245/980"] = (piv["245-d (2x2 block)"] / piv["980-d"]).round(3)
print("\n=== same scorer, same sites, P10 — 980-d vs 245-d block-mean outcome ===")
print(piv.round(3).to_string())
df.to_csv("panel_block_outcome_compare.csv", index=False)
