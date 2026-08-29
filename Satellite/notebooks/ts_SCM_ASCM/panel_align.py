"""panel_align — parcel descriptors, class-constrained Hungarian alignment of donor
parcels to treated positions, and synthetic-historical-control (SHC) blocks.

Design (approved 2026-08-28, plan `quirky-growing-wombat`):
  * Experiment 1: the collaborator's 5 donors per treated site; each donor's 196 latent
    parcels are PERMUTED so that a donor parcel only stands in for a treated position of
    the same NLCD land class (hard constraint); within a class the Hungarian assignment
    minimises a descriptor distance (elevation/slope [+ aspect], TerraMind pre-treatment
    history mean [+ SD], [+ 3x3 class-context histogram]). Descriptors decide the
    permutation only — the outcome vector stays the 980-d latent (5 ch x 196 parcels).
  * Experiment 2: the treated site's own history as donor (SHC blocks, Chen-Yang-Yang,
    SSRN 4995085). No alignment: same parcel, same class, same position by construction.

Parcel geometry: latent parcel (i, j) is the 16x16 patch (i, j) of the 224 tokenizer input,
i.e. cell (i, j) of the uniform 14x14 partition `_EDGES` of the 101-px chip
(`panel_repr.py`, parcel-validity section). Row-major (i, j) -> p = 14*i + j, the same
C-order as `latent.reshape(5, 196)`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.optimize import linear_sum_assignment, minimize

from panel_lib import BIW, ROOT, SENSORS, TRAIN8
from panel_repr import _EDGES, PARCEL_THR, NPARCEL

# ---------------------------------------------------------------- static rasters
BND = ROOT / "Satellite" / "data" / "boundaries"
NLCD_TIF = BND / "landcover" / "processed" / "southwest_virginia_nlcd_2021_utm17n.tif"
DEM_TIF = BND / "dem" / "processed" / "southwest_virginia_dem_utm17n.tif"
SLOPE_TIF = BND / "dem" / "processed" / "southwest_virginia_slope_degrees.tif"

NLCD_NAMES = {0: "nodata", 11: "water", 21: "dev-open", 22: "dev-low", 23: "dev-med",
              24: "dev-high", 31: "barren", 41: "deciduous", 42: "evergreen",
              43: "mixed", 52: "shrub", 71: "grass", 81: "pasture", 82: "crop",
              90: "wood-wet", 95: "herb-wet"}
NLCD_CLASSES = [k for k in NLCD_NAMES if k != 0]          # 15 classes, context hist order
MERGE_FOREST_DEV = {42: 41, 43: 41, 22: 21, 23: 21, 24: 21}   # E1.4 alternative

MIN_CLEAN = 3          # E1.7: parcel needs >= this many clean P01-P08 periods
MIN_MATCHED = 60       # E1.11: donor dropped for a site below this many matched parcels
BIG = 1e6              # "infinite" cost for a class mismatch / nodata parcel
GRID = 14


def _mode(a):
    a = a.ravel()
    return int(np.bincount(a).argmax()) if a.size else 0


def site_bounds(panel, site, sensor="sentinel1"):
    """Bounds of the chip raster (any existing period) — the ground footprint that the
    14x14 parcels partition uniformly."""
    for (s, se, pid), ok in panel.fexists.items():
        if s == site and se == sensor and ok:
            m = panel.meta[(s, se, pid)]
            tif = m.get("tif") or str(BIW / se / m["group"] / m["half"] /
                                      f"{s}_{pid}_{m['period_start']}_{m['period_end']}"
                                      f"_biweekly_{se}.tif")
            with rasterio.open(tif) as r:
                return r.bounds, r.crs
    raise KeyError(site)


def chip_rasters(bounds, crs):
    """NLCD class (nearest), elevation and slope (bilinear) resampled onto the chip's
    101x101 10 m grid, so `_EDGES` windows apply exactly as for the pixel validity map."""
    out = {}
    for key, tif, rs in (("cls", NLCD_TIF, Resampling.nearest),
                         ("elev", DEM_TIF, Resampling.bilinear),
                         ("slope", SLOPE_TIF, Resampling.bilinear)):
        with rasterio.open(tif) as r:
            assert r.crs == crs, (tif, r.crs, crs)
            w = from_bounds(*bounds, transform=r.transform)
            out[key] = r.read(1, window=w, out_shape=(101, 101), resampling=rs,
                              boundless=True, fill_value=0)
    return out


def parcel_static(rasters, merge=None):
    """Per-parcel class (mode), elevation, slope, aspect (sin, cos) and the 3x3-neighbour
    class histogram (15 NLCD classes, row-normalised)."""
    cls101 = rasters["cls"].astype(np.int64)
    if merge:
        cls101 = np.vectorize(lambda c: merge.get(c, c))(cls101)
    elev = rasters["elev"].astype(np.float64)
    gy, gx = np.gradient(elev, 10.0)                  # 10 m pixels
    aspect = np.arctan2(-gx, gy)                      # direction the slope faces
    cls = np.zeros(NPARCEL, np.int64)
    phys = np.zeros((NPARCEL, 2)); asp = np.zeros((NPARCEL, 2))
    for i in range(GRID):
        for j in range(GRID):
            sl = (slice(_EDGES[i], _EDGES[i + 1]), slice(_EDGES[j], _EDGES[j + 1]))
            p = GRID * i + j
            cls[p] = _mode(cls101[sl])
            phys[p] = (elev[sl].mean(), rasters["slope"][sl].mean())
            a = aspect[sl]
            asp[p] = (np.sin(a).mean(), np.cos(a).mean())
    cg = cls.reshape(GRID, GRID)
    ctx = np.zeros((NPARCEL, len(NLCD_CLASSES)))
    col = {c: k for k, c in enumerate(NLCD_CLASSES)}
    for i in range(GRID):
        for j in range(GRID):
            nb = cg[max(0, i - 1):i + 2, max(0, j - 1):j + 2].ravel()
            for c in nb:
                if c in col:
                    ctx[GRID * i + j, col[c]] += 1
            ctx[GRID * i + j] /= max(1, len(nb))
    return {"cls": cls, "phys": phys, "aspect": asp, "ctx": ctx}


def parcel_history(panel, validity, site, sensor, periods=TRAIN8, thr=PARCEL_THR):
    """Mean and SD over CLEAN pre-treatment periods of each parcel's 5-ch latent, plus the
    clean-period count. A period is clean for parcel p if its finite fraction >= thr."""
    vals, oks = [], []
    for q in periods:
        v = panel.L(site, sensor, q)
        pv = validity.get((site, sensor, panel.pid_of_seq[q]))
        if v is None or pv is None:
            continue
        vals.append(v.reshape(5, NPARCEL)); oks.append(pv >= thr)
    if not vals:
        return {"hmean": np.full((NPARCEL, 5), np.nan),
                "hsd": np.full((NPARCEL, 5), np.nan), "n_clean": np.zeros(NPARCEL, int)}
    V = np.stack(vals)                          # (T, 5, 196)
    M = np.stack(oks)[:, None, :]               # (T, 1, 196)
    n = M.sum(axis=0)[0]                        # (196,)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(M, V, 0).sum(0) / np.maximum(n, 1)
        var = (np.where(M, (V - mean[None]) ** 2, 0).sum(0) / np.maximum(n - 1, 1))
    mean[:, n == 0] = np.nan; var[:, n < 2] = np.nan
    return {"hmean": mean.T, "hsd": np.sqrt(var).T, "n_clean": n}


def build_descriptors(panel, validity, sites=None, merge=None, thr=PARCEL_THR,
                      periods=TRAIN8):
    """{(site, sensor): dict of per-parcel arrays}. Static blocks are shared by both
    sensors; the TerraMind-history block is per sensor. Numeric blocks are standardised
    later (`standardize`), pooled over all sites x parcels."""
    sites = sites or sorted(panel.roster["site_id"])
    desc = {}
    for s in sites:
        b, crs = site_bounds(panel, s)
        st = parcel_static(chip_rasters(b, crs), merge=merge)
        for sensor in SENSORS:
            d = dict(st); d.update(parcel_history(panel, validity, s, sensor, periods, thr))
            desc[(s, sensor)] = d
    return desc


def save_descriptors(desc, path):
    np.savez_compressed(path, **{f"{s}|{se}|{k}": v for (s, se), d in desc.items()
                                 for k, v in d.items()})


def load_descriptors(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        s, se, name = k.split("|")
        out.setdefault((s, se), {})[name] = z[k]
    return out


NUMERIC = ("phys", "aspect", "hmean", "hsd", "ctx")


def standardize(desc):
    """Pooled per-dimension mean/SD (ddof=1) over all (site, sensor, parcel) rows of each
    numeric block; returns a new dict with standardised copies (NaN kept)."""
    out = {k: dict(v) for k, v in desc.items()}
    for blk in NUMERIC:
        X = np.concatenate([d[blk] for d in desc.values()])
        mu = np.nanmean(X, 0); sd = np.nanstd(X, 0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        for k in out:
            out[k][blk] = (desc[k][blk] - mu) / sd
    return out


def _blockdist(a, b):
    """mean squared standardised difference across the block's dims: (196_t, 196_j)."""
    d = a[:, None, :] - b[None, :, :]
    return np.nanmean(d * d, axis=-1)


def cost_matrix(dt, dj, alpha=0.0, beta=0.0, gamma=0.0, variant="minimal",
                hard=True, soft_pen=10.0, min_clean=MIN_CLEAN):
    """C[p, q] for treated parcel p and donor parcel q (both 196). variant 'minimal':
    phys=(elev,slope), terramind=hmean; 'full': phys+(aspect), terramind=hmean+hsd,
    and context (gamma). Rows/cols with nodata class or too little clean history get
    BIG everywhere (they can only go unmatched)."""
    C = np.zeros((NPARCEL, NPARCEL))
    if alpha:
        C += alpha * _blockdist(dt["phys"], dj["phys"])
        if variant == "full":
            C += alpha * _blockdist(dt["aspect"], dj["aspect"])
    if beta:
        C += beta * _blockdist(dt["hmean"], dj["hmean"])
        if variant == "full":
            C += beta * _blockdist(dt["hsd"], dj["hsd"])
    if gamma and variant == "full":
        C += gamma * _blockdist(dt["ctx"], dj["ctx"])
    C = np.nan_to_num(C, nan=BIG)
    mism = dt["cls"][:, None] != dj["cls"][None, :]
    C = C + (BIG if hard else soft_pen) * mism
    bad_t = (dt["cls"] == 0) | (dt["n_clean"] < min_clean)
    bad_j = (dj["cls"] == 0) | (dj["n_clean"] < min_clean)
    C[bad_t, :] = BIG; C[:, bad_j] = BIG
    return C


def align_pair(dt, dj, hard=True, dummy_cost=0.0, **kw):
    """Hungarian assignment treated positions -> donor parcels. Hard mode adds, per
    class c, max(0, n_t(c) - n_j(c)) dummy columns (E1.10) so only unavoidable
    positions go unmatched; any assignment still costing >= BIG is unmatched too.
    Returns perm (196,) int, -1 = unmatched."""
    C = cost_matrix(dt, dj, hard=hard, **kw)
    if hard:
        cols = []
        for c in np.unique(dt["cls"]):
            n_t = int((dt["cls"] == c).sum()); n_j = int((dj["cls"] == c).sum())
            for _ in range(max(0, n_t - n_j) + (1 if c == 0 else 0)):
                col = np.full(NPARCEL, BIG); col[dt["cls"] == c] = dummy_cost
                cols.append(col)
        # positions unusable for history reasons also need an exit
        col = np.full(NPARCEL, BIG); col[C.min(axis=1) >= BIG] = dummy_cost
        cols.append(col)
        Cx = np.column_stack([C] + cols) if cols else C
    else:
        Cx = C
    r, c = linear_sum_assignment(Cx)
    perm = np.full(NPARCEL, -1, dtype=int)
    for p, q in zip(r, c):
        if q < NPARCEL and Cx[p, q] < BIG:
            perm[p] = q
    return perm


def aligned_vec(panel, site, sensor, seq, perm):
    """Donor latent at period seq, re-ordered into the treated layout; NaN where the
    treated position has no donor parcel. Same-coordinate baseline: perm = arange(196)."""
    v = panel.L(site, sensor, seq)
    if v is None:
        return np.full(5 * NPARCEL, np.nan)
    A = v.reshape(5, NPARCEL)
    out = np.full((5, NPARCEL), np.nan)
    ok = perm >= 0
    out[:, ok] = A[:, perm[ok]]
    return out.ravel()


def block_mean(vec980, k=2):
    """(5,14,14) -> (5, 14/k, 14/k) block means (NaN-aware) -> flat. k=2 gives 245-d."""
    A = vec980.reshape(5, GRID, GRID)
    n = GRID // k
    out = np.full((5, n, n), np.nan)
    for i in range(n):
        for j in range(n):
            blk = A[:, i * k:(i + 1) * k, j * k:(j + 1) * k].reshape(5, -1)
            if np.isfinite(blk).any(axis=1).all():
                out[:, i, j] = np.nanmean(blk, axis=1)
    return out.ravel()


def same_position_agreement(dt, dj):
    return float((dt["cls"] == dj["cls"]).mean())


def class_hist(d):
    c = pd.Series(d["cls"]).map(NLCD_NAMES).value_counts()
    return c.to_dict()


# ---------------------------------------------------------------- estimators
def simplex_scm(y, X, ridge=0.0):
    """Notebook-11 `collab_scm` (SLSQP, bounds [0,1], sum w = 1, maxiter 5000,
    ftol 1e-12; rows with any non-finite entry dropped). `ridge` adds lam*||w||^2
    (Wang et al. 2025 use lam = 1) while keeping the simplex."""
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y_, X_ = y[valid], X[valid]
    Jn = X.shape[1]
    if valid.sum() == 0:
        return np.full(Jn, np.nan)
    obj = lambda w: float(np.mean((y_ - X_ @ w) ** 2) + ridge * np.sum(w * w))
    res = minimize(obj, np.repeat(1.0 / Jn, Jn), method="SLSQP",
                   bounds=[(0.0, 1.0)] * Jn,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 5000, "ftol": 1e-12})
    assert res.success, res.message
    return res.x


def rmse(e):
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e * e))) if e.size else np.nan


# ---------------------------------------------------------------- SHC blocks
def shc_blocks(train_p, m, n=1):
    """Chen-Yang-Yang historical blocks inside the training window. Treated block =
    (last m train periods, target). Historical block i = the same (m pre, n post) window
    shifted back by z_i = n + i - 1 periods so its pseudo-post period lies inside the
    training window. Returns [(pre_seqs, post_seqs), ...], N = len(train_p) - n - (m - 1)."""
    tp = list(train_p)
    T0 = len(tp)
    blocks = []
    for i in range(1, T0 - n - (m - 1) + 1):
        z = n + i - 1
        end = T0 - z                     # index (1-based within tp) of the block's last pre period
        pre = tp[end - m:end]
        post = tp[end:end + n]
        if len(pre) == m and len(post) == n:
            blocks.append((pre, post))
    return blocks


def treated_block(train_p, m):
    tp = list(train_p)
    return tp[-m:]
