"""
L2 -- weight identifiability, both pre-registered designs.

DESIGN 1 (H1) semi-synthetic weight recovery
    Pseudo-target z* = sum_j w*_j z_j (+ noise) over REAL visit latents; refit
    by simplex QP and ask whether w-hat recovers w*.
    Cells: K in {2,5,10} x w* in {dirichlet, spiky} x sigma in {0, residual-
    matched} x pool in {oracle-K, m=50, LARGE} x 2 encoder arms.

DESIGN 2 (H2) cross-space weight consistency
    Real patients, T0=1.  Fit SC in bag space (the observable machinery: trial
    StateSet -> Panel -> cosine donor schema, m=50) and in each latent space
    ON THE SAME DONOR POOL, then compare the per-patient weight vectors against
    a permuted-target null.

Outputs
    latent/results/weight_recovery.parquet        one row per pseudo-target
    latent/results/weight_recovery_summary.json   per-cell metrics + verdicts
    latent/results/cross_space.parquet            one row per (target, arm)
    latent/results/cross_space_summary.json       H2 metrics, null, verdict

DEVIATION FROM THE PRE-REGISTRATION, recorded here and in the summary JSON
--------------------------------------------------------------------------
The pre-registered `pool = m=all` (9,092 donors) is not computable with the
pre-registered solver.  `solve_simplex_qp` is SLSQP, whose per-iteration cost is
cubic in the number of donors; measured on this machine at d=768:

    m=200   0.13 s      m=1000   15.1 s      m=4000  did not finish in 900 s
    m=500   1.91 s      m=2000  133.5 s

which extrapolates to ~1.2e4 s (~3.5 h) for ONE m=9092 solve, i.e. ~4,800 CPU-
hours for the 24 m=all cells even at the pre-registration's own N=50 fallback.
The fallback it authorises (N=50) therefore does not rescue the cell.

What is run instead: the third pool level is **m=500** at **N=50** -- a 10x
larger pool than the primary m=50 and 50x the largest K, which is what the
m=all level was there to test (does a large pool of near-collinear donors
destroy identification?).  The measured cost curve above is reported as the
evidence, and no m=9092 number is quoted anywhere.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_weight_recovery.py
"""
from __future__ import annotations

import json
import os

# Parallelism here is over pseudo-targets; a BLAS pool per worker would
# oversubscribe the box (see mimic/scripts/run_phase2_fit.py's header).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "src"), str(_PRELIM / "mimic"),
          str(_PRELIM / "observable" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.synthctl.fit import Panel, solve_simplex_qp  # noqa: E402
from src.vendored.stats import boot_continuous  # noqa: E402
import encode as enc  # noqa: E402

# `trial_nct00174655` / `synthctl_trial` are imported INSIDE run_design2: they
# are specific to the NCT00174655 bag space, and the n2c2 rerun
# (run_n2c2_weight_recovery.py) reuses this module's design-1 machinery only.

RESULTS = _LATENT / "results"
DATA = _LATENT / "data"

#: (arm label, npz).  `qwen3` here is the 768-d MRL truncation: the primary.
ARM_FILES = {"qwen3": DATA / "states_trial_qwen3.npz",
             "biolord": DATA / "states_trial_biolord.npz"}

SEED = 0
BOOT_B = 2000
BOOT_SEED = 0

K_GRID = (2, 5, 10)
WSTAR_GRID = ("dirichlet", "spiky")
SPIKE = 0.9
#: pool label -> (m or None for oracle, N pseudo-targets)
POOLS: Dict[str, Tuple[Optional[int], int]] = {
    "oracle": (None, 500),
    "m50": (50, 500),
    "m500": (500, 50),      # DEVIATION: stands in for the m=all level
}
#: optional per-pool restriction of the K grid, e.g. `{"mall": (5,)}` to run the
#: m=all *control* only at K=5.  Empty means every pool sees the full K grid.
POOL_K_FILTER: Dict[str, Tuple[int, ...]] = {}

#: measured single-solve cost, reported as the justification for the deviation
SOLVE_COST_SECONDS = {200: 0.13, 500: 1.91, 1000: 15.1, 2000: 133.5,
                      4000: float('inf')}   # 4000 did not finish in 900 s

T0 = 1
M_CROSS = 50
CROSS_SCHEMA = "cosine"

#: how many real visits the residual-matched sigma is estimated from
N_RESIDUAL_REF = 500

_G: Dict[str, Any] = {}


def _jsonable(o):
    """numpy scalars survive `groupby` keys and pandas aggregation."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ==========================================================================
# residual-matched sigma
# ==========================================================================


def residual_sigma(Z: np.ndarray, pid: np.ndarray, n_ref: int = N_RESIDUAL_REF,
                   m: int = 50, seed: int = SEED) -> Dict[str, float]:
    """Per-coordinate noise scale matched to REAL m=50 SC residuals.

    The pre-registration says "residual-matched", defines it as "the median L2
    residual of real m=50 SC fits in that latent space", and delegates the exact
    operationalisation.  What is done here, stated so the realised number is
    interpretable:

    * `n_ref` real visit rows are drawn (seed 0) as targets;
    * the pool is the `m` nearest OTHER-PATIENT visits by cosine -- other-patient
      because a synthetic control never uses the target's own unit as a donor,
      and the target's own adjacent visits are near-duplicates that would drive
      the residual to ~0 and make the noise cell vacuous;
    * the fit is the same `solve_simplex_qp` used everywhere else;
    * `sigma` is the MEDIAN residual L2 norm divided by sqrt(d), i.e. a
      per-coordinate RMS, so that isotropic noise `N(0, sigma^2 I_d)` added to a
      d-dimensional pseudo-target has the same expected per-coordinate size as a
      real fit's residual.  Matching the norm rather than the norm-per-coordinate
      would be ill-defined across the two designs, whose targets have different
      dimension (d vs (T0+1)d).
    """
    n, d = Z.shape
    rng = np.random.default_rng([seed, 7])
    rows = rng.choice(n, size=min(n_ref, n), replace=False)
    res: List[float] = []
    for r in rows:
        sims = Z @ Z[r]
        sims[pid == pid[r]] = -np.inf
        pool = np.argsort(-sims, kind="stable")[:m]
        sol = solve_simplex_qp(Z[pool], Z[r])
        res.append(float(np.sqrt(max(sol.sse, 0.0))))
    res_a = np.array(res)
    med = float(np.median(res_a))
    return {"n_ref": int(len(rows)), "m": m, "d": int(d),
            "median_residual_l2": med,
            "mean_residual_l2": float(res_a.mean()),
            "p25_residual_l2": float(np.percentile(res_a, 25)),
            "p75_residual_l2": float(np.percentile(res_a, 75)),
            "sigma_per_coordinate": med / np.sqrt(d)}


# ==========================================================================
# design 1
# ==========================================================================


def draw_replicates(n_corpus: int, K: int, wstar: str, N: int,
                    seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Donor indices and true weights, shared across pools, sigmas and arms.

    Keying the stream on (K, w* family) only makes every cell that differs in
    pool / sigma / encoder a PAIRED comparison on the identical pseudo-targets,
    so a cell-to-cell difference is the manipulated factor and not a different
    random draw.
    """
    code = WSTAR_GRID.index(wstar)
    rng = np.random.default_rng([seed, K, code])
    D = np.empty((N, K), dtype=np.int64)
    W = np.empty((N, K), dtype=np.float64)
    for r in range(N):
        D[r] = rng.choice(n_corpus, size=K, replace=False)
        if wstar == "dirichlet":
            W[r] = rng.dirichlet(np.ones(K))
        else:
            j = int(rng.integers(K))
            W[r] = (1.0 - SPIKE) / (K - 1)
            W[r, j] = SPIKE
    return D, W


def draw_noise(d: int, K: int, wstar: str, N: int, seed: int = SEED) -> np.ndarray:
    """Standard normal draws, shared across arms so the arms stay paired."""
    code = WSTAR_GRID.index(wstar)
    rng = np.random.default_rng([seed, K, code, 99])
    return rng.standard_normal((N, d))


def _cell_init(Z: np.ndarray, n_corpus: int) -> None:
    _G["Z"] = Z
    _G["n_corpus"] = n_corpus


def _one_replicate(args):
    (rep, donors, wstar_vec, eps, sigma, pool_m) = args
    Z = _G["Z"]
    n_corpus = _G["n_corpus"]
    z_mix = wstar_vec @ Z[donors]
    z = z_mix + sigma * eps if sigma > 0 else z_mix

    if pool_m is None:                       # oracle: exactly the true donors
        rows = np.asarray(donors)
    else:
        sims = Z @ z                          # unit rows: ranking == cosine
        rows = np.argsort(-sims, kind="stable")[:pool_m]

    t0 = time.time()
    sol = solve_simplex_qp(Z[rows], z)
    dt = time.time() - t0

    w_hat = sol.w
    full_hat = np.zeros(n_corpus)
    full_hat[rows] = w_hat
    full_star = np.zeros(n_corpus)
    full_star[donors] = wstar_vec
    l1 = float(np.abs(full_hat - full_star).sum())

    true_top = int(donors[int(np.argmax(wstar_vec))])
    hat_top = int(rows[int(np.argmax(w_hat))])
    return {
        "rep": int(rep),
        "l1": l1,
        "top1_hit": bool(hat_top == true_top),
        "w_max": float(w_hat.max()),
        "w_on_true_top1": float(full_hat[true_top]),
        "n_true_in_pool": int(np.isin(donors, rows).sum()),
        "pool_size": int(len(rows)),
        "sse": float(sol.sse),
        "residual_l2": float(np.sqrt(max(sol.sse, 0.0))),
        "status": int(sol.status),
        "nit": int(sol.nit),
        "simplex_violation": float(sol.simplex_violation),
        "solve_seconds": dt,
    }


def run_design1(arms: Dict[str, np.ndarray], pid: np.ndarray,
                sigmas: Dict[str, float], n_jobs: int) -> pd.DataFrame:
    import multiprocessing as mp

    print(f"[d1] pools={POOLS}", flush=True)
    rows: List[Dict[str, Any]] = []
    for arm, Z in arms.items():
        n_corpus, d = Z.shape
        ctx = mp.get_context("fork")
        _cell_init(Z, n_corpus)
        for K in K_GRID:
            for wstar in WSTAR_GRID:
                Dmax, Wmax = draw_replicates(n_corpus, K, wstar, 500)
                Emax = draw_noise(d, K, wstar, 500)
                for pool, (pool_m, N) in POOLS.items():
                    ks = POOL_K_FILTER.get(pool)
                    if ks is not None and K not in ks:
                        continue
                    for sig_name in ("sigma0", "sigma_res"):
                        sigma = 0.0 if sig_name == "sigma0" else sigmas[arm]
                        tasks = [(r, Dmax[r], Wmax[r], Emax[r], sigma, pool_m)
                                 for r in range(N)]
                        t0 = time.time()
                        if n_jobs > 1 and (pool_m or 0) >= 200:
                            with ctx.Pool(n_jobs, initializer=_cell_init,
                                          initargs=(Z, n_corpus)) as pool_p:
                                out = pool_p.map(_one_replicate, tasks,
                                                 chunksize=1)
                        else:
                            out = [_one_replicate(t) for t in tasks]
                        for o in out:
                            o.update(arm=arm, K=K, wstar=wstar, pool=pool,
                                     sigma_name=sig_name, sigma=sigma, N=N)
                        rows.extend(out)
                        print(f"[d1] {arm} K={K} {wstar} {pool} {sig_name} "
                              f"N={N} in {time.time()-t0:.1f}s "
                              f"meanL1={np.mean([o['l1'] for o in out]):.4f}",
                              flush=True)
    return pd.DataFrame(rows)


def summarise_design1(df: pd.DataFrame) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    keys = ["arm", "K", "wstar", "pool", "sigma_name"]
    for key, g in df.groupby(keys, sort=True):
        l1 = g["l1"].to_numpy()
        lo, hi = boot_continuous(l1, B=BOOT_B, seed=BOOT_SEED)
        cell: Dict[str, Any] = {k: (int(v) if isinstance(v, np.integer) else v)
                                for k, v in zip(keys, key)}
        cell.update({
            "N": int(len(g)),
            "sigma": float(g["sigma"].iloc[0]),
            "mean_l1": float(l1.mean()),
            "l1_ci_lo": float(lo), "l1_ci_hi": float(hi),
            "median_l1": float(np.median(l1)),
            "top1_recovery": float(g["top1_hit"].mean()),
            "mean_w_max": float(g["w_max"].mean()),
            "mean_true_donors_in_pool": float(g["n_true_in_pool"].mean()),
            "frac_all_true_donors_in_pool":
                float((g["n_true_in_pool"] == g["K"]).mean()),
            "mean_residual_l2": float(g["residual_l2"].mean()),
            "n_not_converged": int((g["status"] != 0).sum()),
            "max_simplex_violation": float(g["simplex_violation"].max()),
            "mean_solve_seconds": float(g["solve_seconds"].mean()),
        })
        if key[keys.index("wstar")] == "spiky":
            err = np.abs(g["w_max"].to_numpy() - SPIKE)
            elo, ehi = boot_continuous(err, B=BOOT_B, seed=BOOT_SEED)
            cell.update({"mean_wmax_err": float(err.mean()),
                         "wmax_err_ci_lo": float(elo),
                         "wmax_err_ci_hi": float(ehi)})
        cells.append(cell)
    return {"cells": cells}


def h1_verdicts(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered thresholds, applied literally.

    Reading of the criteria, fixed here because the text leaves one thing to
    the analyst: "mean L1 <= 0.30 at K=5" and "top-1 recovery >= 90%" are read
    on the **Dirichlet** K=5 cell (the generic mixture; the spiky cell has its
    own criterion), and "|w_max - 0.9| <= 0.10" on the **spiky** K=5 cell.  Both
    at pool=m50, sigma=0.  Top-1 recovery in the spiky cell is reported too.
    """
    def get(arm, K, wstar, pool, sig):
        for c in cells:
            if (c["arm"] == arm and c["K"] == K and c["wstar"] == wstar
                    and c["pool"] == pool and c["sigma_name"] == sig):
                return c
        return None

    out: Dict[str, Any] = {}
    gate = {}
    for arm in ARM_FILES:
        worst = max(get(arm, K, w, "oracle", "sigma0")["mean_l1"]
                    for K in K_GRID for w in WSTAR_GRID)
        gate[arm] = {"max_mean_l1_over_oracle_sigma0_cells": worst,
                     "threshold": 0.01, "passed": bool(worst <= 0.01)}
    out["oracle_gate"] = gate
    out["oracle_gate_passed_all_arms"] = all(v["passed"] for v in gate.values())

    for arm in ARM_FILES:
        d5 = get(arm, 5, "dirichlet", "m50", "sigma0")
        s5 = get(arm, 5, "spiky", "m50", "sigma0")
        sup = (d5["mean_l1"] <= 0.30 and s5["mean_wmax_err"] <= 0.10
               and d5["top1_recovery"] >= 0.90)
        ref = (s5["mean_wmax_err"] > 0.25 or d5["top1_recovery"] < 0.60)
        out[arm] = {
            "K5_m50_sigma0_dirichlet_mean_l1": d5["mean_l1"],
            "K5_m50_sigma0_dirichlet_l1_ci": [d5["l1_ci_lo"], d5["l1_ci_hi"]],
            "K5_m50_sigma0_dirichlet_top1": d5["top1_recovery"],
            "K5_m50_sigma0_spiky_top1": s5["top1_recovery"],
            "K5_m50_sigma0_spiky_wmax_err": s5["mean_wmax_err"],
            "K5_m50_sigma0_spiky_wmax_err_ci": [s5["wmax_err_ci_lo"],
                                                s5["wmax_err_ci_hi"]],
            "K5_m50_sigma0_spiky_mean_w_max": s5["mean_w_max"],
            "supported": bool(sup),
            "refuted": bool(ref),
            "verdict": ("supported" if sup else
                        "refuted" if ref else "neither (between thresholds)"),
        }
    prim = {a: out[a]["K5_m50_sigma0_dirichlet_mean_l1"] for a in ARM_FILES}
    out["encoder_selection"] = {
        "rule": "arm with lower mean L1 at K=5, m=50, sigma=0 (Dirichlet)",
        "values": prim,
        "winner": min(prim, key=prim.get),
        "tie": bool(abs(prim["qwen3"] - prim["biolord"]) < 1e-6),
    }
    return out


# ==========================================================================
# design 2
# ==========================================================================


def latent_panel(panel: Panel, Z: np.ndarray, pid: np.ndarray,
                 t: np.ndarray) -> np.ndarray:
    """`(n, (T0+1)*d)` pre-period blocks in a latent space, row-aligned to `panel`.

    Same patients, same row order, same flattening convention as `Panel.build`
    -- so `panel.P[rows]` and this matrix's `[rows]` name the same donors and
    the two weight vectors are directly comparable coordinate by coordinate.
    """
    lookup = {(int(p), int(tt)): r for r, (p, tt) in enumerate(zip(pid, t))}
    d = Z.shape[1]
    P = np.empty((panel.n, (panel.T0 + 1) * d), dtype=np.float64)
    for i, sid in enumerate(panel.ids):
        p = int(sid)
        P[i] = np.concatenate([Z[lookup[(p, tt)]] for tt in range(panel.T0 + 1)])
    return P


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def run_design2(arms: Dict[str, np.ndarray], pid: np.ndarray, t: np.ndarray
                ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    import trial_nct00174655 as trial
    from synthctl_trial import donor_selection as ds

    ss = trial.load_trial_states()
    panel = Panel.build(ss, T0)
    n = panel.n
    print(f"[d2] bag panel: n={n} eligible at T0={T0}, d={panel.d}", flush=True)

    lat = {arm: latent_panel(panel, Z, pid, t) for arm, Z in arms.items()}

    # one permutation draw per target, seed 0
    rng = np.random.default_rng(SEED)
    perm = np.array([rng.choice(np.delete(np.arange(n), i)) for i in range(n)])

    W_bag: List[np.ndarray] = []
    W_lat: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    diag: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(n):
        rows, label = ds.select_donors(panel, i, M_CROSS, schema=CROSS_SCHEMA)
        assert label == CROSS_SCHEMA and len(rows) == M_CROSS
        sb = solve_simplex_qp(panel.P[rows], panel.P[i])
        W_bag.append(sb.w)
        rec = {"i": i, "subject_id": panel.ids[i],
               "bag_pre_rmse": float(np.sqrt(sb.sse / panel.P.shape[1])),
               "bag_max_w": float(sb.w.max()),
               "bag_eff_donors": float(1.0 / max(float(sb.w @ sb.w), 1e-300)),
               "bag_status": int(sb.status)}
        for arm in arms:
            sl = solve_simplex_qp(lat[arm][rows], lat[arm][i])
            W_lat[arm].append(sl.w)
            rec[f"{arm}_pre_rmse"] = float(np.sqrt(sl.sse / lat[arm].shape[1]))
            rec[f"{arm}_max_w"] = float(sl.w.max())
            rec[f"{arm}_eff_donors"] = float(1.0 / max(float(sl.w @ sl.w), 1e-300))
            rec[f"{arm}_status"] = int(sl.status)
        diag.append(rec)
    print(f"[d2] fitted {n} targets x {1+len(arms)} spaces in "
          f"{time.time()-t0:.1f}s", flush=True)

    rows_out: List[Dict[str, Any]] = []
    for arm in arms:
        for i in range(n):
            wb, wl = W_bag[i], W_lat[arm][i]
            wn = W_lat[arm][int(perm[i])]     # permuted-target null
            rows_out.append({
                "arm": arm, "i": i, "subject_id": diag[i]["subject_id"],
                "cos": _cos(wb, wl), "l1": float(np.abs(wb - wl).sum()),
                "top1_agree": bool(int(np.argmax(wb)) == int(np.argmax(wl))),
                "null_i": int(perm[i]),
                "null_cos": _cos(wb, wn),
                "null_l1": float(np.abs(wb - wn).sum()),
                "null_top1_agree": bool(int(np.argmax(wb)) == int(np.argmax(wn))),
                **{k: v for k, v in diag[i].items()
                   if k.startswith(("bag_", arm + "_"))},
            })
    df = pd.DataFrame(rows_out)

    summary: Dict[str, Any] = {
        "n_targets": n, "T0": T0, "m": M_CROSS, "donor_schema": CROSS_SCHEMA,
        "bag_dim": int(panel.P.shape[1]),
        "null": ("weights of a random OTHER target with the same pool size, one "
                 "draw per target at seed 0; compared positionally, which is "
                 "what destroys the donor correspondence the real comparison "
                 "has"),
        "arms": {},
    }
    for arm in arms:
        g = df[df["arm"] == arm]
        c, nc = g["cos"].to_numpy(), g["null_cos"].to_numpy()
        clo, chi = boot_continuous(c, B=BOOT_B, seed=BOOT_SEED)
        nlo, nhi = boot_continuous(nc, B=BOOT_B, seed=BOOT_SEED)
        l1lo, l1hi = boot_continuous(g["l1"].to_numpy(), B=BOOT_B, seed=BOOT_SEED)
        p95 = float(np.percentile(nc, 95))
        q1, q3 = float(np.percentile(nc, 25)), float(np.percentile(nc, 75))
        t1, nt1 = float(g["top1_agree"].mean()), float(g["null_top1_agree"].mean())
        sup = bool(float(c.mean()) > p95 and
                   (t1 >= 3 * nt1 if nt1 > 0 else t1 > 0))
        ref = bool(q1 <= float(c.mean()) <= q3)
        summary["arms"][arm] = {
            "mean_weight_cosine": float(c.mean()),
            "weight_cosine_ci": [float(clo), float(chi)],
            "median_weight_cosine": float(np.median(c)),
            "mean_l1": float(g["l1"].mean()), "l1_ci": [float(l1lo), float(l1hi)],
            "top1_agreement": t1,
            "null_mean_cosine": float(nc.mean()),
            "null_cosine_ci": [float(nlo), float(nhi)],
            "null_cosine_p95": p95,
            "null_cosine_iqr": [q1, q3],
            "null_mean_l1": float(g["null_l1"].mean()),
            "null_top1_agreement": nt1,
            "top1_ratio_vs_null": (t1 / nt1) if nt1 > 0 else float("inf"),
            "mean_bag_pre_rmse": float(g["bag_pre_rmse"].mean()),
            "mean_latent_pre_rmse": float(g[f"{arm}_pre_rmse"].mean()),
            "mean_bag_eff_donors": float(g["bag_eff_donors"].mean()),
            "mean_latent_eff_donors": float(g[f"{arm}_eff_donors"].mean()),
            "mean_bag_max_w": float(g["bag_max_w"].mean()),
            "mean_latent_max_w": float(g[f"{arm}_max_w"].mean()),
            "supported": sup, "refuted": ref,
            "verdict": ("supported" if sup else
                        "refuted (within null IQR)" if ref else
                        "neither (above null IQR, criteria not both met)"),
        }
    return df, summary


# ==========================================================================


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--skip-design1", action="store_true")
    ap.add_argument("--design1-from-parquet", action="store_true",
                    help="re-summarise an existing weight_recovery.parquet "
                         "instead of refitting; the fits are unchanged")
    ap.add_argument("--skip-design2", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    arms: Dict[str, np.ndarray] = {}
    pid = t = None
    for arm, path in ARM_FILES.items():
        s = enc.load_states(path)
        arms[arm] = np.ascontiguousarray(s["X"], dtype=np.float64)
        if pid is None:
            pid, t = s["patient_idx"], s["t"]
        else:
            assert np.array_equal(pid, s["patient_idx"])
            assert np.array_equal(t, s["t"])
    print({a: Z.shape for a, Z in arms.items()}, flush=True)

    timings: Dict[str, float] = {}
    meta = {
        "stage": "L2", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED, "boot_B": BOOT_B, "boot_seed": BOOT_SEED,
        "arm_state_files": {a: str(p) for a, p in ARM_FILES.items()},
        "deviations": [],
    }

    if not args.skip_design1:
        t0 = time.time()
        sig = {}
        for arm, Z in arms.items():
            s = residual_sigma(Z, pid)
            sig[arm] = s["sigma_per_coordinate"]
            meta.setdefault("residual_matched_sigma", {})[arm] = s
            print(f"[sigma] {arm}: {s}", flush=True)
        timings["residual_sigma_seconds"] = round(time.time() - t0, 1)

        t0 = time.time()
        if args.design1_from_parquet:
            df1 = pd.read_parquet(RESULTS / "weight_recovery.parquet")
            timings["design1_seconds"] = "reused existing parquet"
        else:
            df1 = run_design1(arms, pid, sig, args.jobs)
            timings["design1_seconds"] = round(time.time() - t0, 1)
            df1.to_parquet(RESULTS / "weight_recovery.parquet", index=False)
        summ = summarise_design1(df1)
        summ["verdicts"] = h1_verdicts(summ["cells"])
        summ.update(meta)
        summ["deviations"] = [{
            "item": "pool level 'm=all' (9,092 donors)",
            "replaced_by": "m=500 at N=50",
            "reason": ("solve_simplex_qp is SLSQP with cubic per-iteration cost "
                       "in m; measured single-solve times at d=768 were "
                       f"{SOLVE_COST_SECONDS}, extrapolating to ~1.2e4 s per "
                       "m=9092 solve (~3.5 h), i.e. ~4,800 CPU-hours for the 24 "
                       "m=all cells even at the pre-registration's own N=50 "
                       "fallback. No m=9092 number is quoted."),
            "measured_solve_seconds_by_m": SOLVE_COST_SECONDS,
        }, {
            "item": "residual-matched sigma",
            "definition": ("median L2 residual of real m=50 SC fits (500 real "
                           "visits, seed 0, 50 nearest other-patient visits by "
                           "cosine) divided by sqrt(d) -> per-coordinate sigma; "
                           "realised values in residual_matched_sigma"),
            "reason": "the pre-registration delegates the operationalisation",
        }]
        summ["timings"] = timings
        (RESULTS / "weight_recovery_summary.json").write_text(
            json.dumps(summ, indent=2, default=_jsonable))
        print(json.dumps(summ["verdicts"], indent=2, default=_jsonable), flush=True)

    if not args.skip_design2:
        t0 = time.time()
        df2, summ2 = run_design2(arms, pid, t)
        timings["design2_seconds"] = round(time.time() - t0, 1)
        df2.to_parquet(RESULTS / "cross_space.parquet", index=False)
        summ2.update({k: v for k, v in meta.items() if k != "deviations"})
        summ2["timings"] = timings
        (RESULTS / "cross_space_summary.json").write_text(
            json.dumps(summ2, indent=2, default=_jsonable))
        print(json.dumps(summ2["arms"], indent=2, default=_jsonable), flush=True)

    print(f"[L2] total {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
