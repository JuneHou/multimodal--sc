"""Post-period holdout: equal-weight donor mean vs estimated-weight SC
(POST-HOC, NOT pre-registered).

Jun's question (2026-08-13): mimic the collaborator's satellite comparison --
equal-weight donor counterfactual vs solver-estimated weights -- on data where
the counterfactual can be scored against an observed truth.  Her pipeline
(REAP data/scripts/09_test_counterfactual_average.ipynb) is a 2x2 DiD with
fixed 1/10 weights over 10 matched control sites and NO holdout: parallel
trends is assumed via matching, never tested.  n2c2 has what the satellite
lacks: every patient is untreated and the next visit is observed, so every
patient is a direct test of "did the counterfactual construction predict
reality?"  Conversely n2c2 has no treatment event, so no effect is estimated
here -- this experiment checks the ESTIMATOR, not an effect.

Design (primary cell T0=1): fit on visits t=0,1; forecast the OBSERVED t=2
embedding.  Arms per target, all scored in the same space:

  sc         solve_simplex_qp over the 50 nearest donors (cosine on the
             flattened pre-period block); forecast = w_hat @ Y[pool]
  equal10    mean of the 10 nearest donors' Y -- the collaborator's
             estimator verbatim (matched set, fixed 1/k weights)
  equal50    mean over the full 50-donor pool (equal weights without the
             tight matching)
  lvcf       carry the last pre-period visit forward (do-nothing floor)
  sc_random  SC over 50 uniformly random donors (placebo: prices donor
             SELECTION separately from donor WEIGHTING)

Scoring: per-coordinate RMSE and cosine of forecast vs the observed next
visit; PAIRED per-patient contrasts with percentile-bootstrap CIs (B=2000,
seed 0) and win fractions; Spearman(pre-fit RMSE, forecast RMSE) for the
"does pre-fit quality predict counterfactual quality" assumption.

POST-HOC: no pre-registered thresholds exist, so the summary reports effect
sizes and CIs only -- no supported/refuted language.  Robustness cells:
T0=0 (n=288) and T0=2 (n=274).  CPU-only, no generation.

    /data/wang/junh/envs/medrag/bin/python preliminary/did/run_postperiod_holdout.py
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_DID = Path(__file__).resolve().parent
_PRELIM = _DID.parent
_LATENT = _PRELIM / "latent"
for p in (str(_LATENT / "src"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import n2c2_concepts as nc  # noqa: E402
from src.synthctl.fit import solve_simplex_qp  # noqa: E402
from src.vendored.stats import boot_continuous  # noqa: E402

RESULTS = _DID / "results"
DATA = _LATENT / "data"
H2_PARQUET = _LATENT / "results" / "n2c2_cross_space.parquet"

ARM_FILES = {"qwen3": DATA / "states_n2c2_qwen3.npz",
             "biolord": DATA / "states_n2c2_biolord.npz"}
T0_GRID = (1, 0, 2)      # primary first; 0 and 2 are robustness cells
M_POOL = 50              # SC pool -- H2's pre-registered m
K_EQUAL = 10             # the collaborator's matched-set size
SEED = 0
BOOT_B = 2000
BOOT_SEED = 0
ESTIMATORS = ("sc", "equal10", "equal50", "lvcf", "sc_random")
CONTRASTS = (("sc", "equal10"), ("sc", "equal50"), ("sc", "lvcf"),
             ("sc", "sc_random"), ("equal10", "lvcf"),
             ("equal10", "equal50"))


def cosine_pool(P: np.ndarray, i: int, m: int) -> np.ndarray:
    """Top-m donors by cosine on the flattened pre-period block, self
    excluded, ties broken by row order -- the H2 donor schema
    (run_n2c2_h2.cosine_donors), computed here in the latent block itself
    because this experiment has no bag space in the loop."""
    nrm = np.linalg.norm(P, axis=1)
    denom = nrm * nrm[i]
    sim = np.divide(P @ P[i], denom, out=np.full(P.shape[0], -np.inf),
                    where=denom > 0)
    sim[i] = -np.inf
    return np.argsort(-sim, kind="stable")[:m]


def next_states(X: np.ndarray, index: pd.DataFrame, T0: int,
                ids: List[int]) -> np.ndarray:
    """`(n, d)` observed state at t = T0+1, row-aligned to
    `cohort_flat_blocks(..., T0)`s `ids` -- the held-out truth.  Eligibility
    (`t_max >= T0+1`) already guarantees the row exists."""
    rowmap = {(int(p), int(tt)): r for r, (p, tt) in
              enumerate(zip(index["patient_idx"], index["t"]))}
    return np.stack([X[rowmap[(p, T0 + 1)]] for p in ids])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def run_cell(arm: str, Z: np.ndarray, index: pd.DataFrame, T0: int
             ) -> pd.DataFrame:
    ids, P = nc.cohort_flat_blocks(Z, index, T0)
    Y = next_states(Z, index, T0, ids)
    n, d = len(ids), Z.shape[1]
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(n):
        pool = cosine_pool(P, i, M_POOL)
        sol = solve_simplex_qp(P[pool], P[i])

        rng = np.random.default_rng([SEED, T0, i])
        rand_pool = rng.choice(np.delete(np.arange(n), i), size=M_POOL,
                               replace=False)
        sol_r = solve_simplex_qp(P[rand_pool], P[i])

        forecasts = {
            "sc": sol.w @ Y[pool],
            "equal10": Y[pool[:K_EQUAL]].mean(axis=0),
            "equal50": Y[pool].mean(axis=0),
            "lvcf": P[i][T0 * d:(T0 + 1) * d],
            "sc_random": sol_r.w @ Y[rand_pool],
        }
        for est, f in forecasts.items():
            rec: Dict[str, Any] = {
                "arm": arm, "T0": T0, "patient_idx": int(ids[i]),
                "estimator": est,
                "rmse": float(np.linalg.norm(f - Y[i]) / np.sqrt(d)),
                "cos": _cos(f, Y[i]),
            }
            if est == "sc":
                rec.update({
                    "pre_rmse": float(np.sqrt(max(sol.sse, 0.0)
                                              / P.shape[1])),
                    "w_max": float(sol.w.max()),
                    "eff_donors": float(1.0 / max(float(sol.w @ sol.w),
                                                  1e-300)),
                    "status": int(sol.status),
                })
            if est == "sc_random":
                rec["status"] = int(sol_r.status)
            rows.append(rec)
    print(f"[{arm} T0={T0}] n={n} fitted in {time.time()-t0:.1f}s",
          flush=True)
    return pd.DataFrame(rows)


def summarise_cell(df: pd.DataFrame) -> Dict[str, Any]:
    by_est = {e: df[df["estimator"] == e].sort_values("patient_idx")
              for e in ESTIMATORS}
    n = len(next(iter(by_est.values())))
    out: Dict[str, Any] = {"n_targets": n, "estimators": {}, "contrasts": {}}
    for e, g in by_est.items():
        r = g["rmse"].to_numpy()
        lo, hi = boot_continuous(r, B=BOOT_B, seed=BOOT_SEED)
        out["estimators"][e] = {
            "mean_rmse": float(r.mean()), "rmse_ci": [float(lo), float(hi)],
            "median_rmse": float(np.median(r)),
            "mean_cos": float(g["cos"].mean()),
            "median_cos": float(g["cos"].median()),
        }
    sc = by_est["sc"]
    out["estimators"]["sc"].update({
        "mean_pre_rmse": float(sc["pre_rmse"].mean()),
        "mean_w_max": float(sc["w_max"].mean()),
        "mean_eff_donors": float(sc["eff_donors"].mean()),
        "n_not_converged": int((sc["status"] != 0).sum()),
    })
    rho = spearmanr(sc["pre_rmse"], sc["rmse"])
    out["spearman_pre_rmse_vs_forecast_rmse_sc"] = {
        "rho": float(rho.statistic), "p": float(rho.pvalue)}
    for a, b in CONTRASTS:
        diff = (by_est[a]["rmse"].to_numpy()
                - by_est[b]["rmse"].to_numpy())      # paired: same patients
        lo, hi = boot_continuous(diff, B=BOOT_B, seed=BOOT_SEED)
        out["contrasts"][f"{a}_minus_{b}"] = {
            "mean_paired_rmse_diff": float(diff.mean()),
            "diff_ci": [float(lo), float(hi)],
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            f"win_frac_{a}": float((diff < 0).mean()),
        }
    return out


def main() -> int:
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(DATA / "n2c2_corpus.parquet",
                            columns=["patient_idx", "t"])

    frames: List[pd.DataFrame] = []
    summary: Dict[str, Any] = {
        "stage": "post-period holdout (post-hoc, not pre-registered)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "design": "fit on t=0..T0, forecast the observed t=T0+1 embedding",
        "primary_cell": {"arm": "qwen3", "T0": 1},
        "m_pool": M_POOL, "k_equal": K_EQUAL,
        "donor_schema": "cosine on the flattened pre-period latent block "
                        "(H2 convention), self excluded",
        "seed": SEED, "boot_B": BOOT_B, "boot_seed": BOOT_SEED,
        "cells": {},
    }
    for arm, path in ARM_FILES.items():
        z = np.load(path, allow_pickle=True)
        assert np.array_equal(z["patient_idx"],
                              index["patient_idx"].to_numpy())
        assert np.array_equal(z["t"], index["t"].to_numpy())
        Z = np.ascontiguousarray(z["X"], dtype=np.float64)
        for T0 in T0_GRID:
            df = run_cell(arm, Z, index, T0)
            frames.append(df)
            summary["cells"][f"{arm}_T0{T0}"] = summarise_cell(df)

    out = pd.concat(frames, ignore_index=True)

    # anchor: the T0=1 cohort must be exactly H2's 287 patients
    h2_ids = set(pd.read_parquet(H2_PARQUET, columns=["patient_idx"])
                 ["patient_idx"].unique().tolist())
    ours = set(out.loc[out["T0"] == 1, "patient_idx"].unique().tolist())
    summary["anchor_t01_cohort_equals_h2"] = {
        "n_ours": len(ours), "n_h2": len(h2_ids),
        "identical": bool(ours == h2_ids)}
    assert ours == h2_ids, "T0=1 cohort differs from H2's 287 patients"

    out.to_parquet(RESULTS / "postperiod_holdout.parquet", index=False)
    summary["wall_clock_seconds"] = round(time.time() - t_start, 1)
    (RESULTS / "postperiod_holdout_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary["cells"].items()
                      if k == "qwen3_T01"}, indent=2), flush=True)
    print(f"[holdout] total {summary['wall_clock_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
