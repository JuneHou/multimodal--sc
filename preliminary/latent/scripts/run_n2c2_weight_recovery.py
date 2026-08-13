"""
L2 on the n2c2 corpus -- H1 (semi-synthetic weight recovery) ONLY.

H2 needs the UMLS concept-bag observable space, whose Metathesaurus download has
not landed (addendum, "H2 bag space"); it is not attempted here.  Everything
else is `run_weight_recovery.py`'s design-1 machinery, unchanged, pointed at the
n2c2 latents:

    cells   K in {2,5,10} x w* in {dirichlet, spiky-0.9} x sigma in {0,
            residual-matched} x pool in {oracle-K, m=50, m=all=1263} x 2 arms
    N       500 per cell, seed 0; bootstrap B=2000, seed 0
    gate    oracle-K, sigma=0 -> mean L1 <= 0.01

The m=all level is the addendum's: the corpus is 1,264 records, so the pool that
was computationally impossible at 9,092 donors is feasible.  Pool size 1,263 is
the addendum's number (all donors bar one; the semi-synthetic target is not a
corpus row, so no row has to be excluded -- it is the top-1263 by cosine).

Outputs
    latent/results/n2c2_weight_recovery.parquet
    latent/results/n2c2_weight_recovery_summary.json

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_n2c2_weight_recovery.py
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
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "scripts"), str(_LATENT / "src"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import encode as enc  # noqa: E402
import run_weight_recovery as wr  # noqa: E402

RESULTS = _LATENT / "results"
DATA = _LATENT / "data"

ARM_FILES = {"qwen3": DATA / "states_n2c2_qwen3.npz",
             "biolord": DATA / "states_n2c2_biolord.npz"}
M_ALL = 1263
POOLS = {"oracle": (None, 500), "m50": (50, 500), "mall": (M_ALL, 500)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=96)
    ap.add_argument("--n-mall", type=int, default=500,
                    help="N for the m=all cells (500 = pre-registered)")
    ap.add_argument("--mall-k", default="",
                    help="restrict the m=all pool to these K (comma-separated); "
                         "empty = the full K grid")
    ap.add_argument("--from-parquet", action="store_true",
                    help="re-summarise the existing parquet, no refitting")
    ap.add_argument("--pools", default="oracle,m50,mall",
                    help="which pool levels to fit (comma-separated); the "
                         "cost of an m=all solve is measured, not assumed")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output filenames (staging runs)")
    args = ap.parse_args()

    POOLS["mall"] = (M_ALL, args.n_mall)
    keep = args.pools.split(",")
    for k in list(POOLS):
        if k not in keep:
            del POOLS[k]
    mall_k = tuple(int(x) for x in args.mall_k.split(",")) if args.mall_k else None
    if mall_k:
        wr.POOL_K_FILTER = {"mall": mall_k}
    sfx = args.out_suffix
    parquet_path = RESULTS / f"n2c2_weight_recovery{sfx}.parquet"
    summary_path = RESULTS / f"n2c2_weight_recovery_summary{sfx}.json"
    # point the reused machinery at this corpus
    wr.ARM_FILES = ARM_FILES
    wr.POOLS = POOLS

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
    n_corpus = arms["qwen3"].shape[0]
    assert n_corpus == 1264, f"corpus is {n_corpus} rows, expected 1264"
    print({a: Z.shape for a, Z in arms.items()}, flush=True)

    timings: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "stage": "L2 (n2c2 rerun, H1 only)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": str(DATA / "n2c2_corpus.parquet"),
        "n_corpus_rows": int(n_corpus),
        "n_patients": int(len(np.unique(pid))),
        "seed": wr.SEED, "boot_B": wr.BOOT_B, "boot_seed": wr.BOOT_SEED,
        "pools": {k: {"m": v[0], "N": v[1]} for k, v in POOLS.items()},
        "arm_state_files": {a: str(p) for a, p in ARM_FILES.items()},
        "h2_not_run": ("H2 needs the UMLS concept-bag observable space; the "
                       "Metathesaurus download has not landed (addendum)."),
    }

    t0 = time.time()
    sig: Dict[str, float] = {}
    for arm, Z in arms.items():
        s = wr.residual_sigma(Z, pid)
        sig[arm] = s["sigma_per_coordinate"]
        meta.setdefault("residual_matched_sigma", {})[arm] = s
        print(f"[sigma] {arm}: {s}", flush=True)
    timings["residual_sigma_seconds"] = round(time.time() - t0, 1)

    t0 = time.time()
    if args.from_parquet:
        df1 = pd.read_parquet(parquet_path)
        timings["design1_seconds"] = "reused existing parquet"
    else:
        df1 = wr.run_design1(arms, pid, sig, args.jobs)
        timings["design1_seconds"] = round(time.time() - t0, 1)
        df1.to_parquet(parquet_path, index=False)

    summ = wr.summarise_design1(df1)
    summ["verdicts"] = wr.h1_verdicts(summ["cells"])
    summ.update(meta)
    dev = [{
        "item": "H2 (cross-space weight consistency)",
        "status": "not run",
        "reason": ("the n2c2 observable space is UMLS concept bags via "
                   "QuickUMLS; the Metathesaurus download has not landed. "
                   "H1 does not depend on it (addendum)."),
    }, {
        "item": "pool level 'm=all'",
        "status": (f"run at m={M_ALL}, N={POOLS['mall'][1]}"
                   f"{', K in ' + str(mall_k) if mall_k else ''}"
                   if "mall" in POOLS else "not fitted in this run (--pools)"),
        "pre_registered": "m=all=1,263 donors, N=500, full K grid {2,5,10}",
        "deviation": (
            "m=all run at N=50 and K=5 only (8 cells: 2 w* x 2 sigma x 2 arms) "
            "instead of N=500 over the full K grid (24 cells)"
            if (mall_k == (5,) and POOLS.get("mall", (0, 0))[1] == 50) else None),
        "reason": (
            "Jun, mid-run: the m=all pool is a DIAGNOSTIC CONTROL -- it "
            "separates retrieval failure from geometry failure -- not the "
            "method. The method is the retrieval-reduced pool (m=50), so the "
            "control does not need headline-grade precision. Measured cost of "
            "the pre-registered version: ~50 s per m=1263 SLSQP solve, 512-1088 "
            "s per 500-replicate cell at 80 workers, ~5 h for the 24 cells. The "
            "m=50 decision cells and the oracle gate are untouched (N=500, full "
            "K grid, as pre-registered)."),
        "superseded_partial_run_at_N500": {
            "note": ("the first attempt ran the pre-registered N=500 m=all "
                     "cells until it was stopped at Jun's instruction; "
                     "per-replicate rows were never written (the runner "
                     "persists the parquet only after all cells), so only the "
                     "logged cell means survive. Reported as observed, without "
                     "CIs, and not used for any verdict."),
            "qwen3_mean_l1": {
                "K2_dirichlet_sigma0": 0.0000, "K2_dirichlet_sigma_res": 0.2654,
                "K2_spiky_sigma0": 0.0000, "K2_spiky_sigma_res": 0.2709,
                "K5_dirichlet_sigma0": 0.0001, "K5_dirichlet_sigma_res": 0.5627,
                "K5_spiky_sigma0": 0.0000, "K5_spiky_sigma_res": 0.3239,
                "K10_dirichlet_sigma0": 0.0000, "K10_dirichlet_sigma_res": 0.8957,
                "K10_spiky_sigma0": 0.0002},
            "biolord_mean_l1": "not reached before the stop",
        },
    }, {
        "item": "residual-matched sigma",
        "definition": ("median L2 residual of real m=50 SC fits (500 real "
                       "records, seed 0, 50 nearest other-patient records by "
                       "cosine) / sqrt(d) -> per-coordinate sigma; realised "
                       "values in residual_matched_sigma"),
        "reason": "the pre-registration delegates the operationalisation",
    }]
    if POOLS.get("mall", (0, 500))[1] != 500:
        dev[1]["deviation"] = (
            f"N reduced from the pre-registered 500 to {POOLS['mall'][1]} for "
            "the m=all cells only")
    summ["deviations"] = dev

    # retrieval diagnostic asked for by name: does the m=50 cosine pool contain
    # the true donors more often than it did in the invalid digit-string run?
    diag = {}
    for arm in ARM_FILES:
        for K in wr.K_GRID:
            for wstar in wr.WSTAR_GRID:
                g = df1[(df1.arm == arm) & (df1.K == K) & (df1.wstar == wstar)
                        & (df1["pool"] == "m50") & (df1.sigma_name == "sigma0")]
                if len(g):
                    diag[f"{arm}_K{K}_{wstar}"] = {
                        "mean_true_donors_in_pool": float(g.n_true_in_pool.mean()),
                        "frac_all_K_in_pool": float((g.n_true_in_pool == K).mean()),
                        "K": int(K)}
    summ["pool_contains_true_donors_m50_sigma0"] = diag
    summ["timings"] = timings
    summ["wall_clock_seconds"] = round(time.time() - t_start, 1)

    summary_path.write_text(json.dumps(summ, indent=2, default=wr._jsonable))
    print(json.dumps(summ["verdicts"], indent=2, default=wr._jsonable), flush=True)
    print(json.dumps(diag, indent=2), flush=True)
    print(f"[L2] total {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
