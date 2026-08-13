"""
Diagnostic for L2's hard gate: pool = the true K donors, sigma = 0.

The pre-registration makes `mean L1 <= 0.01` in this cell a gate rather than a
finding: "if it fails, the solver or the design is broken".  This script
separates those two, because the answer changes what the whole L2 result means:

* **solver**: the argmin really is `w*` but SLSQP stops short of it.  Detected
  by re-solving with a much tighter `ftol` / higher `maxiter`, and by warm-
  starting AT `w*` -- if the objective at the returned point is no better than at
  `w*` while `w` is far from it, the optimiser stopped on a flat direction.
* **design/geometry**: the donor latents are so near-collinear that many `w` give
  a numerically identical fit, so `w*` is not identified even at sigma=0.
  Detected by the Gram condition number and by the residual gap between the
  returned point and `w*`.

Writes latent/results/oracle_gate_diag.json.  Reads only the L1 npz files.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "src"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.synthctl.fit import solve_simplex_qp  # noqa: E402
import encode as enc  # noqa: E402

sys.path.insert(0, str(_LATENT / "scripts"))
from run_weight_recovery import (ARM_FILES, K_GRID, WSTAR_GRID,  # noqa: E402
                                 draw_replicates)

N = 200          # subset of the pre-registered 500: this is a diagnostic
OUT = _LATENT / "results" / "oracle_gate_diag.json"

#: the n2c2 rerun reuses this diagnostic unchanged, on its own states
N2C2_ARM_FILES = {"qwen3": _LATENT / "data" / "states_n2c2_qwen3.npz",
                  "biolord": _LATENT / "data" / "states_n2c2_biolord.npz"}
N2C2_OUT = _LATENT / "results" / "n2c2_oracle_gate_diag.json"


def main() -> int:
    import argparse

    global ARM_FILES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("trial", "n2c2"), default="trial")
    args = ap.parse_args()
    if args.corpus == "n2c2":
        ARM_FILES, OUT = N2C2_ARM_FILES, N2C2_OUT

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for arm, path in ARM_FILES.items():
        Z = np.ascontiguousarray(enc.load_states(path)["X"], dtype=np.float64)
        n_corpus = Z.shape[0]
        for K in K_GRID:
            for wstar in WSTAR_GRID:
                D, W = draw_replicates(n_corpus, K, wstar, 500)
                l1_def, l1_tight, l1_warm = [], [], []
                f_def, f_star, cond, mincos = [], [], [], []
                for r in range(N):
                    A, w = Z[D[r]], W[r]
                    x = w @ A
                    G = A @ A.T
                    def f(v):
                        return float(v @ (G @ v) - 2.0 * (v @ (A @ x)) + x @ x)
                    s = solve_simplex_qp(A, x)
                    st = solve_simplex_qp(A, x, ftol=1e-14, maxiter=5000)
                    sw = solve_simplex_qp(A, x, w0=w.copy())
                    l1_def.append(float(np.abs(s.w - w).sum()))
                    l1_tight.append(float(np.abs(st.w - w).sum()))
                    l1_warm.append(float(np.abs(sw.w - w).sum()))
                    f_def.append(max(s.sse, 0.0))
                    f_star.append(max(f(w), 0.0))
                    cond.append(float(np.linalg.cond(G)))
                    C = A @ A.T
                    iu = np.triu_indices(K, 1)
                    mincos.append(float(C[iu].min()))
                rows.append({
                    "arm": arm, "K": K, "wstar": wstar, "N": N,
                    "mean_l1_default": float(np.mean(l1_def)),
                    "mean_l1_ftol1e-14_maxiter5000": float(np.mean(l1_tight)),
                    "mean_l1_warmstart_at_wstar": float(np.mean(l1_warm)),
                    "mean_sse_at_solution": float(np.mean(f_def)),
                    "mean_sse_at_wstar": float(np.mean(f_star)),
                    "median_gram_condition_number": float(np.median(cond)),
                    "max_gram_condition_number": float(np.max(cond)),
                    "min_pairwise_donor_cosine_mean": float(np.mean(mincos)),
                })
                print(rows[-1], flush=True)
    out = {"stage": "L2 oracle-gate diagnostic",
           "corpus": args.corpus,
           "arm_state_files": {a: str(p) for a, p in ARM_FILES.items()},
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "note": ("donor latents are unit-norm, so the Gram IS the pairwise "
                    "cosine matrix; `min_pairwise_donor_cosine_mean` is the "
                    "mean over replicates of the LEAST similar donor pair"),
           "cells": rows,
           "wall_clock_seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
