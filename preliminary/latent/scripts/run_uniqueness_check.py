"""
Is the m=all EXACT weight recovery an identified optimum or solver bias?

Context: latent/results/n2c2_weight_recovery_summary.json reports mean L1 ~2.3e-5
for K=5, pool=m=all=1263, sigma=0 (qwen3).  Abadie & L'Hour (JASA 2021) warn that
with N donors > d dimensions the exact-fit set on the simplex can be a POLYTOPE,
so an exact hit on w* could be an artefact of the SLSQP path from the uniform
1/m warm start.

Two independent probes, on the SAME pseudo-targets the headline run used
(seed 0, K=5, Dirichlet(1) and spiky-0.9, sigma=0, reps 0..9 of each family):

  1. GEOMETRY (exact, LP).  The exact-fit set is
         P = { w : A' w = z*,  1'w = 1,  w >= 0 }.
     - rank of the equality system -> dimension of the AFFINE solution set
       (nonnegativity dropped), i.e. the Abadie/L'Hour object.
     - per-coordinate ranges over P by linprog/HiGHS: max w_j for every j, then
       min w_j for the coordinates whose max is above tolerance.  All ranges 0
       => P is the single point w*.
     - hull position of z*: number of donors that can carry positive weight, and
       an explicit supporting hyperplane (LP) if one exists.

  2. SOLVER BIAS (empirical).  Rerun the project solver
     (mimic/src/synthctl/fit.py::solve_simplex_qp) from 10 RANDOM Dirichlet(1)
     warm starts instead of the uniform 1/m default, and measure the scatter.

  3. m=50 control: the same two probes at the retrieval-reduced pool.

Writes latent/results/uniqueness_check.json.  Nothing else on disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

_LATENT = Path("/data/wang/junh/githubs/latent-synthetic-control/preliminary/latent")
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "scripts"), str(_LATENT / "src"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_weight_recovery as wr        # noqa: E402  (draw_replicates, SEED)
from src.synthctl.fit import solve_simplex_qp  # noqa: E402

ARM = "qwen3"
STATES = _LATENT / "data" / "states_n2c2_qwen3.npz"
OUT = _LATENT / "results" / "uniqueness_check.json"

K = 5
N_TARGETS_PER_FAMILY = 10          # reps 0..9 of each w* family -> 20 targets
M_ALL = 1263
M_50 = 50
N_RESTARTS = 10
RESTART_SEED = 12345
ZERO_TOL = 1e-9                    # a coordinate range above this is "nonzero"
N_JOBS = 96

_G: dict = {}


# ---------------------------------------------------------------------------
# pseudo-targets: recreated bit-for-bit from run_weight_recovery.draw_replicates
# ---------------------------------------------------------------------------

def build_targets(Z: np.ndarray):
    n_corpus = Z.shape[0]
    targets = []
    for wstar in wr.WSTAR_GRID:                      # ("dirichlet", "spiky")
        D, W = wr.draw_replicates(n_corpus, K, wstar, 500, seed=wr.SEED)
        for rep in range(N_TARGETS_PER_FAMILY):
            donors, w = D[rep], W[rep]
            z = w @ Z[donors]                        # sigma = 0
            sims = Z @ z
            order = np.argsort(-sims, kind="stable")
            targets.append({
                "id": f"{wstar}_rep{rep}",
                "wstar": wstar, "rep": int(rep),
                "donors": donors.copy(), "w": w.copy(), "z": z,
                "rows_mall": order[:M_ALL].copy(),
                "rows_m50": order[:M_50].copy(),
            })
    return targets


def _pool(tgt, which):
    return tgt["rows_mall"] if which == "mall" else tgt["rows_m50"]


def _eq_system(Z, tgt, which):
    rows = _pool(tgt, which)
    A = Z[rows]                                     # (m, d)
    m = A.shape[0]
    Aeq = np.vstack([A.T, np.ones((1, m))])         # (d+1, m)
    beq = np.concatenate([tgt["z"], [1.0]])
    return rows, A, Aeq, beq


def _support_pos(tgt, rows):
    """Positions of the K true donors inside the pool (-1 if absent)."""
    lut = {int(r): i for i, r in enumerate(rows)}
    return np.array([lut.get(int(dj), -1) for dj in tgt["donors"]])


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

def _init():
    Z = np.ascontiguousarray(np.load(STATES)["X"], dtype=np.float64)
    _G["Z"] = Z
    _G["targets"] = build_targets(Z)
    _G["cache"] = {}


def _sys_cached(ti, which):
    key = (ti, which)
    if key not in _G["cache"]:
        _G["cache"].clear()
        _G["cache"][key] = _eq_system(_G["Z"], _G["targets"][ti], which)
    return _G["cache"][key]


def _lp_coord(args):
    """max (sense=+1) or min (sense=-1) of w_j over the exact-fit polytope."""
    ti, which, j, sense = args
    rows, A, Aeq, beq = _sys_cached(ti, which)
    m = Aeq.shape[1]
    c = np.zeros(m)
    c[j] = -1.0 if sense > 0 else 1.0
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0.0, None)] * m, method="highs")
    val = float(-r.fun) if (r.status == 0 and sense > 0) else (
        float(r.fun) if r.status == 0 else float("nan"))
    return (ti, which, int(j), int(sense), val, int(r.status))


def _restart(args):
    """One solve_simplex_qp run from a given warm start."""
    ti, which, k = args           # k = -1 -> the uniform 1/m default
    Z = _G["Z"]
    tgt = _G["targets"][ti]
    rows = _pool(tgt, which)
    A = Z[rows]
    m = A.shape[0]
    if k < 0:
        w0 = None
    else:
        rng = np.random.default_rng([RESTART_SEED, ti, 0 if which == "mall" else 1, k])
        w0 = rng.dirichlet(np.ones(m))
    t0 = time.time()
    sol = solve_simplex_qp(A, tgt["z"], w0=w0)
    dt = time.time() - t0
    full = np.zeros(Z.shape[0]); full[rows] = sol.w
    star = np.zeros(Z.shape[0]); star[tgt["donors"]] = tgt["w"]
    return {"ti": ti, "which": which, "k": int(k),
            "l1_to_wstar": float(np.abs(full - star).sum()),
            "sse": float(sol.sse),
            "residual_l2": float(np.sqrt(max(sol.sse, 0.0))),
            "status": int(sol.status), "nit": int(sol.nit),
            "simplex_violation": float(sol.simplex_violation),
            "seconds": dt, "w_full": full}


def _geometry(args):
    """Per-target geometry: rank, aggregate off-support LP, hull position."""
    ti, which = args
    Z = _G["Z"]
    tgt = _G["targets"][ti]
    rows, A, Aeq, beq = _eq_system(Z, tgt, which)
    m = A.shape[0]
    pos = _support_pos(tgt, rows)
    out = {"ti": ti, "which": which, "m": int(m),
           "n_true_donors_in_pool": int((pos >= 0).sum())}

    rank = int(np.linalg.matrix_rank(Aeq))
    out["rank_eq_system"] = rank
    out["affine_solution_dim"] = int(m - rank)   # nonnegativity DROPPED

    # strict convexity of the QP: G = A A' PD <=> the argmin is unique for ANY
    # target, independent of the exact-fit geometry.  m > d forces G singular.
    ev = np.linalg.eigvalsh(A @ A.T)
    out["gram_min_eig"] = float(ev[0])
    out["gram_max_eig"] = float(ev[-1])
    out["gram_numeric_rank"] = int((ev > ev[-1] * 1e-10).sum())
    out["qp_strictly_convex"] = bool(ev[0] > ev[-1] * 1e-10)

    # feasibility of the exact fit at all (m=50 is expected infeasible)
    r0 = linprog(np.zeros(m), A_eq=Aeq, b_eq=beq,
                 bounds=[(0.0, None)] * m, method="highs")
    out["exact_fit_feasible"] = bool(r0.status == 0)
    out["exact_fit_lp_status"] = int(r0.status)

    if out["exact_fit_feasible"]:
        wl = np.asarray(r0.x)
        out["lp_vertex_residual_l2"] = float(np.linalg.norm(A.T @ wl - tgt["z"]))
        # aggregate certificate: max total weight OFF the true support
        c = np.ones(m)
        c[pos[pos >= 0]] = 0.0
        r1 = linprog(-c, A_eq=Aeq, b_eq=beq, bounds=[(0.0, None)] * m,
                     method="highs")
        out["max_offsupport_mass"] = (float(-r1.fun) if r1.status == 0
                                      else float("nan"))
        out["offsupport_lp_status"] = int(r1.status)

    # hull position of z*: explicit supporting hyperplane u with
    #   <a_j - z*, u> <= 0 for every donor j, normalised <cbar - z*, u> = -1.
    # feasible => z* is on the boundary of conv(A).
    d = A.shape[1]
    Aub = A - tgt["z"]
    cbar = A.mean(axis=0) - tgt["z"]
    r2 = linprog(np.zeros(d), A_ub=Aub, b_ub=np.zeros(m),
                 A_eq=cbar.reshape(1, -1), b_eq=[-1.0],
                 bounds=[(None, None)] * d, method="highs")
    out["supporting_hyperplane_found"] = bool(r2.status == 0)
    if r2.status == 0:
        u = np.asarray(r2.x)
        out["max_donor_margin"] = float((Aub @ u).max())   # <= 0 => supports
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    import multiprocessing as mp

    t_start = time.time()
    _init()
    Z, targets = _G["Z"], _G["targets"]
    n_corpus, d = Z.shape
    ctx = mp.get_context("fork")

    def run(fn, tasks, jobs=N_JOBS):
        with ctx.Pool(jobs, initializer=_init) as p:
            return p.map(fn, tasks, chunksize=1)

    # ---- 1. geometry -----------------------------------------------------
    t0 = time.time()
    geo_tasks = [(ti, which) for which in ("mall", "m50")
                 for ti in range(len(targets))]
    geo = {(g["ti"], g["which"]): g for g in run(_geometry, geo_tasks, 40)}
    print(f"[geometry] {time.time()-t0:.1f}s", flush=True)

    # ---- 2. per-coordinate LP ranges -------------------------------------
    # max first for every coordinate; min only where the max is above tolerance
    # (w >= 0, so max_j == 0 forces min_j == 0).
    lp_ranges = {}
    for which in ("mall", "m50"):
        feas = [ti for ti in range(len(targets)) if geo[(ti, which)]["exact_fit_feasible"]]
        if not feas:
            continue
        m = M_ALL if which == "mall" else M_50
        t0 = time.time()
        tasks = [(ti, which, j, +1) for ti in feas for j in range(m)]
        maxes = run(_lp_coord, tasks)
        print(f"[lp-max {which}] {len(tasks)} LPs in {time.time()-t0:.1f}s", flush=True)
        mx = {ti: np.full(m, np.nan) for ti in feas}
        bad = 0
        for ti, _w, j, _s, val, st in maxes:
            mx[ti][j] = val
            bad += (st != 0)
        need = [(ti, which, j, -1) for ti in feas
                for j in np.where(mx[ti] > ZERO_TOL)[0]]
        t0 = time.time()
        mins = run(_lp_coord, need) if need else []
        print(f"[lp-min {which}] {len(need)} LPs in {time.time()-t0:.1f}s", flush=True)
        mn = {ti: np.zeros(m) for ti in feas}
        for ti, _w, j, _s, val, st in mins:
            mn[ti][j] = val
            bad += (st != 0)
        lp_ranges[which] = {"max": mx, "min": mn, "n_lp_failed": int(bad),
                            "feasible_targets": feas}

    # ---- 3. warm-start restarts ------------------------------------------
    t0 = time.time()
    res_tasks = [(ti, which, k) for which in ("mall", "m50")
                 for ti in range(len(targets)) for k in range(-1, N_RESTARTS)]
    restarts = run(_restart, res_tasks)
    print(f"[restarts] {len(res_tasks)} solves in {time.time()-t0:.1f}s", flush=True)

    # ---- assemble --------------------------------------------------------
    per_target = []
    for ti, tgt in enumerate(targets):
        rec = {"id": tgt["id"], "wstar": tgt["wstar"], "rep": tgt["rep"],
               "true_donor_rows": tgt["donors"].tolist(),
               "wstar_weights": tgt["w"].tolist()}
        for which in ("mall", "m50"):
            g = dict(geo[(ti, which)]); g.pop("ti"); g.pop("which")
            blk = dict(g)

            lr = lp_ranges.get(which)
            if lr and ti in lr["feasible_targets"]:
                mx, mn = lr["max"][ti], lr["min"][ti]
                rng_ = mx - mn
                rows = _pool(tgt, which)
                pos = _support_pos(tgt, rows)
                on = pos[pos >= 0]
                off = np.setdiff1d(np.arange(len(rows)), on)
                blk["lp_ranges"] = {
                    "n_coords": int(len(rows)),
                    "n_coords_range_gt_tol": int((rng_ > ZERO_TOL).sum()),
                    "zero_tol": ZERO_TOL,
                    "max_range_over_all_coords": float(rng_.max()),
                    "max_range_offsupport": float(rng_[off].max()),
                    "max_range_onsupport": float(rng_[on].max()) if len(on) else None,
                    "max_upper_bound_offsupport": float(mx[off].max()),
                    "n_donors_that_can_carry_weight":
                        int((mx > ZERO_TOL).sum()),
                    "onsupport_lp_bounds": [
                        {"donor_row": int(rows[j]), "wstar": float(tgt["w"][i]),
                         "lp_min": float(mn[j]), "lp_max": float(mx[j])}
                        for i, j in enumerate(pos) if j >= 0
                    ],
                    "polytope_is_single_point": bool((rng_ > ZERO_TOL).sum() == 0),
                }
            rs = [r for r in restarts if r["ti"] == ti and r["which"] == which]
            rs.sort(key=lambda r: r["k"])
            Wm = np.array([r["w_full"] for r in rs])
            pw = [float(np.abs(Wm[a] - Wm[b]).sum())
                  for a in range(len(rs)) for b in range(a + 1, len(rs))]
            blk["restarts"] = {
                "n_starts": len(rs),
                "uniform_start_l1_to_wstar":
                    float([r for r in rs if r["k"] == -1][0]["l1_to_wstar"]),
                "random_start_l1_to_wstar":
                    [float(r["l1_to_wstar"]) for r in rs if r["k"] >= 0],
                "max_l1_to_wstar": float(max(r["l1_to_wstar"] for r in rs)),
                "mean_l1_to_wstar": float(np.mean([r["l1_to_wstar"] for r in rs])),
                "max_pairwise_l1": float(max(pw)),
                "max_residual_l2": float(max(r["residual_l2"] for r in rs)),
                "n_not_converged": int(sum(r["status"] != 0 for r in rs)),
                "max_simplex_violation":
                    float(max(r["simplex_violation"] for r in rs)),
                "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
            }
            blk.pop("m", None)
            rec[which] = blk
        per_target.append(rec)

    def agg(which):
        b = [r[which] for r in per_target]
        o = {
            "n_targets": len(b),
            "m": M_ALL if which == "mall" else M_50,
            "d": int(d),
            "n_exact_fit_feasible": int(sum(x["exact_fit_feasible"] for x in b)),
            "rank_eq_system": sorted({x["rank_eq_system"] for x in b}),
            "affine_solution_dim": sorted({x["affine_solution_dim"] for x in b}),
            "n_supporting_hyperplane_found":
                int(sum(x["supporting_hyperplane_found"] for x in b)),
            "gram_numeric_rank": sorted({x["gram_numeric_rank"] for x in b}),
            "n_qp_strictly_convex": int(sum(x["qp_strictly_convex"] for x in b)),
            "gram_min_eig_range": [float(min(x["gram_min_eig"] for x in b)),
                                   float(max(x["gram_min_eig"] for x in b))],
            "restarts": {
                "max_pairwise_l1_over_targets":
                    float(max(x["restarts"]["max_pairwise_l1"] for x in b)),
                "max_l1_to_wstar_over_targets":
                    float(max(x["restarts"]["max_l1_to_wstar"] for x in b)),
                "mean_l1_to_wstar_over_targets":
                    float(np.mean([x["restarts"]["mean_l1_to_wstar"] for x in b])),
                "max_residual_l2": float(max(x["restarts"]["max_residual_l2"] for x in b)),
                "n_not_converged": int(sum(x["restarts"]["n_not_converged"] for x in b)),
                "mean_seconds_per_solve":
                    float(np.mean([x["restarts"]["mean_seconds"] for x in b])),
            },
        }
        lp = [x["lp_ranges"] for x in b if "lp_ranges" in x]
        if lp:
            o["lp_ranges"] = {
                "n_targets_with_lp": len(lp),
                "n_lp_failed": int(lp_ranges[which]["n_lp_failed"]),
                "total_coords_probed": int(sum(x["n_coords"] for x in lp)),
                "n_coords_with_nonzero_range": int(sum(x["n_coords_range_gt_tol"] for x in lp)),
                "max_range_over_everything": float(max(x["max_range_over_all_coords"] for x in lp)),
                "max_offsupport_upper_bound": float(max(x["max_upper_bound_offsupport"] for x in lp)),
                "n_donors_that_can_carry_weight":
                    sorted({x["n_donors_that_can_carry_weight"] for x in lp}),
                "all_targets_single_point":
                    bool(all(x["polytope_is_single_point"] for x in lp)),
            }
            o["max_offsupport_mass"] = float(max(x["max_offsupport_mass"] for x in b
                                                 if "max_offsupport_mass" in x))
        return o

    summary = {
        "question": ("does the m=all exact weight recovery reflect a UNIQUE "
                     "optimum, or a lucky pick from an exact-fit polytope "
                     "(Abadie & L'Hour JASA 2021; Eslami arXiv 2607.25074)?"),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": ARM, "states": str(STATES),
        "corpus_rows": int(n_corpus), "d": int(d), "K": K, "sigma": 0.0,
        "pseudo_targets": (f"seed {wr.SEED}, K={K}, reps 0..{N_TARGETS_PER_FAMILY-1} "
                           f"of each of {wr.WSTAR_GRID} (spike={wr.SPIKE}), "
                           "rebuilt via run_weight_recovery.draw_replicates"),
        "n_pseudo_targets": len(targets),
        "solver": "mimic/src/synthctl/fit.py::solve_simplex_qp (SLSQP)",
        "lp": "scipy.optimize.linprog method='highs'",
        "n_restarts_per_target": N_RESTARTS,
        "restart_start": "Dirichlet(1) over the pool, seed [12345, target, pool, k]",
        "zero_tol": ZERO_TOL,
        "aggregate": {"mall": agg("mall"), "m50": agg("m50")},
        "per_target": per_target,
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    OUT.write_text(json.dumps(summary, indent=2, default=wr._jsonable))
    print(json.dumps(summary["aggregate"], indent=2, default=wr._jsonable), flush=True)
    print(f"[done] {time.time()-t_start:.1f}s -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
