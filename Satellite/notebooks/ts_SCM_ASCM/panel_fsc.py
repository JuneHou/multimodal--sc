"""panel_fsc — drive Okano & Kurisu's FSC R code on the biweekly latent panel.

Their reference implementation at `FSC_DIR` runs end-to-end in the local R
(/data/wang/junh/envs/rsynth, R 4.3.3) once quadprog, cubicBsplines, Rearrangement,
Matrix and the MASS/quantreg chain are installed — verified by smoke-testing their
UNMODIFIED `main_functions.R` on their own published data:

    service.R  (covariance outcomes)  FSCM -> 3 non-zero weights, sum(w)=1, min(w)=0,
                                      pre-treatment fit 39.3429; FSCM_aug_covmat and
                                      cross_val_covmat run; Matrix::nearPD ok
    mortality.R (quantile outcomes)   FSCM -> sum(w)=1; FSCM_aug (K=50, B-splines) and
                                      cross_val run; modif() monotone and in bounds

So **their R code is the estimator here**, called through `fsc_bridge.R`; nothing in this
module reimplements FSC. Python only builds the outcome blocks (`panel_repr`), marshals
them, and computes the comparison metrics our earlier notebooks use.

Two facts the smoke test settled, both worth knowing before reading results:

1. **Their augmented weights DO sum to 1** (measured 1.000000 on both of their datasets)
   even though no constraint imposes it. The reason is their centring step
   (main_functions.R L76-85 / L272-281): after demeaning across control units at each
   pre-period, the rows of `r_0dot` sum to zero over units, so the augmentation term
   contributes nothing to the sum. It is a property of the centring, not a constraint —
   which is a real difference from our `panel_ascm`, where Σw=1 is imposed via KKT.
2. **The augmentation extrapolates hard.** On their own service data the augmented
   weights reach min(w) = -42.6 with 11 of 22 negative. Large negative weights are
   normal for this estimator, not a bug — and are exactly why the outcome must be
   projected back (`modif` for quantiles, `nearPD` for covariance matrices).
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import panel_repr as pr
from panel_lib import TS

FSC_DIR = Path("/data/wang/junh/githubs/FSC")          # read-only, never modified
RSCRIPT = Path("/data/wang/junh/envs/rsynth/bin/Rscript")
RBIN = Path("/data/wang/junh/envs/rsynth/bin")          # gfortran etc. live here
BRIDGE = TS / "fsc_bridge.R"

# their optimise() intervals: service.R L61 used c(0,1); mortality.R L148 used c(0,10)
LAM_INTERVAL = {"covmat": (0.0, 1.0), "bspline": (0.0, 10.0)}
K_BSPLINE = 50                                          # mortality.R L141
BLAS_THREADS = 8            # this is a shared machine; see RBridge.run


class RBridge:
    """Marshal an outcome block to `fsc_bridge.R` and read the results back.

    The block is written as raw float64 in C-order with shape
    (n_groups, n_units, n_periods, M); R reads it with readBin and reshapes so that M
    varies fastest, then rebuilds their `func_vals_list` (unit -> period -> vector) with
    unit 1 = the treated site.
    """

    def __init__(self, workdir=None):
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fsc_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def _write(self, block, meta, tag):
        d = self.workdir
        (d / f"{tag}_data.bin").write_bytes(
            np.ascontiguousarray(block, dtype=np.float64).tobytes())
        lines = [f"{k}={v}" for k, v in meta.items()]
        (d / f"{tag}_meta.txt").write_text("\n".join(lines) + "\n")
        return d / f"{tag}_meta.txt", d / f"{tag}_data.bin", d / tag

    def run(self, block, op, method, T_0, tag, lambdas=None, **kw):
        n_groups, n_units, n_periods, M = block.shape
        a, b = LAM_INTERVAL[method]
        meta = {"fsc_dir": str(FSC_DIR), "op": op, "method": method,
                "n_groups": n_groups, "n_units": n_units, "n_periods": n_periods,
                "M": M, "T_0": T_0, "K": K_BSPLINE,
                "lam_a": kw.get("lam_a", a), "lam_b": kw.get("lam_b", b),
                "lambda": kw.get("lambda_", "cv"),
                "post_period": kw.get("post_period", T_0 + 1),
                # projection metadata (their modif / nearPD); n_grid=0 or n_gram=0
                # switches the corresponding block off
                "repr_kind": kw.get("repr_kind", "none"),
                "n_channels": kw.get("n_channels", 5),
                "n_grid": kw.get("n_grid", 0), "n_gram": kw.get("n_gram", 0),
                "low": kw.get("low", 0.0), "upp": kw.get("upp", 1.0)}
        mp, dp, op_prefix = self._write(block, meta, tag)
        if lambdas is not None:
            Path(f"{op_prefix}_lambdas.txt").write_text(
                "\n".join(f"{x:.10g}" for x in lambdas) + "\n")
        import os
        # Cap BLAS threading. R's `solve()` on the (M*T_0)² system otherwise fans out
        # across every core — measured at ~4700% CPU per process, which on a shared
        # 128-core box with other users drove the load average past 300 and made three
        # concurrent notebooks ~4x slower than running them one at a time. These
        # matrices are small enough that a handful of threads costs nothing.
        env = dict(os.environ, PATH=f"{RBIN}:{os.environ.get('PATH','')}",
                   OMP_NUM_THREADS=str(BLAS_THREADS),
                   OPENBLAS_NUM_THREADS=str(BLAS_THREADS),
                   MKL_NUM_THREADS=str(BLAS_THREADS))
        res = subprocess.run([str(RSCRIPT), str(BRIDGE), str(mp), str(dp),
                              str(op_prefix)],
                             capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"fsc_bridge.R failed:\n{res.stdout[-3000:]}\n"
                               f"{res.stderr[-3000:]}")
        out = {"log": res.stdout}
        for suffix in ("weights", "fit", "placebo", "lamgrid"):
            p = Path(f"{op_prefix}_{suffix}.csv")
            if p.exists():
                out[suffix] = pd.read_csv(p)
        return out


# ---------------------------------------------------------------- lambda search
# λ = 0 makes their solve(RᵀR + λI) singular: RᵀR has rank N_0 = 5 but size M·T_0, so the
# penalty is the only thing making it invertible. Their optimise() never evaluates an
# endpoint exactly, so they never hit this; an explicit grid does. Start strictly positive.
LAM_GRID = np.logspace(-6, 6, 25)
INTERIOR_TOL = 1e-3            # RELATIVE; their optimise() converges to ~1.2e-4 relative


def repr_meta(name, grids):
    """Projection metadata for a representation — which block gets `modif`, which gets
    `nearPD`. See Docs/8-27-fsc-fidelity.md Issue 2."""
    ng = len(grids)
    return {"quantile":  dict(repr_kind="quantile", n_grid=ng, n_gram=0),
            "gram":      dict(repr_kind="gram", n_grid=0, n_gram=15),
            "combined":  dict(repr_kind="combined", n_grid=ng, n_gram=15),
            "chip_mean": dict(repr_kind="none", n_grid=0, n_gram=0)}[name]


def lambda_search(block, method, T_0, tag, bridge, grid=LAM_GRID, n_groups=None, **kw):
    """Select λ by evaluating THEIR `cross_val` / `cross_val_covmat` on a log grid.

    Disclosed deviation from their `optimise()` on a linear interval — see
    Docs/8-27-fsc-fidelity.md Issue 1. The objective is theirs; only the search changes,
    because a linear interval chosen for their outcome scales put λ on a boundary in
    20 of 28 of our configurations, and the interiority of the optimum could then never
    be checked.

    Returns (lambda, tidy CV table, diagnostics dict).
    """
    # `n_groups` restricts the search to the first few groups. Their own scripts select
    # λ once for a single treated unit, so this is not a further shortcut; it keeps the
    # O((M·T_0)²) CV affordable for the high-M representations.
    sub = block if n_groups is None else block[:n_groups]
    res = bridge.run(sub, "lambda_grid", method, T_0, tag, lambdas=grid, **kw)
    cv = res["lamgrid"]
    tot = cv.groupby("lambda")["cv"].sum().sort_index()      # pooled over groups
    lam = float(tot.idxmin())
    lo, hi = float(grid.min()), float(grid.max())
    diag = {"lambda": lam,
            "at_floor": lam <= lo or (hi > 0 and lam / hi < INTERIOR_TOL and lam == lo),
            "at_ceiling": hi > 0 and (hi - lam) / hi < INTERIOR_TOL,
            "grid_lo": lo, "grid_hi": hi,
            "cv_at_opt": float(tot.min()), "n_grid": len(grid)}
    diag["interior"] = not (diag["at_ceiling"] or lam == lo)
    return lam, cv, diag


# ---------------------------------------------------------------- metrics
def synth(block, w, gi):
    """Synthetic outcome for group `gi` at every period: Σ_j w_j × donor_j.
    Mirrors their `t(weight) %*% control_matrix` (service.R L37-41)."""
    return np.tensordot(w, block[gi, 1:], axes=(0, 0))       # (n_periods, M)


def metrics(block, weights, groups, periods, T_0, label):
    """Per-group fit and holdout error, plus two reference predictors.

    RMSE is in the representation's own units, so it is NOT comparable across arms.
    `ratio_own_mean` is: it divides by the error of predicting the treated unit's own
    P01–P08 mean (an intercept-only model with no donors at all). Below 1 means the
    donors added something. `ratio_equal` uses the equal-weight donor average instead.
    """
    rows = []
    hold = list(range(T_0, min(10, len(periods))))           # P09, P10 -> idx 8,9
    tr = list(range(T_0))
    for gi, g in enumerate(groups):
        obs = block[gi, 0]                                   # (n_periods, M)
        own = obs[tr].mean(axis=0)                           # treated's own pre-mean
        eq = np.tensordot(np.full(block.shape[1] - 1, 1.0 / (block.shape[1] - 1)),
                          block[gi, 1:], axes=(0, 0))
        for est in ("fsc", "afsc"):
            w = weights.query("group == @gi + 1").sort_values("donor")[
                f"weight_{est}"].to_numpy()
            s = synth(block, w, gi)
            rms = lambda idx, pred: float(np.sqrt(np.mean((obs[idx] - pred) ** 2)))
            rows.append({
                "label": label, "estimator": est,
                "treatment_site_id": g[0], "group": gi + 1,
                "train_rmse": rms(tr, s[tr]),
                "holdout_rmse": rms(hold, s[hold]),
                "null_own_mean": rms(hold, own),
                "null_equal_weight": rms(hold, eq[hold]),
                "ratio_own_mean": rms(hold, s[hold]) / rms(hold, own),
                "ratio_equal": rms(hold, s[hold]) / rms(hold, eq[hold]),
                "sum_w": float(w.sum()), "min_w": float(w.min()),
                "n_negative": int((w < 0).sum()),
                "n_train_periods": len(tr), "n_holdout_periods": len(hold)})
    return pd.DataFrame(rows)


def effects(block, weights, groups, periods, T_0, label, est="fsc"):
    """Post-period gap ‖observed − synthetic‖ per group per period."""
    rows = []
    for gi, g in enumerate(groups):
        w = weights.query("group == @gi + 1").sort_values("donor")[
            f"weight_{est}"].to_numpy()
        s = synth(block, w, gi)
        for ti, q in enumerate(periods):
            rows.append({"label": label, "estimator": est,
                         "treatment_site_id": g[0], "period": f"P{q:02d}",
                         "period_index": ti + 1, "is_post": q >= 11,
                         "gap_norm": float(np.linalg.norm(block[gi, 0, ti] - s[ti])),
                         "gap_rmse": float(np.sqrt(np.mean(
                             (block[gi, 0, ti] - s[ti]) ** 2)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- placebo
def placebo_ranks(placebo_df, groups):
    """Abadie in-space placebo, their `placebo_covmat` output turned into ranks.

    Each group is 1 treated + 5 matched controls, so rotating every unit into the
    treated slot gives 6 gap magnitudes and the treated unit's rank is 1..6 — the
    smallest attainable per-site p-value is 1/6 = 0.167.

    Under the sharp null of no effect the rank is uniform on 1..6, so across the G
    groups the number achieving rank 1 is Binomial(G, 1/6). That combined test is what
    gives this design its power: with G = 10, 4 rank-1 sites give p = 0.070,
    5 give p = 0.015, 6 give p = 0.002 (exact binom.sf values).
    """
    from scipy.stats import binom
    rows = []
    for gi, g in enumerate(groups):
        sub = placebo_df.query("group == @gi + 1").sort_values(
            "magnitude", ascending=False).reset_index(drop=True)
        rank = int(sub.index[sub["is_treated"]][0]) + 1
        n = len(sub)
        rows.append({"treatment_site_id": g[0], "group": gi + 1, "rank": rank,
                     "n_units": n, "p_site": rank / n,
                     "treated_magnitude": float(
                         sub.loc[sub["is_treated"], "magnitude"].iloc[0]),
                     "max_control_magnitude": float(
                         sub.loc[~sub["is_treated"], "magnitude"].max())})
    df = pd.DataFrame(rows)
    n_units = int(df["n_units"].iloc[0])
    k = int((df["rank"] == 1).sum())
    G = len(df)
    df.attrs["combined"] = {
        "n_groups": G, "n_rank1": k, "p_per_site_null": 1.0 / n_units,
        "p_combined_binomial": float(binom.sf(k - 1, G, 1.0 / n_units))
        if k > 0 else 1.0}
    return df


def combined_line(pl_df):
    c = pl_df.attrs["combined"]
    return (f"placebo: {c['n_rank1']}/{c['n_groups']} sites rank 1 "
            f"(per-site null {c['p_per_site_null']:.3f}) -> "
            f"combined binomial p = {c['p_combined_binomial']:.4f}")


# ---------------------------------------------------------------- one-call driver
def run_placebo(arm, method, label, name, sensor, post_period_q=11, lambda_=None):
    """Their `placebo` / `placebo_covmat` on an arm already built by `run_arm`.

    `post_period` in their code is a 1-based index into the period list, not a period
    label — P15 is missing panel-wide, so the two differ after P14. Reuses the arm's
    block, so nothing is rebuilt.
    """
    periods, T_0 = arm["periods"], arm["T_0"]
    assert post_period_q in periods, (post_period_q, periods)
    idx = periods.index(post_period_q) + 1
    if lambda_ is None:                      # reuse the arm's once-selected lambda
        lambda_ = float(arm["weights"]["lambda"].iloc[0])
    res = arm["bridge"].run(arm["block"], "placebo", method, T_0,
                            f"{label}_{name}_{sensor}_pl{post_period_q}",
                            post_period=idx, lambda_=lambda_)
    df = placebo_ranks(res["placebo"], arm["groups"])
    df.insert(0, "label", label)
    df.insert(1, "sensor", sensor)
    df.insert(2, "period", f"P{post_period_q:02d}")
    return df


def run_scheme(panel, sensor, name, method, label, workdir=None, need=range(1, 11),
               grids=None, lam_grid=LAM_GRID, bridge=None, lam_groups=None):
    """Both validation schemes of notebooks 02–06, with λ chosen on a log grid.

        frozen_joint : fit P01–P08 (T_0=8), score P09 and P10 with the same weights
        expanding    : fit P01–P08 → predict P09;  REFIT P01–P09 (T_0=9) → predict P10

    Returns a tidy per-site table of prediction RMSE for FSC and augmented FSC (both
    unprojected and projected), against two reference predictors.
    """
    grids = pr.GRIDS_Q if grids is None else grids
    groups_all = pr.build_groups(panel)
    groups, dropped = pr.complete_groups(panel, sensor, groups_all, need=need)
    periods = pr.complete_periods(panel, sensor, groups)
    block, _ = pr.build_block(panel, sensor, name, groups, periods, grids=grids)
    br = bridge or RBridge(workdir)
    rk = repr_meta(name, grids)

    fits, lam_rows = {}, []
    for T_0 in (8, 9):
        tag = f"{label}_{name}_{sensor}_T{T_0}"
        lam, _cv, diag = lambda_search(block, method, T_0, tag, br, grid=lam_grid,
                                       n_groups=lam_groups, **rk)
        lam_rows.append({"label": label, "sensor": sensor, "repr": name,
                         "method": method, "T_0": T_0, **diag})
        res = br.run(block, "fit", method, T_0, tag, lambda_=lam, **rk)
        fits[T_0] = res
    LAM = pd.DataFrame(lam_rows)

    def wvec(res, gi, est):
        return (res["weights"].query("group == @gi + 1").sort_values("donor")
                [f"weight_{est}"].to_numpy())

    rows = []
    i09, i10 = periods.index(9), periods.index(10)
    for gi, g in enumerate(groups):
        obs = block[gi, 0]
        pre8 = list(range(8))
        own8 = obs[pre8].mean(axis=0)
        eqw = np.full(block.shape[1] - 1, 1.0 / (block.shape[1] - 1))
        eq = np.tensordot(eqw, block[gi, 1:], axes=(0, 0))
        rms = lambda a, b_: float(np.sqrt(np.mean((a - b_) ** 2)))
        for est in ("fsc", "afsc"):
            s8 = synth(block, wvec(fits[8], gi, est), gi)      # weights fit on P01–P08
            s9 = synth(block, wvec(fits[9], gi, est), gi)      # weights fit on P01–P09
            for scheme, ti, s in (("frozen_joint", i09, s8), ("frozen_joint", i10, s8),
                                  ("expanding", i09, s8), ("expanding", i10, s9)):
                rows.append({
                    "label": label, "sensor": sensor, "repr": name, "method": method,
                    "estimator": est, "scheme": scheme,
                    "treatment_site_id": g[0], "eval_period": f"P{periods[ti]:02d}",
                    "fit_window": "P01-08" if (scheme == "frozen_joint" or ti == i09)
                                  else "P01-09",
                    "pred_rmse": rms(obs[ti], s[ti]),
                    "null_own_mean": rms(obs[ti], own8),
                    "null_equal_weight": rms(obs[ti], eq[ti]),
                    "ratio_own_mean": rms(obs[ti], s[ti]) / rms(obs[ti], own8),
                    "train_rmse": float(np.sqrt(np.mean(
                        (obs[pre8] - s[pre8]) ** 2)))})
    return {"metrics": pd.DataFrame(rows), "lambda": LAM, "fits": fits,
            "block": block, "groups": groups, "dropped": dropped, "periods": periods,
            "bridge": br, "grids": grids, "repr_meta": rk}


def run_arm(panel, sensor, name, method, label, workdir=None, need=range(1, 11),
            grids=None, lambda_="cv_once"):
    """Build the block, run their FSCM + augmented FSC through R, return everything.

    `lambda_="cv_once"` selects λ by their `optimise()` on the first group and reuses it,
    mirroring their own scripts, which run the CV once and then hardcode the value.
    """
    grids = pr.GRIDS_Q if grids is None else grids
    groups_all = pr.build_groups(panel)
    groups, dropped = pr.complete_groups(panel, sensor, groups_all, need=need)
    periods = pr.complete_periods(panel, sensor, groups)
    block, scale = pr.build_block(panel, sensor, name, groups, periods, grids=grids)
    T_0 = 8
    br = RBridge(workdir)
    res = br.run(block, "fit", method, T_0, f"{label}_{name}_{sensor}",
                 lambda_=lambda_)
    m = metrics(block, res["weights"], groups, periods, T_0, label)
    e = effects(block, res["weights"], groups, periods, T_0, label)
    return {"block": block, "groups": groups, "dropped": dropped, "periods": periods,
            "T_0": T_0, "weights": res["weights"], "fit": res["fit"],
            "metrics": m, "effects": e, "block_scale": scale, "bridge": br,
            "grids": grids, "name": name, "method": method, "sensor": sensor,
            "label": label, "log": res["log"]}
