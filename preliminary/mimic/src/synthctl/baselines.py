"""
Phase 2 baselines.  Every one forecasts the state at ``T0+1`` and is scored by
exactly the same code as the synthetic control.

The list is deliberately unglamorous, because the honest question is not
"is synthetic control clever?" but "does it beat the obvious thing?".  In EHR
the obvious thing is **last value carried forward**: the patient looks about the
same next visit, and usually does.  LVCF is the baseline that matters; the rest
are there to locate the estimator between "does nothing" and "does what the
source repo did".

None of these is tuned.  There is no hyperparameter search anywhere in this file
and no baseline is weakened to flatter the estimator: `k=10` and the `MAX_K=1`
one-hot rule are both fixed a priori (the latter is literally what the source
repo `moa-clinical-rag` used).

Baselines
---------
    lvcf              the target's own state at T0
    nn1               the single nearest donor's state at T0+1  (source repo's rule)
    topk_mean         unweighted mean of the k nearest donors' states at T0+1
    cohort_mean       mean of ALL eligible donors' states at T0+1
    ridge             supervised ridge X[0:T0] -> X[T0+1], fit on the donor pool
    sc_random         m donors drawn at random, then the identical SLSQP fit

`ridge` is the "just fit a model" comparator.  Without it the table only
answers "does SC beat copying?"; every other baseline here copies or averages,
so nothing in the original set learned the pre-to-post map at all and the
question "does SC beat plain supervised regression?" could not be asked.  Its
absence was a genuine gap, not a stylistic one.

`sc_random` lives in `fit.py` (it is the estimator with a different donor
selection rule); it is registered here so the evaluation treats it as one more
row in the table.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .fit import Panel

#: Neighbourhood size for the unweighted-mean baseline.  Fixed a priori, not
#: tuned: 10 is the conventional "top-10 donors" figure Phase 1 already reported
#: its neighbour-overlap diagnostics at, so it is the value that was going to be
#: quoted whatever the answer turned out to be.
DEFAULT_K = 10


def lvcf(panel: Panel, i: int) -> np.ndarray:
    """Last value carried forward: predict `X_i[T0+1]` = `X_i[T0]`.

    Uses no donor and no fitting.  **This is the baseline that matters.**
    """
    return panel.target_last_state(i).copy()


def _nearest(panel: Panel, i: int, k: int) -> np.ndarray:
    donors = panel.donor_index(i)
    dist = panel.distances(i, donors)
    # stable so ties break by canonical subject_id order; bag distances are
    # heavily tied (Phase 1: ~100 distinct values over 748 donors), and an
    # unstable sort would make the "nearest donor" a function of BLAS ordering.
    return donors[np.argsort(dist, kind="stable")[:k]]


def nn1(panel: Panel, i: int) -> np.ndarray:
    """The single nearest donor's state at `T0+1`.

    A synthetic control with all its weight on one unit -- and exactly what the
    source repo did (`MAX_K = 1`).  Included so the table shows what the convex
    combination buys over the 1-sparse special case.
    """
    return panel.Y[_nearest(panel, i, 1)[0]].copy()


def topk_mean(panel: Panel, i: int, k: int = DEFAULT_K) -> np.ndarray:
    """Unweighted mean of the `k` nearest donors' states at `T0+1`.

    The ablation that removes the *fitting* while keeping the *neighbourhood*:
    if this matches the synthetic control, solving the QP bought nothing beyond
    picking the neighbourhood.
    """
    return panel.Y[_nearest(panel, i, k)].mean(axis=0)


def cohort_mean(panel: Panel, i: int) -> np.ndarray:
    """Mean of every eligible donor's state at `T0+1`.

    Removes the neighbourhood too: the population's average next visit.  Any
    method that cannot beat this is not using patient identity at all.
    """
    return panel.Y[panel.donor_index(i)].mean(axis=0)


# --------------------------------------------------------------------------
# Ridge: the supervised comparator.
# --------------------------------------------------------------------------

#: L2 penalty for the ridge baseline.  See `ridge`.
DEFAULT_RIDGE_ALPHA = 1.0


def _ridge_predict(
    Ppool: np.ndarray, Ypool: np.ndarray, p_query: np.ndarray, alpha: float,
    K: Optional[np.ndarray] = None, kq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-output ridge prediction in DUAL (kernel) form, with intercept.

    Solves ``min_W ||Y - Xc W||_F^2 + alpha ||W||_F^2`` on column-centred data
    and predicts `ybar + (x - xbar)' W`.  The dual identity
    `(Xc'Xc + aI)^-1 Xc' = Xc' (Xc Xc' + aI)^-1` is used because in this design
    the pre-period dimension `D = (T0+1)*d` (1170 for the bag encoder at T0=1)
    exceeds the number of donors `n` (748), so the `n x n` system is both
    smaller and better conditioned than the `D x D` one.

    `K` (= Xc Xc', centred) and `kq` (= Xc (x - xbar)) may be supplied
    pre-computed; `ridge_all` does so, sharing one Gram across all targets.
    """
    n = Ypool.shape[0]
    if K is None or kq is None:
        xbar = Ppool.mean(axis=0)
        Xc = Ppool - xbar
        K = Xc @ Xc.T
        kq = Xc @ (p_query - xbar)
    ybar = Ypool.mean(axis=0)
    Yc = Ypool - ybar
    # v = (K + alpha I)^-1 kq, then prediction = ybar + v' Yc.  One RHS, so the
    # cost is the O(n^3) factorisation, not O(n^2 d).
    v = np.linalg.solve(K + alpha * np.eye(n), kq)
    return ybar + v @ Yc


def ridge(panel: Panel, i: int, alpha: float = DEFAULT_RIDGE_ALPHA) -> np.ndarray:
    """Ridge regression from the pre-period block to the next visit.

    **The "just fit a model" comparator, and the one baseline the original set
    was missing.**  Every other baseline here either copies a single vector
    (`lvcf`, `nn1`) or averages a neighbourhood (`topk_mean`, `cohort_mean`);
    none of them *learns* the map `X[0:T0] -> X[T0+1]`.  Without a supervised
    regression in the table there is no way to tell whether synthetic control
    beats plain supervised learning, only whether it beats copying.

    Protocol, chosen to be exactly as fair as the synthetic control's:

    * fit on the **donor pool** for this target, i.e. every eligible patient
      except `i` (`panel.donor_index(i)`) -- the same leave-one-patient-out
      convention the SC fit uses, so the target never appears in its own
      training set;
    * inputs are the flattened pre-period block `P` (all of `t = 0..T0`),
      outputs the state at `T0+1`, multi-output over all `d` coordinates;
    * an intercept is fitted (both sides column-centred), which is what makes
      it a fair opponent for an SC *with* an intercept;
    * `alpha` is the L2 penalty, **fixed a priori, never tuned** -- see below.
      No inner CV, no sweep. Like every other baseline in this file it is a
      floor, not a tuned opponent.

    Args:
        panel: the donor panel.
        i:     target row index.
        alpha: L2 penalty.  1.0.

    Returns:
        (d,) forecast of the state at `T0+1`.
    """
    donors = panel.donor_index(i)
    return _ridge_predict(panel.P[donors], panel.Y[donors], panel.P[i],
                          float(alpha))


def ridge_all(panel: Panel, alpha: float = DEFAULT_RIDGE_ALPHA) -> np.ndarray:
    """`ridge` for every target, sharing one Gram matrix.  Returns (n, d).

    Mathematically identical to `[ridge(panel, i) for i in range(n)]` (asserted
    in `scripts/smoke_phase2_fixes.py`), but the `n x D` by `D x n` product is
    formed ONCE instead of `n` times, which is the whole cost at D=1170.  The
    per-target leave-one-out centring is then recovered from row/column sums of
    the full Gram, so the target really is excluded from its own fit.
    """
    n = panel.n
    P, Y = panel.P, panel.Y
    Gf = P @ P.T                       # (n, n), the only O(n^2 D) work
    out = np.empty((n, panel.d), dtype=np.float64)
    for i in range(n):
        idx = np.arange(n) != i
        Kd = Gf[np.ix_(idx, idx)]      # donors x donors, uncentred
        c = Gf[idx, i]                 # donors . target
        nd = n - 1
        # Centre the Gram and the query kernel using only row/col sums, i.e.
        # without ever re-touching the D-dimensional rows:
        #   K~ = K - u 1' - 1 u' + (sum K / nd^2) 11',   u = K.sum(1)/nd
        #   k~ = c - u - (c.sum()/nd) 1 + (sum K / nd^2) 1
        u = Kd.sum(axis=1) / nd
        gg = Kd.sum() / (nd * nd)
        Kc = Kd - u[:, None] - u[None, :] + gg
        kq = c - u - c.sum() / nd + gg
        out[i] = _ridge_predict(None, Y[idx], None, float(alpha), K=Kc, kq=kq)
    return out


#: name -> callable(panel, i) -> forecast.  `sc` and `sc_random` are produced by
#: `fit.py` and merged in by the runner, so they are absent here.
SIMPLE_BASELINES: Dict[str, object] = {
    "lvcf": lvcf,
    "nn1": nn1,
    "topk_mean": topk_mean,
    "cohort_mean": cohort_mean,
    "ridge": ridge,
}

#: Human-readable labels for the headline table.
LABELS: Dict[str, str] = {
    "sc": "SC (SLSQP simplex)",
    "sc_intercept": "SC + intercept mu",
    "sc_linf": "SC + L-inf (dense weights)",
    "lvcf": "LVCF (own state at T0)",
    "nn1": "1-NN donor (source repo MAX_K=1)",
    "topk_mean": f"mean of {DEFAULT_K} nearest donors",
    "cohort_mean": "cohort mean",
    "ridge": f"ridge regression (a={DEFAULT_RIDGE_ALPHA:g}, LOO)",
    "sc_random": "SC on RANDOM donors (placebo)",
}


def forecast_all(
    panel: Panel, name: str, k: int = DEFAULT_K,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> np.ndarray:
    """Run one simple baseline over every target.  Returns (n, d)."""
    fn = SIMPLE_BASELINES[name]
    if name == "topk_mean":
        return np.stack([fn(panel, i, k=k) for i in range(panel.n)])
    if name == "ridge":
        return ridge_all(panel, alpha=ridge_alpha)
    return np.stack([fn(panel, i) for i in range(panel.n)])


def all_simple_forecasts(
    panel: Panel, k: int = DEFAULT_K, names: Optional[List[str]] = None,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> Dict[str, np.ndarray]:
    names = list(SIMPLE_BASELINES) if names is None else names
    return {nm: forecast_all(panel, nm, k=k, ridge_alpha=ridge_alpha)
            for nm in names}
