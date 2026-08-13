"""
Phase 2: the synthetic-control estimator.

For target patient *i* with pre-period ``t in [0, T0]``, fit convex weights over
a donor pool of other patients so that the weighted donors reproduce *i*'s
pre-period visits::

    min_w  || X_i[0:T0] - sum_j w_j X_j[0:T0] ||^2     s.t.  w_j >= 0,  sum_j w_j = 1

and then forecast the held-out next visit as ``sum_j w_j X_j[T0+1]``.

Solved with ``scipy.optimize.minimize(method='SLSQP')``, which enforces the
simplex exactly (`cvxpy` is deliberately not installed in this project).  The
objective is reduced to its Gram form before it reaches the solver -- see
`solve_simplex_qp` -- so the cost per iteration is O(m^2) in the number of
donors rather than O(m * (T0+1) * d).  That is what makes an m=748 fit take ~10 s
instead of minutes.

**Read `overfitting` in MODULE.md before quoting a pre-period RMSE.**  At the
primary T0=1 the pre-period is two timesteps against up to 748 donors, so a
near-perfect pre-period fit is expected and carries no information about the
forecast.  Pre-period RMSE is emitted as a *diagnostic*, never as the validation
metric; `evaluate.py` measures the correlation between the two and reports it.

Two optional extensions, both **off by default** so the classical estimator
above is bit-for-bit unchanged unless asked for:

* an **intercept** ``mu^0`` (Wang, Xing & Ye Eq. 3, after Doudchenko & Imbens),
  so the weights mimic the *trajectory* instead of spending themselves matching
  a patient's chronic code *level*.  `fit_intercept=True`.  `mu` is
  unconstrained in sign and is profiled out analytically, preserving the Gram
  form -- see `_gram`.
* an **L-infinity (dense) penalty** on the weights (their Eq. 5),
  `penalty="linf"`, which caps the largest weight instead of accepting the
  sparsity the simplex mechanically produces -- see `solve_linf`, including why
  the composite penalty's `alpha` is not identified on the simplex.

Public API
----------
    Panel.build(ss, T0)                          -> Panel      (donor pool, once)
    Panel.rownorm_l2()                           -> Panel      (bag ablation)
    solve_simplex_qp(A, x, ..., fit_intercept)   -> SolveResult
    solve_linf(A, x, lam, alpha, ...)            -> SolveResult
    fit_target(panel, i, m, ...)                 -> TargetFit
    fit_all(panel, m, ...)                       -> List[TargetFit]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.state import StateSet  # noqa: E402

# --------------------------------------------------------------------------
# Solver constants.  Every one is recorded in MODULE.md; none is a silent
# default.
# --------------------------------------------------------------------------

#: SLSQP iteration cap.  Observed nit is 12-20 for every fit at every m, so this
#: is an order of magnitude of headroom, not a binding constraint.
SLSQP_MAXITER = 500

#: SLSQP convergence tolerance on the objective.  The objective is a squared
#: distance, so this is tight relative to the residuals we actually see
#: (post-period squared errors are O(1e-1) for medcpt, O(1e1) for raw bag).
SLSQP_FTOL = 1e-10

#: Weights below this count as "not a donor" for the effective-donor count.
#: 1e-4 = one part in 10^4 of the simplex; a donor at this weight moves the
#: forecast by less than float32 noise.
WEIGHT_EPS = 1e-4

#: Tolerance on the simplex constraints for calling a solution feasible.
SIMPLEX_TOL = 1e-6

#: Default pre-filter size: solve over the m nearest donors only.
DEFAULT_M = 50


# --------------------------------------------------------------------------
# The donor panel: built once per (encoder, scaling, T0), reused by every
# target.
# --------------------------------------------------------------------------


@dataclass
class Panel:
    """The full eligible cohort as two dense matrices, plus its id order.

    Every eligible patient is simultaneously a *target* (row `i` of `P`/`Y`) and
    a *donor* for everyone else.  `StateSet.donor_matrix` rebuilds a
    (748, 1536) array on every call; at 749 targets x several configurations
    that dominates the runtime, so the pool is materialised once here and the
    target is excluded by index instead.

    Attributes:
        ids: eligible `subject_id`s in the canonical numeric order that
            `state.sorted_ids` defines.  Row-aligned to `P` and `Y`.
        P: (n, (T0+1)*d) pre-period blocks, flattened t-major -- the design the
            fit matches on.  float64, because the Gram matrix is formed from it.
        Y: (n, d) the state at `T0+1`: the donor's value to forecast *from*, and
            simultaneously the target's held-out truth.
    """

    ids: List[str]
    P: np.ndarray
    Y: np.ndarray
    T0: int
    d: int
    encoder: str
    scale: str
    #: Free-text tag for the row-normalisation ablation: "raw" or "rowl2".
    rownorm: str = "raw"

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def key(self) -> str:
        """Filename-safe identifier for this (encoder, scaling, T0) panel."""
        return f"{self.encoder}-{self.scale}-{self.rownorm}-T{self.T0}"

    @classmethod
    def build(cls, ss: StateSet, T0: int) -> "Panel":
        """Materialise the eligible cohort for one `T0`.

        Eligibility is `StateSet.eligible_subjects(T0)`, i.e. `>= T0+2` visits:
        `t=0..T0` to fit on plus `t=T0+1` to forecast.  At the primary T0=1 this
        is the 749-patient `>=3`-visit cohort.
        """
        T0 = int(T0)
        ids = ss.eligible_subjects(T0)
        d = ss.d
        P = np.empty((len(ids), (T0 + 1) * d), dtype=np.float64)
        Y = np.empty((len(ids), d), dtype=np.float64)
        for i, s in enumerate(ids):
            blk = ss.block(s, 0, T0 + 1)  # (T0+2, d)
            P[i] = blk[: T0 + 1].reshape(-1)
            Y[i] = blk[T0 + 1]
        return cls(
            ids=ids, P=P, Y=Y, T0=T0, d=d, encoder=ss.encoder, scale=ss.scale
        )

    def rownorm_l2(self) -> "Panel":
        """Row-L2-normalised copy -- the required bag-encoder ablation.

        Euclidean distance between two binary bags is `sqrt(|A symdiff B|)`,
        which makes a *low-cardinality* patient a near neighbour of everybody
        regardless of clinical content (Phase 1 measured the median cardinality
        of picked neighbours at 26 against a population median of 39).  Dividing
        each **visit** vector by its own L2 norm converts the geometry to a
        cosine one and removes that bias.

        Normalisation is per visit, not per flattened pre-period block: each of
        the `T0+1` pre-period visits and the `T0+1` forecast target is scaled by
        its own norm, so a patient with a big visit 0 and a small visit 1 is not
        allowed to have visit 0 dominate the fit.  Zero-norm rows (none exist in
        this cohort; asserted) are left at zero.
        """

        def _norm(M: np.ndarray) -> np.ndarray:
            nr = np.linalg.norm(M, axis=-1, keepdims=True)
            return np.divide(M, nr, out=np.zeros_like(M), where=nr > 0)

        P = _norm(self.P.reshape(self.n, self.T0 + 1, self.d)).reshape(self.n, -1)
        Y = _norm(self.Y)
        return Panel(
            ids=list(self.ids), P=P, Y=Y, T0=self.T0, d=self.d,
            encoder=self.encoder, scale=self.scale, rownorm="rowl2",
        )

    # -- target-side views -------------------------------------------------

    def target_last_state(self, i: int) -> np.ndarray:
        """The target's own state at `T0` -- i.e. the LVCF forecast."""
        return self.P[i, -self.d:]

    def donor_index(self, i: int) -> np.ndarray:
        """Every row except `i`.  A patient is never their own donor."""
        idx = np.arange(self.n)
        return idx[idx != i]

    def distances(self, i: int, donors: np.ndarray) -> np.ndarray:
        """Euclidean pre-period distance from target `i` to each donor row."""
        return np.linalg.norm(self.P[donors] - self.P[i], axis=1)


# --------------------------------------------------------------------------
# The solver.
# --------------------------------------------------------------------------


@dataclass
class SolveResult:
    w: np.ndarray
    sse: float          #: || x - mu*1 - A' w ||^2 at the returned (mu, w)
    status: int         #: scipy status; 0 == success
    message: str
    nit: int
    nfev: int
    simplex_violation: float  #: max(|sum(w) - 1|, max(0, -min(w)))
    #: The fitted level shift mu^0 of Wang/Xing/Ye Eq. 3.  Exactly 0.0 when
    #: `fit_intercept=False`; UNCONSTRAINED IN SIGN when True.
    mu: float = 0.0


def _gram(
    A: np.ndarray, x: np.ndarray, fit_intercept: bool
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, int]:
    """Gram-form data for ``||x - mu*1 - A'w||^2``, with mu profiled out.

    Returns `(G, b, c, a, sx, D)` such that the objective as a function of `w`
    alone is `w'Gw - 2w'b + c`, plus the pieces needed to recover `mu`.

    WITHOUT an intercept this is the plain expansion
    `||x - A'w||^2 = w'(AA')w - 2w'(Ax) + x'x`, i.e. `G = AA'`, `b = Ax`,
    `c = x'x`.

    WITH an intercept, `mu` is unconstrained in sign, so it can be minimised out
    in closed form rather than handed to the solver as an extra variable.
    Writing `a = A @ 1_D` (each donor's coordinate sum), `sx = 1_D'x` and
    `D = len(x)`::

        f(mu, w) = w'Gw + 2 mu (a'w) + mu^2 D - 2 w'b - 2 mu sx + x'x
        df/dmu = 0   ->   mu*(w) = (sx - a'w) / D
        f(w)     = w'(G - a a'/D) w - 2 w'(b - sx a/D) + (x'x - sx^2/D)

    which is exactly the SAME Gram form on **mean-centred** data: `G - a a'/D`
    is `A~ A~'` and `b - sx a/D` is `A~ x~`, where each row of `A~` and `x~`
    has had its own coordinate mean removed.  So the intercept costs one rank-1
    update of an already-computed Gram matrix and NOTHING per solver iteration;
    the O(m^2) cost per objective/gradient evaluation is preserved exactly, and
    the augmented (m+1)x(m+1) normal-equation system is never formed.

    Profiling is legitimate here precisely because `mu` is unconstrained: the
    inner minimisation is unconstrained and exact, the problem is jointly
    convex, and the simplex constraints touch only `w`.  It would NOT be
    legitimate if `mu` were sign-constrained or penalised.
    """
    G = A @ A.T
    b = A @ x
    c = float(x @ x)
    D = int(A.shape[1])
    a = A.sum(axis=1)
    sx = float(x.sum())
    if fit_intercept:
        G = G - np.outer(a, a) / D
        b = b - (sx / D) * a
        c = c - sx * sx / D
    return G, b, c, a, sx, D


def _mu_of(w: np.ndarray, a: np.ndarray, sx: float, D: int,
           fit_intercept: bool) -> float:
    """The profiled level shift `mu*(w) = (1'x - w'(A 1)) / D`.

    In words: the mean of the target's pre-period block minus the mean of the
    synthetic control's, i.e. exactly the level difference the weights are no
    longer asked to absorb.
    """
    return float((sx - float(w @ a)) / D) if fit_intercept else 0.0


def _finish(
    res, f, m: int, a: np.ndarray, sx: float, D: int, fit_intercept: bool,
) -> SolveResult:
    w = np.asarray(res.x, dtype=np.float64)[:m]
    viol = max(abs(float(w.sum()) - 1.0), max(0.0, float(-w.min())))
    return SolveResult(
        w=w, sse=max(f(w), 0.0), status=int(res.status),
        message=str(res.message), nit=int(res.nit), nfev=int(res.nfev),
        simplex_violation=float(viol),
        mu=_mu_of(w, a, sx, D, fit_intercept),
    )


def solve_simplex_qp(
    A: np.ndarray,
    x: np.ndarray,
    w0: Optional[np.ndarray] = None,
    maxiter: int = SLSQP_MAXITER,
    ftol: float = SLSQP_FTOL,
    fit_intercept: bool = False,
) -> SolveResult:
    """``min_{mu,w} ||x - mu*1 - A'w||^2  s.t.  w >= 0, sum w = 1`` by SLSQP.

    With `fit_intercept=False` (the default, and what every previously recorded
    run used) `mu` is pinned to 0 and this is the classical Abadie estimator.

    The objective is expanded into its Gram form once, before the solver is
    entered::

        ||x - A'w||^2  =  w' (A A') w  -  2 w' (A x)  +  x'x

    so each of SLSQP's objective/gradient evaluations costs O(m^2) instead of
    O(m * len(x)).  This is an exact rewrite, not an approximation, and it is
    what makes the m=748 configuration affordable.  Analytic gradients are
    supplied for both the objective and the equality constraint; without them
    SLSQP would finite-difference m+1 times per iteration.

    THE INTERCEPT
    -------------
    `fit_intercept=True` adds the level shift `mu^0` of Wang, Xing & Ye Eq. 3,
    `Y_1t(0) = mu^0 + sum_j omega_j Y_jt + u_1t`, which they take from
    Doudchenko & Imbens (2016): with it "the goal of omega is no longer to
    perfectly match the treatment group, but only to mimic the trend ... we can
    correct it later with a unit fixed effect".  That matters in this
    application specifically: one patient chronically carries more diagnosis
    codes than another, and without `mu` the fit spends weight matching that
    *level* instead of the trajectory.

    `mu` is **unconstrained in sign** -- it is not on the simplex, is not
    bounded, and is not penalised.  It is therefore profiled out analytically
    (see `_gram`) rather than appended to the decision vector, which keeps the
    Gram-form efficiency exactly: the intercept costs one rank-1 downdate of
    `AA'` up front and nothing at all per iteration.  The solver still sees an
    m-dimensional simplex problem.

    Note that `mu` is ONE scalar added to every coordinate of the flattened
    pre-period block -- a uniform level shift across code dimensions and across
    pre-period visits alike, not a per-visit or per-code offset.  That is the
    quantity the motivating argument is about (a patient who is uniformly
    "more coded"), and it is what makes `mu: float` rather than a vector.

    Args:
        A: (m, D) donor pre-period rows.
        x: (D,) target pre-period row.
        w0: warm start.  Defaults to uniform `1/m` -- i.e. the fit starts at the
            unweighted donor mean and has to earn any sparsity it reports.
        fit_intercept: fit `mu^0` as above.  Default False, so this argument
            changes nothing unless it is passed.

    Returns:
        SolveResult, with `sse` the residual AT the fitted `(mu, w)` and `mu`
        the fitted level shift (0.0 when `fit_intercept=False`).
        `status == 0` is convergence; callers must not drop non-zero statuses
        silently, they are counted in the report.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    m = A.shape[0]
    if m == 0:
        raise ValueError("empty donor pool")

    G, b, c, a, sx, D = _gram(A, x, fit_intercept)

    def f(w: np.ndarray) -> float:
        return float(w @ (G @ w) - 2.0 * (w @ b) + c)

    if m == 1:
        w = np.ones(1)
        return SolveResult(
            w=w, sse=max(f(w), 0.0), status=0,
            message="trivial (m=1)", nit=0, nfev=0, simplex_violation=0.0,
            mu=_mu_of(w, a, sx, D, fit_intercept),
        )

    ones = np.ones(m)

    def g(w: np.ndarray) -> np.ndarray:
        return 2.0 * (G @ w - b)

    w0 = np.full(m, 1.0 / m) if w0 is None else np.asarray(w0, dtype=np.float64)
    res = minimize(
        f, w0, jac=g, method="SLSQP",
        bounds=[(0.0, None)] * m,
        constraints=[{"type": "eq",
                      "fun": lambda w: float(w.sum() - 1.0),
                      "jac": lambda w: ones}],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return _finish(res, f, m, a, sx, D, fit_intercept)


# --------------------------------------------------------------------------
# The L-infinity (dense-weight) solver variant.
# --------------------------------------------------------------------------


def solve_linf(
    A: np.ndarray,
    x: np.ndarray,
    lam: float,
    alpha: float = 1.0,
    w0: Optional[np.ndarray] = None,
    maxiter: int = SLSQP_MAXITER,
    ftol: float = SLSQP_FTOL,
    fit_intercept: bool = False,
) -> SolveResult:
    """Max-norm-penalised synthetic control (Wang, Xing & Ye Eq. 5).

    ::

        min_{mu,w}  ||x - mu*1 - A'w||^2 + P(w)     s.t.  w >= 0, sum w = 1

        alpha == 1.0    ->  P(w) = lam * ||w||_inf                  (Eq. 5, line 1)
        0 < alpha < 1   ->  P(w) = lam * (alpha*||w||_1
                                          + (1-alpha)*||w||_inf)    (Eq. 5, line 2)

    Note `alpha=1.0` selects the PURE max-norm penalty, i.e. it drops the L1
    term entirely; it does not mean "alpha=1 substituted into the composite
    formula".  That is how the paper writes its two penalties and how this
    project's API contract defines the argument.

    WHY
    ---
    The simplex constraints are not a neutral choice.  Doudchenko & Imbens
    (2016) show they are equivalent to a Lasso-type regularisation, so the
    sparsity classical SC reports is, in Wang/Xing/Ye's words, "a mechanical
    byproduct of the optimization problem ... neither intentional nor
    necessarily advantageous": concentrating the whole forecast on a handful of
    donors gives high variance and bias.  Penalising the LARGEST weight pushes
    the solution towards a denser, more uniform weighting.

    WHAT THE CONSTRAINT SET DOES TO THE PENALTY -- read before using `alpha`
    ------------------------------------------------------------------------
    This variant keeps the simplex, because in this application the forecast
    must remain a convex combination of real patients (and every downstream
    diagnostic -- `simplex_violation`, the effective-donor counts, the weight
    parquet -- assumes it).  The paper's own setting is *unconstrained* `omega`,
    where L1 and L-inf are genuinely different penalties.  On the simplex they
    are not::

        w >= 0 and sum w = 1   =>   ||w||_1 == 1 identically

    so the L1 half of the composite penalty is a CONSTANT `lam*alpha` that
    cannot change the argmin, and the composite collapses to a pure max-norm
    penalty at an effective strength `lam*(1-alpha)`.  `alpha` is therefore
    **not identified** here: `solve_linf(..., lam=L, alpha=a)` returns the same
    weights as `solve_linf(..., lam=L*(1-a), alpha=1.0)`.  The argument is
    accepted for API compatibility and so an unconstrained variant could use it
    later, but a sweep over `alpha` at fixed `lam` is a sweep over
    `lam*(1-alpha)` and nothing else.  This is a real negative finding about
    the port, not a limitation of the paper.

    The returned `sse` is the UNPENALISED residual `||x - mu*1 - A'w||^2` at the
    returned point, so it stays comparable with `solve_simplex_qp`'s and with
    `pre_rmse` everywhere else.  The penalty enters the optimisation only.

    HOW IT IS SOLVED
    ----------------
    The paper uses an interior point method and notes that closed forms are
    unavailable for these penalties, so no Lasso/ridge path is reused here.
    `cvxpy` is deliberately absent from this project.  Instead the max-norm is
    given its standard **epigraph reformulation**: introduce a scalar `s` and
    minimise::

        min_{w,s}  w'Gw - 2w'b + c + lam*(1-alpha)*s     [+ lam*alpha, dropped]
        s.t.       sum w = 1,   w >= 0,   s >= w_j  for every j,   s >= 0

    At the optimum `s = max_j w_j = ||w||_inf` (positive `lam` pushes `s` down
    until a constraint binds), so this is EXACTLY equivalent, not a relaxation
    -- and because `w >= 0` on the simplex, `|w_j| <= s` is the single linear
    inequality `w_j <= s`, with no absolute value and no non-smoothness left.
    The result is a smooth QP in `m+1` variables with `m+1` linear constraints,
    which is precisely what SLSQP is for.  Analytic gradients are supplied for
    the objective and for every constraint.

    Cost: the objective/gradient stay O(m^2) (same Gram form, `s` enters
    linearly), but the `m` epigraph inequalities make each SLSQP iteration more
    expensive than the plain simplex solve and the iteration count higher.

    Args:
        A:   (m, D) donor pre-period rows.
        x:   (D,) target pre-period row.
        lam: penalty strength, on the scale of the SUM-of-squares residual
            (this project does not carry the paper's 1/2 factor in Eq. 4).
            `lam=0` reduces exactly to `solve_simplex_qp` and is delegated to it.
        alpha: see above.  `1.0` = pure L-inf.  `0 < alpha < 1` = composite.
        fit_intercept: as in `solve_simplex_qp`; the penalty never touches `mu`,
            so `mu` is still profiled out exactly.

    Returns:
        SolveResult.  `sse` excludes the penalty.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    m = A.shape[0]
    if m == 0:
        raise ValueError("empty donor pool")
    lam = float(lam)
    alpha = float(alpha)
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    # Effective max-norm coefficient.  alpha == 1 is the pure L-inf penalty;
    # otherwise the L1 half is the constant lam*alpha (see the docstring) and
    # only lam*(1-alpha) multiplies ||w||_inf.
    lam_inf = lam if alpha == 1.0 else lam * (1.0 - alpha)

    if lam_inf == 0.0:
        # Nothing to penalise: the epigraph variable would be unconstrained
        # from above and SLSQP would wander over a flat direction.  Delegate.
        return solve_simplex_qp(A, x, w0=w0, maxiter=maxiter, ftol=ftol,
                                fit_intercept=fit_intercept)

    G, b, c, a, sx, D = _gram(A, x, fit_intercept)

    def f_w(w: np.ndarray) -> float:
        """Unpenalised residual -- what SolveResult.sse reports."""
        return float(w @ (G @ w) - 2.0 * (w @ b) + c)

    if m == 1:
        w = np.ones(1)
        return SolveResult(
            w=w, sse=max(f_w(w), 0.0), status=0,
            message="trivial (m=1)", nit=0, nfev=0, simplex_violation=0.0,
            mu=_mu_of(w, a, sx, D, fit_intercept),
        )

    eye = np.eye(m)
    # Epigraph constraint Jacobian: d(s - w_j)/d(w, s) = (-e_j, 1).
    jac_epi = np.hstack([-eye, np.ones((m, 1))])
    eq_jac = np.concatenate([np.ones(m), [0.0]])

    def f(z: np.ndarray) -> float:
        w, s = z[:m], z[m]
        return f_w(w) + lam_inf * float(s)

    def g(z: np.ndarray) -> np.ndarray:
        w = z[:m]
        out = np.empty(m + 1)
        out[:m] = 2.0 * (G @ w - b)
        out[m] = lam_inf
        return out

    w_start = (np.full(m, 1.0 / m) if w0 is None
               else np.asarray(w0, dtype=np.float64))
    z0 = np.concatenate([w_start, [float(w_start.max())]])
    res = minimize(
        f, z0, jac=g, method="SLSQP",
        bounds=[(0.0, None)] * m + [(0.0, None)],
        constraints=[
            {"type": "eq",
             "fun": lambda z: float(z[:m].sum() - 1.0),
             "jac": lambda z: eq_jac},
            {"type": "ineq",                      # s - w_j >= 0 for every j
             "fun": lambda z: z[m] - z[:m],
             "jac": lambda z: jac_epi},
        ],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return _finish(res, f_w, m, a, sx, D, fit_intercept)


# --------------------------------------------------------------------------
# Per-target fit.
# --------------------------------------------------------------------------


@dataclass
class TargetFit:
    """One target's synthetic control: weights, forecast, and diagnostics."""

    subject_id: str
    i: int
    T0: int
    m: int
    #: Panel row indices of the selected donors, aligned to `w`.
    donor_rows: np.ndarray
    w: np.ndarray
    forecast: np.ndarray
    #: PRE-period fit quality.  A DIAGNOSTIC, not the validation metric.
    pre_rmse: float
    #: 1 / sum(w^2): the inverse participation ratio.  m if uniform, 1 if
    #: all weight is on one donor.
    eff_donors_ipr: float
    #: count of weights above WEIGHT_EPS.
    eff_donors_count: int
    max_weight: float
    status: int
    message: str
    nit: int
    simplex_violation: float
    #: Which donor selection rule produced `donor_rows` ("nearest" | "random").
    selection: str = "nearest"
    #: Fitted level shift mu^0.  0.0 unless the fit was run with
    #: `fit_intercept=True`.  Already added into `forecast`.
    mu: float = 0.0
    #: Which penalty produced `w`: None for the classical simplex objective,
    #: "linf" for the max-norm variant.
    penalty: Optional[str] = None

    def top_donors(self, k: int = 5) -> List[Tuple[int, float]]:
        o = np.argsort(-self.w)[:k]
        return [(int(self.donor_rows[j]), float(self.w[j])) for j in o]


def _summarise(
    panel: Panel, i: int, rows: np.ndarray, sol: SolveResult, m_req: int,
    selection: str, penalty: Optional[str] = None,
) -> TargetFit:
    w = sol.w
    D_pre_len = panel.P.shape[1]
    # The forecast carries the intercept: Wang/Xing/Ye Eq. 6 imputes
    # Y_1t(0) = mu_hat + sum_j omega_hat_j Y_jt, so a level shift fitted on the
    # pre-period is applied to the post-period too.  With fit_intercept=False,
    # mu is exactly 0.0 and this is the unchanged classical forecast.
    return TargetFit(
        subject_id=panel.ids[i], i=i, T0=panel.T0, m=m_req,
        donor_rows=rows, w=w,
        forecast=w @ panel.Y[rows] + sol.mu,
        pre_rmse=float(np.sqrt(sol.sse / D_pre_len)),
        eff_donors_ipr=float(1.0 / max(float(w @ w), 1e-300)),
        eff_donors_count=int((w > WEIGHT_EPS).sum()),
        max_weight=float(w.max()),
        status=sol.status, message=sol.message, nit=sol.nit,
        simplex_violation=sol.simplex_violation, selection=selection,
        mu=float(sol.mu), penalty=penalty,
    )


def select_donors(
    panel: Panel, i: int, m: Optional[int],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, str]:
    """Choose which donors enter the optimisation for target `i`.

    Two rules:

    * `rng is None` (the estimator) -- the `m` **nearest** donors by Euclidean
      distance on the flattened pre-period block, the same geometry the
      objective minimises.  `np.argsort(kind="stable")` so that ties break by
      panel row order, which is canonical numeric `subject_id` order.  This is
      load-bearing for the bag encoder, whose distances are integer-valued and
      heavily tied (Phase 1: ~100 distinct values across 748 donors).
    * `rng` supplied (the placebo) -- `m` donors sampled uniformly without
      replacement, ignoring distance entirely.

    `m=None` or `m >= n-1` means no pre-filter: every eligible donor.

    THE SELECTION LABEL IS PART OF THE CONTRACT.  It is written to the fits
    cache and to `phase2_targets.parquet`, and the placebo's whole meaning is
    that it did NOT use distance.  An earlier version returned the full pool
    tagged `"all"` for BOTH branches once `m >= n-1`, so a placebo requested at
    a large `m` silently stopped being labelled a placebo.  That is harmless at
    the primary `m=50` (50 << 748) but wrong under an m-sweep, and it would
    have made a `sc_random` row at `m=all` indistinguishable from `sc`.  The
    random branch is now always labelled `"random"`; when the requested `m`
    covers the whole pool the draw degenerates to the whole pool (a uniform
    sample of size n-1 without replacement from n-1 units IS the pool), which
    is reported honestly rather than relabelled.
    """
    donors = panel.donor_index(i)
    n_av = len(donors)
    if rng is not None:
        if m is None or int(m) >= n_av:
            # Sampling n_av of n_av units without replacement is the identity;
            # keep the label so downstream still knows this row is the placebo.
            return donors, "random"
        return np.sort(rng.choice(donors, size=int(m), replace=False)), "random"
    if m is None or int(m) >= n_av:
        return donors, "all"
    dist = panel.distances(i, donors)
    order = np.argsort(dist, kind="stable")[: int(m)]
    return donors[order], "nearest"


def fit_target(
    panel: Panel,
    i: int,
    m: Optional[int] = DEFAULT_M,
    rng: Optional[np.random.Generator] = None,
    maxiter: int = SLSQP_MAXITER,
    ftol: float = SLSQP_FTOL,
    fit_intercept: bool = False,
    penalty: Optional[str] = None,
    lam: float = 0.0,
    alpha: float = 1.0,
) -> TargetFit:
    """Pre-filter, solve, and package one target's synthetic control.

    Args:
        fit_intercept: fit the level shift `mu^0` (Wang/Xing/Ye Eq. 3).  The
            fitted `mu` is added to the forecast, per their Eq. 6.
        penalty: `None` for the classical simplex objective (the default and
            what every recorded run used), or `"linf"` for the max-norm variant
            `solve_linf`.
        lam:   penalty strength; only read when `penalty == "linf"`.
        alpha: `1.0` = pure L-inf, `0 < alpha < 1` = composite L1 + L-inf.  Only
            read when `penalty == "linf"`, and see `solve_linf` on why it is not
            identified under the simplex constraints.
    """
    rows, selection = select_donors(panel, i, m, rng=rng)
    if penalty is None:
        sol = solve_simplex_qp(panel.P[rows], panel.P[i], maxiter=maxiter,
                               ftol=ftol, fit_intercept=fit_intercept)
    elif penalty == "linf":
        sol = solve_linf(panel.P[rows], panel.P[i], lam=lam, alpha=alpha,
                         maxiter=maxiter, ftol=ftol,
                         fit_intercept=fit_intercept)
    else:
        raise ValueError(f"unknown penalty {penalty!r}; expected None or 'linf'")
    m_req = len(rows) if m is None else int(m)
    return _summarise(panel, i, rows, sol, m_req, selection, penalty=penalty)


# --------------------------------------------------------------------------
# Whole-cohort driver.  Kept importable and picklable so the runner can fan it
# out over processes; this phase is CPU/scipy work and touches no GPU.
# --------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _worker_init(panel: Panel, m, seed, maxiter, ftol, random_donors,
                 fit_intercept=False, penalty=None, lam=0.0, alpha=1.0) -> None:
    _WORKER.update(panel=panel, m=m, seed=seed, maxiter=maxiter, ftol=ftol,
                   random_donors=random_donors, fit_intercept=fit_intercept,
                   penalty=penalty, lam=lam, alpha=alpha)


def _worker_fit(i: int) -> TargetFit:
    w = _WORKER
    # Per-target seed derived from (run seed, target index), so a target's
    # random donor set does not depend on how the work was chunked across
    # processes.  Reproducible under any -j.
    rng = (np.random.default_rng([w["seed"], i]) if w["random_donors"] else None)
    return fit_target(w["panel"], i, m=w["m"], rng=rng,
                      maxiter=w["maxiter"], ftol=w["ftol"],
                      fit_intercept=w.get("fit_intercept", False),
                      penalty=w.get("penalty"), lam=w.get("lam", 0.0),
                      alpha=w.get("alpha", 1.0))


def fit_all(
    panel: Panel,
    m: Optional[int] = DEFAULT_M,
    seed: int = 0,
    random_donors: bool = False,
    maxiter: int = SLSQP_MAXITER,
    ftol: float = SLSQP_FTOL,
    n_jobs: int = 1,
    verbose: bool = True,
    fit_intercept: bool = False,
    penalty: Optional[str] = None,
    lam: float = 0.0,
    alpha: float = 1.0,
) -> List[TargetFit]:
    """Fit every eligible target in `panel`.

    Args:
        m: pre-filter size; `None` for the full pool.
        random_donors: placebo mode -- draw the `m` donors at random instead of
            by nearest-neighbour.  Everything downstream is identical.
        n_jobs: processes.  Targets are independent, so this is embarrassingly
            parallel; results are re-sorted into panel order regardless of
            completion order, so `n_jobs` never changes the output.
        fit_intercept, penalty, lam, alpha: forwarded verbatim to `fit_target`;
            defaults reproduce the classical estimator exactly.
    """
    import time

    t0 = time.time()
    idx = list(range(panel.n))
    init = (panel, m, seed, maxiter, ftol, random_donors, fit_intercept,
            penalty, lam, alpha)
    if n_jobs and n_jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_jobs, initializer=_worker_init, initargs=init,
        ) as pool:
            fits = pool.map(_worker_fit, idx, chunksize=4)
    else:
        _worker_init(*init)
        fits = [_worker_fit(i) for i in idx]
    fits.sort(key=lambda f: f.i)
    if verbose:
        nbad = sum(1 for f in fits if f.status != 0)
        print(
            f"[fit] {panel.key} m={m} random={random_donors} "
            f"n={len(fits)} in {time.time()-t0:.1f}s  "
            f"non-converged={nbad}",
            flush=True,
        )
    return fits


def convergence_report(fits: Sequence[TargetFit]) -> Dict[str, Any]:
    """Solver health.  Reported, never used to filter targets."""
    status = np.array([f.status for f in fits])
    viol = np.array([f.simplex_violation for f in fits])
    nit = np.array([f.nit for f in fits])
    by_status: Dict[str, int] = {}
    msgs: Dict[str, int] = {}
    for f in fits:
        by_status[str(f.status)] = by_status.get(str(f.status), 0) + 1
        if f.status != 0:
            msgs[f.message] = msgs.get(f.message, 0) + 1
    return {
        "n": int(len(fits)),
        "n_converged": int((status == 0).sum()),
        "n_not_converged": int((status != 0).sum()),
        "frac_not_converged": float((status != 0).mean()),
        "status_counts": by_status,
        "nonconvergence_messages": msgs,
        "nit_mean": float(nit.mean()),
        "nit_max": int(nit.max()),
        "max_simplex_violation": float(viol.max()),
        "n_infeasible_beyond_tol": int((viol > SIMPLEX_TOL).sum()),
        "simplex_tol": SIMPLEX_TOL,
    }
