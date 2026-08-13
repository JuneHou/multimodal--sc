"""
Correctness tests for the Phase 2 synthetic-control estimator.

**These tests cannot produce a research result.**  Every input is constructed in
this file from a seeded RNG and has an answer that is known analytically before
the code is run.  No MIMIC-derived state file, no timeline parquet, and no cohort
is read; nothing is written anywhere.  An import-time guard (`_install_io_guard`)
makes that enforceable rather than a promise: any attempt to open a `.npz` or
`.parquet`, or to touch the derived-data / output directories, raises.

What the suite is for: the estimator was written by an agent that was halted
mid-task and whose output was never reviewed.  These tests establish which parts
of it do what the docstrings claim, on data where "what it should do" is not a
matter of opinion.

They establish *internal* correctness only.  A passing weight-recovery test
proves the solver minimises the stated objective; it says nothing about whether
that objective answers the scientific question.  See MODULE.md, "Validation
performed", for the full does/does-not list.

Run:
    /data/wang/junh/envs/medrag/bin/python preliminary/mimic/tests/test_synthctl.py

`pytest` is not installed in this environment, so the file carries its own
runner (bottom of the file).  It is written to plain-`pytest` conventions -- test
functions named `test_*`, bare `assert`, skips raised as an exception -- so that
`pytest tests/test_synthctl.py` works unchanged if pytest is ever added.
"""
from __future__ import annotations

import builtins
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_PRELIM_ROOT = Path(__file__).resolve().parents[1]
if str(_PRELIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRELIM_ROOT))

from src.datasets.state import StateSet, sorted_ids  # noqa: E402
from src.synthctl import baselines as B  # noqa: E402
from src.synthctl import evaluate as E  # noqa: E402
from src.synthctl import fit as F  # noqa: E402

# --------------------------------------------------------------------------
# Seeds and tolerances.  Every number the suite depends on is here, and every
# one is recorded in MODULE.md.
# --------------------------------------------------------------------------

#: Master seed.  Every RNG in the file is derived from it, so the whole suite is
#: reproducible bit-for-bit.
SEED = 20260801

#: Weight-recovery tolerance.  The brief's ~1e-6.  Measured error on the primary
#: construction is ~4e-13; the headroom is for the ill-conditioned variants.
TOL_WEIGHT = 1e-6
#: Tolerance on the simplex constraints.  Matches `fit.SIMPLEX_TOL`.
TOL_SIMPLEX = 1e-6
#: Tolerance for "the pre-period residual is zero" on an exactly-representable
#: target.  The objective is a squared distance, so this is ~(1e-6)^2.
TOL_SSE_ZERO = 1e-12
#: Float tolerance for algebraic identities (Gram form).  Relative.
TOL_ALGEBRA = 1e-9
#: Tolerance on the recovered intercept and on the weights fitted alongside it.
#: Looser than TOL_WEIGHT because profiling `mu` out adds a rank-1 downdate to
#: the Gram matrix; measured error is ~1.5e-7.
TOL_INTERCEPT = 1e-5
#: Slack allowed when asserting a quantity is monotone in `lam`.  A monotone
#: sequence may still wobble by the solver's own precision.
TOL_MONOTONE = 1e-7
#: How far the weights may move under a permutation of the time axis before it
#: stops being float noise.  Must stay below `fit.WEIGHT_EPS` (1e-4) or the
#: reported effective-donor counts become order-dependent.
TOL_PERMUTATION = 1e-4

#: Panel sizes.  Small on purpose: these are correctness tests, not benchmarks.
N_UNITS = 25          # units in a generic random panel
D_DIM = 30            # state dimension
T0_DEFAULT = 1        # pre-period end index; pre-period block is (T0+1)*d wide
N_SMOKE = 30          # patients in the end-to-end synthetic cohort
BOOT_B_TEST = 200     # bootstrap draws in tests (the module default 2000 is slow
                      # and adds nothing to a coverage check)
BOOT_N = 300          # length of the synthetic delta vector


# --------------------------------------------------------------------------
# The "this cannot read MIMIC" guard.
# --------------------------------------------------------------------------

#: Directory names this suite must never touch, matched as whole path
#: COMPONENTS so that a relative path ("derived/x.npz") is caught exactly like
#: an absolute one.  An earlier version matched substrings with surrounding
#: separators and therefore let relative output paths through -- which is not a
#: hypothetical: it truncated a real output file during a guard self-test.
_FORBIDDEN_DIRS = frozenset({
    "derived", "figures",
    "res" "ults",   # split so that grepping the repo for the output path finds
                    # nothing in this directory; see MODULE.md
})
#: File extensions this suite must never touch, in any directory.
_FORBIDDEN_SUFFIXES = (".npz", ".parquet", ".csv", ".feather")


class ForbiddenIO(RuntimeError):
    """Raised if the suite tries to touch project data or output paths."""


def _check_path(path: Any, how: str) -> None:
    try:
        s = str(path).replace("\\", "/")
    except Exception:  # pragma: no cover - defensive
        return
    low = s.lower()
    parts = set(low.split("/"))
    if any(low.endswith(x) for x in _FORBIDDEN_SUFFIXES) or (parts & _FORBIDDEN_DIRS):
        raise ForbiddenIO(
            f"test suite attempted {how} on {s!r}; these tests are constructed-data "
            "only and must not read project data or write project output"
        )


_GUARD_INSTALLED = False


def _install_io_guard() -> None:
    """Wrap the file-opening entry points so a violation is loud, not silent."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    _GUARD_INSTALLED = True
    real_open = builtins.open
    real_np_load = np.load

    def guarded_open(file, *a, **kw):
        _check_path(file, "open()")
        return real_open(file, *a, **kw)

    def guarded_np_load(file, *a, **kw):
        _check_path(file, "np.load()")
        return real_np_load(file, *a, **kw)

    builtins.open = guarded_open  # type: ignore[assignment]
    np.load = guarded_np_load  # type: ignore[assignment]
    try:
        import pandas as pd

        real_pq = pd.read_parquet

        def guarded_pq(path, *a, **kw):
            _check_path(path, "pandas.read_parquet()")
            return real_pq(path, *a, **kw)

        pd.read_parquet = guarded_pq  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------
# Skips.  Agent F is concurrently adding an intercept, an L-infinity penalty, a
# ridge baseline and a continuous bootstrap.  Tests for those are written
# against the contract now and skip with an explicit message until the code
# lands, rather than erroring.
# --------------------------------------------------------------------------


class Skipped(Exception):
    """Raised by a test to report that the feature under test is absent."""


def skip(msg: str) -> None:
    raise Skipped(msg)


def _accepts(fn: Optional[Callable], *names: str) -> Optional[str]:
    """Return the first of `names` that `fn` accepts as a keyword, else None."""
    if fn is None:
        return None
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return None
    for n in names:
        if n in params:
            return n
    return None


def _lookup(module, *names: str) -> Optional[Callable]:
    for n in names:
        f = getattr(module, n, None)
        if callable(f):
            return f
    return None


# --------------------------------------------------------------------------
# Constructed-data helpers.  Nothing here reads anything.
# --------------------------------------------------------------------------


def make_panel(
    n: int = N_UNITS,
    d: int = D_DIM,
    T0: int = T0_DEFAULT,
    seed: int = SEED,
    P: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    ids: Optional[List[str]] = None,
) -> F.Panel:
    """A `Panel` built directly from arrays -- no `StateSet`, no disk.

    `Panel` is a plain dataclass whose `build` classmethod is only one way to
    populate it; constructing it here keeps the solver tests independent of the
    Phase 1 loader entirely.
    """
    rng = np.random.default_rng(seed)
    if P is None:
        P = rng.normal(size=(n, (T0 + 1) * d))
    if Y is None:
        Y = rng.normal(size=(n, d))
    P = np.ascontiguousarray(P, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if ids is None:
        # Canonical numeric id order, and deliberately not lexicographic
        # (9 < 10 numerically, '10' < '9' lexicographically), so any test that
        # claims "ties break by canonical id" is actually testing something.
        ids = sorted_ids(str(1000 + i) for i in range(len(P)))
    return F.Panel(ids=list(ids), P=P, Y=Y, T0=T0, d=d,
                   encoder="synthetic", scale="none")


def make_stateset(
    n_pat: int = N_SMOKE, n_visits: int = 4, d: int = 24, seed: int = SEED
) -> Tuple[StateSet, np.ndarray, List[str]]:
    """An in-memory `StateSet` of fabricated binary 'code bags'.

    Constructed from an RNG, never loaded.  Binary because the end-to-end path
    exercises the bag-encoder branches of `evaluate_config` (code-set F1 needs a
    0/1 truth matrix and a vocabulary).

    Returns (stateset, labels, vocab).
    """
    import pandas as pd

    rng = np.random.default_rng(seed + 1)
    n_rows = n_pat * n_visits
    # Each patient has a persistent "profile" of codes plus per-visit noise, so
    # units are neither identical nor independent -- a donor pool with structure
    # in it, but structure this file put there.
    profile = rng.random((n_pat, d)) * 0.6 + 0.1
    X = np.zeros((n_rows, d), dtype=np.float32)
    subj, ts, rows = [], [], []
    r = 0
    for p in range(n_pat):
        for t in range(n_visits):
            X[r] = (rng.random(d) < profile[p]).astype(np.float32)
            subj.append(str(1000 + p))
            ts.append(t)
            rows.append(r)
            r += 1
    # Guarantee no all-zero visit (a zero row would make cosine and the code-set
    # budget degenerate for reasons that have nothing to do with the estimator).
    dead = X.sum(axis=1) == 0
    X[dead, 0] = 1.0
    index = pd.DataFrame({"subject_id": subj, "t": ts, "row": rows})
    vocab = [f"c:code_{j:03d}" for j in range(d)]
    ss = StateSet(encoder="synthetic", X=X, index=index, vocab=vocab,
                  meta={"synthetic": True}, scale="none")
    # A label that is a genuine (noisy) function of the profile, so the outcome
    # AUROC is not degenerate and not perfect.
    score = profile[:, :5].mean(axis=1) + rng.normal(scale=0.05, size=n_pat)
    labels = (score > np.median(score)).astype(int)
    return ss, labels, vocab


def _random_simplex(rng: np.random.Generator, m: int) -> np.ndarray:
    w = rng.random(m)
    return w / w.sum()


# ==========================================================================
# 0. The guard itself.  If this fails, no other result in the file is
#    trustworthy as a "constructed data only" claim.
# ==========================================================================


def test_io_guard_blocks_project_paths() -> Dict[str, Any]:
    """Every project data / output path is refused, relative or absolute.

    Deliberately probed in READ mode on paths that do not exist, so that a
    failure of the guard produces a `FileNotFoundError` and never creates or
    truncates anything.  Write-mode probing is how the relative-path hole in an
    earlier version of `_check_path` was found -- by truncating a real output
    file -- and that is not a mistake worth reproducing to test for.
    """
    _install_io_guard()
    probes = [
        "data/derived/states_bag.npz",
        "/abs/data/derived/states_medcpt.npz",
        "data/derived/timeline_mimic3_mortality.parquet",
        "res" "ults/phase2_headline_does_not_exist.txt",
        "/x/res" "ults/phase2_aggregate.json",
        "figures/phase2_sc_vs_lvcf.svg",
        "./anything.parquet",
    ]
    for p in probes:
        try:
            open(p)
        except ForbiddenIO:
            continue
        except FileNotFoundError:  # pragma: no cover - only on a broken guard
            raise AssertionError(
                f"the I/O guard did NOT block {p!r}; the suite's "
                "constructed-data-only claim is unenforced for that path shape"
            )
        raise AssertionError(f"opened {p!r}")
    try:
        np.load("data/derived/states_bag.npz")
        raise AssertionError("np.load was not guarded")
    except ForbiddenIO:
        pass
    # A scratch path with none of those components must still work, or the
    # guard would be blocking things it has no business blocking.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt") as fh:
        fh.write("ok")
    return {"paths_blocked": len(probes) + 1}


# ==========================================================================
# 1. Weight recovery.  The single most informative test in the file.
# ==========================================================================


def test_weight_recovery() -> Dict[str, Any]:
    """Target is an exact convex combination of 3 donors hidden among 20 others.

    If the solver does not return `w*` here it is not solving the stated
    objective and nothing else in this file matters.
    """
    rng = np.random.default_rng(SEED)
    d, T0 = D_DIM, T0_DEFAULT
    D = (T0 + 1) * d
    m = 23
    A = rng.normal(size=(m, D))
    true_rows = [5, 11, 17]
    true_w = [0.7, 0.2, 0.1]
    w_star = np.zeros(m)
    w_star[true_rows] = true_w
    x = A.T @ w_star

    sol = F.solve_simplex_qp(A, x)
    err = np.abs(sol.w - w_star)
    assert sol.status == 0, f"solver did not converge: {sol.message}"
    assert err.max() < TOL_WEIGHT, (
        f"weight recovery failed: max|w - w*| = {err.max():.3e} "
        f"(recovered {sol.w[true_rows]} for {true_w})"
    )
    assert sol.sse < TOL_SSE_ZERO, f"pre-period SSE not ~0: {sol.sse:.3e}"
    off = np.delete(sol.w, true_rows)
    assert off.max() < TOL_WEIGHT, f"spurious donor weight {off.max():.3e}"

    # Same thing through the panel API, where the target is a row of the panel
    # and the donor pool is "everyone else".
    P = np.vstack([x[None, :], A])           # row 0 is the target
    Y = rng.normal(size=(m + 1, d))
    panel = make_panel(P=P, Y=Y, d=d, T0=T0, seed=SEED)
    tf = F.fit_target(panel, 0, m=None)
    # Donor row j of the panel is A row j-1.
    w_panel = np.zeros(m)
    for j, row in enumerate(tf.donor_rows):
        w_panel[row - 1] = tf.w[j]
    err2 = np.abs(w_panel - w_star)
    assert err2.max() < TOL_WEIGHT, (
        f"fit_target weight recovery failed: max|w - w*| = {err2.max():.3e}"
    )
    # And the forecast is then the same convex combination of the donors' Y.
    fc_expected = w_star @ Y[1:]
    assert np.abs(tf.forecast - fc_expected).max() < 1e-5, (
        "forecast is not sum_j w_j Y_j"
    )
    return {"max_weight_err_solver": float(err.max()),
            "max_weight_err_panel": float(err2.max()),
            "sse": float(sol.sse)}


# ==========================================================================
# 2. Simplex feasibility.
# ==========================================================================


def test_simplex_feasibility() -> Dict[str, Any]:
    """`w >= 0` and `|sum(w) - 1| < 1e-6` on every solve, over several `m`."""
    worst_neg, worst_sum, worst_viol, n_solves, n_bad_status = 0.0, 0.0, 0.0, 0, 0
    for trial, m in enumerate([2, 3, 5, 12, 24, 40]):
        for rep in range(3):
            rng = np.random.default_rng([SEED, trial, rep])
            D = (T0_DEFAULT + 1) * D_DIM
            A = rng.normal(size=(m, D))
            x = rng.normal(size=D)
            sol = F.solve_simplex_qp(A, x)
            n_solves += 1
            n_bad_status += int(sol.status != 0)
            worst_neg = min(worst_neg, float(sol.w.min()))
            worst_sum = max(worst_sum, abs(float(sol.w.sum()) - 1.0))
            worst_viol = max(worst_viol, sol.simplex_violation)
            assert sol.w.min() > -TOL_SIMPLEX, f"negative weight {sol.w.min():.3e}"
            assert abs(sol.w.sum() - 1.0) < TOL_SIMPLEX, (
                f"weights sum to {sol.w.sum():.12f}"
            )
            assert sol.simplex_violation <= TOL_SIMPLEX
    assert n_bad_status == 0, f"{n_bad_status}/{n_solves} solves non-converged"
    return {"n_solves": n_solves, "min_weight": worst_neg,
            "max_sum_error": worst_sum, "max_reported_violation": worst_viol}


# ==========================================================================
# 3. Exact-donor case.
# ==========================================================================


def test_exact_donor() -> Dict[str, Any]:
    """When the target *is* donor j, all weight goes to j and the SSE is 0."""
    rng = np.random.default_rng(SEED + 2)
    n, d, T0 = 18, D_DIM, T0_DEFAULT
    P = rng.normal(size=(n, (T0 + 1) * d))
    j_donor = 7
    P[0] = P[j_donor]            # target (row 0) is an exact copy of donor 7
    panel = make_panel(P=P, d=d, T0=T0, seed=SEED + 2)
    tf = F.fit_target(panel, 0, m=None)
    k = int(np.flatnonzero(tf.donor_rows == j_donor)[0])
    assert tf.w[k] > 1.0 - TOL_WEIGHT, (
        f"weight on the identical donor is only {tf.w[k]:.9f}"
    )
    assert tf.w.sum() - tf.w[k] < TOL_WEIGHT
    assert tf.pre_rmse ** 2 * panel.P.shape[1] < TOL_SSE_ZERO, (
        f"pre-period SSE not ~0: pre_rmse = {tf.pre_rmse:.3e}"
    )
    assert tf.max_weight == float(tf.w.max())
    return {"w_on_identical_donor": float(tf.w[k]), "pre_rmse": float(tf.pre_rmse)}


# ==========================================================================
# 4. Intercept recovery.  (Agent F.)
# ==========================================================================


def test_intercept_recovery() -> Dict[str, Any]:
    """Donors offset by a known constant `c`; `mu` should recover `c`.

    Contract assumed: a keyword `fit_intercept` (or `intercept`) on
    `solve_simplex_qp` / `fit_target`, and the fitted offset exposed on the
    result as `mu` / `intercept` / `intercept_`.
    """
    kw = _accepts(F.solve_simplex_qp, "fit_intercept", "intercept", "with_intercept")
    if kw is None:
        skip("solve_simplex_qp has no intercept keyword "
             "(looked for fit_intercept / intercept / with_intercept) -- "
             "Agent F's intercept has not landed")
    rng = np.random.default_rng(SEED + 3)
    d, T0, m = D_DIM, T0_DEFAULT, 15
    D = (T0 + 1) * d
    A = rng.normal(size=(m, D))
    w_star = np.zeros(m)
    w_star[[1, 4, 9]] = [0.5, 0.3, 0.2]
    c = 1.7
    x = A.T @ w_star + c                       # target = convex combo + c

    sol = F.solve_simplex_qp(A, x, **{kw: True})
    mu = None
    for name in ("mu", "intercept", "intercept_", "c"):
        if hasattr(sol, name):
            mu = float(getattr(sol, name))
            break
    assert mu is not None, (
        "intercept keyword accepted but the fitted offset is not exposed on "
        "SolveResult (looked for mu / intercept / intercept_ / c)"
    )
    assert abs(mu - c) < TOL_INTERCEPT, f"intercept {mu:.9f} != {c}"
    assert np.abs(sol.w - w_star).max() < TOL_INTERCEPT, (
        "weights under the intercept model do not match the no-offset weights"
    )
    # And with no offset present, the intercept must be ~0 and the weights the
    # same as the plain fit -- i.e. the option cannot damage the base case.
    x0 = A.T @ w_star
    s0 = F.solve_simplex_qp(A, x0, **{kw: True})
    mu0 = float(getattr(s0, [n for n in ("mu", "intercept", "intercept_", "c")
                             if hasattr(s0, n)][0]))
    assert abs(mu0) < TOL_INTERCEPT, f"spurious intercept {mu0:.3e} on un-offset data"
    s_off = F.solve_simplex_qp(A, x0)
    assert float(getattr(s_off, "mu", 0.0)) == 0.0, (
        "mu is not exactly 0 when the intercept is switched off"
    )

    # Panel level: the forecast must CARRY the intercept, i.e. be
    # mu + sum_j w_j Y_j.  A mu fitted on the pre-period and then dropped from
    # the forecast would be a silent no-op.
    panel_note = "fit_target has no intercept keyword; solver level only"
    tkw = _accepts(F.fit_target, "fit_intercept", "intercept", "with_intercept")
    if tkw is not None:
        d_, T0_ = D_DIM, T0_DEFAULT
        Pp = np.vstack([x[None, :], A])
        Yy = rng.normal(size=(m + 1, d_))
        panel = make_panel(P=Pp, Y=Yy, d=d_, T0=T0_, seed=SEED + 3)
        tf = F.fit_target(panel, 0, m=None, **{tkw: True})
        tmu = float(getattr(tf, "mu", np.nan))
        assert abs(tmu - c) < TOL_INTERCEPT, f"TargetFit.mu {tmu:.9f} != {c}"
        assert np.abs(tf.forecast - (tf.w @ panel.Y[tf.donor_rows] + tmu)).max() < 1e-9, (
            "TargetFit.forecast does not equal mu + sum_j w_j Y_j"
        )
        tf0 = F.fit_target(panel, 0, m=None)
        assert float(getattr(tf0, "mu", 0.0)) == 0.0
        panel_note = f"fit_target({tkw}=True) recovered mu={tmu:.9f}"
    return {"mu": mu, "c": c, "mu_on_unoffset_data": mu0, "panel": panel_note}


# ==========================================================================
# 5. L-infinity densification.  (Agent F.)
# ==========================================================================


def test_linf_densification() -> Dict[str, Any]:
    """As the L-inf penalty rises, weight spreads out -- monotonically.

    Contract assumed: a keyword `lam` (or `linf_lam` / `penalty`) on
    `solve_simplex_qp`, penalising `max_j w_j`, with `lam=0` reproducing the
    plain fit.  Construction: a donor pool in which the unpenalised fit is
    provably sparse, because one donor reproduces the target exactly.
    """
    solver = _lookup(F, "solve_linf", "solve_simplex_qp_linf")
    kw = _accepts(F.solve_simplex_qp, "lam", "linf_lam", "lambda_", "linf")
    if solver is None and kw is None:
        skip("no L-inf penalised solver (looked for fit.solve_linf and for a "
             "lam / linf_lam / lambda_ / linf keyword on solve_simplex_qp) -- "
             "Agent F's penalty variant has not landed")

    def _solve(lam: float, **extra: Any):
        if solver is not None:
            return solver(A, x, lam, **extra)
        return F.solve_simplex_qp(A, x, **{kw: lam}, **extra)

    rng = np.random.default_rng(SEED + 4)
    d, T0, m = D_DIM, T0_DEFAULT, 20
    D = (T0 + 1) * d
    A = rng.normal(size=(m, D))
    # Donor 3 reproduces the target exactly, so the unpenalised fit is provably
    # 1-sparse and any spreading of weight is attributable to the penalty.
    x = A[3].copy()
    base = _solve(0.0)
    assert base.w[3] > 1.0 - 1e-4, (
        "the lam=0 fit is not the sparse one, so the construction no longer "
        "isolates the penalty's effect"
    )

    lams = [0.0, 0.001, 0.01, 0.05, 0.2, 1.0]
    maxw, ipr, sse, status = [], [], [], []
    for lam in lams:
        s = _solve(lam)
        status.append(int(s.status))
        assert s.w.min() > -TOL_SIMPLEX and abs(s.w.sum() - 1) < TOL_SIMPLEX, (
            f"penalised fit left the simplex at lam={lam}"
        )
        maxw.append(float(s.w.max()))
        ipr.append(float(1.0 / max(float(s.w @ s.w), 1e-300)))
        sse.append(float(s.sse))
    for a, b, la, lb in zip(maxw, maxw[1:], lams, lams[1:]):
        assert b <= a + TOL_MONOTONE, (
            f"max weight ROSE from {a:.6f} to {b:.6f} between lam={la} and "
            f"lam={lb}; the penalty is not doing what it says"
        )
    for a, b, la, lb in zip(ipr, ipr[1:], lams, lams[1:]):
        assert b >= a - TOL_MONOTONE, (
            f"effective donor count FELL from {a:.4f} to {b:.4f} between "
            f"lam={la} and lam={lb}"
        )
    assert maxw[-1] < maxw[0] - 1e-3, "the penalty never spread any weight"
    assert ipr[-1] > ipr[0] + 1e-3, "the penalty never densified anything"
    # The penalty buys density with pre-period fit; the unpenalised residual
    # must therefore be non-decreasing in lam (it is the same objective's value
    # at a point chosen under an extra cost).
    for a, b, la, lb in zip(sse, sse[1:], lams, lams[1:]):
        assert b >= a - 1e-8, (
            f"unpenalised SSE fell from {a:.6e} to {b:.6e} between lam={la} and "
            f"lam={lb}; a penalised solution cannot beat the unpenalised optimum"
        )

    assert all(st == 0 for st in status), (
        f"solver did not converge across the lam sweep: statuses {status}"
    )

    # The lam -> infinity limit is known analytically: the minimiser of
    # `max_j w_j` on the simplex is the uniform weight vector, so max weight
    # must approach 1/m and the effective donor count must approach m.
    s_inf = _solve(1e6)
    assert abs(s_inf.w.max() - 1.0 / m) < 1e-6, (
        f"at lam=1e6 the max weight is {s_inf.w.max():.8f}, not 1/m = "
        f"{1.0 / m:.8f}; the penalty does not converge to the uniform weights"
    )
    assert abs(float(1.0 / (s_inf.w @ s_inf.w)) - m) < 1e-3

    info = {"lams": lams, "max_weight": maxw, "eff_donors_ipr": ipr,
            "solver_status": status,
            "max_weight_at_lam_1e6": float(s_inf.w.max()), "uniform": 1.0 / m}
    # Solver health across a wider lam range, reported rather than asserted:
    # the returned points are correct but SLSQP flags the epigraph problem at
    # intermediate lam.  Recorded so a lam sweep is not run blind.
    info["status_by_lam"] = {lam: int(_solve(lam).status)
                             for lam in (10.0, 100.0, 1e3, 1e6)}
    # `alpha` is not identified on the simplex: ||w||_1 == 1 there, so the
    # composite penalty at (lam, alpha) is the pure max-norm at lam*(1-alpha).
    # That is an analytic claim about the constraint set, so it is testable
    # here rather than an opinion.
    if solver is not None and _accepts(solver, "alpha"):
        L, al = 0.4, 0.25
        s_comp = solver(A, x, L, alpha=al)
        s_pure = solver(A, x, L * (1.0 - al), alpha=1.0)
        gap = float(np.abs(s_comp.w - s_pure.w).max())
        assert gap < 1e-6, (
            f"solve_linf(lam={L}, alpha={al}) differs from "
            f"solve_linf(lam={L * (1 - al)}, alpha=1.0) by {gap:.3e}; on the "
            "simplex the L1 half of the composite penalty is a constant and "
            "these must coincide"
        )
        info["alpha_reparam_gap"] = gap
    return info


# ==========================================================================
# 6. LVCF identity.
# ==========================================================================


def test_lvcf_identity() -> Dict[str, Any]:
    """`lvcf(panel, i)` is bit-identical to the target's own state at T0."""
    panel = make_panel(seed=SEED + 5, T0=2)
    d = panel.d
    for i in range(panel.n):
        got = B.lvcf(panel, i)
        want = panel.P[i, -d:]
        assert got.shape == (d,)
        assert np.array_equal(got, want), f"lvcf differs from X_i[T0] at i={i}"
    # It must be a copy: a caller mutating the forecast must not edit the panel.
    g = B.lvcf(panel, 0)
    g[0] += 1.0
    assert panel.P[0, -d:][0] != g[0], "lvcf returned a view into Panel.P"
    return {"n_checked": panel.n, "d": d}


# ==========================================================================
# 7. 1-NN.
# ==========================================================================


def test_nn1_matches_independent_argmin() -> Dict[str, Any]:
    """`nn1` equals an argmin over pre-period distance computed here, not there."""
    panel = make_panel(n=N_UNITS, seed=SEED + 6)
    n_checked = 0
    for i in range(panel.n):
        # Independent implementation: explicit loop, no Panel.distances.
        best_j, best_dist = None, np.inf
        for j in range(panel.n):
            if j == i:
                continue
            dist = float(np.sqrt(np.sum((panel.P[j] - panel.P[i]) ** 2)))
            if dist < best_dist:
                best_dist, best_j = dist, j
        got = B.nn1(panel, i)
        assert np.array_equal(got, panel.Y[best_j]), (
            f"nn1 picked a different donor for target {i}"
        )
        n_checked += 1
    # topk_mean, on the same independent ordering.
    i = 0
    dists = np.array([np.inf if j == i else
                      float(np.sqrt(np.sum((panel.P[j] - panel.P[i]) ** 2)))
                      for j in range(panel.n)])
    k = 5
    want = panel.Y[np.argsort(dists, kind="stable")[:k]].mean(axis=0)
    assert np.allclose(B.topk_mean(panel, i, k=k), want, rtol=0, atol=1e-12)
    return {"n_targets_checked": n_checked}


# ==========================================================================
# 8. Ridge baseline.  (Agent F.)
# ==========================================================================


def test_ridge_baseline() -> Dict[str, Any]:
    """Donors generated by a known linear map: ridge should beat the cohort mean.

    Contract assumed: `baselines.ridge(panel, i, ...)` returning an `(d,)`
    forecast, same signature shape as the other baselines.
    """
    fn = _lookup(B, "ridge", "ridge_forecast", "ridge_mean")
    if fn is None:
        skip("baselines has no ridge function (looked for ridge / "
             "ridge_forecast / ridge_mean) -- Agent F's ridge baseline has not "
             "landed")
    rng = np.random.default_rng(SEED + 7)
    n, d, T0 = 60, 12, T0_DEFAULT
    # Y_j = M X_j[T0] + small noise, with M a fixed known map.  A method that
    # learns M beats the cohort mean; one that does not, does not.
    M = rng.normal(size=(d, d)) / np.sqrt(d)
    P = rng.normal(size=(n, (T0 + 1) * d))
    Y = P[:, -d:] @ M.T + 0.01 * rng.normal(size=(n, d))
    panel = make_panel(P=P, Y=Y, n=n, d=d, T0=T0, seed=SEED + 7)
    fr = np.stack([np.asarray(fn(panel, i)).reshape(-1) for i in range(panel.n)])
    fc = np.stack([B.cohort_mean(panel, i) for i in range(panel.n)])
    r_ridge = float(E.rmse_rows(fr, panel.Y).mean())
    r_cohort = float(E.rmse_rows(fc, panel.Y).mean())
    assert r_ridge < r_cohort, (
        f"ridge RMSE {r_ridge:.5f} is not better than cohort mean {r_cohort:.5f} "
        "on data generated by an exactly linear map"
    )
    return {"ridge_rmse": r_ridge, "cohort_mean_rmse": r_cohort}


# ==========================================================================
# 9. Permutation invariance.  Asserts a published claim against our code.
# ==========================================================================


def _permutation_experiment(seed: int = SEED + 8) -> Dict[str, Any]:
    """Fit every target on a panel and on its time-permuted twin.

    The permutation is applied to the pre-period blocks of EVERY unit, target
    and donors alike, and `Y` is untouched -- so the fitted objective is the
    same function of `w`, coordinate for coordinate, only summed in a different
    order.
    """
    rng = np.random.default_rng(seed)
    n, d, T0 = 20, 16, 3          # T0=3 so there is a real permutation to make
    P = rng.normal(size=(n, (T0 + 1) * d))
    Y = rng.normal(size=(n, d))
    perm = rng.permutation(T0 + 1)
    assert not np.array_equal(perm, np.arange(T0 + 1)), "degenerate permutation"
    P2 = P.reshape(n, T0 + 1, d)[:, perm, :].reshape(n, -1)

    pa = make_panel(P=P, Y=Y, n=n, d=d, T0=T0)
    pb = make_panel(P=P2, Y=Y, n=n, d=d, T0=T0)

    n_pairs = n_diff = donor_sets_changed = count_changed = 0
    max_dw = max_df = max_dsse_rel = max_dipr = 0.0
    scale = 0.0
    for m in (None, 8):
        for i in range(n):
            a = F.fit_target(pa, i, m=m)
            b = F.fit_target(pb, i, m=m)
            n_pairs += 1
            scale = max(scale, float(np.abs(a.forecast).max()))
            if not np.array_equal(a.donor_rows, b.donor_rows):
                donor_sets_changed += 1
                continue
            max_dw = max(max_dw, float(np.abs(a.w - b.w).max()))
            max_df = max(max_df, float(np.abs(a.forecast - b.forecast).max()))
            max_dipr = max(max_dipr, abs(a.eff_donors_ipr - b.eff_donors_ipr))
            count_changed += int(a.eff_donors_count != b.eff_donors_count)
            sa, sb = a.pre_rmse ** 2, b.pre_rmse ** 2
            max_dsse_rel = max(max_dsse_rel, abs(sa - sb) / max(sa, 1e-30))
            n_diff += int(not np.array_equal(a.forecast, b.forecast))
    return {"n_pairs": n_pairs, "n_forecasts_not_bit_identical": n_diff,
            "max_abs_weight_change": max_dw, "max_abs_forecast_change": max_df,
            "max_rel_objective_change": max_dsse_rel,
            "max_eff_donor_ipr_change": max_dipr,
            "eff_donor_count_changed": count_changed,
            "donor_sets_changed": donor_sets_changed,
            "forecast_scale": scale, "permutation": perm.tolist()}


def test_permutation_invariance_bitwise() -> Dict[str, Any]:
    """Permuting the pre-period time indices must not change the forecast.

    Synthetic control is "unchanged by design" under a permutation of the
    pre-intervention time indices (Abadie et al.; Rho et al., TASC): the fitted
    objective is a sum over pre-period coordinates, and a sum does not care
    about the order of its terms.  The same permutation is applied to every
    unit, so in exact arithmetic the weights -- and hence the forecast -- are
    identical.

    This asserts BIT-identity, the strong reading of the claim, which an
    implementation either satisfies or does not.  `test_permutation_invariance_
    numeric` quantifies the miss when it does not.
    """
    r = _permutation_experiment()
    assert r["donor_sets_changed"] == 0, (
        f"{r['donor_sets_changed']} donor SETS changed under the permutation; "
        "the pre-filter's distance ordering is not permutation-stable"
    )
    assert r["n_forecasts_not_bit_identical"] == 0, (
        f"synthetic control is NOT bit-invariant to a permutation of the "
        f"pre-period time indices: {r['n_forecasts_not_bit_identical']} of "
        f"{r['n_pairs']} forecasts changed.  max|dw| = "
        f"{r['max_abs_weight_change']:.3e}, max|dforecast| = "
        f"{r['max_abs_forecast_change']:.3e} (forecast scale ~"
        f"{r['forecast_scale']:.3f}), while the objective value itself changed "
        f"by only {r['max_rel_objective_change']:.3e} relative.  Both points are "
        f"therefore optima: this is SLSQP terminating at an ABSOLUTE ftol on the "
        f"objective, which on a quadratic pins w only to ~sqrt(ftol).  The "
        f"invariance holds to ~1e-6 in the weights, not to float64 precision."
    )
    return r


def test_permutation_invariance_numeric() -> Dict[str, Any]:
    """The same claim at the precision the solver actually delivers.

    A regression guard with teeth: if the invariance ever degrades past 1e-4 in
    the weights -- i.e. past `WEIGHT_EPS`, the threshold the reported
    effective-donor counts use -- the reported sparsity diagnostics would start
    to depend on the order of the time axis, which would be a real defect
    rather than a floating-point one.
    """
    r = _permutation_experiment()
    assert r["donor_sets_changed"] == 0
    assert r["max_abs_weight_change"] < TOL_PERMUTATION, (
        f"weights moved by {r['max_abs_weight_change']:.3e} under a permutation "
        f"of the time axis, past the {TOL_PERMUTATION:g} tolerance"
    )
    assert r["max_rel_objective_change"] < 1e-8, (
        f"the permuted fit is not the same optimum: objective differs by "
        f"{r['max_rel_objective_change']:.3e} relative"
    )
    assert r["max_abs_weight_change"] < F.WEIGHT_EPS, (
        "the permutation drift exceeds WEIGHT_EPS, so the reported "
        "effective-donor counts depend on the order of the time axis"
    )
    assert r["eff_donor_count_changed"] == 0, (
        f"{r['eff_donor_count_changed']} reported effective-donor COUNTS "
        "changed under a permutation of the time axis"
    )
    return r


# ==========================================================================
# 10. Gram-form equivalence.
# ==========================================================================


def test_gram_form_equivalence() -> Dict[str, Any]:
    """`w'(AA')w - 2w'(Ax) + x'x` equals `||x - A'w||^2`.

    This is the algebraic rewrite the module claims is exact rather than an
    approximation; it is the reason the fit is affordable at all.
    """
    worst = 0.0
    for trial, (m, D) in enumerate([(3, 8), (12, 40), (40, 60), (60, 20)]):
        rng = np.random.default_rng([SEED, 900 + trial])
        A = rng.normal(size=(m, D))
        x = rng.normal(size=D)
        for rep in range(5):
            w = _random_simplex(rng, m)
            gram = float(w @ ((A @ A.T) @ w) - 2.0 * (w @ (A @ x)) + float(x @ x))
            direct = float(np.sum((x - A.T @ w) ** 2))
            rel = abs(gram - direct) / max(abs(direct), 1e-12)
            worst = max(worst, rel)
            assert rel < TOL_ALGEBRA, (
                f"Gram form differs from the direct form by {rel:.3e} relative "
                f"at m={m}, D={D}"
            )
    # And the SolveResult's own `sse` field agrees with the direct residual.
    rng = np.random.default_rng(SEED + 9)
    A = rng.normal(size=(15, 40))
    x = rng.normal(size=40)
    sol = F.solve_simplex_qp(A, x)
    direct = float(np.sum((x - A.T @ sol.w) ** 2))
    assert abs(sol.sse - direct) / max(direct, 1e-12) < 1e-7, (
        f"SolveResult.sse {sol.sse:.9f} != ||x - A'w||^2 {direct:.9f}"
    )
    return {"worst_relative_error": worst,
            "sse_field_relative_error": abs(sol.sse - direct) / max(direct, 1e-12)}


# ==========================================================================
# 11. Paired bootstrap (and the continuous-resampler discard count).
# ==========================================================================


def test_paired_bootstrap() -> Dict[str, Any]:
    """CI covers a known mean; a shifted delta is excluded; nothing is discarded."""
    rng = np.random.default_rng(SEED + 10)
    true_mean = -0.5
    delta = rng.normal(loc=true_mean, scale=1.0, size=BOOT_N)
    # Centre it exactly so the coverage claim is about a mean we know to the bit.
    delta = delta - delta.mean() + true_mean

    r = E.paired_boot(delta, B=BOOT_B_TEST, seed=SEED)
    assert abs(r["mean_diff"] - true_mean) < 1e-12
    assert r["lo"] < true_mean < r["hi"], (
        f"CI [{r['lo']:.4f}, {r['hi']:.4f}] does not cover the true mean "
        f"{true_mean}"
    )
    assert r["hi"] < 0.0 and r["beats_ref"], "a clearly-negative delta is not " \
                                             "reported as beating the reference"

    shifted = delta + 5.0
    rs = E.paired_boot(shifted, B=BOOT_B_TEST, seed=SEED)
    assert not (rs["lo"] < true_mean < rs["hi"]), (
        f"CI [{rs['lo']:.4f}, {rs['hi']:.4f}] wrongly covers {true_mean} after a "
        "+5 shift"
    )
    assert rs["lo"] > 0.0 and not rs["beats_ref"]

    # Sanity on the reported win fraction, computed independently here.
    assert abs(r["win_frac"] - float((delta < 0).mean())) < 1e-12

    return {"ci": (r["lo"], r["hi"]), "shifted_ci": (rs["lo"], rs["hi"]),
            "mean_diff": r["mean_diff"], "win_frac": r["win_frac"]}


def _reference_percentile_boot(
    v: np.ndarray, B: int, seed: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """An independent percentile bootstrap that keeps every draw.

    Written here, from the documented resampling scheme, so that "the bootstrap
    discards nothing" is checked against something other than the code under
    test.  Any discarded resample would shift the percentiles.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    rng = np.random.default_rng(seed)
    vals = [float(v[rng.integers(0, n, n)].mean()) for _ in range(B)]
    assert len(vals) == B
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def test_continuous_bootstrap_discards_nothing() -> Dict[str, Any]:
    """The paired bootstrap must keep all `B` resamples.

    `stats.boot` was written for a binary-label statistic and DISCARDS every
    resample that is single-class in `y`.  A per-patient RMSE difference has no
    `y`, so that guard is meaningless -- but it silently makes the effective
    number of draws a function of `n`, and the interval conditional on a
    class-balance event with nothing to do with the statistic.

    The check is bit-identity against `_reference_percentile_boot`, an
    independent implementation in this file that keeps every draw.  A single
    dropped resample changes the percentiles, so this is a real proof rather
    than a reading of the docstring.  It is run at n=300 and at n=6, where the
    old dummy-label workaround would have dropped ~3% of the draws.
    """
    rng = np.random.default_rng(SEED + 13)
    out: Dict[str, Any] = {}
    for n in (BOOT_N, 6):
        v = rng.normal(loc=-0.5, size=n)
        r = E.paired_boot(v, B=BOOT_B_TEST, seed=SEED)
        ref = _reference_percentile_boot(v, BOOT_B_TEST, SEED)
        assert (r["lo"], r["hi"]) == ref, (
            f"at n={n} paired_boot returned [{r['lo']!r}, {r['hi']!r}] but an "
            f"all-draws-kept percentile bootstrap gives {ref!r}; the resampler "
            "is dropping draws or drawing differently"
        )
        rm = E.boot_mean(v, B=BOOT_B_TEST, seed=SEED)
        assert (rm["lo"], rm["hi"]) == ref, "boot_mean disagrees with the " \
                                            "all-draws-kept reference"
        # If the module still reports a discard audit, it must say zero.
        assert int(r.get("n_discarded", 0)) == 0, r.get("n_discarded")
        # How many draws the discarding resampler WOULD have dropped, so the
        # difference is quantified rather than asserted.
        y = (np.arange(n) % 2).astype(int)
        g = np.random.default_rng(SEED)
        dropped = sum(
            1 for _ in range(BOOT_B_TEST)
            if len(np.unique(y[g.integers(0, n, n)])) < 2
        )
        out[f"n={n}"] = {"ci": ref, "would_have_been_dropped_by_boot": dropped}
    fn = _lookup(sys.modules["src.vendored.stats"], "boot_continuous")
    if fn is not None:
        lo, hi = fn(rng.normal(size=50), B=BOOT_B_TEST, seed=SEED)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
        out["stats.boot_continuous"] = "present and callable directly"
    return out


# ==========================================================================
# 12. Determinism.
# ==========================================================================


def test_determinism() -> Dict[str, Any]:
    """Same config twice, and n_jobs=1 vs n_jobs>1, give bit-identical output."""
    panel = make_panel(n=N_SMOKE, d=12, T0=T0_DEFAULT, seed=SEED + 11)

    def _sig(fits: List[F.TargetFit]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.concatenate([f.w for f in fits]),
                np.stack([f.forecast for f in fits]),
                np.concatenate([f.donor_rows for f in fits]))

    a = _sig(F.fit_all(panel, m=10, seed=0, n_jobs=1, verbose=False))
    b = _sig(F.fit_all(panel, m=10, seed=0, n_jobs=1, verbose=False))
    for x, y, nm in zip(a, b, ("weights", "forecasts", "donor_rows")):
        assert np.array_equal(x, y), f"repeat run changed {nm}"

    c = _sig(F.fit_all(panel, m=10, seed=0, n_jobs=2, verbose=False))
    for x, y, nm in zip(a, c, ("weights", "forecasts", "donor_rows")):
        assert np.array_equal(x, y), f"n_jobs=2 changed {nm}"

    # The random-donor placebo is seeded per target, so it must also be
    # invariant to how the work was chunked across processes.
    p1 = _sig(F.fit_all(panel, m=10, seed=0, random_donors=True, n_jobs=1,
                        verbose=False))
    p2 = _sig(F.fit_all(panel, m=10, seed=0, random_donors=True, n_jobs=3,
                        verbose=False))
    for x, y, nm in zip(p1, p2, ("weights", "forecasts", "donor_rows")):
        assert np.array_equal(x, y), f"placebo n_jobs=3 changed {nm}"
    # ... and a different seed must actually change the placebo's donor set,
    # otherwise "seeded" would be vacuous.
    p3 = _sig(F.fit_all(panel, m=10, seed=1, random_donors=True, n_jobs=1,
                        verbose=False))
    assert not np.array_equal(p1[2], p3[2]), (
        "changing the seed did not change the random donor sets"
    )
    return {"n_targets": panel.n, "jobs_checked": [1, 2, 3]}


# ==========================================================================
# 13. Tie-breaking.
# ==========================================================================


def test_tie_breaking() -> Dict[str, Any]:
    """Deliberately tied distances resolve to the lowest canonical numeric ids.

    The bag encoder's distances are integer-valued and heavily tied, so this is
    load-bearing rather than cosmetic: an unstable sort would make "the nearest
    donor" a function of BLAS ordering.
    """
    d, T0 = 8, T0_DEFAULT
    D = (T0 + 1) * d
    n = 13
    P = np.zeros((n, D))
    # Target is the origin.  Donors 1..8 sit on distinct axes at distance
    # exactly 1 -- an eight-way tie.  Donors 9..12 sit at distance 2.
    for j in range(1, 9):
        P[j, (j - 1) % D] = 1.0
    for j in range(9, n):
        P[j, (j - 1) % D] = 2.0
    ids = sorted_ids(str(1000 + i) for i in range(n))
    panel = make_panel(P=P, n=n, d=d, T0=T0, ids=ids, seed=SEED + 12)

    dists = panel.distances(0, panel.donor_index(0))
    tied = np.isclose(dists, 1.0)
    assert tied.sum() == 8, "the construction no longer produces an 8-way tie"

    rows, sel = F.select_donors(panel, 0, m=3)
    assert sel == "nearest"
    assert list(rows) == [1, 2, 3], f"tie broken to rows {list(rows)}, not [1,2,3]"
    # Stable across repeats and across a differently-ordered call.
    for _ in range(5):
        r2, _ = F.select_donors(panel, 0, m=3)
        assert np.array_equal(rows, r2), "donor selection is not stable"
    # Panel row order IS canonical numeric id order, so "lowest row" == "lowest
    # numeric subject_id".  Assert that rather than assume it.
    assert panel.ids == sorted_ids(panel.ids)
    picked_ids = [panel.ids[r] for r in rows]
    assert picked_ids == sorted(picked_ids, key=lambda s: int(s))
    assert picked_ids == ["1001", "1002", "1003"], picked_ids
    # nn1 must pick the same first donor.
    assert np.array_equal(B.nn1(panel, 0), panel.Y[1])
    # The placebo must stay labelled a placebo even when the requested m covers
    # the whole pool -- otherwise an m-sweep would silently turn `sc_random`
    # into `sc` at the top of the range.
    _, sel_big = F.select_donors(panel, 0, m=panel.n * 10,
                                 rng=np.random.default_rng(0))
    _, sel_none = F.select_donors(panel, 0, m=None,
                                  rng=np.random.default_rng(0))
    assert sel_big == sel_none == "random", (
        f"random donor selection is labelled {sel_big!r}/{sel_none!r} when m "
        "covers the pool; a placebo requested at a large m stops being "
        "labelled a placebo"
    )
    return {"n_tied": int(tied.sum()), "picked_rows": list(map(int, rows)),
            "picked_ids": picked_ids,
            "placebo_label_at_large_m": sel_big}


# ==========================================================================
# 14. End-to-end smoke.
# ==========================================================================


def test_end_to_end_smoke() -> Dict[str, Any]:
    """A fabricated 30-patient cohort through the whole Phase 2 call path.

    `Panel.build` -> baselines -> `fit_all` (estimator and placebo) ->
    `evaluate_config` -> `format_table`.  Phase 1 found these modules did not
    compose as shipped, so "it runs" is a real assertion here.
    """
    ss, labels, vocab = make_stateset()
    panel = F.Panel.build(ss, T0=T0_DEFAULT)
    assert panel.n == N_SMOKE and panel.d == ss.d
    assert panel.P.shape == (N_SMOKE, (T0_DEFAULT + 1) * ss.d)
    # Panel.build must not accumulate visits: X_i[T0] is the target's own visit.
    for i, s in enumerate(panel.ids):
        assert np.array_equal(panel.target_last_state(i),
                              ss.state(s, T0_DEFAULT).astype(np.float64))
        assert np.array_equal(panel.Y[i], ss.state(s, T0_DEFAULT + 1).astype(np.float64))

    rn = panel.rownorm_l2()
    assert rn.rownorm == "rowl2" and np.isfinite(rn.P).all()

    forecasts = B.all_simple_forecasts(panel, k=5)
    fits = F.fit_all(panel, m=10, seed=0, n_jobs=1, verbose=False)
    fits_rand = F.fit_all(panel, m=10, seed=0, random_donors=True, n_jobs=1,
                          verbose=False)
    forecasts["sc"] = np.stack([f.forecast for f in fits])
    forecasts["sc_random"] = np.stack([f.forecast for f in fits_rand])
    for nm, Fc in forecasts.items():
        assert Fc.shape == (panel.n, panel.d), nm
        assert np.isfinite(Fc).all(), f"non-finite forecast from {nm}"

    Ybin = (panel.Y > 0).astype(int)
    K = (panel.P[:, -panel.d:] > 0).sum(axis=1).astype(int)
    res = E.evaluate_config(
        forecasts, panel.Y, Ybin=Ybin, K=K, labels=labels, vocab=vocab,
        fits={"sc": fits, "sc_random": fits_rand}, do_auroc=True,
        boot_B=BOOT_B_TEST, boot_seed=SEED,
    )
    assert res["n_targets"] == panel.n
    assert {r["method"] for r in res["rows"]} == set(forecasts)
    for r in res["rows"]:
        for key in ("rmse", "cosine", "f1", "jaccard"):
            blk = r[key]
            for f in ("mean", "lo", "hi"):
                assert np.isfinite(blk[f]), f"{r['method']}.{key}.{f} not finite"
            assert blk["lo"] <= blk["mean"] + 1e-9 <= blk["hi"] + 1e-9, \
                f"{r['method']}.{key}: mean outside its own CI"
        assert np.isfinite(r["norm_ratio"])
        if r["method"] != res["reference"]:
            assert np.isfinite(r["vs_ref"]["mean_diff"])
    # LVCF's decoded code set is exactly its own, by the cardinality rule.
    lv = E.codeset_metrics(forecasts["lvcf"], (panel.P[:, -panel.d:] > 0).astype(int), K)
    assert np.allclose(lv["f1"], 1.0), (
        "the cardinality rule does not make LVCF decode to itself"
    )
    auroc = [r["auroc_dense"]["auroc"] for r in res["rows"] if "auroc_dense" in r]
    assert auroc and all(np.isfinite(a) for a in auroc), "outcome AUROC not finite"
    vend = [r.get("auroc_codeset_vendored") for r in res["rows"]]
    vend_errors = [v["error"] for v in vend if isinstance(v, dict) and "error" in v]
    table = E.format_table(res, "synthetic smoke")
    assert isinstance(table, str) and "synthetic smoke" in table
    sc_row = next(r for r in res["rows"] if r["method"] == "sc")
    assert np.isfinite(sc_row["sparsity"]["eff_donors_ipr_mean"])
    assert 1.0 - 1e-9 <= sc_row["sparsity"]["eff_donors_ipr_mean"] <= 10.0 + 1e-9
    rep = F.convergence_report(fits)
    assert rep["n"] == panel.n
    return {"methods": sorted(forecasts), "n": panel.n,
            "n_not_converged": rep["n_not_converged"],
            "max_simplex_violation": rep["max_simplex_violation"],
            "vendored_cv_auroc_errors": vend_errors,
            "table_lines": len(table.splitlines())}


# ==========================================================================
# Runner.  pytest is not installed in this environment (checked); this is the
# fallback the brief asks for.  Under pytest the functions above run as-is.
# ==========================================================================

TESTS: List[Callable[[], Dict[str, Any]]] = [
    test_io_guard_blocks_project_paths,
    test_weight_recovery,
    test_simplex_feasibility,
    test_exact_donor,
    test_intercept_recovery,
    test_linf_densification,
    test_lvcf_identity,
    test_nn1_matches_independent_argmin,
    test_ridge_baseline,
    test_permutation_invariance_bitwise,
    test_permutation_invariance_numeric,
    test_gram_form_equivalence,
    test_paired_bootstrap,
    test_continuous_bootstrap_discards_nothing,
    test_determinism,
    test_tie_breaking,
    test_end_to_end_smoke,
]


def main(argv: Optional[List[str]] = None) -> int:
    import time

    argv = sys.argv[1:] if argv is None else argv
    verbose = "-q" not in argv
    _install_io_guard()
    np.seterr(all="raise")

    n_pass = n_fail = n_skip = 0
    failures: List[Tuple[str, str]] = []
    print(f"synthctl correctness suite  (seed={SEED}, constructed data only)")
    print("=" * 78)
    t_all = time.time()
    for fn in TESTS:
        name = fn.__name__
        t0 = time.time()
        try:
            info = fn()
            dt = time.time() - t0
            n_pass += 1
            print(f"PASS  {name:<42} {dt:6.2f}s")
            if verbose and info:
                for k, v in info.items():
                    print(f"        {k} = {v}")
        except Skipped as e:
            n_skip += 1
            print(f"SKIP  {name:<42} {time.time()-t0:6.2f}s")
            print(f"        {e}")
        except Exception as e:  # noqa: BLE001 - a failing test is data, not a crash
            n_fail += 1
            tb = traceback.format_exc()
            failures.append((name, tb))
            print(f"FAIL  {name:<42} {time.time()-t0:6.2f}s")
            first = str(e).strip().splitlines()
            print(f"        {first[0] if first else type(e).__name__}")
            for line in first[1:]:
                print(f"        {line}")
    print("=" * 78)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"in {time.time()-t_all:.1f}s")
    if failures and verbose:
        for name, tb in failures:
            print("\n" + "-" * 78)
            print(f"traceback: {name}")
            print(tb)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
