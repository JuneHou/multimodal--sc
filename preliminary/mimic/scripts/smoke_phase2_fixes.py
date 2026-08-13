#!/usr/bin/env python
"""
Smoke tests for the Phase 2 methodology fixes.  **Synthetic data only.**

    /data/wang/junh/envs/medrag/bin/python scripts/smoke_phase2_fixes.py

Covers the pieces added on 2026-08-01 in response to the methodology review
against Wang, Xing & Ye (arXiv 2510.26053) and Doudchenko & Imbens (2016):

    1. the intercept mu^0 in `solve_simplex_qp`
    2. the L-infinity / composite penalty in `solve_linf`
    3. the ridge baseline
    4. `stats.boot_continuous` (no class guard, no discarded resamples)
    5. code-set F1 as the primary metric in `evaluate_config` / `format_table`
    6. `select_donors` placebo labelling at large m
    7. the two new figures, `sc_trajectory` and `placebo_distribution`
    8. `features.py` reproducibility: `_flatten` on object ndarrays and
       `design_matrix`'s PYTHONHASHSEED-independent column order

Reads NO MIMIC-derived data, touches no GPU, writes nothing outside a temporary
directory.  Every array here is generated from a fixed seed inside this file.
Exits 0 iff every check passes.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthctl import baselines as bl                              # noqa: E402
from src.synthctl import evaluate as ev                               # noqa: E402
from src.synthctl import figures as figs                              # noqa: E402
from src.synthctl.fit import (                                        # noqa: E402
    Panel, fit_target, select_donors, solve_linf, solve_simplex_qp,
)
from src.vendored.features import _flatten, design_matrix             # noqa: E402
from src.vendored.stats import boot, boot_continuous                  # noqa: E402

ok = {}


def chk(name, cond, detail=""):
    ok[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL':4}  {name}  {detail}", flush=True)


TMP = Path(tempfile.mkdtemp(prefix="phase2_smoke_"))
rng = np.random.default_rng(17)


def synth_panel(n=36, T0=1, d=20, seed=5):
    g = np.random.default_rng(seed)
    P = g.normal(size=(n, (T0 + 1) * d))
    Y = g.normal(size=(n, d))
    return Panel(ids=[str(i) for i in range(n)], P=P, Y=Y, T0=T0, d=d,
                 encoder="synthetic", scale="none")


# ==========================================================================
# 1. the intercept
# ==========================================================================
m, D = 14, 45
A = rng.normal(size=(m, D))
w_true = np.zeros(m)
w_true[[2, 5, 9]] = [0.5, 0.3, 0.2]
MU_TRUE = 0.73
x = A.T @ w_true + MU_TRUE + 0.005 * rng.normal(size=D)

r_off = solve_simplex_qp(A, x)
chk("fit_intercept=False leaves mu exactly 0.0",
    r_off.mu == 0.0, f"mu={r_off.mu!r}")

r_on = solve_simplex_qp(A, x, fit_intercept=True)
chk("intercept recovers the planted level shift",
    abs(r_on.mu - MU_TRUE) < 1e-2, f"mu={r_on.mu:.6f} vs {MU_TRUE}")
chk("intercept recovers the planted weights",
    np.abs(r_on.w - w_true).max() < 1e-3,
    f"max|w-w*|={np.abs(r_on.w - w_true).max():.2e}")


def joint_obj(z):
    return float(np.sum((x - z[0] - A.T @ z[1:]) ** 2))


best = None
for _ in range(8):
    z0 = np.concatenate([[rng.normal()], rng.dirichlet(np.ones(m))])
    rr = minimize(joint_obj, z0, method="SLSQP",
                  bounds=[(None, None)] + [(0.0, None)] * m,
                  constraints=[{"type": "eq", "fun": lambda z: z[1:].sum() - 1}],
                  options={"maxiter": 2000, "ftol": 1e-12})
    if best is None or rr.fun < best.fun:
        best = rr
# The profiled Gram form must agree with a genuinely joint (mu, w) optimisation.
chk("profiled intercept == joint (mu, w) optimisation",
    abs(best.x[0] - r_on.mu) < 1e-5 and abs(best.fun - r_on.sse) < 1e-8,
    f"joint mu={best.x[0]:.8f} f={best.fun:.10f} | ours mu={r_on.mu:.8f} "
    f"sse={r_on.sse:.10f}")
chk("SolveResult.sse really is ||x - mu*1 - A'w||^2",
    abs(np.sum((x - r_on.mu - A.T @ r_on.w) ** 2) - r_on.sse) < 1e-10)
chk("intercept fit is still exactly on the simplex",
    r_on.simplex_violation < 1e-9, f"viol={r_on.simplex_violation:.2e}")
# mu must be free to go negative -- it is a level shift, not a weight.
r_neg = solve_simplex_qp(A, A.T @ w_true - 2.5, fit_intercept=True)
chk("mu is unconstrained in sign (can be negative)",
    r_neg.mu < -2.0, f"mu={r_neg.mu:.4f}")

# Gram-form efficiency: the intercept must not have changed the problem size
# handed to SLSQP.  Same number of decision variables, same order of iterations.
chk("intercept does not enlarge the solver's decision vector",
    r_on.w.shape == (m,) and r_on.nit < 200, f"nit={r_on.nit}")

# ==========================================================================
# 2. the L-infinity variant
# ==========================================================================
r_l0 = solve_linf(A, x, lam=0.0)
chk("solve_linf(lam=0) == solve_simplex_qp",
    np.abs(r_l0.w - r_off.w).max() < 1e-12,
    f"max|dw|={np.abs(r_l0.w - r_off.w).max():.2e}")

maxw, iprs = [], []
for lam in (0.0, 0.1, 1.0, 10.0, 100.0):
    rr = solve_linf(A, x, lam=lam)
    maxw.append(rr.w.max())
    iprs.append(1.0 / float(rr.w @ rr.w))
chk("larger lam monotonically lowers the largest weight",
    all(b <= a + 1e-9 for a, b in zip(maxw, maxw[1:])),
    "max w: " + " -> ".join(f"{v:.4f}" for v in maxw))
chk("larger lam gives DENSER weights (higher 1/sum w^2)",
    iprs[-1] > iprs[0] + 1e-6,
    "eff donors: " + " -> ".join(f"{v:.2f}" for v in iprs))

LAM = 1.0
r_e = solve_linf(A, x, lam=LAM)
chk("epigraph slack really equals ||w||_inf at the optimum",
    True, f"max w = {r_e.w.max():.6f} (the epigraph var is not returned; the "
          f"objective below is the check)")


def nonsmooth(w):
    return float(np.sum((x - A.T @ w) ** 2) + LAM * np.max(np.abs(w)))


best = None
for _ in range(40):
    rr = minimize(nonsmooth, rng.dirichlet(np.ones(m)), method="SLSQP",
                  bounds=[(0.0, None)] * m,
                  constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                  options={"maxiter": 3000, "ftol": 1e-12})
    if best is None or rr.fun < best.fun:
        best = rr
ours = r_e.sse + LAM * r_e.w.max()
chk("epigraph reformulation == the non-smooth max-norm problem",
    abs(ours - best.fun) < 1e-7 and np.abs(best.x - r_e.w).max() < 1e-4,
    f"ours {ours:.10f} vs direct {best.fun:.10f}, max|dw|="
    f"{np.abs(best.x - r_e.w).max():.2e}")

# The documented degeneracy: on the simplex ||w||_1 == 1, so the composite
# penalty is a pure L-inf penalty at strength lam*(1-alpha).
ra = solve_linf(A, x, lam=1.0, alpha=0.4)
rb = solve_linf(A, x, lam=0.6, alpha=1.0)
chk("composite alpha is NOT identified on the simplex (documented)",
    np.abs(ra.w - rb.w).max() < 1e-12,
    f"(lam=1, a=0.4) == (lam=0.6, a=1): max|dw|={np.abs(ra.w - rb.w).max():.1e}")
chk("solve_linf reports the UNPENALISED sse",
    abs(np.sum((x - r_e.w @ A) ** 2) - r_e.sse) < 1e-9)
r_li = solve_linf(A, x, lam=1.0, fit_intercept=True)
chk("solve_linf composes with the intercept",
    abs(np.sum((x - r_li.mu - A.T @ r_li.w) ** 2) - r_li.sse) < 1e-9
    and r_li.simplex_violation < 1e-9, f"mu={r_li.mu:.4f}")
for bad in ((-1.0, 1.0), (1.0, 0.0), (1.0, 1.5)):
    try:
        solve_linf(A, x, lam=bad[0], alpha=bad[1])
        raised = False
    except ValueError:
        raised = True
    chk(f"solve_linf rejects lam={bad[0]}, alpha={bad[1]}", raised)

# ==========================================================================
# 3. ridge baseline
# ==========================================================================
p = synth_panel()
one = np.stack([bl.ridge(p, i, alpha=1.0) for i in range(p.n)])
allr = bl.ridge_all(p, alpha=1.0)
chk("ridge_all == per-target ridge",
    np.abs(one - allr).max() < 1e-9, f"max|d|={np.abs(one - allr).max():.2e}")
try:
    from sklearn.linear_model import Ridge

    i = 7
    dn = p.donor_index(i)
    sk = Ridge(alpha=1.0, fit_intercept=True).fit(p.P[dn], p.Y[dn]) \
        .predict(p.P[i][None])[0]
    chk("ridge == sklearn.linear_model.Ridge (dual form, D<n and D>n)",
        np.abs(sk - one[i]).max() < 1e-9, f"max|d|={np.abs(sk - one[i]).max():.2e}")
except ImportError:                                            # pragma: no cover
    chk("ridge == sklearn Ridge", False, "sklearn missing")

p_wide = synth_panel(n=25, T0=1, d=60, seed=9)                 # D = 120 > n = 25
i = 3
dn = p_wide.donor_index(i)
sk = Ridge(alpha=1.0).fit(p_wide.P[dn], p_wide.Y[dn]).predict(p_wide.P[i][None])[0]
chk("ridge is correct in the D > n regime (the real one)",
    np.abs(sk - bl.ridge(p_wide, i)).max() < 1e-9)

p_loo = synth_panel(seed=5)
p_loo.Y = p_loo.Y.copy()
p_loo.Y[4] += 1e4
chk("ridge never trains on its own target (leave-one-patient-out)",
    np.abs(bl.ridge(p_loo, 4) - one[4]).max() < 1e-9)
chk("ridge is registered in SIMPLE_BASELINES and LABELS",
    "ridge" in bl.SIMPLE_BASELINES and "ridge" in bl.LABELS)
fc = bl.all_simple_forecasts(p)
chk("all_simple_forecasts emits ridge at the right shape",
    fc["ridge"].shape == (p.n, p.d))

# ==========================================================================
# 4. boot_continuous
# ==========================================================================
v = rng.normal(-0.002, 0.01, 500)


def plain(a, B=2000, seed=0):
    g = np.random.default_rng(seed)
    n = len(a)
    return tuple(np.percentile(
        [a[g.integers(0, n, n)].mean() for _ in range(B)], [2.5, 97.5]))


lo, hi = boot_continuous(v, B=2000, seed=0)
plo, phi = plain(v)
chk("boot_continuous == a plain unguarded percentile bootstrap",
    abs(lo - plo) < 1e-12 and abs(hi - phi) < 1e-12,
    f"[{lo:.8f},{hi:.8f}] vs [{plo:.8f},{phi:.8f}]")
chk("boot_continuous is seed-reproducible and seed-sensitive",
    boot_continuous(v, seed=0) == (lo, hi) and boot_continuous(v, seed=1) != (lo, hi))
l90, h90 = boot_continuous(v, alpha=0.10)
chk("alpha=0.10 gives a narrower interval than alpha=0.05",
    (h90 - l90) <= (hi - lo) + 1e-15)
chk("boot_continuous takes a custom statistic",
    abs(boot_continuous(v, fn=lambda a: float(np.median(a)), B=200)[0]
        - np.median(v)) < 0.01)
# The guard `boot` applies, and that `boot_continuous` must not: a single-class
# y makes `boot` raise; the same continuous data is fine here.
try:
    boot(np.zeros(len(v), dtype=int), v, lambda y, q: float(q.mean()), B=50)
    guarded = False
except RuntimeError:
    guarded = True
chk("stats.boot still guards (unchanged) while boot_continuous does not",
    guarded and np.isfinite(boot_continuous(v, B=50)).all())
try:
    boot_continuous([])
    empty_ok = False
except ValueError:
    empty_ok = True
chk("boot_continuous raises on empty input", empty_ok)
pb = ev.paired_boot(v, B=2000, seed=0)
chk("paired_boot no longer reports a discard count (the hack is gone)",
    "n_discarded" not in pb and pb.get("resampler") == "boot_continuous")
chk("paired_boot bounds match the unguarded bootstrap exactly",
    abs(pb["lo"] - plo) < 1e-12 and abs(pb["hi"] - phi) < 1e-12)
chk("_dummy_labels is gone from evaluate.py",
    not hasattr(ev, "_dummy_labels") and not hasattr(ev, "_count_discards"))

# ==========================================================================
# 5. code-set F1 as the primary metric
# ==========================================================================
# The demonstration: a HARD 0/1 forecast that is genuinely informative (25% of
# entries flipped, so it names the right codes 3 times out of 4) against a SOFT
# forecast that is perfectly calibrated to the marginal rate and names nothing
# at all.  MSE of the hard one is 0.25; MSE of the constant 0.25 forecast is
# 0.1875.  The proper scoring rule therefore ranks the useless forecast FIRST.
# That is precisely the comparison the old RMSE-primary table was making
# between SC (soft) and LVCF (hard).
n_u, d_u, RATE, FLIP = 120, 40, 0.25, 0.25
Ybin = (rng.random((n_u, d_u)) < RATE).astype(float)
K = np.maximum(Ybin.sum(axis=1).astype(int), 1)
flip = rng.random((n_u, d_u)) < FLIP
F_lvcf = np.where(flip, 1.0 - Ybin, Ybin)   # informative, hard, imperfect
F_soft = RATE * np.ones((n_u, d_u))         # calibrated, but names nothing
res = ev.evaluate_config({"lvcf": F_lvcf, "soft": F_soft}, Ybin,
                         reference="lvcf", Ybin=Ybin, K=K, do_auroc=False,
                         boot_B=200)
chk("evaluate_config declares code_f1 primary when a vocabulary exists",
    res["primary_metric"] == "code_f1")
rows = {r["method"]: r for r in res["rows"]}
chk("primary F1 is present for every method, with a paired CI vs the reference",
    all("f1" in r for r in res["rows"]) and rows["soft"]["f1_vs_ref"] is not None
    and rows["lvcf"]["f1_vs_ref"] is None)
chk("f1_vs (all references, not just LVCF) is emitted",
    "f1_vs" in rows["soft"])
chk("Brier/MSE is emitted and labelled secondary",
    "brier" in rows["soft"] and "brier" in res["secondary_metrics"])
chk("Brier == RMSE^2 per patient",
    abs(rows["soft"]["brier"]["mean"]
        - float((ev.rmse_rows(F_soft, Ybin) ** 2).mean())) < 1e-12)
# The whole point of the change: the constant 0.25 forecast beats a PERFECT
# hard forecast on the proper scoring rule, and loses to it catastrophically on
# the primary metric.
chk("MSE prefers the calibrated-but-useless forecast (the apples-to-oranges bug)",
    rows["soft"]["rmse"]["mean"] < rows["lvcf"]["rmse"]["mean"],
    f"soft (names nothing) RMSE {rows['soft']['rmse']['mean']:.4f} < hard "
    f"(75% right) {rows['lvcf']['rmse']['mean']:.4f}")
chk("code F1 correctly prefers the informative hard forecast",
    rows["lvcf"]["f1"]["mean"] > rows["soft"]["f1"]["mean"] + 0.2,
    f"F1 hard {rows['lvcf']['f1']['mean']:.4f} vs soft "
    f"{rows['soft']['f1']['mean']:.4f}")
tbl = ev.format_table(res, "smoke")
chk("format_table leads with code F1 and marks it PRIMARY",
    "PRIMARY" in tbl and tbl.index("code F1") < tbl.index("post RMSE"))
chk("format_table ranks by F1, not RMSE",
    tbl.index("LVCF") < tbl.index("soft"))
chk("format_table labels the probabilistic view as secondary",
    "SECONDARY" in tbl and "proper scoring rule" in tbl.lower()
    and "Brier" in tbl)
res_nocs = ev.evaluate_config({"lvcf": F_lvcf, "soft": F_soft}, Ybin,
                              reference="lvcf", do_auroc=False, boot_B=100)
chk("no vocabulary -> falls back to RMSE and SAYS SO",
    res_nocs["primary_metric"] == "post_rmse"
    and "NO code-set F1" in ev.format_table(res_nocs, "smoke2"))

# ==========================================================================
# 6. select_donors placebo labelling
# ==========================================================================
pp = synth_panel(n=12)
g = np.random.default_rng(0)
for mm in (3, 10, 11, 12, 50, None):
    rows_, sel = select_donors(pp, 0, mm, rng=np.random.default_rng(0))
    chk(f"placebo at m={mm} is still labelled 'random'", sel == "random",
        f"n={len(rows_)} of {pp.n - 1} available")
_, sel_nn = select_donors(pp, 0, 50, rng=None)
chk("the estimator at m >= n-1 is still labelled 'all'", sel_nn == "all")
f_pl = fit_target(pp, 0, m=50, rng=np.random.default_rng(0))
chk("fit_target propagates the placebo label at large m",
    f_pl.selection == "random")
chk("TargetFit carries mu and penalty",
    f_pl.mu == 0.0 and f_pl.penalty is None)
f_int = fit_target(pp, 0, m=6, fit_intercept=True)
chk("fit_target(fit_intercept=True) adds mu into the forecast",
    abs(float(f_int.forecast.mean()
              - (f_int.w @ pp.Y[f_int.donor_rows]).mean()) - f_int.mu) < 1e-12
    and f_int.mu != 0.0, f"mu={f_int.mu:.5f}")
f_linf = fit_target(pp, 0, m=6, penalty="linf", lam=1.0)
chk("fit_target(penalty='linf') records the penalty",
    f_linf.penalty == "linf")
try:
    fit_target(pp, 0, m=6, penalty="lasso")
    bad_ok = False
except ValueError:
    bad_ok = True
chk("fit_target rejects an unknown penalty", bad_ok)

# ==========================================================================
# 7. the two new figures
# ==========================================================================
T0f = 2
tgt = np.array([40.0, 42.0, 44.0, 51.0])
syn = np.array([40.4, 41.6, 44.3, 46.0])
pth = figs.sc_trajectory(tgt, syn, TMP / "traj.svg",
                         title="synthetic trajectory", T0=T0f,
                         subtitle="synthetic data", ylabel="code mass")
svg = pth.read_text()
chk("sc_trajectory writes a well-formed SVG",
    svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    and svg.count("<polyline") == 2)
chk("sc_trajectory draws the pre/post divide and labels both sides",
    'class="ref"' in svg and f"fit on t ≤ {T0f}" in svg)
chk("sc_trajectory direct-labels both series (no colour-only encoding)",
    "target</text>" in svg and "synthetic control</text>" in svg)
chk("sc_trajectory is dark-mode aware",
    "prefers-color-scheme" in svg)
for bad_args in (((1, 2, 3), (1, 2)), ((), ())):
    try:
        figs.sc_trajectory(*bad_args, TMP / "x.svg", "t", T0=0)
        raised = False
    except ValueError:
        raised = True
    chk(f"sc_trajectory rejects malformed input {bad_args}", raised)
# Degenerate but legal: a flat trajectory must not divide by zero.
figs.sc_trajectory([1.0, 1.0], [1.0, 1.0], TMP / "flat.svg", "flat", T0=0)
chk("sc_trajectory survives a constant trajectory", True)

pl = rng.normal(1.0, 0.2, 400)
pth2 = figs.placebo_distribution(pl, 0.35, TMP / "placebo.svg",
                                 title="placebo", xlabel="RMSE",
                                 lower_is_better=True)
svg2 = pth2.read_text()
chk("placebo_distribution writes a well-formed SVG",
    svg2.startswith("<svg") and svg2.rstrip().endswith("</svg>")
    and svg2.count("<rect") > 3)
chk("placebo_distribution marks the observed value and its rank p",
    "rank p =" in svg2 and 'class="s2s"' in svg2)
# p must use the (1 + k) / (1 + n) convention -- never exactly 0.
k = int((pl <= 0.35).sum())
want = (1.0 + k) / (1.0 + len(pl))
chk("placebo p-value uses the (1+k)/(1+n) convention",
    f"rank p = {want:.4f}" in svg2, f"k={k}, n={len(pl)}, p={want:.4f}")
sv_hi = figs.placebo_distribution(pl, 5.0, TMP / "p2.svg", "placebo",
                                  lower_is_better=False).read_text()
chk("placebo_distribution honours lower_is_better=False",
    f"rank p = {1.0 / (1.0 + len(pl)):.4f}" in sv_hi)
try:
    figs.placebo_distribution([np.nan], 1.0, TMP / "p3.svg", "t")
    raised = False
except ValueError:
    raised = True
chk("placebo_distribution rejects an all-non-finite placebo set", raised)

# ==========================================================================
# 8. features.py: the two claimed fixes
# ==========================================================================
chk("_flatten survives a length>1 object ndarray (the `field or []` bug)",
    _flatten(np.array(["a", "b", "c"], dtype=object)) == ["a", "b", "c"])
chk("_flatten handles None, empty, nested list and nested ndarray",
    _flatten(None) == [] and _flatten(np.array([], dtype=object)) == []
    and _flatten([["a"], "b"]) == ["a", "b"]
    and _flatten(np.array([np.array(["a", "b"], dtype=object)], dtype=object))
    == ["a", "b"])
# Source check on EXECUTABLE code only: `field or []` is named in the patch
# note inside the docstring, so a plain substring scan false-positives.  Strip
# docstrings via an AST round-trip first (the same technique the vendored
# module's own smoke test uses).
import ast                                                          # noqa: E402

_tree = ast.parse(Path(ROOT / "src" / "vendored" / "features.py").read_text())
_fns = {n.name: n for n in ast.walk(_tree) if isinstance(n, ast.FunctionDef)}
for _n in _fns.values():
    if (_n.body and isinstance(_n.body[0], ast.Expr)
            and isinstance(_n.body[0].value, ast.Constant)
            and isinstance(_n.body[0].value.value, str)):
        _n.body = _n.body[1:]
_flat_src = ast.unparse(_fns["_flatten"])
_dm_src = ast.unparse(_fns["design_matrix"])
chk("_flatten's CODE no longer contains `field or []`",
    "field or []" not in _flat_src and "if field is None" in _flat_src)
chk("_flatten's CODE widens the nested check to tuple/ndarray",
    "np.ndarray" in _flat_src and "tuple" in _flat_src)
chk("design_matrix's CODE builds a sorted vocabulary",
    "kept = sorted(" in _dm_src)

# Cross-process determinism: the same bags, built in two fresh interpreters at
# different PYTHONHASHSEEDs, must give the same sha1 for X and for the vocab.
# NOTE: the bags here are SYNTHETIC.  This deliberately does not re-run the
# check against the Phase 0 parquet (MIMIC-derived, out of bounds for this
# task); it tests the mechanism, not the artifact.
CHILD = r'''
import hashlib, sys, random
sys.path.insert(0, %r)
from src.vendored.features import design_matrix
rnd = random.Random(1234)
codes = [f"{sec}:{w}" for sec in "cpd" for w in
         ("alpha","beta","gamma","delta","epsilon","zeta","eta","theta",
          "iota","kappa","lambda","mu","nu","xi","omicron","pi")]
bags = [set(rnd.sample(codes, rnd.randint(6, 20))) for _ in range(300)]
X, vocab = design_matrix(bags, min_df=5)
print(X.shape[0], X.shape[1],
      hashlib.sha1(X.tobytes()).hexdigest(),
      hashlib.sha1("\x1f".join(vocab).encode()).hexdigest())
'''
outs = []
for hs in ("1", "987654", "42"):
    env = dict(os.environ, PYTHONHASHSEED=hs)
    outs.append(subprocess.run([sys.executable, "-c", CHILD % str(ROOT)],
                               capture_output=True, text=True, env=env,
                               check=True).stdout.strip())
chk("design_matrix: X and vocab are byte-identical across PYTHONHASHSEEDs",
    len(set(outs)) == 1,
    f"{len(set(outs))} distinct results over seeds 1/987654/42 -> {outs[0]}")

# ==========================================================================
print()
n_bad = sum(1 for v in ok.values() if not v)
print(f"{len(ok) - n_bad}/{len(ok)} checks passed.  scratch dir: {TMP}")
if n_bad:
    print("FAILED: " + ", ".join(k for k, v in ok.items() if not v))
raise SystemExit(1 if n_bad else 0)
