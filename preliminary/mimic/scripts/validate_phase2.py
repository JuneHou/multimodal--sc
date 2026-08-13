"""Phase 2 correctness gate: 23 independent checks on `src/synthctl`.

This is NOT a metric run.  It checks that the ESTIMATOR is what it claims to be
-- exactly simplex-constrained, actually optimal, warm-start invariant, row-
aligned to Phase 1's public API, and bit-identical under any `--jobs` -- and
that the EVALUATION is what it claims to be: the vendored bootstrap discards no
resamples, every method receives the identical cardinality budget, and the
row-L2 ablation genuinely moves the donor neighbourhood.

    /data/wang/junh/envs/medrag/bin/python mimic/scripts/validate_phase2.py

Exits 0 iff every check passes.  ~90 s, CPU only, no GPU and no network.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic")
sys.path.insert(0, str(ROOT))
from src.datasets.state import load_states
from src.synthctl.fit import Panel, fit_all, fit_target, solve_simplex_qp
from src.synthctl import baselines as bl
from src.synthctl import evaluate as ev
from src.vendored.stats import boot

ok = {}


def chk(name, cond, detail=""):
    ok[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


# ---- 1. solver recovers a known convex combination ------------------------
rng = np.random.default_rng(7)
A = rng.normal(size=(40, 300))
w_true = np.zeros(40)
w_true[[3, 11, 27]] = [0.5, 0.3, 0.2]
x = A.T @ w_true
r = solve_simplex_qp(A, x)
# Tolerance is set by SLSQP_FTOL=1e-10 on the OBJECTIVE, so the recovered
# weights are accurate to ~sqrt(ftol) and the residual to ~ftol -- not to
# machine epsilon.  Checking against machine epsilon would be checking the
# tolerance, not the solver.
chk("solver recovers exact convex combination",
    np.allclose(r.w, w_true, atol=1e-5) and r.sse < 1e-9,
    f"max|w-w*|={np.abs(r.w-w_true).max():.2e} sse={r.sse:.2e} status={r.status}")

# ---- 2. simplex feasibility + optimality vs random simplex points ---------
A2 = rng.normal(size=(60, 200))
x2 = rng.normal(size=200)
r2 = solve_simplex_qp(A2, x2)
f = lambda w: np.sum((x2 - A2.T @ w) ** 2)
rand = [f(d / d.sum()) for d in rng.dirichlet(np.ones(60), size=4000)]
# SIMPLEX_TOL (1e-6) is the documented feasibility bar.  SLSQP satisfies the
# equality constraint to ~1e-9 on this deliberately ill-posed random problem
# (60 donors, 200 dims, no signal); on the real panels the worst observed
# violation across every configuration was ~1e-15.
chk("simplex feasibility within SIMPLEX_TOL",
    abs(r2.w.sum() - 1) < 1e-6 and r2.w.min() >= -1e-12,
    f"sum-1={r2.w.sum()-1:.2e} min={r2.w.min():.2e}")
chk("solution beats 4000 random simplex points",
    r2.sse <= min(rand) + 1e-12, f"sse={r2.sse:.6f} best_random={min(rand):.6f}")

# ---- 3. warm start independence ------------------------------------------
starts = [np.full(60, 1 / 60), rng.dirichlet(np.ones(60)), rng.dirichlet(np.ones(60))]
sses = [solve_simplex_qp(A2, x2, w0=s).sse for s in starts]
chk("objective independent of warm start",
    max(sses) - min(sses) < 1e-8 * max(1.0, abs(sses[0])),
    f"spread={max(sses)-min(sses):.3e} over {len(starts)} starts")

# ---- 4. Panel agrees with the Phase 1 public API --------------------------
ss = load_states("medcpt")
p = Panel.build(ss, 1)
ids = p.ids
j = 137
chk("Panel ids == StateSet.eligible_subjects", ids == ss.eligible_subjects(1),
    f"n={len(ids)}")
chk("Panel.Y[j] == ss.state(id, T0+1)",
    np.allclose(p.Y[j], ss.state(ids[j], 2)), "")
chk("Panel.P[j] == ss.pre_period(id, T0).ravel()",
    np.allclose(p.P[j], ss.pre_period(ids[j], 1).reshape(-1)), "")
chk("Panel.target_last_state == ss.state(id, T0)",
    np.allclose(p.target_last_state(j), ss.state(ids[j], 1)), "")
Dp, Dn, did = ss.donor_matrix(1, exclude=ids[j])
rows = p.donor_index(j)
chk("Panel donor pool == StateSet.donor_matrix (ids and values)",
    [ids[i] for i in rows] == did and np.allclose(p.P[rows], Dp)
    and np.allclose(p.Y[rows], Dn), f"n_donors={len(did)}")
chk("target excluded from its own donor pool", ids[j] not in [ids[i] for i in rows])

# ---- 5. LVCF baseline is literally the target's own state ----------------
chk("lvcf(i) == ss.state(id, T0)",
    all(np.allclose(bl.lvcf(p, i), ss.state(ids[i], 1)) for i in range(0, p.n, 97)))

# ---- 6. n_jobs does not change results -----------------------------------
small = Panel(ids=p.ids[:60], P=p.P[:60], Y=p.Y[:60], T0=1, d=p.d,
              encoder="medcpt", scale="none")
a = fit_all(small, m=20, n_jobs=1, verbose=False)
b = fit_all(small, m=20, n_jobs=4, verbose=False)
chk("fit_all bit-identical at n_jobs=1 vs 4",
    all(np.array_equal(x.w, y.w) and np.array_equal(x.donor_rows, y.donor_rows)
        for x, y in zip(a, b)))
ar = fit_all(small, m=20, n_jobs=1, random_donors=True, seed=0, verbose=False)
br = fit_all(small, m=20, n_jobs=7, random_donors=True, seed=0, verbose=False)
chk("random-donor placebo bit-identical at n_jobs=1 vs 7",
    all(np.array_equal(x.donor_rows, y.donor_rows) for x, y in zip(ar, br)))
cr = fit_all(small, m=20, n_jobs=1, random_donors=True, seed=1, verbose=False)
chk("random-donor placebo DOES change with seed",
    any(not np.array_equal(x.donor_rows, y.donor_rows) for x, y in zip(ar, cr)))

# ---- 7. pre-filter is a superset relation --------------------------------
f25 = fit_target(p, j, m=25)
f50 = fit_target(p, j, m=50)
chk("top-25 donors are a subset of top-50",
    set(f25.donor_rows.tolist()) <= set(f50.donor_rows.tolist()))
chk("larger m cannot worsen the pre-period fit",
    f50.pre_rmse <= f25.pre_rmse + 1e-9,
    f"pre_rmse m=25 {f25.pre_rmse:.6f} -> m=50 {f50.pre_rmse:.6f}")

# ---- 8. paired bootstrap vs an unguarded reimplementation ----------------
# `paired_boot` now calls `stats.boot_continuous`, not `stats.boot`.  The
# earlier version fed `boot` an alternating 0/1 dummy purely to get past its
# binary-class guard and then audited how many resamples the guard had eaten;
# there is no guard and no discard count any more, so what is checked here is
# equality with a plain unguarded bootstrap plus the absence of the hack.
delta = rng.normal(-0.002, 0.01, 749)
res = ev.paired_boot(delta, B=2000, seed=0)


def unguarded(v, B=2000, seed=0):
    g = np.random.default_rng(seed)
    n = len(v)
    return np.percentile([v[g.integers(0, n, n)].mean() for _ in range(B)], [2.5, 97.5])


lo, hi = unguarded(delta)
chk("paired_boot == unguarded percentile bootstrap",
    abs(res["lo"] - lo) < 1e-12 and abs(res["hi"] - hi) < 1e-12,
    f"vendored [{res['lo']:.6f},{res['hi']:.6f}] vs plain [{lo:.6f},{hi:.6f}]")
chk("the dummy-label hack is gone; the continuous resampler is used",
    "n_discarded" not in res and res.get("resampler") == "boot_continuous"
    and not hasattr(ev, "_dummy_labels"),
    f"resampler={res.get('resampler')!r}")

# ---- 9. code-set decoding: LVCF decodes to exactly itself ----------------
sb = load_states("bag")
pb = Panel.build(sb, 1)
K = np.array([int((pb.target_last_state(i) > 0).sum()) for i in range(pb.n)])
Flv = np.stack([bl.lvcf(pb, i) for i in range(pb.n)])
sets = ev.topk_sets(Flv, K)
chk("top-K decoding of LVCF returns exactly its own code set",
    all(set(s.tolist()) == set(np.flatnonzero(Flv[i] > 0).tolist())
        for i, s in enumerate(sets)))
csm = ev.codeset_metrics(Flv, Flv, K)
chk("codeset F1 of a forecast against itself == 1.0",
    np.allclose(csm["f1"], 1.0), f"min={csm['f1'].min():.6f}")
chk("every method gets the identical cardinality budget",
    all(len(s) == K[i] for i, s in enumerate(sets)))

# ---- 10. row-L2 ablation really changes the neighbourhood ---------------
pr = pb.rownorm_l2()
chk("row-L2 rows are unit norm",
    np.allclose(np.linalg.norm(pr.Y, axis=1), 1.0, atol=1e-6))
ov = []
for i in range(0, pb.n, 37):
    d0 = pb.donor_index(i)
    a10 = set(d0[np.argsort(pb.distances(i, d0), kind="stable")[:10]].tolist())
    b10 = set(d0[np.argsort(pr.distances(i, d0), kind="stable")[:10]].tolist())
    ov.append(len(a10 & b10) / 10)
chk("raw vs rowL2 top-10 donor overlap is materially < 1 (the ablation bites)",
    np.mean(ov) < 0.8, f"mean overlap={np.mean(ov):.3f} over {len(ov)} targets")
card_raw, card_l2, pop = [], [], (pb.Y > 0).sum(axis=1)
for i in range(0, pb.n, 7):
    d0 = pb.donor_index(i)
    card_raw += list(pop[d0[np.argsort(pb.distances(i, d0), kind="stable")[:10]]])
    card_l2 += list(pop[d0[np.argsort(pr.distances(i, d0), kind="stable")[:10]]])
print(f"      cardinality of picked neighbours: raw median {np.median(card_raw):.0f}, "
      f"rowL2 median {np.median(card_l2):.0f}, population median {np.median(pop):.0f}")

print("\n%d/%d PASS" % (sum(ok.values()), len(ok)))
sys.exit(0 if all(ok.values()) else 1)
