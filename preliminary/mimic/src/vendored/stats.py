"""
Bootstrap CIs, significance stars, Tjur's D, and tie-corrected Mann-Whitney AUROC.

VENDORED CODE -- copied, not imported.
  source repo:  /data/wang/junh/githubs/moa-clinical-rag
  source paths: src/metrics/tjur_readmission_full.py:29-63  (`tjur`, `boot`, `star`)
                src/metrics/threshold_sweep.py:104-120      (`auc_mw`)
  vendored on:  2026-08-01

Why two source files: `tjur_readmission_full.py` is a BROKEN module. Line 11
reads `RES = str(REPO_ROOT / 'results')` at module scope, and `REPO_ROOT` is
never defined or imported anywhere in that file -- importing it raises
NameError immediately. Only the pure functions were lifted; none of its module
scope (`RES`, `COND`, `load()`, `main()`, the `__main__` block) was copied.
Nothing here touches the filesystem.

`auc_mw` came from threshold_sweep.py rather than from tjur_readmission_full.py
because the source repo carries several near-duplicate hand-rolled AUROC
implementations (see "Discrepancies" below) and the threshold_sweep copy is the
cleanest: it is the only one that guards the degenerate single-class case AND
returns a plain float AND casts its inputs with np.asarray.

Changes from the source:
  1. `auc_mw`: verbatim apart from the docstring, which referenced
     `_analysis_common.auc` and "the published numbers" in the source repo --
     rewritten for this project. Logic is byte-for-byte the same algorithm.
  2. `tjur`, `star`: verbatim (type hints added).
  3. `boot`: two changes.
     a. Added a `RuntimeError` when every resample was rejected. The original
        called `np.percentile` on a possibly-empty `vals`, which yields a
        RuntimeWarning and `nan` CI bounds -- silently. This can only happen
        when y is single-class, which is exactly when you want to be told.
     b. Added the `alpha=0.05` argument. The original hardcoded
        `np.percentile(vals, [2.5, 97.5])`. Default reproduces the original.
     Resampling itself is unchanged: `np.random.default_rng(seed)`, B=2000
     draws of n indices with replacement, and each draw is DISCARDED unless it
     contains both classes.
  4. `auc` from tjur_readmission_full.py:29 was NOT copied -- it is a
     redundant, slightly messier version of `auc_mw` (it computes unused
     `pos`/`neg` locals and lacks the empty-class guard).
  5. `boot_continuous` is NEW, not vendored (added 2026-08-01, Phase 2). It is
     the same percentile bootstrap for a *continuous per-unit* statistic, with
     no binary-label argument and no single-class rejection rule. `boot` is
     unchanged and still the only thing called for label-based statistics
     (AUROC); nothing that used `boot` was migrated off it.


Discrepancies found while vendoring (recorded for a later reader): the task
brief described `auc_mw` as "the cleanest of four duplicate copies". There is
exactly ONE function literally named `auc_mw` in the source repo, but there are
four further hand-rolled tie-corrected AUROC implementations under other names:
  src/metrics/_analysis_common.py:42            def auc(L, S)
  src/metrics/tjur_readmission_full.py:29       def auc(y, p)
  src/metrics/verify_all_tables.py:98           def auc(y, s)
  src/metrics/discrimination/_build_mechanism_nb.py:47  def auc(L, S)
(plus a fifth pasted inside discrimination/analyze_mechanism.ipynb). So the
"four duplicates" characterisation is right in substance; only the naming
differs. threshold_sweep's remains the cleanest and is what is vendored here.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def auc_mw(y: Sequence, p: Sequence) -> float:
    """Tie-corrected Mann-Whitney AUROC.

    Vendored from threshold_sweep.py:104-120. Equivalent to
    sklearn.metrics.roc_auc_score for binary y, computed from average ranks so
    that tied scores contribute 0.5 rather than being ordered arbitrarily. Kept
    hand-rolled (rather than swapped for sklearn) so `boot` can call it 2000
    times without sklearn's per-call validation overhead.

    Returns nan if y is single-class.
    """
    y = np.asarray(y)
    s = np.asarray(p)
    o = np.argsort(s, kind="mergesort")
    sr = s[o]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and sr[j] == sr[i]:
            j += 1
        ranks[i:j] = (i + j - 1) / 2 + 1
        i = j
    r = np.empty(len(s))
    r[o] = ranks
    pos = y == 1
    npos = int(pos.sum())
    nneg = int((~pos).sum())
    if not npos or not nneg:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tjur(y: Sequence, p: Sequence) -> float:
    """Tjur's coefficient of discrimination.

    Vendored from tjur_readmission_full.py:44-45. Mean predicted probability on
    the positives minus mean predicted probability on the negatives. Unlike
    AUROC this is calibration-sensitive: it moves when predictions are
    rescaled, so a model that ranks perfectly but predicts 0.50/0.51 scores
    near zero. Range [-1, 1]; 0 is no discrimination.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    return float(p[y == 1].mean() - p[y == 0].mean())


def boot(
    y: Sequence,
    p: Sequence,
    fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for `fn(y, p)` over paired resamples of units.

    Vendored from tjur_readmission_full.py:48-57.

    Draws B resamples of n unit indices with replacement, keeping y and p
    aligned, and returns the empirical [alpha/2, 1-alpha/2] percentiles of
    `fn`. Resamples that end up single-class are DISCARDED, not counted -- so
    the effective B is slightly below the nominal B on heavily imbalanced data,
    and the interval is conditional on both classes being present.

    Note this is a percentile bootstrap, not BCa: it is not bias-corrected and
    will be somewhat anti-conservative for small n or a skewed statistic.

    Args:
        y:     binary labels.
        p:     predicted scores.
        fn:    statistic, called as fn(y_resampled, p_resampled).
        B:     bootstrap draws.
        seed:  RNG seed. Fixed default so CIs are reproducible.
        alpha: 1 - coverage. 0.05 gives a 95% interval.

    Returns:
        (lo, hi).

    Raises:
        RuntimeError: if every draw was discarded (y is single-class).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if (ys == 1).any() and (ys == 0).any():
            vals.append(fn(ys, ps))
    if not vals:
        raise RuntimeError(
            f"boot: all {B} resamples were single-class (n={n}, "
            f"n_pos={int((y == 1).sum())}); no CI can be formed."
        )
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def boot_continuous(
    v: Sequence[float],
    fn: Optional[Callable[[np.ndarray], float]] = None,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a CONTINUOUS per-unit statistic.

    NOT vendored -- written for this project (2026-08-01).  `boot` above is
    written for a *binary-label* statistic: it takes `(y, p)`, and it DISCARDS
    any resample that is single-class in `y`.  For a continuous per-unit
    quantity -- a per-patient RMSE, a per-patient code-set F1, a per-patient
    paired difference -- there is no `y`, the guard is meaningless, and it
    silently lowers the effective number of resamples below the nominal `B`.
    Phase 2 previously fabricated an alternating 0/1 dummy purely to get past
    that guard; this function replaces the hack.

    Differences from `boot`, stated explicitly because both now exist:

    * one data argument, not two;
    * `fn` is called as `fn(v_resampled)`, not `fn(y_resampled, p_resampled)`;
    * **no class guard and therefore no discarded resamples** -- exactly `B`
      values enter the percentile;
    * it cannot raise `RuntimeError`, because there is no rejection rule.

    The resampling itself is deliberately identical to `boot`'s so the two are
    comparable and so previously-published intervals reproduce: the same
    `np.random.default_rng(seed)`, the same `rng.integers(0, n, n)` draw per
    replicate, in the same order, and the same `np.percentile` at
    `[100*alpha/2, 100*(1-alpha/2)]`.  A call to `boot_continuous(v, seed=s)`
    returns bit-identical bounds to the old
    `boot(dummy_y, v, lambda _y, p: p.mean(), seed=s)` whenever that call
    happened to discard nothing.

    Like `boot` this is a plain percentile bootstrap, not BCa: no bias
    correction, so it is somewhat anti-conservative at small n or for a skewed
    statistic.

    Args:
        v:     (n,) per-unit values.  Units are the resampling unit, so each
               unit must appear exactly once (i.i.d.); for repeated measures on
               the same unit a cluster bootstrap is needed instead.
        fn:    statistic, called as `fn(v_resampled) -> float`.  Defaults to the
               mean.
        B:     bootstrap draws.  All of them are kept.
        seed:  RNG seed.  Fixed default so CIs are reproducible.
        alpha: 1 - coverage.  0.05 gives a 95% interval.

    Returns:
        (lo, hi).

    Raises:
        ValueError: if `v` is empty.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        raise ValueError("boot_continuous: empty input")
    if fn is None:
        def fn(a: np.ndarray) -> float:  # noqa: E306  (mean, the default)
            return float(a.mean())
    rng = np.random.default_rng(seed)
    vals = np.empty(B, dtype=float)
    for b in range(B):
        vals[b] = fn(v[rng.integers(0, n, n)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def star(lo: float, hi: float, null: float) -> str:
    """'*' if the CI [lo, hi] excludes `null`, else ' '.

    Vendored verbatim from tjur_readmission_full.py:60-61. Use null=0.5 for
    AUROC and null=0.0 for Tjur's D. This is an interval-exclusion marker, not
    a p-value, and it carries no multiple-comparison correction.
    """
    return "*" if (lo > null or hi < null) else " "
