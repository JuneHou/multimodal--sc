"""
Phase 2 evaluation.  One scorer for every method, no exceptions.

**PRIMARY METRIC: code-set F1 at matched cardinality**, per target, aggregated
with a PAIRED bootstrap over patients.  Each method decodes its forecast to the
top-`K_i` codes, where `K_i` is *the target's own code count at `T0`* -- a
budget available at forecast time and identical for every method (see
`CARDINALITY RULE` in `codeset_metrics`).  Paired matters: every method
forecasts the same patients, so the quantity with a meaningful confidence
interval is the per-patient *difference* against LVCF, not the difference of two
independently bootstrapped means.  The fraction of patients on which the method
beats LVCF is reported alongside, because a mean difference can be carried by a
handful of targets.

**Why F1 and not RMSE, which this module used to lead with.**  The targets are
binary code indicators.  On a binary target, squared error is a *proper scoring
rule*: it is minimised by the true conditional probability, so a well-calibrated
probabilistic forecast is supposed to beat a hard 0/1 forecast, and it will do
so even when it is no better at saying which codes the patient will actually
have.  SC emits a convex combination of binary vectors -- a probability-like
object.  LVCF emits a hard 0/1 vector.  Comparing them on RMSE is therefore
apples to oranges: the winner is decided partly by the *form* of the prediction,
not its accuracy, and a shrinkage-only "win" is exactly what that produces (the
cohort mean and the random-donor placebo both beat LVCF on RMSE).  Decoding
every method to the same cardinality removes that asymmetry: both sides emit a
set of `K_i` codes and are scored on set overlap.

**Secondary -- the probabilistic view, kept and clearly labelled as such:**

* **Brier / MSE and post-period RMSE** at `T0+1`.  These are the proper-scoring
  numbers.  They are informative about calibration and are still reported in
  full with paired CIs; they are simply not the metric a method is ranked by,
  because of the form asymmetry above.  `brier` is a mean squared error per
  patient; it is the Brier score exactly when the truth matrix is 0/1 (the
  `bag`/`raw` arm) and a plain MSE on the normalised scale otherwise.
* **Cosine similarity** -- scale-free, so it separates "points somewhere better"
  from "predicts a smaller vector".
* **Outcome AUROC** -- does the forecast state predict the mortality label at
  `T0+1`?  Uses the vendored `features.cv_auroc` on the bag code sets, plus a
  pooled out-of-fold variant of the identical protocol so dense forecasts are
  comparable and so a bootstrap CI can be formed.
* **Weight sparsity / effective donor count**, from `fit.TargetFit`.

Code-set F1 needs an interpretable vocabulary, so it exists only for the bag
encoder.  For a configuration where `Ybin`/`K` are not supplied there is no
primary metric and the module falls back to ranking by RMSE, saying so in the
table header rather than silently.

Diagnostic, deliberately kept apart from the above: **pre-period RMSE**.  With
`T0=1` the pre-period is two timesteps against up to 748 donors, so the fit can
be near-perfect while the forecast is worthless.  `pre_post_correlation` measures
whether pre-period fit predicts post-period accuracy at all; if it does not,
that is a finding about the method, not a bug in the code.

Public API
----------
    rmse_rows(F, Y)                       -> (n,) per-target RMSE
    boot_mean(v, ...)                     -> {mean, lo, hi, ...}
    paired_boot(delta, ...)               -> {mean_diff, lo, hi, win_frac, ...}
    codeset_metrics(F, Ybin, K)           -> {"f1": (n,), "jaccard": (n,), ...}
    cv_auroc_oof(X, y, ...)               -> {auroc, lo, hi, ...}
    pre_post_correlation(pre, post)       -> {pearson, spearman, ...}
    evaluate_config(...)                  -> one fully-scored configuration
    format_table(rows, ...)               -> str
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vendored.features import cv_auroc  # noqa: E402
from src.vendored.stats import auc_mw, boot, boot_continuous, star  # noqa: E402

#: Bootstrap draws.  The vendored default.
BOOT_B = 2000
#: Bootstrap seed.  The vendored default; fixed so every CI in the report is
#: reproducible.
BOOT_SEED = 0
BOOT_ALPHA = 0.05

#: Folds / seed / solver settings for the outcome AUROC.  Identical to the
#: vendored `cv_auroc` defaults, so the dense variant is the same protocol.
AUROC_FOLDS = 5
AUROC_SEED = 0
AUROC_C = 1.0
AUROC_MAX_ITER = 2000


# --------------------------------------------------------------------------
# Primary metric.
# --------------------------------------------------------------------------


def rmse_rows(F: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-target RMSE between forecast `F` and truth `Y`, both (n, d).

    Root-*mean*-square, i.e. divided by `d`, so a MedCPT (768-d) and a bag
    (585-d) number are on the same per-coordinate scale.  They are still not
    directly comparable across encoders because the coordinates mean different
    things and have different variance -- which is exactly why every method is
    re-run under every encoder rather than compared across them.
    """
    return np.sqrt(((F - Y) ** 2).mean(axis=1))


def cosine_rows(F: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-target cosine similarity between forecast and truth.

    **Why this is here and not decoration.**  RMSE is not scale-free, and under
    squared error the minimiser of expected loss is a *shrunk* prediction: any
    average of donors sits closer to the cohort centroid than a real visit does,
    so it wins on RMSE partly by predicting a smaller vector, not a better
    direction.  The first run showed exactly this signature -- the cohort mean
    and the random-donor placebo both beat LVCF on RMSE, and neither uses the
    target's identity in any way that could help.

    Cosine ignores magnitude entirely.  If a method beats LVCF on RMSE but not
    on cosine, its RMSE win is shrinkage.  If it wins on both, it is genuinely
    pointing somewhere better.  Zero-norm rows score 0.
    """
    fn = np.linalg.norm(F, axis=1)
    yn = np.linalg.norm(Y, axis=1)
    denom = fn * yn
    return np.divide((F * Y).sum(axis=1), denom, out=np.zeros(len(F)),
                     where=denom > 0)


def boot_mean(
    v: Sequence[float], B: int = BOOT_B, seed: int = BOOT_SEED,
    alpha: float = BOOT_ALPHA,
) -> Dict[str, Any]:
    """Percentile-bootstrap CI for the mean of `v`, resampling patients.

    Uses `stats.boot_continuous`, NOT `stats.boot`.  `boot` is written for a
    binary-label statistic and discards every resample that is single-class in
    `y`; a per-patient RMSE / F1 / paired difference has no `y` at all.  An
    earlier version of this module fabricated an alternating 0/1 dummy purely to
    get past that guard, which was meaningless for the statistic and silently
    made the effective number of resamples a function of `n`.  `boot_continuous`
    draws the resamples identically and keeps all `B` of them.
    """
    v = np.asarray(v, dtype=float)
    lo, hi = boot_continuous(v, B=B, seed=seed, alpha=alpha)
    return {
        "mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
        "median": float(np.median(v)), "n": int(len(v)),
        "B": B, "seed": seed, "resampler": "boot_continuous",
    }


def paired_boot(
    delta: Sequence[float], B: int = BOOT_B, seed: int = BOOT_SEED,
    alpha: float = BOOT_ALPHA, lower_is_better: bool = True,
) -> Dict[str, Any]:
    """Paired bootstrap over patients of a per-patient difference.

    Args:
        delta: `metric_method[i] - metric_reference[i]`, one entry per patient.
            With `lower_is_better` (RMSE), a NEGATIVE mean is the method winning.

    Patients, not observations, are the resampling unit -- each patient appears
    exactly once in `delta`, so `stats.boot`'s i.i.d. resampling is the right
    thing here (unlike the multi-visit case its docstring warns about).

    Returns the mean difference with its CI, the win fraction, and `wins` --
    the CI excluding zero in the winning direction.
    """
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    lo, hi = boot_continuous(delta, B=B, seed=seed, alpha=alpha)
    better = delta < 0 if lower_is_better else delta > 0
    md = float(delta.mean())
    return {
        "mean_diff": md, "lo": float(lo), "hi": float(hi),
        "median_diff": float(np.median(delta)),
        "win_frac": float(better.mean()),
        "n_wins": int(better.sum()), "n": int(n),
        "sig": star(float(lo), float(hi), 0.0),
        "beats_ref": bool((hi < 0.0) if lower_is_better else (lo > 0.0)),
        "B": B, "seed": seed, "resampler": "boot_continuous",
    }


# --------------------------------------------------------------------------
# Secondary: code-set F1 / Jaccard.
# --------------------------------------------------------------------------


def topk_sets(F: np.ndarray, K: np.ndarray) -> List[np.ndarray]:
    """Per-row top-`K[i]` column indices of `F`, ties broken by column index.

    `np.argsort(-f, kind="stable")` so a row of a *binary* forecast (LVCF) with
    exactly `K` ones returns those ones and nothing arbitrary, and so any
    remaining ties are broken deterministically rather than by BLAS ordering.
    """
    out = []
    for i in range(F.shape[0]):
        k = int(K[i])
        out.append(np.argsort(-F[i], kind="stable")[:k] if k > 0
                   else np.empty(0, dtype=int))
    return out


def codeset_metrics(
    F: np.ndarray, Ybin: np.ndarray, K: np.ndarray
) -> Dict[str, np.ndarray]:
    """Set-overlap between the decoded forecast code set and the true one.

    CARDINALITY RULE
    ----------------
    A forecast is a convex combination of binary vectors and so is continuous;
    it has to be thresholded into a set before F1 means anything.  Thresholding
    at a fixed value would be a disaster for a fair comparison: LVCF is exactly
    binary and would emit ~44 codes, while an SC forecast spread over 15 donors
    has almost no coordinate near 1 and would emit nearly none.  The measured
    difference would then be *cardinality*, not accuracy.

    So every method is given the **same cardinality budget**:
    `K_i = |{codes active in the target's own visit at T0}|`, i.e. how many
    codes that patient had last time.  Each method emits its own top-`K_i`
    coordinates.

    Why this K and not another:

    * It is **available at forecast time**.  Using `|codes at T0+1|` would leak
      the answer's cardinality into every method equally but would still be
      information no forecaster has.
    * It is **patient-specific**, so a poly-pharmacy patient is not scored
      against the same budget as a 12-code patient.
    * It makes **LVCF exactly itself**: top-`K_i` of a binary vector with `K_i`
      ones is that vector's own code set, so LVCF is not distorted by the
      decoding rule it is being compared under.
    * A fixed global K (say 40) was rejected for the second reason; "predict a
      code if its weight exceeds 0.5" was rejected for the first.

    Args:
        F:    (n, d) continuous forecasts.
        Ybin: (n, d) the TRUE binary state at T0+1.  Always the raw 0/1 matrix,
              even when `F` came from the row-L2 ablation, so both ablation arms
              are scored against identical ground truth.
        K:    (n,) per-target budget.
    """
    n = F.shape[0]
    preds = topk_sets(F, K)
    f1 = np.zeros(n)
    jac = np.zeros(n)
    prec = np.zeros(n)
    rec = np.zeros(n)
    for i in range(n):
        true = set(np.flatnonzero(Ybin[i] > 0).tolist())
        pred = set(preds[i].tolist())
        inter = len(true & pred)
        union = len(true | pred)
        f1[i] = 2 * inter / (len(true) + len(pred)) if (true or pred) else 1.0
        jac[i] = inter / union if union else 1.0
        prec[i] = inter / len(pred) if pred else 0.0
        rec[i] = inter / len(true) if true else 0.0
    return {"f1": f1, "jaccard": jac, "precision": prec, "recall": rec,
            "pred_sizes": np.array([len(p) for p in preds])}


# --------------------------------------------------------------------------
# Secondary: outcome AUROC.
# --------------------------------------------------------------------------


def cv_auroc_oof(
    X: np.ndarray, y: np.ndarray, folds: int = AUROC_FOLDS,
    seed: int = AUROC_SEED, C: float = AUROC_C, max_iter: int = AUROC_MAX_ITER,
    boot_seed: int = BOOT_SEED, B: int = BOOT_B,
) -> Dict[str, Any]:
    """Pooled out-of-fold AUROC under the vendored `cv_auroc`'s exact protocol.

    Same `StratifiedKFold(folds, shuffle=True, random_state=seed)` and same
    `LogisticRegression(C=C, max_iter=max_iter, class_weight='balanced')`.  The
    only difference is that predictions are pooled across folds before scoring
    instead of averaged fold-wise, which (a) lets a bootstrap CI be formed over
    patients and (b) works on **dense** features, so the MedCPT forecasts can be
    scored at all -- the vendored `cv_auroc` only accepts code bags.

    For the bag encoder the vendored function is called as well and the two
    numbers cross-checked; see `evaluate_config`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_pos": int(y.sum()), "n": int(len(y)),
                "note": "single-class outcome; AUROC undefined"}
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    a = auc_mw(y, oof)
    lo, hi = boot(y, oof, auc_mw, B=B, seed=boot_seed)
    return {
        "auroc": float(a), "lo": float(lo), "hi": float(hi),
        "sig_vs_chance": star(lo, hi, 0.5),
        "n_pos": int(y.sum()), "n": int(len(y)), "folds": folds, "seed": seed,
    }


# --------------------------------------------------------------------------
# The pre-period overfitting check.
# --------------------------------------------------------------------------


def pre_post_correlation(pre: np.ndarray, post: np.ndarray) -> Dict[str, Any]:
    """Does a good pre-period fit predict a good forecast?

    The classical synthetic-control worry, sharpened by this design's very short
    pre-period: with `T0=1` there are two pre-period timesteps and up to 748
    free (if simplex-constrained) weights, so the pre-period residual can be
    driven near zero by donors that have nothing to do with the target.  If the
    correlation reported here is ~0, pre-period fit is not evidence of anything
    and any selection or tuning done on it is selection on noise.
    """
    from scipy.stats import pearsonr, spearmanr

    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    ok = np.isfinite(pre) & np.isfinite(post)
    if ok.sum() < 3 or np.std(pre[ok]) == 0:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"),
                "n": int(ok.sum())}
    pr, pp = pearsonr(pre[ok], post[ok])
    sr, sp = spearmanr(pre[ok], post[ok])
    return {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "n": int(ok.sum()),
        "pre_rmse_mean": float(pre[ok].mean()),
        "pre_rmse_median": float(np.median(pre[ok])),
        "post_rmse_mean": float(post[ok].mean()),
    }


# --------------------------------------------------------------------------
# Whole-configuration scoring.
# --------------------------------------------------------------------------


#: Every method is compared, pairwise and per patient, against ALL of these --
#: not just LVCF.  The first smoke run made the reason obvious: the cohort mean
#: and even the RANDOM-donor placebo both beat LVCF comfortably, because under
#: squared error *any* averaging shrinks variance and a single observed visit is
#: a noisy target.  A table that reports only "SC beats LVCF" would therefore
#: claim a win for donor matching that donor matching did not earn.  The extra
#: references are what separate "averaging helps" from "matching helps".
#: `ridge` joins the list because it is the only reference that actually LEARNS
#: the pre-to-post map; "SC beats copying and averaging" is a much weaker claim
#: than "SC beats supervised regression", and only the latter is interesting.
DEFAULT_REFERENCES: Sequence[str] = ("lvcf", "cohort_mean", "sc_random",
                                     "topk_mean", "ridge")


def evaluate_config(
    forecasts: Dict[str, np.ndarray],
    Y: np.ndarray,
    *,
    reference: str = "lvcf",
    extra_references: Sequence[str] = DEFAULT_REFERENCES,
    Ybin: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    vocab: Optional[Sequence[str]] = None,
    fits: Optional[Dict[str, Any]] = None,
    do_auroc: bool = True,
    boot_B: int = BOOT_B,
    boot_seed: int = BOOT_SEED,
) -> Dict[str, Any]:
    """Score every method in `forecasts` against truth `Y`, on one panel.

    Args:
        forecasts: {method name: (n, d) forecast}.
        Y:         (n, d) truth at `T0+1`, in the SAME representation as the
                   forecasts (raw or row-L2), since RMSE is measured there.
        reference: the method every paired comparison is against.  `lvcf`.
        Ybin, K:   raw binary truth and per-target cardinality budget; supply
                   only for the bag encoder, else code-set metrics are skipped.
        labels:    (n,) mortality label at `T0+1` for the outcome AUROC.
        vocab:     bag column names, so the vendored `cv_auroc` can be called
                   on real code strings and cross-checked against `cv_auroc_oof`.
        fits:      {method name: List[TargetFit]} for sparsity reporting.
    """
    rmse = {nm: rmse_rows(F, Y) for nm, F in forecasts.items()}
    cos = {nm: cosine_rows(F, Y) for nm, F in forecasts.items()}
    ynorm = float(np.linalg.norm(Y, axis=1).mean())
    ref = rmse[reference]

    # ---- PRIMARY: code-set F1 at matched cardinality --------------------
    # Computed once per method, up front, because it is the metric everything
    # else is subordinate to.  Absent for encoders with no interpretable
    # vocabulary (no `Ybin`/`K`), in which case `has_primary` is False and the
    # table says so rather than pretending.
    has_primary = Ybin is not None and K is not None
    cs = ({nm: codeset_metrics(F, Ybin, K) for nm, F in forecasts.items()}
          if has_primary else {})

    rows: List[Dict[str, Any]] = []
    for nm, F in forecasts.items():
        r: Dict[str, Any] = {"method": nm}

        # -- primary ------------------------------------------------------
        if has_primary:
            c = cs[nm]
            r["f1"] = boot_mean(c["f1"], B=boot_B, seed=boot_seed)
            r["jaccard"] = boot_mean(c["jaccard"], B=boot_B, seed=boot_seed)
            r["precision_mean"] = float(c["precision"].mean())
            r["recall_mean"] = float(c["recall"].mean())
            r["pred_size_mean"] = float(c["pred_sizes"].mean())
            r["_f1_rows"] = c["f1"]
            # F1 is higher-is-better, so the paired difference flips sign.
            r["f1_vs_ref"] = (
                paired_boot(c["f1"] - cs[reference]["f1"], B=boot_B,
                            seed=boot_seed, lower_is_better=False)
                if nm != reference else None
            )
            # ... and against every other reference, exactly as for RMSE, so
            # "did matching earn anything?" can be asked on the primary metric
            # and not only on the probabilistic one.
            r["f1_vs"] = {
                other: paired_boot(c["f1"] - cs[other]["f1"], B=boot_B,
                                   seed=boot_seed, lower_is_better=False)
                for other in extra_references
                if other in cs and other != nm
            }

        # -- secondary: the probabilistic view ----------------------------
        # MSE/Brier is a PROPER scoring rule on binary targets, so it rewards a
        # calibrated soft forecast over a hard one independently of which codes
        # are right.  Reported in full, ranked by nothing.
        r["brier"] = boot_mean(rmse[nm] ** 2, B=boot_B, seed=boot_seed)
        r["rmse"] = boot_mean(rmse[nm], B=boot_B, seed=boot_seed)
        # Shrinkage diagnostics: scale-free accuracy, and how much smaller the
        # forecast is than a real visit.
        r["cosine"] = boot_mean(cos[nm], B=boot_B, seed=boot_seed)
        r["cosine_vs_ref"] = (
            paired_boot(cos[nm] - cos[reference], B=boot_B, seed=boot_seed,
                        lower_is_better=False)
            if nm != reference else None
        )
        fn = float(np.linalg.norm(F, axis=1).mean())
        r["forecast_norm_mean"] = fn
        r["truth_norm_mean"] = ynorm
        r["norm_ratio"] = fn / ynorm if ynorm else float("nan")
        r["vs_ref"] = (
            paired_boot(rmse[nm] - ref, B=boot_B, seed=boot_seed)
            if nm != reference else None
        )
        # Same paired bootstrap against every other reference in the table.
        r["vs"] = {
            other: paired_boot(rmse[nm] - rmse[other], B=boot_B, seed=boot_seed)
            for other in extra_references
            if other in rmse and other != nm
        }

        if do_auroc and labels is not None:
            r["auroc_dense"] = cv_auroc_oof(
                np.ascontiguousarray(F, dtype=np.float64), labels,
                boot_seed=boot_seed, B=boot_B,
            )
            if Ybin is not None and K is not None and vocab is not None:
                bags = [{str(vocab[j]) for j in s}
                        for s in topk_sets(F, K)]
                try:
                    r["auroc_codeset_vendored"] = cv_auroc(
                        bags, np.asarray(labels).astype(int),
                        folds=AUROC_FOLDS, seed=AUROC_SEED,
                    )
                except ValueError as e:  # e.g. degenerate fold
                    r["auroc_codeset_vendored"] = {"error": str(e)}

        if fits and nm in fits:
            ff = fits[nm]
            r["sparsity"] = {
                "eff_donors_ipr_mean": float(np.mean([f.eff_donors_ipr for f in ff])),
                "eff_donors_ipr_median": float(np.median([f.eff_donors_ipr for f in ff])),
                "eff_donors_count_mean": float(np.mean([f.eff_donors_count for f in ff])),
                "eff_donors_count_median": float(np.median([f.eff_donors_count for f in ff])),
                "max_weight_mean": float(np.mean([f.max_weight for f in ff])),
            }
            pre = np.array([f.pre_rmse for f in ff])
            r["pre_rmse_diagnostic"] = {
                "mean": float(pre.mean()), "median": float(np.median(pre)),
                "min": float(pre.min()), "max": float(pre.max()),
            }
            r["pre_vs_post"] = pre_post_correlation(pre, rmse[nm])
        rows.append(r)

    return {
        "reference": reference, "n_targets": int(len(Y)), "rows": rows,
        "boot_B": int(boot_B), "boot_seed": int(boot_seed),
        # Which metric this configuration is ranked and reported by.  Explicit
        # in the artifact so a reader of the JSON never has to infer it.
        "primary_metric": "code_f1" if has_primary else "post_rmse",
        "primary_metric_note": (
            "code-set F1 at matched cardinality K_i = |codes in the target's "
            "own visit at T0|; higher is better"
            if has_primary else
            "no interpretable vocabulary for this encoder, so no code-set F1 "
            "is available and ranking falls back to post-period RMSE (a proper "
            "scoring rule, and therefore NOT comparable across hard vs soft "
            "forecasts -- read it as calibration, not accuracy)"
        ),
        "secondary_metrics": ["brier", "post_rmse", "cosine", "auroc"],
        "_rmse_rows": rmse,
    }


def _f1_of(F: np.ndarray, Ybin: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-patient code-set F1.  Kept for callers that want the rows alone."""
    return codeset_metrics(F, Ybin, K)["f1"]


# --------------------------------------------------------------------------
# Presentation.
# --------------------------------------------------------------------------


def _fmt_ci(d: Optional[Dict[str, Any]], key: str = "mean", prec: int = 4) -> str:
    if d is None:
        return "—"
    v = d.get(key)
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{prec}f} [{d['lo']:.{prec}f}, {d['hi']:.{prec}f}]"


def format_table(
    result: Dict[str, Any], title: str, labels: Optional[Dict[str, str]] = None,
    show_codeset: bool = True, show_auroc: bool = True,
) -> str:
    """Human-readable methods x metrics table for one configuration.

    Ordered and ranked by the PRIMARY metric, code-set F1 at matched
    cardinality (higher is better).  Everything to the right of the `||`
    separator is the secondary, probabilistic view -- Brier/MSE, RMSE and
    cosine -- which is reported but is not what ranks the table; see this
    module's header for why a proper scoring rule cannot fairly rank a hard 0/1
    forecast against a soft one.

    When the configuration has no interpretable vocabulary (no code-set F1),
    the table falls back to ranking by RMSE and says so in its own header.
    """
    from .baselines import LABELS

    labels = labels or LABELS
    rows = result["rows"]
    ref = result["reference"]
    has_cs = show_codeset and any("f1" in r for r in rows)
    has_au = show_auroc and any("auroc_dense" in r for r in rows)

    hdr = [("method", 34)]
    if has_cs:
        hdr += [("code F1 [95% CI] (PRIMARY)", 28),
                ("Δ F1 vs LVCF [95% CI]", 30), ("win%", 6)]
    hdr += [("|| Brier/MSE [95% CI]", 30), ("post RMSE [95% CI]", 28),
            ("Δ RMSE vs LVCF [95% CI]", 30), ("cosine", 8), ("|f|/|y|", 8)]
    if has_au:
        hdr += [("AUROC [95% CI]", 24)]
    hdr += [("eff donors", 11)]

    banner = ("PRIMARY = code-set F1 at matched cardinality K_i (higher better)."
              "  Columns after '||' are the SECONDARY probabilistic view."
              if has_cs else
              "NO code-set F1 for this encoder (no interpretable vocabulary); "
              "ranked by post-period RMSE, which is a proper scoring rule and "
              "therefore NOT an apples-to-apples hard-vs-soft comparison.")

    out = [f"\n{title}", "=" * (sum(w for _, w in hdr) + 2 * len(hdr))]
    out.append(f"  [{banner}]")
    out.append("  ".join(h.ljust(w) for h, w in hdr))
    out.append("  ".join("-" * w for _, w in hdr))

    if has_cs:
        def sort_key(r):
            return -r["f1"]["mean"]
        best = max(rows, key=lambda r: r["f1"]["mean"])["method"]
    else:
        def sort_key(r):
            return r["rmse"]["mean"]
        best = min(rows, key=lambda r: r["rmse"]["mean"])["method"]

    for r in sorted(rows, key=sort_key):
        nm = r["method"]
        mark = " *" if nm == best else ""
        cells = [(labels.get(nm, nm) + mark)[:34].ljust(34)]
        if has_cs:
            cells.append(_fmt_ci(r.get("f1"), prec=4).ljust(28))
            fv = r.get("f1_vs_ref")
            if nm == ref or fv is None:
                cells += ["(reference)".ljust(30) if nm == ref else "—".ljust(30),
                          "—".ljust(6)]
            else:
                cells += [
                    f"{fv['mean_diff']:+.4f} [{fv['lo']:+.4f}, {fv['hi']:+.4f}]{fv['sig']}".ljust(30),
                    f"{100*fv['win_frac']:.1f}".ljust(6),
                ]
        cells.append(("|| " + _fmt_ci(r.get("brier"), prec=4)).ljust(30))
        cells.append(_fmt_ci(r["rmse"]).ljust(28))
        if nm == ref:
            cells.append("(reference)".ljust(30))
        else:
            v = r["vs_ref"]
            cells.append(
                f"{v['mean_diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{v['sig']}".ljust(30))
        cells.append(f"{r['cosine']['mean']:.4f}".ljust(8))
        cells.append(f"{r['norm_ratio']:.3f}".ljust(8))
        if has_au:
            a = r.get("auroc_dense")
            cells.append(
                (f"{a['auroc']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]{a.get('sig_vs_chance','')}"
                 if a and np.isfinite(a.get("auroc", np.nan)) else "n/a").ljust(24)
            )
        sp = r.get("sparsity")
        cells.append((f"{sp['eff_donors_ipr_mean']:.1f}" if sp else "—").ljust(11))
        out.append("  ".join(cells))
    out.append(
        f"  n = {result['n_targets']} patients · every Δ is a per-patient paired "
        f"bootstrap (B={result.get('boot_B', BOOT_B)}, "
        f"seed={result.get('boot_seed', BOOT_SEED)}, continuous resampler, no "
        f"resamples discarded); '*' = 95% CI excludes 0"
    )
    out.append("  Δ F1 : POSITIVE = beats LVCF.   Δ RMSE : NEGATIVE = beats LVCF. "
               " (F1 is higher-better, RMSE lower-better.)")
    out.append(f"  '*' after a method name = best {'code F1' if has_cs else 'post-period RMSE'} "
               "in this block")
    out.append("  SECONDARY (right of '||'): Brier/MSE and RMSE are PROPER scoring rules on a "
               "binary target, so a")
    out.append("  calibrated soft forecast beats a hard 0/1 one by construction — read them as "
               "calibration, not accuracy.")
    out.append("  cosine = scale-free accuracy (higher better);  |f|/|y| = mean forecast norm "
               "/ mean truth norm.")
    out.append("  A method with better RMSE but worse cosine than LVCF is winning by SHRINKAGE, "
               "not by pointing better.")

    # The comparison that decides whether MATCHING earned anything, as opposed
    # to averaging.  Printed under every table because the LVCF column alone is
    # genuinely misleading here.  Shown on the PRIMARY metric first.
    sc = next((r for r in rows if r["method"] == "sc"), None)
    if sc and sc.get("f1_vs"):
        out.append("  SC vs the other references on the PRIMARY metric, code F1 "
                   "(paired, same patients; positive = SC better):")
        for other in ("cohort_mean", "sc_random", "topk_mean", "ridge"):
            v = sc["f1_vs"].get(other)
            if v is None:
                continue
            verdict = ("SC better" if v["beats_ref"] else
                       "SC worse" if v["hi"] < 0 else "indistinguishable")
            out.append(
                f"    vs {labels.get(other, other):<34} "
                f"{v['mean_diff']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]{v['sig']} "
                f"win {100*v['win_frac']:.1f}%   -> {verdict}"
            )
    if sc and sc.get("vs"):
        out.append("  SC vs the other references on SECONDARY post-period RMSE "
                   "(negative = SC better):")
        for other in ("cohort_mean", "sc_random", "topk_mean", "ridge"):
            v = sc["vs"].get(other)
            if v is None:
                continue
            verdict = ("SC better" if v["beats_ref"] else
                       "SC worse" if v["lo"] > 0 else "indistinguishable")
            out.append(
                f"    vs {labels.get(other, other):<34} "
                f"{v['mean_diff']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]{v['sig']} "
                f"win {100*v['win_frac']:.1f}%   -> {verdict}"
            )
    return "\n".join(out)
