"""
Bag-of-codes design matrix + cross-validated logistic-regression AUROC.

VENDORED CODE -- copied, not imported.
  source repo:  /data/wang/junh/githubs/moa-clinical-rag
  source path:  src/metrics/feature_baseline.py
  source lines: 30-89 (`_flatten`, `_bag`, `load_cell`, `design_matrix`, `cv_auroc`)
  vendored on:  2026-08-01

Why vendored: self-contained copy, see src/vendored/MODULE.md. The original file
resolves a `ROOT` at module scope by walking parent dirs for a `config/`
directory and raises StopIteration if it finds none -- not importable outside
that repo's layout.

Changes from the source:
  1. **`cv_auroc` signature changed** -- the original was
     `cv_auroc(task, ds, folds=5, seed=0)` and began by calling
     `load_cell(task, ds)`, which reads
     `<ROOT>/KARE/data/ehr_data/{ds}_{task}_samples_test.json`. That path and
     those files do not exist in this project. The vendored version is
     `cv_auroc(bags, y, folds=5, seed=0, min_df=5, C=1.0, max_iter=2000)` and
     takes the data directly. The modelling body is otherwise unchanged.
  2. `load_cell` was NOT copied -- it is the repo-specific JSON reader replaced
     by change #1. Building `bags`/`y` from MIMIC is agent B's job in
     src/datasets/.
  3. Dropped module scope: `ROOT`, `EHR`, `CELLS`, `main()`, `__main__` block.
  4. `min_df` is now threaded through `cv_auroc` into the per-fold
     `design_matrix` call. The original hardcoded `min_df=5` inside `cv_auroc`
     even though `design_matrix` took the argument, so the knob was unreachable.
     The default is still 5, so behaviour is identical unless you pass one.
  5. `C` and `max_iter` for LogisticRegression are likewise now arguments with
     the original's values (1.0, 2000) as defaults. `class_weight='balanced'`
     is kept hardcoded, as in the original.
  6. `cv_auroc` now returns a dict instead of a 6-tuple. The original returned
     `(mean_auc, std_auc, mean_ap, n_pos, n, n_feat)` where `n_feat` was the
     leaked last-fold loop variable `len(vocab)`; the dict makes that explicit
     as `n_feat_last_fold`.
  7. `_bag` copied verbatim (modulo type hints).
  8. **`_flatten` patched 2026-08-01 by agent D (Phase 2).** Was
     `for v in field or []`; `bool(ndarray)` raises ValueError for length > 1,
     and pandas materialises this project's parquet `list<str>` columns as
     object ndarrays, so `_bag` could not consume Phase 0's output at all
     (agent B worked around it in `state._codes` rather than patching). Now
     `if field is None: field = []`. The nested-element check was widened from
     `isinstance(v, list)` to `(list, tuple, np.ndarray)` at the same time, so a
     genuinely nested ndarray flattens instead of being `str()`-ed into garbage.
     Callers passing lists/strings/None are unaffected.
  9. **`design_matrix` patched 2026-08-01 by agent D (Phase 2).** The built
     vocabulary is now `sorted(...)` instead of `Counter.items()` first-seen
     order. Callers pass `set`-valued bags, and `set` iteration order for `str`
     depends on PYTHONHASHSEED, so the column order of `X` was not reproducible
     across processes. This is a column permutation only -- no cell value
     changes -- but it makes the matrix a pure function of its input. Note the
     permutation means a matrix built by the patched code is NOT column-aligned
     with one built by the unpatched code; see MODULE.md.

Preserved on purpose: the vocabulary is fit on the TRAIN fold only and applied
to the test fold, so there is no vocabulary leakage across the CV split. Codes
unseen in training are silently dropped from the test rows.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def _flatten(field: Any) -> List[str]:
    """list-of-lists (per visit) -> flat list of code strings.

    Vendored from feature_baseline.py:30-38. Tolerates a flat list, a nested
    list, a numpy array, and None.

    PATCHED (see change #8 in the header): the source wrote ``for v in field or []``.
    ``field or []`` evaluates ``bool(field)``, which raises
    ``ValueError: The truth value of an array with more than one element is
    ambiguous`` for a numpy array of length > 1 -- and pandas materialises
    parquet ``list<str>`` columns as object ndarrays, so the source could not
    consume this project's Phase 0 parquet at all.  The None case is now
    explicit.
    """
    if field is None:
        field = []
    out: List[str] = []
    for v in field:
        if isinstance(v, (list, tuple, np.ndarray)):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def _bag(rec: Dict[str, Any], prefix: bool = True) -> Set[str]:
    """Record -> set of code strings over conditions/procedures/drugs.

    Vendored verbatim from feature_baseline.py:41-46. With `prefix=True` each
    code is tagged with its section's first letter ('c:', 'p:', 'd:') so the
    same raw code appearing as both a condition and a drug stays distinct.
    Set-valued: a code repeated across visits counts once (binary presence).
    """
    codes: List[str] = []
    for sec in ("conditions", "procedures", "drugs"):
        tag = sec[0] + ":" if prefix else ""
        codes += [tag + c for c in _flatten(rec.get(sec))]
    return set(codes)


def design_matrix(
    bags: Sequence[Set[str]],
    vocab: Optional[Dict[str, int]] = None,
    min_df: int = 5,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Binary code-presence design matrix.

    Vendored from feature_baseline.py:56-68.

    PATCHED (see change #9 in the header): the built vocabulary is now sorted.
    The source took `Counter.items()` in first-seen order; because callers pass
    `set`-valued bags and `set` iteration order for strings depends on
    PYTHONHASHSEED, the column order -- and therefore `X` itself -- differed
    between interpreter processes.  `kept` is now `sorted(...)`, which makes the
    column order a pure function of the input codes.  This changes nothing about
    the matrix's *content* (it is a column permutation) but makes it
    reproducible.

    Args:
        bags:   sequence of code sets, one per unit.
        vocab:  {code: column index}. If None it is built from `bags`, keeping
                codes appearing in at least `min_df` bags, in sorted code order.
                Pass the train fold's vocab when transforming a test fold.
        min_df: minimum document frequency. Ignored when `vocab` is given.

    Returns:
        (X, vocab) with X float32, shape (len(bags), len(vocab)), entries 0/1.
        Codes absent from `vocab` are dropped.
    """
    if vocab is None:
        c: Counter = Counter()
        for b in bags:
            c.update(b)
        kept = sorted(code for code, n in c.items() if n >= min_df)
        vocab = {code: i for i, code in enumerate(kept)}
    X = np.zeros((len(bags), len(vocab)), dtype=np.float32)
    for i, b in enumerate(bags):
        for code in b:
            j = vocab.get(code)
            if j is not None:
                X[i, j] = 1.0
    return X, vocab


def cv_auroc(
    bags: Sequence[Set[str]],
    y: np.ndarray,
    folds: int = 5,
    seed: int = 0,
    min_df: int = 5,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict[str, Any]:
    """Stratified k-fold CV AUROC/AUPRC for L2 logistic regression on code bags.

    Adapted from feature_baseline.py:71-89 (see change #1 in the header: takes
    data directly instead of loading it from the source repo's JSONs).

    Args:
        bags:     sequence of code sets, one per unit.
        y:        binary labels, shape (len(bags),).
        folds:    StratifiedKFold splits.
        seed:     shuffle seed for the fold assignment.
        min_df:   min document frequency, applied per train fold.
        C:        inverse L2 regularisation strength.
        max_iter: lbfgs iteration cap.

    Returns:
        dict with auroc_mean, auroc_std, auprc_mean, auprc_std, n_pos, n,
        n_feat_last_fold, folds, seed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs: List[float] = []
    aps: List[float] = []
    vocab: Dict[str, int] = {}
    for tr, te in skf.split(np.zeros(len(y)), y):
        Xtr, vocab = design_matrix([bags[i] for i in tr], min_df=min_df)
        Xte, _ = design_matrix([bags[i] for i in te], vocab=vocab)
        clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        s = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], s))
        aps.append(average_precision_score(y[te], s))
    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(aps)),
        "auprc_std": float(np.std(aps)),
        "n_pos": int(y.sum()),
        "n": int(len(y)),
        "n_feat_last_fold": len(vocab),
        "folds": folds,
        "seed": seed,
    }
