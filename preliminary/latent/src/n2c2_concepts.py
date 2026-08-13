"""UMLS concept bags for the n2c2 corpus -- the H2 observable space.

The addendum to the pre-registration fixes this space: "UMLS concept bags
extracted per record with QuickUMLS over Jun's UTS Metathesaurus download,
filtered to disorder/drug/procedure/finding semantic types, min-df >= 5,
deterministic index.  Pre-registered caveat: no negation handling in v1 --
negated concepts enter the bag."

Nothing here does negation detection.  That is the caveat, not an oversight.

This module is the **medrag-side** half: semantic-type filtering tables from
`MRSTY.RRF`, preferred names from `MRCONSO.RRF` (for the eyeball report), the
binary per-record design matrix, and the leakage check.  The extraction itself
runs under a separate interpreter (`/data/wang/junh/envs/quickumls`), because
`quickumls` is not installed in medrag and must not be -- see
`scripts/_quickumls_extract.py`.

Assets (read-only):
    /data/wang/junh/datasets/UMLS_Metadata/{MRCONSO,MRSTY}.RRF   (2024AA)
    /data/wang/junh/datasets/quickUMLS/                          (pre-built)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

UMLS_DIR = Path("/data/wang/junh/datasets/UMLS_Metadata")
MRSTY = UMLS_DIR / "MRSTY.RRF"
MRCONSO = UMLS_DIR / "MRCONSO.RRF"
QUICKUMLS_FP = Path("/data/wang/junh/datasets/quickUMLS")

#: The accepted semantic types, exactly as specified for this run.  Grouped for
#: the report; the union is what filters a match.
TUI_GROUPS: Dict[str, Tuple[str, ...]] = {
    # disorders / findings
    "disorder": ("T047", "T048", "T049", "T050", "T191", "T046", "T184"),
    # procedures (incl. laboratory and diagnostic procedures)
    "procedure": ("T059", "T060", "T061"),
    # drugs / chemicals
    "drug": ("T121", "T195", "T200", "T116", "T123"),
}
ACCEPTED_TUIS: Set[str] = {t for g in TUI_GROUPS.values() for t in g}

#: Minimum document frequency for a concept to enter the state vocabulary.
MIN_DF = 5


# --------------------------------------------------------------------------
# UMLS tables
# --------------------------------------------------------------------------


def load_cui_semtypes(path: Path = MRSTY,
                      accepted: Iterable[str] = ACCEPTED_TUIS
                      ) -> Dict[str, Tuple[str, ...]]:
    """`CUI -> sorted accepted TUIs`, restricted to CUIs with >= 1 accepted TUI.

    MRSTY.RRF is `CUI|TUI|STN|STY|ATUI|CVF|`.  A CUI carries several TUIs; a
    CUI is accepted if ANY of them is in `accepted`, and only the accepted ones
    are kept in the value (so the report can attribute a concept to a group).
    """
    accepted = set(accepted)
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cui, tui = line.split("|", 2)[:2]
            if tui in accepted:
                out.setdefault(cui, set()).add(tui)
    return {c: tuple(sorted(v)) for c, v in out.items()}


def load_preferred_names(cuis: Iterable[str],
                         path: Path = MRCONSO) -> Dict[str, str]:
    """`CUI -> English preferred name`, for the eyeball-validation report only.

    MRCONSO field order: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|...|STR(14)|...
    Preference order, best first: (LAT=ENG, TS=P, STT=PF, ISPREF=Y) > (LAT=ENG,
    TS=P) > any ENG string.  Never used for matching -- names are cosmetic here.
    """
    want = set(cuis)
    best: Dict[str, Tuple[int, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.split("|")
            if f[1] != "ENG" or f[0] not in want:
                continue
            rank = 2
            if f[2] == "P" and f[4] == "PF" and f[6] == "Y":
                rank = 0
            elif f[2] == "P":
                rank = 1
            cur = best.get(f[0])
            if cur is None or rank < cur[0]:
                best[f[0]] = (rank, f[14])
    return {c: s for c, (_, s) in best.items()}


# --------------------------------------------------------------------------
# The bag state matrix
# --------------------------------------------------------------------------


def build_design_matrix(bags: Sequence[Sequence[str]],
                        vocab: Dict[str, int] | None = None,
                        min_df: int = MIN_DF
                        ) -> Tuple[np.ndarray, Dict[str, int]]:
    """Binary per-record design matrix over concepts with document frequency
    `>= min_df`.

    One row per record, **never cumulative**: a record's row is a function of
    that record's bag alone.  `vocab` may be supplied to reuse an existing
    column order (that is what the leakage check does); when supplied, `min_df`
    is ignored and out-of-vocabulary concepts are dropped.

    Column order is `sorted(vocab)` -- deterministic, as the addendum requires.
    """
    if vocab is None:
        df: Dict[str, int] = {}
        for bag in bags:
            for c in set(bag):
                df[c] = df.get(c, 0) + 1
        kept = sorted(c for c, k in df.items() if k >= int(min_df))
        vocab = {c: i for i, c in enumerate(kept)}
    X = np.zeros((len(bags), len(vocab)), dtype=np.float64)
    for r, bag in enumerate(bags):
        for c in set(bag):
            j = vocab.get(c)
            if j is not None:
                X[r, j] = 1.0
    return X, vocab


def validate_no_leakage(index: pd.DataFrame, bags: Sequence[Sequence[str]],
                        X: np.ndarray, vocab: Dict[str, int]) -> Dict[str, Any]:
    """Prove the bag states are per-record, not cumulative prefixes.

    Mirrors `mimic/src/datasets/state.py::validate_no_leakage`: rebuild every
    `t=0` row from a frame containing ONLY the `t=0` records -- no later record
    of that patient is visible to the rebuild -- against the same vocabulary,
    and require bit-identical equality with the corresponding rows of the full
    build.  Under a cumulative encoding, patients with later records would
    differ.  Also reports mean concepts-per-record by `t`, which grows
    monotonically under accumulation and is flat under a per-record encoding.
    """
    t = index["t"].to_numpy()
    rows0 = np.flatnonzero(t == 0)
    X0, _ = build_design_matrix([bags[r] for r in rows0], vocab=vocab)
    same = bool(np.array_equal(X0, X[rows0]))

    n_by_patient = index.groupby("patient_id")["t"].size()
    multi = set(n_by_patient[n_by_patient >= 3].index)
    prof = pd.DataFrame({"t": t, "support": X.sum(axis=1),
                         "patient_id": index["patient_id"].to_numpy()})
    by_t = prof.groupby("t")["support"].mean()
    by_t_cohort = (prof[prof["patient_id"].isin(multi)]
                   .groupby("t")["support"].mean())
    return {
        "t0_rebuilt_in_isolation_matches": same,
        "n_t0_rows_checked": int(len(rows0)),
        "n_t0_rows_from_multirecord_patients": int(
            sum(1 for r in rows0 if index["patient_id"].iloc[r] in multi)),
        "max_abs_diff": float(np.abs(X0 - X[rows0]).max()) if len(rows0) else 0.0,
        "mean_active_concepts_by_t_all_patients": {
            int(k): float(v) for k, v in by_t.head(6).items()},
        "mean_active_concepts_by_t_ge3_cohort": {
            int(k): float(v) for k, v in by_t_cohort.head(6).items()},
    }


def cohort_flat_blocks(X: np.ndarray, index: pd.DataFrame, T0: int,
                       ids: Sequence[int] | None = None
                       ) -> Tuple[List[int], np.ndarray]:
    """`(patient_idx, P)` for patients with `>= T0 + 2` records.

    `P` is `(n, (T0+1)*d)`: the pre-period records `t = 0..T0` flattened
    t-major, i.e. `Panel.build`'s convention, so a row of `P` and the same row
    of a latent-space block name the same patient-times.  Eligibility is
    `t_max >= T0+1`, which at `T0=1` is the `>= 3`-record cohort.  `ids` reuses
    a cohort computed elsewhere (that is how the latent blocks are guaranteed
    row-aligned to the bag block).
    """
    if ids is None:
        tmax = index.groupby("patient_idx")["t"].max()
        ids = sorted(int(p) for p, m in tmax.items() if m >= T0 + 1)
    ids = [int(p) for p in ids]
    rowmap = {(int(p), int(tt)): r for r, (p, tt) in
              enumerate(zip(index["patient_idx"], index["t"]))}
    P = np.empty((len(ids), (T0 + 1) * X.shape[1]), dtype=np.float64)
    for i, p in enumerate(ids):
        P[i] = np.concatenate([X[rowmap[(p, tt)]] for tt in range(T0 + 1)])
    return ids, P
