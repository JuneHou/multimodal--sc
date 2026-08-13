"""
Phase 1: per-visit state representations for latent synthetic control.

Turns the Phase 0 timeline parquet (one row per patient-visit) into two numeric
state matrices, row-aligned to a `(subject_id, t) -> row` index, plus a donor-pool
interface that Phase 2's synthetic-control estimator consumes.

Two encoders, both produced:

  ``bag``     binary bag-of-codes over ``c:``/``p:``/``d:``-prefixed CCS/ATC names,
              built with the vendored ``features.design_matrix``.
  ``medcpt``  768-d ``ncbi/MedCPT-Article-Encoder`` CLS embedding of a short
              structured rendering of the visit, L2-normalised, built with the
              vendored ``embedding.embed``.

**Per-visit, never cumulative.**  Each row encodes that visit's own codes only.
The upstream repo this project inherited from stored nested prefixes (key
``{subject}_{k}`` carried visits ``0..k``), which makes any pre-period fit
trivially good.  The Phase 0 parquet is already per-visit; this module consumes
it row-wise and adds no accumulation.  `validate_no_leakage` proves it.

Nothing in `src/vendored/` or `rebuild_timeline.py` is modified.  Two
incompatibilities with the vendored code were found and are worked around here,
not patched upstream -- see MODULE.md "Known limitations".

Public API
----------
    render_visit_text(conditions, procedures, drugs)             -> str
    build_bag_states(df, min_df=5)                               -> (X, vocab, index)
    build_medcpt_states(df, ...)                                 -> (X, index, stats)
    save_states(...) / load_states(encoder, derived_dir, scale)  -> StateSet
    StateSet.pre_period(subject_id, T0)                          -> (T0+1, d)
    StateSet.donor_pool(T0, exclude=...)                         -> ((n, T0+2, d), ids)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Vendored, imported not copied (they live in this project tree).
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vendored.embedding import embed, sha  # noqa: E402
from src.vendored.features import _bag, design_matrix  # noqa: E402

# --------------------------------------------------------------------------
# Constants.  Every one of these is recorded in MODULE.md; none is a silent
# default.
# --------------------------------------------------------------------------

DEFAULT_DERIVED = _PROJECT_ROOT / "data" / "derived"
DEFAULT_TIMELINE = DEFAULT_DERIVED / "timeline_mimic3_mortality.parquet"

ENCODERS: Tuple[str, ...] = ("bag", "medcpt")

#: Minimum document frequency for the bag-of-codes vocabulary.  The vendored
#: default.  647 distinct codes exist; 585 survive min_df=5.
DEFAULT_MIN_DF = 5

MEDCPT_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_DIM = 768
#: BERT positional-embedding ceiling for this checkpoint. Hard, not a choice.
MEDCPT_MAXLEN = 512
MEDCPT_BATCH = 32

#: Section order and English headers used to render a visit as text.
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("conditions", "Conditions"),
    ("procedures", "Procedures"),
    ("drugs", "Medications"),
)

#: The ATC name table carries XML-ish numeric markup, e.g.
#: "vitamin b<n>12</n> and folic acid".  Stripped for the *embedding text only*
#: (it tokenises as junk); the bag-of-codes vocabulary keeps the raw string,
#: because there identity matters and readability does not.
_NTAG = re.compile(r"</?n>")

STATE_DTYPE = np.float32

SCALINGS: Tuple[str, ...] = ("none", "global", "zscore")
DEFAULT_SCALING = "none"


# --------------------------------------------------------------------------
# Record coercion.
# --------------------------------------------------------------------------


def _codes(value: Any, clean: bool = False) -> List[str]:
    """Coerce a parquet list-column cell to a plain ``list[str]``.

    pandas materialises the parquet's ``list<str>`` columns as
    ``numpy.ndarray`` of object (documented in Phase 0's MODULE.md).  The
    vendored ``features._flatten`` does ``for v in field or []``, and
    ``bool(ndarray)`` raises ``ValueError`` for length > 1 -- so the vendored
    bag builder cannot eat the Phase 0 parquet as-is.  Coercing here rather
    than patching the vendored file.
    """
    if value is None:
        return []
    out = [str(x) for x in list(value)]
    if clean:
        out = [_NTAG.sub("", x) for x in out]
    return out


def _record(row: Any) -> Dict[str, List[str]]:
    """Row -> the dict shape ``features._bag`` expects."""
    return {sec: _codes(row[sec]) for sec, _ in SECTIONS}


def _id_key(subject_id: str) -> Tuple[int, Any]:
    """Sort key giving numeric order for MIMIC's numeric subject ids.

    ``"1004" < "10004"`` numerically but not lexically, and every donor-pool
    array is ordered by this, so it is pinned rather than left to str sort.
    Non-numeric ids (none exist today) sort after numeric ones, lexically.
    """
    s = str(subject_id)
    return (0, int(s)) if s.isdigit() else (1, s)


def sorted_ids(ids: Iterable[str]) -> List[str]:
    """Canonical id ordering used by every ``(n_donors, ...)`` array."""
    return sorted({str(i) for i in ids}, key=_id_key)


# --------------------------------------------------------------------------
# Encoder (a): bag of codes.
# --------------------------------------------------------------------------


def build_bag_states(
    df: pd.DataFrame, min_df: int = DEFAULT_MIN_DF
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Binary per-visit bag-of-codes matrix.

    Uses the vendored ``features._bag`` (which prefixes ``c:``/``p:``/``d:`` by
    section, so a condition and a drug sharing a name stay distinct) and
    ``features.design_matrix`` (binary indicators, ``min_df`` document-frequency
    floor).

    The bags are handed to ``design_matrix`` as **sorted lists**, not sets.
    ``design_matrix`` builds its vocabulary from ``Counter.items()``, i.e. in
    first-seen order; iterating a ``set`` of strings makes that order depend on
    PYTHONHASHSEED, so the column order -- and therefore the saved matrix --
    would differ between runs.  A deduplicated sorted list gives ``Counter`` the
    identical document-frequency counts (each code once per bag) while making
    the order deterministic.  Behaviourally identical, reproducibly ordered.

    Returns:
        (X (n_visits, d) float32 0/1, vocab as an ordered list of code strings,
         index DataFrame with subject_id / t / row).
    """
    index = _build_index(df)
    bags = [sorted(_bag(_record(r))) for _, r in df.iterrows()]
    X, vocab_map = design_matrix(bags, min_df=min_df)
    vocab = [None] * len(vocab_map)
    for code, j in vocab_map.items():
        vocab[j] = code
    return X.astype(STATE_DTYPE, copy=False), list(vocab), index


# --------------------------------------------------------------------------
# Encoder (b): MedCPT dense.
# --------------------------------------------------------------------------


def render_visit_text(
    conditions: Sequence[str],
    procedures: Sequence[str],
    drugs: Sequence[str],
) -> str:
    """Render one visit as short structured English.

    ``"Conditions: a; b. Procedures: c. Medications: d; e."``  Empty sections
    are rendered as ``"none"`` rather than dropped, so the string shape is
    constant across visits (Phase 0's code filter means this should never fire;
    it is asserted, not assumed).
    """
    parts = []
    for codes, header in zip((conditions, procedures, drugs), [h for _, h in SECTIONS]):
        items = _codes(codes, clean=True)
        parts.append(f"{header}: {'; '.join(items) if items else 'none'}.")
    return " ".join(parts)


def _render_chunk(sections: Dict[str, List[str]]) -> str:
    """Render a partial visit (a chunk), omitting sections with no items."""
    parts = []
    for sec, header in SECTIONS:
        items = sections.get(sec) or []
        if items:
            parts.append(f"{header}: {'; '.join(items)}.")
    return " ".join(parts) if parts else "Conditions: none."


def chunk_visit_texts(
    conditions: Sequence[str],
    procedures: Sequence[str],
    drugs: Sequence[str],
    tok,
    maxlen: int = MEDCPT_MAXLEN,
    length_cache: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Split one visit into >=1 renderings that each fit under ``maxlen`` tokens.

    **This is not cosmetic.**  The brief assumed the rendered visits were short
    enough that the 512-token ceiling would not bind.  Measured, it binds:
    9.7% of the 9,582 visits exceed 512 tokens and the longest is 977, driven by
    a mean of 29.2 drugs per visit with names like "iv solutions used in
    parenteral administration of fluids, electrolytes and nutrients".  Plain
    truncation would silently drop the tail of the drug list -- always the drugs,
    since they render last -- so the state of a poly-pharmacy patient would be
    systematically missing exactly the codes that distinguish them.

    Instead codes are greedily packed, in section order, into chunks that each
    fit; the caller embeds every chunk and mean-pools.  A visit that fits in one
    chunk (90.3% of them) produces exactly the same vector as plain embedding,
    so this is a strict superset of the naive path.

    Args:
        tok:          a HF tokenizer (used for per-name token counts).
        length_cache: optional ``{name: n_tokens}`` memo shared across visits.

    Returns:
        List of rendered chunk texts, in order.
    """
    if length_cache is None:
        length_cache = {}

    def tlen(s: str) -> int:
        n = length_cache.get(s)
        if n is None:
            n = len(tok(s, add_special_tokens=False)["input_ids"])
            length_cache[s] = n
        return n

    # Reserve: [CLS] + [SEP], plus one header ("Conditions:" etc.) and a
    # trailing "." for each of the three sections that could appear in a chunk.
    overhead = 2 + sum(tlen(f"{h}:") + 1 for _, h in SECTIONS)
    budget = maxlen - overhead

    items: List[Tuple[str, str]] = []
    for codes, (sec, _) in zip((conditions, procedures, drugs), SECTIONS):
        items += [(sec, c) for c in _codes(codes, clean=True)]

    chunks: List[str] = []
    cur: Dict[str, List[str]] = {sec: [] for sec, _ in SECTIONS}
    used = 0
    for sec, name in items:
        cost = tlen(name) + 1  # +1 for the "; " separator
        if used + cost > budget and used > 0:
            chunks.append(_render_chunk(cur))
            cur = {s: [] for s, _ in SECTIONS}
            used = 0
        cur[sec].append(name)
        used += cost
    if any(cur[s] for s, _ in SECTIONS) or not chunks:
        chunks.append(_render_chunk(cur))
    return chunks


def build_medcpt_states(
    df: pd.DataFrame,
    cache_path: Path,
    model_name: str = MEDCPT_MODEL,
    maxlen: int = MEDCPT_MAXLEN,
    batch: int = MEDCPT_BATCH,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Dense per-visit states from MedCPT, chunk-safe against the 512 ceiling.

    Each visit is rendered (and chunked if needed), every chunk is embedded via
    the vendored ``embedding.embed`` (CLS pooling, L2-normalised, cached by sha1
    of the text), and a visit's chunk vectors are mean-pooled and re-normalised
    so every row is unit-norm.

    Returns:
        (X (n_visits, 768) float32, index DataFrame, stats dict).
    """
    from transformers import AutoTokenizer

    index = _build_index(df)
    tok = AutoTokenizer.from_pretrained(model_name)

    length_cache: Dict[str, int] = {}
    per_visit_chunks: List[List[str]] = []
    full_texts: List[str] = []
    empty_section_visits: List[str] = []
    for _, r in df.iterrows():
        c, p, d = (_codes(r[sec]) for sec, _ in SECTIONS)
        if not (c and p and d):
            # Phase 0 documents a `len(c)*len(p)*len(d) != 0` filter, so this
            # should be impossible.  Exactly one row violates it (see MODULE.md
            # "Known limitations").  Recorded, not crashed on, and not patched
            # upstream: `render_visit_text` emits "<Section>: none." so the
            # string shape stays constant.
            empty_section_visits.append(f"{r['subject_id']}/t={r['t']}")
        full_texts.append(render_visit_text(c, p, d))
        per_visit_chunks.append(
            chunk_visit_texts(c, p, d, tok, maxlen=maxlen, length_cache=length_cache)
        )

    # Measure, do not assume: report the observed token-length distribution of
    # the *unchunked* rendering, and confirm every chunk actually fits.
    full_len = np.array(
        [len(tok(t, add_special_tokens=True)["input_ids"]) for t in full_texts]
    )
    chunk_texts = [t for ch in per_visit_chunks for t in ch]
    chunk_len = np.array(
        [len(tok(t, add_special_tokens=True)["input_ids"]) for t in chunk_texts]
    )
    n_over = int((chunk_len > maxlen).sum())
    assert n_over == 0, f"{n_over} chunks still exceed maxlen={maxlen}"

    stats: Dict[str, Any] = {
        "n_visits": int(len(df)),
        "n_unique_full_texts": int(len(set(full_texts))),
        "full_text_tokens_mean": float(full_len.mean()),
        "full_text_tokens_p50": float(np.percentile(full_len, 50)),
        "full_text_tokens_p95": float(np.percentile(full_len, 95)),
        "full_text_tokens_max": int(full_len.max()),
        "full_text_frac_over_maxlen": float((full_len > maxlen).mean()),
        "n_visits_over_maxlen": int((full_len > maxlen).sum()),
        "n_chunks_total": int(len(chunk_texts)),
        "n_unique_chunks": int(len(set(chunk_texts))),
        "chunks_per_visit_max": int(max(len(c) for c in per_visit_chunks)),
        "chunk_tokens_max": int(chunk_len.max()),
        "maxlen": int(maxlen),
        "n_visits_with_empty_section": len(empty_section_visits),
        "empty_section_visits": empty_section_visits[:10],
    }
    if verbose:
        print(
            f"[medcpt] tokens: mean={stats['full_text_tokens_mean']:.1f} "
            f"max={stats['full_text_tokens_max']} "
            f"over-{maxlen}={stats['n_visits_over_maxlen']} "
            f"({stats['full_text_frac_over_maxlen']:.2%}); "
            f"{stats['n_chunks_total']} chunks "
            f"({stats['n_unique_chunks']} unique)",
            flush=True,
        )

    cache = embed(
        model_name,
        chunk_texts,
        cache_path=cache_path,
        batch=batch,
        maxlen=maxlen,
        device=device,
        verbose=verbose,
    )

    X = np.zeros((len(df), MEDCPT_DIM), dtype=STATE_DTYPE)
    for i, chunks in enumerate(per_visit_chunks):
        v = np.mean([cache[sha(t)] for t in chunks], axis=0)
        X[i] = v / (np.linalg.norm(v) + 1e-12)
    assert X.shape[1] == MEDCPT_DIM
    return X, index, stats


def build_medcpt_truncated(
    df: pd.DataFrame,
    cache_path: Path,
    model_name: str = MEDCPT_MODEL,
    maxlen: int = MEDCPT_MAXLEN,
    batch: int = MEDCPT_BATCH,
    device: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """The naive path: embed the full rendering and let the tokenizer truncate.

    Built only so the cost of truncation can be *measured* against the chunked
    states (see the report's ``medcpt_truncation``).  Not saved as a state
    matrix.
    """
    texts = [
        render_visit_text(*(_codes(r[sec]) for sec, _ in SECTIONS))
        for _, r in df.iterrows()
    ]
    cache = embed(
        model_name,
        texts,
        cache_path=cache_path,
        batch=batch,
        maxlen=maxlen,
        device=device,
        verbose=verbose,
    )
    return np.stack([cache[sha(t)] for t in texts]).astype(STATE_DTYPE)


# --------------------------------------------------------------------------
# Index + persistence.
# --------------------------------------------------------------------------


def _build_index(df: pd.DataFrame) -> pd.DataFrame:
    """``(subject_id, t) -> row``, row-aligned to the state matrix by position."""
    idx = pd.DataFrame(
        {
            "subject_id": df["subject_id"].astype(str).to_numpy(),
            "t": df["t"].astype("int32").to_numpy(),
            "row": np.arange(len(df), dtype="int32"),
        }
    )
    assert not idx.duplicated(["subject_id", "t"]).any(), "duplicate (subject_id, t)"
    return idx


def _scaling_params(X: np.ndarray) -> Dict[str, Any]:
    """Scaling constants computed once at build time, stored with the matrix.

    Pinned at build so a load-time rescale never depends on which subset of rows
    happens to be in memory.
    """
    norms = np.linalg.norm(X, axis=1)
    return {
        "row_norm_mean": float(norms.mean()),
        "row_norm_rms": float(np.sqrt((norms**2).mean())),
        "col_mean": X.mean(axis=0).astype(np.float64).tolist(),
        "col_std": X.std(axis=0).astype(np.float64).tolist(),
    }


def save_states(
    encoder: str,
    X: np.ndarray,
    index: pd.DataFrame,
    derived_dir: Path = DEFAULT_DERIVED,
    vocab: Optional[Sequence[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write ``states_<encoder>.npz`` + ``states_<encoder>_index.parquet``.

    The ``.npz`` is self-contained (matrix + index columns + vocab + metadata);
    the parquet is a convenience for pandas consumers.  Both are written; they
    carry the same index.
    """
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=STATE_DTYPE)
    assert len(index) == len(X)

    meta = dict(meta or {})
    meta.setdefault("encoder", encoder)
    meta["shape"] = list(X.shape)
    meta["dtype"] = str(X.dtype)
    meta["scaling_params"] = _scaling_params(X)

    payload: Dict[str, Any] = {
        "X": X,
        "subject_id": index["subject_id"].astype(str).to_numpy().astype("U"),
        "t": index["t"].to_numpy().astype("int32"),
        "row": index["row"].to_numpy().astype("int32"),
        "meta_json": np.array(json.dumps(meta), dtype=object),
    }
    if vocab is not None:
        payload["vocab"] = np.array([str(v) for v in vocab], dtype="U")

    npz_path = derived_dir / f"states_{encoder}.npz"
    idx_path = derived_dir / f"states_{encoder}_index.parquet"
    np.savez_compressed(npz_path, **payload)
    index.to_parquet(idx_path, index=False)
    return npz_path, idx_path


# --------------------------------------------------------------------------
# The consumer-facing object.
# --------------------------------------------------------------------------


@dataclass
class StateSet:
    """A state matrix, its index, and the donor-pool interface over them."""

    encoder: str
    X: np.ndarray
    index: pd.DataFrame
    vocab: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    scale: str = DEFAULT_SCALING

    _row_of: Dict[Tuple[str, int], int] = field(default_factory=dict, repr=False)
    _rows_by_subject: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._row_of:
            self._row_of = {
                (s, int(t)): int(r)
                for s, t, r in zip(
                    self.index["subject_id"], self.index["t"], self.index["row"]
                )
            }
        if not self._rows_by_subject:
            g = self.index.sort_values(["subject_id", "t"]).groupby("subject_id")
            self._rows_by_subject = {
                str(s): sub["row"].to_numpy(dtype=np.int64) for s, sub in g
            }

    # -- basic accessors ---------------------------------------------------

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def subjects(self) -> List[str]:
        """All subject ids present, in canonical id order."""
        return sorted_ids(self._rows_by_subject)

    def n_visits(self, subject_id: str) -> int:
        return int(len(self._rows_by_subject.get(str(subject_id), ())))

    def visit_counts(self) -> pd.Series:
        return pd.Series(
            {s: len(r) for s, r in self._rows_by_subject.items()}, dtype="int64"
        )

    def row(self, subject_id: str, t: int) -> int:
        """Matrix row for ``(subject_id, t)``.  KeyError if absent."""
        return self._row_of[(str(subject_id), int(t))]

    def state(self, subject_id: str, t: int) -> np.ndarray:
        return self.X[self.row(subject_id, t)]

    def block(self, subject_id: str, t_lo: int = 0, t_hi: Optional[int] = None):
        """``X_i[t_lo..t_hi]`` inclusive, shape ``(t_hi - t_lo + 1, d)``.

        ``t`` is contiguous from 0 per Phase 0, so this is a straight slice of
        the subject's t-ordered rows.
        """
        rows = self._rows_by_subject[str(subject_id)]
        hi = len(rows) - 1 if t_hi is None else int(t_hi)
        if hi >= len(rows) or t_lo < 0 or hi < t_lo:
            raise KeyError(
                f"subject {subject_id} has {len(rows)} visits; "
                f"t={t_lo}..{hi} out of range"
            )
        return self.X[rows[t_lo : hi + 1]]

    # -- the Phase 2 contract ---------------------------------------------

    def pre_period(self, subject_id: str, T0: int) -> np.ndarray:
        """The target's pre-period block ``X_i[0..T0]``, shape ``(T0+1, d)``."""
        return self.block(subject_id, 0, int(T0))

    def eligible_subjects(self, T0: int) -> List[str]:
        """Subjects with ``>= T0+2`` visits, in canonical id order.

        ``T0+2`` because a unit needs states ``t=0..T0`` to fit on and a state at
        ``t=T0+1`` -- either the donor's value to forecast *from*, or the
        target's held-out truth.  For the primary ``T0=1`` this is the
        ">= 3 visits" cohort.
        """
        need = int(T0) + 2
        return sorted_ids(s for s, r in self._rows_by_subject.items() if len(r) >= need)

    def donor_pool(
        self, T0: int, exclude: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Eligible donors as one dense id-ordered array.

        Args:
            T0:      last pre-period index.
            exclude: the target's ``subject_id``, dropped from the pool.  A
                     patient is never their own donor.

        Returns:
            ``(D, ids)`` where ``D`` has shape ``(n_donors, T0+2, d)`` --
            ``D[j, 0:T0+1]`` is donor ``j``'s pre-period block and
            ``D[j, T0+1]`` is the state to forecast from -- and ``ids[j]`` is
            donor ``j``'s ``subject_id``.  ``ids`` is in canonical id order and
            ``D`` is row-aligned to it, so the pool can be enumerated into a
            matrix, which a content-addressed cache alone cannot.
        """
        T0 = int(T0)
        ids = [s for s in self.eligible_subjects(T0) if s != str(exclude)]
        D = np.empty((len(ids), T0 + 2, self.d), dtype=STATE_DTYPE)
        for j, s in enumerate(ids):
            D[j] = self.X[self._rows_by_subject[s][: T0 + 2]]
        return D, ids

    def donor_matrix(
        self, T0: int, exclude: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """``donor_pool`` split into the pre-period and the forecast row.

        Returns ``(D_pre (n, (T0+1)*d), D_next (n, d), ids)``; ``D_pre`` is the
        pre-period block flattened in ``t``-major order, which is the design a
        least-squares / projected-gradient SC fit wants directly.
        """
        D, ids = self.donor_pool(T0, exclude=exclude)
        n, T, d = D.shape
        return D[:, : T - 1].reshape(n, (T - 1) * d).copy(), D[:, T - 1].copy(), ids


def _apply_scale(X: np.ndarray, scale: str, params: Dict[str, Any]) -> np.ndarray:
    """Apply one of ``SCALINGS`` using build-time constants."""
    if scale == "none":
        return X
    if scale == "global":
        # A single positive scalar.  Provably cannot change any distance
        # *ordering*; it only puts the two encoders' squared distances -- and
        # therefore any ridge penalty tuned against them -- on one scale.
        rms = float(params.get("row_norm_rms") or 1.0)
        return (X / (rms + 1e-12)).astype(STATE_DTYPE)
    if scale == "zscore":
        mu = np.asarray(params["col_mean"], dtype=np.float32)
        sd = np.asarray(params["col_std"], dtype=np.float32)
        return ((X - mu) / np.maximum(sd, 1e-6)).astype(STATE_DTYPE)
    raise ValueError(f"scale must be one of {SCALINGS}, got {scale!r}")


def load_states(
    encoder: str,
    derived_dir: Path = DEFAULT_DERIVED,
    scale: str = DEFAULT_SCALING,
) -> StateSet:
    """Load ``states_<encoder>.npz`` into a `StateSet`.

    Args:
        encoder: ``"bag"`` or ``"medcpt"``.
        scale:   ``"none"`` (default, as stored), ``"global"`` (divide by the
                 build-time RMS row norm; ordering-preserving), or ``"zscore"``
                 (per-column standardisation; **does** change ordering).
    """
    if scale not in SCALINGS:
        raise ValueError(f"scale must be one of {SCALINGS}, got {scale!r}")
    path = Path(derived_dir) / f"states_{encoder}.npz"
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"].item()))
        X = np.ascontiguousarray(z["X"], dtype=STATE_DTYPE)
        index = pd.DataFrame(
            {
                "subject_id": z["subject_id"].astype(str),
                "t": z["t"].astype("int32"),
                "row": z["row"].astype("int32"),
            }
        )
        vocab = [str(v) for v in z["vocab"]] if "vocab" in z.files else None
    X = _apply_scale(X, scale, meta.get("scaling_params", {}))
    return StateSet(
        encoder=encoder, X=X, index=index, vocab=vocab, meta=meta, scale=scale
    )


# --------------------------------------------------------------------------
# Validation helpers (called by the runner; kept here so they are importable).
# --------------------------------------------------------------------------


def validate_no_leakage(
    df: pd.DataFrame, X_bag: np.ndarray, vocab: Sequence[str], index: pd.DataFrame
) -> Dict[str, Any]:
    """Prove the states are per-visit, not cumulative prefixes.

    Rebuilds every ``t=0`` row from a dataframe containing *only* the ``t=0``
    rows -- so no later visit of that patient is even visible -- against the
    same vocabulary, and requires exact equality with the corresponding rows of
    the full build.  If the encoder accumulated over ``t``, patients with later
    visits would differ; patients with a single visit would not.

    Also reports mean active-code count by ``t``, which grows monotonically
    under a cumulative encoding and is flat under a per-visit one.
    """
    vocab_map = {c: i for i, c in enumerate(vocab)}
    df0 = df[df["t"] == 0].reset_index(drop=True)
    bags0 = [sorted(_bag(_record(r))) for _, r in df0.iterrows()]
    X0, _ = design_matrix(bags0, vocab=vocab_map)

    rowmap = {(s, int(t)): int(r) for s, t, r in index.itertuples(index=False)}
    rows = np.array(
        [rowmap[(str(s), 0)] for s in df0["subject_id"].astype(str)], dtype=np.int64
    )
    same = np.array_equal(X0.astype(STATE_DTYPE), X_bag[rows])

    n_by_subject = df.groupby("subject_id").size()
    counts = df["subject_id"].map(n_by_subject)
    t0_multi_rows = rows[(counts >= 3)[df["t"] == 0].to_numpy()]

    support = X_bag.sum(axis=1)
    prof = pd.DataFrame({"t": index["t"].to_numpy(), "support": support,
                         "subject_id": index["subject_id"].to_numpy()})
    by_t = prof.groupby("t")["support"].mean()
    # Restricted to the >=3-visit cohort, so the comparison across t is
    # within a fixed set of patients.  Pooled over all patients the mean drifts
    # up with t purely because only sicker, more-admitted patients reach high t.
    keep = set(n_by_subject[n_by_subject >= 3].index.astype(str))
    by_t_cohort = (
        prof[prof["subject_id"].isin(keep)].groupby("t")["support"].mean()
    )
    return {
        "t0_rebuilt_in_isolation_matches": bool(same),
        "n_t0_rows_checked": int(len(rows)),
        "n_t0_rows_from_multivisit_patients": int(len(t0_multi_rows)),
        "mean_active_codes_by_t_all_patients": {
            int(k): float(v) for k, v in by_t.head(6).items()
        },
        "mean_active_codes_by_t_ge3_cohort": {
            int(k): float(v) for k, v in by_t_cohort.head(6).items()
        },
    }


def sparsity(X: np.ndarray) -> Dict[str, float]:
    nz = np.count_nonzero(X)
    return {
        "n_rows": int(X.shape[0]),
        "d": int(X.shape[1]),
        "frac_nonzero": float(nz / X.size),
        "mean_nonzero_per_row": float(nz / X.shape[0]),
        "mean_row_l2": float(np.linalg.norm(X, axis=1).mean()),
    }


def nearest_donors(
    ss: StateSet, target: str, T0: int, k: int = 3
) -> List[Tuple[str, float]]:
    """Top-``k`` donors by Euclidean distance on the flattened pre-period block.

    The same squared-distance geometry the SC objective uses, with all weight on
    one donor -- i.e. the best possible 1-sparse synthetic control.  A smell
    test, not the estimator.
    """
    D_pre, _, ids = ss.donor_matrix(T0, exclude=target)
    q = ss.pre_period(target, T0).reshape(-1)
    dist = np.linalg.norm(D_pre - q, axis=1)
    order = np.argsort(dist)[:k]
    return [(ids[j], float(dist[j])) for j in order]


def distance_profile(
    ss: StateSet, T0: int, targets: Sequence[str], k: int = 10
) -> Dict[str, float]:
    """How well-separated are donors in this state space?

    Binary bag-of-codes gives integer-valued squared distances, so the donor
    distance vector is heavily tied and the identity of the "nearest" donor is
    not well determined.  Quantified here because it is the reason a global
    rescale appears to perturb neighbour sets when it provably cannot: the
    ordering is identical, only the arbitrary tie-break differs.
    """
    n_distinct, tie_at_k, spread = [], 0, []
    for s in targets:
        D_pre, _, _ = ss.donor_matrix(T0, exclude=s)
        q = ss.pre_period(s, T0).reshape(-1)
        d = np.linalg.norm(D_pre - q, axis=1)
        n_distinct.append(len(np.unique(np.round(d, 5))))
        o = np.sort(d)
        if k < len(o) and np.isclose(o[k - 1], o[k]):
            tie_at_k += 1
        spread.append(float((o[-1] - o[0]) / (o[0] + 1e-12)))
    return {
        "k": k,
        "n_donors": int(len(ss.eligible_subjects(T0)) - 1),
        "mean_distinct_distances": float(np.mean(n_distinct)),
        "frac_targets_tied_at_k": float(tie_at_k / max(len(targets), 1)),
        "mean_relative_spread": float(np.mean(spread)),
    }


def neighbour_overlap(
    ss_a: StateSet, ss_b: StateSet, T0: int, targets: Sequence[str], k: int = 10
) -> Dict[str, float]:
    """Mean top-``k`` set overlap between two state spaces, over ``targets``."""
    ov = []
    for s in targets:
        a = {i for i, _ in nearest_donors(ss_a, s, T0, k=k)}
        b = {i for i, _ in nearest_donors(ss_b, s, T0, k=k)}
        ov.append(len(a & b) / k)
    return {
        "k": k,
        "n_targets": len(targets),
        "mean_overlap": float(np.mean(ov)) if ov else float("nan"),
        "min_overlap": float(np.min(ov)) if ov else float("nan"),
    }
