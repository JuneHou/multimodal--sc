"""
L1 -- embed the rendered visit corpus with the two pre-registered encoders.

Both arms go through **sentence-transformers directly**.  The vendored
`mimic/src/vendored/embedding.py::embed` is deliberately NOT used: it is
`transformers` code that takes `last_hidden_state[:, 0]` (CLS pooling), which is
correct for the MedCPT BERT it was written for and wrong for both encoders here
-- Qwen3-Embedding is **last-token** pooled and BioLORD-2023 is **mean** pooled.
Using it would silently produce a different representation than the model card
defines.

ARM 1 -- Qwen/Qwen3-Embedding-8B
    bf16 on one A40.  **No instruction prefix.**  The Qwen3-Embedding family
    prepends an instruction to *queries only* in asymmetric retrieval; every use
    downstream here is symmetric visit-to-visit similarity, where query and
    document are the same kind of object, so prefixing one side would break the
    symmetry the synthetic control assumes.  Native output is 4096-d and is
    stored; the primary representation is the **MRL truncation to the first 768
    dims, re-L2-normalised** -- Matryoshka models place the coarse structure in
    the leading coordinates, so truncate-then-renormalise is the documented
    reduction, and it puts the arm at exactly BioLORD's dimensionality so the
    two are compared at equal width.

ARM 2 -- FremyCompany/BioLORD-2023
    Mean-pooled 768-d.  `max_seq_length` is set to **512** after load: the repo
    ships `128`, which would truncate the longer visits in this corpus.  The
    pre-registration names this explicitly as a trap to avoid.

Both arms return float32 rows with unit L2 norm, row-aligned to the order of
`latent/data/rendered_visits.parquet`.

Public API
----------
    load_encoder(arm, device)             -> SentenceTransformer
    encode_texts(model, texts, ...)       -> (n, d) float32
    l2_normalise(X)                       -> (n, d) float32 unit rows
    mrl_truncate(X, dim)                  -> (n, dim) float32 unit rows
    save_states(path, X, patient_idx, t, meta)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

QWEN3_REPO = "Qwen/Qwen3-Embedding-8B"
BIOLORD_REPO = "FremyCompany/BioLORD-2023"

#: BioLORD ships `max_seq_length=128`; the corpus reaches 192 tokens.  Fixed to
#: 512 by the pre-registration.
BIOLORD_MAX_SEQ = 512

#: MRL truncation width for the primary Qwen3 representation.
MRL_DIM = 768

ARMS = ("qwen3", "biolord")


def load_encoder(arm: str, device: str = "cuda"):
    """One `SentenceTransformer`, configured as the arm's model card requires."""
    from sentence_transformers import SentenceTransformer

    if arm == "qwen3":
        import torch

        model = SentenceTransformer(
            QWEN3_REPO,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        )
    elif arm == "biolord":
        model = SentenceTransformer(BIOLORD_REPO, device=device)
        # Pre-registered: the shipped 128 would truncate this corpus.
        model.max_seq_length = BIOLORD_MAX_SEQ
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return model


def encode_texts(model, texts: Sequence[str], batch_size: int = 32,
                 show_progress: bool = True) -> np.ndarray:
    """Encode in the given order.  No instruction prefix, no query template.

    Normalisation is done here in float32 rather than via
    `normalize_embeddings=True`, so an arm running in bf16 does not normalise in
    bf16.
    """
    X = model.encode(list(texts), batch_size=batch_size,
                     show_progress_bar=show_progress,
                     convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(X, dtype=np.float32)


#: chunking budget for a 512-token encoder: 510 content tokens + [CLS] + [SEP].
CHUNK_MAX_TOKENS = 510


def chunk_by_tokens(text: str, tokenizer, max_tokens: int = CHUNK_MAX_TOKENS
                    ) -> Sequence[str]:
    """Split `text` into <= `max_tokens`-token chunks on sentence/whitespace.

    Added for the n2c2 corpus (records reach 2,985 tokens) so BioLORD's 512-token
    window does not silently truncate; the addendum pre-registers chunk+mean-pool
    for exactly this.  Rules:

    * a text that already fits is returned **unchanged, as a single chunk**, so
      the chunked path is byte-identical to direct encoding for short texts;
    * units are sentences (a word run ending in `.`/`!`/`?`, or a line), split
      further into words only if one unit alone exceeds the budget -- the
      concatenation of the units is exactly the input text;
    * packing is greedy on summed unit token counts (wordpiece never merges
      across whitespace, so the sum is an upper bound on the chunk's length),
      then verified by re-tokenising each chunk.
    """
    import re

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [text]

    parts = re.split(r"(\s+)", text)          # words and whitespace, alternating
    units: list = []
    buf = ""
    for i, p in enumerate(parts):
        buf += p
        is_ws = bool(i % 2)
        if is_ws and (("\n" in p) or buf.rstrip().endswith((".", "!", "?"))):
            units.append(buf)
            buf = ""
    if buf:
        units.append(buf)
    assert "".join(units) == text, "unit split lost characters"

    def toklen(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    # a single over-long unit -> break it at whitespace
    fixed: list = []
    for u in units:
        if toklen(u) <= max_tokens:
            fixed.append(u)
            continue
        words = re.split(r"(\s+)", u)
        cur = ""
        for j, w in enumerate(words):
            if cur and toklen(cur + w) > max_tokens and not (j % 2):
                fixed.append(cur)
                cur = ""
            cur += w
        if cur:
            fixed.append(cur)
    units = fixed
    assert "".join(units) == text, "word split lost characters"

    lens = [toklen(u) for u in units]
    chunks: list = []
    cur, cur_len = "", 0
    for u, ln in zip(units, lens):
        if cur and cur_len + ln > max_tokens:
            chunks.append(cur)
            cur, cur_len = "", 0
        cur += u
        cur_len += ln
    if cur:
        chunks.append(cur)
    for c in chunks:
        assert toklen(c) <= max_tokens, f"chunk of {toklen(c)} tokens > {max_tokens}"
    assert "".join(chunks) == text, "chunking lost characters"
    return chunks


def encode_texts_chunked(model, texts: Sequence[str],
                         max_tokens: int = CHUNK_MAX_TOKENS,
                         batch_size: int = 64, show_progress: bool = True):
    """Chunk + encode + **mean-pool over chunks** + re-L2-normalise.

    Returns `(X, chunk_counts)`.  Texts that fit in one chunk go through the
    identical code path as direct encoding, so their rows are unchanged by the
    chunking (asserted in the L1 runner on a sample).
    """
    tok = model.tokenizer
    per_text = [chunk_by_tokens(t, tok, max_tokens) for t in texts]
    flat: list = []
    owner: list = []
    for i, cs in enumerate(per_text):
        for c in cs:
            flat.append(c)
            owner.append(i)
    E = encode_texts(model, flat, batch_size=batch_size,
                     show_progress=show_progress)
    owner_a = np.asarray(owner)
    d = E.shape[1]
    X = np.zeros((len(texts), d), dtype=np.float32)
    np.add.at(X, owner_a, E)
    counts = np.bincount(owner_a, minlength=len(texts)).astype(np.int32)
    X = X / counts[:, None]                    # mean-pool over chunks
    return l2_normalise(X), counts


def l2_normalise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    assert float(n.min()) > 0, "zero-norm embedding row"
    return (X / n).astype(np.float32)


def mrl_truncate(X: np.ndarray, dim: int = MRL_DIM) -> np.ndarray:
    """First `dim` coordinates, re-L2-normalised.  The Matryoshka reduction."""
    if X.shape[1] < dim:
        raise ValueError(f"cannot truncate {X.shape[1]}-d to {dim}-d")
    return l2_normalise(X[:, :dim])


def save_states(path: Path, X: np.ndarray, patient_idx: np.ndarray,
                t: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Path:
    """`.npz` with the row-alignment keys the rest of L2-L4 joins on."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"rows not unit norm: [{norms.min()}, {norms.max()}]")
    np.savez(path, X=X,
             patient_idx=np.asarray(patient_idx, dtype=np.int32),
             t=np.asarray(t, dtype=np.int32),
             meta_json=json.dumps(meta or {}))
    return path


def load_states(path: Path) -> Dict[str, Any]:
    import json

    z = np.load(Path(path), allow_pickle=False)
    return {"X": z["X"], "patient_idx": z["patient_idx"], "t": z["t"],
            "meta": json.loads(str(z["meta_json"]))}
