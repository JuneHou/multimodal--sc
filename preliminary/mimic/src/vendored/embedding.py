"""
Content-addressed HuggingFace text embedding with an idempotent on-disk cache.

VENDORED CODE -- copied, not imported.
  source repo:  /data/wang/junh/githubs/moa-clinical-rag
  source path:  src/metrics/build_embeddings.py
  source lines: 62-85 (`embed`), 32 (`sha` helper)
  vendored on:  2026-08-01

Why vendored: the source repo is under active edits and several of its modules
are broken at import time (e.g. src/metrics/tjur_readmission_full.py references
an undefined REPO_ROOT at module scope). We keep a self-contained copy so this
project has no dependency on that repo's state.

Changes from the source:
  1. Dropped the module-scope script context entirely -- the original file had
     `HERE`, `sys.path.insert`, `import _analysis_common as A`, a hardcoded
     `EMB = HERE/'.emb_cache'`, a `CELLS` list, a `collect()` function that
     walks that repo's result manifests, and an `if __name__ == '__main__'`
     block. None of it applies here. Only `sha()` and `embed()` were lifted.
  2. `DEV` was a module-level global in the original. It is now a per-call
     argument `device=None` (auto-detects cuda, same rule as the original) so
     the module has no import-time CUDA dependency and is testable on CPU.
  3. `cache_path` is now coerced with `Path(...)`, so a plain str works. The
     original assumed a Path (it called `cache_path.exists()`).
  4. Added `mkdir(parents=True, exist_ok=True)` on the cache's parent dir. The
     original relied on a module-level `EMB.mkdir(exist_ok=True)` side effect
     that no longer exists.
  5. `texts` is now deduplicated in a deterministic (first-seen) order before
     embedding. The original was called with a `set`, so its `todo` order --
     and therefore its checkpoint boundaries -- were nondeterministic across
     runs. Results were unaffected (the cache is keyed by content hash), but
     determinism is worth having.
  6. Added the `verbose` flag to silence progress prints under test.
  7. Type hints and docstrings added. The embedding math is UNCHANGED:
     CLS pooling at position 0, float32, L2-normalise with a 1e-12 epsilon,
     fp16 model weights on GPU / fp32 on CPU, batch=32, max_length=512,
     checkpoint every 20 batches.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def sha(s: Optional[str]) -> str:
    """Content address for a text: sha1 of its utf-8 bytes.

    Vendored verbatim from build_embeddings.py:32. `None` and `''` both hash to
    the empty-string digest, which is intentional -- the caller is expected to
    have discarded empties before calling `embed`.
    """
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()


def embed(
    model_name: str,
    texts: Iterable[str],
    cache_path,
    batch: int = 32,
    maxlen: int = 512,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Embed `texts` with a HF encoder, caching by sha1 of the text.

    Idempotent: texts whose sha1 is already in the cache file are skipped, and
    the model is not even loaded if nothing is left to do. The cache is
    checkpointed to disk every 20 batches so a killed run loses at most that
    much work.

    Args:
        model_name: HF model id or local path, e.g. 'ncbi/MedCPT-Article-Encoder'.
        texts:      iterable of strings. Deduplicated, order-preserving.
        cache_path: pickle file holding {sha1: np.float32 vector}.
        batch:      texts per forward pass.
        maxlen:     tokenizer truncation length.
        device:     'cuda' | 'cpu'. None auto-detects (cuda if available).
        verbose:    print progress.

    Returns:
        The full cache dict {sha1: vector}, including pre-existing entries.
        Look a text up with `cache[sha(text)]`.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, np.ndarray] = (
        pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
    )

    # Deterministic dedup (see change #5 in the header).
    seen, uniq = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    todo = [t for t in uniq if sha(t) not in cache]
    if verbose:
        print(
            f"[{model_name}] cached={len(cache)} todo={len(todo)} "
            f"(of {len(uniq)} unique)",
            flush=True,
        )
    if not todo:
        return cache

    # Imported lazily so that a fully-cached call, and `import embedding`
    # itself, cost nothing.
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    if device == "cuda":
        model = AutoModel.from_pretrained(model_name).to(device).eval().half()
    else:
        model = AutoModel.from_pretrained(model_name).eval()

    with torch.no_grad():
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            enc = tok(
                chunk,
                truncation=True,
                max_length=maxlen,
                padding=True,
                return_tensors="pt",
            ).to(device)
            # CLS pooling: token position 0 of the last hidden state.
            cls = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            # L2-normalise so dot product == cosine similarity.
            cls /= np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12
            for tx, v in zip(chunk, cls):
                cache[sha(tx)] = v.astype(np.float32)
            if (i // batch) % 20 == 0:
                if verbose:
                    print(
                        f"  {model_name.split('/')[-1]}: {i + len(chunk)}/{len(todo)}",
                        flush=True,
                    )
                pickle.dump(cache, open(cache_path, "wb"))  # checkpoint

    pickle.dump(cache, open(cache_path, "wb"))
    if verbose:
        print(f"[{model_name}] wrote {len(cache)} -> {cache_path}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return cache
