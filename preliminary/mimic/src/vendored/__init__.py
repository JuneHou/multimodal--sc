"""Vendored, self-contained copies of code from an external repo.

Copied on 2026-08-01 from /data/wang/junh/githubs/moa-clinical-rag. Nothing in
this package imports from that repo at runtime -- it is under active edits and
several of its modules do not import cleanly. Each file's header records its
exact source path, line range, and every change made. See MODULE.md.

`llm` is intentionally NOT re-exported here: importing it pulls in vllm, which
is slow and GPU-adjacent. Import it explicitly when you need it.
"""

from . import embedding, features, stats  # noqa: F401

__all__ = ["embedding", "features", "stats"]
