"""QuickUMLS extraction worker -- runs under /data/wang/junh/envs/quickumls.

Deliberately isolated: `quickumls` is NOT installed in the medrag environment
and must not be, so this script talks to the rest of the pipeline through JSONL
on disk and imports nothing from `latent/src`.

    in   JSONL, one object per record: {"row": int, "text": str}
    out  JSONL, one object per record: {"row": int, "matches": [
             {"cui", "term", "ngram", "start", "end", "similarity",
              "semtypes": [...]}, ...]}

Every match QuickUMLS returns after its own semantic-type filter is written;
the min-df cut and the bag construction happen on the medrag side.

    python _quickumls_extract.py --in in.jsonl --out out.jsonl \
        --index /data/wang/junh/datasets/quickUMLS --threshold 0.9 --jobs 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

TUIS = ["T046", "T047", "T048", "T049", "T050", "T059", "T060", "T061",
        "T116", "T121", "T123", "T184", "T191", "T195", "T200"]

_M = {}


def _init(index_fp: str, threshold: float, similarity: str, window: int,
          overlapping: str, min_match_length: int) -> None:
    from quickumls import QuickUMLS

    _M["m"] = QuickUMLS(
        quickumls_fp=index_fp,
        overlapping_criteria=overlapping,
        threshold=threshold,
        similarity_name=similarity,
        window=window,
        min_match_length=min_match_length,
        accepted_semtypes=set(TUIS),
    )


def _one(payload):
    row, text = payload
    m = _M["m"]
    out = []
    for group in m.match(text, best_match=True, ignore_syntax=False):
        for c in group:
            out.append({
                "cui": c["cui"], "term": c["term"], "ngram": c["ngram"],
                "start": int(c["start"]), "end": int(c["end"]),
                "similarity": float(c["similarity"]),
                "semtypes": sorted(c["semtypes"]),
            })
    return row, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--index", default="/data/wang/junh/datasets/quickUMLS")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--similarity", default="jaccard")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--overlapping", default="score")
    ap.add_argument("--min-match-length", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--probe", default="",
                    help="match this string, print the result, and exit -- the "
                         "index-compatibility probe")
    a = ap.parse_args()

    init_args = (a.index, a.threshold, a.similarity, a.window, a.overlapping,
                 a.min_match_length)
    if a.probe:
        t0 = time.time()
        _init(*init_args)
        print(f"[probe] matcher built in {time.time()-t0:.1f}s", flush=True)
        _, res = _one((0, a.probe))
        print(json.dumps(res, indent=2))
        return 0

    with open(a.inp, encoding="utf-8") as fh:
        docs = [(d["row"], d["text"]) for d in map(json.loads, fh)]
    print(f"[extract] {len(docs)} records, jobs={a.jobs}, "
          f"threshold={a.threshold} {a.similarity}", flush=True)

    t0 = time.time()
    results = {}
    if a.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(a.jobs, initializer=_init, initargs=init_args) as pool:
            for k, (row, res) in enumerate(
                    pool.imap_unordered(_one, docs, chunksize=1), 1):
                results[row] = res
                if k % 100 == 0:
                    print(f"  {k}/{len(docs)} in {time.time()-t0:.0f}s",
                          flush=True)
    else:
        _init(*init_args)
        for k, d in enumerate(docs, 1):
            row, res = _one(d)
            results[row] = res
            if k % 50 == 0:
                print(f"  {k}/{len(docs)} in {time.time()-t0:.0f}s", flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        for row, _ in docs:
            fh.write(json.dumps({"row": row, "matches": results[row]}) + "\n")
    print(f"[extract] done in {time.time()-t0:.1f}s -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
