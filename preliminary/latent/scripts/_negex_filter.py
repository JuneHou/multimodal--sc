"""NegEx negation detection over QuickUMLS spans -- the v2 filter.

Runs under /data/wang/junh/envs/quickumls (needs spacy + negspacy), same
isolation rule as `_quickumls_extract.py`: JSONL in, JSONL out, no imports from
`latent/src`.

The addendum's permitted follow-up after H2 fails in v1 is "a NegEx-filtered
v2".  This does NOT re-extract: it reads the v1 matches and flags each one, so
the concept inventory is identical and the only difference between v1 and v2 is
which matches survive.

    in   records.jsonl {"row", "text"}, matches.jsonl {"row", "matches": [...]}
    out  JSONL {"row", "negated": [bool per match, aligned to matches]}

Method: negspacy's `Negex` (NegEx of Chapman et al.), `en_clinical` termset,
applied to the QuickUMLS character spans rather than to spaCy's own NER --
`doc.ents` is overwritten with our spans and the component is called directly.
`doc.ents` cannot hold overlapping spans and QuickUMLS returns plenty ("type 2
diabetes mellitus" contains "diabetes"), so the unique spans are partitioned
into non-overlapping LAYERS by a greedy sweep and the component is run once per
layer; every span is therefore scored, none is dropped for overlapping another.
Sentence boundaries (which bound NegEx's search window) come from
en_core_web_sm's parser and do not depend on the layering.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple

_S: Dict = {}


def _init() -> None:
    import spacy
    import negspacy.negation  # noqa: F401  -- registers the "negex" factory

    # The NER is not used -- entities are supplied by QuickUMLS -- but the
    # parser is, because NegEx's window is the sentence.
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    negex = nlp.add_pipe("negex", config={"ent_types": [], "chunk_prefix": []})
    _S.update(nlp=nlp, negex=negex)


def _layers(spans: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Greedy partition of char spans into non-overlapping layers."""
    out: List[List[Tuple[int, int]]] = []
    ends: List[int] = []
    for s in sorted(spans):
        for k, e in enumerate(ends):
            if s[0] >= e:
                out[k].append(s)
                ends[k] = s[1]
                break
        else:
            out.append([s])
            ends.append(s[1])
    return out


def _one(payload):
    row, text, spans = payload
    nlp, negex = _S["nlp"], _S["negex"]
    doc = nlp(text)
    flags: Dict[Tuple[int, int], bool] = {s: False for s in spans}
    for layer in _layers(list(spans)):
        ents = []
        for a, b in layer:
            sp = doc.char_span(a, b, label="CUI", alignment_mode="expand")
            if sp is not None:
                ents.append((sp, (a, b)))
        try:
            doc.ents = [sp for sp, _ in ents]
        except ValueError:
            # a char_span expanded into another one's tokens; fall back to
            # one span at a time for this layer rather than skipping it
            for sp, key in ents:
                doc.ents = [sp]
                negex(doc)
                flags[key] = bool(doc.ents[0]._.negex)
            continue
        negex(doc)
        for sp, key in zip(doc.ents, [k for _, k in ents]):
            flags[key] = bool(sp._.negex)
    return row, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--probe", default="")
    a = ap.parse_args()

    if a.probe:
        _init()
        doc = _S["nlp"](a.probe)
        print([(t.idx, t.text) for t in doc][:40])
        return 0

    texts = {}
    with open(a.records, encoding="utf-8") as fh:
        for d in map(json.loads, fh):
            texts[d["row"]] = d["text"]
    order: List[int] = []
    spans_by_row: Dict[int, List[Tuple[int, int]]] = {}
    match_spans: Dict[int, List[Tuple[int, int]]] = {}
    with open(a.matches, encoding="utf-8") as fh:
        for d in map(json.loads, fh):
            r = d["row"]
            order.append(r)
            ms = [(int(m["start"]), int(m["end"])) for m in d["matches"]]
            match_spans[r] = ms
            spans_by_row[r] = sorted(set(ms))
    print(f"[negex] {len(order)} records, "
          f"{sum(len(v) for v in match_spans.values())} matches, "
          f"{sum(len(v) for v in spans_by_row.values())} unique spans",
          flush=True)

    payloads = [(r, texts[r], spans_by_row[r]) for r in order]
    t0 = time.time()
    res: Dict[int, Dict[Tuple[int, int], bool]] = {}
    if a.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(a.jobs, initializer=_init) as pool:
            for k, (row, flags) in enumerate(
                    pool.imap_unordered(_one, payloads, chunksize=1), 1):
                res[row] = flags
                if k % 200 == 0:
                    print(f"  {k}/{len(payloads)} in {time.time()-t0:.0f}s",
                          flush=True)
    else:
        _init()
        for k, p in enumerate(payloads, 1):
            row, flags = _one(p)
            res[row] = flags
            if k % 50 == 0:
                print(f"  {k}/{len(payloads)} in {time.time()-t0:.0f}s",
                      flush=True)

    n_neg = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in order:
            flags = res[r]
            neg = [bool(flags[s]) for s in match_spans[r]]
            n_neg += sum(neg)
            fh.write(json.dumps({"row": r, "negated": neg}) + "\n")
    print(f"[negex] {n_neg} negated matches in {time.time()-t0:.1f}s -> {a.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
