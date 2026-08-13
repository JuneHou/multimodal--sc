"""H2 step 1 -- per-record UMLS concept bags for the n2c2 corpus.

The addendum fixes this as H2's observable space: QuickUMLS over the 2024AA
Metathesaurus, disorder/drug/procedure/finding semantic types, min-df >= 5,
deterministic index, **no negation handling in v1** (pre-registered caveat).

`quickumls` is not installed in medrag and must not be, so extraction is
delegated to `/data/wang/junh/envs/quickumls` via JSONL on disk; see
`_quickumls_extract.py`.  This script owns the corpus, the semantic-type
re-check against MRSTY, the bags, and the report.

v2 (`--v2`) is the addendum's ONE permitted follow-up, triggered because H2 was
refuted in v1.  It re-uses the v1 QuickUMLS matches unchanged -- no
re-extraction -- and differs from v1 in exactly two documented ways:

    1. NegEx (negspacy, `en_clinical` termset) over the QuickUMLS spans;
       matches inside a negation scope are dropped.  See `_negex_filter.py`.
    2. the "his" -> C0019602 histidine artifact is dropped.  All 1,026 v1
       matches of that CUI come from the pronoun, so dropping the CUI and
       dropping the artifact are the same operation here.

Outputs
    latent/data/n2c2_concept_bags.parquet        patient_id, t, cuis, n_concepts
    latent/results/n2c2_concept_extraction.json  the report
    ...{_v2}.parquet / ...{_v2}.json             under --v2

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_n2c2_concepts.py
    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_n2c2_concepts.py --v2
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_LATENT = Path(__file__).resolve().parents[1]
for p in (str(_LATENT / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

import n2c2_concepts as nc  # noqa: E402

DATA = _LATENT / "data"
RESULTS = _LATENT / "results"
TMP = DATA / "_h2_extract"
QUICKUMLS_PY = Path("/data/wang/junh/envs/quickumls/bin/python")
EXTRACTOR = Path(__file__).resolve().parent / "_quickumls_extract.py"
NEGEX = Path(__file__).resolve().parent / "_negex_filter.py"

THRESHOLD = 0.9
SIMILARITY = "jaccard"

#: v2 only.  Every one of the 1,026 v1 matches of C0019602 (histidine) is the
#: pronoun "his"; the CUI and the artifact are the same thing in this corpus.
V2_BLOCKED_CUIS = {"C0019602": "histidine, matched only by the pronoun 'his'"}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--reuse-matches", action="store_true",
                    help="skip extraction, re-read the existing matches JSONL")
    ap.add_argument("--v2", action="store_true",
                    help="NegEx-filtered v2 (implies --reuse-matches): drop "
                         "negated matches and the histidine artifact")
    ap.add_argument("--reuse-negation", action="store_true",
                    help="v2 only: re-read the existing negation JSONL")
    a = ap.parse_args()
    if a.v2:
        a.reuse_matches = True
    sfx = "_v2" if a.v2 else ""

    t_start = time.time()
    TMP.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(DATA / "n2c2_corpus.parquet")
    # The parquet is already in (patient asc, t asc) order and the state .npz
    # files are row-aligned to it (their meta_json says so, and it is asserted
    # in the H2 runner).  Do not reorder -- the bag rows must stay aligned too.
    srt = corpus.sort_values(["patient_idx", "t"])
    assert np.array_equal(srt.index.to_numpy(), np.arange(len(corpus))), \
        "corpus parquet is not in (patient_idx, t) order"
    n = len(corpus)
    print(f"[corpus] {n} records / {corpus.patient_id.nunique()} patients",
          flush=True)

    in_jsonl, out_jsonl = TMP / "records.jsonl", TMP / "matches.jsonl"
    timings: Dict[str, Any] = {}
    if not a.reuse_matches:
        with open(in_jsonl, "w", encoding="utf-8") as fh:
            for r, txt in enumerate(corpus["text"]):
                fh.write(json.dumps({"row": r, "text": txt}) + "\n")
        t0 = time.time()
        cmd = [str(QUICKUMLS_PY), str(EXTRACTOR), "--in", str(in_jsonl),
               "--out", str(out_jsonl), "--index", str(nc.QUICKUMLS_FP),
               "--threshold", str(THRESHOLD), "--similarity", SIMILARITY,
               "--jobs", str(a.jobs)]
        print("[extract] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        timings["extraction_seconds"] = round(time.time() - t0, 1)

    with open(out_jsonl, encoding="utf-8") as fh:
        matches = {d["row"]: d["matches"] for d in map(json.loads, fh)}
    assert len(matches) == n, f"{len(matches)} match rows for {n} records"

    # -- v2 only: NegEx over the SAME matches, no re-extraction ------------
    neg_jsonl = TMP / "negation.jsonl"
    negated: Dict[int, List[bool]] = {}
    if a.v2:
        if not a.reuse_negation:
            t0 = time.time()
            cmd = [str(QUICKUMLS_PY), str(NEGEX), "--records", str(in_jsonl),
                   "--matches", str(out_jsonl), "--out", str(neg_jsonl),
                   "--jobs", str(a.jobs)]
            print("[negex] " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            timings["negex_seconds"] = round(time.time() - t0, 1)
        with open(neg_jsonl, encoding="utf-8") as fh:
            negated = {d["row"]: d["negated"] for d in map(json.loads, fh)}
        assert len(negated) == n
        for r in range(n):
            assert len(negated[r]) == len(matches[r]), f"row {r} misaligned"

    # Re-check every returned CUI's semantic types against MRSTY directly, so
    # the filter does not rest on the pre-built index's own semtype table.
    t0 = time.time()
    cui2tui = nc.load_cui_semtypes()
    timings["mrsty_load_seconds"] = round(time.time() - t0, 1)
    print(f"[mrsty] {len(cui2tui)} CUIs carry an accepted TUI", flush=True)

    bags: List[List[str]] = []
    bags_v1: List[List[str]] = []
    n_raw = n_dropped = n_negated = n_blocked = 0
    sim: List[float] = []
    tui_counter: Counter = Counter()
    negated_tui: Counter = Counter()
    negated_cui: Counter = Counter()
    ngram_by_cui: Dict[str, Counter] = {}
    for r in range(n):
        keep: set = set()
        keep_v1: set = set()
        for k, m in enumerate(matches[r]):
            n_raw += 1
            tuis = cui2tui.get(m["cui"])
            if not tuis:
                n_dropped += 1
                continue
            keep_v1.add(m["cui"])
            if a.v2:
                if negated[r][k]:
                    n_negated += 1
                    negated_tui.update(tuis)
                    negated_cui[m["cui"]] += 1
                    continue
                if m["cui"] in V2_BLOCKED_CUIS:
                    n_blocked += 1
                    continue
            keep.add(m["cui"])
            sim.append(m["similarity"])
            tui_counter.update(tuis)
            ngram_by_cui.setdefault(m["cui"], Counter())[m["ngram"].lower()] += 1
        bags.append(sorted(keep))
        bags_v1.append(sorted(keep_v1))

    bags_df = pd.DataFrame({
        "patient_id": corpus["patient_id"].to_numpy(),
        "patient_idx": corpus["patient_idx"].to_numpy(),
        "t": corpus["t"].to_numpy(),
        "cuis": bags,
        "n_concepts": [len(b) for b in bags],
    })
    bags_df.to_parquet(DATA / f"n2c2_concept_bags{sfx}.parquet", index=False)

    # vocabulary before / after the min-df cut
    df_counts: Counter = Counter()
    for b in bags:
        df_counts.update(b)
    vocab_all = len(df_counts)
    vocab_kept = sum(1 for v in df_counts.values() if v >= nc.MIN_DF)

    npc = np.array([len(b) for b in bags], dtype=float)
    names = nc.load_preferred_names(set(df_counts))

    # three example records, spread over the concepts-per-record distribution,
    # with the ten most frequent concepts in each and the span that matched.
    order = np.argsort(npc, kind="stable")
    picks = [int(order[len(order) // 10]), int(order[len(order) // 2]),
             int(order[-len(order) // 10])]
    examples = []
    for r in picks:
        top = sorted(bags[r], key=lambda c: (-df_counts[c], c))[:10]
        examples.append({
            "row": r, "patient_id": str(corpus["patient_id"].iloc[r]),
            "t": int(corpus["t"].iloc[r]),
            "n_chars": int(len(corpus["text"].iloc[r])),
            "n_concepts": int(len(bags[r])),
            "text_head": corpus["text"].iloc[r][:300],
            "concepts": [
                {"cui": c, "name": names.get(c, "?"),
                 "semtypes": list(cui2tui[c]),
                 "matched_text": ngram_by_cui[c].most_common(1)[0][0]}
                for c in top],
        })

    report: Dict[str, Any] = {
        "stage": f"H2 step 1 (concept extraction){', v2 NegEx-filtered' if a.v2 else ''}",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": str(DATA / "n2c2_corpus.parquet"),
        "n_records": int(n), "n_patients": int(corpus.patient_id.nunique()),
        "extraction": {
            "tool": "QuickUMLS",
            "index": str(nc.QUICKUMLS_FP),
            "umls_release": "2024AA",
            "threshold": THRESHOLD, "similarity": SIMILARITY,
            "window": 5, "min_match_length": 3,
            "overlapping_criteria": "score", "best_match": True,
            "accepted_semtypes": sorted(nc.ACCEPTED_TUIS),
            "semtype_groups": {k: list(v) for k, v in nc.TUI_GROUPS.items()},
            "negation_handling": (
                "negspacy 1.0.4 Negex, en_clinical termset, over the QuickUMLS "
                "spans (see _negex_filter.py); negated matches dropped"
                if a.v2 else "NONE (pre-registered v1 caveat)"),
        },
        "matches_raw": int(n_raw),
        "matches_dropped_by_mrsty_recheck": int(n_dropped),
        "similarity_mean": float(np.mean(sim)) if sim else None,
        "similarity_min": float(np.min(sim)) if sim else None,
        "concepts_per_record": {
            "mean": float(npc.mean()), "std": float(npc.std(ddof=1)),
            "min": int(npc.min()), "max": int(npc.max()),
            "quantiles": {q: float(np.quantile(npc, q / 100))
                          for q in (5, 25, 50, 75, 95)},
            "n_records_with_zero": int((npc == 0).sum()),
        },
        "vocabulary": {
            "distinct_cuis_before_min_df": int(vocab_all),
            "min_df": nc.MIN_DF,
            "distinct_cuis_after_min_df": int(vocab_kept),
            "dropped": int(vocab_all - vocab_kept),
        },
        "semtype_hits": {t: int(c) for t, c in
                         sorted(tui_counter.items(), key=lambda kv: -kv[1])},
        "examples": examples,
        "timings": timings,
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "outputs": {"bags": str(DATA / f"n2c2_concept_bags{sfx}.parquet")},
    }

    if a.v2:
        npc1 = np.array([len(b) for b in bags_v1], dtype=float)
        # per-record concept mass removed, as a fraction of the v1 bag
        frac = np.divide(npc1 - npc, npc1, out=np.zeros_like(npc1),
                         where=npc1 > 0)
        v1c: Counter = Counter()
        for b in bags_v1:
            v1c.update(b)
        neg_names = nc.load_preferred_names(
            set(list(dict(negated_cui.most_common(15)))))
        report["v2"] = {
            "definition": [
                "1. NegEx (negspacy 1.0.4, en_clinical termset: 62 preceding, "
                "11 following, 24 pseudo-negation, 34 termination phrases) run "
                "over the v1 QuickUMLS character spans; a match inside a "
                "negation scope is dropped",
                "2. C0019602 (histidine) dropped -- all 1,026 v1 matches of it "
                "are the pronoun 'his'",
            ],
            "reuses_v1_matches": True,
            "matches_negated_dropped": int(n_negated),
            "matches_blocked_cui_dropped": int(n_blocked),
            "matches_kept": int(n_raw - n_dropped - n_negated - n_blocked),
            "frac_of_accepted_matches_negated": float(
                n_negated / max(n_raw - n_dropped, 1)),
            "blocked_cuis": V2_BLOCKED_CUIS,
            "concepts_per_record_v1": {
                "mean": float(npc1.mean()), "median": float(np.median(npc1))},
            "concept_mass_removed_per_record": {
                "mean_frac": float(frac.mean()),
                "median_frac": float(np.median(frac)),
                "q25_frac": float(np.quantile(frac, .25)),
                "q75_frac": float(np.quantile(frac, .75)),
                "max_frac": float(frac.max()),
                "mean_concepts_removed": float((npc1 - npc).mean()),
            },
            "vocabulary_v1": {
                "distinct_cuis_before_min_df": int(len(v1c)),
                "distinct_cuis_after_min_df": int(
                    sum(1 for x in v1c.values() if x >= nc.MIN_DF))},
            "negated_semtype_hits": {t: int(c) for t, c in
                                     sorted(negated_tui.items(),
                                            key=lambda kv: -kv[1])},
            "most_negated_concepts": [
                {"cui": c, "name": neg_names.get(c, "?"), "n_negated": int(k)}
                for c, k in negated_cui.most_common(15)],
        }

    (RESULTS / f"n2c2_concept_extraction{sfx}.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"},
                     indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
