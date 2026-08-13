"""H3 metric (d) -- code recovery F1 of the generated records.

The pre-registration lists (d) as an H3 metric but the H3 success criteria use
only (a), (b) and (c); this is descriptive.  It was to be **deferred with H2**
unless a UMLS extraction existed.  When L3/L4 ran, the concurrent H2
workstream's extraction landed
(`latent/data/n2c2_concept_bags.parquet`, 17:43), so the condition triggered
and (d) is computed here rather than deferred.

Method: the 500 generated records go through the SAME extractor and the SAME
settings as the corpus (`scripts/_quickumls_extract.py` under the quickumls
interpreter, QuickUMLS 2024AA, jaccard 0.9, window 5, min match 3, the 15
accepted TUIs, no negation -- the pre-registered v1 space), then every bag is
restricted to the corpus vocabulary (document frequency >= 5, 1,915 CUIs).

"The mixture's expected code set" is not defined by the pre-registration.
Three readings are reported, the weighted one first:
    weighted   {c : sum_j w_j * 1[c in bag_j] >= 0.5}   -- the concepts
               carrying at least half the mixture's weight
    union      the union of the K donor bags
    top_donor  the top-weight donor's bag alone
and each is scored for the generated record AND for the baseline decode (the
top-weight donor's real record), so the F1 has something to be compared to.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_l3_code_recovery.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd

_LATENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LATENT / "src"))

RESULTS = _LATENT / "results"
DATA = _LATENT / "data"
QU_PY = Path("/data/wang/junh/envs/quickumls/bin/python")
EXTRACT = _LATENT / "scripts" / "_quickumls_extract.py"
IN_JSONL = RESULTS / "_l3_gen_extract_in.jsonl"
OUT_JSONL = RESULTS / "_l3_gen_extract_out.jsonl"
MIN_DF = 5


def prf(pred: Set[str], exp: Set[str]) -> Dict[str, float]:
    tp = len(pred & exp)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(exp) if exp else float("nan")
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp,
            "n_pred": len(pred), "n_expected": len(exp)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    l3 = pd.read_parquet(RESULTS / "l3_generation.parquet")
    bags_df = pd.read_parquet(DATA / "n2c2_concept_bags.parquet")
    assert len(bags_df) == 1264
    corpus_bags: List[Set[str]] = [set(x) for x in bags_df["cuis"]]

    df_count: Dict[str, int] = {}
    for b in corpus_bags:
        for c in b:
            df_count[c] = df_count.get(c, 0) + 1
    vocab = {c for c, k in df_count.items() if k >= MIN_DF}
    corpus_bags = [b & vocab for b in corpus_bags]

    # ------------------------------------------------- extract from generated
    if not (args.reuse and OUT_JSONL.exists()):
        with IN_JSONL.open("w", encoding="utf-8") as fh:
            for i, t in enumerate(l3["generated_text"]):
                fh.write(json.dumps({"row": i, "text": t}) + "\n")
        t0 = time.time()
        r = subprocess.run([str(QU_PY), str(EXTRACT), "--in", str(IN_JSONL),
                            "--out", str(OUT_JSONL), "--jobs", str(args.jobs)])
        if r.returncode != 0:
            raise SystemExit(f"extraction failed: {r.returncode}")
        extract_seconds = round(time.time() - t0, 1)
    else:
        extract_seconds = "reused"

    gen_bags: List[Set[str]] = [set() for _ in range(len(l3))]
    n_raw = 0
    for ln in OUT_JSONL.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        n_raw += len(d["matches"])
        gen_bags[d["row"]] = {m["cui"] for m in d["matches"]} & vocab

    # ------------------------------------------------------------- expected
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(l3.itertuples()):
        donors = [int(x) for x in r.donor_rows]
        w = [float(x) for x in r.weights]
        support: Dict[str, float] = {}
        for wj, dj in zip(w, donors):
            for c in corpus_bags[dj]:
                support[c] = support.get(c, 0.0) + wj
        exp = {
            "weighted": {c for c, s in support.items() if s >= 0.5},
            "union": set(support),
            "top_donor": set(corpus_bags[int(r.top_donor_row)]),
        }
        base_bag = corpus_bags[int(r.top_donor_row)]
        rec: Dict[str, Any] = {"mixture_id": r.mixture_id, "family": r.family,
                               "n_concepts_generated": len(gen_bags[i])}
        for name, e in exp.items():
            g = prf(gen_bags[i], e)
            b = prf(base_bag, e)
            rec[f"code_f1_{name}"] = g["f1"]
            rec[f"code_precision_{name}"] = g["precision"]
            rec[f"code_recall_{name}"] = g["recall"]
            rec[f"code_f1_{name}_baseline"] = b["f1"]
            rec[f"n_expected_{name}"] = g["n_expected"]
        rows.append(rec)
    cr = pd.DataFrame(rows)

    def summ(col: str) -> Dict[str, float]:
        v = cr[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "mean": float(v.mean()),
                "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)),
                "q75": float(np.percentile(v, 75))}

    report: Dict[str, Any] = {
        "stage": "H3 metric (d): code recovery F1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "computed",
        "why_not_deferred": ("the pre-registration defers (d) with H2 unless a "
                             "UMLS extraction exists; the concurrent H2 "
                             "workstream's extraction landed during the L3 run "
                             "(latent/data/n2c2_concept_bags.parquet, 17:43), "
                             "so the condition triggered"),
        "note": ("H3's success criteria use (a), (b) and (c) only; (d) is "
                 "descriptive and no verdict depends on it"),
        "extraction": {
            "worker": str(EXTRACT), "interpreter": str(QU_PY),
            "settings": "the corpus extraction's defaults: QuickUMLS 2024AA, "
                        "jaccard, threshold 0.9, window 5, min match 3, 15 "
                        "accepted TUIs, best_match, no negation (v1)",
            "corpus_bags": str(DATA / "n2c2_concept_bags.parquet"),
            "seconds": extract_seconds,
            "raw_matches_generated": n_raw,
        },
        "vocabulary": {"min_df": MIN_DF, "size": len(vocab),
                       "scope": "concepts with document frequency >= 5 in the "
                                "1,264-record corpus; every bag is restricted "
                                "to it"},
        "concepts_per_record": {
            "generated": summ("n_concepts_generated"),
            "real_corpus": {"mean": float(np.mean([len(b) for b in corpus_bags])),
                            "median": float(np.median([len(b) for b in corpus_bags]))},
        },
        "expected_set_definitions": {
            "weighted": "concepts whose donor weight sums to >= 0.5",
            "union": "union of the K donor bags",
            "top_donor": "the top-weight donor's bag",
        },
        "f1": {},
    }
    for name in ("weighted", "union", "top_donor"):
        report["f1"][name] = {
            "generated": summ(f"code_f1_{name}"),
            "precision": summ(f"code_precision_{name}"),
            "recall": summ(f"code_recall_{name}"),
            "baseline_top_weight_donor_record": summ(f"code_f1_{name}_baseline"),
            "expected_set_size": summ(f"n_expected_{name}"),
            "by_family": {f: float(np.mean(cr.loc[cr.family == f,
                                                  f"code_f1_{name}"]))
                          for f in ("dirichlet", "spiky")},
        }

    (RESULTS / "l3_code_recovery.json").write_text(json.dumps(report, indent=2))

    # fold the per-mixture columns into the L3 parquet
    l3 = l3.merge(cr.drop(columns=["family"]), on="mixture_id", how="left")
    l3.to_parquet(RESULTS / "l3_generation.parquet", index=False)

    # and the summary into l3_summary.json
    sp = RESULTS / "l3_summary.json"
    s = json.loads(sp.read_text())
    s["d_code_recovery"] = report
    for d in s.get("deviations", []):
        if d.get("item") == "H3 metric (d), code recovery F1":
            d["deviation"] = ("computed after all -- the H2 extraction landed "
                              "mid-run; see l3_code_recovery.json")
            d["reason"] = report["why_not_deferred"]
    sp.write_text(json.dumps(s, indent=2))

    print(json.dumps(report["f1"], indent=2), flush=True)
    print(f"[l3d] {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
