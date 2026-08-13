"""How much of a "generated" record is copied from its donors?

NOT pre-registered.  Added after the L3 worked examples showed the decoder
reproducing the dominant donor's record near-verbatim -- de-identification
surrogate name, MRN and record date included.  That possibility matters for
reading H3: a decoder that copies its top-weight donor scores well on
cycle-consistency and on perplexity without having synthesised anything, so the
copy rate has to be on the table next to those numbers.

Measure: word 8-gram shingles.  `copy_fraction` is the share of the generated
record's shingles that also occur in a donor record, per donor and pooled over
donors.  1.0 means every 8-word run in the output appears in some source.

Writes latent/results/copy_diagnostic.json and folds the summary into
l3_summary.json / l4_summary.json.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_copy_diagnostic.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

_LATENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LATENT / "src"))

RESULTS = _LATENT / "results"
N = 8


def shingles(text: str, n: int = N) -> Set[str]:
    w = re.findall(r"\S+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def frac(gen: Set[str], src: Set[str]) -> float:
    return len(gen & src) / len(gen) if gen else float("nan")


def _q(a) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "max": float(a.max())}


def main() -> int:
    t_start = time.time()
    import genmix as gm

    C = gm.load_corpus(_LATENT)
    df = C["df"]
    real = df["text"].tolist()
    real_sh = [shingles(t) for t in real]

    l3 = pd.read_parquet(RESULTS / "l3_generation.parquet")
    l4 = pd.read_parquet(RESULTS / "l4_interpolation.parquet")

    # ------------------------------------------------------------------ L3
    rows: List[Dict[str, Any]] = []
    for r in l3.itertuples():
        g = shingles(r.generated_text)
        donors = [int(x) for x in r.donor_rows]
        w = [float(x) for x in r.weights]
        per = [frac(g, real_sh[d]) for d in donors]
        j = int(np.argmax(per))
        pooled = frac(g, set().union(*[real_sh[d] for d in donors]))
        rows.append({
            "mixture_id": r.mixture_id, "family": r.family,
            "copy_fraction_pooled": pooled,
            "copy_fraction_best_donor": per[j],
            "best_donor_weight": w[j],
            "best_donor_is_top_weight": bool(j == int(np.argmax(w))),
            "copy_fraction_top_weight_donor": per[int(np.argmax(w))],
            "n_donors_with_copy_over_0.2": int(sum(1 for x in per if x > 0.2)),
        })
    c3 = pd.DataFrame(rows)

    # ------------------------------------------------------------------ L4
    rows4: List[Dict[str, Any]] = []
    for r in l4.itertuples():
        g = shingles(r.generated_text)
        fm = frac(g, real_sh[int(r.met_row)])
        fn = frac(g, real_sh[int(r.notmet_row)])
        rows4.append({
            "pair_id": r.pair_id, "alpha": float(r.alpha),
            "copy_fraction_met": fm, "copy_fraction_notmet": fn,
            "copy_fraction_pooled": frac(
                g, real_sh[int(r.met_row)] | real_sh[int(r.notmet_row)]),
            "copies_the_heavier_source": bool(
                (fm > fn) if r.alpha > 0.5 else (fn > fm) if r.alpha < 0.5
                else True),
        })
    c4 = pd.DataFrame(rows4)

    report: Dict[str, Any] = {
        "stage": "copy diagnostic (not pre-registered)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "why": ("the L3 worked examples showed the decoder reproducing the "
                "dominant donor near-verbatim; a copying decoder scores well "
                "on cycle-consistency and perplexity without synthesising, so "
                "the copy rate belongs beside those numbers"),
        "measure": f"share of the generated record's word {N}-gram shingles "
                   f"that also occur in a source record",
        "l3": {
            "pooled_over_donors": _q(c3["copy_fraction_pooled"]),
            "best_single_donor": _q(c3["copy_fraction_best_donor"]),
            "top_weight_donor": _q(c3["copy_fraction_top_weight_donor"]),
            "share_best_donor_is_the_top_weight_donor":
                float(c3["best_donor_is_top_weight"].mean()),
            "share_pooled_copy_over_50pct":
                float((c3["copy_fraction_pooled"] > 0.5).mean()),
            "share_pooled_copy_over_80pct":
                float((c3["copy_fraction_pooled"] > 0.8).mean()),
            "mean_donors_contributing_over_20pct":
                float(c3["n_donors_with_copy_over_0.2"].mean()),
            "by_family": {f: {
                "pooled_median": float(np.median(
                    c3.loc[c3.family == f, "copy_fraction_pooled"])),
                "top_weight_donor_median": float(np.median(
                    c3.loc[c3.family == f, "copy_fraction_top_weight_donor"])),
            } for f in ("dirichlet", "spiky")},
        },
        "l4": {
            "pooled_over_the_two_sources": _q(c4["copy_fraction_pooled"]),
            "by_alpha": {f"{a:.2f}": {
                "pooled_median": float(np.median(
                    c4.loc[c4.alpha == a, "copy_fraction_pooled"])),
                "met_median": float(np.median(
                    c4.loc[c4.alpha == a, "copy_fraction_met"])),
                "notmet_median": float(np.median(
                    c4.loc[c4.alpha == a, "copy_fraction_notmet"])),
            } for a in sorted(c4["alpha"].unique())},
        },
    }

    l3 = l3.merge(c3.drop(columns=["family"]), on="mixture_id", how="left")
    l4 = l4.merge(c4, on=["pair_id", "alpha"], how="left")
    l3.to_parquet(RESULTS / "l3_generation.parquet", index=False)
    l4.to_parquet(RESULTS / "l4_interpolation.parquet", index=False)
    (RESULTS / "copy_diagnostic.json").write_text(json.dumps(report, indent=2))
    for name, key in (("l3_summary.json", "l3"), ("l4_summary.json", "l4")):
        p = RESULTS / name
        s = json.loads(p.read_text())
        s["copy_diagnostic"] = {k: report[k] for k in
                                ("measure", "why", key)}
        p.write_text(json.dumps(s, indent=2))

    print(json.dumps(report["l3"], indent=2))
    print(json.dumps(report["l4"], indent=2))
    print(f"[copy] {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
