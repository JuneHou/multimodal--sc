"""
L0 runner: render the NCT00174655 visit corpus to text and report on it.

Writes
    latent/data/rendered_visits.parquet   patient_idx, t, text  (9,092 rows)
    latent/results/render_report.json     corpus stats + token-length
                                          distributions under BOTH encoders'
                                          tokenizers

Reads the PyTrial release read-only.  Touches nothing under `mimic/` or
`observable/` except by import.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_render.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "src"), str(_PRELIM / "mimic"), str(_PRELIM / "observable" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_trial import render_corpus, SECTIONS  # noqa: E402

DATA_DIR = Path("/data/wang/junh/datasets/PDS/NCT00174655/seq_patient")
OUT_PARQUET = _LATENT / "data" / "rendered_visits.parquet"
OUT_REPORT = _LATENT / "results" / "render_report.json"

TOKENIZERS = {
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "biolord": "FremyCompany/BioLORD-2023",
}
#: Encoder input limits the corpus is asserted against.  BioLORD's 512 is the
#: value the pre-registration sets `max_seq_length` to (it ships 128).
LIMITS = {"qwen3": 32768, "biolord": 512}


def _describe(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=float)
    q = np.percentile(v, [0, 25, 50, 75, 90, 95, 99, 100])
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "max": float(q[7])}


def main() -> int:
    t_start = time.time()
    import dill

    with open(DATA_DIR / "visit.pkl", "rb") as f:
        visit = dill.load(f)
    with open(DATA_DIR / "voc.pkl", "rb") as f:
        voc = dill.load(f)

    df = render_corpus(visit, voc)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # ---- corpus stats -----------------------------------------------------
    texts = df["text"].tolist()
    n_codes: Dict[str, List[int]] = {k: [] for k, _p, _h in SECTIONS}
    n_empty: Dict[str, int] = {k: 0 for k, _p, _h in SECTIONS}
    for patient in visit:
        for rec in patient:
            for key, pos, _h in SECTIONS:
                codes = {int(c) for c in rec[pos] if int(c) != 0}
                n_codes[key].append(len(codes))
                if not codes:
                    n_empty[key] += 1

    chars = np.array([len(s) for s in texts])
    stats: Dict[str, Any] = {
        "n_patients": int(len(visit)),
        "n_visits": int(len(df)),
        "visits_per_patient": _describe(np.array([len(p) for p in visit])),
        "unique_texts": int(df["text"].nunique()),
        "duplicate_text_fraction": float(1 - df["text"].nunique() / len(df)),
        "char_length": _describe(chars),
        "codes_per_visit": {k: _describe(np.array(v)) for k, v in n_codes.items()},
        "empty_section_counts": {k: int(v) for k, v in n_empty.items()},
        "empty_both_sections": int(sum(
            1 for a, b in zip(n_codes["medication"], n_codes["adverse_event"])
            if a == 0 and b == 0)),
        "vocab_sizes": {k: len(voc[k].idx2word) - 1 for k, _p, _h in SECTIONS},
        "examples": [texts[i] for i in (0, 1, 2, 4000, 9091)],
        "longest_example": texts[int(np.argmax(chars))],
    }

    # ---- token lengths under both tokenizers ------------------------------
    from transformers import AutoTokenizer

    tok_stats: Dict[str, Any] = {}
    for arm, repo in TOKENIZERS.items():
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo)
        lens = np.array([len(x) for x in
                         tok(texts, add_special_tokens=True)["input_ids"]])
        tok_stats[arm] = {
            "model": repo,
            "limit_asserted": LIMITS[arm],
            "n_over_limit": int((lens > LIMITS[arm]).sum()),
            "length": _describe(lens),
            "seconds": round(time.time() - t0, 1),
        }
        assert lens.max() <= LIMITS[arm], (
            f"{arm}: max token length {lens.max()} exceeds {LIMITS[arm]}")

    report = {
        "stage": "L0",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(DATA_DIR),
        "output_parquet": str(OUT_PARQUET),
        "render_rule": ("'Medications: <a>; <b>. Adverse events: <c>.' -- names "
                        "from voc idx2word, [PAD] dropped, treatment and "
                        "ae_serious excluded, alphabetised within section, "
                        "empty section -> 'none.'"),
        "code_names_are_opaque": ("medication = WHO drug codes (C_CTXWHO), "
                                  "adverse_event = COSTART codes (C_AECOS); the "
                                  "release ships no code->term dictionary, so "
                                  "the rendered text is digit strings"),
        "corpus": stats,
        "tokenizers": tok_stats,
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "corpus"}, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
