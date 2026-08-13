"""
L0 runner: parse the n2c2 corpus and cross-check it against the verification pass.

Writes
    latent/data/n2c2_corpus.parquet     one row per record (date-sorted, t=0..)
    latent/results/n2c2_parse_report.json

The cross-check against `latent/results/n2c2_verify.json` is a HARD assert on
patient count (288), record count (1,264) and **per-patient n_records**; dates
and criteria tags are compared too and any mismatch is fatal.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_n2c2_parse.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

_LATENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LATENT / "src"))

import numpy as np  # noqa: E402

import n2c2_corpus as nc  # noqa: E402

ROOT = _LATENT / "data" / "n2c2"
OUT_PARQUET = _LATENT / "data" / "n2c2_corpus.parquet"
VERIFY = _LATENT / "results" / "n2c2_verify.json"
REPORT = _LATENT / "results" / "n2c2_parse_report.json"


def main() -> int:
    t0 = time.time()
    df, per_patient = nc.build_corpus(ROOT)
    ver = json.loads(VERIFY.read_text())
    vpat = {p["id"]: p for p in ver["patients"]}

    n_pat = df["patient_id"].nunique()
    n_rec = len(df)
    mism_counts, mism_dates, mism_crit, mism_split = [], [], [], []
    for p in per_patient:
        v = vpat.get(p["patient_id"])
        if v is None:
            mism_counts.append({"patient_id": p["patient_id"],
                                "issue": "absent from n2c2_verify.json"})
            continue
        if v["n_records"] != p["n_records"]:
            mism_counts.append({"patient_id": p["patient_id"],
                                "parsed": p["n_records"], "verify": v["n_records"]})
        if sorted(v["dates_iso"]) != p["dates"]:
            mism_dates.append({"patient_id": p["patient_id"],
                               "parsed": p["dates"],
                               "verify_sorted": sorted(v["dates_iso"])})
        if v["criteria"] != p["criteria"]:
            mism_crit.append({"patient_id": p["patient_id"]})
        if v["split"] != p["split"]:
            mism_split.append({"patient_id": p["patient_id"]})

    # ---- HARD GATE -------------------------------------------------------
    assert n_pat == 288, f"patients {n_pat} != 288"
    assert n_rec == 1264, f"records {n_rec} != 1264"
    assert not mism_counts, f"per-patient n_records mismatch: {mism_counts[:5]}"
    assert not mism_dates, f"per-patient dates mismatch: {mism_dates[:5]}"
    assert not mism_crit, f"criteria mismatch: {mism_crit[:5]}"
    assert not mism_split, f"split mismatch: {mism_split[:5]}"
    assert ver["release_completeness"]["found"]["total"] == 288

    # every patient's t is a dense 0..n-1 range
    for pid, g in df.groupby("patient_id"):
        assert list(g["t"]) == list(range(len(g))), f"{pid}: t not 0..n-1"
        assert list(g["record_date"]) == sorted(g["record_date"]), (
            f"{pid}: records not date-sorted")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    reordered = [p["patient_id"] for p in per_patient
                 if not p["dates_ascending_in_file_order"]]
    ws = df["n_ws_tokens"].to_numpy()
    ch = df["n_chars"].to_numpy()
    report: Dict[str, Any] = {
        "stage": "L0 (parse; the addendum replaces rendering with parsing)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(ROOT),
        "sources_used": {"train": nc.TRAIN_GLOB, "test": nc.TEST_GLOB},
        "sources_ignored": [
            "n2c2-t1_gold_standard_test_data-20260811T175327Z-1-001 "
            "(AppleDouble stub, empty test/)",
            "n2c2_track1_abstracts-* (papers)",
            "n2c2-community-annotations_2018-khetan-MIMICause (other task)",
            "track1_eval_script-* (organisers' scorer)",
        ],
        "record_separator": "^Record date: at line start (re.M lookahead split)",
        "record_text": ("verbatim record incl. its `Record date:` header line; "
                        "lines consisting only of `*` (banner separators) "
                        "removed; leading/trailing whitespace stripped"),
        "ordering": "records sorted by parsed date (stable); t = 0-based index",
        "n_patients": int(n_pat),
        "n_records": int(n_rec),
        "records_per_patient_hist": dict(sorted(
            Counter(df.groupby("patient_id").size()).items())),
        "by_split": {s: {"n_patients": int(g["patient_id"].nunique()),
                         "n_records": int(len(g))}
                     for s, g in df.groupby("split")},
        "patients_non_chronological_in_file_order": reordered,
        "ws_tokens_per_record": {
            "mean": float(ws.mean()), "median": float(np.median(ws)),
            "min": int(ws.min()), "max": int(ws.max()),
            "p90": float(np.percentile(ws, 90)),
            "p99": float(np.percentile(ws, 99))},
        "chars_per_record": {
            "mean": float(ch.mean()), "median": float(np.median(ch)),
            "min": int(ch.min()), "max": int(ch.max())},
        "criteria_counts": {
            name: dict(Counter(
                df.drop_duplicates("patient_id")[f"crit_{name}"]))
            for name in nc.CRITERIA},
        "cross_check_vs_n2c2_verify": {
            "file": str(VERIFY),
            "patient_count_match": True,
            "record_count_match": True,
            "per_patient_n_records_match": True,
            "per_patient_dates_match": True,
            "per_patient_criteria_match": True,
            "split_assignment_match": True,
            "verify_expected_non_chronological": ["112", "344"],
            "parsed_non_chronological": reordered,
            "non_chronological_agrees": sorted(reordered) == ["112", "344"],
            "verify_n_patients_dates_ascending": ver["aggregate"][
                "n_patients_dates_ascending"],
        },
        "output_parquet": str(OUT_PARQUET),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
