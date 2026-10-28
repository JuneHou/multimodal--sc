#!/usr/bin/env python3
"""Validate the shape and basic completeness of an SHC parcel-band forecast answer.

Example:
  python3 validate_answer.py --input data.csv --answer answer.json --out validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_band_names(path: Path) -> list[str]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        lowered = {name.strip().lower(): name for name in reader.fieldnames}
        band_column = None
        for candidate in ("band", "band_name", "channel", "feature", "variable"):
            if candidate in lowered:
                band_column = lowered[candidate]
                break
        if band_column is None:
            raise ValueError("Could not find a band column in the input CSV")
        seen = []
        seen_set = set()
        for row in reader:
            band = (row.get(band_column) or "").strip()
            if band and band not in seen_set:
                seen_set.add(band)
                seen.append(band)
    if not seen:
        raise ValueError(f"{path} did not contain any bands")
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input long-format data.csv file.")
    parser.add_argument("--answer", type=Path, required=True, help="answer.json produced by the forecast script.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON validation summary path.")
    args = parser.parse_args()

    bands = load_band_names(args.input)
    payload = json.loads(args.answer.read_text(encoding="utf-8"))
    predictions = payload.get("prediction")
    rationale = payload.get("rationale")
    if not isinstance(predictions, dict):
        raise SystemExit("answer.json is missing a top-level 'prediction' object")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SystemExit("answer.json is missing a non-empty top-level 'rationale' string")

    missing = [band for band in bands if band not in predictions]
    extras = sorted(set(predictions) - set(bands))
    non_numeric = [
        band for band, value in predictions.items()
        if not isinstance(value, (int, float))
    ]
    summary = {
        "band_count": len(bands),
        "missing_bands": missing,
        "extra_bands": extras,
        "non_numeric_bands": non_numeric,
        "rationale_length": len(rationale.strip()),
        "valid": not missing and not extras and not non_numeric,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
