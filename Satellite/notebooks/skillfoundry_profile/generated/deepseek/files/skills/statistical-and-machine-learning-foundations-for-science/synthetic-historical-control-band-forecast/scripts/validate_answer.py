#!/usr/bin/env python3
"""Validate an SHC ``answer.json`` against the required output contract.

The grading contract for the parcel-forecast task is::

    {"prediction": {"<band>": <number>, ...}, "rationale": "<text>"}

This script checks that a candidate ``answer.json`` (a) parses as JSON, (b)
has exactly the keys ``prediction`` and ``rationale``, (c) maps every band to
a finite JSON number, and (d) carries a non-empty rationale string. It exits
non-zero with a descriptive message on any violation so it can gate an
automated pipeline.

Optionally pass ``--expected-bands`` to require an exact set of band keys, and
``--truth`` to compute per-band absolute error against known next-time values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def validate(answer: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(answer, dict):
        return ["answer.json must be a JSON object"]
    extra = set(answer) - {"prediction", "rationale"}
    if extra:
        errors.append(f"unexpected top-level keys: {sorted(extra)}")
    if "prediction" not in answer:
        errors.append("missing required key 'prediction'")
    if "rationale" not in answer:
        errors.append("missing required key 'rationale'")

    prediction = answer.get("prediction")
    if not isinstance(prediction, dict) or not prediction:
        errors.append("'prediction' must be a non-empty object mapping band -> number")
    else:
        for band, value in prediction.items():
            if not isinstance(band, str) or not band:
                errors.append(f"band key must be a non-empty string, got {band!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"prediction for band {band!r} must be a JSON number, got {type(value).__name__}")
            elif not math.isfinite(float(value)):
                errors.append(f"prediction for band {band!r} is not finite: {value!r}")

    rationale = answer.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("'rationale' must be a non-empty string")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 validate_answer.py --answer answer.json "
            "--expected-bands ndvi,nir,red,swir\n"
        ),
    )
    parser.add_argument("--answer", type=Path, default=Path("answer.json"),
                        help="Path to the answer.json to validate (default: answer.json).")
    parser.add_argument("--expected-bands", type=str, default=None,
                        help="Comma-separated exact set of band keys that must be present.")
    parser.add_argument("--truth", type=Path, default=None,
                        help="Optional JSON file mapping band -> true next value for error reporting.")
    args = parser.parse_args(argv)

    if not args.answer.exists():
        print(f"error: answer file not found: {args.answer}", file=sys.stderr)
        return 2
    try:
        answer = json.loads(args.answer.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: {args.answer} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    errors = validate(answer)
    if args.expected_bands:
        expected = {b.strip() for b in args.expected_bands.split(",") if b.strip()}
        actual = set(answer.get("prediction", {})) if isinstance(answer.get("prediction"), dict) else set()
        if expected != actual:
            errors.append(f"band set mismatch: expected {sorted(expected)}, got {sorted(actual)}")

    metrics = None
    if args.truth is not None:
        try:
            truth = json.loads(args.truth.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: could not read truth file {args.truth}: {exc}", file=sys.stderr)
            return 2
        errors_out = []
        for band, true_value in truth.items():
            pred = answer.get("prediction", {}).get(band)
            if pred is None:
                continue
            errors_out.append({"band": band, "predicted": pred, "true": true_value,
                               "abs_error": round(abs(pred - true_value), 8)})
        if errors_out:
            metrics = {
                "per_band": errors_out,
                "mean_abs_error": round(
                    sum(item["abs_error"] for item in errors_out) / len(errors_out), 8),
            }

    if errors:
        for message in errors:
            print(f"error: {message}", file=sys.stderr)
        return 1

    summary = {
        "status": "valid",
        "answer": str(args.answer),
        "bands": sorted(answer["prediction"]),
        "rationale_chars": len(answer["rationale"]),
    }
    if metrics is not None:
        summary["accuracy"] = metrics
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
