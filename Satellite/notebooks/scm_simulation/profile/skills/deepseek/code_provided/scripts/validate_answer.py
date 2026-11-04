#!/usr/bin/env python3
"""Validate a synthetic-control ``answer.json`` against the required contract.

Checks that the answer file parses and contains:
  - ``counterfactual``: an object mapping each requested treated period (as a
    string) to a finite number.
  - ``att``: a finite number equal (within tolerance) to the mean of
    ``observed - counterfactual`` over the requested periods.
  - ``rationale``: a non-empty string.

If ``--panel`` is supplied, the observed outcomes for the requested periods are
read from the panel so the ATT consistency check can be enforced.

Prints a JSON verdict on stdout (or to ``--report``) and exits non-zero when the
answer is invalid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 validate_answer.py --answer ./answer.json \\\n"
            "    --panel ./tobacco_panel.csv --treated-periods 1989,1990,1991\n"
        ),
    )
    parser.add_argument("--answer", required=True, help="Path to answer.json.")
    parser.add_argument("--panel", default=None, help="Optional panel CSV for the ATT consistency check.")
    parser.add_argument(
        "--treated-periods",
        default=None,
        help="Comma-separated treated periods expected as keys in counterfactual.",
    )
    parser.add_argument("--treated-unit", default="treated", help="Treated unit_id (default treated).")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="ATT tolerance (default 5e-3, since the estimator rounds to 4 decimals).")
    parser.add_argument("--report", default=None, help="Optional path to write the JSON verdict.")
    return parser.parse_args(argv)


def finish(payload: dict, report: str | None, code: int) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    sys.exit(code)


def to_period_key(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return ("%g" % number)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    answer_path = Path(args.answer)
    if not answer_path.is_file():
        finish({"status": "error", "error": f"answer file not found: {answer_path}"}, args.report, 2)

    try:
        payload = json.loads(answer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        finish({"status": "error", "error": f"answer.json is not valid JSON: {exc}"}, args.report, 2)

    problems: list[str] = []

    counterfactual = payload.get("counterfactual")
    if not isinstance(counterfactual, dict) or not counterfactual:
        problems.append("counterfactual must be a non-empty object")
        counterfactual = {}

    for key, value in counterfactual.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            problems.append(f"counterfactual[{key!r}] is not a finite number: {value!r}")

    att = payload.get("att")
    if not isinstance(att, (int, float)) or isinstance(att, bool) or not math.isfinite(att):
        problems.append(f"att is not a finite number: {att!r}")
        att = None

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        problems.append("rationale must be a non-empty string")

    requested = None
    if args.treated_periods:
        requested = [to_period_key(p) for p in args.treated_periods.split(",")]
        missing = [p for p in requested if p not in {to_period_key(k) for k in counterfactual}]
        if missing:
            problems.append(f"counterfactual is missing requested periods: {missing}")

    observed: dict[str, float] = {}
    if args.panel:
        panel_path = Path(args.panel)
        if not panel_path.is_file():
            problems.append(f"panel not found: {panel_path}")
        else:
            with panel_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["unit_id"] != args.treated_unit:
                        continue
                    if requested is not None and to_period_key(row["period"]) not in requested:
                        continue
                    try:
                        observed[to_period_key(row["period"])] = float(row["Y_observed"])
                    except (TypeError, ValueError):
                        problems.append(f"non-numeric Y_observed for period {row['period']}")
            if requested is not None:
                for p in requested:
                    if p not in observed:
                        problems.append(f"panel has no observed outcome for treated period {p}")

    computed_att = None
    if observed and counterfactual and att is not None and not problems:
        cf_lookup = {to_period_key(k): v for k, v in counterfactual.items()}
        gaps = [
            observed[p] - cf_lookup[p]
            for p in observed
            if p in cf_lookup and isinstance(cf_lookup[p], (int, float))
        ]
        if gaps and len(gaps) == len(observed):
            computed_att = sum(gaps) / len(gaps)
            if abs(computed_att - att) > args.tolerance:
                problems.append(
                    f"att {att} does not equal mean(observed - counterfactual) {computed_att}"
                )

    verdict = {
        "status": "ok" if not problems else "error",
        "answer": str(answer_path),
        "counterfactual_periods": sorted(to_period_key(k) for k in counterfactual),
        "att": att,
        "computed_att": computed_att,
        "problems": problems,
    }
    finish(verdict, args.report, 0 if not problems else 1)


if __name__ == "__main__":
    main()
