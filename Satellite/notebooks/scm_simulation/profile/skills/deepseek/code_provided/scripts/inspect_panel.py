#!/usr/bin/env python3
"""Inspect and validate a synthetic-control panel CSV before fitting.

Reads a panel with columns ``unit_id``, ``period``, ``treatment_group``, ``Z1``,
``Z2`` and ``Y_observed`` (one row per unit and period) and reports whether the
data is shaped correctly for the Synth-based estimator in this skill:
balanced panels, enough donors, a treated unit, numeric outcome, and an
inferable pre-treatment window.

Emits a JSON report on stdout (or to ``--out``). Exit code is non-zero when a
*blocking* issue is found (e.g. missing columns or fewer than two donors), and
zero for warnings only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = ["unit_id", "period", "treatment_group", "Z1", "Z2", "Y_observed"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 inspect_panel.py --panel ./tobacco_panel.csv --treated-unit treated\n"
        ),
    )
    parser.add_argument("--panel", required=True, help="Path to the panel CSV file.")
    parser.add_argument(
        "--treated-unit",
        default="treated",
        help="unit_id value of the treated unit (default: treated).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON report (default: stdout).",
    )
    return parser.parse_args(argv)


def emit(payload: dict, out: str | None, exit_code: int) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    sys.exit(exit_code)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = Path(args.panel)
    if not path.is_file():
        emit({"status": "error", "error": f"panel not found: {path}"}, args.out, 2)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    blocking: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        emit(
            {
                "status": "error",
                "error": "missing required columns",
                "missing_columns": missing,
                "found_columns": columns,
            },
            args.out,
            2,
        )
    if not rows:
        emit({"status": "error", "error": "panel has zero data rows"}, args.out, 2)

    units: dict[str, dict[float, dict[str, str]]] = defaultdict(dict)
    bad_outcome = 0
    for row in rows:
        unit = row["unit_id"]
        try:
            period = float(row["period"])
        except (TypeError, ValueError):
            blocking.append(f"non-numeric period in row for unit {unit!r}")
            continue
        try:
            float(row["Y_observed"])
        except (TypeError, ValueError):
            bad_outcome += 1
        units[unit][period] = row
    if bad_outcome:
        blocking.append(f"{bad_outcome} rows have a non-numeric Y_observed value")

    if args.treated_unit not in units:
        blocking.append(
            f"treated unit {args.treated_unit!r} not found; units are {sorted(units)[:20]}"
        )
    else:
        treated_group_periods = sorted(
            p
            for p, row in units[args.treated_unit].items()
            if row["treatment_group"].strip().lower() == "treated"
        )
        if not treated_group_periods:
            warnings.append(
                "treated unit has no period flagged treatment_group == 'treated'; "
                "pass --treated-periods explicitly."
            )

    donors = sorted(u for u in units if u != args.treated_unit)
    if len(donors) < 2:
        blocking.append(f"synthetic control needs >= 2 donors; found {len(donors)}")

    period_sets = [set(pmap.keys()) for pmap in units.values()]
    common = set.intersection(*period_sets) if period_sets else set()
    unbalanced = [u for u, pmap in units.items() if set(pmap.keys()) != common]
    if unbalanced:
        warnings.append(
            f"{len(unbalanced)} unit(s) do not share the common period set; "
            "estimator will align to complete periods only."
        )

    mid = sorted(common)
    payload = {
        "status": "ok" if not blocking else "error",
        "panel": str(path),
        "row_count": len(rows),
        "columns": columns,
        "unit_count": len(units),
        "donor_count": len(donors),
        "treated_unit": args.treated_unit,
        "period_min": mid[0] if mid else None,
        "period_max": mid[-1] if mid else None,
        "common_period_count": len(common),
        "treated_group_periods": treated_group_periods if args.treated_unit in units else [],
        "blocking_issues": blocking,
        "warnings": warnings,
    }

    if "Z1" in columns and "Z2" in columns:
        z1_const = all(
            units[u][p]["Z1"] == units[u][sorted(units[u])[0]]["Z1"] for u in units for p in units[u]
        ) if units else True
        payload["Z1_time_varying"] = not z1_const

    emit(payload, args.out, 0 if not blocking else 2)


if __name__ == "__main__":
    main()
