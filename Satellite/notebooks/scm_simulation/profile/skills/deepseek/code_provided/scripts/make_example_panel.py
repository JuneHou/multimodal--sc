#!/usr/bin/env python3
"""Generate a deterministic synthetic-control panel with a known treatment effect.

Builds a balanced panel of one treated unit plus donor units across a range of
periods using the canonical synthetic-control data-generating process: the
treated unit's pre-treatment outcome is a fixed convex combination of a few
donor units' outcome series, so a well-behaved synthetic control can reproduce
it closely. After the intervention period the treated unit receives a constant
additive treatment effect, making the ground-truth ATT recoverable.

Columns written: ``unit_id``, ``period``, ``treatment_group``, ``Z1``, ``Z2``,
``Y_observed`` (one row per unit and period), matching the estimator's input
contract. ``treatment_group`` is a unit-level label (``treated``/``donor``); the
estimator receives the treated periods explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 make_example_panel.py --out ./tobacco_panel.csv --treated-effect -8\n"
        ),
    )
    parser.add_argument("--out", required=True, help="Destination CSV path for the panel.")
    parser.add_argument("--donors", type=int, default=15, help="Number of donor units (default 15).")
    parser.add_argument("--period-start", type=int, default=1980, help="First period (default 1980).")
    parser.add_argument("--period-end", type=int, default=2000, help="Last period (default 2000).")
    parser.add_argument(
        "--first-treated-period", type=int, default=1989, help="First treated period (default 1989)."
    )
    parser.add_argument(
        "--treated-effect",
        type=float,
        default=-8.0,
        help="Constant additive effect applied to the treated unit post-treatment (default -8).",
    )
    parser.add_argument(
        "--n-sponsors",
        type=int,
        default=4,
        help="Donors combined (with positive weights) to build the treated unit series (default 4).",
    )
    parser.add_argument("--noise-sd", type=float, default=0.25, help="Outcome noise sd (default 0.25).")
    parser.add_argument("--seed", type=int, default=20240531, help="RNG seed (default 20240531).")
    parser.add_argument("--treated-unit", default="treated", help="unit_id for the treated unit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.donors < 2:
        print("ERROR: --donors must be >= 2 for a synthetic control.", file=sys.stderr)
        raise SystemExit(2)
    if args.first_treated_period <= args.period_start:
        print("ERROR: --first-treated-period must be after --period-start.", file=sys.stderr)
        raise SystemExit(2)
    if not 1 <= args.n_sponsors <= args.donors:
        print("ERROR: --n-sponsors must be between 1 and --donors.", file=sys.stderr)
        raise SystemExit(2)

    rng = random.Random(args.seed)
    periods = list(range(args.period_start, args.period_end + 1))
    donors = [f"donor{i:02d}" for i in range(1, args.donors + 1)]
    units = [args.treated_unit] + donors

    # Shared latent factor path (a common macro trend all units load onto).
    factor = {p: rng.uniform(-1.0, 1.0) for p in periods}

    donor_series: dict[str, list[float]] = {}
    donor_z: dict[str, tuple[float, float]] = {}
    for unit in donors:
        intercept = rng.uniform(10.0, 60.0)
        slope = rng.uniform(0.0, 0.6)
        load = rng.uniform(0.5, 2.0)
        z1 = round(rng.uniform(1.0, 5.0), 3)
        z2 = round(rng.uniform(10.0, 20.0), 3)
        donor_z[unit] = (z1, z2)
        series = [
            intercept
            + slope * (p - args.period_start)
            + load * factor[p]
            + 0.3 * z1
            + 0.1 * z2
            + rng.gauss(0.0, args.noise_sd)
            for p in periods
        ]
        donor_series[unit] = series

    # Fixed convex weights over a small sponsor set for the treated unit.
    sponsors = rng.sample(donors, args.n_sponsors)
    raw = [rng.uniform(0.5, 2.0) for _ in sponsors]
    total = sum(raw)
    weights = [r / total for r in raw]

    treated_series = [
        sum(w * donor_series[u][i] for w, u in zip(weights, sponsors)) + rng.gauss(0.0, args.noise_sd)
        for i in range(len(periods))
    ]
    treated_z = (
        round(sum(w * donor_z[u][0] for w, u in zip(weights, sponsors)), 3),
        round(sum(w * donor_z[u][1] for w, u in zip(weights, sponsors)), 3),
    )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit_id", "period", "treatment_group", "Z1", "Z2", "Y_observed"])

        for unit in units:
            for i, period in enumerate(periods):
                if unit == args.treated_unit:
                    z1, z2 = treated_z
                    y = treated_series[i]
                    if period >= args.first_treated_period:
                        y += args.treated_effect
                    group = "treated"
                else:
                    z1, z2 = donor_z[unit]
                    y = donor_series[unit][i]
                    group = "donor"
                writer.writerow([unit, period, group, z1, z2, round(y, 4)])
                rows_written += 1

    payload = {
        "status": "ok",
        "panel": str(path),
        "rows_written": rows_written,
        "unit_count": len(units),
        "donor_count": len(donors),
        "period_start": args.period_start,
        "period_end": args.period_end,
        "first_treated_period": args.first_treated_period,
        "treated_effect": args.treated_effect,
        "sponsors": sponsors,
        "sponsor_weights": [round(w, 4) for w in weights],
        "seed": args.seed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
