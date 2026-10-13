#!/usr/bin/env python3
"""Fit the synthetic historical control to one land parcel's nine months under a specification and print the
forecast for the next month as a JSON summary.

Run from the skill directory:
    python scripts/fit.py --spec spec.json
    python scripts/fit.py --data examples/data.csv --spec examples/spec.json --out example_fit.json

The data file is checked as scripts/show_data.py checks it (nine periods P01 to P09, one row per band per month)
and must be named data.csv, because the bundled fitting modules read data.csv in the folder the fit runs in.
The specification is a JSON object with every field listed in SKILL.md; there are no defaults. The fit reads
P01 to P09 only. Every run appends one JSON line to the log file.

Exit status 0 on success; 1 when the data or the specification is refused or the fit cannot run, with the
reason on stderr; 2 on a usage error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from show_data import DataError, read_parcel  # noqa: E402

FIT_MARK = "@@FIT@@"


def fail(message):
    sys.exit(f"fit: {message}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data.csv"), metavar="PATH",
                    help="path (string) of the parcel's data file, named data.csv; default: data.csv in the current directory")
    ap.add_argument("--spec", type=Path, required=True, metavar="PATH",
                    help="path (string) of the specification JSON file, for example spec.json; required")
    ap.add_argument("--out", type=Path, default=None, metavar="PATH",
                    help="path (string) of a file to write the JSON summary to as well; default: none written")
    ap.add_argument("--log", type=Path, default=Path(".fit_log.jsonl"), metavar="PATH",
                    help="path (string) of the run log, one JSON line appended per run; default: .fit_log.jsonl in the current directory")
    a = ap.parse_args(argv)
    data, spec_path, log = a.data.resolve(), a.spec.resolve(), a.log.resolve()
    out = a.out.resolve() if a.out is not None else None

    try:
        sensor, bands, rows, target = read_parcel(a.data)
    except DataError as e:
        fail(str(e))
    if data.name != "data.csv":
        fail(f"{a.data}: the data file must be named data.csv, because the bundled fitting modules read data.csv "
             "in the folder the fit runs in")
    try:
        given = json.loads(spec_path.read_text())
    except FileNotFoundError:
        fail(f"{a.spec} not found; write it first")
    except json.JSONDecodeError as e:
        fail(f"{a.spec} is not valid JSON: {e}")

    os.chdir(data.parent)                       # the bundled modules read ./data.csv when they are imported
    import lb_lib as lb
    import lb_her_panel
    import lb_spec
    import lb_fit

    w, sensor = lb.wide_from_data()
    panel = lb_her_panel.panel_from_wide(w)
    try:
        spec = lb_spec.validate_spec(given, sensor)
    except lb_spec.SpecError as e:
        fail(f"refused: {e}")
    try:
        res = lb_fit.run_cell(panel, sensor, "site", spec)
    except (RuntimeError, ValueError) as e:
        fail(f"the fit could not run: {e}")

    blocks = {b["donor_id"]: b for b in lb_fit.her.make_historical_blocks_from_periods(lb_fit.TRAIN, m=spec["m"], n=spec["n"])}
    summary = {
        "data": str(a.data),
        "spec": str(a.spec),
        "m": spec["m"],
        "n": spec["n"],
        "matching_window": f"{lb_fit.TRAIN[-spec['m']]}-{lb_fit.TRAIN[-1]}",
        "blocks_available": res["n_blocks"],
        "blocks_kept": [{"block": i, "matching_months": f"{blocks[i]['pre_periods'][0]}-{blocks[i]['pre_periods'][-1]}",
                         "forward_month": blocks[i]["post_periods"][0], "weight": round(float(wt), 6)}
                        for i, wt in zip(res["donor_ids"], res["weights"])],
    }
    if spec["intercept"] == "fitted":
        summary["intercept"] = round(float(res["intercept"]), 6)
    summary["matching_mse"] = round(float(res["fit_mse"]), 6)
    summary["prediction"] = {"period_id": f"P{lb.TARGET:02d}", "month": lb.CALENDAR[lb.TARGET],
                             "values": {b: round(float(v), 6) for b, v in res["predictions"].items()}}
    text = json.dumps(summary, indent=2)

    record = {"file": str(a.spec), "data": str(a.data), "spec": spec, "donor_ids": res["donor_ids"],
              "weights": [float(x) for x in res["weights"]], "intercept": res["intercept"], "fit_mse": res["fit_mse"],
              "predictions": res["predictions"], "engine_used": res["engine_used"], "zeta_used": res["zeta_used"],
              "near_ties": res["near_ties"]}
    try:
        with open(log, "a") as f:
            f.write(json.dumps(record) + "\n")
        if out is not None:
            out.write_text(text + "\n")
    except OSError as e:
        fail(f"could not write {e.filename}: {e.strerror}")
    print(text)
    print(FIT_MARK + json.dumps(record))        # captured by the harness and removed from what the model sees
    return 0


if __name__ == "__main__":
    sys.exit(main())
