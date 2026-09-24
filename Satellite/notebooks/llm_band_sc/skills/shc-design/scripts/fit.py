"""Run the synthetic historical control on this parcel's nine months under the given file.
    python scripts/fit.py spec.json
Prints the blocks kept, the weights, the matching error over P01-P09, and the prediction; appends a record of the
run to .fit_log.jsonl. The fit sees P01-P09 only."""
import json
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb
import lb_her_panel
import lb_spec
import lb_fit



def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/fit.py spec.json")
    try:
        given = json.load(open(sys.argv[1]))
    except FileNotFoundError:
        raise SystemExit(f"{sys.argv[1]} not found; write it first")
    except json.JSONDecodeError as e:
        raise SystemExit(f"{sys.argv[1]} is not valid JSON: {e}")
    w, sensor = lb.wide_from_data()
    panel = lb_her_panel.panel_from_wide(w)
    try:
        spec = lb_spec.validate_spec(given, sensor)
    except lb_spec.SpecError as e:
        raise SystemExit(f"refused: {e}")
    try:
        out = lb_fit.run_cell(panel, sensor, "site", spec)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"the fit could not run: {e}")
    blocks = lb_fit.her.make_historical_blocks_from_periods(lb_fit.TRAIN, m=spec["m"], n=spec["n"])
    by_id = {b["donor_id"]: b for b in blocks}
    kept = [f"block {i} ({by_id[i]['pre_periods'][0]}-{by_id[i]['pre_periods'][-1]} -> {by_id[i]['post_periods'][0]})" for i in out["donor_ids"]]
    d = lb.DECIMALS[sensor]
    print(f"blocks available: {out['n_blocks']} (m = {spec['m']}, n = {spec['n']})")
    print("blocks kept: " + "; ".join(kept))
    print("weights: " + ", ".join(f"{w_:.4f}" for w_ in out["weights"]) + (f"; intercept {out['intercept']:.4f}" if spec["intercept"] == "fitted" else ""))
    print(f"matching error (mean squared error over the matching window, P01-P09 only): {out['fit_mse']:.6f}")
    print("prediction for P%02d = %s: " % (lb.TARGET, lb.CALENDAR[lb.TARGET]) + ", ".join(f"{b} {v:.{d}f}" for b, v in out["predictions"].items()))
    rec = {"file": sys.argv[1], "spec": spec, "donor_ids": out["donor_ids"], "weights": [float(x) for x in out["weights"]],
           "intercept": out["intercept"], "fit_mse": out["fit_mse"], "predictions": out["predictions"],
           "engine_used": out["engine_used"], "zeta_used": out["zeta_used"], "near_ties": out["near_ties"]}
    with open(".fit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("@@FIT@@" + json.dumps(rec))          # captured by the harness and removed from what the model sees


if __name__ == "__main__":
    main()
