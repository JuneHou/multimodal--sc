"""Run the synthetic historical control on this parcel's visits given under the given file.
    python scripts/fit.py spec.json
Prints the matching window length chosen, the blocks kept, the weights, the matching error over the visits given, and
the prediction at every visit to predict; appends a record of the run to .fit_log.jsonl. The fit sees the visits
given only."""
import json
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb
import lb_spec_daily
import lb_fit_daily



def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/fit.py spec.json")
    try:
        given = json.load(open(sys.argv[1]))
    except FileNotFoundError:
        raise SystemExit(f"{sys.argv[1]} not found; write it first")
    except json.JSONDecodeError as e:
        raise SystemExit(f"{sys.argv[1]} is not valid JSON: {e}")
    d, sensor = lb.given()
    tp = lb.to_predict()
    pid_of = dict(zip(d["date"], d["period_id"]))
    pre = d[["band", "date", "y"]]
    n_given = d["date"].nunique()
    try:
        spec = lb_spec_daily.validate_spec(given, sensor, n_given, tp)
    except lb_spec_daily.SpecError as e:
        raise SystemExit(f"refused: {e}")
    try:
        out = lb_fit_daily.run_case(pre, tp, sensor, spec)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"the fit could not run: {e}")
    dates = sorted(d["date"].unique())
    pid = [pid_of[x] for x in dates]
    m, n = out["m_star"], len(tp)
    kept = [f"block {i} ({pid[s]}-{pid[s + m - 1]} -> {pid[s + m]}-{pid[s + m + n - 1]})"
            for i, s in zip(out["donor_ids"], out["block_starts"])]
    rule = ("the longest m tried with R <= nu" if out["selection_mode"] == "paper_rule_stacked"
            else "no m tried had R <= nu; the longest m tried")
    dec = lb.DECIMALS[sensor]
    print(f"visits given: {n_given}; visits to predict: {n}")
    print("bandwidths chosen, days: " + ", ".join(f"{lb.disp(b)} {h:g}" for b, h in out["bandwidths"].items()))
    print(f"matching window length: m = {m} ({rule}); blocks available: {out['n_blocks']}")
    print("blocks kept: " + "; ".join(kept))
    print("weights: " + ", ".join(f"{w_:.4f}" for w_ in out["weights"]) + (f"; intercept {out['intercept']:.4f}" if spec["intercept"] == "fitted" else ""))
    print(f"matching error over the matching window, visits given only: R = {out['R']:.6f}, mean squared error = {out['fit_mse']:.6f}")
    print("prediction:")
    for p, day in tp:
        print(f"{p} = {day}: " + ", ".join(f"{b} {v[p]:.{dec}f}" for b, v in out["predictions"].items()))
    rec = {"file": sys.argv[1], "spec": spec, "m_star": m, "selection_mode": out["selection_mode"],
           "donor_ids": out["donor_ids"], "weights": [float(x) for x in out["weights"]], "intercept": out["intercept"],
           "R": out["R"], "fit_mse": out["fit_mse"], "bandwidths": out["bandwidths"], "predictions": out["predictions"],
           "engine_used": out["engine_used"], "near_ties": out["near_ties"]}
    with open(".fit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("@@FIT@@" + json.dumps(rec))          # captured by the harness and removed from what the model sees


if __name__ == "__main__":
    main()
