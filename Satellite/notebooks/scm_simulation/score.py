"""Score every answer against the ground truth of the simulated data (README Section 4.1 of data/simulation/).

For each record of runs/simulation*/*.jsonl (the last record of each case in a file), with the true untreated path
Y0_true and the true per-period effect tau_true of that case's DGP from ground_truth.csv, over periods 41 to 60:

    cf_rmse            sqrt(mean over 41..60 of (answer counterfactual - Y0_true)^2)
    att_error          answer att - mean over 41..60 of tau_true
    rel_att_error      |att_error| / |mean over 41..60 of tau_true|

A record without an answer is kept as a row with its ending and no scores. No reference way is a row.

    python score.py              score the records, write results/scores.csv and print the table
    python score.py --selftest   a scripted answer equal to Y0_true scores 0 on all three; one shifted by 1 in every
                                 period scores a counterfactual RMSE of 1
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sc_skills"))
import scs_method  # noqa: E402

mod = scs_method.load_module("../../scm_simulation")
POST = [str(t) for t in range(mod.FIRST_TREATED, 61)]


def truth():
    g = pd.read_csv(mod.TRUTH)
    out = {}
    for dgp, d in g.groupby("dgp"):
        d = d.set_index(d.period.astype(int).astype(str))
        out[dgp] = {"y0": {p: float(d.loc[p, "Y0_true"]) for p in POST},
                    "att": float(d.loc[POST, "tau_true"].mean())}
    return out


def score(case, counterfactual, att, t):
    tr = t[case]
    cf_rmse = math.sqrt(sum((counterfactual[p] - tr["y0"][p]) ** 2 for p in POST) / len(POST))
    err = att - tr["att"]
    return {"cf_rmse": cf_rmse, "att_error": err, "rel_att_error": abs(err) / abs(tr["att"])}


def arm(folder, skill):
    """runs/simulation/ holds scm-configure and scm-design; runs/simulation_profile-<generator>/ one profile skill."""
    return skill if folder == "simulation" else folder[len("simulation_"):]


def records():
    for p in sorted((HERE / "runs").glob("simulation*/*.jsonl")):
        last = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            last[r["prompt_id"]] = r
        for r in last.values():
            yield p.parent.name, p.stem, r


def table():
    t, rows = truth(), []
    for folder, stem, r in records():
        row = {"arm": arm(folder, r["skill"]), "model": r["model"], "case": r["case"], "ok": bool(r["ok"]),
               "ended": r.get("ended"), "n_steps": r.get("n_steps"), "n_fit_runs": len(r.get("fit_runs") or [])}
        if r["ok"]:
            a = r["answer"]
            row.update(score(r["case"], a["prediction"]["counterfactual"], a["att"], t))
        rows.append(row)
    return pd.DataFrame(rows)


def selftest():
    t, bad = truth(), []
    for case in t:
        y0 = t[case]["y0"]
        observed = mod._observed_post({"files": mod._case_files(case)})
        for shift, want in ((0.0, 0.0), (1.0, 1.0)):
            cf = {p: y0[p] + shift for p in POST}
            att = sum(observed[p] - cf[p] for p in POST) / len(POST)
            s = score(case, cf, att, t)
            ok = math.isclose(s["cf_rmse"], want, abs_tol=1e-12) and (shift or abs(s["att_error"]) < 1e-12)
            print(f"{case} shift {shift}: cf_rmse {s['cf_rmse']:.12f}, att_error {s['att_error']:.12f}, "
                  f"rel_att_error {s['rel_att_error']:.12f}")
            if not ok:
                bad.append((case, shift))
    print("selftest:", "pass" if not bad else f"FAIL {bad}")
    return not bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    df = table()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)
    print(df.to_string(index=False) if len(df) else "no records under runs/simulation*/")
