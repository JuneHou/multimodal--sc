"""Score every answer against the ground truth of the simulated data (README Section 4.1 of data/simulation/).

For each record in the record folders of one noise level (method.record_folders: scm-configure/sigma_<s>/,
scm-design/sigma_<s>/, skill_profile/<generator>/sigma_<s>/; the last record of each case in a file), with the true
untreated path Y0_true and the true per-period effect tau_true of that case's DGP from that level's ground_truth.csv,
over periods 41 to 60:

    cf_rmse            sqrt(mean over 41..60 of (answer counterfactual - Y0_true)^2)
    att_error          ATT - mean over 41..60 of tau_true, where ATT is the mean over 41..60 of the treated unit's
                       observed outcome minus the answer's counterfactual, as the collaborator's code computes it
                       (data/simulation/mc_evaluation_r.R line 94; Jun "there should be no difference between her
                       scoring and what we use", 2026-11-06). The att number the model wrote is not scored;
                       report/att_check.py compares it with this ATT.
    rel_att_error      |att_error| / |mean over 41..60 of tau_true|

A record without an answer is kept as a row with its ending and no scores. No reference way is a row.

    python score.py [--sigma 1.0]             score the records of one noise level (1.0, 0.5 or 0.2), write
                                              results/sigma_<s>/scores.csv and print the table
    python score.py [--sigma 1.0] --selftest  a scripted answer equal to Y0_true scores 0 on all three; one shifted
                                              by 1 in every period scores a counterfactual RMSE of 1
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


def truth(sigma="1.0"):
    g = pd.read_csv(mod.TRUTH_BY_SIGMA[sigma])
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


def records(sigma="1.0"):
    """(arm, file stem, last record of each case) over the record folders of one noise level."""
    for arm, folder in mod.record_folders(sigma).items():
        for p in sorted(folder.glob("*.jsonl")):
            last = {}
            for line in p.read_text().splitlines():
                r = json.loads(line)
                last[r["prompt_id"]] = r
            for r in last.values():
                yield arm, p.stem, r


def counterfactual_att(case, counterfactual, sigma="1.0"):
    """The ATT of a counterfactual: mean over 41..60 of the observed outcome minus it (mc_evaluation_r.R line 94)."""
    observed = mod._observed_post({"files": mod._case_files(case, sigma)})
    return sum(observed[p] - counterfactual[p] for p in POST) / len(POST)


def table(sigma="1.0"):
    t, rows = truth(sigma), []
    for arm, stem, r in records(sigma):
        row = {"arm": arm, "model": r["model"], "case": r["case"], "ok": bool(r["ok"]),
               "ended": r.get("ended"), "n_steps": r.get("n_steps"), "n_fit_runs": len(r.get("fit_runs") or [])}
        if r["ok"]:
            cf = r["answer"]["prediction"]["counterfactual"]
            row.update(score(r["case"], cf, counterfactual_att(r["case"], cf, sigma), t))
        rows.append(row)
    return pd.DataFrame(rows)


def selftest(sigma="1.0"):
    t, bad = truth(sigma), []
    for case in t:
        y0 = t[case]["y0"]
        observed = mod._observed_post({"files": mod._case_files(case, sigma)})
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


def sigma_arg(argv):
    s = argv[argv.index("--sigma") + 1] if "--sigma" in argv else "1.0"
    if s not in mod.SIGMAS:
        raise SystemExit(f"--sigma {s}: one of {', '.join(mod.SIGMAS)}")
    return s


if __name__ == "__main__":
    sig = sigma_arg(sys.argv)
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest(sig) else 1)
    df = table(sig)
    out = mod.results_folder(sig)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)
    print(df.to_string(index=False) if len(df) else f"no records for sigma {sig}")
