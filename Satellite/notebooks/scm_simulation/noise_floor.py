"""Random noise of every answer, the row of the README Section 5 evaluation tables, at one noise level.

README Section 5: "The noise term is computed per replication from the realised weights", sigma * sqrt(1 + sum_j w_j^2)
(mc_evaluation_r.R line 118). The weights are those of the fit behind each answer, as in Section 4.2 of
her_tables.py: the recorded fit of a configure or design answer, and the Synth fit of the skill script's specification
for a profile answer. Each fit's pre-treatment RMSPE is checked against results/sigma_<s>/table_4_2.csv, so the weights
are the ones behind the tables. No model is called.

    python noise_floor.py --sigma 1.0   writes results/sigma_1.0/noise_floor.csv
"""
import argparse
import math

import pandas as pd

import her_tables as h


def main(sigma):
    use = h.mod.USES[h.mod.use_name(sigma)]
    out = h.mod.results_folder(sigma)
    t42 = pd.read_csv(out / "table_4_2.csv")
    s = float(sigma)
    profile_fits, rows = {}, []
    for arm, r in h.answers(sigma):
        if not r["ok"]:
            continue
        fit, note = h.fit_record(arm, r, use)
        if h.no_fit_gives(r, fit, note):            # no fit behind the answer, so no weights: left out
            print(f"{arm} {r['model']} {r['case']}: no recorded fit gives this counterfactual, no random noise")
            continue
        if fit is None:
            key = (arm, r["case"])
            if key not in profile_fits:
                profile_fits[key] = h.run_checks(r["case"], h.PROFILE_SPECS[arm], sigma, label=f"{arm} {r['case']}")
            c = profile_fits[key]
        else:
            c = h.recorded(fit, r["case"], sigma)
        want = t42[(t42.arm == arm) & (t42.model == r["model"]) & (t42.case == r["case"])].pre_rmspe.iloc[0]
        assert abs(c["pre_rmspe"] - want) <= h.SAME, (arm, r["model"], r["case"], c["pre_rmspe"], want)
        ss = sum(w * w for w in c["weights"].values())
        rows.append({"arm": arm, "model": r["model"], "case": r["case"], "sum_sq_weights": ss,
                     "noise_floor": s * math.sqrt(1 + ss)})
    pd.DataFrame(rows).to_csv(out / "noise_floor.csv", index=False)
    print(f"sigma {sigma}: {len(rows)} answers, pre-treatment RMSPE of every fit equal to table 4.2 within {h.SAME}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", default="1.0", choices=h.mod.SIGMAS)
    main(ap.parse_args().sigma)
