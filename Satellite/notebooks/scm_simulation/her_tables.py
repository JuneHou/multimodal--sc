"""Three tables, one per section of the collaborator's checks (data/simulation/README.md Section 4.1, 4.2, 4.3). The
first rows are three references (plan use-nature-skills-analyze-wiggly-coral.md, departure 1, Jun "Both rows"):
    README Section 3        the README's own SCM, computed from its formulas by her_section3.R
    every factor known      DGP 1 only: the counterfactual delta_t + theta_t'Z_1 + lambda_t'mu_1 from the generator
                            rebuilt with the shipped seed, so its error is the treated unit's own noise; DGP 2 shares no
                            term across units, so it has no such row; no donor weights, so no 4.2 balance and no 4.3
    ADH paper specification Z1 and Z2 averaged over periods 23 to 40 and Y_observed at 12, 23 and 40, V searched over
                            1 to 40 (method.ADH_WAY), fitted with Synth through arm_checks.R
Every arm's answers follow.

Which fit stands behind an answer:
    configure, design   the fit scs_steps matches to the answer; for an answer it does not match, the recorded fit
                        whose counterfactual is closest to the answer, with that distance reported
    profile             the skill script's fixed specification (the gpt-5.4 skill: Z1, Z2 averaged over 1..40 and
                        Y_observed at each of 1..40; the DeepSeek skill: Z1, Z2 averaged over 1..40; both V searched by
                        Synth's defaults), refitted with Synth and kept only if it reproduces every answer within 0.0001
4.1 uses the answer itself; 4.2 and the leave-one-donor-out row of 4.3 use the fit (arm_checks.R); the
leave-one-period-out and block rows of 4.3 do not depend on the answer and are her values, once per DGP. Leave one
donor out runs over every donor whose weight in the fit is above 0 (her "w_j > 0"): her exact weights have 8 (DGP 1)
and 10 (DGP 2), Synth's interior-point weights are all above 0, so an arm's fit leaves out each of its donors.

    python her_tables.py     writes results/table_4_1.csv, table_4_2.csv, table_4_3.csv and prints them
"""
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sc_skills"))
import scs_method  # noqa: E402
import scs_steps   # noqa: E402

mod = scs_method.load_module("../../scm_simulation")
use = mod.USES["simulation"]
POST = [str(t) for t in range(41, 61)]
SAME = 1e-4                                   # Jun 2026-10-31 "just 4 decimal is enough" for reproduction checks
D = mod.DONORS
BASE = {"predictors_op": "mean", "time_predictors_prior": list(range(1, 41)), "time_optimize_ssr": list(range(1, 41)),
        "controls_identifier": list(D), "custom_v": None, "optimxmethod": ["Nelder-Mead", "BFGS"], "genoud": False,
        "quadopt": "ipop", "margin_ipop": 0.0005, "sigf_ipop": 5, "bound_ipop": 10,
        "weights": "convex", "interpolation_penalty": "none"}
PROFILE_SPECS = {
    "profile-gpt-5.4": {**BASE, "predictors": ["Z1", "Z2"],
                        "special_predictors": [{"variable": "Y_observed", "periods": [t], "op": "mean"} for t in range(1, 41)]},
    "profile-deepseek": {**BASE, "predictors": ["Z1", "Z2"], "special_predictors": []}}


def answers():
    for p in sorted((HERE / "runs").glob("simulation*/*.jsonl")):
        last = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            last[r["prompt_id"]] = r
        arm = p.parent.name[len("simulation_"):] if p.parent.name != "simulation" else None
        for r in last.values():
            yield (arm or r["skill"]), r


def fit_behind(arm, r):
    """-> (spec, the answer's own fit predictions or None, note)."""
    if arm in PROFILE_SPECS:
        return PROFILE_SPECS[arm], None, "skill script's specification"
    k = scs_steps.matched_run(r, use)
    if k:
        return r["fit_runs"][k - 1]["spec"], r["fit_runs"][k - 1]["predictions"], "matched fit"
    cf = r["answer"]["prediction"]["counterfactual"]
    dist = [(max(abs(cf[t] - f["predictions"][t]) for t in POST), f) for f in r["fit_runs"]]
    dmin, f = min(dist, key=lambda x: x[0])
    return f["spec"], f["predictions"], f"closest fit, {dmin:.1e} from the answer"


def run_checks(case, spec, donor=None):
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "spec.json"
        sp.write_text(json.dumps(spec))
        p = subprocess.run([str(mod.RSCRIPT), str(HERE / "arm_checks.R"), str(mod.PANELS[case]), str(sp),
                            *([donor] if donor else [])], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-600:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def observed_and_truth(case):
    panel = pd.read_csv(mod.PANELS[case])
    t = panel[panel.unit_id == "treated"].set_index("period")["Y_observed"]
    tr = pd.read_csv(mod.TRUTH)
    tr = tr[tr.dgp == case].set_index("period")
    return t, tr


def known_factor_row():
    """DGP 1 rebuilt with the shipped seed; the counterfactual is the treated unit's shared part."""
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("gen", mod.DATA / "generate_simulation_data.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    rng = np.random.default_rng(g.SEED_DGP1)
    Z = rng.uniform(-1, 1, size=(g.N_UNITS, g.N_COVARIATES))
    mu = rng.uniform(-1, 1, size=(g.N_UNITS, g.N_FACTORS))
    delta = rng.normal(size=g.T)
    theta = rng.normal(size=(g.T, g.N_COVARIATES))
    lam = rng.normal(size=(g.T, g.N_FACTORS))
    eps = rng.normal(size=(g.T, g.N_UNITS))
    shared = delta + theta @ Z[0] + lam @ mu[0]
    obs, tr = observed_and_truth("dgp1")
    assert np.allclose(shared + eps[:, 0], tr["Y0_true"].to_numpy()), "rebuild does not match ground_truth.csv"
    cf = dict(zip(range(1, 61), shared))
    post, pre = range(41, 61), range(1, 41)
    att = sum(obs[t] - cf[t] for t in post) / 20
    true_att = tr.loc[list(post), "tau_true"].mean()
    return {"arm": "every factor known", "model": "", "case": "dgp1", "answered": True, "note": "generator, shipped seed",
            "cf_rmse": math.sqrt(sum((cf[t] - tr.loc[t, "Y0_true"]) ** 2 for t in post) / 20),
            "att_error": att - true_att, "rel_att_error": abs(att - true_att) / abs(true_att),
            "pre_rmspe": math.sqrt(sum((obs[t] - cf[t]) ** 2 for t in pre) / 40)}


def adh_spec():
    s = dict(mod.ADH_WAY)
    s["time_predictors_prior"] = [int(x) for x in s["time_predictors_prior"]]
    s["time_optimize_ssr"] = [int(x) for x in s["time_optimize_ssr"]]
    return s


def adh_row(case, fit):
    obs, tr = observed_and_truth(case)
    cf = {int(t): fit["predictions"][t] for t in POST}
    att = sum(obs[t] - cf[t] for t in cf) / 20
    true_att = tr.loc[list(cf), "tau_true"].mean()
    return {"arm": "ADH paper specification", "model": "", "case": case, "answered": True, "note": "arm_checks.R",
            "cf_rmse": math.sqrt(sum((cf[t] - tr.loc[t, "Y0_true"]) ** 2 for t in cf) / 20),
            "att_error": att - true_att, "rel_att_error": abs(att - true_att) / abs(true_att),
            "pre_rmspe": fit["pre_rmspe"], "covariate_balance": fit["covariate_balance"],
            "lodo_max_abs_att_change": fit["lodo_max_abs_att_change"], "lodo_donor": fit["lodo_donor"],
            "lodo_donor_weight": fit["lodo_donor_weight"], "n_positive": fit["n_positive"]}


def main():
    sc = pd.read_csv(HERE / "results" / "scores.csv")
    her = pd.read_csv(HERE / "results" / "her_section3_errors.csv").set_index("dgp")
    rows, jobs = [], {}
    for arm, r in answers():
        row = {"arm": arm, "model": r["model"], "case": r["case"], "answered": bool(r["ok"])}
        if r["ok"]:
            spec, own, note = fit_behind(arm, r)
            key = (r["case"], json.dumps(spec, sort_keys=True))
            jobs[key] = spec
            row.update(key=key, own=own, note=note, cf=r["answer"]["prediction"]["counterfactual"])
            s = sc[(sc.arm == arm) & (sc.model == r["model"]) & (sc.case == r["case"])].iloc[0]
            row.update(cf_rmse=s.cf_rmse, att_error=s.att_error, rel_att_error=s.rel_att_error)
        rows.append(row)
    adh_keys = [(c, json.dumps(adh_spec(), sort_keys=True)) for c in ("dgp1", "dgp2")]
    for k in adh_keys:
        jobs[k] = adh_spec()
    with ThreadPoolExecutor(64) as ex:
        res = dict(zip(jobs, ex.map(lambda k: run_checks(k[0], jobs[k]), jobs)))
        drops = [(k, j) for k in jobs for j, wj in res[k]["weights"].items() if wj > 0]
        att_j = dict(zip(drops, ex.map(lambda kj: run_checks(kj[0][0], jobs[kj[0]], kj[1])["att"], drops)))
    for k in jobs:
        ch = {j: abs(att_j[(k, j)] - res[k]["att"]) for (kk, j) in drops if kk == k}
        j = max(ch, key=ch.get)
        res[k].update(lodo_max_abs_att_change=ch[j], lodo_donor=j, lodo_donor_weight=res[k]["weights"][j],
                      n_positive=len(ch))
    bad = []
    for row in rows:
        if not row["answered"]:
            continue
        c = res[row["key"]]
        ref = row["own"] if row["own"] is not None else row["cf"]
        d = max(abs(c["predictions"][t] - ref[t]) for t in POST)
        if d > SAME:
            bad.append((row["arm"], row["model"], row["case"], d))
        row.update(refit_vs_fit=d, pre_rmspe=c["pre_rmspe"], covariate_balance=c["covariate_balance"],
                   lodo_max_abs_att_change=c["lodo_max_abs_att_change"], lodo_donor=c["lodo_donor"],
                   lodo_donor_weight=c["lodo_donor_weight"], n_positive=c["n_positive"])
    print(f"{len(jobs)} distinct specifications refitted; refits that do not reproduce their fit within {SAME}: {bad}")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("key", "own", "cf")} for r in rows])
    her_rows = [{"arm": "README Section 3", "model": "", "case": g, "answered": True, "note": "her_section3.R",
                 **{k: her.loc[g, k] for k in ("cf_rmse", "att_error", "rel_att_error", "pre_rmspe",
                                               "covariate_balance", "lodo_max_abs_att_change", "lodo_donor",
                                               "lodo_donor_weight")}, "n_positive": her.loc[g, "donors_positive"]}
                for g in ("dgp1", "dgp2")]
    refs = her_rows + [known_factor_row()] + [adh_row(k[0], res[k]) for k in adh_keys]
    df = pd.concat([pd.DataFrame(refs), df], ignore_index=True)
    keys = ["arm", "model", "case", "answered", "note"]
    tables = {"4_1": keys + ["cf_rmse", "att_error", "rel_att_error"],
              "4_2": keys + ["pre_rmspe", "covariate_balance"],
              "4_3": keys + ["lodo_max_abs_att_change", "lodo_donor", "lodo_donor_weight", "n_positive"]}
    for name, cols in tables.items():
        t = df[cols]
        t.to_csv(HERE / "results" / f"table_{name}.csv", index=False)
        print(f"\n== table {name}\n" + t.to_string(index=False))
    print("\n4.3 rows that do not depend on the answer (her values):",
          {g: {"cv_rmse": her.loc[g, "cv_rmse"], "block_36_40_rmse": her.loc[g, "block_36_40_rmse"]} for g in her.index})
    print("refit vs the answer's own fit, max over answers:", df["refit_vs_fit"].max())


if __name__ == "__main__":
    main()
