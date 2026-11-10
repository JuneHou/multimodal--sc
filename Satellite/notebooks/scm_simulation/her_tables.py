"""Three tables, one per section of the collaborator's checks (data/simulation/README.md Section 4.1, 4.2, 4.3), at one
noise level (outcome shock SD 1.0, 0.5 or 0.2, data/simulation/simulation_data/sigma_<s>/). The first rows are two
references (plan use-nature-skills-analyze-wiggly-coral.md: the slide-deck plan's departure 1, Jun "Both rows",
2026-11-01; the noise-level plan, Jun "Read her result files" and "drop all, b", 2026-11-04, which dropped the ADH
paper specification row):
    README Section 3        replication 1 of the collaborator's Monte Carlo files,
                            data/simulation/mc_results_r/sigma_<s>/mc_replications_dgp{1,2}.csv, read as they are;
                            replication 1 is the shipped panel of that level (generate_simulation_data.py, line 167).
                            Her files hold her leave-one-period-out and block values, so this row has them; they do
                            not name the donor of the largest leave-one-donor-out change, so those two cells are empty
    every factor known      DGP 1 only: the counterfactual delta_t + theta_t'Z_1 + lambda_t'mu_1 from the generator
                            rebuilt with the shipped seed, whose noise the generator multiplies by sigma, so its error
                            is the treated unit's own noise; DGP 2 shares no term across units, so it has no such row;
                            no donor weights, so no 4.2 balance and no 4.3
Every arm's answers follow.

Which fit stands behind an answer:
    configure, design   the fit scs_steps matches to the answer; for an answer it does not match, the recorded fit
                        whose counterfactual is closest to the answer, with that distance reported
    profile             the skill script's fixed specification (the gpt-5.4 skill: Z1, Z2 averaged over 1..40 and
                        Y_observed at each of 1..40; the DeepSeek skill: Z1, Z2 averaged over 1..40; both V searched by
                        Synth's defaults), refitted with Synth and kept only if it reproduces every answer within 0.0001
4.1 uses the answer itself. 4.2 and the starting ATT of leave one donor out use the fit behind the answer, as the
collaborator's code takes them from its main fit (mc_evaluation_r.R lines 90 to 104; Jun "please do this", 2026-11-04):
for a configure or design answer its recorded fit, the weights and gaps the box recorded, with no second fit; for a
profile answer, whose fit is not recorded, the Synth fit of the script's specification (arm_checks.R). Each
leave-one-donor-out refit runs through arm_checks.R. Leave one donor out runs over every donor whose weight in the fit
is above 0.001, and n_positive counts those donors, as the collaborator's
code counts donors (data/simulation/mc_evaluation_r.R, lines 105 and 148; Jun "what is used in her method of secton 3
corresponding code, we follow it", 2026-11-04). Leave one period out and the block check refit each answer's own
specification, V searched again with its own optimizers where it was searched (the collaborator's option A; Jun "yes,
build option A.", 2026-11-04):
    leave one period out  for each s in 1..40 the outcome at s is removed wherever it enters (a single-period outcome
                          predictor at s dropped, an outcome average covering s recomputed without s, an outcome
                          summarized through `predictors` turned into the same average as a special predictor), s is
                          removed from time_optimize_ssr, the covariates stay, and the refit's prediction at s is
                          scored: CV-RMSE = sqrt(mean over s of (Y_1s - prediction)^2)
    block                 every predictor limited to periods 1..35 (a covariate whose periods all fall after 35 taken
                          at 35, an outcome predictor whose periods all fall after 35 dropped), time_optimize_ssr
                          limited to 1..35, and the refit's predictions at 36..40 scored by their RMSE
A specification rewritten with nothing removed (an outcome in `predictors` moved to a special predictor) must reproduce
the row's own fit within 0.0001. The README Section 3 row keeps the collaborator's own values.
A refit that fails stops only itself: each check uses the refits that worked, and failed_refits in table 4.3 names the
others (Jun "yes, stop for taht datapoint", 2026-11-05).

scm-four-design answers (plan use-nature-skills-analyze-wiggly-coral.md, top section) are refitted the same way: their
recorded spec holds Synth's arguments with outcome_combinations, each of which loses its weight on a period left out, and
arm_checks.R fits every spec as the fit behind the answer ran, Synth or the native engine.

    python her_tables.py [--sigma 1.0]   writes results/sigma_<s>/table_4_1.csv, table_4_2.csv, table_4_3.csv
    python her_tables.py --sigma <s> --arms scm-four-design
                                         computes only the named arms; every other row of the tables stays as it is
    python her_tables.py --sigma <s> --scores-only
                                         table 4.1's three scores of every answer copied again from scores.csv
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
POST = [str(t) for t in range(41, 61)]
SAME = 1e-4                                   # Jun 2026-10-31 "just 4 decimal is enough" for reproduction checks
DONOR_MIN = 1e-3                              # mc_evaluation_r.R lines 105 and 148: donors with weight above 1e-3
D = mod.DONORS
BASE = {"predictors_op": "mean", "time_predictors_prior": list(range(1, 41)), "time_optimize_ssr": list(range(1, 41)),
        "controls_identifier": list(D), "custom_v": None, "optimxmethod": ["Nelder-Mead", "BFGS"], "genoud": False,
        "quadopt": "ipop", "margin_ipop": 0.0005, "sigf_ipop": 5, "bound_ipop": 10,
        "weights": "convex", "interpolation_penalty": "none"}
PROFILE_SPECS = {
    "profile-gpt-5.4": {**BASE, "predictors": ["Z1", "Z2"],
                        "special_predictors": [{"variable": "Y_observed", "periods": [t], "op": "mean"} for t in range(1, 41)]},
    "profile-deepseek": {**BASE, "predictors": ["Z1", "Z2"], "special_predictors": []}}
README_ROW_FILES = mod.DATA / "mc_results_r"
PRE = list(range(1, 41))
BLOCK, BLOCK_LAST_TRAIN = list(range(36, 41)), 35


def as_special(spec):
    """The same specification, with an outcome summarized through `predictors` written as a special predictor. A fixed
    V (custom_v) moves that predictor's entry with it: Synth orders the predictors as `predictors`, then the special
    predictors, then (scm-four-design) the outcome combinations."""
    s = json.loads(json.dumps(spec))
    if "Y_observed" in s["predictors"]:
        if s.get("custom_v") is not None:
            v, i = s["custom_v"], s["predictors"].index("Y_observed")
            n = len(s["predictors"]) + len(s["special_predictors"])
            s["custom_v"] = v[:i] + v[i + 1:n] + [v[i]] + v[n:]
        s["predictors"] = [v for v in s["predictors"] if v != "Y_observed"]
        s["special_predictors"] = s["special_predictors"] + [
            {"variable": "Y_observed", "periods": [int(t) for t in s["time_predictors_prior"]], "op": s["predictors_op"]}]
    return s


def _drop(s, kept_special, kept_combinations):
    """Keep the special predictors and outcome combinations at the given positions, and a fixed V's entries of the
    predictors that stay (the refit removes only the held-out information, README Section 4.3)."""
    n0, n1 = len(s["predictors"]), len(s["special_predictors"])
    if s.get("custom_v") is not None:
        v = s["custom_v"]
        s["custom_v"] = (v[:n0] + [v[n0 + j] for j, _ in kept_special]
                         + [v[n0 + n1 + m] for m, _ in kept_combinations])
    s["special_predictors"] = [p for _, p in kept_special]
    if "outcome_combinations" in s:
        s["outcome_combinations"] = [c for _, c in kept_combinations]
    return s


def _combinations(s, keep_period):
    """(position, combination) of each outcome combination with the weights of the periods kept, dropped when no
    weight other than 0 is left."""
    out = []
    for m, c in enumerate(s.get("outcome_combinations") or []):
        c2 = {k: w for k, w in c.items() if keep_period(int(k))}
        if any(w != 0 for w in c2.values()):
            out.append((m, c2))
    return out


def without_period(spec, t0):
    """Leave one period out: the outcome at t0 removed from every predictor and from the V search; covariates stay.
    An outcome combination loses its weight on t0, which under Synth's scaling of each predictor by its SD is the same
    as recomputing an average without t0."""
    s = as_special(spec)
    ks = []
    for j, p in enumerate(s["special_predictors"]):
        if p["variable"] == "Y_observed":
            per = [int(t) for t in p["periods"] if int(t) != t0]
            if per:
                ks.append((j, {**p, "periods": per}))
        else:
            ks.append((j, p))
    s = _drop(s, ks, _combinations(s, lambda t: t != t0))
    s["time_optimize_ssr"] = [int(t) for t in s["time_optimize_ssr"] if int(t) != t0]
    return s


def block_spec(spec):
    """The block check: every predictor and the V search limited to periods 1..35."""
    s = json.loads(json.dumps(spec))
    keep = lambda per: [int(t) for t in per if int(t) <= BLOCK_LAST_TRAIN]   # noqa: E731
    s["time_predictors_prior"] = keep(s["time_predictors_prior"]) or [BLOCK_LAST_TRAIN]
    ks = []
    for j, p in enumerate(s["special_predictors"]):
        per = keep(p["periods"])
        if per:
            ks.append((j, {**p, "periods": per}))
        elif p["variable"] != "Y_observed":
            ks.append((j, {**p, "periods": [BLOCK_LAST_TRAIN]}))
    s = _drop(s, ks, _combinations(s, lambda t: t <= BLOCK_LAST_TRAIN))
    s["time_optimize_ssr"] = keep(s["time_optimize_ssr"])
    return s


def tried(f):
    """f(), or None when the refit fails: a failed refit stops only itself and the build goes on (Jun "yes, stop for
    taht datapoint", 2026-11-05)."""
    try:
        return f()
    except RuntimeError as e:
        print(f"refit failed, skipped: {str(e).splitlines()[0]}", flush=True)
        return None


def own_refits(jobs, fits_behind, sigma, ex, who):
    """Leave one period out and block for every distinct specification: {key: (cv_rmse, block_rmse)}, and the
    specifications whose rewritten form with nothing removed does not reproduce a fit behind an answer within SAME
    (fits_behind: {key: [predictions of each fit behind an answer with that specification]}), and the failed refits of
    each specification. CV-RMSE is taken over the periods whose refit worked; a failed block refit leaves it None."""
    obs = {c: observed_and_truth(c, sigma)[0] for c in ("dgp1", "dgp2")}
    tasks = [(k, "lopo", t) for k in jobs for t in PRE] + [(k, "block", None) for k in jobs] + \
            [(k, "same", None) for k in jobs if as_special(jobs[k]) != jobs[k]]
    spec_of = {"lopo": lambda k, t: without_period(jobs[k], t), "block": lambda k, t: block_spec(jobs[k]),
               "same": lambda k, t: as_special(jobs[k])}
    fits = dict(zip(tasks, ex.map(lambda x: tried(lambda: run_checks(
        x[0][0], spec_of[x[1]](x[0], x[2]), sigma, label=f"{who[x[0]]}, {x[1]} {x[2] or ''}")["predictions"]), tasks)))
    out, bad, failed = {}, [], {}
    for k in jobs:
        y = obs[k[0]]
        miss = [t for t in PRE if fits[(k, "lopo", t)] is None]
        e = [y[t] - fits[(k, "lopo", t)][str(t)] for t in PRE if t not in miss]
        bf = fits[(k, "block", None)]
        out[k] = (math.sqrt(sum(x * x for x in e) / len(e)) if e else None,
                  math.sqrt(sum((y[t] - bf[str(t)]) ** 2 for t in BLOCK) / len(BLOCK)) if bf is not None else None)
        failed[k] = [f"period {t} left out" for t in miss] + (["block"] if bf is None else [])
        if (k, "same", None) in fits:
            sf = fits[(k, "same", None)]
            for pr in fits_behind.get(k, []):
                d = max(abs(sf[t] - pr[t]) for t in pr) if sf is not None else "refit failed"
                if sf is None or d > SAME:
                    bad.append((k[0], d))
    return out, bad, failed


def answers(sigma="1.0"):
    """(arm, last record of each case) over the record folders of one noise level."""
    for arm, folder in mod.record_folders(sigma).items():
        for p in sorted(folder.glob("*.jsonl")):
            last = {}
            for line in p.read_text().splitlines():
                r = json.loads(line)
                last[r["prompt_id"]] = r
            for r in last.values():
                yield arm, r


def fit_record(arm, r, use):
    """-> (the recorded fit behind a configure or design answer, note), or (None, note) for a profile answer."""
    if arm in PROFILE_SPECS:
        return None, "skill script's specification"
    k = scs_steps.matched_run(r, use)
    if k:
        return r["fit_runs"][k - 1], "matched fit"
    cf = r["answer"]["prediction"]["counterfactual"]
    dist = [(max(abs(cf[t] - f["predictions"][t]) for t in POST), f) for f in r["fit_runs"]]
    dmin, f = min(dist, key=lambda x: x[0])
    return f, f"closest fit, {dmin:.1e} from the answer"


def no_fit_gives(r, fit, note):
    """True when the answer's counterfactual comes from none of its recorded fits: the closest fit differs from it by
    more than SAME. Such an answer has no fit behind it, so the rows computed from a fit (4.2, 4.3, random noise) stay
    empty, and a row never mixes the answer with another path (~/.claude/CLAUDE.md, "Only comparable numbers in one
    table"; Jun "i need all table comparable", 2026-11-06)."""
    if fit is None or not note.startswith("closest fit"):
        return False
    cf = r["answer"]["prediction"]["counterfactual"]
    return max(abs(cf[t] - fit["predictions"][t]) for t in POST) > SAME


def fit_behind(arm, r, use):
    """-> (spec, the answer's own fit predictions or None, note)."""
    f, note = fit_record(arm, r, use)
    if f is None:
        return PROFILE_SPECS[arm], None, note
    return f["spec"], f["predictions"], note


def recorded(fit, case, sigma):
    """A recorded fit in the form arm_checks.R gives a fit: its path, donor weights, ATT and 4.2, computed as
    arm_checks.R computes them (the gaps are the treated unit's outcome minus the synthetic control; covariate balance
    at period 40)."""
    gaps = {int(t): g for t, g in fit["gaps"].items()}
    w = dict(zip(fit["donor_ids"], fit["weights"]))
    panel = pd.read_csv(mod.PANELS_BY_SIGMA[sigma][case])
    z = panel[panel.period == 40].set_index("unit_id")
    zb = [z.loc["treated", c] - sum(wj * z.loc[j, c] for j, wj in w.items()) for c in ("Z1", "Z2")]
    return {"predictions": fit["predictions"], "weights": w, "att": sum(gaps[t] for t in range(41, 61)) / 20,
            "pre_rmspe": math.sqrt(sum(gaps[t] ** 2 for t in PRE) / len(PRE)),
            "covariate_balance": math.sqrt(sum(x * x for x in zb) / 2)}


def run_checks(case, spec, sigma="1.0", donor=None, label=""):
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "spec.json"
        sp.write_text(json.dumps(spec))
        p = subprocess.run([str(mod.RSCRIPT), str(HERE / "arm_checks.R"), str(mod.PANELS_BY_SIGMA[sigma][case]), str(sp),
                            *([donor] if donor else [])], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{label or case}: {p.stderr[-600:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def observed_and_truth(case, sigma="1.0"):
    panel = pd.read_csv(mod.PANELS_BY_SIGMA[sigma][case])
    t = panel[panel.unit_id == "treated"].set_index("period")["Y_observed"]
    tr = pd.read_csv(mod.TRUTH_BY_SIGMA[sigma])
    tr = tr[tr.dgp == case].set_index("period")
    return t, tr


def known_factor_row(sigma="1.0"):
    """DGP 1 rebuilt with the shipped seed, its noise multiplied by sigma as the generator does (line 72); the
    counterfactual is the treated unit's shared part."""
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
    eps = float(sigma) * rng.normal(size=(g.T, g.N_UNITS))
    shared = delta + theta @ Z[0] + lam @ mu[0]
    obs, tr = observed_and_truth("dgp1", sigma)
    assert np.allclose(shared + eps[:, 0], tr["Y0_true"].to_numpy(), atol=1e-12, rtol=0), \
        f"rebuild does not match sigma_{sigma}/ground_truth.csv"
    cf = dict(zip(range(1, 61), shared))
    post, pre = range(41, 61), range(1, 41)
    att = sum(obs[t] - cf[t] for t in post) / 20
    true_att = tr.loc[list(post), "tau_true"].mean()
    return {"arm": "every factor known", "model": "", "case": "dgp1", "answered": True, "note": "generator, shipped seed",
            "cf_rmse": math.sqrt(sum((cf[t] - tr.loc[t, "Y0_true"]) ** 2 for t in post) / 20),
            "att_error": att - true_att, "rel_att_error": abs(att - true_att) / abs(true_att),
            "pre_rmspe": math.sqrt(sum((obs[t] - cf[t]) ** 2 for t in pre) / 40)}


def readme_rows(sigma="1.0"):
    """README Section 3: replication 1 of the collaborator's replication files of that level, read as they are."""
    out = []
    for g in ("dgp1", "dgp2"):
        r = pd.read_csv(README_ROW_FILES / f"sigma_{sigma}" / f"mc_replications_{g}.csv")
        r = r[r.replication == 1].iloc[0]
        out.append({"arm": "README Section 3", "model": "", "case": g, "answered": True,
                    "note": f"mc_results_r/sigma_{sigma}, replication 1",
                    "cf_rmse": r.counterfactual_rmse, "att_error": r.att_error, "rel_att_error": r.relative_att_error,
                    "pre_rmspe": r.pre_rmspe, "covariate_balance": r.covariate_balance,
                    "lodo_max_abs_att_change": r.loo_donor_max_shift, "lodo_donor": None, "lodo_donor_weight": None,
                    "n_positive": int(r.n_donors_used), "cv_rmse": r.cv_rmse, "block_rmse": r.block_rmse})
    return out


def merge_rows(path, new, arms):
    """Write table `path` with the rows of `arms` replaced by `new` and every other row kept as its text in the file
    (Jun "keep the original two variable one as comparison", 2026-11-05): other rows are never recomputed, since refits
    with optimizer "All" draw random numbers. A column the file lacks is added, empty in its old rows."""
    import io
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    buf = io.StringIO()
    new.to_csv(buf, index=False)
    new = pd.read_csv(io.StringIO(buf.getvalue()), dtype=str, keep_default_na=False)
    assert set(new.arm) <= set(arms), (set(new.arm), arms)
    cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
    kept = old[~old.arm.isin(arms)].reindex(columns=cols, fill_value="")
    pd.concat([kept, new.reindex(columns=cols, fill_value="")], ignore_index=True).to_csv(path, index=False)


def refresh_scores(sigma="1.0"):
    """Table 4.1's cf_rmse, att_error and rel_att_error of every answer, from scores.csv, which table 4.1 copies; every
    other cell, and tables 4.2 and 4.3, kept as their text, so no refit runs again (plan step 11)."""
    import io
    out = mod.results_folder(sigma)
    path = out / "table_4_1.csv"
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    sc = pd.read_csv(out / "scores.csv")
    cols, n = ["cf_rmse", "att_error", "rel_att_error"], 0
    for i, row in t.iterrows():
        m = sc[(sc.arm == row.arm) & (sc.model == row.model) & (sc.case == row.case) & sc.ok]
        if len(m) != 1:
            continue                                  # the reference rows and the cases with no answer
        buf = io.StringIO()
        m[cols].to_csv(buf, index=False)
        vals = pd.read_csv(io.StringIO(buf.getvalue()), dtype=str, keep_default_na=False).iloc[0]
        for c in cols:
            t.at[i, c] = vals[c]
        n += 1
    t.to_csv(path, index=False)
    print(f"sigma {sigma}: table 4.1 scores of {n} answers taken from scores.csv")


def main(sigma="1.0", arms=None):
    use = mod.USES[mod.use_name(sigma)]
    out = mod.results_folder(sigma)
    sc = pd.read_csv(out / "scores.csv")
    rows, jobs = [], {}
    for arm, r in answers(sigma):
        if arms and arm not in arms:
            continue
        row = {"arm": arm, "model": r["model"], "case": r["case"], "answered": bool(r["ok"])}
        if r["ok"]:
            fit, note = fit_record(arm, r, use)
            if no_fit_gives(r, fit, note):
                s = sc[(sc.arm == arm) & (sc.model == r["model"]) & (sc.case == r["case"])].iloc[0]
                row.update(note="no recorded fit gives this counterfactual", cf_rmse=s.cf_rmse,
                           att_error=s.att_error, rel_att_error=s.rel_att_error)
                rows.append(row)
                continue
            spec = PROFILE_SPECS[arm] if fit is None else fit["spec"]
            key = (r["case"], json.dumps(spec, sort_keys=True))
            jobs[key] = spec
            row.update(key=key, fit=fit, note=note, cf=r["answer"]["prediction"]["counterfactual"])
            s = sc[(sc.arm == arm) & (sc.model == r["model"]) & (sc.case == r["case"])].iloc[0]
            row.update(cf_rmse=s.cf_rmse, att_error=s.att_error, rel_att_error=s.rel_att_error)
        rows.append(row)
    done = [row for row in rows if row["answered"] and "key" in row]       # answers with a fit behind them
    who = {}                                                   # the answers behind each specification, for errors
    for row in done:
        who[row["key"]] = "; ".join(filter(None, [who.get(row["key"]), f"{row['arm']} {row['model']} {row['case']}"]))
    with ThreadPoolExecutor(64) as ex:
        synth = sorted({row["key"] for row in done if row["fit"] is None})       # profile answers: no recorded fit
        synth = dict(zip(synth, ex.map(lambda k: tried(lambda: run_checks(k[0], jobs[k], sigma,
                                                                          label=f"{who[k]}, Synth fit")), synth)))
        for row in done:
            row["base"] = synth[row["key"]] if row["fit"] is None else recorded(row["fit"], row["case"], sigma)
        based = [row for row in done if row["base"] is not None]
        drops = sorted({(row["key"], j) for row in based for j, wj in row["base"]["weights"].items() if wj > DONOR_MIN})
        att_j = dict(zip(drops, ex.map(lambda kj: tried(lambda: run_checks(
            kj[0][0], jobs[kj[0]], sigma, kj[1], label=f"{who[kj[0]]}, without {kj[1]}")["att"]), drops)))
        behind = {}
        for row in based:
            behind.setdefault(row["key"], []).append(row["base"]["predictions"])
        own, not_same, failed = own_refits({k: jobs[k] for k in behind}, behind, sigma, ex, who)
    print(f"sigma {sigma}: rewritten specifications that do not reproduce their fit within {SAME}: {not_same}")
    bad = []
    for row in done:
        c = row["base"]
        if c is None:
            row.update(failed_refits="Synth fit")
            continue
        above = [j for j, wj in c["weights"].items() if wj > DONOR_MIN]
        ch = {j: abs(att_j[(row["key"], j)] - c["att"]) for j in above if att_j[(row["key"], j)] is not None}
        j = max(ch, key=ch.get) if ch else None
        row["failed_refits"] = "; ".join([f"donor {d} left out" for d in above if d not in ch] +
                                         failed[row["key"]]) or None
        if row["fit"] is None:
            d = max(abs(c["predictions"][t] - row["cf"][t]) for t in POST)
            row["synth_vs_answer"] = d
            if d > SAME:
                bad.append((row["arm"], row["model"], row["case"], d))
        row.update(pre_rmspe=c["pre_rmspe"], covariate_balance=c["covariate_balance"],
                   lodo_max_abs_att_change=ch[j] if j else None, lodo_donor=j,
                   lodo_donor_weight=c["weights"][j] if j else None, n_positive=len(above),
                   cv_rmse=own[row["key"]][0], block_rmse=own[row["key"]][1])
    print(f"sigma {sigma}: {len(jobs)} distinct specifications; profile answers the Synth fit of the script's "
          f"specification does not reproduce within {SAME}: {bad}")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("key", "fit", "cf", "base")} for r in rows])
    if not arms:
        refs = readme_rows(sigma) + [known_factor_row(sigma)]
        df = pd.concat([pd.DataFrame(refs), df], ignore_index=True)
    for c in ("cv_rmse", "block_rmse", "failed_refits"):
        if c not in df:
            df[c] = None
    keys = ["arm", "model", "case", "answered", "note"]
    tables = {"4_1": keys + ["cf_rmse", "att_error", "rel_att_error"],
              "4_2": keys + ["pre_rmspe", "covariate_balance"],
              "4_3": keys + ["lodo_max_abs_att_change", "lodo_donor", "lodo_donor_weight", "n_positive",
                             "cv_rmse", "block_rmse", "failed_refits"]}
    out.mkdir(parents=True, exist_ok=True)
    for name, cols in tables.items():
        t = df.reindex(columns=cols) if arms else df[cols]     # the named arms can have no answered case
        if arms:
            merge_rows(out / f"table_{name}.csv", t, arms)
        else:
            t.to_csv(out / f"table_{name}.csv", index=False)
        print(f"\n== sigma {sigma} table {name}\n" + t.to_string(index=False))
    print("Synth fit vs a profile answer, max over answers:",
          df["synth_vs_answer"].max() if "synth_vs_answer" in df else None)


if __name__ == "__main__":
    s = sys.argv[sys.argv.index("--sigma") + 1] if "--sigma" in sys.argv else "1.0"
    if s not in mod.SIGMAS:
        raise SystemExit(f"--sigma {s}: one of {', '.join(mod.SIGMAS)}")
    if "--scores-only" in sys.argv:
        refresh_scores(s)
        raise SystemExit(0)
    a = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else None
    if a and not set(a) <= set(mod.record_folders(s)):
        raise SystemExit(f"--arms {','.join(a)}: arms are {', '.join(mod.record_folders(s))}")
    main(s, a)
