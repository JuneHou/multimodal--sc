"""Number registry rows for the SCM report deck (number-check skill, section 0).

Every value is read from its source file here, formatted as the slides print it, and written as one row of the
registry section between the two markers below in NUMBERS.md at the repository root. The same rows are also written to
report/deck/numbers_section.md, so the deck can be checked against its own rows alone.

    python report/number_rows.py  rewrites the section and prints the rows
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]                 # notebooks/scm_simulation
ROOT = HERE.parents[2]                                     # repository root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "report"))
import her_tables as h  # noqa: E402
import luck  # noqa: E402

REG = ROOT / "NUMBERS.md"
START, END = "<!-- scm-report-deck start -->", "<!-- scm-report-deck end -->"
TODAY = dt.date.today().isoformat()
S = "Satellite/notebooks/scm_simulation/"
DATA = "Satellite/data/simulation/"
MODELS = [("gpt-oss-120b", "gpt-oss-120b"), ("GLM-5.3", "GLM-5.3"), ("openai:gpt-5.5", "GPT-5.5")]


def f3(v):
    return f"{v:.3f}".replace("-", "−")


def table(name):
    return pd.read_csv(HERE / "results" / f"{name}.csv").fillna({"model": ""})


def pick(t, arm, case, col, model=""):
    r = t[(t.arm == arm) & (t.case == case) & (t.model == model)] if model else t[(t.arm == arm) & (t.case == case)]
    vals = r[col].dropna().unique()
    assert len(vals) == 1, (arm, model, case, col, vals)
    return float(vals[0])


def main():
    rows = []

    def add(name, values, what, source, how):
        rows.append((name, ", ".join(str(v) for v in values), what, source, how))

    cfg = json.loads((ROOT / DATA / "simulation_data/config.json").read_text())
    add("scm_design", [cfg["n_units"], cfg["n_donors"], cfg["T"], cfg["T0"], cfg["T_post"],
                       cfg["first_treated_period"], cfg["T"], f"{cfg['true_ATT']:.0f}".replace("-", "−"),
                       f"{abs(cfg['true_ATT']):.0f}", cfg["dgp1"]["n_factors"], cfg["dgp2"]["alpha"]],
        "units, donors, periods, pre-treatment periods, post-treatment periods, first treated period, last period, "
        "true effect per treated period, its size, DGP 1 latent factors, DGP 2 outcome persistence alpha",
        f"`{DATA}simulation_data/config.json`", "keys n_units, n_donors, T, T0, T_post, first_treated_period, "
        "true_ATT, dgp1.n_factors, dgp2.alpha")
    add("scm_paper", [493, 495, 496, "2.2", "2.3"],
        "pages of Abadie, Diamond and Hainmueller: abstract, Section 2.2 with Equation 1 and the estimand, Section 2.3 "
        "with the convex weights, V and Equation 5; the two section numbers",
        f"`{S}profile/work/gpt-5.4/paper/` (the paper PDF)", "pdftotext of the paper, page headers")
    add("scm_readme_sections", ["4.1", "4.2", "4.3"],
        "section numbers of the README's three checks", f"`{DATA}README.md`", "section headings")
    add("scm_readme_spec", [42, 40, 1, 40, 30],
        "README Section 3: predictors (Z1, Z2 at period 40 and the outcome at each of periods 1 to 40), the period of "
        "the covariates, the outcome periods, the units over which each predictor's standard deviation is taken",
        f"`{DATA}README.md`; `{S}her_section3.R`", "Section 3 Steps 1 and 2")
    adh = h.adh_spec()
    add("scm_adh_spec", [len(adh["predictors"]) + len(adh["special_predictors"]), int(adh["time_predictors_prior"][0]),
                         int(adh["time_predictors_prior"][-1]),
                         *[s["periods"][0] for s in adh["special_predictors"]],
                         int(adh["time_optimize_ssr"][0]), int(adh["time_optimize_ssr"][-1])],
        "ADH paper specification on this data: predictors; Z1 and Z2 averaged over periods 23 to 40; the outcome at "
        "periods 12, 23 and 40; V searched over the outcome error of periods 1 to 40",
        f"`{S}method.py`", "ADH_WAY")
    add("scm_checks", [20, 3, 40, 39, 36, 40, 1, 35],
        "periods averaged by the counterfactual RMSE; the true effect size in the relative ATT error; pre-treatment "
        "periods of the RMSPE; periods each leave-one-period-out fit uses; the held-out block 36 to 40 and its fit "
        "periods 1 to 35", f"`{DATA}README.md`", "Section 4.1, 4.2 and 4.3 formulas")

    t1, t2, t3 = table("table_4_1"), table("table_4_2"), table("table_4_3")
    refs = [("README Section 3", "readme"), ("ADH paper specification", "adh")]
    for t, cols, key in ((t1, ["cf_rmse", "att_error", "rel_att_error"], "t41"),
                         (t2, ["pre_rmspe", "covariate_balance"], "t42"),
                         (t3, ["lodo_max_abs_att_change", "n_positive"], "t43")):
        src = f"`{S}results/table_{key[1]}_{key[2]}.csv`"
        for arm, short in refs:
            vals = [pick(t, arm, c, col) for c in ("dgp1", "dgp2") for col in cols]
            add(f"scm_{key}_{short}", [f3(v) if col != "n_positive" else int(v) for v, col in zip(vals, cols * 2)],
                f"{arm}: {', '.join(cols)} for DGP 1 then DGP 2", src, "her_tables.py")
        if key != "t43":
            known = [pick(t, "every factor known", "dgp1", col) for col in cols if col != "covariate_balance"]
            add(f"scm_{key}_known", [f3(v) for v in known],
                f"every factor known, DGP 1: {', '.join(c for c in cols if c != 'covariate_balance')}", src,
                "her_tables.py known_factor_row")
        for arm in ("scm-configure", "scm-design"):
            for m, label in MODELS:
                vals = [pick(t, arm, c, col, m) for c in ("dgp1", "dgp2") for col in cols]
                add(f"scm_{key}_{arm[4:]}_{label}",
                    [f3(v) if col != "n_positive" else int(v) for v, col in zip(vals, cols * 2)],
                    f"{arm}, {label}: {', '.join(cols)} for DGP 1 then DGP 2", src, "her_tables.py")
        for arm in ("profile-gpt-5.4", "profile-deepseek"):
            vals = [pick(t, arm, c, col) for c in ("dgp1", "dgp2") for col in cols]
            add(f"scm_{key}_{arm}", [f3(v) if col != "n_positive" else int(v) for v, col in zip(vals, cols * 2)],
                f"{arm}, all four models (identical answers): {', '.join(cols)} for DGP 1 then DGP 2", src,
                "her_tables.py; pick() asserts one value over the four models")
    fixed = {"dgp1": (1.17762004448812, 1.74638539766192), "dgp2": (0.962351927945658, 1.01983827295471)}
    her = pd.read_csv(HERE / "results" / "her_section3_errors.csv")
    for c in ("dgp1", "dgp2"):
        r = her[her.dgp == c] if "dgp" in her.columns else her[her.case == c]
        cv = float(r[[k for k in r.columns if "cv" in k.lower()][0]].iloc[0])
        bl = float(r[[k for k in r.columns if "block" in k.lower()][0]].iloc[0])
        assert abs(cv - fixed[c][0]) < 1e-9 and abs(bl - fixed[c][1]) < 1e-9, (c, cv, bl)
    add("scm_t43_fixed", [f3(fixed["dgp1"][0]), f3(fixed["dgp1"][1]), f3(fixed["dgp2"][0]), f3(fixed["dgp2"][1])],
        "Section 4.3 rows that do not depend on the answer: leave-one-period-out CV-RMSE and block 36 to 40 RMSE, "
        "DGP 1 then DGP 2", f"`{S}results/her_section3_errors.csv`", "her_section3.R, printed by her_tables.py")

    sc = pd.read_csv(HERE / "results" / "scores.csv")
    ans = sc.groupby("arm").ok.agg(["sum", "count"])
    ds = sc[sc.model.str.startswith("DeepSeek") & sc.arm.isin(["scm-configure", "scm-design"])]
    assert (~ds.ok).all() and (ds.n_steps == 12).all()
    add("scm_answered", [int(ans.loc["scm-configure", "sum"]), int(ans.loc["scm-design", "sum"]),
                         int(ans.loc["profile-gpt-5.4", "sum"]), int(ans.loc["profile-deepseek", "sum"]),
                         int(ans.loc["scm-configure", "count"]), int(ds.n_steps.max()), len(ds)],
        "cases answered with scm-configure, scm-design and each profile skill, of the cases per format; steps every "
        "DeepSeek-V4.1-Flash configure and design case used; those cases", f"`{S}results/scores.csv`", "ok, n_steps")
    prof = sc[sc.arm.str.startswith("profile")]
    add("scm_profile_runs", [len(prof)], "profile runs: two skills, four models, two cases, each the skill's script "
        "at its defaults", f"`{S}results/steps_profile.csv`; `{S}results/scores.csv`", "rows of the profile arms")

    recs = [json.loads(l) for l in (HERE / "runs/simulation/scm-configure_GLM-5.3_low.jsonl").read_text().splitlines()]
    caps = sorted({r["max_tokens"] for p in (HERE / "runs").glob("simulation*/*.jsonl")
                   for r in map(json.loads, p.read_text().splitlines())})
    assert caps == [16000, 32000], caps
    add("scm_run", ["16,000", "32,000", recs[-1]["env"]["step_cap"], 120, "4.3", "1.1"],
        "token cap per reply for gpt-oss-120b, GLM-5.3 and GPT-5.5, and for DeepSeek-V4.1-Flash; step cap; seconds "
        "per command; R version 4.3.3; Synth version 1.1-10",
        f"`{S}runs/simulation*/*.jsonl`; `Satellite/notebooks/llm_band_sc/lb_skill_loop.py`",
        "max_tokens and env.step_cap of every record; CMD_TIMEOUT")
    add("scm_skill_fields", [13, 13], "settings the model decides in scm-configure, and the Synth settings of "
        "scm-design beside its three choices", f"`{S}skills/scm-configure/SKILL.md`; `{S}skills/scm-design/SKILL.md`",
        "the SKILL.md lists")

    lk = pd.read_csv(HERE / "results" / "luck_replications.csv").att_error_known_factors
    add("scm_luck", [len(lk), f3(lk.std(ddof=1)), f3(lk.min()), "+" + f3(lk.max()), f3(lk.iloc[0]), len(lk) - 1],
        "replications of DGP 1 (shipped and seeds 1 to 10); standard deviation, minimum and maximum of the "
        "known-factor ATT error over them; the shipped replication's; the new replications",
        f"`{S}results/luck_replications.csv`", "report/luck.py")

    d2 = pd.read_csv(ROOT / DATA / "simulation_data/dgp2_panel.csv")
    pre = d2[d2.period <= 40].pivot(index="period", columns="unit_id", values="Y_observed")
    corr = float(np.corrcoef(pre["treated"], pre.drop(columns="treated").mean(axis=1))[0, 1])
    add("scm_dgp2_corr", [f"{corr:.2f}"], "DGP 2, periods 1 to 40: correlation of the treated unit's outcome with the "
        "donor average", f"`{DATA}simulation_data/dgp2_panel.csv`", "np.corrcoef over periods 1 to 40")

    obs = {c: h.observed_and_truth(c) for c in ("dgp1", "dgp2")}
    gaps, other = [], []
    for arm, r in h.answers():
        if not r["ok"]:
            continue
        cf = {int(k): v for k, v in r["answer"]["prediction"]["counterfactual"].items()}
        g = float(np.mean([obs[r["case"]][0][t] - cf[t] for t in cf]))
        if r["model"] == "gpt-oss-120b" and arm in ("scm-configure", "scm-design"):
            gaps.append((arm, r["case"], r["answer"]["att"], g, abs(r["answer"]["att"] - g)))
        else:
            other.append(abs(r["answer"]["att"] - g))
    ex = [x for x in gaps if x[0] == "scm-configure" and x[1] == "dgp1"][0]
    diffs = sorted(x[4] for x in gaps)
    assert max(other) < 1e-4, max(other)
    add("scm_att_gap", [f3(ex[2]), f"{ex[2]:.3f}", f3(ex[3]), f"{diffs[0]:.2f}", f"{diffs[-1]:.2f}", len(gaps), "0.0001"],
        "gpt-oss-120b, scm-configure, DGP 1: reported ATT (also as its rationale prints it, ASCII minus) and the mean gap of its own counterfactual; smallest and "
        "largest difference between the two over its configure and design answers; those answers; every other "
        "answer's difference is below 0.0001", f"`{S}runs/simulation/*.jsonl`; `{DATA}simulation_data/*_panel.csv`",
        "answer att against mean over 41..60 of Y_observed minus the counterfactual")

    sys.path.insert(0, str(HERE / "report"))
    import figures as fg
    tr, an = fg.truth("dgp1"), fg.answers("dgp1")
    E = pd.DataFrame({k: s - tr.loc[s.index] for k, s in an.items()
                      if not (k[0].startswith("profile") and k[1] != "openai:gpt-5.5")})
    common = [t for t in E.index if (E.loc[t] > 1).all() or (E.loc[t] < -1).all()]
    c = E.corr().values
    add("scm_dgp1_misses", [*common, f"{c[np.triu_indices_from(c, 1)].min():.2f}", E.shape[1]],
        "DGP 1: periods where every answer misses the true path by more than 1 in the same direction; smallest "
        "correlation between two answers' error paths over periods 41 to 60; distinct answers (profile shown once)",
        f"`{S}runs/simulation*/*.jsonl`; `{DATA}simulation_data/ground_truth.csv`", "report/figures.py answers()")

    sp = pd.read_csv(HERE / "results" / "specs_answers.csv")
    add("scm_specs", [*sorted(sp.n_predictors.unique()), sp.donors.unique()[0], 10, 11, 20, 21, 30, 31, 40],
        "predictors per specification; donors in every specification; period bounds of the outcome blocks 1 to 10, "
        "11 to 20, 21 to 30 and 31 to 40, and GLM-5.3's covariate periods 1 and 20",
        f"`{S}results/specs_answers.csv`", "report/specs.py")
    add("scm_model_names", ["5.3", "5.4", "5.5"], "version numbers inside the model names GLM-5.3, GPT-5.4 and GPT-5.5",
        f"`{S}runs/simulation*/*.jsonl`; `{S}profile/out/gpt-5.4/run.log`", "model of every record; the generator name")
    add("scm_refit", [f"{h.SAME:.4f}"], "largest difference allowed between a profile answer and the refit of the skill "
        "script's specification, in every period", f"`{S}her_tables.py`", "SAME; the rebuild printed a largest "
        "difference of 4.9e-05")
    add("scm_fig_ticks", [1, 10, 20, 30, 40, 50, 60, 41, 2, 4, "0.0", "0.5", "1.0", "1.5", "2.5"],
        "axis tick labels of the three figures (periods; outcome; ATT error)", f"`{S}report/figures.py`",
        "matplotlib ticks")

    head = "| name | value | what it counts | source file | how it is computed | date read |\n|---|---|---|---|---|---|\n"
    body = "".join(f"| {n} | {v} | {w} | {s} | {hw} | {TODAY} |\n" for n, v, w, s, hw in rows)
    section = (f"{START}\n## SCM simulation report deck (Satellite/notebooks/scm_simulation/report)\n\n"
               f"Written by `{S}report/number_rows.py`.\n\n{head}{body}{END}\n")
    text = REG.read_text()
    if START in text:
        text = text[:text.index(START)] + section + text[text.index(END) + len(END) + 1:]
    else:
        text = text.rstrip("\n") + "\n\n" + section
    REG.write_text(text)
    (HERE / "report" / "deck" / "numbers_section.md").write_text(section)
    print(section)


if __name__ == "__main__":
    main()
