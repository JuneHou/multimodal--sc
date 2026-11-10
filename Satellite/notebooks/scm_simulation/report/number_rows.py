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
# the arms the deck shows; the deck's scm-design is the four-choice design skill, arm scm-four-design, and the
# two-choice scm-design leaves the deck (Jun 2026-11-06 "keep only four variable case as scm-design")
DESIGN = "scm-four-design"
DECK_ARMS = ("scm-configure", DESIGN, "profile-gpt-5.4", "profile-deepseek")
# DeepSeek-V4.1-Flash is not a solver model of the deck (Jun "drop all deepseek in this artifact", "Solver model only",
# 2026-11-06); the profile skill the DeepSeek generator wrote stays
DROPPED_SOLVER = "DeepSeek"


def deck_rows(t):
    """Rows of a table whose model is a solver model of the deck."""
    return t[~t.model.astype(str).str.startswith(DROPPED_SOLVER)]


def f3(v):
    return f"{v:.3f}".replace("-", "−")


def table(name):
    return pd.read_csv(HERE / "results" / "sigma_1.0" / f"{name}.csv").fillna({"model": ""})


def pick(t, arm, case, col, model=""):
    t = deck_rows(t)
    r = t[(t.arm == arm) & (t.case == case) & (t.model == model)] if model else t[(t.arm == arm) & (t.case == case)]
    vals = r[col].dropna().unique()
    assert len(vals) == 1, (arm, model, case, col, vals)
    return float(vals[0])


def main():
    rows = []

    def add(name, values, what, source, how):
        rows.append((name, ", ".join(str(v) for v in values), what, source, how))

    cfg = json.loads((ROOT / DATA / "simulation_data/sigma_1.0/config.json").read_text())
    add("scm_design", [cfg["n_units"], cfg["n_donors"], cfg["T"], cfg["T0"], cfg["T_post"],
                       cfg["first_treated_period"], cfg["T"], f"{cfg['true_ATT']:.0f}".replace("-", "−"),
                       f"{abs(cfg['true_ATT']):.0f}", cfg["dgp1"]["n_factors"], cfg["dgp2"]["alpha"]],
        "units, donors, periods, pre-treatment periods, post-treatment periods, first treated period, last period, "
        "true effect per treated period, its size, DGP 1 latent factors, DGP 2 outcome persistence alpha",
        f"`{DATA}simulation_data/sigma_1.0/config.json`", "keys n_units, n_donors, T, T0, T_post, first_treated_period, "
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
    add("scm_checks", [20, 3, 40, 39, 36, 40, 1, 35],
        "periods averaged by the counterfactual RMSE; the true effect size in the relative ATT error; pre-treatment "
        "periods of the RMSPE; periods each leave-one-period-out fit uses; the held-out block 36 to 40 and its fit "
        "periods 1 to 35", f"`{DATA}README.md`", "Section 4.1, 4.2 and 4.3 formulas")

    t1, t2, t3 = table("table_4_1"), table("table_4_2"), table("table_4_3")
    refs = [("README Section 3", "readme")]
    for t, cols, key in ((t1, ["cf_rmse", "att_error", "rel_att_error"], "t41"),
                         (t2, ["pre_rmspe", "covariate_balance"], "t42"),
                         (t3, ["lodo_max_abs_att_change", "n_positive", "cv_rmse", "block_rmse"], "t43")):
        src = f"`{S}results/sigma_1.0/table_{key[1]}_{key[2]}.csv`"
        for arm, short in refs:
            vals = [pick(t, arm, c, col) for c in ("dgp1", "dgp2") for col in cols]
            add(f"scm_{key}_{short}", [f3(v) if col != "n_positive" else int(v) for v, col in zip(vals, cols * 2)],
                f"{arm}: {', '.join(cols)} for DGP 1 then DGP 2", src, "her_tables.py")
        if key != "t43":
            known = [pick(t, "every factor known", "dgp1", col) for col in cols if col != "covariate_balance"]
            add(f"scm_{key}_known", [f3(v) for v in known],
                f"every factor known, DGP 1: {', '.join(c for c in cols if c != 'covariate_balance')}", src,
                "her_tables.py known_factor_row")
        for arm in ("scm-configure", DESIGN):
            for m, label in MODELS:
                dt_ = deck_rows(t)
                if not len(dt_[(dt_.arm == arm) & (dt_.model == m)]):
                    continue                                # GPT-5.5 has not run scm-design (Jun "Rebuild without it")
                vals = [pick(t, arm, c, col, m) for c in ("dgp1", "dgp2") for col in cols]
                add(f"scm_{key}_{arm[4:]}_{label}",
                    [f3(v) if col != "n_positive" else int(v) for v, col in zip(vals, cols * 2)],
                    f"{arm}, {label}: {', '.join(cols)} for DGP 1 then DGP 2", src, "her_tables.py")
        for arm in ("profile-gpt-5.4", "profile-deepseek"):
            vals = [pick(t, arm, c, col) for c in ("dgp1", "dgp2") for col in cols]
            add(f"scm_{key}_{arm}", [f3(v) if col != "n_positive" else int(v) for v, col in zip(vals, cols * 2)],
                f"{arm}, all three models (identical answers): {', '.join(cols)} for DGP 1 then DGP 2", src,
                "her_tables.py; pick() asserts one value over the three models")
    sc = deck_rows(pd.read_csv(HERE / "results" / "sigma_1.0" / "scores.csv"))
    sc = sc[sc.arm.isin(DECK_ARMS)]
    ans = sc.groupby("arm").ok.agg(["sum", "count"])
    add("scm_answered", [int(ans.loc["scm-configure", "sum"]), int(ans.loc[DESIGN, "sum"]),
                         int(ans.loc["profile-gpt-5.4", "sum"]), int(ans.loc["profile-deepseek", "sum"]),
                         int(ans.loc["scm-configure", "count"]), int(ans.loc[DESIGN, "count"])],
        "cases answered with scm-configure, scm-design (arm scm-four-design) and each profile skill; the cases run "
        "with scm-configure and with scm-design (GPT-5.5 has not run scm-design)", f"`{S}results/sigma_1.0/scores.csv`", "ok")
    prof = sc[sc.arm.str.startswith("profile")]
    add("scm_profile_runs", [len(prof)], "profile runs: two skills, three models, two cases, each the skill's script "
        "at its defaults", f"`{S}results/sigma_1.0/steps_profile.csv`; `{S}results/sigma_1.0/scores.csv`",
        "rows of the profile arms")

    recs = [json.loads(l) for l in (HERE / "scm-configure/sigma_1.0/scm-configure_GLM-5.3_low.jsonl").read_text().splitlines()]
    caps = {arm: sorted({str(r["max_tokens"]) for p in folder.glob("*.jsonl") if DROPPED_SOLVER not in p.name
                         for r in map(json.loads, p.read_text().splitlines())})
            for arm, folder in h.mod.record_folders("1.0").items() if arm in DECK_ARMS}
    assert caps == {a: (["None"] if a == DESIGN else ["16000"]) for a in DECK_ARMS}, caps
    add("scm_run", ["16,000", recs[-1]["env"]["step_cap"], 120, "4.3", "1.1"],
        "token cap per reply at sigma 1 under scm-configure and the profile skills (scm-design runs set none); step cap; seconds "
        "per command; R version 4.3.3; Synth version 1.1-10",
        f"`{S}scm-configure/sigma_1.0/scm-configure_GLM-5.3_low.jsonl`; `Satellite/notebooks/llm_band_sc/lb_skill_loop.py`",
        "max_tokens and env.step_cap of every record; CMD_TIMEOUT")
    add("scm_skill_fields", [13, 4, 29, 496], "settings the model decides in scm-configure; the choices of scm-design "
        "(skill scm-four-design), the donors every scm-design fit uses, the paper page of its options",
        f"`{S}skills/scm-configure/SKILL.md`; `{S}four_design/skills/scm-four-design/SKILL.md`", "the SKILL.md lists")

    lk = pd.read_csv(HERE / "results" / "sigma_1.0" / "luck_replications.csv").att_error_known_factors
    add("scm_luck", [len(lk), f3(lk.std(ddof=1)), f3(lk.min()), "+" + f3(lk.max()), f3(lk.iloc[0]), len(lk) - 1],
        "replications of DGP 1 (shipped and seeds 1 to 10); standard deviation, minimum and maximum of the "
        "known-factor ATT error over them; the shipped replication's; the new replications",
        f"`{S}results/sigma_1.0/luck_replications.csv`", "report/luck.py")

    d2 = pd.read_csv(ROOT / DATA / "simulation_data/sigma_1.0/dgp2_panel.csv")
    pre = d2[d2.period <= 40].pivot(index="period", columns="unit_id", values="Y_observed")
    corr = float(np.corrcoef(pre["treated"], pre.drop(columns="treated").mean(axis=1))[0, 1])
    add("scm_dgp2_corr", [f"{corr:.2f}"], "DGP 2, periods 1 to 40: correlation of the treated unit's outcome with the "
        "donor average", f"`{DATA}simulation_data/sigma_1.0/dgp2_panel.csv`", "np.corrcoef over periods 1 to 40")

    obs = {c: h.observed_and_truth(c) for c in ("dgp1", "dgp2")}
    gaps, other = [], []
    for arm, r in h.answers():
        if not r["ok"] or arm not in DECK_ARMS or r["model"].startswith(DROPPED_SOLVER):
            continue
        cf = {int(k): v for k, v in r["answer"]["prediction"]["counterfactual"].items()}
        g = float(np.mean([obs[r["case"]][0][t] - cf[t] for t in cf]))
        if r["model"] == "gpt-oss-120b" and arm in ("scm-configure", DESIGN):
            gaps.append((arm, r["case"], r["answer"]["att"], g, abs(r["answer"]["att"] - g)))
        else:
            other.append(abs(r["answer"]["att"] - g))
    ex = [x for x in gaps if x[0] == "scm-configure" and x[1] == "dgp1"][0]
    diffs = sorted(x[4] for x in gaps if x[4] > 1e-4)
    assert max(other) < 1e-4, max(other)
    add("scm_att_gap", [f3(ex[2]), f"{ex[2]:.3f}", f3(ex[3]), f"{diffs[0]:.2f}", f"{diffs[-1]:.2f}", len(diffs),
                        len(gaps), "0.0001"],
        "gpt-oss-120b, scm-configure, DGP 1: reported ATT (also as its rationale prints it, ASCII minus) and the mean "
        "gap of its own counterfactual; smallest and largest difference between the two over its configure and design "
        "answers that differ; those that differ; its configure and design answers; every other answer's difference "
        "is below 0.0001", f"`{S}scm-configure/sigma_1.0/scm-configure_gpt-oss-120b_low.jsonl`; "
        f"`{DATA}simulation_data/sigma_1.0/dgp1_panel.csv`",
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
        f"`{S}scm-configure/sigma_1.0/`, `{S}scm-four-design/sigma_1.0/`, `{S}skill_profile/`; "
        f"`{DATA}simulation_data/sigma_1.0/ground_truth.csv`", "report/figures.py answers()")

    sp = pd.read_csv(HERE / "results" / "sigma_1.0" / "specs_answers.csv")
    add("scm_specs", [*sorted(sp.n_predictors.unique()), sp.donors.unique()[0], 10, 11, 20, 21, 30, 31, 40],
        "predictors per specification; donors in every specification; period bounds of the outcome blocks 1 to 10, "
        "11 to 20, 21 to 30 and 31 to 40, and GLM-5.3's covariate periods 1 and 20",
        f"`{S}results/sigma_1.0/specs_answers.csv`", "report/specs.py")
    add("scm_model_names", ["5.3", "5.4", "5.5"], "version numbers inside the model names GLM-5.3, GPT-5.4 and GPT-5.5",
        f"`{S}scm-configure/sigma_1.0/`; `{S}profile/out/gpt-5.4/run.log`", "model of every record; the generator name")
    add("scm_donor_min", [f"{h.DONOR_MIN:.3f}"], "weight above which a donor is left out in leave one donor out and "
        "counted, as the collaborator's code counts donors", f"`{DATA}mc_evaluation_r.R`; `{S}her_tables.py`",
        "mc_evaluation_r.R lines 105 and 148, w > 1e-3; her_tables.DONOR_MIN")
    add("scm_refit", [f"{h.SAME:.4f}"], "largest difference allowed between a profile answer and the refit of the skill "
        "script's specification, in every period", f"`{S}her_tables.py`", "SAME; the rebuild printed a largest "
        "difference of 4.9e-05")
    ac = pd.concat([pd.read_csv(h.mod.results_folder(s) / "att_check.csv") for s in h.mod.SIGMAS], ignore_index=True)
    ac = deck_rows(ac[ac.arm.isin(DECK_ARMS)])
    by = ac.groupby("how").differs.agg(["size", "sum"])
    add("scm_att_how", [len(ac), *[int(by.loc[k, c]) for k in ("typed", "computed in code", "profile script")
                                   for c in ("size", "sum")],
                        int(((ac.how == "typed") & ac.differs & ac.typed_seen_period.notna()).sum()),
                        int(((ac.how == "typed") & ac.differs & ac.typed_seen_at_step.isna()).sum()),
                        *sorted({int(p) for p in ac.typed_seen_period.dropna()})],
        "answers at the three noise levels; answers whose ATT was typed, and those that differ from their own "
        "counterfactual; computed in code, and those that differ; written by a profile skill's script, and those "
        "that differ; typed ATTs that differ and equal one period's gap in the fitter's table; typed ATTs that differ "
        "and appear in no earlier output; the periods of those gaps",
        f"`{S}results/sigma_1.0/att_check.csv`, `sigma_0.5/`, `sigma_0.2/`", "report/att_check.py")
    sg = ac[(ac.how == "computed in code") & ac.differs].sort_values("sigma", ascending=False)
    assert list(zip(sg.model, sg.arm, sg.case)) == [("gpt-oss-120b", DESIGN, "dgp1"), ("GLM-5.3", DESIGN, "dgp2")], sg
    a, b = sg.iloc[0], sg.iloc[1]
    add("scm_att_computed", [a.sigma, f"{a.reported_att:.0f}", f3(a.counterfactual_att), b.sigma, f3(b.reported_att),
                             f3(b.counterfactual_att), 41],
        "the two computed ATTs that differ, both under scm-design: gpt-oss-120b at sigma 0.5 on DGP 1 (its "
        "counterfactual and ATT are 0 in every period), its reported ATT and the mean gap of its counterfactual; "
        "GLM-5.3 at sigma 0.2 on DGP 2, its reported ATT and the mean gap of its counterfactual; the period whose "
        "observed outcome GLM-5.3 subtracted from in every period",
        f"`{S}results/sigma_0.5/att_check.csv`, `sigma_0.2/att_check.csv`; the records' last commands",
        "report/att_check.py; read from the commands that wrote the answers")
    add("scm_sigmas", list(h.mod.SIGMAS), "outcome shock SD of the three noise levels",
        f"`{DATA}simulation_data/`", "method.SIGMAS, the data folders sigma_<s>")

    import eval_slides as ev                                   # the results slides in the README Section 5 layout
    her = ev.readme_tables()
    for g in ("dgp1", "dgp2"):
        add(f"scm_readme_s5_{g}", [her[g][row][k] for row in ev.ROWS for k in range(3)],
            f"README Section 5 evaluation results, DGP {g[-1]}: {', '.join(ev.ROWS)}, each at sigma 1.0, 0.5 and 0.2, "
            "as printed (means over 1,000 replications, the last two rows over 100)",
            f"`{DATA}README.md`", "Section 5 Evaluation results tables, copied; her per-replication files reproduce them")
    add("scm_s5_layout", ["0.25", "0.50", "1,000", 100, 1, "0.001"],
        "the two bounds of the within rows; replications behind the README column and behind its leave-one-out rows; "
        "the 1 in sigma*sqrt(1 + sum w^2); the donor weight above which a donor is counted",
        f"`{DATA}README.md`; `{DATA}mc_evaluation_r.R`", "Section 5 tables and text; lines 105, 118 and 148")
    add("scm_scoring_lines", [101, 169, 170, 95, 173, 118, 182, 96, 175, 104, 176, 187, 188, 105, 190, 126, 134, 194,
                              147, 156, 196, 5],
        "lines of the collaborator's evaluation code for each row of the scoring slide (per replication, then the "
        "average over replications): ATT error, bias, RMSE, counterfactual RMSE, random noise, pre-treatment RMSPE, "
        "covariate balance, the two within rows, donors, CV-RMSE and its ratio, leave one donor out; README Section 5",
        f"`{DATA}mc_evaluation_r.R`; `{DATA}README.md`", "grep of the row and summary assignments; section headings")
    for sigma in ev.SIGMAS:
        tag = "r" + sigma.replace(".", "")
        src = (f"`{S}results/sigma_{sigma}/table_4_1.csv`, `table_4_2.csv`, `table_4_3.csv`, `noise_floor.csv`")
        for case in ("dgp1", "dgp2"):
            rows_, heads = ev.grid(sigma, case)
            for j, head in enumerate(heads):
                vals = [v.strip("()") for _, _, vs in rows_ for v in vs[j].split() if v not in ("–", "yes", "no")]
                if vals:
                    name = head.replace("<br>", "_").replace(ev.NB, "-")
                    add(f"scm_{tag}_{case}_{name}", vals,
                        f"sigma {sigma}, {case}, {head.replace('<br>', ' ')}: the README Section 5 rows as the slide "
                        "prints them (ATT error, its absolute value, counterfactual RMSE, random noise, pre-treatment "
                        "RMSPE, covariate balance, donors above 0.001, CV-RMSE over pre-treatment RMSPE, largest "
                        "leave-one-donor-out change), a bracketed count being the refits that worked", src,
                        "report/eval_slides.py value(); noise_floor.py")
    add("scm_fig_ticks", [1, 10, 20, 30, 40, 50, 60, 41, 2, 4, "0.0", "0.2", "0.4", "0.6", "2.5"],
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
