"""SHC daily method package: the collaborator's notebook 14 SHC as the existing daily simple skills use it.

Every value points at an existing file of notebooks/llm_band_sc/ (or the paper copy of the profile experiment); nothing
is retyped. The texts are the sections of the skill texts in lb_skills_build_daily.py, so rebuilding the skills from
this package through the generic pipeline must give skills_daily/shc-configure and skills_daily/shc-design byte for
byte. Import this module before numpy is loaded: notebook 14's fit depends on the BLAS thread count (lb_one_thread).
"""
import sys
from pathlib import Path

SC = Path(__file__).resolve().parents[2]
LB = SC.parent / "llm_band_sc"
for p in (str(SC), str(LB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import lb_one_thread  # noqa: E402,F401  first import: one thread, as the skill sandbox runs
import json  # noqa: E402

import lb_lib as lb                     # noqa: E402
import lb_spec_daily as S               # noqa: E402
import lb_skills_build_daily as B       # noqa: E402
import lb_extract_her14 as X            # noqa: E402
from scs_method import (Field, Check, Spec, CopyRule, ReferenceCode, Engine, Texts,  # noqa: E402
                        MethodPackage)

KIND = {"bwlist": "positive_list"}       # lb_spec_daily's list-of-bandwidths kind, by its generic name


def between(text, start, end):
    i = text.index(start) + len(start)
    return text[i:] if end is None else text[i:text.index(end, i)]


# ----------------------------------------------------------------------------- spec: notebook 14 settings and choices
def _fields(table):
    return {k: Field(KIND.get(kind, kind), allowed, meaning) for k, (kind, allowed, meaning) in table.items()}


def _band_subset(given, out, ctx, errors):
    sensor = ctx["sensor"]
    bands = [lb.disp(b) for b in lb.BANDS[sensor]]
    if out.get("band_weighting") == "subset":
        sub = given.get("band_subset")
        if not isinstance(sub, list) or not sub:
            errors.append("field 'band_subset': required as a non-empty list when band_weighting is subset")
        else:
            bad = [b for b in sub if b not in bands]
            if bad:
                errors.append(f"field 'band_subset': {bad} are not bands of {sensor} ({bands})")
            if len(set(sub)) != len(sub):
                errors.append("field 'band_subset': repeated band")
            out["band_subset"] = list(sub)
    elif "band_subset" in given and "band_weighting" in out and given["band_subset"] not in (None, []):
        errors.append("field 'band_subset': given, but band_weighting is not subset")


def _window(given, out, ctx, errors):
    n, n_given = len(ctx["to_predict"]), ctx["n_given"]
    if isinstance(out.get("m_min"), int) and isinstance(out.get("m_max"), int) and out["m_min"] > out["m_max"]:
        errors.append(f"fields 'm_min','m_max': m_min = {out['m_min']} exceeds m_max = {out['m_max']}")
    if isinstance(out.get("m_max"), int) and out["m_max"] + n > n_given:
        errors.append(f"fields 'm_max': m_max + {n} visits to predict = {out['m_max'] + n} exceeds the {n_given} "
                      f"visits given; no historical block would exist")
    if "m_max" in out and out["m_max"] is None and isinstance(out.get("m_min"), int) and out["m_min"] + n > n_given:
        errors.append(f"fields 'm_min','m_max': m_max is null and m_min + {n} visits to predict = {out['m_min'] + n} "
                      f"exceeds the {n_given} visits given; no historical block would exist")


def _readd(given, out, ctx, errors):
    if out.get("readd_seasonal") is True and out.get("fit_space") == "raw":
        errors.append("fields 'readd_seasonal','fit_space': readd_seasonal requires fit_space = latent_std")


def _intercept(given, out, ctx, errors):
    if out.get("intercept") == "fitted" and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'intercept','fit_space','readd_seasonal': intercept = fitted requires fit_space = raw or readd_seasonal = true")


def _constraint(given, out, ctx, errors):
    if out.get("weight_constraint") in ("nonneg", "unconstrained") and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'weight_constraint','fit_space','readd_seasonal': nonneg or unconstrained weights require fit_space = raw or readd_seasonal = true")


def _no_visit_to_predict(given, out, ctx, errors):
    names = [str(x) for pair in ctx["to_predict"] for x in pair]
    for k, v in given.items():
        text = json.dumps(v) + " " + str(k)
        hit = [x for x in names if x in text]
        if hit:
            errors.append(f"field '{k}': refers to {hit[0]}, a visit to predict, which is withheld")


SPEC = Spec(
    settings=_fields(S.CLASS1),
    choices=_fields(S.CLASS2),
    optional={"band_subset": (Field(*S.OPTIONAL["band_subset"]), "list of band names",
                              "<list of band names, only with subset>")},
    rules=list(S.CROSS_FIELD_RULES),
    configure_rules=list(S.CONFIGURE_RULES),
    checks=[Check(_band_subset, False), Check(_window, True), Check(_readd, False), Check(_intercept, False),
            Check(_constraint, False), Check(_no_visit_to_predict, True)],
    reference=dict(S.HER_SPEC),
    configure_label="shc-configure",
)


# ----------------------------------------------------------------------------- reference code: notebook 14 cells
def _header(sha):
    return ('"""GENERATED by lb_extract_her14.py from '
            'data/scripts/scripts/validation/shc/14_shc_daily_sentinel1_sentinel2.ipynb - DO NOT EDIT.\n'
            f'notebook sha256: {sha}\n'
            'CELL_SRC[i] holds the top-level statements of her cell i, each a byte copy sliced by ast line span, joined in\n'
            'notebook order; the manifest at the foot gives the kept and skipped line spans. lb_her14_run.py executes\n'
            'the strings in order in one namespace. Regenerate: python lb_extract_her14.py --write.\n'
            'Verify: python lb_extract_her14.py --check (gate DK1).\n"""\n')


def _parity():
    """The copied cells run in notebook order as notebook 14 runs them (lb_her14_run.run_cells), against what her
    notebook printed in the cells gate DK2 compares, In [6], In [9] and In [13] (effect tables), line for line (departure
    2 of the plan, Jun "a" 2026-10-30; numbers the same within 0.0001, departure 3, Jun 2026-10-31). A cell whose line
    count differs fails on its line count."""
    import lb_her14_run as H
    _, printed, _ = H.run_cells()
    hers = H.her_printed()
    nb, _ = X.notebook()
    for i in H.GATE_EXACT + H.GATE_0001:
        cell = f"notebook 14 In [{nb['cells'][i]['execution_count']}]"
        a, b = printed[i].splitlines(), hers[i].splitlines()
        yield cell, "line count", len(a), len(b)
        for k, (x, y) in enumerate(zip(a, b)):
            yield cell, f"line {k + 1}", x, y


REFERENCE_CODE = ReferenceCode(
    origin="collaborator", language="python",
    copy=CopyRule(mode="cells", source=X.NB, out=LB / "lb_her_nb14.py", cells=list(X.CELLS), skip=dict(X.SKIP),
                  header=_header),
    parity=lambda: _parity())


# ----------------------------------------------------------------------------- engine: the copied cells, else lb_fit_daily
def _reference(ctx, spec):
    import lb_fit_daily
    return lb_fit_daily.run_case(ctx["pre"], ctx["to_predict"], ctx["sensor"], spec, engine="her")


def _native(ctx, spec, forced):
    import lb_fit_daily
    return lb_fit_daily.run_case(ctx["pre"], ctx["to_predict"], ctx["sensor"], spec,
                                 engine="native" if forced else "auto")


ENGINE = Engine(reference=_reference, native=_native)


# ----------------------------------------------------------------------------- texts: sections of the daily skill texts
_A, _B = B.SKILL_A, B.SKILL_B
TEXTS = Texts(
    method_name="synthetic historical control",
    skill_prefix="shc",
    method_md=B.METHOD_MD,
    configure_description=between(_A, "description: ", "\n---\n"),
    design_description=between(_B, "description: ", "\n---\n"),
    configure_intro=between(_A, "# Configure and run the synthetic historical control\n\n", "\n\n## What is fixed"),
    design_intro=between(_B, "# Design and run the synthetic historical control\n\n", "\n\n## Everything"),
    fixed_choices=between(_A, "The fitter makes these choices for you: ", "\n\nYou decide the settings"),
    settings_list=between(_A, "in `settings.json`:\n\n", "\n\n`references/settings.md`"),
    settings_rules="`references/settings.md`" + between(_A, "\n\n`references/settings.md`", "\n\n## Workflow"),
    configure_workflow=between(_A, "## Workflow\n\n", None),
    design_overview=between(_B, "## Everything is yours to decide\n\n", "\n\n- `"),
    choices_list="- `" + between(_B, "\n\n- `", "\n\n`references/settings.md`"),
    design_rules="`references/settings.md`" + between(_B, "\n\n`references/settings.md`", "\n\n## Workflow"),
    design_workflow=between(_B, "## Workflow\n\n", None),
    scripts={"show_data.py": B.SHOW_DATA, "lb_lib.py": B.LB_LIB_SHIM},
    fit_script=B.FIT_PY,
    fit_script_name="fit.py",
    fixed_comment=B.FIXED_A_CODE.splitlines()[0],
    validate_configure=B.VALIDATE_A,
    validate_design=B.VALIDATE_B,
    authors="Chen, Yang and Yang",
    paper_title="Synthetic Historical Control for Policy Evaluation",
    profile_code="",
    profile_focus="Causal inference",
)

DAILY = lb.ROOT / "Satellite" / "data" / "daily_Sep2020_Sep2025_datasets"
PACKAGE = MethodPackage(
    name="shc_daily",
    paper=SC.parent / "skillfoundry_profile" / "work" / "gpt-5.4" / "paper" / "ssrn-4995085.pdf",
    reference_code=REFERENCE_CODE,
    data=[DAILY / "sentinel1" / "treatment.xlsx", DAILY / "sentinel2" / "treatment.xlsx"],
    spec=SPEC,
    engine=ENGINE,
    texts=TEXTS,
    skill_modules=[(m, LB / m) for m in B.BUNDLED],
)


# ----------------------------------------------------------------------------- the existing implementation, for the reproduction checks
def _contexts():
    import pandas as pd
    cases = pd.read_csv(LB / "lb_cases_daily.csv", parse_dates=["date"], float_precision="round_trip")
    out = []
    for sensor, site in cases[["sensor", "site_id"]].drop_duplicates().itertuples(index=False, name=None):
        g = cases.loc[cases["sensor"].eq(sensor) & cases["site_id"].eq(site)]
        pre = g.loc[g["part"].eq("pre"), ["band", "date", "y"]].reset_index(drop=True)
        post = g.loc[g["part"].eq("post")].drop_duplicates("period_id").sort_values("period_id")
        tp = [(p, d.strftime("%Y-%m-%d")) for p, d in zip(post["period_id"], post["date"])]
        out.append({"pre": pre, "to_predict": tp, "sensor": sensor, "site": site, "n_given": pre["date"].nunique()})
    return out


def _old_fit(ctx, spec):
    import lb_fit_daily
    return lb_fit_daily.run_case(ctx["pre"], ctx["to_predict"], ctx["sensor"], spec, engine="auto")


def _old_at_reference(spec):
    import lb_fit_daily
    return lb_fit_daily._matches_her_code_behaviour(spec)


OLD = {"validate": lambda given, ctx: S.validate_spec(given, ctx["sensor"], ctx["n_given"], ctx["to_predict"]),
       "settings_only": lambda given, ctx: S.class1_only(given, ctx["sensor"], ctx["n_given"], ctx["to_predict"]),
       "error": S.SpecError, "fit": _old_fit, "at_reference": _old_at_reference, "contexts": _contexts,
       "native_choices": {"fit_space": "raw"},
       "profile": {"request": SC.parent / "skillfoundry_profile" / "prompts" / "design_task.md", "runs": {}},
       "skills": LB / "skills_daily"}


# ----------------------------------------------------------------------------- the use of the SHC daily runs
def _use_cases():
    """The 52 cases of lb_cases_daily.csv; each case folder holds data.csv (the visits given: band, period_id, date,
    y) and visits_to_predict.csv (period_id, date), as lb_skill_run_daily.DailyCell wrote them."""
    import pandas as pd
    allc = pd.read_csv(LB / "lb_cases_daily.csv", parse_dates=["date"], float_precision="round_trip")
    res = []
    for ctx in _contexts():
        sensor, site = ctx["sensor"], ctx["site"]
        g = allc.loc[allc["sensor"].eq(sensor) & allc["site_id"].eq(site)]
        given = g.loc[g["part"].eq("pre"), ["band", "period_id", "date", "y"]].reset_index(drop=True)
        d = given.assign(band=[lb.disp(b) for b in given["band"]], date=given["date"].dt.strftime("%Y-%m-%d"))
        data = d.sort_values(["date", "band"])[["band", "period_id", "date", "y"]].to_csv(index=False)
        tp = pd.DataFrame(ctx["to_predict"], columns=["period_id", "date"]).to_csv(index=False)
        res.append({"prompt_id": f"daily_{lb.SHORT[sensor]}_{site}", "id": {"sensor": sensor, "site_id": site},
                    "files": {"data.csv": data, "visits_to_predict.csv": tp}, "ctx": ctx})
    return res


def _use_answer(case, a):
    """lb_skill_run_daily.DailyCell.answer: every band at every visit to predict."""
    bands = [lb.disp(b) for b in lb.BANDS[case["id"]["sensor"]]]
    pids = [p for p, _ in case["ctx"]["to_predict"]]
    if not isinstance(a, dict) or not isinstance(a.get("prediction"), dict):
        return None, 'answer.json must be an object with a "prediction" object and a "rationale" string'
    pred = a["prediction"]
    if any(p_ not in pred or not isinstance(pred[p_], dict) or any(b not in pred[p_] for b in bands) for p_ in pids):
        return None, ("answer.json: prediction must give, for every visit listed in visits_to_predict.csv under "
                      "its period_id, a number for every band of this parcel under the band names the parcel's "
                      "data uses")
    try:
        vals = {p_: {b: float(pred[p_][b]) for b in bands} for p_ in pids}
    except (TypeError, ValueError) as e:
        return None, f"answer.json: every prediction must be a number ({e})"
    if any(v != v for row in vals.values() for v in row.values()):
        return None, "answer.json: a prediction is NaN"
    return {"prediction": vals, "rationale": str(a.get("rationale", ""))}, None


def _task_text(skill_dir, case):
    import lb_skill_run_daily
    return lb_skill_run_daily.task_text_daily(skill_dir)


def _dry_answer(case, fit):
    pids = [p for p, _ in case["ctx"]["to_predict"]]
    return {"prediction": {p: {b: v[p] for b, v in fit["predictions"].items()} for p in pids}, "rationale": "dry run"}


def _fit_values(fit):
    """A daily fit's predictions {band: {period_id: value}} in the answer's shape {period_id: {band: value}}."""
    out = {}
    for b, pv in (fit.get("predictions") or {}).items():
        for p_, v in pv.items():
            out.setdefault(p_, {})[lb.disp(b)] = v
    return out


from scs_loop import Use  # noqa: E402

PROFILE_REQUEST = {          # the request of the SHC profile runs (skillfoundry_profile/prompts/design_task.md, row G3)
    "profile_task": "forecasting a land parcel's satellite band values",
    "profile_data": "`data.csv`, one parcel's own history of several bands, one row per band and time point.",
    "profile_answer": ("predicts every band's value at the time point after that history and writes `answer.json` as "
                       '{"prediction": {"<band>": <number>, ...}, "rationale": "<text>"}.')}

USES = {"shc_validation": Use(name="shc_validation", cases=_use_cases, task_text=_task_text, answer=_use_answer,
                              **PROFILE_REQUEST, fit_values=_fit_values,
                              dry_answer=_dry_answer)}


# ----------------------------------------------------------------------------- compare steps: the SHC daily pieces (lb_donors_reasoning.py)
_CASES = {}


def _steps_cases():
    import pandas as pd
    if "c" not in _CASES:
        _CASES["c"] = pd.read_csv(LB / "lb_cases_daily.csv")
    return _CASES["c"]


def _steps_records():
    import lb_donors_reasoning as D
    return {(skill, key): D.last_records(D.record_file("daily", skill, fname))
            for skill in D.SKILLS for key, fname in D.MODELS}


def _steps_case(rec):
    import lb_donors_reasoning as D
    return D.DailyCase(_steps_cases(), rec["sensor"], rec["site_id"])


def _steps_place(rec, case):
    import lb_donors_reasoning as D
    return D.place_fits(rec, "daily", case)[0]


def _steps_describe(fr, rec, case):
    spec = fr["spec"]
    mstar, maxd = fr["m_star"], spec["max_donors"]
    blocks = [case.block(i, mstar, maxd) for i in fr["donor_ids"]]
    return {"m": mstar, "n": spec.get("n", case.n), "blocks_available": len(case.starts(mstar, maxd)),
            "J": len(fr["donor_ids"]), "donor_ids": " ".join(map(str, fr["donor_ids"])),
            "weights": " ".join(f"{w:.4f}" for w in fr["weights"]), "max_weight": max(fr["weights"]),
            "blocks": "; ".join(b[0] for b in blocks),
            "block_dates": "; ".join(f"{b[1]}..{b[2]} -> {b[3]}..{b[4]}" for b in blocks),
            "keeps_most_recent": None, "selection_mode": fr.get("selection_mode"), "fit_mse": fr["fit_mse"]}


def _steps_reference_rows():
    """Notebook 14's Table 1 as printed, with dates (lb_donors_reasoning departure 2)."""
    import lb_donors_reasoning as D
    return D.notebook14_rows(_steps_cases())


def _steps_spec_columns():
    import lb_donors_reasoning as D
    return D.SETTINGS["daily"] + D.CHOICES + ["band_subset"]


STEPS = {"data": "daily", "records": _steps_records, "case": _steps_case, "place": _steps_place,
         "describe": _steps_describe, "reference_rows": _steps_reference_rows, "spec_columns": _steps_spec_columns(),
         "reference_marker": "HER_SPEC",
         "column_names": {"saw_reference": "saw_notebook_settings", "settings_equal_reference": "settings_equal_notebook"},
         "old_tables": {"donors": LB / "lb_donors_simple_daily.csv", "reasoning": LB / "lb_answer_reasoning_simple.csv"},
         "old_match": LB / "lb_match_skill_daily.csv"}
