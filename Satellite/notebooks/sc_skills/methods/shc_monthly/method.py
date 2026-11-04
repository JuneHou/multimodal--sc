"""SHC monthly method package: the collaborator's notebook 10 SHC as the existing simple skills use it.

Every value points at an existing file of notebooks/llm_band_sc/ (or the paper copy of the profile experiment); nothing
is retyped. The texts are the sections of the approved skill texts in lb_skills_build.py, so rebuilding the skills
from this package through the generic pipeline must give skills/shc-configure and skills/shc-design byte for byte.
"""
import ast
import json
import sys
from pathlib import Path

SC = Path(__file__).resolve().parents[2]
LB = SC.parent / "llm_band_sc"
for p in (str(SC), str(LB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import lb_lib as lb                     # noqa: E402
import lb_spec                          # noqa: E402
import lb_skills_build as B             # noqa: E402
import lb_extract_her as X              # noqa: E402
from scs_method import (Field, Check, Spec, CopyRule, ReferenceCode, Engine, Texts,  # noqa: E402
                        MethodPackage)


def between(text, start, end):
    """The text strictly between `start` and `end`; `end` None means to the end."""
    i = text.index(start) + len(start)
    return text[i:] if end is None else text[i:text.index(end, i)]


# ----------------------------------------------------------------------------- spec: notebook 10 settings and choices
def _fields(table):
    return {k: Field(kind, allowed, meaning) for k, (kind, allowed, meaning) in table.items()}


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


def _m_plus_n(given, out, ctx, errors):
    n_train = ctx.get("n_train", lb_spec.N_TRAIN)
    if "m" in out and "n" in out and out["m"] + out["n"] > n_train:
        errors.append(f"fields 'm','n': m + n = {out['m'] + out['n']} exceeds {n_train}; no historical block would exist")


def _readd(given, out, ctx, errors):
    if out.get("readd_seasonal") is True and out.get("fit_space") == "raw":
        errors.append("fields 'readd_seasonal','fit_space': readd_seasonal requires fit_space = latent_std")


def _intercept(given, out, ctx, errors):
    if out.get("intercept") == "fitted" and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'intercept','fit_space','readd_seasonal': intercept = fitted requires fit_space = raw or readd_seasonal = true")


def _constraint(given, out, ctx, errors):
    if out.get("weight_constraint") in ("nonneg", "unconstrained") and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'weight_constraint','fit_space','readd_seasonal': nonneg or unconstrained weights require fit_space = raw or readd_seasonal = true")


def _no_p10(given, out, ctx, errors):
    for k, v in given.items():
        if "P10" in json.dumps(v) or "P10" in str(k):
            errors.append(f"field '{k}': refers to P10, which is withheld")


SPEC = Spec(
    settings=_fields(lb_spec.CLASS1),
    choices=_fields(lb_spec.CLASS2),
    optional={"band_subset": (Field(*lb_spec.OPTIONAL["band_subset"]), "list of band names",
                              "<list of band names, only with subset>")},
    rules=list(lb_spec.CROSS_FIELD_RULES),
    configure_rules=[0, len(lb_spec.CROSS_FIELD_RULES) - 1],
    checks=[Check(_band_subset, False), Check(_m_plus_n, True), Check(_readd, False), Check(_intercept, False),
            Check(_constraint, False), Check(_no_p10, True)],
    reference=dict(lb_spec.HER_SPEC),
    configure_label="arm A",
)


# ----------------------------------------------------------------------------- reference code: notebook 10 functions
def _cell7(tree, lines):
    """Her derived-column statements: every top-level Assign after the `if feature_panel.empty:` guard and before the
    first print() (lb_extract_her.slices, kind cell7)."""
    guard = next(n for n in tree.body if isinstance(n, ast.If) and "empty" in ast.dump(n.test))
    first_print = next(n.lineno for n in tree.body
                       if isinstance(n, ast.Expr) and n.lineno > guard.end_lineno
                       and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "print")
    return [n for n in tree.body if isinstance(n, ast.Assign) and guard.end_lineno < n.lineno < first_print]


def _blocks():
    out = []
    for cell, kind, names in X.BLOCKS:
        if kind == "cell7":
            out.append({"cell": cell, "kind": "cell7", "name": "CELL7_DERIVED", "const": "CELL7_DERIVED_SRC",
                        "select": _cell7})
        else:
            out.append((cell, kind, names))
    return out


def _header(sha):
    return ('"""GENERATED by lb_extract_her.py from data/shc/10_shc_monthly_multiple_outcomes.ipynb - DO NOT EDIT.\n'
            f'notebook sha256: {sha}\n'
            'Each block is a byte copy of the named statement(s) of the named notebook cell, sliced by ast line span;\n'
            'comments between statements are not carried. Regenerate: python lb_extract_her.py --write.\n'
            'Verify: python lb_extract_her.py --check (gate G1). Manifest at the foot.\n"""\n')


PRINTED = "joint standardized P10 error="          # notebook 10 In [14]: f"  OK {site_id}: joint standardized P10 error={joint_nmse:.6f}"


def _printed():
    """{(sensor, site): text} from what notebook 10 In [14] printed: '<sensor>: N sites', then one OK line per site."""
    import re
    nb = json.loads(Path(lb.NB26).read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and PRINTED in "".join(c["source"])]
    assert len(cells) == 1 and cells[0]["execution_count"] == 14, "expected one cell, In [14], printing the error"
    text = "".join("".join(o["text"]) for o in cells[0]["outputs"] if o.get("name") == "stdout")
    out, sensor = {}, None
    for line in text.splitlines():
        m = re.match(r"^(sentinel[12]): \d+ sites", line)
        if m:
            sensor = m.group(1)
        m = re.match(r"^\s+OK (treatment_\d{4}): " + re.escape(PRINTED) + r"(\S+)$", line)
        if m:
            out[(sensor, m.group(1))] = m.group(2)
    return out


def _parity():
    """The copied code at notebook 10's values (her validation function, lb_her_run.heldout_cell, on the monthly
    data), each site's validation error written as her In [14] prints it, against the text notebook 10 printed
    (departure 1 of the plan, Jun "A" 2026-10-30)."""
    import lb_her_panel
    import lb_her_run
    panel = lb_her_panel.her_feature_panel(lb.FEATURES_CSV)
    for (sensor, site), printed in _printed().items():
        r = lb_her_run.heldout_cell(panel, sensor, site)
        yield f"{sensor} {site}", "validation error, In [14]", f"{r['joint_validation_nmse']:.6f}", printed


REFERENCE_CODE = ReferenceCode(
    origin="collaborator", language="python",
    copy=CopyRule(mode="functions", source=lb.NB26, out=LB / "lb_her_nb10.py", blocks=_blocks(), header=_header,
                  imports="import numpy as np\nimport pandas as pd\nfrom scipy.optimize import minimize\n"),
    parity=lambda: _parity())


# ----------------------------------------------------------------------------- engine: the copied code, else lb_fit
def _reference(ctx, spec):
    import lb_fit
    return lb_fit.run_cell(ctx["panel"], ctx["sensor"], ctx["site"], spec, engine="her")


def _native(ctx, spec, forced):
    import lb_fit
    return lb_fit.run_cell(ctx["panel"], ctx["sensor"], ctx["site"], spec, engine="native" if forced else "auto")


ENGINE = Engine(reference=_reference, native=_native)


# ----------------------------------------------------------------------------- texts: sections of the approved skill texts
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

PACKAGE = MethodPackage(
    name="shc_monthly",
    paper=SC.parent / "skillfoundry_profile" / "work" / "gpt-5.4" / "paper" / "ssrn-4995085.pdf",
    reference_code=REFERENCE_CODE,
    data=[lb.FEATURES_CSV],
    spec=SPEC,
    engine=ENGINE,
    texts=TEXTS,
    skill_modules=[(m, LB / m) for m in B.BUNDLED],
)


# ----------------------------------------------------------------------------- the existing implementation, for the reproduction checks
def _contexts():
    import lb_guard
    panel = lb_guard.training_panel()
    sites = {s: sorted(panel.loc[panel.sensor == s, "site_id"].unique()) for s in lb.SENSORS}
    return [{"panel": panel, "sensor": s, "site": site} for s in lb.SENSORS for site in sites[s]]


def _old_fit(ctx, spec):
    import lb_fit
    return lb_fit.run_cell(ctx["panel"], ctx["sensor"], ctx["site"], spec, engine="auto")


def _old_at_reference(spec):
    import lb_fit
    return lb_fit._matches_her_code_behaviour(spec)


OLD = {"validate": lambda given, ctx: lb_spec.validate_spec(given, ctx["sensor"]),
       "settings_only": lambda given, ctx: lb_spec.class1_only(given, ctx["sensor"]),
       "error": lb_spec.SpecError, "fit": _old_fit, "at_reference": _old_at_reference, "contexts": _contexts,
       "native_choices": {"selection": "exhaustive", "fit_space": "raw"},
       "profile": {"request": SC.parent / "skillfoundry_profile" / "prompts" / "design_task.md",
                   "runs": ("gpt-5.4", "deepseek"), "root": SC.parent / "skillfoundry_profile"},
       "skills": LB / "skills"}


# ----------------------------------------------------------------------------- the use of the SHC monthly runs
def _use_cases():
    """The 27 cells of lb_prompts_shc.jsonl; each case folder holds data.csv as lb_skill_loop.Cell wrote it (P01 to
    P09: band, period_id, month, mean_value)."""
    import lb_guard
    panel = lb_guard.training_panel()
    out = []
    for p in map(json.loads, open(LB / "lb_prompts_shc.jsonl")):
        sensor, site = p["sensor"], p["target_site"]
        cell = lb_guard.cell_panel(panel, sensor, site)
        d = cell.assign(band=[lb.disp(b) for b in cell["band"]], seq=cell["period_id"].str[1:].astype(int))
        csv = d.sort_values(["seq", "band"])[["band", "period_id", "month", "mean_value"]].to_csv(index=False)
        out.append({"prompt_id": p["prompt_id"][:-4], "id": {"sensor": sensor, "site_id": site},
                    "files": {"data.csv": csv}, "ctx": {"panel": panel, "sensor": sensor, "site": site, "n_train": 9}})
    return out


def _use_answer(case, a):
    """lb_skill_loop.Cell.answer: a number for every band, under the band names the data uses."""
    bands = [lb.disp(b) for b in lb.BANDS[case["id"]["sensor"]]]
    if not isinstance(a, dict) or not isinstance(a.get("prediction"), dict):
        return None, 'answer.json must be an object with a "prediction" object and a "rationale" string'
    pred = a["prediction"]
    if any(b not in pred for b in bands):
        return None, ("answer.json: prediction must give a number for every band of this parcel, "
                      "under the band names the parcel's data uses")
    try:
        vals = {b: float(pred[b]) for b in bands}
    except (TypeError, ValueError) as e:
        return None, f"answer.json: every prediction must be a number ({e})"
    if any(v != v for v in vals.values()):
        return None, "answer.json: a prediction is NaN"
    return {"prediction": vals, "rationale": str(a.get("rationale", ""))}, None


def _task_text(skill_dir, case):
    import lb_skill_loop
    return lb_skill_loop.task_text(skill_dir)


from scs_loop import Use  # noqa: E402

PROFILE_REQUEST = {          # the request of the SHC profile runs (skillfoundry_profile/prompts/design_task.md, row G3)
    "profile_task": "forecasting a land parcel's satellite band values",
    "profile_data": "`data.csv`, one parcel's own history of several bands, one row per band and time point.",
    "profile_answer": ("predicts every band's value at the time point after that history and writes `answer.json` as "
                       '{"prediction": {"<band>": <number>, ...}, "rationale": "<text>"}.')}

USES = {"shc_validation": Use(name="shc_validation", cases=_use_cases, task_text=_task_text, answer=_use_answer,
                              **PROFILE_REQUEST, fit_values=lambda fit: fit.get("predictions") or {},
                              out_limit=6000,
                              dry_answer=lambda case, fit: {"prediction": fit["predictions"], "rationale": "dry run"})}


# ----------------------------------------------------------------------------- compare steps: the SHC monthly pieces (lb_donors_reasoning.py)
def _steps_records():
    import lb_donors_reasoning as D
    return {(skill, key): D.last_records(D.record_file("monthly", skill, fname))
            for skill in D.SKILLS for key, fname in D.MODELS}


def _steps_place(rec, case):
    import lb_donors_reasoning as D
    return D.place_fits(rec, "monthly", case)[0]


def _steps_describe(fr, rec, case):
    import lb_donors_reasoning as D
    spec = fr["spec"]
    n_blocks = 9 - spec["m"] - spec["n"] + 1
    return {"m": spec["m"], "n": spec.get("n"), "blocks_available": n_blocks, "J": len(fr["donor_ids"]),
            "donor_ids": " ".join(map(str, fr["donor_ids"])), "weights": " ".join(f"{w:.4f}" for w in fr["weights"]),
            "max_weight": max(fr["weights"]),
            "blocks": "; ".join(D.monthly_block(i, spec["m"], spec["n"]) for i in fr["donor_ids"]), "block_dates": "",
            "keeps_most_recent": n_blocks in fr["donor_ids"], "selection_mode": fr.get("selection_mode"),
            "fit_mse": fr["fit_mse"]}


def _steps_reference_rows():
    """Notebook 10's block per case (lb_donors_reasoning departure 3)."""
    import pandas as pd
    import lb_donors_reasoning as D
    nb10 = D.notebook10_blocks()
    return pd.DataFrame([{"data": "monthly", "skill": "notebook 10", "model": "", "sensor": x.sensor, "site_id": x.site_id,
                          "m": 4, "n": 1, "blocks_available": 5, "J": 1,
                          "donor_ids": "" if pd.isna(x.block) else str(int(x.block)),
                          "blocks": "not identified" if pd.isna(x.block) else D.monthly_block(int(x.block), 4, 1),
                          "keeps_most_recent": None if pd.isna(x.block) else int(x.block) == 5,
                          "printed_error": x.printed_error} for x in nb10.itertuples()])


def _steps_spec_columns():
    import lb_donors_reasoning as D
    return D.SETTINGS["monthly"] + D.CHOICES + ["band_subset"]


STEPS = {"data": "monthly", "records": _steps_records, "place": _steps_place, "describe": _steps_describe,
         "reference_rows": _steps_reference_rows, "spec_columns": _steps_spec_columns(), "reference_marker": "HER_SPEC",
         "column_names": {"saw_reference": "saw_notebook_settings", "settings_equal_reference": "settings_equal_notebook"},
         "old_tables": {"donors": LB / "lb_donors_simple_monthly.csv", "reasoning": LB / "lb_answer_reasoning_simple.csv"}}
