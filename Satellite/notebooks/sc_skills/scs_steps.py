"""scs_steps — compare each answer's steps with the reference code's (the compare-steps stage of every pipeline).

Configure and design (as llm_band_sc/lb_donors_reasoning.py does, without the search path):
    the fit behind the answer   the last stored fit whose predictions, each rounded to the decimals of the answer value
                                at the same place, equal the answer, over the same places (Use.fit_values gives a fit's
                                predictions in the answer's shape)
    donors table                that fit's engine, donors and weights (and the method's own columns), whether the record
                                shows the reference values, whether its settings equal them, and every field of its spec
    weights table               that fit's donors and weights beside the reference code's on the same case
    reasoning table             reasoning text and replies from the step that wrote the fit's file to the answer step,
                                and the rationale; "not logged" for a model whose records hold no reasoning text at all
    logged table                per skill and model: records, steps, steps with reasoning text, median length
Profile (README row G12 of the profile experiment): for each reference step, what the generator's scripts do and what
each solver's code does; written as a table to fill from the skill and the programs (profile_table).

The method's own pieces come from the module's STEPS hooks: data label, records, placement of fits at steps, the
method's columns, the reference rows, the spec columns, the reference marker and the column names.

    python scs_steps.py <method> <out folder>
"""
import json
import statistics
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

import scs_method


def ndec(v):
    return max(0, -Decimal(repr(v)).as_tuple().exponent)


def flat(d, prefix=()):
    """{a: {b: v}} -> {(a, b): v}."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def matches(fit_values, values):
    f, a = flat(fit_values), flat(values)
    if set(f) != set(a):
        return False
    return all(round(float(f[k]), ndec(a[k])) == a[k] for k in a)


def matched_run(rec, use):
    """1-based index of the fit behind the answer, or None."""
    if not rec.get("ok"):
        return None
    hit = [i for i, fr in enumerate(rec["fit_runs"]) if matches(use.fit_values(fr), rec["values"])]
    return hit[-1] + 1 if hit else None


def cmd_text(step):
    c = step.get("command") or {}
    return (c.get("text") or c.get("message") or "") if isinstance(c, dict) else str(c)


def reasoning(step):
    rs = (step.get("usage") or {}).get("reasoning")
    return rs if isinstance(rs, str) else ""


def same(a, b):
    if isinstance(a, dict) or isinstance(b, dict):      # a predictor entry {variable, periods, op}
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a is b or a == b and type(a) is type(b)
    return float(a) == float(b)


def show(v):
    return json.dumps(v) if not isinstance(v, str) else v


def default_place(rec, case):
    return [fr.get("step") for fr in rec["fit_runs"]]


def default_describe(fit, rec, case):
    w = [float(x) for x in fit["weights"]]
    return {"J": len(fit["donor_ids"]), "donor_ids": " ".join(map(str, fit["donor_ids"])),
            "weights": " ".join(f"{x:.4f}" for x in w), "max_weight": max(w), "fit_mse": fit.get("fit_mse")}


def tables(pkg, mod, use, hooks):
    """-> {"donors", "reasoning", "logged", "matched"} DataFrames for one method's records."""
    S = pkg.spec
    data = hooks["data"]
    recs = hooks["records"]()                            # {(skill, model key): {prompt_id: record}}
    place = hooks.get("place", default_place)
    describe = hooks.get("describe", default_describe)
    case_of = hooks.get("case", lambda rec: None)
    spec_cols = hooks.get("spec_columns", list(S.settings) + list(S.choices) + list(S.optional))
    marker = hooks.get("reference_marker")
    names = {"saw_reference": "saw_reference_values", "settings_equal_reference": "settings_equal_reference",
             **hooks.get("column_names", {})}
    logged_models = {key for (skill, key), d in recs.items()
                     if any(reasoning(s) for r in d.values() for s in r["steps"])}
    logged, matched, donors, reason = [], [], [], []
    for (skill, key), d in recs.items():
        steps_all = [s for r in d.values() for s in r["steps"]]
        lens = [len(reasoning(s)) for s in steps_all if reasoning(s)]
        logged.append({"data": data, "skill": skill, "model": key, "records": len(d), "steps": len(steps_all),
                       "steps_with_reasoning": len(lens), "median_chars": statistics.median(lens) if lens else None})
        for r in d.values():
            k1 = matched_run(r, use)
            matched.append({"data": data, "skill": skill, "model": key, "prompt_id": r["prompt_id"],
                            "ok": bool(r["ok"]), "ended": r.get("ended"), "n_fit_runs": len(r["fit_runs"]),
                            "matched_run": k1})
            if k1 is None:
                continue
            case = case_of(r)
            steps = place(r, case)
            k = k1 - 1
            fr = r["fit_runs"][k]
            spec = fr["spec"]
            saw = any(marker in (s.get("output") or "") for s in r["steps"]) if marker else None
            donors.append({"data": data, "skill": skill, "model": key, "sensor": r.get("sensor"),
                           "site_id": r.get("site_id"), "prompt_id": r["prompt_id"], "matched_run": k1,
                           "engine": fr.get("engine_used"), **describe(fr, r, case),
                           names["saw_reference"]: saw,
                           names["settings_equal_reference"]: all(same(spec.get(f), S.reference[f]) for f in S.settings),
                           **{f"spec_{f}": show(spec.get(f)) for f in spec_cols}})
            # the reasoning when the answer was made: from the step that wrote the fit's file to the answer step
            fnames = {fr["file"], Path(fr["file"]).name}
            cmds = [cmd_text(s) for s in r["steps"]]
            ans = max((i for i, c in enumerate(cmds) if "answer.json" in c), default=None)
            fit_at = next((i for i, s in enumerate(r["steps"]) if s["step"] == steps[k]), None)
            upto = fit_at if fit_at is not None else ans
            start = max((i for i in range(0, (upto if upto is not None else -1) + 1) if any(n in cmds[i] for n in fnames)),
                        default=upto)
            end = ans if ans is not None else upto
            win = r["steps"][start:end + 1] if start is not None and end is not None and end >= start else []
            nl = key not in logged_models
            reason.append({"data": data, "skill": skill, "model": key, "sensor": r.get("sensor"),
                           "site_id": r.get("site_id"), "prompt_id": r["prompt_id"], "matched_run": k1,
                           "fit_step": steps[k], "window": f"{win[0]['step']}-{win[-1]['step']}" if win else "",
                           "reasoning": "not logged" if nl else "\n".join(f"[step {s['step']}] {reasoning(s)}"
                                                                          for s in win if reasoning(s)),
                           "replies": "\n".join(f"[step {s['step']}] {s.get('reply') or ''}" for s in win),
                           "rationale": r.get("rationale") or "",
                           "reasoning_chars": 0 if nl else sum(len(reasoning(s)) for s in win)})
    ref_rows = hooks["reference_rows"]() if "reference_rows" in hooks else pd.DataFrame()
    return {"donors": pd.concat([ref_rows, pd.DataFrame(donors)], ignore_index=True), "reasoning": pd.DataFrame(reason),
            "logged": pd.DataFrame(logged), "matched": pd.DataFrame(matched)}


def weights_table(pkg, use, donors, contexts):
    """Each answer's donors and weights beside the reference code's at the reference values on the same case."""
    import scs_fit
    import scs_spec
    ref = {}
    rows = []
    for _, x in donors[donors["prompt_id"].notna()].iterrows():
        key = (x["sensor"], x["site_id"])
        if key not in ref:
            ctx = contexts[key]
            spec = scs_spec.validate(pkg.spec, dict(pkg.spec.reference), ctx)
            ref[key] = scs_fit.run(pkg, ctx, spec, engine="reference")
        rf = ref[key]
        a = dict(zip(str(x["donor_ids"]).split(), map(float, str(x["weights"]).split())))
        b = {str(i): float(w) for i, w in zip(rf["donor_ids"], rf["weights"])}
        rows.append({"data": x["data"], "skill": x["skill"], "model": x["model"], "sensor": x["sensor"],
                     "site_id": x["site_id"], "prompt_id": x["prompt_id"], "donor_ids": x["donor_ids"],
                     "weights": x["weights"], "reference_donor_ids": " ".join(b),
                     "reference_weights": " ".join(f"{w:.4f}" for w in b.values()), "same_donors": set(a) == set(b),
                     "max_abs_weight_difference": max(abs(a.get(i, 0.0) - b.get(i, 0.0)) for i in set(a) | set(b))})
    return pd.DataFrame(rows)


def profile_table(pkg, steps_of_method, generators=("gpt-5.4", "deepseek")):
    """README row G12: one row per reference step; the columns to fill are what each generator's scripts do and what
    each solver model's code does, next to the scores."""
    return pd.DataFrame([{"reference step": s, **{f"{g} scripts": "" for g in generators}} for s in steps_of_method])


def main(method, out):
    mod = scs_method.load_module(method)
    pkg, use = mod.PACKAGE, next(iter(mod.USES.values()))
    t = tables(pkg, mod, use, mod.STEPS)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in t.items():
        df.to_csv(out / f"steps_{name}.csv", index=False)
    print(f"{method}: {len(t['donors'])} donor rows, {len(t['reasoning'])} reasoning rows -> {out}")
    return t


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
