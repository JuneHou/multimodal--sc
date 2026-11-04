"""The specification behind every answer, in words (the specifications slide of the SCM report deck).

The fit behind an answer is her_tables.fit_behind: the fit scs_steps matches to a configure or design answer, else the
recorded fit closest to it, and for a profile answer the skill script's fixed specification. The two reference
specifications (README Section 3 and the ADH paper's) come first, from method.README_WAY and method.ADH_WAY.

    python report/specs.py      writes results/specs_answers.csv and prints it
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import her_tables as h  # noqa: E402

ORDER_ARM = ["README Section 3", "ADH paper specification", "scm-configure", "scm-design", "profile-gpt-5.4",
             "profile-deepseek"]
ORDER_MODEL = ["", "gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low", "openai:gpt-5.5", "all four models"]


def span(periods):
    p = [int(x) for x in periods]
    if len(p) == 1:
        return f"{p[0]}"
    if p == list(range(p[0], p[-1] + 1)):
        return f"{p[0]}–{p[-1]}"
    return ", ".join(str(x) for x in p)


def describe(spec):
    """Predictors in words, in the order Synth builds them: predictors first, then special predictors."""
    parts = []
    if spec["predictors"]:
        parts.append(f"{', '.join(spec['predictors'])} {spec['predictors_op']} over {span(spec['time_predictors_prior'])}")
    sp = spec["special_predictors"]
    i = 0
    while i < len(sp):
        s = sp[i]
        if s["variable"] == "Y_observed" and len(s["periods"]) == 1:
            j = i
            while j + 1 < len(sp) and sp[j + 1]["variable"] == "Y_observed" and len(sp[j + 1]["periods"]) == 1:
                j += 1
            ts = [x["periods"][0] for x in sp[i:j + 1]]
            parts.append(f"Y_observed at each of {span(ts)}" if len(ts) > 2 and span(ts).count("–")
                         else f"Y_observed at {span(ts)}")
            i = j + 1
            continue
        if s["variable"] == "Y_observed":
            j = i
            while j + 1 < len(sp) and sp[j + 1]["variable"] == "Y_observed" and len(sp[j + 1]["periods"]) > 1 \
                    and sp[j + 1]["op"] == s["op"]:
                j += 1
            parts.append(f"Y_observed {s['op']} over " + ", ".join(span(x["periods"]) for x in sp[i:j + 1]))
            i = j + 1
            continue
        j = i
        while j + 1 < len(sp) and sp[j + 1]["variable"] != "Y_observed" and sp[j + 1]["periods"] == s["periods"] \
                and sp[j + 1]["op"] == s["op"]:
            j += 1
        names = ", ".join(x["variable"] for x in sp[i:j + 1])
        parts.append(f"{names} {s['op']} over {span(s['periods'])}" if len(s["periods"]) > 1
                     else f"{names} at {s['periods'][0]}")
        i = j + 1
    n = len(spec["predictors"]) + len(sp)
    return n, " · ".join(parts)


def row(arm, model, case, spec, note):
    n, words = describe(spec)
    v = "searched" if spec.get("custom_v") is None else ("equal" if len(set(spec["custom_v"])) == 1 else "given")
    return {"arm": arm, "model": model, "case": case, "fit": note, "n_predictors": n, "predictors": words,
            "V": v, "optimizers": " ".join(spec["optimxmethod"]) if v == "searched" else "",
            "time_optimize_ssr": span(spec["time_optimize_ssr"]) if v == "searched" else "",
            "donors": len(spec["controls_identifier"]), "weights": spec.get("weights", "convex"),
            "interpolation_penalty": spec.get("interpolation_penalty", "none")}


def main():
    readme = dict(h.mod.README_WAY)
    rows = [row("README Section 3", "", "dgp1 and dgp2", readme, "README Section 3, computed from its formulas"),
            row("ADH paper specification", "", "dgp1 and dgp2", h.adh_spec(), "the ADH paper's choices on this data")]
    seen = set()
    for arm, r in h.answers():
        if not r["ok"]:
            continue
        spec, _, note = h.fit_behind(arm, r)
        if arm.startswith("profile"):
            if (arm, r["case"]) in seen:
                continue
            seen.add((arm, r["case"]))
            rows.append(row(arm, "all four models", r["case"], spec, note))
        else:
            rows.append(row(arm, r["model"], r["case"], spec, note))
    df = pd.DataFrame(rows)
    df["_a"] = df.arm.map(ORDER_ARM.index)
    df["_m"] = df.model.map(ORDER_MODEL.index)
    df = df.sort_values(["_a", "_m", "case"]).drop(columns=["_a", "_m"])
    df.to_csv(HERE / "results" / "specs_answers.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
