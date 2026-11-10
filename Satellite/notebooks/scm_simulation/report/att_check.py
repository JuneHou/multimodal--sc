"""Reported ATT against the answer's own counterfactual, for every answer at every noise level.

For each answer: the ATT it reported, the mean over periods 41..60 of the treated unit's observed outcome minus the
answer's own counterfactual, and how the reported ATT reached answer.json. The command read is the last one that
writes answer.json:
    profile script     a profile answer, written by the skill's script
    typed              the command writes "att": <number>, a number the model typed
    computed in code   otherwise; every such command computes the ATT from gaps or from observed minus counterfactual
A typed ATT is also searched, as text, in the output of every earlier step. Where it appears in the fitter's table
(period, observed, synthetic control, gap), the period of that line and its column are recorded.
An ATT differs from its counterfactual when the two are more than her_tables.SAME (0.0001) apart.

    python report/att_check.py   writes results/sigma_<s>/att_check.csv for every noise level and prints the summary
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]                 # notebooks/scm_simulation
sys.path.insert(0, str(HERE))
import her_tables as h  # noqa: E402

WRITES = re.compile(r'>\s*answer\.json|open\(\s*["\']answer\.json["\']\s*,\s*["\']w|--out\s+answer\.json'
                    r'|answer\.json["\']?\s*,\s*["\']w|write_json|toJSON|writeLines')
TYPED = re.compile(r'"att"\s*:\s*(-?\d[\d.eE+-]*)')
COLUMNS = ("period", "observed", "synthetic control", "gap")


def where_printed(text, steps, before):
    """(step, period, column) of the first fitter-table line before step `before` that shows `text`, else None;
    (step, None, None) when the text appears elsewhere in an earlier output."""
    for st in steps:
        if st["step"] >= before or text not in (st.get("output") or ""):
            continue
        for line in st["output"].splitlines():
            f = line.split()
            if len(f) == 4 and f[0].isdigit():
                hit = [text in x for x in f[1:]]
                if any(hit):
                    return st["step"], int(f[0]), COLUMNS[1 + hit.index(True)]
        return st["step"], None, None
    return None


def rows(sigma):
    obs = {c: h.observed_and_truth(c, sigma)[0] for c in ("dgp1", "dgp2")}
    out = []
    for arm, r in h.answers(sigma):
        if not r["ok"]:
            continue
        cf = {int(k): v for k, v in r["answer"]["prediction"]["counterfactual"].items()}
        own = sum(obs[r["case"]][t] - cf[t] for t in range(41, 61)) / 20
        w = [st for st in r["steps"] if isinstance(st.get("command"), dict)
             and "answer.json" in (st["command"].get("text") or "") and WRITES.search(st["command"]["text"])][-1]
        text = w["command"]["text"]
        m = TYPED.search(text)
        if arm in h.PROFILE_SPECS:
            assert "scripts/" in text and not m, (sigma, arm, r["model"], r["case"])
            how = "profile script"
        else:
            how = "typed" if m else "computed in code"
        seen = where_printed(m.group(1).rstrip(".,"), r["steps"], w["step"]) if m else None
        att = r["answer"]["att"]
        out.append({"sigma": sigma, "arm": arm, "model": r["model"], "case": r["case"], "answer_step": w["step"],
                    "how": how, "reported_att": att, "counterfactual_att": own, "difference": att - own,
                    "differs": abs(att - own) > h.SAME,
                    "typed_seen_at_step": seen[0] if seen else None,
                    "typed_seen_period": seen[1] if seen else None,
                    "typed_seen_column": seen[2] if seen else None})
    return pd.DataFrame(out)


def main():
    allr = []
    for sigma in h.mod.SIGMAS:
        t = rows(sigma)
        t.to_csv(h.mod.results_folder(sigma) / "att_check.csv", index=False)
        allr.append(t)
    t = pd.concat(allr, ignore_index=True)
    pd.set_option("display.width", 200)
    print(t.groupby(["how", "model"]).agg(answers=("differs", "size"), differ=("differs", "sum")).to_string())
    print(f"\n{len(t)} answers; differ: {int(t.differs.sum())}")
    print(t[t.differs].drop(columns=["arm"]).to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
