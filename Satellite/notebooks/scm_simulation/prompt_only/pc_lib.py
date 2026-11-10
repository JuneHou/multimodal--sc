"""Language models asked by prompt alone for the counterfactual of a simulated disaster, without and with 29 untreated
donors (plan ~/.claude/plans/use-nature-skills-analyze-wiggly-coral.md, top section, 2026-11-06).

Shared pieces: the two prompts, the answer parser, the five rules of the step analysis, the collaborator's README
Section 5 cells as printed, and scoring through ../score.py (the ATT from a counterfactual, mc_evaluation_r.R line 94).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                    # notebooks/scm_simulation/prompt_only
SIM = HERE.parent                                         # notebooks/scm_simulation
sys.path.insert(0, str(SIM))
import score                                              # noqa: E402  loads the scm package as score.mod

mod = score.mod
SIGMAS = mod.SIGMAS                                       # ("1.0", "0.5", "0.2")
DGPS = ("dgp1", "dgp2")
METHODS = ("without_donors", "with_donors")
MODELS = ("gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low")   # Jun "replace GPT-5.5 with deepseek first"
EFFORTS = ("default", "low")
assert mod.FIRST_TREATED == 41
PRE = list(range(1, 41))
POST = list(range(41, 61))
PROMPTS = HERE / "prompts.jsonl"
RESULTS = HERE / "results"

HEAD = ("This is a vegetation index time series of a target site, periods 1 to 40.\n"
        "There is a natural disaster. It hits the target at period 41.\n"
        "We want to know the counterfactual state of the target and how large the impact of the disaster is.\n")
DONOR_LINE = "We also have 29 control sites as donors, which have no natural disaster.\n"
ASK = ("Generate the counterfactual state of the target for each period from 41 to 60, that is, the vegetation index "
       "the target would have if the disaster had not happened.\n"
       "End your reply with this line:\n"
       "counterfactual: 41 <value>, 42 <value>, ..., 60 <value>")
RULE_NAMES = {"pre_mean": "the target's mean over periods 1 to 40",
              "last_value": "the target's value at period 40",
              "linear_trend": "the least squares line through the target's periods 1 to 40",
              "did_donor_mean": "the donor mean plus the target's pre-disaster mean minus the donors' pre-disaster mean",
              "readme_sc": "the README Section 3 synthetic control on the same panel"}
README_SC_PATHS = RESULTS / "readme_sc_paths.csv"            # pc_readme_sc.R (Jun "copy from readme", 2026-11-07)


def wide(sigma, dgp):
    """Y_observed by period (rows 1..60) and unit (treated, donor_01..donor_29)."""
    d = pd.read_csv(mod.PANELS_BY_SIGMA[sigma][dgp])
    w = d.pivot(index="period", columns="unit_id", values="Y_observed").sort_index()
    return w[["treated"] + donors(w)]


def donors(w):
    return sorted(c for c in w.columns if c != "treated")


def fmt(v):
    return f"{v:.3f}"                                     # departure 2: three decimals


def prompt(method, sigma, dgp):
    """The prompt text of one method on one panel (plan section Prompt texts)."""
    w = wide(sigma, dgp)
    target = "period,target\n" + "".join(f"{t},{fmt(w.loc[t, 'treated'])}\n" for t in PRE)
    if method == "without_donors":
        return HEAD + "\n" + target + "\n" + ASK
    ds = donors(w)
    table = ("period," + ",".join(ds) + "\n"
             + "".join(f"{t}," + ",".join(fmt(w.loc[t, d]) for d in ds) + "\n" for t in PRE + POST))
    return HEAD + DONOR_LINE + "\nTarget:\n" + target + "\nDonors:\n" + table + "\n" + ASK


def prompt_id(method, sigma, dgp):
    return f"{method}_sigma_{sigma}_{dgp}"


NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"


def parse(content):
    """The counterfactual {"41": v, ..., "60": v} of a reply, or None (departure 1, plan decision "answer line and
    parsing"): the last line that, with every `*` and backtick removed and spaces stripped, starts with
    "counterfactual:" in any case, holding exactly the periods 41 to 60, each once, each followed by one number."""
    lines = [re.sub(r"[*`]", "", l).strip() for l in (content or "").splitlines()]
    cands = [l for l in lines if l.lower().startswith("counterfactual:")]
    if not cands:
        return None
    out = {}
    for part in cands[-1][len("counterfactual:"):].split(","):
        m = re.fullmatch(rf"(\d+)\s+({NUM})", part.strip())
        if not m or int(m.group(1)) in out:
            return None
        out[int(m.group(1))] = float(m.group(2))
    if sorted(out) != POST:
        return None
    return {str(t): out[t] for t in POST}


def rules_from(y, donor_frame=None):
    """The arithmetic rules over 41..60 from the target series y (indexed by period, 1..40 used) and, for method 2,
    the donors (rows 1..60, one column per donor). Values are the panel's, not the rounded prompt values."""
    pre = np.array([y[t] for t in PRE], dtype=float)
    slope, icpt = np.polyfit(np.array(PRE, dtype=float), pre, 1)
    out = {"pre_mean": {str(t): float(pre.mean()) for t in POST},
           "last_value": {str(t): float(y[40]) for t in POST},
           "linear_trend": {str(t): float(icpt + slope * t) for t in POST}}
    if donor_frame is not None:
        dm = donor_frame.mean(axis=1)
        shift = pre.mean() - dm.loc[PRE].mean()
        out["did_donor_mean"] = {str(t): float(dm.loc[t] + shift) for t in POST}
    return out


def readme_sc(sigma, dgp):
    """{"41": y_syn, ..., "60": y_syn} of the README Section 3 synthetic control on this panel (pc_readme_sc.R, checked
    against her replication 1 row by gate G7)."""
    p = pd.read_csv(README_SC_PATHS, dtype={"sigma": str})
    s = p[(p.sigma == sigma) & (p.dgp == dgp)].set_index("period")["y_syn"]
    return {str(t): float(s[t]) for t in POST}


def rules(method, sigma, dgp):
    """Every rule an answer of this method is set against: three for method 1, five for method 2, the fifth the README
    Section 3 synthetic control (Jun "2. b", 2026-11-07, departure 4 declined)."""
    w = wide(sigma, dgp)
    if method == "without_donors":
        return rules_from(w["treated"])
    return {**rules_from(w["treated"], w[donors(w)]), "readme_sc": readme_sc(sigma, dgp)}


README_ROWS = ["ATT bias", "ATT RMSE", "counterfactual RMSE", "within ±0.25 of the truth", "within ±0.50 of the truth"]


def readme_column(sigma, dgp):
    """{row: cell} of README Section 5 for one DGP and noise level, exactly as the collaborator printed it (Jun "what
    collaborator report in readme table is what i will seee EXACTLY in result table for her numbers", 2026-11-06)."""
    sys.path.insert(0, str(SIM / "report"))
    import eval_slides
    t = eval_slides.readme_tables()[dgp]
    k = list(SIGMAS).index(sigma)
    return {row: t[row][k] for row in README_ROWS}


def rmse(a, b):
    return float(np.sqrt(np.mean([(a[str(t)] - b[str(t)]) ** 2 for t in POST])))


def scored(sigma, dgp, cf):
    """ATT, ATT error, relative ATT error and counterfactual RMSE of one counterfactual, as score.py scores answers."""
    att = score.counterfactual_att(dgp, cf, sigma)
    return {"att": att, **score.score(dgp, cf, att, score.truth(sigma))}


def record_path(method, sigma, model, effort):
    safe = model.replace(":", "_").replace("/", "_")      # lb_lib.safe_name
    return HERE / method / f"sigma_{sigma}" / f"{safe}_{effort}.jsonl"


def last_records(path):
    """{prompt_id: last record} of one record file."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def chosen(method, sigma, dgp, model):
    """(record, effort) scored for one case: the default record with a valid answer, else the low record with one,
    else (None, None) (departure 3)."""
    pid = prompt_id(method, sigma, dgp)
    for effort in EFFORTS:
        r = last_records(record_path(method, sigma, model, effort)).get(pid)
        if r and r.get("ok"):
            return r, effort
    return None, None
