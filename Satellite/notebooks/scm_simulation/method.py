"""SCM method package: synthetic control (Abadie, Diamond and Hainmueller 2010) with the Synth R package 1.1-10 as
reference code, on the simulated Abadie data in data/simulation/. Plan ~/.claude/plans/scm-simulation-skills.md.

Every pipeline script in notebooks/sc_skills/ takes this package as ../../scm_simulation.

    settings   Synth's own arguments (dataprep and synth), the model writes them in configure and design
    choices    the paper's alternatives to what Synth fixes in its body (p. 496): weights, interpolation_penalty
    engine     the reference engine runs Synth unchanged (synth_fit.R), the native engine runs scm_native.R
    use        "simulation": the two shipped panels at outcome shock SD 1 (simulation_data/sigma_1.0/), one case each,
               scored later against that folder's ground_truth.csv; "sigma_0.5" and "sigma_0.2": the same two panels
               at the collaborator's lower noise levels, with the same file names and task text
    records    record_folders(sigma): one folder per skill format and noise level, scm-configure/sigma_<s>/,
               scm-design/sigma_<s>/, skill_profile/<generator>/sigma_<s>/; tables in results/sigma_<s>/
"""
import atexit
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                          # notebooks/scm_simulation
SC = HERE.parent / "sc_skills"
if str(SC) not in sys.path:
    sys.path.insert(0, str(SC))

from scs_method import Field, Check, Spec, CopyRule, ReferenceCode, Engine, Texts, MethodPackage  # noqa: E402
from scs_loop import Use                                                                          # noqa: E402
import scs_loop as _loop                                                                          # noqa: E402

DATA = HERE.parents[1] / "data" / "simulation"
SIGMAS = ("1.0", "0.5", "0.2")           # the collaborator's three noise levels, simulation_data/sigma_<s>/
PANELS_BY_SIGMA = {s: {g: DATA / "simulation_data" / f"sigma_{s}" / f"{g}_panel.csv" for g in ("dgp1", "dgp2")}
                   for s in SIGMAS}
TRUTH_BY_SIGMA = {s: DATA / "simulation_data" / f"sigma_{s}" / "ground_truth.csv" for s in SIGMAS}
PANELS = PANELS_BY_SIGMA["1.0"]
TRUTH = TRUTH_BY_SIGMA["1.0"]
GENERATORS = ("gpt-5.4", "deepseek")


def record_folders(sigma):
    """{arm: folder of its records} at one noise level (plan use-nature-skills-analyze-wiggly-coral.md, departure 3 of
    the noise-level plan; scm-four-design added by the plan at the top of that file)."""
    return {"scm-configure": HERE / "scm-configure" / f"sigma_{sigma}", "scm-design": HERE / "scm-design" / f"sigma_{sigma}",
            **{f"profile-{g}": HERE / "skill_profile" / g / f"sigma_{sigma}" for g in GENERATORS},
            "scm-four-design": HERE / "scm-four-design" / f"sigma_{sigma}"}


def results_folder(sigma):
    return HERE / "results" / f"sigma_{sigma}"


def use_name(sigma):
    return "simulation" if sigma == "1.0" else f"sigma_{sigma}"
PAPER = DATA / ("Synthetic Control Methods for Comparative Case Studies- Estimating the Effect of California’s "
                "Tobacco Control Program.pdf")
R_ENV = Path("/data/wang/junh/envs/rsynth")
RSCRIPT = R_ENV / "bin" / "Rscript"
FIRST_TREATED = 41                       # README lines 42 to 44: fixed by the design, recorded only in config.json


def _units(path):
    import pandas as pd
    ids = pd.read_csv(path, usecols=["unit_id"])["unit_id"]
    return sorted(set(ids) - {"treated"})


DONORS = tuple(_units(PANELS["dgp1"]))
assert DONORS == tuple(_units(PANELS["dgp2"])), "the two panels hold different donors"
VARIABLES = ("Z1", "Z2", "Y_observed")
# predictors.op: "A character string identifying the method (operator) to be used on the predictors ... rm.na = T is
# hardwired" (dataprep help); dataprep calls apply(X, 2, paste(op), na.rm = TRUE), so the base R summaries that take na.rm
OPS = ("mean", "median", "sd", "var", "min", "max", "sum")
# optimxmethod: "all optimization algorithms currently implemented in the function optimx ... If "All" is specified, all
# algorithms in optimx are tried" (synth help); the list is optimx 2025.4.9 ctrldefault()$allmeth
OPTIMX = ("BFGS", "CG", "Nelder-Mead", "L-BFGS-B", "nlm", "nlminb", "lbfgsb3c", "Rcgmin", "Rtnmin", "Rvmmin", "snewton",
          "snewtonm", "spg", "ucminf", "newuoa", "bobyqa", "uobyqa", "nmkb", "hjkb", "hjn", "lbfgs", "subplex", "ncg",
          "nvm", "mla", "slsqp", "tnewt", "anms", "pracmanm", "nlnm", "snewtm", "All")


def _positive_list(v):
    return (isinstance(v, list) and len(v) > 0
            and all(not isinstance(x, bool) and isinstance(x, (int, float)) and x > 0 for x in v))


def _positive(v):
    return v > 0


# ----------------------------------------------------------------------------- spec
SETTINGS = {
    "special_predictors": Field("predictor_list", {"variables": VARIABLES, "ops": OPS},
                                "list of {variable, periods, op}, variable one of " + ", ".join(VARIABLES)
                                + ", op one of " + ", ".join(OPS) + ": dataprep's special.predictors. Each entry is "
                                "one predictor, the op of the variable over the listed periods"),
    "predictors": Field("name_list", VARIABLES,
                        "list of names from " + ", ".join(VARIABLES) + ", may be empty: dataprep's predictors. Each is one predictor, the "
                        "predictors_op of the variable over time_predictors_prior"),
    "predictors_op": Field("enum", OPS, "one of " + ", ".join(OPS) + ": dataprep's predictors.op"),
    "time_predictors_prior": Field("positive_list", _positive_list,
                                   "list of periods: dataprep's time.predictors.prior, the periods over which "
                                   "predictors are summarized"),
    "time_optimize_ssr": Field("positive_list", _positive_list,
                               "list of periods: dataprep's time.optimize.ssr, the periods of the outcome over which "
                               "the mean squared prediction error is minimized when V is searched"),
    "controls_identifier": Field("name_list", DONORS,
                                 f"list of donor names from {DONORS[0]} to {DONORS[-1]}: dataprep's "
                                 "controls.identifier, the donor pool"),
    "custom_v": Field("positive_list_or_null", _positive_list,
                      "list of numbers above 0, one per predictor, or null: synth's custom.v. Null means V is "
                      "searched"),
    "optimxmethod": Field("name_list", OPTIMX, "list of names from " + ", ".join(OPTIMX)
                          + ": synth's optimxmethod, the optimx methods of the V search (All runs every one)"),
    "genoud": Field("bool", None, "true or false: synth's genoud"),
    "quadopt": Field("enum", ("ipop",), "ipop: synth's quadopt"),
    "margin_ipop": Field("float", _positive, "number above 0: synth's Margin.ipop"),
    "sigf_ipop": Field("int", _positive, "integer above 0: synth's Sigf.ipop"),
    "bound_ipop": Field("float", _positive, "number above 0: synth's Bound.ipop"),
}
CHOICES = {
    "weights": Field("enum", ("convex", "extrapolation"),
                     "convex = each donor weight at least 0 and the weights summing to one (the paper's definition "
                     "and Synth); extrapolation = the weights sum to one and any weight may be negative or above one "
                     "(p. 496)"),
    "interpolation_penalty": Field("enum", ("none", "distance"),
                                   "none = no penalty (Synth); distance = penalty_lambda times the sum over donors of "
                                   "the donor's weight times its V-weighted squared distance from the treated unit's "
                                   "scaled predictors (p. 496)"),
}
OPTIONAL = {"penalty_lambda": (Field("float", lambda v: v >= 0,
                                     "a number of 0 or above, given exactly when interpolation_penalty is distance"),
                               "number of 0 or above", "<number, only with distance>")}
RULES = [
    "Every period in time_predictors_prior, time_optimize_ssr and the periods of special_predictors is a whole "
    "period number.",
    "penalty_lambda is given, a number of 0 or above, exactly when interpolation_penalty is distance.",
    "interpolation_penalty = distance requires weights = convex.",
]


def _whole(given, out, ctx, errors):
    for k in ("time_predictors_prior", "time_optimize_ssr"):
        v = out.get(k)
        if v is not None and not all(float(x).is_integer() for x in v):
            errors.append(f"field '{k}': every period must be a whole period number")


def _lambda(given, out, ctx, errors):
    pen = out.get("interpolation_penalty")
    if pen == "distance":
        if "penalty_lambda" not in given:
            errors.append("field 'penalty_lambda': required when interpolation_penalty is distance")
        else:
            lam = given["penalty_lambda"]
            if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not lam >= 0:
                errors.append(f"field 'penalty_lambda': expected a number of 0 or above, got {lam!r}")
            else:
                out["penalty_lambda"] = float(lam)
    elif "penalty_lambda" in given and pen is not None:
        errors.append("field 'penalty_lambda': given, but interpolation_penalty is not distance")


def _convex(given, out, ctx, errors):
    if out.get("interpolation_penalty") == "distance" and out.get("weights") == "extrapolation":
        errors.append("fields 'interpolation_penalty','weights': interpolation_penalty = distance requires "
                      "weights = convex")


README_SPECIAL = ([{"variable": "Z1", "periods": [40], "op": "mean"}, {"variable": "Z2", "periods": [40], "op": "mean"}]
                  + [{"variable": "Y_observed", "periods": [t], "op": "mean"} for t in range(1, 41)])
README_WAY = {"special_predictors": README_SPECIAL, "predictors": [], "predictors_op": "mean",
              "time_predictors_prior": [float(t) for t in range(1, 41)],
              "time_optimize_ssr": [float(t) for t in range(1, 41)],
              "controls_identifier": list(DONORS), "custom_v": [1.0] * len(README_SPECIAL),
              "optimxmethod": ["Nelder-Mead", "BFGS"], "genoud": False, "quadopt": "ipop",
              "margin_ipop": 0.0005, "sigf_ipop": 5, "bound_ipop": 10.0,
              "weights": "convex", "interpolation_penalty": "none"}
ADH_WAY = {**README_WAY,
           "special_predictors": [{"variable": "Y_observed", "periods": [t], "op": "mean"} for t in (12, 23, 40)],
           "predictors": ["Z1", "Z2"], "time_predictors_prior": [float(t) for t in range(23, 41)],
           "custom_v": None}
REFERENCE_WAYS = {"README way": README_WAY, "ADH way": ADH_WAY}

SPEC = Spec(settings=SETTINGS, choices=CHOICES, optional=OPTIONAL, rules=RULES, configure_rules=[0],
            checks=[Check(_whole, True), Check(_lambda, False), Check(_convex, False)],
            reference=dict(README_WAY), configure_label="scm-configure")


# ----------------------------------------------------------------------------- the use's cases and contexts
def _case_files(name, sigma="1.0"):
    path = PANELS_BY_SIGMA[sigma][name]
    return {path.name: path.read_text()}


def CONTEXTS():
    return [{"case": name, "panel": str(PANELS[name])} for name in PANELS]


_TRIAL_DIR = None


def _trial_cases():
    """One folder per case holding that case's panel, as the case folder of a run holds it (made once per process)."""
    global _TRIAL_DIR
    if _TRIAL_DIR is None:
        _TRIAL_DIR = Path(tempfile.mkdtemp(prefix="scm_trial_"))
        atexit.register(shutil.rmtree, _TRIAL_DIR, ignore_errors=True)   # Jun "if we don't need, deleted after finish."
        for name in PANELS:
            (_TRIAL_DIR / name).mkdir()
            for fname, text in _case_files(name).items():
                (_TRIAL_DIR / name / fname).write_text(text)
    return [_TRIAL_DIR / name for name in PANELS]


class _Trial(dict):
    """The R trial: its case folders are made only when the trial runs."""

    def __getitem__(self, k):
        if k == "cases":
            return _trial_cases()
        return super().__getitem__(k)


# ----------------------------------------------------------------------------- engines (R, run outside the box)
def _run_engine(engine, ctx, spec):
    """engine.R <engine> <spec.json> <panel> -> the fit record fit.R prints after its marker."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "spec.json"
        sp.write_text(json.dumps(spec))
        p = subprocess.run([str(RSCRIPT), str(HERE / "engine.R"), engine, str(sp), ctx["panel"]],
                           capture_output=True, text=True, cwd=HERE,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    if p.returncode != 0:
        raise RuntimeError(f"engine.R {engine} failed: {p.stderr.strip()[-500:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


ENGINE = Engine(reference=lambda ctx, spec: _run_engine("synth", ctx, spec),
                native=lambda ctx, spec, forced: _run_engine("native", ctx, spec))


# ----------------------------------------------------------------------------- texts (drafted in plan step 7)
METHOD_MD = """# Synthetic control for one treated unit

Abadie, Diamond and Hainmueller (2010), Sections 2.2 and 2.3 (pp. 494 to 496), as the Synth R package implements it.

## The quantity estimated
There are J + 1 units. Unit 1 is exposed to the treatment from period T0 + 1 on, and the other J units, the donors,
are never exposed. For each treated period t the effect is the treated unit's observed outcome minus the outcome it
would have had without treatment, Y1t − Y1t(N). The second term is never observed and is what the method estimates.

## The synthetic control
A synthetic control is a weighted combination of donors, W = (w2, ..., wJ+1), with every weight at least 0 and the
weights summing to one. Its outcome Σj wj Yjt estimates Y1t(N), and Y1t − Σj wj Yjt estimates the effect in period t.
The paper notes that weights below 0 or above one can be used, at the cost of extrapolating beyond the donors (p. 496).

## Predictors
The weights are chosen so the synthetic control resembles the treated unit before treatment. X1 holds k
pre-treatment characteristics of the treated unit, its covariates and any linear combinations of its pre-treatment
outcomes (for example the outcome in one period, or its average over several periods). X0 holds the same
characteristics for each donor. One choice is the outcome in every pre-treatment period. The paper notes that a few
linear combinations can be used instead, checking that the resulting weights still match the pre-treatment outcomes
approximately (p. 496). In Synth, `predictors` are summarized with `predictors_op` over `time_predictors_prior`, and
each entry of `special_predictors` is one predictor with its own periods and summary. Synth divides every predictor by
its standard deviation across the treated unit and the donors before matching.

## The weights given V
W is chosen to minimize (X1 − X0 W)' V (X1 − X0 W) under the weight constraints, where V is a diagonal matrix of
non-negative predictor weights. Synth solves this quadratic program with `kernlab::ipop` (`quadopt`), whose
precision and bounds are `margin_ipop`, `sigf_ipop` and `bound_ipop`. The paper notes that W can instead minimize this
distance plus penalty terms that grow with the distance between the treated unit and each donor with positive weight,
which keeps distant donors out (p. 496).

## Choosing V
The choice of V affects the estimator's error (p. 496). V can come from a judgment of how well each predictor predicts
the outcome (`custom_v`, one positive number per predictor, normalized by Synth to sum to one). V can be chosen from
the data so the synthetic control reproduces the treated unit's pre-treatment outcome path, minimizing the mean
squared prediction error of the outcome over pre-treatment periods (`custom_v` null and `time_optimize_ssr` those
periods). This is the paper's own choice in its application. With enough pre-treatment periods, W can be computed
on an earlier training stretch and V chosen to minimize the prediction error over a later validation stretch.
When V is searched, Synth starts its optimizer (`optimxmethod`, optionally `genoud` first) from equal weights and from
regression-based weights and keeps the better result.

## Donors and fit
The donor pool can be restricted to units similar to the treated unit, which reduces interpolation bias
(`controls_identifier`). The paper adds that when no weighted combination of donors matches the treated unit's
pre-treatment characteristics, the fit may be poor, and the analyst decides whether the match is close enough (p. 495).

## The result
After fitting, the synthetic control's outcome in every period is Σj wj Yjt, and the gap is the treated unit's
observed outcome minus it. Before treatment the gap shows the fit. After treatment it is the estimated effect, and
its average over the treated periods is the average treatment effect on the treated unit.
"""

SHOW_DATA = """\"\"\"Show the panel in the working directory: the units, the periods and one variable by period and unit.

    python scripts/show_data.py            Y_observed
    python scripts/show_data.py Z1         another column (Z1, Z2 or Y_observed)
\"\"\"
import glob
import sys

import pandas as pd

files = glob.glob("*_panel.csv")
if len(files) != 1:
    raise SystemExit(f"expected one *_panel.csv file in the working directory, found {files}")
d = pd.read_csv(files[0])
var = sys.argv[1] if len(sys.argv) > 1 else "Y_observed"
if var not in ("Z1", "Z2", "Y_observed"):
    raise SystemExit("the variable must be Z1, Z2 or Y_observed")
units = ["treated"] + sorted(u for u in d.unit_id.unique() if u != "treated")
print(f"file {files[0]}: {len(d)} rows, columns {list(d.columns)}")
print(f"units: {len(units)} ({units[0]}, then {units[1]} to {units[-1]}); periods {d.period.min()} to {d.period.max()}")
print(f"treatment_group is 1 for {sorted(d.loc[d.treatment_group == 1, 'unit_id'].unique())} and 0 for the others")
print(f"\\n{var} by period (rows) and unit (columns):")
wide = d.pivot(index="period", columns="unit_id", values=var)[units]
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(wide.round(3).to_string())
"""

FIT_R = """#!/usr/bin/env Rscript
# Fit the synthetic control method with the Synth R package, version 1.1-10, unchanged, at the settings in a file:
#   Rscript scripts/fit.R %(file)s
# The arguments of dataprep() that describe the data file are filled here, the same for every file:
#   dependent "Y_observed", unit.variable unit numbers (treated 1, donor_jj jj + 1), time.variable "period",
#   treatment.identifier "treated", unit.names.variable "unit_id", time.plot every period of the file, verbose FALSE.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "scm_fields.R"))
source(file.path(here, "synth_fit.R"))
source(file.path(here, "scm_native.R"))
args <- commandArgs(trailingOnly = TRUE)
spec_file <- if (length(args)) args[1] else "%(file)s"
given <- tryCatch(fromJSON(spec_file, simplifyVector = FALSE), error = function(e) {
  cat("refused: ", spec_file, " is not readable JSON (", conditionMessage(e), ")\\n", sep = "")
  quit(status = 1)
})
%(validate)s
panel <- list.files(".", pattern = "_panel[.]csv$")
if (length(panel) != 1) stop("expected one *_panel.csv file in the working directory")
d <- scm_data(panel)
synth_choices <- identical(s$weights, "convex") && identical(s$interpolation_penalty, "none")
rec <- if (synth_choices) fit_synth(d, s) else fit_native(d, s)
rec$file <- basename(spec_file)
log <- Sys.getenv("FIT_LOG")                         # the record of this fit, for the run log
if (nzchar(log)) cat(scm_json(rec), "\\n", file = log, append = TRUE, sep = "")

cat(if (synth_choices) "fit: Synth 1.1-10 dataprep() and synth(), unchanged\\n" else
      sprintf("fit: Synth's code with weights = %%s, interpolation_penalty = %%s\\n", s$weights, s$interpolation_penalty))
cat(sprintf("predictors and their weights in V (%%d):\\n", length(rec$v)))
for (i in seq_along(rec$v)) cat(sprintf("  %%-28s %%.6f\\n", rec$predictor_names[[i]], rec$v[[i]]))
cat(sprintf("donor weights (%%d donors in the pool):\\n", length(rec$weights)))
for (i in seq_along(rec$weights)) cat(sprintf("  %%-10s %%.6f\\n", rec$donor_ids[[i]], rec$weights[[i]]))
cat(sprintf("pre-treatment fit: mean squared prediction error over time_optimize_ssr (%%d periods) %%.6f\\n",
            length(s$time_optimize_ssr), rec$fit_mse))
cat("period, treated unit observed, synthetic control, gap (observed minus synthetic control):\\n")
for (p in names(rec$predictions))
  cat(sprintf("  %%3s %%12.6f %%12.6f %%12.6f\\n", p, rec$predictions[[p]] + rec$gaps[[p]], rec$predictions[[p]],
              rec$gaps[[p]]))
"""

VALIDATE_CONFIGURE = """errors <- check_settings(given, "configure")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\\n", sep = "")
  quit(status = 1)
}
s <- c(given, SCM_SYNTH_CHOICES)          # the two choices this skill fixes, at what Synth does"""

VALIDATE_DESIGN = """errors <- check_settings(given, "design")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\\n", sep = "")
  quit(status = 1)
}
s <- given"""

INTRO = ("A weighted combination of donor units, the synthetic control, is chosen so that it resembles the treated unit "
         "before treatment. Its outcome after treatment estimates what the treated unit's outcome would have been "
         "without treatment, and the difference is the effect. `references/method.md` describes the construction. "
         "Read it if anything below is unclear.")

SETTINGS_LIST = """- `special_predictors`, predictors given one by one, each a variable (`Z1`, `Z2` or `Y_observed`), the periods it is summarized over and the summary `op`
- `predictors` and `predictors_op`, variables summarized with one summary over `time_predictors_prior`
- `time_predictors_prior`, the periods over which `predictors` are summarized
- `time_optimize_ssr`, the outcome periods whose mean squared prediction error chooses V when V is searched
- `controls_identifier`, the donor pool
- `custom_v`, the predictor weights V, one positive number per predictor, or null to let Synth search V
- `optimxmethod` and `genoud`, the optimizers of the V search
- `quadopt`, `margin_ipop`, `sigf_ipop` and `bound_ipop`, the solver of the donor weights and its precision"""

SETTINGS_RULES = ("`references/settings.md` gives each field's type, range and meaning, the rule that every listed "
                  "period is a whole period number, and the shape of the file. There are no defaults. The fitter "
                  "refuses an incomplete file and names what is missing. Each field is the Synth argument of the same "
                  "name, so Synth's own errors, for example a `custom_v` whose length is not the number of predictors, "
                  "are printed as Synth gives them.")


def _workflow(spec_file, what, why):
    return f"""1. Look at the data: `python scripts/show_data.py`, and `python scripts/show_data.py Z1` or `Z2` for the
   covariates. It shows the treated unit and the donors period by period.
2. Read `references/method.md` and `references/settings.md`. Decide the {what} from the data and the method.
3. Write `{spec_file}`, then run `Rscript scripts/fit.R {spec_file}`. It prints the predictors and V, the donor
   weights, the pre-treatment fit over `time_optimize_ssr`, and for every period the treated unit's observed outcome,
   the synthetic control and their gap.
4. You may change the {what} and run again. Every run is recorded. The pre-treatment fit is an in-sample quantity, and
   a closer fit before treatment is not by itself a better estimate after it.
5. Finish by writing `answer.json`, with one value for every treated period the task names:
   ```json
   {{"counterfactual": {{"<period>": <value>, ...}}, "att": <value>, "rationale": "<{why}, in a few sentences>"}}
   ```
   `counterfactual` is the treated unit's outcome without treatment in each treated period, and `att` the average
   effect over those periods. The values you submit are your answer. If they differ from the fitter's, say why.
"""


TEXTS = Texts(
    method_name="synthetic control",
    skill_prefix="scm",
    method_md=METHOD_MD,
    configure_description=("Estimate the effect of a treatment on one treated unit from a panel of that unit and "
                           "untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller "
                           "(2010), by deciding the arguments of the Synth R package (predictors, donor pool, predictor "
                           "weights V, optimizer) and running the bundled fitter. Use this whenever one treated unit's "
                           "outcome without treatment is to be estimated from donors and the Synth code is provided."),
    design_description=("Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated "
                        "donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by "
                        "designing the method, the Synth R package's arguments and the paper's alternatives for the "
                        "donor weights, and running the bundled fitter. Use this whenever one treated unit's outcome "
                        "without treatment is to be estimated from donors and the method's design is yours to choose."),
    configure_intro=INTRO,
    design_intro=INTRO,
    fixed_choices=("every donor weight is at least 0 and the weights sum to one, as the paper defines a synthetic "
                   "control, and no penalty is added to the weight fit. Everything else is Synth's own code, unchanged."),
    settings_list=SETTINGS_LIST,
    settings_rules=SETTINGS_RULES,
    configure_workflow=_workflow("settings.json", "settings", "why these settings"),
    design_overview=("`spec.json` holds two kinds of field. The settings are the 13 arguments of Synth's `dataprep` and "
                     "`synth` listed in `references/settings.md`. The choices are the paper's alternatives to what "
                     "Synth fixes in its code (Abadie, Diamond and Hainmueller 2010, p. 496):"),
    choices_list=("- `weights`, `convex` (every donor weight at least 0, summing to one, the paper's definition and "
                  "Synth's) or `extrapolation` (summing to one, any weight may be below 0 or above one)\n"
                  "- `interpolation_penalty`, `none` or `distance`, which adds `penalty_lambda` times each donor's "
                  "weight times its distance from the treated unit's predictors, so that distant donors are kept out"),
    design_rules=("`references/settings.md` defines every field and the rules that tie them. `penalty_lambda` is given "
                  "exactly when `interpolation_penalty` is `distance`, and `distance` requires `convex` weights. The "
                  "fitter refuses a specification that breaks a rule and names the fields, so read the rules before "
                  "writing the file. There are no defaults."),
    design_workflow=_workflow("spec.json", "design", "why this design"),
    scripts={"show_data.py": SHOW_DATA},
    fit_script=FIT_R,
    fit_script_name="fit.R",
    fixed_comment="# the two choices this skill fixes, at what Synth does",
    validate_configure=VALIDATE_CONFIGURE,
    validate_design=VALIDATE_DESIGN,
    authors="Abadie, Diamond and Hainmueller",
    paper_title=("Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's "
                 "Tobacco Control Program"),
    profile_code=(" The paper's own software, the Synth R package version 1.1-10, is at `code/Synth/`, and R 4.3.3 "
                  "with Synth and jsonlite installed is available to the agent."),
    profile_focus="Causal inference",
    profile_code_paths=[HERE / "code" / "Synth"])

PACKAGE = MethodPackage(
    name="scm",
    paper=PAPER,
    reference_code=ReferenceCode(
        origin="paper package", language="R",
        copy=CopyRule(mode="package", source=R_ENV / "lib" / "R" / "library" / "Synth", package="Synth",
                      out=HERE / "code" / "synth_installed.json"),
        trial=_Trial(script=str(HERE / "trial.R"), r_env=str(R_ENV))),
    data=[PANELS["dgp1"], PANELS["dgp2"]],
    spec=SPEC,
    engine=ENGINE,
    texts=TEXTS,
    skill_modules=[("scm_fields.R", HERE / "scm_fields.R"), ("synth_fit.R", HERE / "synth_fit.R"),
                   ("scm_native.R", HERE / "scm_native.R")],
)


# ----------------------------------------------------------------------------- the use "simulation"
def _observed_post(case):
    import io
    import pandas as pd
    d = pd.read_csv(io.StringIO(next(iter(case["files"].values()))))
    t = d[(d.unit_id == "treated") & (d.period >= FIRST_TREATED)].sort_values("period")
    return {str(int(p)): float(y) for p, y in zip(t.period, t.Y_observed)}


def _post_periods():
    return [str(t) for t in range(FIRST_TREATED, 61)]


def _use_cases(sigma="1.0"):
    out = []
    for name, path in PANELS_BY_SIGMA[sigma].items():
        out.append({"prompt_id": f"scm_{name}", "id": {"case": name}, "files": _case_files(name, sigma),
                    "ctx": {"case": name, "panel": str(path)}})
    return out


def _use_answer(case, a):
    """README Section 3 Step 4: the counterfactual for periods 41 to 60 and the ATT."""
    shape = ('answer.json must be an object {"counterfactual": {"41": <number>, ..., "60": <number>}, '
             '"att": <number>, "rationale": "<text>"}')
    if not isinstance(a, dict) or not isinstance(a.get("counterfactual"), dict) or "att" not in a:
        return None, shape
    cf = a["counterfactual"]
    if sorted(cf, key=lambda k: (len(k), k)) != _post_periods():
        return None, "answer.json: counterfactual must have exactly the keys \"41\" to \"60\""
    try:
        vals = {k: float(cf[k]) for k in _post_periods()}
        att = float(a["att"])
    except (TypeError, ValueError) as e:
        return None, f"answer.json: every counterfactual value and att must be a number ({e})"
    if any(math.isnan(v) for v in [*vals.values(), att]):
        return None, "answer.json: a value is NaN"
    return {"prediction": {"counterfactual": vals}, "att": att, "rationale": str(a.get("rationale", ""))}, None


def _att(path, observed):
    return sum(observed[k] - path[k] for k in _post_periods()) / len(_post_periods())


def _fit_values(fit):
    """A fit's counterfactual for 41 to 60, in the shape of the answer's prediction (the fit behind an answer is found
    by its counterfactual; the answer's att is scored beside it)."""
    return {"counterfactual": {k: fit["predictions"][k] for k in _post_periods()}}


def _dry_answer(case, fit):
    cf = {k: fit["predictions"][k] for k in _post_periods()}
    return {"counterfactual": cf, "att": _att(cf, _observed_post(case)), "rationale": "dry run"}


def _task_text(skill_dir, case):
    """The first message. A profile skill (its folder sits beside the solver.json that scs_profile.collect writes)
    gets no skill line, as row G16 of the profile experiment (Jun "16. follow the paper", 2026-10-13): the paper varies
    only whether the skill is there. A configure or design skill is named from its SKILL.md front matter."""
    import lb_skill_loop
    panel = next(iter(case["files"]))
    task = (f"Estimate the effect of a treatment on the unit `treated` in the panel `{panel}`. The unit `treated` was "
            f"treated in periods {FIRST_TREATED} to 60, and the units `donor_01` to `donor_29` were never treated. "
            f"Estimate the treated unit's outcome `Y_observed` without treatment in each period from {FIRST_TREATED} to "
            "60, and the average effect over those periods.\n")
    rest = ("Its instructions are in SKILL.md. Read that file first and follow it. Reply with one ```bash block per "
            "turn. It is run by bash in the working directory and its output comes back in the next message. "
            "Finish by writing answer.json as the skill describes.")
    if (Path(skill_dir).parent / "solver.json").exists():
        return task + "The working directory holds a skill for this task. " + rest
    fm = lb_skill_loop.frontmatter(skill_dir)
    return task + f"The working directory holds a skill for this task:\n  {fm['name']}. {fm['description']}\n" + rest


USES = {"simulation": Use(name="simulation", cases=_use_cases, task_text=_task_text, answer=_use_answer,
                          mounts=[R_ENV],
                          # the solver's Python first: the R environment holds a Python of its own (3.13, no pandas)
                          path_dirs=[Path(_loop.L.PY_DIR), R_ENV / "bin"], nofile=256, fit_values=_fit_values, dry_answer=_dry_answer,
                          profile_task=("estimating the effect of a treatment on one treated unit from panel data on that unit "
                                        "and untreated donor units"),
                          profile_data=("one panel file `*_panel.csv` with the columns `unit_id`, `period`, "
                                        "`treatment_group`, `Z1`, `Z2` and `Y_observed`, one row per unit and period, "
                                        "for the treated unit `treated` and the untreated donor units."),
                          profile_answer=("estimates the treated unit's outcome without treatment in each treated "
                                          "period the task names and the average effect over those periods, and "
                                          "writes `answer.json` as {\"counterfactual\": {\"<period>\": <number>, "
                                          "...}, \"att\": <number>, \"rationale\": \"<text>\"}."))}
for _s in SIGMAS[1:]:                    # the lower noise levels: the same use, that level's panels
    USES[use_name(_s)] = dataclasses.replace(USES["simulation"], name=use_name(_s),
                                             cases=(lambda s=_s: _use_cases(s)))


# ----------------------------------------------------------------------------- compare steps
def _steps_records():
    """{(arm, model): {prompt_id: last record}} over the sigma = 1 record folders (record_folders)."""
    out = {}
    for arm, folder in record_folders("1.0").items():
        for p in sorted(folder.glob("*.jsonl")):
            recs = {}
            for line in p.read_text().splitlines():
                r = json.loads(line)
                recs[r["prompt_id"]] = r                 # the last record of a case is the one kept
            first = recs[next(iter(recs))]
            out[(arm, first["model"])] = recs
    return out


def _steps_describe(fit, rec, case):
    w = [float(x) for x in fit["weights"]]
    s = fit["spec"]
    return {"case": rec.get("case"), "engine": fit.get("engine_used"), "J": len(fit["donor_ids"]),
            "donor_ids": " ".join(map(str, fit["donor_ids"])), "weights": " ".join(f"{x:.4f}" for x in w),
            "max_weight": max(w), "n_predictors": len(fit["v"]), "v_searched": s.get("custom_v") is None,
            "fit_mse": fit.get("fit_mse")}


def _steps_reference_rows():
    import pandas as pd
    import scs_spec
    rows = []
    for way, spec in {"README way": README_WAY}.items():     # the ADH row dropped (Jun "drop all, b", 2026-11-04)
        for ctx in CONTEXTS():
            fit = ENGINE.reference(ctx, scs_spec.validate(SPEC, dict(spec), ctx))
            rows.append({"data": "simulation", "skill": way, "model": "", "case": ctx["case"],
                         **{k: v for k, v in _steps_describe(fit, {"case": ctx["case"]}, None).items() if k != "case"}})
    return pd.DataFrame(rows)


STEPS = {"data": "simulation", "records": _steps_records, "describe": _steps_describe,
         "reference_rows": _steps_reference_rows, "reference_marker": None,
         "case": lambda rec: rec.get("case")}
