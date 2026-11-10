"""scm-four-design package: synthetic control (Abadie, Diamond and Hainmueller 2010) where the model designs four of the
paper's method items, each limited to the paper's options (p. 496), and every fit uses all 29 donors and Synth 1.1-10's
default optimizers and solver settings. Plan ~/.claude/plans/use-nature-skills-analyze-wiggly-coral.md (top section).

Every pipeline script in notebooks/sc_skills/ takes this package as ../../scm_simulation/four_design. Its paper, Synth
copy, R trial, data, uses and method.md are the scm package's (../../scm_simulation).

    choices    four, of the field kind "option" (departure 2): predictors, v, weights, penalty
    settings   none
    engine     four_design/engine.R: the design turned into Synth's arguments (four_translate.R), then Synth unchanged
               (synth_fit.R) or the native engine (scm_native.R)
    skills     scm-four-design, and scm-four-configure (departure 1), built and tested, never run

    from notebooks/sc_skills/:  python scs_build.py ../../scm_simulation/four_design ../scm_simulation/four_design/skills
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                          # notebooks/scm_simulation/four_design
SIM = HERE.parent
SC = SIM.parent / "sc_skills"
if str(SC) not in sys.path:
    sys.path.insert(0, str(SC))

import scs_method                                                                  # noqa: E402
from scs_method import Field, Check, Spec, Engine, Texts, MethodPackage            # noqa: E402

base = scs_method.load_module("../../scm_simulation")

DONORS = base.DONORS
USES = base.USES
CONTEXTS = base.CONTEXTS
PRE = list(range(1, 41))
RSCRIPT = base.RSCRIPT


def _positive_list(v):
    return (isinstance(v, list) and len(v) > 0
            and all(not isinstance(x, bool) and isinstance(x, (int, float)) and x > 0 for x in v))


# ----------------------------------------------------------------------------- the four design choices
COV = Field(None, None, "list of whole periods from 1 to 40: the periods over which Z1 and Z2 are each averaged; one "
                        "period gives the value in that period")
SUMMARIES = Field(None, None, "list of objects, each mapping periods (text keys from \"1\" to \"40\") to weights, at "
                              "least one weight not 0: each summary is, for every unit, the sum of its weights times "
                              "the unit's outcome in those periods")
CHOICES = {
    "predictors": Field("option", {
        "every_outcome": {"values": {"covariate_periods": COV}, "example": {"covariate_periods": [40]}},
        "summaries": {"values": {"covariate_periods": COV, "summaries": SUMMARIES},
                      "example": {"covariate_periods": [40],
                                  "summaries": [{"40": 1}, {str(t): 0.025 for t in PRE}]}}},
        "every_outcome = Z1, Z2 and the outcome in each pre-treatment period; summaries = Z1, Z2 and a few linear "
        "combinations of pre-treatment outcomes (p. 496)"),
    "v": Field("option", {
        "judgement": {"values": {"values": Field("positive_list", _positive_list,
                                                 "list of numbers above 0: one weight for each predictor, in the "
                                                 "order Z1, Z2, then the outcome predictors in their order")},
                      "example": {"values": [1.0] * 42}},
        "pre_period_fit": {"values": {}, "example": {}},
        "validation": {"values": {"training_end": Field("int", lambda v: 1 <= v <= 39,
                                                        "whole period from 1 to 39: the last training period s; the "
                                                        "donor weights are fitted on periods 1 to s and V is chosen "
                                                        "to fit the outcome in periods s + 1 to 40")},
                       "example": {"training_end": 30}}},
        "judgement = V set by the given weights; pre_period_fit = V chosen to fit the treated unit's outcome in periods "
        "1 to 40; validation = V chosen on a validation period after a training period (p. 496)"),
    "weights": Field("option", {"convex": {"values": {}, "example": {}},
                                "extrapolation": {"values": {}, "example": {}}},
                     "convex = each donor weight at least 0 and the weights summing to one (the paper's definition "
                     "and Synth); extrapolation = the weights sum to one and any weight may be below 0 or above one "
                     "(p. 496)"),
    "penalty": Field("option", {
        "none": {"values": {}, "example": {}},
        "distance": {"values": {"lambda": Field("float", lambda v: v >= 0,
                                                "number of 0 or above: how strongly distant donors are penalized")},
                     "example": {"lambda": 0.1}}},
        "none = no penalty (Synth); distance = lambda times the sum over donors of the donor's weight times its "
        "V-weighted squared distance from the treated unit's scaled predictors (p. 496)"),
}
REFERENCE = {"predictors": {"option": "every_outcome", "covariate_periods": [40]}, "v": {"option": "pre_period_fit"},
             "weights": {"option": "convex"}, "penalty": {"option": "none"}}
RULES = [
    'Every period is a whole number from 1 to 40, a number in `covariate_periods` and a text key such as "40" in a '
    "summary.",
    "Each choice holds `option` and exactly the values that option carries.",
    "Under `v` `validation`, every period of `covariate_periods` and of the summaries is at most `training_end`, and "
    "`every_outcome` uses the periods 1 to `training_end`.",
    "Under `v` `judgement`, `values` has one number for each predictor, Z1, Z2, then each outcome predictor in its "
    "order (40 outcome predictors under `every_outcome`, one per summary under `summaries`).",
    "`penalty` `distance` requires `weights` `convex`.",
]


def _period_list_ok(v):
    return (isinstance(v, list) and len(v) > 0 and all(not isinstance(x, bool) and isinstance(x, (int, float))
                                                       and float(x).is_integer() and 1 <= x <= 40 for x in v))


def _summary_ok(s):
    if not isinstance(s, dict) or not s:
        return False
    keys = all(isinstance(k, str) and re.fullmatch(r"[1-9][0-9]*", k) and 1 <= int(k) <= 40 for k in s)
    nums = all(not isinstance(w, bool) and isinstance(w, (int, float)) for w in s.values())
    return keys and nums and any(w != 0 for w in s.values())


def _rules(given, out, ctx, errors):
    """Rules 1, 3, 4 and 5, on the choices that are well formed (the R mirror is four_fields.R check_four)."""
    p, v = out.get("predictors"), out.get("v")
    if p is not None:
        if not _period_list_ok(p["covariate_periods"]):
            errors.append("field 'predictors.covariate_periods': expected a non-empty list of whole periods from 1 "
                          "to 40")
            p = None
        elif p["option"] == "summaries" and not (isinstance(p["summaries"], list) and p["summaries"]
                                                 and all(_summary_ok(s) for s in p["summaries"])):
            errors.append("field 'predictors.summaries': expected a non-empty list of objects, each mapping periods "
                          'from "1" to "40" to numbers, at least one not 0')
            p = None
    if p is not None and v is not None and v["option"] == "validation":
        used = [float(x) for x in p["covariate_periods"]]
        if p["option"] == "summaries":
            used += [float(k) for s in p["summaries"] for k in s]
        if any(x > v["training_end"] for x in used):
            errors.append("fields 'predictors','v': under validation every period of the predictors is at most "
                          "training_end")
    if p is not None and v is not None and v["option"] == "judgement":
        n = 2 + (len(p["summaries"]) if p["option"] == "summaries" else 40)
        if len(v["values"]) != n:
            errors.append(f"field 'v.values': {len(v['values'])} numbers for {n} predictors")
    w, pen = out.get("weights"), out.get("penalty")
    if w is not None and pen is not None and pen["option"] == "distance" and w["option"] == "extrapolation":
        errors.append("fields 'penalty','weights': penalty distance requires weights convex")


SPEC = Spec(settings={}, choices=CHOICES, optional={}, rules=RULES, configure_rules=[],
            checks=[Check(_rules, False)], reference=dict(REFERENCE), configure_label="scm-four-configure")


# ----------------------------------------------------------------------------- engine (R, run outside the box)
def _run_engine(engine, ctx, spec):
    """four_design/engine.R <synth|native|auto> <spec.json> <panel> -> the fit record."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "spec.json"
        sp.write_text(json.dumps(spec))
        p = subprocess.run([str(RSCRIPT), str(HERE / "engine.R"), engine, str(sp), ctx["panel"]],
                           capture_output=True, text=True, cwd=HERE,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    if p.returncode != 0:
        raise RuntimeError(f"four_design/engine.R {engine} failed: {p.stderr.strip()[-500:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


ENGINE = Engine(reference=lambda ctx, spec: _run_engine("synth", ctx, spec),
                native=lambda ctx, spec, forced: _run_engine("native" if forced else "auto", ctx, spec))


# ----------------------------------------------------------------------------- texts
FIT_R = """#!/usr/bin/env Rscript
# Fit the synthetic control method at the design in a file:
#   Rscript scripts/fit.R %(file)s
# The design is turned into the arguments of the Synth R package, version 1.1-10 (four_translate.R): Z1 and Z2
# averaged over covariate_periods, the outcome predictors, all 29 donors, V, and Synth's default optimizers and solver
# settings. Synth fits it unchanged when the weights are convex and the penalty is none, and Synth's code with the
# paper's alternative (scm_native.R) otherwise. The arguments of dataprep() that describe the data file are filled
# here, the same for every file: dependent "Y_observed", unit.variable unit numbers (treated 1, donor_jj jj + 1),
# time.variable "period", treatment.identifier "treated", unit.names.variable "unit_id", time.plot every period of the
# file, verbose FALSE.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "four_fields.R"))
source(file.path(here, "synth_fit.R"))
source(file.path(here, "scm_native.R"))
source(file.path(here, "four_translate.R"))
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
s <- four_to_synth(given)
rec <- four_fit(d, s)
rec$design <- given
rec$file <- basename(spec_file)
log <- Sys.getenv("FIT_LOG")                         # the record of this fit, for the run log
if (nzchar(log)) cat(scm_json(rec), "\\n", file = log, append = TRUE, sep = "")

cat(if (four_synth_choices(s)) "fit: Synth 1.1-10 dataprep() and synth(), unchanged\\n" else
      sprintf("fit: Synth's code with weights %%s and penalty %%s\\n", s$weights, s$interpolation_penalty))
cat(sprintf("predictors and their weights in V (%%d):\\n", length(rec$v)))
for (i in seq_along(rec$v)) cat(sprintf("  %%-36s %%.6f\\n", rec$predictor_names[[i]], rec$v[[i]]))
cat(sprintf("donor weights (%%d donors in the pool):\\n", length(rec$weights)))
for (i in seq_along(rec$weights)) cat(sprintf("  %%-10s %%.6f\\n", rec$donor_ids[[i]], rec$weights[[i]]))
ssr <- unlist(s$time_optimize_ssr)
cat(sprintf("pre-treatment fit: mean squared prediction error over periods %%d to %%d %%.6f\\n", min(ssr), max(ssr),
            rec$fit_mse))
cat("period, treated unit observed, synthetic control, gap (observed minus synthetic control):\\n")
for (p in names(rec$predictions))
  cat(sprintf("  %%3s %%12.6f %%12.6f %%12.6f\\n", p, rec$predictions[[p]] + rec$gaps[[p]], rec$predictions[[p]],
              rec$gaps[[p]]))
"""

VALIDATE_CONFIGURE = """errors <- check_four(given, "configure")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\\n", sep = "")
  quit(status = 1)
}
given <- FOUR_REFERENCE                   # the design this skill fixes"""

VALIDATE_DESIGN = """errors <- check_four(given, "design")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\\n", sep = "")
  quit(status = 1)
}"""

_FIT_PHRASE = "the pre-treatment fit over `time_optimize_ssr`"


def _workflow(spec_file, what, why):
    text = base._workflow(spec_file, what, why)
    assert _FIT_PHRASE in text
    return text.replace(_FIT_PHRASE, "the pre-treatment fit over the periods V is fitted to")


TEXTS = Texts(
    method_name="synthetic control",
    skill_prefix="scm-four",
    method_md=base.METHOD_MD,
    configure_description=("Estimate the effect of a treatment on one treated unit from a panel of that unit and "
                           "untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller "
                           "(2010) at a fixed design, by running the bundled fitter. Use this whenever one treated "
                           "unit's outcome without treatment is to be estimated from donors with that design."),
    design_description=("Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated "
                        "donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by "
                        "designing the method from the paper's options for the predictors, the predictor weights V, "
                        "the donor weights and a distance penalty, and running the bundled fitter. Use this whenever "
                        "one treated unit's outcome without treatment is to be estimated from donors and the method's "
                        "design is yours to choose."),
    configure_intro=base.INTRO,
    design_intro=base.INTRO,
    fixed_choices=("Z1 and Z2 in period 40 and the outcome in each pre-treatment period as the predictors, V chosen to "
                   "fit the outcome in periods 1 to 40, every donor weight at least 0 and the weights summing to one, "
                   "no penalty, all 29 donors, and Synth's default optimizers and solver settings. Everything else is "
                   "Synth's own code, unchanged."),
    settings_list="There are none, so `settings.json` holds an empty object, `{}`.",
    settings_rules=("`references/settings.md` gives the shape of the file. The fitter refuses any field in it and "
                    "names the field."),
    configure_workflow=_workflow("settings.json", "settings", "why these settings"),
    design_overview=("`spec.json` holds four choices, the paper's options for building the synthetic control (Abadie, "
                     "Diamond and Hainmueller 2010, p. 496). Each choice is an object with `option`, one of its "
                     "options, and the values that option carries. Every fit uses all 29 donors, and the Synth R "
                     "package fits it with its default optimizers and solver settings."),
    choices_list=(
        "- `predictors`, `every_outcome` (Z1, Z2 and the outcome in each pre-treatment period, \"the values of the "
        "outcome variable for all the available preintervention periods\") or `summaries` (Z1, Z2 and \"only a few "
        "linear combination of preintervention outcomes\", each summary a weight for each period it uses). Both carry "
        "`covariate_periods`, the periods over which Z1 and Z2 are averaged.\n"
        "- `v`, how the predictor weights V are chosen. `judgement` sets V from \"subjective assessments of the "
        "predictive power of the variables\" and carries `values`, one weight per predictor. `pre_period_fit` chooses "
        "V so that the synthetic control fits the treated unit's outcome in the pre-treatment periods. `validation` "
        "fits the donor weights on \"an initial training period\" ending at `training_end`, which it carries, and "
        "chooses V to fit the outcome in \"a subsequent validation period\", the periods after `training_end`.\n"
        "- `weights`, `convex` (every donor weight at least 0, summing to one, the paper's definition and Synth's) or "
        "`extrapolation` (summing to one, any weight may be below 0 or above one, \"at the cost of allowing "
        "extrapolation\").\n"
        "- `penalty`, `none` or `distance`, \"penalty terms specified as increasing functions of the distances\" "
        "between the treated unit and the donors. `distance` carries `lambda` and adds `lambda` times each donor's "
        "weight times its squared distance from the treated unit's predictors, so that distant donors are kept out."),
    design_rules=("`references/settings.md` lists every option, the values it carries and the rules that tie them. "
                  "`distance` requires `convex` weights, and under `validation` every period the predictors use is at "
                  "most `training_end`. The fitter refuses a specification that breaks a rule and names the field, "
                  "so read the rules before writing the file. There are no defaults."),
    design_workflow=_workflow("spec.json", "design", "why this design"),
    scripts={"show_data.py": base.SHOW_DATA},
    fit_script=FIT_R,
    fit_script_name="fit.R",
    fixed_comment="# the design this skill fixes",
    validate_configure=VALIDATE_CONFIGURE,
    validate_design=VALIDATE_DESIGN,
    authors=base.TEXTS.authors,
    paper_title=base.TEXTS.paper_title,
    profile_code=base.TEXTS.profile_code,
    profile_focus=base.TEXTS.profile_focus,
    profile_code_paths=base.TEXTS.profile_code_paths)

PACKAGE = MethodPackage(
    name="scm-four",
    paper=base.PAPER,
    reference_code=base.PACKAGE.reference_code,
    data=base.PACKAGE.data,
    spec=SPEC,
    engine=ENGINE,
    texts=TEXTS,
    skill_modules=[("four_fields.R", HERE / "four_fields.R"), ("four_translate.R", HERE / "four_translate.R"),
                   ("synth_fit.R", SIM / "synth_fit.R"), ("scm_native.R", SIM / "scm_native.R")],
)
