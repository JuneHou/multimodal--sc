# scm-four-design: every text a model reads

Built from `notebooks/scm_simulation/four_design/method.py` into `notebooks/scm_simulation/four_design/skills/scm-four-design/`. The build also makes `scm-four-configure`, which is never run, so it is not shown.

## 1. System message (the pipeline's, unchanged)

```
You are operating a bash shell in a working directory that contains a skill: files with instructions (SKILL.md), reference documents and scripts for one task. Each reply you send is executed as one bash command block, and its output is returned to you in the next message. Nothing outside the working directory is visible and there is no network. Read the skill's instructions and follow them.
```

## 2. First message, case dgp1 (dgp2 differs only in the file name)

```
Estimate the effect of a treatment on the unit `treated` in the panel `dgp1_panel.csv`. The unit `treated` was treated in periods 41 to 60, and the units `donor_01` to `donor_29` were never treated. Estimate the treated unit's outcome `Y_observed` without treatment in each period from 41 to 60, and the average effect over those periods.
The working directory holds a skill for this task:
  scm-four-design. Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by designing the method from the paper's options for the predictors, the predictor weights V, the donor weights and a distance penalty, and running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors and the method's design is yours to choose.
Its instructions are in SKILL.md. Read that file first and follow it. Reply with one ```bash block per turn. It is run by bash in the working directory and its output comes back in the next message. Finish by writing answer.json as the skill describes.
```

## 3. SKILL.md

```markdown
---
name: scm-four-design
description: Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by designing the method from the paper's options for the predictors, the predictor weights V, the donor weights and a distance penalty, and running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors and the method's design is yours to choose.
---

# Design and run the synthetic control

A weighted combination of donor units, the synthetic control, is chosen so that it resembles the treated unit before treatment. Its outcome after treatment estimates what the treated unit's outcome would have been without treatment, and the difference is the effect. `references/method.md` describes the construction. Read it if anything below is unclear.

## Everything is yours to decide

`spec.json` holds four choices, the paper's options for building the synthetic control (Abadie, Diamond and Hainmueller 2010, p. 496). Each choice is an object with `option`, one of its options, and the values that option carries. Every fit uses all 29 donors, and the Synth R package fits it with its default optimizers and solver settings.

- `predictors`, `every_outcome` (Z1, Z2 and the outcome in each pre-treatment period, "the values of the outcome variable for all the available preintervention periods") or `summaries` (Z1, Z2 and "only a few linear combination of preintervention outcomes", each summary a weight for each period it uses). Both carry `covariate_periods`, the periods over which Z1 and Z2 are averaged.
- `v`, how the predictor weights V are chosen. `judgement` sets V from "subjective assessments of the predictive power of the variables" and carries `values`, one weight per predictor. `pre_period_fit` chooses V so that the synthetic control fits the treated unit's outcome in the pre-treatment periods. `validation` fits the donor weights on "an initial training period" ending at `training_end`, which it carries, and chooses V to fit the outcome in "a subsequent validation period", the periods after `training_end`.
- `weights`, `convex` (every donor weight at least 0, summing to one, the paper's definition and Synth's) or `extrapolation` (summing to one, any weight may be below 0 or above one, "at the cost of allowing extrapolation").
- `penalty`, `none` or `distance`, "penalty terms specified as increasing functions of the distances" between the treated unit and the donors. `distance` carries `lambda` and adds `lambda` times each donor's weight times its squared distance from the treated unit's predictors, so that distant donors are kept out.

`references/settings.md` lists every option, the values it carries and the rules that tie them. `distance` requires `convex` weights, and under `validation` every period the predictors use is at most `training_end`. The fitter refuses a specification that breaks a rule and names the field, so read the rules before writing the file. There are no defaults.

## Workflow

1. Look at the data: `python scripts/show_data.py`, and `python scripts/show_data.py Z1` or `Z2` for the
   covariates. It shows the treated unit and the donors period by period.
2. Read `references/method.md` and `references/settings.md`. Decide the design from the data and the method.
3. Write `spec.json`, then run `Rscript scripts/fit.R spec.json`. It prints the predictors and V, the donor
   weights, the pre-treatment fit over the periods V is fitted to, and for every period the treated unit's observed outcome,
   the synthetic control and their gap.
4. You may change the design and run again. Every run is recorded. The pre-treatment fit is an in-sample quantity, and
   a closer fit before treatment is not by itself a better estimate after it.
5. Finish by writing `answer.json`, with one value for every treated period the task names:
   ```json
   {"counterfactual": {"<period>": <value>, ...}, "att": <value>, "rationale": "<why this design, in a few sentences>"}
   ```
   `counterfactual` is the treated unit's outcome without treatment in each treated period, and `att` the average
   effect over those periods. The values you submit are your answer. If they differ from the fitter's, say why.
```

## 4. references/method.md, identical to scm-design's

```markdown
# Synthetic control for one treated unit

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
```

## 5. references/settings.md

```markdown
# Fields for scm-four-design

`spec.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|

## Choices

| field | options | meaning |
|---|---|---|
| `predictors` | `every_outcome` with `covariate_periods`, `summaries` with `covariate_periods` and `summaries` | every_outcome = Z1, Z2 and the outcome in each pre-treatment period; summaries = Z1, Z2 and a few linear combinations of pre-treatment outcomes (p. 496) |
| `v` | `judgement` with `values`, `pre_period_fit`, `validation` with `training_end` | judgement = V set by the given weights; pre_period_fit = V chosen to fit the treated unit's outcome in periods 1 to 40; validation = V chosen on a validation period after a training period (p. 496) |
| `weights` | `convex`, `extrapolation` | convex = each donor weight at least 0 and the weights summing to one (the paper's definition and Synth); extrapolation = the weights sum to one and any weight may be below 0 or above one (p. 496) |
| `penalty` | `none`, `distance` with `lambda` | none = no penalty (Synth); distance = lambda times the sum over donors of the donor's weight times its V-weighted squared distance from the treated unit's scaled predictors (p. 496) |

## Values the options carry

| choice | option | value | type and range | meaning |
|---|---|---|---|---|
| `predictors` | `every_outcome` | `covariate_periods` | list of whole periods from 1 to 40 | the periods over which Z1 and Z2 are each averaged; one period gives the value in that period |
| `predictors` | `summaries` | `covariate_periods` | list of whole periods from 1 to 40 | the periods over which Z1 and Z2 are each averaged; one period gives the value in that period |
| `predictors` | `summaries` | `summaries` | list of objects, each mapping periods (text keys from "1" to "40") to weights, at least one weight not 0 | each summary is, for every unit, the sum of its weights times the unit's outcome in those periods |
| `v` | `judgement` | `values` | list of numbers above 0 | one weight for each predictor, in the order Z1, Z2, then the outcome predictors in their order |
| `v` | `validation` | `training_end` | whole period from 1 to 39 | the last training period s; the donor weights are fitted on periods 1 to s and V is chosen to fit the outcome in periods s + 1 to 40 |
| `penalty` | `distance` | `lambda` | number of 0 or above | how strongly distant donors are penalized |

## Rules that tie the fields together

- Every period is a whole number from 1 to 40, a number in `covariate_periods` and a text key such as "40" in a summary.
- Each choice holds `option` and exactly the values that option carries.
- Under `v` `validation`, every period of `covariate_periods` and of the summaries is at most `training_end`, and `every_outcome` uses the periods 1 to `training_end`.
- Under `v` `judgement`, `values` has one number for each predictor, Z1, Z2, then each outcome predictor in its order (40 outcome predictors under `every_outcome`, one per summary under `summaries`).
- `penalty` `distance` requires `weights` `convex`.

## Shape of the file (every value is yours)

```json
{
 "predictors": {
  "option": "<every_outcome|summaries>",
  "covariate_periods": "list of whole periods from 1 to 40, only with every_outcome or summaries",
  "summaries": "list of objects, each mapping periods (text keys from \"1\" to \"40\") to weights, at least one weight not 0, only with summaries"
 },
 "v": {
  "option": "<judgement|pre_period_fit|validation>",
  "values": "list of numbers above 0, only with judgement",
  "training_end": "whole period from 1 to 39, only with validation"
 },
 "weights": {
  "option": "<convex|extrapolation>"
 },
 "penalty": {
  "option": "<none|distance>",
  "lambda": "number of 0 or above, only with distance"
 }
}
```
```

## 6a. scripts/show_data.py, identical to scm-design's

```python
"""Show the panel in the working directory: the units, the periods and one variable by period and unit.

    python scripts/show_data.py            Y_observed
    python scripts/show_data.py Z1         another column (Z1, Z2 or Y_observed)
"""
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
print(f"\n{var} by period (rows) and unit (columns):")
wide = d.pivot(index="period", columns="unit_id", values=var)[units]
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(wide.round(3).to_string())
```

## 6b. scripts/fit.R

```r
#!/usr/bin/env Rscript
# Fit the synthetic control method at the design in a file:
#   Rscript scripts/fit.R spec.json
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
spec_file <- if (length(args)) args[1] else "spec.json"
given <- tryCatch(fromJSON(spec_file, simplifyVector = FALSE), error = function(e) {
  cat("refused: ", spec_file, " is not readable JSON (", conditionMessage(e), ")\n", sep = "")
  quit(status = 1)
})
errors <- check_four(given, "design")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\n", sep = "")
  quit(status = 1)
}
panel <- list.files(".", pattern = "_panel[.]csv$")
if (length(panel) != 1) stop("expected one *_panel.csv file in the working directory")
d <- scm_data(panel)
s <- four_to_synth(given)
rec <- four_fit(d, s)
rec$design <- given
rec$file <- basename(spec_file)
log <- Sys.getenv("FIT_LOG")                         # the record of this fit, for the run log
if (nzchar(log)) cat(scm_json(rec), "\n", file = log, append = TRUE, sep = "")

cat(if (four_synth_choices(s)) "fit: Synth 1.1-10 dataprep() and synth(), unchanged\n" else
      sprintf("fit: Synth's code with weights %s and penalty %s\n", s$weights, s$interpolation_penalty))
cat(sprintf("predictors and their weights in V (%d):\n", length(rec$v)))
for (i in seq_along(rec$v)) cat(sprintf("  %-36s %.6f\n", rec$predictor_names[[i]], rec$v[[i]]))
cat(sprintf("donor weights (%d donors in the pool):\n", length(rec$weights)))
for (i in seq_along(rec$weights)) cat(sprintf("  %-10s %.6f\n", rec$donor_ids[[i]], rec$weights[[i]]))
ssr <- unlist(s$time_optimize_ssr)
cat(sprintf("pre-treatment fit: mean squared prediction error over periods %d to %d %.6f\n", min(ssr), max(ssr),
            rec$fit_mse))
cat("period, treated unit observed, synthetic control, gap (observed minus synthetic control):\n")
for (p in names(rec$predictions))
  cat(sprintf("  %3s %12.6f %12.6f %12.6f\n", p, rec$predictions[[p]] + rec$gaps[[p]], rec$predictions[[p]],
              rec$gaps[[p]]))
```

## 6c. scripts/four_fields.R (new, copied unchanged from notebooks/scm_simulation/four_design/)

```r
# The checks of spec.json in scm-four-design (and of settings.json in scm-four-configure), the same rules as the
# Python checks of four_design/method.py (scs_spec.validate with the field kind "option", and _rules). A refusal names
# the same fields in both, which four_design/checks.py tests. Plan use-nature-skills-analyze-wiggly-coral.md, section
# "Fields of scm-four-design".
suppressPackageStartupMessages(library(jsonlite))

FOUR_DONORS <- sprintf("donor_%02d", 1:29)
FOUR_OPTIONS <- list(
  predictors = list(every_outcome = c("covariate_periods"), summaries = c("covariate_periods", "summaries")),
  v = list(judgement = c("values"), pre_period_fit = character(0), validation = c("training_end")),
  weights = list(convex = character(0), extrapolation = character(0)),
  penalty = list(none = character(0), distance = c("lambda")))
# what scm-four-configure fixes: the package's reference design
FOUR_REFERENCE <- list(predictors = list(option = "every_outcome", covariate_periods = list(40L)),
                       v = list(option = "pre_period_fit"), weights = list(option = "convex"),
                       penalty = list(option = "none"))

four_is_object <- function(v) is.list(v) && !is.null(names(v))
four_is_array <- function(v) is.list(v) && is.null(names(v))
four_is_number <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
four_shown <- function(v) if (is.null(v)) "null" else as.character(toJSON(v, auto_unbox = TRUE, null = "null"))

four_periods_ok <- function(v) four_is_array(v) && length(v) > 0 &&
  all(vapply(v, function(x) four_is_number(x) && x == round(x) && x >= 1 && x <= 40, TRUE))

four_summary_ok <- function(s) {
  if (!four_is_object(s) || !length(s)) return(FALSE)
  keys_ok <- all(grepl("^[1-9][0-9]*$", names(s))) && all(as.integer(names(s)) %in% 1:40)
  keys_ok && all(vapply(s, four_is_number, TRUE)) && any(vapply(s, function(w) w != 0, TRUE))
}

# One choice: NULL when it is well formed, else its error (the first, as the Python typing stops at the first).
four_check_choice <- function(f, v) {
  opts <- FOUR_OPTIONS[[f]]
  if (!(four_is_object(v) && "option" %in% names(v)))
    return(sprintf("field '%s': expected an object with \"option\" and the values that option carries, got %s",
                   f, four_shown(v)))
  o <- v$option
  if (!(is.character(o) && length(o) == 1 && o %in% names(opts)))
    return(sprintf("field '%s': option %s is not one of %s", f, four_shown(o), paste(names(opts), collapse = ", ")))
  got <- setdiff(names(v), "option")
  want <- opts[[o]]
  if (!setequal(got, want))
    return(sprintf("field '%s': option '%s' carries %s, got %s", f, o,
                   if (length(want)) paste0("'", want, "'", collapse = ", ") else "no values",
                   if (length(got)) paste0("'", got, "'", collapse = ", ") else "none"))
  if (f == "v" && o == "judgement") {
    x <- v$values
    if (!(four_is_array(x) && length(x) > 0 && all(vapply(x, function(y) four_is_number(y) && y > 0, TRUE))))
      return(sprintf("field 'v.values': expected a non-empty list of numbers > 0, got %s", four_shown(x)))
  }
  if (f == "v" && o == "validation") {
    x <- v$training_end
    if (!(four_is_number(x) && x == round(x))) return(sprintf("field 'v.training_end': expected an integer, got %s",
                                                             four_shown(x)))
    if (!(x >= 1 && x <= 39)) return(sprintf("field 'v.training_end': %s is outside the allowed range", four_shown(x)))
  }
  if (f == "penalty" && o == "distance") {
    x <- v$lambda
    if (!four_is_number(x)) return(sprintf("field 'penalty.lambda': expected a number, got %s", four_shown(x)))
    if (!(x >= 0)) return(sprintf("field 'penalty.lambda': %s is outside the allowed range", four_shown(x)))
  }
  NULL
}

check_four <- function(given, mode) {
  if (!four_is_object(given)) return("spec must be a JSON object")     # {} is an object here, [] is not
  if (mode == "configure") {
    present <- intersect(names(given), names(FOUR_OPTIONS))
    if (length(present))
      return(paste0("protocol violation: scm-four-configure takes Class-1 fields only, but got ",
                    paste(present, collapse = ", ")))
    unknown <- names(given)
    return(if (length(unknown)) paste0("unknown field(s): ", paste(unknown, collapse = ", ")) else character(0))
  }
  errors <- character(0)
  missing <- setdiff(names(FOUR_OPTIONS), names(given))
  if (length(missing)) errors <- c(errors, paste0("missing required field(s): ", paste(missing, collapse = ", ")))
  unknown <- setdiff(names(given), names(FOUR_OPTIONS))
  if (length(unknown)) errors <- c(errors, paste0("unknown field(s): ", paste(unknown, collapse = ", ")))
  ok <- list()
  for (f in names(FOUR_OPTIONS)) {
    if (!(f %in% names(given))) next
    e <- four_check_choice(f, given[[f]])
    if (is.null(e)) ok[[f]] <- given[[f]] else errors <- c(errors, e)
  }
  p <- ok$predictors; v <- ok$v
  # rule 1: every period is a whole number from 1 to 40
  if (!is.null(p)) {
    if (!four_periods_ok(p$covariate_periods)) {
      errors <- c(errors, "field 'predictors.covariate_periods': expected a non-empty list of whole periods from 1 to 40")
      p <- NULL
    } else if (identical(p$option, "summaries") &&
               !(four_is_array(p$summaries) && length(p$summaries) > 0 && all(vapply(p$summaries, four_summary_ok, TRUE)))) {
      errors <- c(errors, paste0("field 'predictors.summaries': expected a non-empty list of objects, each mapping ",
                                 "periods from \"1\" to \"40\" to numbers, at least one not 0"))
      p <- NULL
    }
  }
  # rule 3: under validation every period of the predictors is at most training_end
  if (!is.null(p) && !is.null(v) && identical(v$option, "validation")) {
    used <- as.numeric(unlist(p$covariate_periods))
    if (identical(p$option, "summaries")) used <- c(used, as.numeric(unlist(lapply(p$summaries, names))))
    if (any(used > v$training_end))
      errors <- c(errors, "fields 'predictors','v': under validation every period of the predictors is at most training_end")
  }
  # rule 4: under judgement, one number in values for each predictor
  if (!is.null(p) && !is.null(v) && identical(v$option, "judgement")) {
    n <- 2 + if (identical(p$option, "summaries")) length(p$summaries) else 40
    if (length(v$values) != n)
      errors <- c(errors, sprintf("field 'v.values': %d numbers for %d predictors", length(v$values), n))
  }
  # rule 5: penalty distance requires weights convex
  if (!is.null(ok$penalty) && !is.null(ok$weights) && identical(ok$penalty$option, "distance") &&
      identical(ok$weights$option, "extrapolation"))
    errors <- c(errors, "fields 'penalty','weights': penalty distance requires weights convex")
  errors
}
```

## 6c. scripts/four_translate.R (new, copied unchanged from notebooks/scm_simulation/four_design/)

```r
# scm-four-design: the four design choices turned into Synth's arguments, and the fit (plan
# use-nature-skills-analyze-wiggly-coral.md, "How the fit script turns the spec into Synth's arguments").
#   four_to_synth(g)        spec.json (checked) -> the translated spec: Synth's arguments, outcome_combinations and the
#                           two choices weights and interpolation_penalty (with penalty_lambda under distance)
#   four_fit(d, s)          the fit of a translated spec: each outcome combination added to the panel as a column and
#                           appended as a special predictor, then Synth when the weights are convex and the penalty is
#                           none, the native engine (scm_native.R) otherwise. The record keeps s as given here.
# Every fit uses all 29 donors and Synth's default optimizers and solver settings (synth.R lines 8 to 13).
# Needs synth_fit.R and scm_native.R sourced first.

FOUR_ALL_DONORS <- sprintf("donor_%02d", 1:29)

four_to_synth <- function(g) {
  p <- g$predictors; v <- g$v
  last <- if (identical(v$option, "validation")) as.integer(v$training_end) else 40L
  s <- list(
    predictors = list("Z1", "Z2"), predictors_op = "mean",
    time_predictors_prior = lapply(unlist(p$covariate_periods), as.numeric),
    special_predictors = if (identical(p$option, "every_outcome"))
      lapply(seq_len(last), function(t) list(variable = "Y_observed", periods = list(t), op = "mean")) else list(),
    outcome_combinations = if (identical(p$option, "summaries")) p$summaries else list(),
    controls_identifier = as.list(FOUR_ALL_DONORS),
    custom_v = if (identical(v$option, "judgement")) lapply(unlist(v$values), as.numeric) else NULL,
    time_optimize_ssr = as.list(if (identical(v$option, "validation")) (last + 1L):40L else 1:40),
    optimxmethod = list("Nelder-Mead", "BFGS"), genoud = FALSE, quadopt = "ipop",
    margin_ipop = 0.0005, sigf_ipop = 5, bound_ipop = 10,
    weights = g$weights$option, interpolation_penalty = g$penalty$option)
  if (identical(g$penalty$option, "distance")) s$penalty_lambda <- as.numeric(g$penalty$lambda)
  s
}

# Each combination m becomes the column outcome_combination_m: for each unit, the sum of the weights times its
# outcome in their periods, constant over the periods, and a special predictor averaged over the combination's first
# period. A spec with no outcome_combinations is returned unchanged.
four_add_combinations <- function(d, s) {
  comb <- s$outcome_combinations
  if (!length(comb)) return(list(d = d, s = s))
  for (m in seq_along(comb)) {
    w <- comb[[m]]
    per <- as.integer(names(w)); k <- as.numeric(unlist(w))
    col <- sprintf("outcome_combination_%d", m)
    val <- vapply(split(d, d$unit_id), function(u) sum(k * u$Y_observed[match(per, u$period)]), 0)
    d[[col]] <- unname(val[d$unit_id])
    s$special_predictors <- c(s$special_predictors, list(list(variable = col, periods = list(min(per)), op = "mean")))
  }
  list(d = d, s = s)
}

four_synth_choices <- function(s) {
  w <- if (is.null(s$weights)) "convex" else s$weights
  pen <- if (is.null(s$interpolation_penalty)) "none" else s$interpolation_penalty
  identical(w, "convex") && identical(pen, "none")
}

four_fit <- function(d, s) {
  a <- four_add_combinations(d, s)
  rec <- if (four_synth_choices(s)) fit_synth(a$d, a$s) else fit_native(a$d, a$s)
  rec$spec <- s
  rec
}
```

## 6d. scripts/synth_fit.R (identical to scm-design's, copied unchanged from notebooks/scm_simulation/)

```r
# Synth 1.1-10, run unchanged at a spec: dataprep() then synth(), and the fit record.
# Shared by scripts/fit.R in both skills and by engine.R (the package's reference engine).
#
# The arguments that describe the data file are filled the same way for every spec:
#   dependent "Y_observed", unit.variable a column of unit numbers added in memory (treated 1, donor_jj jj + 1),
#   time.variable "period", treatment.identifier "treated", unit.names.variable "unit_id",
#   time.plot every period of the file, verbose FALSE.
suppressPackageStartupMessages({library(jsonlite); library(Synth)})

scm_data <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  donors <- sort(unique(d$unit_id[d$unit_id != "treated"]))
  d$unit_num <- match(d$unit_id, c("treated", donors))
  d
}

scm_dataprep <- function(d, s) {
  special <- lapply(s$special_predictors, function(e) list(e$variable, as.numeric(unlist(e$periods)), e$op))
  dataprep(foo = d,
           predictors = if (length(s$predictors)) unlist(s$predictors) else NULL,
           predictors.op = s$predictors_op,
           special.predictors = if (length(special)) special else NULL,
           dependent = "Y_observed", unit.variable = "unit_num", time.variable = "period",
           treatment.identifier = "treated", controls.identifier = unlist(s$controls_identifier),
           time.predictors.prior = as.numeric(unlist(s$time_predictors_prior)),
           time.optimize.ssr = as.numeric(unlist(s$time_optimize_ssr)),
           time.plot = sort(unique(d$period)), unit.names.variable = "unit_id")
}

scm_custom_v <- function(s) if (is.null(s$custom_v)) NULL else as.numeric(unlist(s$custom_v))

fit_synth <- function(d, s) {
  dp <- scm_dataprep(d, s)
  so <- synth(data.prep.obj = dp, custom.v = scm_custom_v(s), optimxmethod = unlist(s$optimxmethod),
              genoud = s$genoud, quadopt = s$quadopt, Margin.ipop = s$margin_ipop, Sigf.ipop = s$sigf_ipop,
              Bound.ipop = s$bound_ipop)
  scm_record(d, dp, so, s, "synth")
}

# The fit record: donors and weights, V and its predictors, the losses, and the synthetic path and gaps over every
# period of the file, computed as Synth's help computes them (Y0plot %*% solution.w, Y1plot minus that). Vectors are
# written as JSON arrays even when they hold one value.
scm_record <- function(d, dp, so, s, engine) {
  w <- as.numeric(so$solution.w)
  path <- as.numeric(dp$Y0plot %*% so$solution.w)
  gaps <- as.numeric(dp$Y1plot) - path
  periods <- as.character(sort(unique(d$period)))
  names_by_num <- setNames(d$unit_id, d$unit_num)
  list(engine_used = engine,
       donor_ids = as.list(unname(names_by_num[rownames(so$solution.w)])),
       weights = as.list(w),
       predictor_names = as.list(names(so$solution.v)),
       v = as.list(as.numeric(so$solution.v)),
       fit_mse = as.numeric(so$loss.v),
       loss_w = as.numeric(so$loss.w),
       predictions = setNames(as.list(path), periods),
       gaps = setNames(as.list(gaps), periods),
       spec = s)
}

scm_json <- function(x) toJSON(x, auto_unbox = TRUE, digits = NA, null = "null")
```

## 6d. scripts/scm_native.R (identical to scm-design's, copied unchanged from notebooks/scm_simulation/)

```r
# The design choices Synth lacks, as a copy of Synth 1.1-10's synth.R and fn.V.R (CRAN source, code/Synth/R/) with
# a branch at each choice. With weights = "convex" and interpolation_penalty = "none" every line Synth runs is run,
# in Synth's order, so the result is Synth's.
#
#   weights = "extrapolation"        (ADH 2010 p. 496: "negative weights or weights larger than one can be used at
#                                    the cost of allowing extrapolation") the weights sum to one and have no bounds.
#                                    The V-weighted least squares under that one constraint is solved through its
#                                    linear optimality equations; MASS::ginv gives the minimum norm solution when the
#                                    solution is not unique.
#   interpolation_penalty = "distance" (p. 496: "penalty terms specified as increasing functions of the distances
#                                    between X1 and the corresponding values for the control units with positive
#                                    weights in W") adds penalty_lambda * sum_j w_j * d_j, d_j the V-weighted squared
#                                    distance between the treated unit's scaled predictors and donor j's. In ipop's
#                                    form (c'x + x'Hx/2, half of Synth's loss) this adds penalty_lambda * d / 2 to c.
#                                    Allowed with convex weights only.
# Changed lines are marked "# native". Uses scm_data, scm_dataprep, scm_custom_v and scm_record from synth_fit.R.
suppressPackageStartupMessages({library(Synth); library(kernlab); library(optimx); library(MASS)})

solve.w.native <- function(X0.scaled, X1.scaled, V, quadopt, margin.ipop, sigf.ipop, bound.ipop,   # native
                           weights, penalty.lambda) {                                               # native
    if (weights == "extrapolation") {                                                               # native
      H <- t(X0.scaled) %*% V %*% (X0.scaled)                                                       # native
      K <- rbind(cbind(H, rep(1, ncol(H))), c(rep(1, ncol(H)), 0))                                  # native
      rhs <- c(c(t(X0.scaled) %*% V %*% X1.scaled), 1)                                              # native
      return(as.matrix(as.numeric(ginv(K) %*% rhs)[1:ncol(H)]))                                     # native
    }                                                                                               # native
    # set up QP problem
    H <- t(X0.scaled) %*% V %*% (X0.scaled)
    a <- X1.scaled
    c <- -1*c(t(a) %*% V %*% (X0.scaled) )
    if (!is.null(penalty.lambda)) {                                                                 # native
      dist <- apply(X0.scaled, 2, function(x0) as.numeric(t(a - x0) %*% V %*% (a - x0)))            # native
      c <- c + penalty.lambda * dist / 2                                                            # native
    }                                                                                               # native
    A <- t(rep(1, length(c)))
    b <- 1
    l <- rep(0, length(c))
    u <- rep(1, length(c))
    r <- 0

    if (quadopt == "ipop") {
      res <- ipop(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                  margin = margin.ipop, maxiter = 1000,
                  sigf = sigf.ipop, bound = bound.ipop)
      solution.w <- as.matrix(primal(res))
    } else if (quadopt == "LowRankQP") {
      stop("LowRankQP is no longer supported; please use quadopt = \"ipop\" instead")
    } else {
      stop(sprintf("Unknown quadopt = \"%s\"; supported value is \"ipop\"", quadopt))
    }
    solution.w
}

fn.V.native <-
function(
           variables.v = stop("variables.v missing"),
           X0.scaled = stop("X0.scaled missing"),
           X1.scaled = stop("X1.scaled missing"),
           Z0 = stop("Z0 missing"),
           Z1 = stop("Z1 missing"),
           margin.ipop = 0.0005,
           sigf.ipop = 5,
           bound.ipop = 10,
           quadopt = "ipop",
           weights = "convex",                                                                      # native
           penalty.lambda = NULL                                                                    # native
           )

  {

    # check quadopt
    if (!(quadopt %in% c("ipop", "LowRankQP"))) {
      stop("option quadopt must be \"ipop\" (\"LowRankQP\" is no longer supported)")
    }

    # rescale par
    #V <- diag( abs(variables.v)/sum(abs(variables.v)) )
    V <- diag(x=as.numeric(abs(variables.v)/sum(abs(variables.v))),
              nrow=length(variables.v),ncol=length(variables.v))

    solution.w <- solve.w.native(X0.scaled, X1.scaled, V, quadopt, margin.ipop, sigf.ipop,          # native
                                 bound.ipop, weights, penalty.lambda)                               # native

    # compute losses
    loss.w <- as.numeric(t(X1.scaled - X0.scaled %*% solution.w) %*%
      (V) %*% (X1.scaled - X0.scaled %*% solution.w))

    loss.v <- as.numeric(t(Z1 - Z0 %*% solution.w) %*%
      ( Z1 - Z0 %*% solution.w ))
    loss.v <- loss.v/nrow(Z0)

    return(invisible(loss.v))
  }

synth.native <-
function(           data.prep.obj = NULL,
                      X1 = NULL,
                      X0 = NULL,
                      Z0 = NULL,
                      Z1 = NULL,
                      custom.v = NULL,
                      optimxmethod = c("Nelder-Mead","BFGS"),
                      genoud = FALSE,
                      quadopt = "ipop",
                      Margin.ipop = 0.0005,
                      Sigf.ipop = 5,
                      Bound.ipop = 10,
                      verbose = FALSE,
                      weights = "convex",                                                           # native
                      penalty.lambda = NULL,                                                        # native
                       ...
                      )
  {
    # Retrieve dataprep objects
    if (!is.null(data.prep.obj)) {
      if (verbose) cat("\nX1, X0, Z1, Z0 all come directly from dataprep object.\n\n")
      X1 <- data.prep.obj$X1
      Z1 <- data.prep.obj$Z1
      X0 <- data.prep.obj$X0
      Z0 <- data.prep.obj$Z0
    } else {
      if (verbose) cat("X1, X0, Z1, Z0 were individually input (not dataprep object).\n\n")
    }

     # routine checks
     store <- list(X1=X1,X0=X0,Z1=Z1,Z0=Z0)
     for(i in 1:4){
     if(is.null(store[[i]]))
      {stop(paste("\n",names(store)[i],"is missing \n"))}
     if(sum(is.na(store[[i]]))>0)
      {stop(paste("\n NAs in",names(store)[i],"\n"))}
     if(is.matrix(store[[i]]) == FALSE)
      {stop(paste("\n",names(store)[i],"is not a matrix object\n"))}
     }

    # geometry checks
    if(ncol(X1)!=1){stop("\n Please specify only one treated unit: X1 has to have ncol= 1")}
    if(ncol(Z1)!=1){stop("\n Please specify only one treated unit: Z1 has to have ncol= 1")}

    if(ncol(X0)<2){stop("\n Please specify at least two control units: X0 has to have ncol >= 2 ")}
    if(ncol(Z0)<2){stop("\n Please specify at least two control units: Z0 has to have ncol >= 2")}

    if(nrow(Z0)!=nrow(Z1)){stop("\n Different number of periods for treated and controls: nrow(Z0) unequal nrow(Z1)")}
    if(nrow(X0)!=nrow(X1)){stop("\n Different number of predictors for treated and controls: nrow(X0) unequal nrow(X1)")}

    if(nrow(X0)==0){stop("No predictors specified. Please specify at least one predictor")}
    if(nrow(Z0)==0){stop("No periods specified for Z1 and Z0. Please specify at least one period")}

    if(0 %in% apply(X0,1,sd))
     {stop("\n At least one predictor in X0 has no variation across control units. Please remove this predictor.")}

    # Normalize X
    nvarsV <- dim(X0)[1]
    big.dataframe <- cbind(X0, X1)
    divisor <- sqrt(apply(big.dataframe, 1, var))
    scaled.matrix <-
      t(t(big.dataframe) %*% ( 1/(divisor) *
                              diag(rep(dim(big.dataframe)[1], 1)) ))

    X0.scaled <- scaled.matrix[, 1:ncol(X0), drop = FALSE]
    X1.scaled <- scaled.matrix[, ncol(scaled.matrix)]


    # check if custom v weights are supplied or
    # if only on predictor is specified,
    # we jump to quadratic optimization over W weights
    # if not start optimization over V and W
    if(is.null(custom.v) & nrow(X0) != 1)
      {

      # two attempts for best V are made:
      # equal weights and regression based starting values
      if (verbose) cat("\n****************",
                       "\n searching for synthetic control unit  \n", "\n")

      if (genoud == TRUE) { # if user wants genoud as well
      # we run genoud first
      if (verbose) cat("\n****************",
                       "\n genoud() requested for optimization\n", "\n")

      rgV.genoud <- rgenoud::genoud(
                             fn.V.native,                                                           # native
                             nvarsV,
                             X0.scaled = X0.scaled,
                             X1.scaled = X1.scaled,
                             Z0 = Z0,
                             Z1 = Z1,
                             quadopt = quadopt,
                             margin.ipop = Margin.ipop,
                             sigf.ipop = Sigf.ipop,
                             bound.ipop = Bound.ipop,
                             weights = weights,                                                     # native
                             penalty.lambda = penalty.lambda                                        # native
                             )
      SV1 <- rgV.genoud$par  # and use these as starting values

      if (verbose) cat("\n****************",
                       "\n genoud() finished, now running local optimization using optim()\n", "\n")

      } else {
      # if we don't use genoud first: set of starting values: equal weights
      SV1 <- rep(1/nvarsV,nvarsV)
      }

      # now we run optimization
      all.methods <- FALSE
      if(sum(optimxmethod %in% c("All"))==1){ all.methods <- TRUE }
     rgV.optim.1 <- optimx(par=SV1, fn=fn.V.native,                                                # native
                             gr=NULL, hess=NULL,
                             method=optimxmethod, itnmax=NULL, hessian=FALSE,
                             control=list(kkt=FALSE,
                                          starttests=FALSE,
                                          dowarn=FALSE,
                                          all.methods=all.methods),
                             X0.scaled = X0.scaled,
                             X1.scaled = X1.scaled,
                             Z0 = Z0,
                             Z1 = Z1,
                             quadopt = quadopt,
                             margin.ipop = Margin.ipop,
                             sigf.ipop = Sigf.ipop,
                             bound.ipop = Bound.ipop,
                             weights = weights,                                                     # native
                             penalty.lambda = penalty.lambda                                        # native
                            )
      # get minimum
      if(verbose==TRUE){print(rgV.optim.1)}
      rgV.optim.1 <- collect.optimx(rgV.optim.1,"min")

      # second set of starting values: regression method
      # will sometimes not work because of collinear Xs
      # so it's wrapped in a try command
      Xall <- cbind(X1.scaled,X0.scaled)
      Xall <- cbind(rep(1,ncol(Xall)),t(Xall))
      Zall <- cbind(Z1,Z0)
      Beta <- try(solve(t(Xall)%*%Xall)%*%t(Xall)%*%t(Zall),silent=TRUE)

      # if inverses did not work, we
      # stick with first results
      if(inherits(Beta,"try-error"))
       {
        rgV.optim <- rgV.optim.1
       } else {
      # otherwise we run a second optimization with regression starting values
        Beta <- Beta[-1,]
        V    <- Beta%*%t(Beta)
        SV2  <- diag(V)
        SV2 <- SV2 / sum(SV2)

      rgV.optim.2 <- optimx(par=SV2, fn=fn.V.native,                                                # native
                             gr=NULL, hess=NULL,
                             method=optimxmethod, itnmax=NULL, hessian=FALSE,
                             control=list(kkt=FALSE,
                                          starttests=FALSE,
                                          dowarn=FALSE,
                                          all.methods=all.methods),
                             X0.scaled = X0.scaled,
                             X1.scaled = X1.scaled,
                             Z0 = Z0,
                             Z1 = Z1,
                             quadopt = quadopt,
                             margin.ipop = Margin.ipop,
                             sigf.ipop = Sigf.ipop,
                             bound.ipop = Bound.ipop,
                             weights = weights,                                                     # native
                             penalty.lambda = penalty.lambda                                        # native
                            )
      # get minimum
      if(verbose==TRUE){print(rgV.optim.2)}
      rgV.optim.2 <- collect.optimx(rgV.optim.2,"min")

      # ouput
      if(verbose == TRUE){
      cat("\n Equal weight loss is:",rgV.optim.1$value,"\n")
      cat("\n Regression Loss is:",rgV.optim.2$value,"\n")
      }
      # and keep the better optim results
      if(rgV.optim.1$value < rgV.optim.2$value)
       {
        rgV.optim <- rgV.optim.1
       } else {
        rgV.optim <- rgV.optim.2
       }
      } # close if statement for second regression based optimization attempt

      # final V weights from optimization
      solution.v   <- abs(rgV.optim$par)/sum(abs(rgV.optim$par))
     } else { # jump here if only optimize over W

     if (verbose) cat("\n****************",
                      "\n optimization over w weights: computing synthetic control unit \n", "\n\n")

     if (nrow(X0) == 1) {
       custom.v <- 1 # only one predictor: V is the identity matrix
     } else {
       # user-supplied v
       if (verbose) cat("\n****************",
                        "\n v weights supplied manually: computing synthetic control unit \n", "\n\n")
       if (length(custom.v) != nvarsV) {
         stop("custom.v misspecified: length(custom.v) != nrow(X0)")
       }
       if (mode(custom.v) != "numeric") {
         stop("custom.v must be numeric")
       }
     }

    # enter solution.V
    rgV.optim  <- NULL
    solution.v <- abs(custom.v)/sum(custom.v)

  } # close else statment for by-passing V optimization


    # last step: now recover solution.w
    V <- diag(x=as.numeric(solution.v),nrow=nvarsV,ncol=nvarsV)
    solution.w <- solve.w.native(X0.scaled, X1.scaled, V, quadopt, Margin.ipop, Sigf.ipop,          # native
                                 Bound.ipop, weights, penalty.lambda)                               # native

    rownames(solution.w) <- colnames(X0)
    colnames(solution.w) <- "w.weight"
    names(solution.v) <- rownames(X0)

    loss.w <- t(X1.scaled - X0.scaled %*% solution.w) %*%
      V %*% (X1.scaled - X0.scaled %*% solution.w)

    loss.v <-
      t(Z1 - Z0 %*% as.matrix(solution.w)) %*%
        (Z1 - Z0 %*% as.matrix(solution.w))
    loss.v <- loss.v/nrow(Z0)

    # produce viewable output
    if (verbose) {
      cat("\n****************",
          "\n****************",
          "\n****************",
          "\n\nMSPE (LOSS V):", loss.v,
          "\n\nsolution.v:\n", round(as.numeric(solution.v), 10),
          "\n\nsolution.w:\n", round(as.numeric(solution.w), 10),
          "\n\n")
    }

    optimize.out <- list(
                         solution.v = solution.v,
                         solution.w = solution.w,
                         loss.v = loss.v,
                         loss.w = loss.w,
                         custom.v = custom.v,
                         rgV.optim = rgV.optim
                         )

    return(invisible(optimize.out))

  }

fit_native <- function(d, s) {
  dp <- scm_dataprep(d, s)
  so <- synth.native(data.prep.obj = dp, custom.v = scm_custom_v(s), optimxmethod = unlist(s$optimxmethod),
                     genoud = s$genoud, quadopt = s$quadopt, Margin.ipop = s$margin_ipop, Sigf.ipop = s$sigf_ipop,
                     Bound.ipop = s$bound_ipop, weights = s$weights,
                     penalty.lambda = if (s$interpolation_penalty == "distance") s$penalty_lambda else NULL)
  scm_record(d, dp, so, s, "native")
}
```
