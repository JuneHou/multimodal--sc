# SCM skills: every text a model reads

Built from `notebooks/scm_simulation/method.py` into `notebooks/scm_simulation/skills/`. The solver models see sections 1 to 6. The two SkillFoundry generators see section 7, with the paper in `paper/` and Synth 1.1-10's CRAN source in `code/Synth/`.

## 1. System message (the pipeline's, unchanged)

```
You are operating a bash shell in a working directory that contains a skill: files with instructions (SKILL.md), reference documents and scripts for one task. Each reply you send is executed as one bash command block, and its output is returned to you in the next message. Nothing outside the working directory is visible and there is no network. Read the skill's instructions and follow them.
```

## 2a. First message, case dgp1, skill scm-configure (dgp2 differs only in the file name)

```
Estimate the effect of a treatment on the unit `treated` in the panel `dgp1_panel.csv`. The unit `treated` was treated in periods 41 to 60, and the units `donor_01` to `donor_29` were never treated. Estimate the treated unit's outcome `Y_observed` without treatment in each period from 41 to 60, and the average effect over those periods.
The working directory holds a skill for this task:
  scm-configure. Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by deciding the arguments of the Synth R package (predictors, donor pool, predictor weights V, optimizer) and running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors and the Synth code is provided.
Its instructions are in SKILL.md. Read that file first and follow it. Reply with one ```bash block per turn. It is run by bash in the working directory and its output comes back in the next message. Finish by writing answer.json as the skill describes.
```

## 2b. First message, case dgp1, skill scm-design (dgp2 differs only in the file name)

```
Estimate the effect of a treatment on the unit `treated` in the panel `dgp1_panel.csv`. The unit `treated` was treated in periods 41 to 60, and the units `donor_01` to `donor_29` were never treated. Estimate the treated unit's outcome `Y_observed` without treatment in each period from 41 to 60, and the average effect over those periods.
The working directory holds a skill for this task:
  scm-design. Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by designing the method, the Synth R package's arguments and the paper's alternatives for the donor weights, and running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors and the method's design is yours to choose.
Its instructions are in SKILL.md. Read that file first and follow it. Reply with one ```bash block per turn. It is run by bash in the working directory and its output comes back in the next message. Finish by writing answer.json as the skill describes.
```

## 2c. First message for a profile skill, case dgp1 (row G16 of the profile experiment: no skill line)

```
Estimate the effect of a treatment on the unit `treated` in the panel `dgp1_panel.csv`. The unit `treated` was treated in periods 41 to 60, and the units `donor_01` to `donor_29` were never treated. Estimate the treated unit's outcome `Y_observed` without treatment in each period from 41 to 60, and the average effect over those periods.
The working directory holds a skill for this task. Its instructions are in SKILL.md. Read that file first and follow it. Reply with one ```bash block per turn. It is run by bash in the working directory and its output comes back in the next message. Finish by writing answer.json as the skill describes.
```

## 3a. scm-configure/SKILL.md

```markdown
---
name: scm-configure
description: Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by deciding the arguments of the Synth R package (predictors, donor pool, predictor weights V, optimizer) and running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors and the Synth code is provided.
---

# Configure and run the synthetic control

A weighted combination of donor units, the synthetic control, is chosen so that it resembles the treated unit before treatment. Its outcome after treatment estimates what the treated unit's outcome would have been without treatment, and the difference is the effect. `references/method.md` describes the construction. Read it if anything below is unclear.

## What is fixed in this skill, and what you decide

The fitter makes these choices for you: every donor weight is at least 0 and the weights sum to one, as the paper defines a synthetic control, and no penalty is added to the weight fit. Everything else is Synth's own code, unchanged.

You decide the settings, all 13, in `settings.json`:

- `special_predictors`, predictors given one by one, each a variable (`Z1`, `Z2` or `Y_observed`), the periods it is summarized over and the summary `op`
- `predictors` and `predictors_op`, variables summarized with one summary over `time_predictors_prior`
- `time_predictors_prior`, the periods over which `predictors` are summarized
- `time_optimize_ssr`, the outcome periods whose mean squared prediction error chooses V when V is searched
- `controls_identifier`, the donor pool
- `custom_v`, the predictor weights V, one positive number per predictor, or null to let Synth search V
- `optimxmethod` and `genoud`, the optimizers of the V search
- `quadopt`, `margin_ipop`, `sigf_ipop` and `bound_ipop`, the solver of the donor weights and its precision

`references/settings.md` gives each field's type, range and meaning, the rule that every listed period is a whole period number, and the shape of the file. There are no defaults. The fitter refuses an incomplete file and names what is missing. Each field is the Synth argument of the same name, so Synth's own errors, for example a `custom_v` whose length is not the number of predictors, are printed as Synth gives them.

## Workflow

1. Look at the data: `python scripts/show_data.py`, and `python scripts/show_data.py Z1` or `Z2` for the
   covariates. It shows the treated unit and the donors period by period.
2. Read `references/method.md` and `references/settings.md`. Decide the settings from the data and the method.
3. Write `settings.json`, then run `Rscript scripts/fit.R settings.json`. It prints the predictors and V, the donor
   weights, the pre-treatment fit over `time_optimize_ssr`, and for every period the treated unit's observed outcome,
   the synthetic control and their gap.
4. You may change the settings and run again. Every run is recorded. The pre-treatment fit is an in-sample quantity, and
   a closer fit before treatment is not by itself a better estimate after it.
5. Finish by writing `answer.json`, with one value for every treated period the task names:
   ```json
   {"counterfactual": {"<period>": <value>, ...}, "att": <value>, "rationale": "<why these settings, in a few sentences>"}
   ```
   `counterfactual` is the treated unit's outcome without treatment in each treated period, and `att` the average
   effect over those periods. The values you submit are your answer. If they differ from the fitter's, say why.
```

## 3b. scm-design/SKILL.md

```markdown
---
name: scm-design
description: Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010), by designing the method, the Synth R package's arguments and the paper's alternatives for the donor weights, and running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors and the method's design is yours to choose.
---

# Design and run the synthetic control

A weighted combination of donor units, the synthetic control, is chosen so that it resembles the treated unit before treatment. Its outcome after treatment estimates what the treated unit's outcome would have been without treatment, and the difference is the effect. `references/method.md` describes the construction. Read it if anything below is unclear.

## Everything is yours to decide

`spec.json` holds two kinds of field. The settings are the 13 arguments of Synth's `dataprep` and `synth` listed in `references/settings.md`. The choices are the paper's alternatives to what Synth fixes in its code (Abadie, Diamond and Hainmueller 2010, p. 496):

- `weights`, `convex` (every donor weight at least 0, summing to one, the paper's definition and Synth's) or `extrapolation` (summing to one, any weight may be below 0 or above one)
- `interpolation_penalty`, `none` or `distance`, which adds `penalty_lambda` times each donor's weight times its distance from the treated unit's predictors, so that distant donors are kept out

`references/settings.md` defines every field and the rules that tie them. `penalty_lambda` is given exactly when `interpolation_penalty` is `distance`, and `distance` requires `convex` weights. The fitter refuses a specification that breaks a rule and names the fields, so read the rules before writing the file. There are no defaults.

## Workflow

1. Look at the data: `python scripts/show_data.py`, and `python scripts/show_data.py Z1` or `Z2` for the
   covariates. It shows the treated unit and the donors period by period.
2. Read `references/method.md` and `references/settings.md`. Decide the design from the data and the method.
3. Write `spec.json`, then run `Rscript scripts/fit.R spec.json`. It prints the predictors and V, the donor
   weights, the pre-treatment fit over `time_optimize_ssr`, and for every period the treated unit's observed outcome,
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

## 4. references/method.md (both skills)

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

## 5a. scm-configure/references/settings.md

```markdown
# Fields for scm-configure

`settings.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|
| `special_predictors` | list of {variable, periods, op}, variable one of Z1, Z2, Y_observed, op one of mean, median, sd, var, min, max, sum | dataprep's special.predictors. Each entry is one predictor, the op of the variable over the listed periods |
| `predictors` | list of names from Z1, Z2, Y_observed, may be empty | dataprep's predictors. Each is one predictor, the predictors_op of the variable over time_predictors_prior |
| `predictors_op` | one of mean, median, sd, var, min, max, sum | dataprep's predictors.op |
| `time_predictors_prior` | list of periods | dataprep's time.predictors.prior, the periods over which predictors are summarized |
| `time_optimize_ssr` | list of periods | dataprep's time.optimize.ssr, the periods of the outcome over which the mean squared prediction error is minimized when V is searched |
| `controls_identifier` | list of donor names from donor_01 to donor_29 | dataprep's controls.identifier, the donor pool |
| `custom_v` | list of numbers above 0, one per predictor, or null | synth's custom.v. Null means V is searched |
| `optimxmethod` | list of names from BFGS, CG, Nelder-Mead, L-BFGS-B, nlm, nlminb, lbfgsb3c, Rcgmin, Rtnmin, Rvmmin, snewton, snewtonm, spg, ucminf, newuoa, bobyqa, uobyqa, nmkb, hjkb, hjn, lbfgs, subplex, ncg, nvm, mla, slsqp, tnewt, anms, pracmanm, nlnm, snewtm, All | synth's optimxmethod, the optimx methods of the V search (All runs every one) |
| `genoud` | true or false | synth's genoud |
| `quadopt` | ipop | synth's quadopt |
| `margin_ipop` | number above 0 | synth's Margin.ipop |
| `sigf_ipop` | integer above 0 | synth's Sigf.ipop |
| `bound_ipop` | number above 0 | synth's Bound.ipop |

## Rules

- Every period in time_predictors_prior, time_optimize_ssr and the periods of special_predictors is a whole period number.

## Shape of the file (every value is yours)

```json
{
 "special_predictors": "<list of {\"variable\": <name>, \"periods\": <list of periods>, \"op\": <name>}>",
 "predictors": "<list of names>",
 "predictors_op": "<one of the listed values>",
 "time_predictors_prior": "<list of numbers>",
 "time_optimize_ssr": "<list of numbers>",
 "controls_identifier": "<list of names>",
 "custom_v": "<list of numbers, or null>",
 "optimxmethod": "<list of names>",
 "genoud": "<true or false>",
 "quadopt": "<one of the listed values>",
 "margin_ipop": "<float>",
 "sigf_ipop": "<int>",
 "bound_ipop": "<float>"
}
```
```

## 5b. scm-design/references/settings.md

```markdown
# Fields for scm-design

`spec.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|
| `special_predictors` | list of {variable, periods, op}, variable one of Z1, Z2, Y_observed, op one of mean, median, sd, var, min, max, sum | dataprep's special.predictors. Each entry is one predictor, the op of the variable over the listed periods |
| `predictors` | list of names from Z1, Z2, Y_observed, may be empty | dataprep's predictors. Each is one predictor, the predictors_op of the variable over time_predictors_prior |
| `predictors_op` | one of mean, median, sd, var, min, max, sum | dataprep's predictors.op |
| `time_predictors_prior` | list of periods | dataprep's time.predictors.prior, the periods over which predictors are summarized |
| `time_optimize_ssr` | list of periods | dataprep's time.optimize.ssr, the periods of the outcome over which the mean squared prediction error is minimized when V is searched |
| `controls_identifier` | list of donor names from donor_01 to donor_29 | dataprep's controls.identifier, the donor pool |
| `custom_v` | list of numbers above 0, one per predictor, or null | synth's custom.v. Null means V is searched |
| `optimxmethod` | list of names from BFGS, CG, Nelder-Mead, L-BFGS-B, nlm, nlminb, lbfgsb3c, Rcgmin, Rtnmin, Rvmmin, snewton, snewtonm, spg, ucminf, newuoa, bobyqa, uobyqa, nmkb, hjkb, hjn, lbfgs, subplex, ncg, nvm, mla, slsqp, tnewt, anms, pracmanm, nlnm, snewtm, All | synth's optimxmethod, the optimx methods of the V search (All runs every one) |
| `genoud` | true or false | synth's genoud |
| `quadopt` | ipop | synth's quadopt |
| `margin_ipop` | number above 0 | synth's Margin.ipop |
| `sigf_ipop` | integer above 0 | synth's Sigf.ipop |
| `bound_ipop` | number above 0 | synth's Bound.ipop |

## Choices

| field | options | meaning |
|---|---|---|
| `weights` | `convex`, `extrapolation` | convex = each donor weight at least 0 and the weights summing to one (the paper's definition and Synth); extrapolation = the weights sum to one and any weight may be negative or above one (p. 496) |
| `interpolation_penalty` | `none`, `distance` | none = no penalty (Synth); distance = penalty_lambda times the sum over donors of the donor's weight times its V-weighted squared distance from the treated unit's scaled predictors (p. 496) |
| `penalty_lambda` | number of 0 or above | a number of 0 or above, given exactly when interpolation_penalty is distance |

## Rules that tie the fields together

- Every period in time_predictors_prior, time_optimize_ssr and the periods of special_predictors is a whole period number.
- penalty_lambda is given, a number of 0 or above, exactly when interpolation_penalty is distance.
- interpolation_penalty = distance requires weights = convex.

## Shape of the file (every value is yours)

```json
{
 "special_predictors": "<list of {\"variable\": <name>, \"periods\": <list of periods>, \"op\": <name>}>",
 "predictors": "<list of names>",
 "predictors_op": "<one of the listed values>",
 "time_predictors_prior": "<list of numbers>",
 "time_optimize_ssr": "<list of numbers>",
 "controls_identifier": "<list of names>",
 "custom_v": "<list of numbers, or null>",
 "optimxmethod": "<list of names>",
 "genoud": "<true or false>",
 "quadopt": "<one of the listed values>",
 "margin_ipop": "<float>",
 "sigf_ipop": "<int>",
 "bound_ipop": "<float>",
 "weights": "<convex|extrapolation>",
 "interpolation_penalty": "<none|distance>",
 "penalty_lambda": "<number, only with distance>"
}
```
```

## 6a. scripts/show_data.py (both skills)

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

## 6b. scm-configure/scripts/fit.R

```r
#!/usr/bin/env Rscript
# Fit the synthetic control method with the Synth R package, version 1.1-10, unchanged, at the settings in a file:
#   Rscript scripts/fit.R settings.json
# The arguments of dataprep() that describe the data file are filled here, the same for every file:
#   dependent "Y_observed", unit.variable unit numbers (treated 1, donor_jj jj + 1), time.variable "period",
#   treatment.identifier "treated", unit.names.variable "unit_id", time.plot every period of the file, verbose FALSE.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "scm_fields.R"))
source(file.path(here, "synth_fit.R"))
source(file.path(here, "scm_native.R"))
args <- commandArgs(trailingOnly = TRUE)
spec_file <- if (length(args)) args[1] else "settings.json"
given <- tryCatch(fromJSON(spec_file, simplifyVector = FALSE), error = function(e) {
  cat("refused: ", spec_file, " is not readable JSON (", conditionMessage(e), ")\n", sep = "")
  quit(status = 1)
})
errors <- check_settings(given, "configure")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\n", sep = "")
  quit(status = 1)
}
s <- c(given, SCM_SYNTH_CHOICES)          # the two choices this skill fixes, at what Synth does
panel <- list.files(".", pattern = "_panel[.]csv$")
if (length(panel) != 1) stop("expected one *_panel.csv file in the working directory")
d <- scm_data(panel)
synth_choices <- identical(s$weights, "convex") && identical(s$interpolation_penalty, "none")
rec <- if (synth_choices) fit_synth(d, s) else fit_native(d, s)
rec$file <- basename(spec_file)
log <- Sys.getenv("FIT_LOG")                         # the record of this fit, for the run log
if (nzchar(log)) cat(scm_json(rec), "\n", file = log, append = TRUE, sep = "")

cat(if (synth_choices) "fit: Synth 1.1-10 dataprep() and synth(), unchanged\n" else
      sprintf("fit: Synth's code with weights = %s, interpolation_penalty = %s\n", s$weights, s$interpolation_penalty))
cat(sprintf("predictors and their weights in V (%d):\n", length(rec$v)))
for (i in seq_along(rec$v)) cat(sprintf("  %-28s %.6f\n", rec$predictor_names[[i]], rec$v[[i]]))
cat(sprintf("donor weights (%d donors in the pool):\n", length(rec$weights)))
for (i in seq_along(rec$weights)) cat(sprintf("  %-10s %.6f\n", rec$donor_ids[[i]], rec$weights[[i]]))
cat(sprintf("pre-treatment fit: mean squared prediction error over time_optimize_ssr (%d periods) %.6f\n",
            length(s$time_optimize_ssr), rec$fit_mse))
cat("period, treated unit observed, synthetic control, gap (observed minus synthetic control):\n")
for (p in names(rec$predictions))
  cat(sprintf("  %3s %12.6f %12.6f %12.6f\n", p, rec$predictions[[p]] + rec$gaps[[p]], rec$predictions[[p]],
              rec$gaps[[p]]))
```

## 6c. scm-design/scripts/fit.R, its difference from 6b

```diff
--- configure
+++ design
@@ -2,3 +2,3 @@
 # Fit the synthetic control method with the Synth R package, version 1.1-10, unchanged, at the settings in a file:
-#   Rscript scripts/fit.R settings.json
+#   Rscript scripts/fit.R spec.json
 # The arguments of dataprep() that describe the data file are filled here, the same for every file:
@@ -11,3 +11,3 @@
 args <- commandArgs(trailingOnly = TRUE)
-spec_file <- if (length(args)) args[1] else "settings.json"
+spec_file <- if (length(args)) args[1] else "spec.json"
 given <- tryCatch(fromJSON(spec_file, simplifyVector = FALSE), error = function(e) {
@@ -16,3 +16,3 @@
 })
-errors <- check_settings(given, "configure")
+errors <- check_settings(given, "design")
 if (length(errors)) {
@@ -21,3 +21,3 @@
 }
-s <- c(given, SCM_SYNTH_CHOICES)          # the two choices this skill fixes, at what Synth does
+s <- given
 panel <- list.files(".", pattern = "_panel[.]csv$")
```

## 6d. scripts/scm_fields.R (both skills, copied unchanged from notebooks/scm_simulation/)

```r
# Field checks of the settings file, the same rules as the package's checks (method.py and the pipeline's scs_spec.py).
# check_settings(given, mode) returns the problems found, each naming its field; an empty vector means the file is
# accepted. mode "configure" takes the settings only; mode "design" takes the settings and the choices.

SCM_VARIABLES <- c("Z1", "Z2", "Y_observed")
SCM_OPS <- c("mean", "median", "sd", "var", "min", "max", "sum")
SCM_DONORS <- sprintf("donor_%02d", 1:29)
SCM_OPTIMX <- c("BFGS", "CG", "Nelder-Mead", "L-BFGS-B", "nlm", "nlminb", "lbfgsb3c", "Rcgmin", "Rtnmin", "Rvmmin",
                "snewton", "snewtonm", "spg", "ucminf", "newuoa", "bobyqa", "uobyqa", "nmkb", "hjkb", "hjn", "lbfgs",
                "subplex", "ncg", "nvm", "mla", "slsqp", "tnewt", "anms", "pracmanm", "nlnm", "snewtm", "All")

SCM_SETTINGS <- list(
  special_predictors = list(kind = "predictor_list"),
  predictors = list(kind = "name_list", allowed = SCM_VARIABLES),
  predictors_op = list(kind = "enum", allowed = SCM_OPS),
  time_predictors_prior = list(kind = "positive_list"),
  time_optimize_ssr = list(kind = "positive_list"),
  controls_identifier = list(kind = "name_list", allowed = SCM_DONORS),
  custom_v = list(kind = "positive_list_or_null"),
  optimxmethod = list(kind = "name_list", allowed = SCM_OPTIMX),
  genoud = list(kind = "bool"),
  quadopt = list(kind = "enum", allowed = c("ipop")),
  margin_ipop = list(kind = "float"),
  sigf_ipop = list(kind = "int"),
  bound_ipop = list(kind = "float"))
SCM_CHOICES <- list(
  weights = list(kind = "enum", allowed = c("convex", "extrapolation")),
  interpolation_penalty = list(kind = "enum", allowed = c("none", "distance")))
SCM_OPTIONAL <- c("penalty_lambda")
SCM_SYNTH_CHOICES <- list(weights = "convex", interpolation_penalty = "none")     # what Synth does in its body

is_array <- function(v) is.list(v) && is.null(names(v))
is_number <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
is_whole <- function(v) is_number(v) && v == round(v)
shown <- function(v) if (is.null(v)) "null" else as.character(jsonlite::toJSON(v, auto_unbox = TRUE, null = "null"))

positive_list_ok <- function(v) is_array(v) && length(v) > 0 && all(vapply(v, function(x) is_number(x) && x > 0, TRUE))

check_value <- function(field, spec, v) {
  kind <- spec$kind
  if (kind == "int") {
    if (!is_whole(v)) return(sprintf("field '%s': expected an integer, got %s", field, shown(v)))
    if (!(v > 0)) return(sprintf("field '%s': %s is outside the allowed range", field, shown(v)))
  } else if (kind == "float") {
    if (!is_number(v)) return(sprintf("field '%s': expected a number, got %s", field, shown(v)))
    if (!(v > 0)) return(sprintf("field '%s': %s is outside the allowed range", field, shown(v)))
  } else if (kind == "bool") {
    if (!(is.logical(v) && length(v) == 1 && !is.na(v)))
      return(sprintf("field '%s': expected true or false, got %s", field, shown(v)))
  } else if (kind == "enum") {
    if (!(is.character(v) && length(v) == 1 && v %in% spec$allowed))
      return(sprintf("field '%s': %s is not one of %s", field, shown(v), shown(spec$allowed)))
  } else if (kind == "positive_list" || (kind == "positive_list_or_null" && !is.null(v))) {
    if (!positive_list_ok(v))
      return(sprintf("field '%s': expected a non-empty list of numbers > 0, got %s", field, shown(v)))
  } else if (kind == "name_list") {
    if (!is_array(v) || !all(vapply(v, function(x) is.character(x) && length(x) == 1, TRUE)))
      return(sprintf("field '%s': expected a list of names, got %s", field, shown(v)))
    bad <- setdiff(unlist(v), spec$allowed)
    if (length(bad)) return(sprintf("field '%s': %s are not among %s", field, shown(bad), shown(spec$allowed)))
    if (anyDuplicated(unlist(v))) return(sprintf("field '%s': a name is repeated", field))
  } else if (kind == "predictor_list") {
    if (!is_array(v)) return(sprintf("field '%s': expected a list of predictor entries, got %s", field, shown(v)))
    for (i in seq_along(v)) {
      e <- v[[i]]
      if (!(is.list(e) && !is.null(names(e)) && setequal(names(e), c("variable", "periods", "op")) && length(e) == 3))
        return(sprintf("field '%s': entry %d must be an object with exactly \"variable\", \"periods\" and \"op\", got %s",
                       field, i - 1, shown(e)))
      if (!(is.character(e$variable) && length(e$variable) == 1 && e$variable %in% SCM_VARIABLES))
        return(sprintf("field '%s': entry %d variable %s is not one of %s", field, i - 1, shown(e$variable),
                       shown(SCM_VARIABLES)))
      p <- e$periods
      if (!(is_array(p) && length(p) > 0 && all(vapply(p, function(x) is_whole(x) && x >= 1, TRUE))))
        return(sprintf("field '%s': entry %d periods must be a non-empty list of whole periods, got %s",
                       field, i - 1, shown(p)))
      if (!(is.character(e$op) && length(e$op) == 1 && e$op %in% SCM_OPS))
        return(sprintf("field '%s': entry %d op %s is not one of %s", field, i - 1, shown(e$op), shown(SCM_OPS)))
    }
  }
  NULL
}

check_settings <- function(given, mode) {
  if (!(is.list(given) && !is.null(names(given)))) return("spec must be a JSON object")
  fields <- if (mode == "design") c(SCM_SETTINGS, SCM_CHOICES) else SCM_SETTINGS
  if (mode == "configure") {
    present <- names(given)[names(given) %in% c(names(SCM_CHOICES), SCM_OPTIONAL)]
    if (length(present))
      return(paste0("protocol violation: scm-configure takes Class-1 fields only, but got ",
                    paste(present, collapse = ", ")))
  }
  errors <- character(0)
  missing <- setdiff(names(fields), names(given))
  if (length(missing)) errors <- c(errors, paste0("missing required field(s): ", paste(missing, collapse = ", ")))
  known <- if (mode == "design") c(names(fields), SCM_OPTIONAL) else names(fields)
  unknown <- setdiff(names(given), known)
  if (length(unknown)) errors <- c(errors, paste0("unknown field(s): ", paste(unknown, collapse = ", ")))
  ok <- character(0)
  for (f in names(fields)) {
    if (!(f %in% names(given))) next
    e <- check_value(f, fields[[f]], given[[f]])
    if (is.null(e)) ok <- c(ok, f) else errors <- c(errors, e)
  }
  # rule: every period in time_predictors_prior and time_optimize_ssr is a whole period number
  for (f in c("time_predictors_prior", "time_optimize_ssr")) {
    if (f %in% ok && !all(vapply(given[[f]], function(x) x == round(x), TRUE)))
      errors <- c(errors, sprintf("field '%s': every period must be a whole period number", f))
  }
  if (mode == "design") {
    pen <- if ("interpolation_penalty" %in% ok) given$interpolation_penalty else NULL
    # rule: penalty_lambda is given, a number of 0 or above, exactly when interpolation_penalty is distance
    if (identical(pen, "distance")) {
      if (!("penalty_lambda" %in% names(given))) {
        errors <- c(errors, "field 'penalty_lambda': required when interpolation_penalty is distance")
      } else {
        lam <- given$penalty_lambda
        if (!(is_number(lam) && lam >= 0))
          errors <- c(errors, sprintf("field 'penalty_lambda': expected a number of 0 or above, got %s", shown(lam)))
      }
    } else if ("penalty_lambda" %in% names(given) && !is.null(pen)) {
      errors <- c(errors, "field 'penalty_lambda': given, but interpolation_penalty is not distance")
    }
    # rule: interpolation_penalty = distance requires weights = convex
    if (identical(pen, "distance") && "weights" %in% ok && identical(given$weights, "extrapolation"))
      errors <- c(errors, paste0("fields 'interpolation_penalty','weights': interpolation_penalty = distance ",
                                 "requires weights = convex"))
  }
  errors
}
```

## 6d. scripts/synth_fit.R (both skills, copied unchanged from notebooks/scm_simulation/)

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

## 6d. scripts/scm_native.R (both skills, copied unchanged from notebooks/scm_simulation/)

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

## 7. SkillFoundry request (both generators)

```
Design a skill for estimating the effect of a treatment on one treated unit from panel data on that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller. The paper, Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program, is at `paper/Synthetic Control Methods for Comparative Case Studies- Estimating the Effect of California’s Tobacco Control Program.pdf`. The paper's own software, the Synth R package version 1.1-10, is at `code/Synth/`, and R 4.3.3 with Synth and jsonlite installed is available to the agent. The agent that uses the skill works in a directory holding one panel file `*_panel.csv` with the columns `unit_id`, `period`, `treatment_group`, `Z1`, `Z2` and `Y_observed`, one row per unit and period, for the treated unit `treated` and the untreated donor units. It estimates the treated unit's outcome without treatment in each treated period the task names and the average effect over those periods, and writes `answer.json` as {"counterfactual": {"<period>": <number>, ...}, "att": <number>, "rationale": "<text>"}. The agent has a bash shell and no network access.
```
