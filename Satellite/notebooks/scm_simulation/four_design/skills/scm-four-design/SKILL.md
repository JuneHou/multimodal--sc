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
