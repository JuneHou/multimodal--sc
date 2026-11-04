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
