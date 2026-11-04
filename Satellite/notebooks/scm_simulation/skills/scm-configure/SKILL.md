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
