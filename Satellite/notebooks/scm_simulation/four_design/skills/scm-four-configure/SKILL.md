---
name: scm-four-configure
description: Estimate the effect of a treatment on one treated unit from a panel of that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller (2010) at a fixed design, by running the bundled fitter. Use this whenever one treated unit's outcome without treatment is to be estimated from donors with that design.
---

# Configure and run the synthetic control

A weighted combination of donor units, the synthetic control, is chosen so that it resembles the treated unit before treatment. Its outcome after treatment estimates what the treated unit's outcome would have been without treatment, and the difference is the effect. `references/method.md` describes the construction. Read it if anything below is unclear.

## What is fixed in this skill, and what you decide

The fitter makes these choices for you: Z1 and Z2 in period 40 and the outcome in each pre-treatment period as the predictors, V chosen to fit the outcome in periods 1 to 40, every donor weight at least 0 and the weights summing to one, no penalty, all 29 donors, and Synth's default optimizers and solver settings. Everything else is Synth's own code, unchanged.

You decide the settings, all zero, in `settings.json`:

There are none, so `settings.json` holds an empty object, `{}`.

`references/settings.md` gives the shape of the file. The fitter refuses any field in it and names the field.

## Workflow

1. Look at the data: `python scripts/show_data.py`, and `python scripts/show_data.py Z1` or `Z2` for the
   covariates. It shows the treated unit and the donors period by period.
2. Read `references/method.md` and `references/settings.md`. Decide the settings from the data and the method.
3. Write `settings.json`, then run `Rscript scripts/fit.R settings.json`. It prints the predictors and V, the donor
   weights, the pre-treatment fit over the periods V is fitted to, and for every period the treated unit's observed outcome,
   the synthetic control and their gap.
4. You may change the settings and run again. Every run is recorded. The pre-treatment fit is an in-sample quantity, and
   a closer fit before treatment is not by itself a better estimate after it.
5. Finish by writing `answer.json`, with one value for every treated period the task names:
   ```json
   {"counterfactual": {"<period>": <value>, ...}, "att": <value>, "rationale": "<why these settings, in a few sentences>"}
   ```
   `counterfactual` is the treated unit's outcome without treatment in each treated period, and `att` the average
   effect over those periods. The values you submit are your answer. If they differ from the fitter's, say why.
