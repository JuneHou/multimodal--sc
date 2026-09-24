---
name: shc-design
description: Predict the next month of a land parcel's satellite band values from its own
  nine-month history with the synthetic historical control method, deciding both its
  settings and its methodological choices — what is matched, how weights are constrained,
  how bands are weighted, how blocks are selected, how the prediction is formed — and
  running the bundled fitter. Use this whenever a parcel's monthly band values are to be
  forecast one month ahead from its own past and the design of the estimator is open.
---

# Design and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier months are weighted to
reproduce its most recent months, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction and the alternatives at each step.

## Everything is yours to decide

`spec.json` holds two kinds of field. The settings (`m`, `n`, `kernel_bandwidth`, `zeta`,
`beta_trend_degree`) are numbers. The choices are each one of a few options:

- `fit_space` — match the prepared series (`latent_std`) or the observed values (`raw`)
- `weight_constraint` — `simplex` (non-negative, sum to one), `nonneg`, or `unconstrained`
- `intercept` — `none` or `fitted`
- `band_weighting` — `equal`, `inverse_variance`, or `subset` (with `band_subset`)
- `selection` — `greedy_forward` or `exhaustive`
- `readd_seasonal` — predict from observed forward values (`false`) or in the prepared
  space with the seasonal term added back (`true`)

`references/settings.md` defines every field and the rules that tie them, for instance
that weights fitted on the standardized prepared series keep the observed scale only when
they sum to one, so `nonneg` and `unconstrained` need `raw` or `readd_seasonal`. The fitter
refuses a specification that breaks a rule and names the fields, so read the rules before
writing the file. There are no defaults.

## Workflow

1. `python scripts/show_data.py` — the nine months, every band.
2. Read `references/method.md` and `references/settings.md`; decide the design from the data.
3. Write `spec.json`; run `python scripts/fit.py spec.json`. It prints the blocks kept,
   weights (and intercept if any), the matching error over P01–P09, and the prediction.
4. Revise and run again if you judge it right; every run is recorded. The matching error is
   in-sample: a design that fits P01–P09 better does not necessarily predict P10 better.
5. Finish by writing `answer.json` as in the format below. The prediction you submit is
   your answer; if it differs from the fitter's, say why.
   ```json
   {"prediction": {"<band>": <value>, ...}, "rationale": "<why this design, in a few sentences>"}
   ```
