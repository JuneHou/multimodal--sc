---
name: shc-design
description: Predict a land parcel's satellite band values at each visit listed in
  visits_to_predict.csv from its own history of visits with the synthetic historical control
  method, deciding both its settings and its methodological choices — what is matched, how
  weights are constrained, how bands are weighted, how blocks are selected, how the
  prediction is formed — and running the bundled fitter. Use this whenever a parcel's band
  values are to be forecast at listed later visits from its own past and the design of the
  estimator is open.
---

# Design and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier visits are weighted to
reproduce its most recent visits, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction and the alternatives at each step.

## Everything is yours to decide

`spec.json` holds two kinds of field. The settings (`time_bandwidth_candidates`, `m_min`,
`m_step`, `m_max`, `max_donors`, `nu`, `zeta`) are numbers. The choices are each one of a
few options:

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

1. `python scripts/show_data.py` — every visit given, every band, and the visits to predict.
2. Read `references/method.md` and `references/settings.md`; decide the design from the data.
3. Write `spec.json`; run `python scripts/fit.py spec.json`. It prints the matching window
   length chosen, the blocks kept, weights (and intercept if any), the matching error over
   the visits given, and the prediction at every visit to predict.
4. Revise and run again if you judge it right; every run is recorded. The matching error is
   in-sample: a design that fits the visits given better does not necessarily predict the
   visits to predict better.
5. Finish by writing `answer.json` as in the format below, with every band at every visit
   to predict. The prediction you submit is your answer; if it differs from the fitter's,
   say why.
   ```json
   {"prediction": {"<period_id>": {"<band>": <value>, ...}, ...}, "rationale": "<why this design, in a few sentences>"}
   ```
