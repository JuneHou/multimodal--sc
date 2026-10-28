---
name: shc-configure
description: Predict a land parcel's satellite band values at each visit listed in
  visits_to_predict.csv from its own history of visits with the synthetic historical control
  method, by deciding the method's settings — candidate smoothing bandwidths, the matching
  window lengths tried, the number of blocks, the error threshold that picks the window
  length, selection tolerance — and running the bundled fitter. Use this whenever a parcel's
  band values are to be forecast at listed later visits from its own past and the fitting
  code is provided.
---

# Configure and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier visits are weighted to
reproduce its most recent visits, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction; read it if anything below is unclear.

## What is fixed in this skill, and what you decide

The fitter makes these choices for you: each band is smoothed and standardized before
matching; every band enters with equal weight; weights are non-negative and sum to one,
with no constant term; selection is stepwise; the prediction is the weighted combination
of the observed values that followed each kept block.

You decide the settings, all seven, in `settings.json`:

- `time_bandwidth_candidates` — the bandwidths of the smoother, in days, among which each
  band's is chosen by leave-one-out cross-validation
- `m_min`, `m_step`, `m_max` — the matching window lengths tried, in visits
- `max_donors` — the most blocks used
- `nu` — the matching window length is the longest one tried whose matching error R is at
  most this
- `zeta` — a further block is kept only if it lowers the matching error R by more than this

`references/settings.md` gives each field's type, range and meaning, and the rules that
tie them: `m_min` must be at most `m_max`, and `m_max` plus the number of visits to predict
must be at most the number of visits given, so that at least one block exists. There are no
defaults; the fitter refuses an incomplete file and names what is missing.

## Workflow

1. Look at the data: `python scripts/show_data.py`. Every visit given, every band, with
   the date of each, and the visits to predict. Nothing else about this parcel exists.
2. Decide the settings from what you see. Short records reward thought here: a longer
   matching window (`m_min` to `m_max`) matches more shape but leaves fewer blocks; a wider
   bandwidth smooths more; a large `zeta` keeps one block, a small one lets more in.
3. Write `settings.json`, then run `python scripts/fit.py settings.json`. It prints the
   matching window length chosen, the blocks it kept, their weights, the matching error
   over the visits given, and the prediction at every visit to predict.
4. You may change the settings and run again. Every run is recorded. The matching error is
   an in-sample quantity; a smaller one is not by itself a better forecast.
5. Finish by writing `answer.json`, with every band at every visit to predict:
   ```json
   {"prediction": {"<period_id>": {"<band>": <value>, ...}, ...}, "rationale": "<why these settings, in a few sentences>"}
   ```
   The prediction you submit is your answer. If it differs from the fitter's, say why.
