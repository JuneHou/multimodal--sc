---
name: shc-configure
description: Predict the next month of a land parcel's satellite band values from its own
  nine-month history with the synthetic historical control method, by deciding the method's
  settings — block length, forecast horizon, smoothing bandwidth, selection tolerance, trend
  degree — and running the bundled fitter. Use this whenever a parcel's monthly band values
  are to be forecast one month ahead from its own past and the fitting code is provided.
---

# Configure and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier months are weighted to
reproduce its most recent months, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction; read it if anything below is unclear.

## What is fixed in this skill, and what you decide

The fitter makes these choices for you: each band is deseasonalized, smoothed and
standardized before matching; every band enters with equal weight; weights are
non-negative and sum to one, with no constant term; selection is stepwise; the prediction
is the weighted combination of the observed values that followed each kept block.

You decide the settings, all five, in `settings.json`:

- `m` — length of the matching window and of every block's matching part, in months
- `n` — how many months after a block it is taken to predict; the first is used
- `kernel_bandwidth` — bandwidth of the smoother, in months
- `zeta` — a further block is kept only if it lowers the matching error by more than this
- `beta_trend_degree` — 0, 1 or 2, the trend partialled out when the seasonal terms are estimated

`references/settings.md` gives each field's type, range and meaning, and the one rule that
ties them: `m + n` must be at most 9, so that at least one block exists. There are no
defaults; the fitter refuses an incomplete file and names what is missing.

## Workflow

1. Look at the data: `python scripts/show_data.py`. Nine months, every band, with the
   calendar month of each. Nothing else about this parcel exists.
2. Decide the settings from what you see. Short records reward thought here: a longer `m`
   matches more shape but leaves fewer blocks; a wider bandwidth smooths more; a large
   `zeta` keeps one block, a small one lets more in.
3. Write `settings.json`, then run `python scripts/fit.py settings.json`. It prints the
   blocks it kept, their weights, the matching error over P01–P09, and the prediction.
4. You may change the settings and run again. Every run is recorded. The matching error is
   an in-sample quantity; a smaller one is not by itself a better forecast.
5. Finish by writing `answer.json`:
   ```json
   {"prediction": {"<band>": <value>, ...}, "rationale": "<why these settings, in a few sentences>"}
   ```
   The prediction you submit is your answer. If it differs from the fitter's, say why.
