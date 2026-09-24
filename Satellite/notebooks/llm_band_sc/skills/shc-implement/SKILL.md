---
name: shc-implement
description: Predict the next month of a land parcel's satellite band values from its own
  nine-month history by implementing the synthetic historical control method yourself in
  Python and running it. Use this whenever a parcel's monthly band values are to be
  forecast one month ahead from its own past and no fitting code is provided.
---

# Implement and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier months are weighted to
reproduce its most recent months, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction; every setting and every choice in it
is yours, made inside your program.

## Workflow

1. `python scripts/show_data.py` — the nine months, every band. The same data is in
   `data.csv` in this directory: one row per band per month, columns `band`, `period_id`
   (P01–P09 in time order), `month` (YYYY-MM), `mean_value`.
2. Read `references/method.md`. Decide how to build the blocks, what to match, how to fit
   and constrain the weights, how to select blocks, and how to form the prediction.
3. Write `predict.py`. It reads `data.csv` and prints one line per band, `BAND VALUE`, the
   predicted mean value for the next month. numpy, scipy and pandas are available; nothing
   outside this directory is, and there is no network.
4. Run `python predict.py`. Fix what fails; run again. Every run is recorded.
5. Finish by writing `answer.json`:
   ```json
   {"prediction": {"<band>": <value>, ...}, "rationale": "<what the program does and why>"}
   ```
