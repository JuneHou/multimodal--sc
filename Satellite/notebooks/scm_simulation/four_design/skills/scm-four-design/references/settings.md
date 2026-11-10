# Fields for scm-four-design

`spec.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|

## Choices

| field | options | meaning |
|---|---|---|
| `predictors` | `every_outcome` with `covariate_periods`, `summaries` with `covariate_periods` and `summaries` | every_outcome = Z1, Z2 and the outcome in each pre-treatment period; summaries = Z1, Z2 and a few linear combinations of pre-treatment outcomes (p. 496) |
| `v` | `judgement` with `values`, `pre_period_fit`, `validation` with `training_end` | judgement = V set by the given weights; pre_period_fit = V chosen to fit the treated unit's outcome in periods 1 to 40; validation = V chosen on a validation period after a training period (p. 496) |
| `weights` | `convex`, `extrapolation` | convex = each donor weight at least 0 and the weights summing to one (the paper's definition and Synth); extrapolation = the weights sum to one and any weight may be below 0 or above one (p. 496) |
| `penalty` | `none`, `distance` with `lambda` | none = no penalty (Synth); distance = lambda times the sum over donors of the donor's weight times its V-weighted squared distance from the treated unit's scaled predictors (p. 496) |

## Values the options carry

| choice | option | value | type and range | meaning |
|---|---|---|---|---|
| `predictors` | `every_outcome` | `covariate_periods` | list of whole periods from 1 to 40 | the periods over which Z1 and Z2 are each averaged; one period gives the value in that period |
| `predictors` | `summaries` | `covariate_periods` | list of whole periods from 1 to 40 | the periods over which Z1 and Z2 are each averaged; one period gives the value in that period |
| `predictors` | `summaries` | `summaries` | list of objects, each mapping periods (text keys from "1" to "40") to weights, at least one weight not 0 | each summary is, for every unit, the sum of its weights times the unit's outcome in those periods |
| `v` | `judgement` | `values` | list of numbers above 0 | one weight for each predictor, in the order Z1, Z2, then the outcome predictors in their order |
| `v` | `validation` | `training_end` | whole period from 1 to 39 | the last training period s; the donor weights are fitted on periods 1 to s and V is chosen to fit the outcome in periods s + 1 to 40 |
| `penalty` | `distance` | `lambda` | number of 0 or above | how strongly distant donors are penalized |

## Rules that tie the fields together

- Every period is a whole number from 1 to 40, a number in `covariate_periods` and a text key such as "40" in a summary.
- Each choice holds `option` and exactly the values that option carries.
- Under `v` `validation`, every period of `covariate_periods` and of the summaries is at most `training_end`, and `every_outcome` uses the periods 1 to `training_end`.
- Under `v` `judgement`, `values` has one number for each predictor, Z1, Z2, then each outcome predictor in its order (40 outcome predictors under `every_outcome`, one per summary under `summaries`).
- `penalty` `distance` requires `weights` `convex`.

## Shape of the file (every value is yours)

```json
{
 "predictors": {
  "option": "<every_outcome|summaries>",
  "covariate_periods": "list of whole periods from 1 to 40, only with every_outcome or summaries",
  "summaries": "list of objects, each mapping periods (text keys from \"1\" to \"40\") to weights, at least one weight not 0, only with summaries"
 },
 "v": {
  "option": "<judgement|pre_period_fit|validation>",
  "values": "list of numbers above 0, only with judgement",
  "training_end": "whole period from 1 to 39, only with validation"
 },
 "weights": {
  "option": "<convex|extrapolation>"
 },
 "penalty": {
  "option": "<none|distance>",
  "lambda": "number of 0 or above, only with distance"
 }
}
```
