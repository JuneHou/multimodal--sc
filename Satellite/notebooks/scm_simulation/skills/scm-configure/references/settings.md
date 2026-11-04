# Fields for scm-configure

`settings.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|
| `special_predictors` | list of {variable, periods, op}, variable one of Z1, Z2, Y_observed, op one of mean, median, sd, var, min, max, sum | dataprep's special.predictors. Each entry is one predictor, the op of the variable over the listed periods |
| `predictors` | list of names from Z1, Z2, Y_observed, may be empty | dataprep's predictors. Each is one predictor, the predictors_op of the variable over time_predictors_prior |
| `predictors_op` | one of mean, median, sd, var, min, max, sum | dataprep's predictors.op |
| `time_predictors_prior` | list of periods | dataprep's time.predictors.prior, the periods over which predictors are summarized |
| `time_optimize_ssr` | list of periods | dataprep's time.optimize.ssr, the periods of the outcome over which the mean squared prediction error is minimized when V is searched |
| `controls_identifier` | list of donor names from donor_01 to donor_29 | dataprep's controls.identifier, the donor pool |
| `custom_v` | list of numbers above 0, one per predictor, or null | synth's custom.v. Null means V is searched |
| `optimxmethod` | list of names from BFGS, CG, Nelder-Mead, L-BFGS-B, nlm, nlminb, lbfgsb3c, Rcgmin, Rtnmin, Rvmmin, snewton, snewtonm, spg, ucminf, newuoa, bobyqa, uobyqa, nmkb, hjkb, hjn, lbfgs, subplex, ncg, nvm, mla, slsqp, tnewt, anms, pracmanm, nlnm, snewtm, All | synth's optimxmethod, the optimx methods of the V search (All runs every one) |
| `genoud` | true or false | synth's genoud |
| `quadopt` | ipop | synth's quadopt |
| `margin_ipop` | number above 0 | synth's Margin.ipop |
| `sigf_ipop` | integer above 0 | synth's Sigf.ipop |
| `bound_ipop` | number above 0 | synth's Bound.ipop |

## Rules

- Every period in time_predictors_prior, time_optimize_ssr and the periods of special_predictors is a whole period number.

## Shape of the file (every value is yours)

```json
{
 "special_predictors": "<list of {\"variable\": <name>, \"periods\": <list of periods>, \"op\": <name>}>",
 "predictors": "<list of names>",
 "predictors_op": "<one of the listed values>",
 "time_predictors_prior": "<list of numbers>",
 "time_optimize_ssr": "<list of numbers>",
 "controls_identifier": "<list of names>",
 "custom_v": "<list of numbers, or null>",
 "optimxmethod": "<list of names>",
 "genoud": "<true or false>",
 "quadopt": "<one of the listed values>",
 "margin_ipop": "<float>",
 "sigf_ipop": "<int>",
 "bound_ipop": "<float>"
}
```
