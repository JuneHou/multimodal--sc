# Fields for shc-design

`spec.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|
| `m` | integer >= 1 | length in months of the matching window and of every historical block |
| `n` | integer >= 1 | how many months after each historical block that block is taken to predict; the first of them is used |
| `kernel_bandwidth` | number > 0 | Gaussian bandwidth, in months, of the local-linear smoother that estimates each band's latent trajectory |
| `zeta` | number >= 0 | in stepwise selection a further block is kept only if it lowers the matching MSE by more than this |
| `beta_trend_degree` | 0, 1 or 2 | degree of the time trend partialled out when the annual sine/cosine seasonal terms are estimated |

## Choices

| field | options | meaning |
|---|---|---|
| `weight_constraint` | `simplex`, `nonneg`, `unconstrained` | simplex = weights >= 0 summing to 1; nonneg = weights >= 0, any sum; unconstrained = any real weights |
| `intercept` | `none`, `fitted` | none = no constant term; fitted = a constant is fitted alongside the block weights and added to the prediction |
| `band_weighting` | `equal`, `inverse_variance`, `subset` | equal = every band's rows enter the joint fit with weight 1; inverse_variance = each band's rows scaled by 1/SD of that band over P01-P09 in the fitting space; subset = fit on the bands listed in band_subset only (all bands are still predicted) |
| `fit_space` | `latent_std`, `raw` | latent_std = weights fitted on each band's smoothed, deseasonalized, standardized trajectory; raw = weights fitted on the observed band values |
| `readd_seasonal` | `true`, `false` | false = predict as the weighted combination of the observed values that followed each kept block; true = predict in the latent space, undo the standardization, and add the target month's seasonal term |
| `selection` | `greedy_forward`, `exhaustive` | greedy_forward = stepwise: best single block first, then add one at a time while the MSE falls by more than zeta; exhaustive = every non-empty subset of the blocks, lowest MSE wins (zeta unused) |
| `band_subset` | list of band names | list of band names, required if and only if band_weighting is subset |

## Rules that tie the fields together

- m + n must be at most 9, so that at least one historical block exists (blocks = 9 - m - n + 1).
- readd_seasonal = true requires fit_space = latent_std (there is no seasonal term to add back in raw space).
- intercept = fitted requires fit_space = raw or readd_seasonal = true (a constant fitted in standardized latent units has no meaning added to an observed-value prediction).
- weight_constraint = nonneg or unconstrained requires fit_space = raw or readd_seasonal = true (weights fitted in standardized latent space keep the observed values' scale only when they sum to one).
- band_subset is given if and only if band_weighting = subset, and lists band names of this sensor without repeats.
- No field may refer to P10. The fit sees P01-P09 only.

## Shape of the file (every value is yours)

```json
{
 "m": "<int>",
 "n": "<int>",
 "kernel_bandwidth": "<float>",
 "zeta": "<float>",
 "beta_trend_degree": "<int>",
 "weight_constraint": "<simplex|nonneg|unconstrained>",
 "intercept": "<none|fitted>",
 "band_weighting": "<equal|inverse_variance|subset>",
 "fit_space": "<latent_std|raw>",
 "readd_seasonal": "<true|false>",
 "selection": "<greedy_forward|exhaustive>",
 "band_subset": "<list of band names, only with subset>"
}
```
