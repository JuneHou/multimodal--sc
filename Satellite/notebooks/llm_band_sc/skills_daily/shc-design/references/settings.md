# Fields for shc-design

`spec.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|
| `time_bandwidth_candidates` | non-empty list of numbers > 0 | the Gaussian bandwidths, in days, among which leave-one-out cross-validation picks each band's bandwidth for the local-linear smoother that estimates the band's latent trajectory |
| `m_min` | integer >= 1 | the shortest matching window tried, in visits |
| `m_step` | integer >= 1 | the step between the matching window lengths tried, in visits |
| `m_max` | integer >= 1, or null | the longest matching window tried, in visits; null = the longest that leaves at least one historical block |
| `max_donors` | integer >= 1 | the most historical blocks used; when more exist, this many are kept, spread evenly over the history |
| `nu` | number >= 0 | the matching window length is the longest m tried whose fit on all blocks has R <= nu, where R is the mean over bands of the band's mean squared matching error divided by the variance of its values over the visits given; if no m passes, the longest m tried |
| `zeta` | number >= 0 | in stepwise selection a further block is kept only if it lowers R by more than this |

## Choices

| field | options | meaning |
|---|---|---|
| `weight_constraint` | `simplex`, `nonneg`, `unconstrained` | simplex = weights >= 0 summing to 1; nonneg = weights >= 0, any sum; unconstrained = any real weights |
| `intercept` | `none`, `fitted` | none = no constant term; fitted = a constant is fitted alongside the block weights and added to the prediction |
| `band_weighting` | `equal`, `inverse_variance`, `subset` | equal = every band's rows enter the joint fit with weight 1; inverse_variance = each band's rows scaled by 1/SD of that band over the visits given in the fitting space; subset = fit on the bands listed in band_subset only (all bands are still predicted) |
| `fit_space` | `latent_std`, `raw` | latent_std = weights fitted on each band's smoothed, standardized trajectory; raw = weights fitted on the observed band values |
| `readd_seasonal` | `true`, `false` | false = predict as the weighted combination of the observed values that followed each kept block; true = predict in the latent space, undo the standardization, and add the seasonal term, which is zero for this method |
| `selection` | `greedy_forward`, `exhaustive` | greedy_forward = stepwise: the block with the lowest matching MSE first, then add one at a time the block that lowers R most, while R falls by more than zeta; exhaustive = every non-empty subset of the blocks, lowest R wins (zeta unused) |
| `band_subset` | list of band names | list of band names, required if and only if band_weighting is subset |

## Rules that tie the fields together

- m_min must be at most m_max.
- m_max plus the number of visits to predict must be at most the number of visits given, so that at least one historical block exists; with m_max null the same holds for m_min.
- readd_seasonal = true requires fit_space = latent_std (there is no seasonal term to add back in raw space).
- intercept = fitted requires fit_space = raw or readd_seasonal = true (a constant fitted in standardized latent units has no meaning added to an observed-value prediction).
- weight_constraint = nonneg or unconstrained requires fit_space = raw or readd_seasonal = true (weights fitted in standardized latent space keep the observed values' scale only when they sum to one).
- band_subset is given if and only if band_weighting = subset, and lists band names of this sensor without repeats.
- No field may refer to a visit to predict. The fit sees the visits given only.

## Shape of the file (every value is yours)

```json
{
 "time_bandwidth_candidates": "<list of numbers>",
 "m_min": "<int>",
 "m_step": "<int>",
 "m_max": "<int or null>",
 "max_donors": "<int>",
 "nu": "<float>",
 "zeta": "<float>",
 "weight_constraint": "<simplex|nonneg|unconstrained>",
 "intercept": "<none|fitted>",
 "band_weighting": "<equal|inverse_variance|subset>",
 "fit_space": "<latent_std|raw>",
 "readd_seasonal": "<true|false>",
 "selection": "<greedy_forward|exhaustive>",
 "band_subset": "<list of band names, only with subset>"
}
```
