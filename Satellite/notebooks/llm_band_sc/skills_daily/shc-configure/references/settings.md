# Fields for shc-configure

`settings.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

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

## Rules

- m_min must be at most m_max.
- m_max plus the number of visits to predict must be at most the number of visits given, so that at least one historical block exists; with m_max null the same holds for m_min.
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
 "zeta": "<float>"
}
```
