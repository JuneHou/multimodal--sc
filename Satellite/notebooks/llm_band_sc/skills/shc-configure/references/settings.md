# Fields for shc-configure

`settings.json` holds exactly the fields below; there are no defaults, and a file with a field missing, a field misspelled, or a value outside its range is refused with the field named.

## Settings

| field | type and range | meaning |
|---|---|---|
| `m` | integer >= 1 | length in months of the matching window and of every historical block |
| `n` | integer >= 1 | how many months after each historical block that block is taken to predict; the first of them is used |
| `kernel_bandwidth` | number > 0 | Gaussian bandwidth, in months, of the local-linear smoother that estimates each band's latent trajectory |
| `zeta` | number >= 0 | in stepwise selection a further block is kept only if it lowers the matching MSE by more than this |
| `beta_trend_degree` | 0, 1 or 2 | degree of the time trend partialled out when the annual sine/cosine seasonal terms are estimated |

## Rules

- m + n must be at most 9, so that at least one historical block exists (blocks = 9 - m - n + 1).
- No field may refer to P10. The fit sees P01-P09 only.

## Shape of the file (every value is yours)

```json
{
 "m": "<int>",
 "n": "<int>",
 "kernel_bandwidth": "<float>",
 "zeta": "<float>",
 "beta_trend_degree": "<int>"
}
```
