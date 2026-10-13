# Synthetic historical control for one parcel

The parcel's own past is its comparison. Nothing from any other parcel is used.

## Historical blocks
From the nine observed months P01–P09, form overlapping blocks. A block is m consecutive
months (its matching part) followed by the n months that came after it (its forward part).
Blocks slide one month at a time, so there are 9 − m − n + 1 of them; the last block's
forward part ends at P09. The matching window is the parcel's most recent m months.

## What is matched
Each band's nine months can be matched as observed, or after preparation: an annual sine
and cosine term is estimated for the band (after partialling out a low-order time trend in
the month index) and removed, the remainder is smoothed along time with a local-linear
Gaussian kernel of a chosen bandwidth, and the smoothed series is standardized to mean 0
and standard deviation 1 over P01–P09. Bands are stacked, so one set of block weights
serves every band.

## Weights and selection
Weights over the blocks are fitted so the weighted combination of the blocks' matching
parts reproduces the matching window, across all bands jointly, minimizing mean squared
error. A selection rule decides which blocks carry weight: stepwise (keep the single
best-fitting block, then add one block at a time while the error falls by more than a
tolerance) or exhaustive (every non-empty subset, lowest error wins).

## Prediction
The next month is predicted as the same weighted combination of what followed each kept
block — its first forward month, as observed — or, when the fit was made on the prepared
series, in the prepared space with the standardization undone and the target month's
seasonal term added back.

## What the fit can and cannot see
The fit sees P01–P09 only. The month to predict, P10 (2024-08), is withheld: no script
can read it and no setting may refer to it. The matching error reported by the fitter is
the error on the matching window inside P01–P09, not the error on P10.
