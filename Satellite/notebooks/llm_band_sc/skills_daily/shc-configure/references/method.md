# Synthetic historical control for one parcel

The parcel's own past is its comparison. Nothing from any other parcel is used.

## Historical blocks
From the visits given, form overlapping blocks. A block is m consecutive visits (its
matching part) followed by the n visits that came after it (its forward part), where n is
the number of visits to predict. Blocks slide one visit at a time, so there are
(visits given) − m − n + 1 of them; when there are more than max_donors, that many are kept,
spread evenly over the history. The last block's forward part ends at the last visit given.
The matching window is the parcel's most recent m visits.

## What is matched
Each band's visits can be matched as observed, or after preparation: the series is
smoothed along time, measured in days since the first visit, with a local-linear Gaussian
kernel whose bandwidth is chosen for each band by leave-one-out cross-validation among the
candidate bandwidths, and the smoothed series is standardized to mean 0 and standard
deviation 1 over the visits given. Bands are stacked, so one set of block weights serves
every band.

## Matching window length
The matching error R is the mean over bands of each band's mean squared matching error
divided by the variance of the band's values over the visits given. For each m from m_min
to m_max in steps of m_step, weights are fitted on all blocks, and the longest m whose R
is at most nu is used. If no m passes, the longest m tried is used.

## Weights and selection
Weights over the blocks are fitted so the weighted combination of the blocks' matching
parts reproduces the matching window, across all bands jointly, minimizing mean squared
error. A selection rule decides which blocks carry weight: stepwise (keep the single
best-fitting block, then add one block at a time while R falls by more than a
tolerance) or exhaustive (every non-empty subset, lowest R wins).

## Prediction
Each visit to predict is predicted as the same weighted combination of what followed each
kept block at the same position of its forward part — the k-th visit to predict from the
k-th visit of each block's forward part, as observed — or, when the fit was made on the
prepared series, in the prepared space with the standardization undone and the seasonal
term added back, which is zero for this method.

## What the fit can and cannot see
The fit sees the visits given only. The visits to predict, listed with their dates in
`visits_to_predict.csv`, are withheld: no script can read their values and no setting may
refer to them. The matching error reported by the fitter is the error on the matching
window inside the visits given, not the error on the visits to predict.
