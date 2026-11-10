# Synthetic control for one treated unit

Abadie, Diamond and Hainmueller (2010), Sections 2.2 and 2.3 (pp. 494 to 496), as the Synth R package implements it.

## The quantity estimated
There are J + 1 units. Unit 1 is exposed to the treatment from period T0 + 1 on, and the other J units, the donors,
are never exposed. For each treated period t the effect is the treated unit's observed outcome minus the outcome it
would have had without treatment, Y1t − Y1t(N). The second term is never observed and is what the method estimates.

## The synthetic control
A synthetic control is a weighted combination of donors, W = (w2, ..., wJ+1), with every weight at least 0 and the
weights summing to one. Its outcome Σj wj Yjt estimates Y1t(N), and Y1t − Σj wj Yjt estimates the effect in period t.
The paper notes that weights below 0 or above one can be used, at the cost of extrapolating beyond the donors (p. 496).

## Predictors
The weights are chosen so the synthetic control resembles the treated unit before treatment. X1 holds k
pre-treatment characteristics of the treated unit, its covariates and any linear combinations of its pre-treatment
outcomes (for example the outcome in one period, or its average over several periods). X0 holds the same
characteristics for each donor. One choice is the outcome in every pre-treatment period. The paper notes that a few
linear combinations can be used instead, checking that the resulting weights still match the pre-treatment outcomes
approximately (p. 496). In Synth, `predictors` are summarized with `predictors_op` over `time_predictors_prior`, and
each entry of `special_predictors` is one predictor with its own periods and summary. Synth divides every predictor by
its standard deviation across the treated unit and the donors before matching.

## The weights given V
W is chosen to minimize (X1 − X0 W)' V (X1 − X0 W) under the weight constraints, where V is a diagonal matrix of
non-negative predictor weights. Synth solves this quadratic program with `kernlab::ipop` (`quadopt`), whose
precision and bounds are `margin_ipop`, `sigf_ipop` and `bound_ipop`. The paper notes that W can instead minimize this
distance plus penalty terms that grow with the distance between the treated unit and each donor with positive weight,
which keeps distant donors out (p. 496).

## Choosing V
The choice of V affects the estimator's error (p. 496). V can come from a judgment of how well each predictor predicts
the outcome (`custom_v`, one positive number per predictor, normalized by Synth to sum to one). V can be chosen from
the data so the synthetic control reproduces the treated unit's pre-treatment outcome path, minimizing the mean
squared prediction error of the outcome over pre-treatment periods (`custom_v` null and `time_optimize_ssr` those
periods). This is the paper's own choice in its application. With enough pre-treatment periods, W can be computed
on an earlier training stretch and V chosen to minimize the prediction error over a later validation stretch.
When V is searched, Synth starts its optimizer (`optimxmethod`, optionally `genoud` first) from equal weights and from
regression-based weights and keeps the better result.

## Donors and fit
The donor pool can be restricted to units similar to the treated unit, which reduces interpolation bias
(`controls_identifier`). The paper adds that when no weighted combination of donors matches the treated unit's
pre-treatment characteristics, the fit may be poor, and the analyst decides whether the match is close enough (p. 495).

## The result
After fitting, the synthetic control's outcome in every period is Σj wj Yjt, and the gap is the treated unit's
observed outcome minus it. Before treatment the gap shows the fit. After treatment it is the estimated effect, and
its average over the treated periods is the average treatment effect on the treated unit.
