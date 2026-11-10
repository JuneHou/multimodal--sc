# How three language models built the counterfactual of a simulated disaster from a prompt alone

## The replies and chains were read in full, and every quote was found in its record.

This file describes how gpt-oss-120b, GLM-5.3 and DeepSeek-V4.1-Flash-thinking-low (DeepSeek below) built the counterfactual of the target for periods 41 to 60, first from the target alone and then with 29 donors. It covers all 39 records, 36 at default effort and three at low effort. Grok 4.7, run by Jun, read every reply and every reasoning chain in full and wrote one reading per record. The 23 records whose quotes left more than half of the chain unquoted were read again in parts of 20,000 characters, 112 parts in all. Every quote in the reading and in this file was found by exact search in its record, and each part's quote lies inside its part.

Each scored answer is set beside the rules of the step analysis over periods 41 to 60. The pre-period mean is the target's mean over periods 1 to 40. The value at period 40 is the target's value at period 40. The least squares line is the line fitted to the target's periods 1 to 40. The shifted donor mean is the donor mean at each period plus the target's mean over periods 1 to 40 minus the donor mean over periods 1 to 40. The README synthetic control is the README Section 3 synthetic control on the same panel. Answers without donors are set beside the first three rules and answers with donors beside all five. The distance to a rule is the root mean squared difference over periods 41 to 60, and the nearest rule is the rule at the smallest distance. A case agrees when the method its values come from is its nearest rule.

Eight GLM-5.3 records have no reply, because the ARC server closed each reply at 30 minutes before any reply text. Their entries describe the chain up to where it stops. Three gpt-oss-120b default records hold an answer line the parser did not accept, and the low effort rerun of each is the scored answer.

## Without donors, the models forecast the target from its own first 40 periods.

Method of each record without donors, the method its reply names, its nearest rule and the distance to that rule.

| σ | DGP | model | effort | method the values come from | method the reply names | nearest rule | distance | agrees |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1 | gpt-oss-120b | default | line from the value at period 40 with a slope from two half means | linear trend extrapolation | value at period 40 | 0.159 | no |
| 1.0 | 1 | GLM-5.3 | default | pre-period mean | white noise around the mean | pre-period mean | 0.002 | yes |
| 1.0 | 1 | DeepSeek | default | AR(1) with intercept | AR(1) | pre-period mean | 0.047 | no |
| 1.0 | 2 | gpt-oss-120b | default | decay from the value at period 40 toward the pre-period mean | exponential smoothing type decay | least squares line | 0.229 | no |
| 1.0 | 2 | GLM-5.3 | default | seasonal regression with two harmonics of a ten period cycle | two harmonic seasonal model | pre-period mean | 0.507 | no |
| 1.0 | 2 | DeepSeek | default | AR(1) with intercept | AR(1) | pre-period mean | 0.030 | no |
| 0.5 | 1 | gpt-oss-120b | default | exponential smoothing level repeated | exponential smoothing | no answer line accepted | no answer | no answer |
| 0.5 | 1 | gpt-oss-120b | low | mean of periods 31 to 40 with random variation | autoregressive type projection | least squares line | 0.175 | no |
| 0.5 | 1 | GLM-5.3 | default | pre-period mean | white noise around the mean | pre-period mean | 0.002 | yes |
| 0.5 | 1 | DeepSeek | default | AR(1) with intercept | AR(1) | pre-period mean | 0.049 | no |
| 0.5 | 2 | gpt-oss-120b | default | least squares line on the period index | linear trend | least squares line | 0.001 | yes |
| 0.5 | 2 | GLM-5.3 | default | Yule-Walker autoregression, cut before any value | no reply | no answer | no answer | no answer |
| 0.5 | 2 | DeepSeek | default | AR(1) around the pre-period mean | AR(1) | pre-period mean | 0.024 | no |
| 0.2 | 1 | gpt-oss-120b | default | path from the value at period 40 down to the pre-period mean | linear trend converging to the mean | pre-period mean | 0.156 | no |
| 0.2 | 1 | GLM-5.3 | default | seasonal means with MA(1), cut before any value | no reply | no answer | no answer | no answer |
| 0.2 | 1 | DeepSeek | default | AR(1) with intercept | AR(1) | pre-period mean | 0.035 | no |
| 0.2 | 2 | gpt-oss-120b | default | pre-period mean with small noise | mean reversion by exponential smoothing | pre-period mean | 0.005 | yes |
| 0.2 | 2 | GLM-5.3 | default | sine and cosine of a twenty period cycle with AR(1), cut before any value | no reply | no answer | no answer | no answer |
| 0.2 | 2 | DeepSeek | default | AR(2) by Yule-Walker | AR(2) | pre-period mean | 0.021 | no |

### At σ = 1.0 in DGP 1, GLM-5.3 gave the pre-period mean and the other two models started from the value at period 40.

**gpt-oss-120b, default effort.** The chain names ARIMA, exponential smoothing and a moving average, then takes a slope from the means of periods 1 to 20 and periods 21 to 40. It starts the line at the value at period 40 and extends it to period 60. The nearest rule is the value at period 40, since the line starts there and falls slowly. The method is a line, so the case does not agree.

> Chain. Add note that this is simple linear trend extrapolation.

> Reply. Counterfactual forecast (simple linear‑trend extrapolation)

**GLM-5.3, default effort.** The chain computes the mean of periods 1 to 40, checks the lag one correlation and a five period cycle, and drops both. It sets every period from 41 to 60 to the mean. The method is the nearest rule.

> Chain. use the sample mean ≈ 0.10 for all periods

**DeepSeek, default effort.** The chain fits an AR(1) with intercept from the lag one products and iterates it from the value at period 40. It notes that the lag one correlation is not significant, compares the AR(1) with a white noise mean, and keeps the AR(1). The path returns to its mean within a few periods, so the nearest rule is the pre-period mean, which is not the method.

> Chain. I'll go with AR(1) forecasts.

> Reply. Using an AR(1) model fitted to the pre-disaster series

### At σ = 1.0 in DGP 2, no answer came from the method of its nearest rule.

**gpt-oss-120b, default effort.** The chain computes the mean of periods 1 to 40 and moves from the value at period 40 toward that mean, keeping nine tenths of the remaining gap each period. No rule describes such a decay, and the nearest is the least squares line.

> Reply. The series is modelled as a simple exponential‑smoothing‑type decay

**GLM-5.3, default effort.** The chain treats the series as a seasonal cycle with AR(1) noise and fits two harmonics of a ten period cycle. Periods 51 to 60 repeat periods 41 to 50. None of the three rules has a cycle, and every distance is 0.507 or more.

> Chain. two-harmonic seasonal model

> Reply. A two-harmonic seasonal model gives the no-disaster baseline.

**DeepSeek, default effort.** The chain fits an AR(1) with intercept, notes that its coefficient is not significant and that a flat mean might be better, and keeps the AR(1). The path returns to the mean within a few periods, and the nearest rule is the pre-period mean.

> Chain. We can state using AR(1).

> Reply. Using an AR(1) model fitted to the pre-disaster periods 1–40

### At σ = 0.5 in DGP 1, GLM-5.3 gave the pre-period mean and gpt-oss-120b was scored at low effort.

**gpt-oss-120b, default effort.** The chain updates an exponential smoothing level through period 40 and repeats that level for every later period. The parser did not accept its answer line.

> Reply. Using a simple exponential‑smoothing model

**gpt-oss-120b, low effort.** The chain names ARIMA and a linear trend, then averages periods 31 to 40 and decides on a random walk around that average. No rule uses the last ten periods alone, and the nearest is the least squares line.

> Chain. I'll generate a simple random walk around -0.2.

> Reply. a simple autoregressive‑type projection was generated

**GLM-5.3, default effort.** The chain estimates an AR(1) by Yule-Walker and finds that it changes only the first steps. It checks a trend and cycles of ten and twelve periods, finds none, and gives the mean for every period. The method is the nearest rule.

> Chain. I think giving the mean for all periods is cleanest and defensible.

**DeepSeek, default effort.** The chain fits an AR(1) with intercept on consecutive pairs of periods 1 to 40, finds the lag two correlation near zero, and keeps the AR(1). The path returns to its mean within a few periods, and the nearest rule is the pre-period mean.

> Chain. We can mention fitted AR(1) model.

> Reply. Using an AR(1) model fitted to the pre-disaster periods 1–40

### At σ = 0.5 in DGP 2, gpt-oss-120b fitted the least squares line and GLM-5.3 gave no reply.

**gpt-oss-120b, default effort.** The chain regresses the value on the period index over periods 1 to 40 and evaluates the line at periods 41 to 60. The method is the nearest rule.

> Chain. Maybe use simple linear regression: period vs value.

> Reply. Using a simple linear trend fitted to the pre‑disaster data

**GLM-5.3, default effort.** The chain names ARIMA and fits a Yule-Walker autoregression of order three. It then starts the autocorrelations again and stops while multiplying the series by its values two periods earlier.

> Chain. Need maybe use ARIMA/autoregression.

**DeepSeek, default effort.** The chain fits an AR(1) around the mean of periods 1 to 40 and lets the gap at period 40 die out. The nearest rule is the pre-period mean.

> Chain. I used an AR(1) forecast fitted to the pre-disaster series.

> Reply. Using an AR(1) forecast fitted to the pre-disaster periods 1–40

### At σ = 0.2 in DGP 1, both answers end at a mean and GLM-5.3 gave no reply.

**gpt-oss-120b, default effort.** The chain computes the mean and a slope from the first to the last value, then writes a path from the value at period 40 down to the mean. The chain calls this path a moving average trending to the mean, and the reply calls it a linear trend converging to the mean. The path ends at the mean, and the nearest rule is the pre-period mean.

> Chain. I'll produce a simple forecast using a moving average of last 5 periods trending to mean.

> Reply. I fitted a very basic linear‑trend model to the pre‑disaster data

**GLM-5.3, default effort.** The chain computes the mean, finds a negative lag one correlation and names an MA(1). It groups the series by position in a twelve period cycle and stops while recomputing the MA(1) residuals.

> Chain. MA(1) fits better. Go with MA(1).

**DeepSeek, default effort.** The chain fits an AR(1) with intercept and notes that its coefficient may not be significant, so a mean model might be preferred. It keeps the AR(1), whose path settles at the model's own mean rather than the mean of periods 1 to 40. The nearest rule is the pre-period mean.

> Chain. So mean model may be preferred.

> Reply. Using an AR(1) forecast fitted to periods 1–40

### At σ = 0.2 in DGP 2, gpt-oss-120b gave the pre-period mean with noise and GLM-5.3 gave no reply.

**gpt-oss-120b, default effort.** The chain computes the mean, averages the last five values, and rejects a line through periods 31 to 40 as too low. It sets an exponential smoothing forecast equal to the mean and adds small noise. The method is the pre-period mean with noise added, which is the nearest rule.

> Chain. We can generate using simple exponential smoothing forecast = mean = 0.04.

**GLM-5.3, default effort.** The chain names an AR(1) or AR(2) with a seasonal term and defines a sine and cosine of a twenty period cycle. It stops while computing the residuals of a plain AR(1).

> Chain. an AR(1) or AR(2) process with some seasonal component.

**DeepSeek, default effort.** The chain fits an AR(2) by Yule-Walker on deviations from the mean and iterates it from the values at periods 39 and 40. It also checks a line, an AR(1) and seasonal lags, and keeps the AR(2). The nearest rule is the pre-period mean.

> Chain. using an AR(2) model fitted to periods 1-40

> Reply. Using an AR(2) forecast fitted to periods 1–40

## With 29 donors, the models built the counterfactual from the donor mean or from a few donors.

Method of each record with donors, the method its reply names, its nearest rule and the distance to that rule.

| σ | DGP | model | effort | method the values come from | method the reply names | nearest rule | distance | agrees |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1 | gpt-oss-120b | default | equal weight donor mean | ordinary least squares synthetic control | no answer line accepted | no answer | no answer |
| 1.0 | 1 | gpt-oss-120b | low | no step in the chain gives the printed values | ordinary least squares weights that are nonnegative and sum to one | pre-period mean | 0.593 | no |
| 1.0 | 1 | GLM-5.3 | default | ordinary least squares on the donor mean, slope 1.27 | synthetic control style predictor | shifted donor mean | 0.278 | no |
| 1.0 | 1 | DeepSeek | default | ordinary least squares on the donor mean, slope 1.253 | donor average regression | shifted donor mean | 0.260 | no |
| 1.0 | 2 | gpt-oss-120b | default | equal weight donor mean | weighted synthetic control | shifted donor mean | 0.104 | no |
| 1.0 | 2 | GLM-5.3 | default | five donor weights, cut before any value | no reply | no answer | no answer | no answer |
| 1.0 | 2 | DeepSeek | default | ordinary least squares on the donor mean, slope 0.31 | none | pre-period mean | 0.062 | no |
| 0.5 | 1 | gpt-oss-120b | default | equal weight donor mean | donor average | no answer line accepted | no answer | no answer |
| 0.5 | 1 | gpt-oss-120b | low | no step in the chain gives the printed values | ridge synthetic control weights | pre-period mean | 0.243 | no |
| 0.5 | 1 | GLM-5.3 | default | ranking of donors, cut before any value | no reply | no answer | no answer | no answer |
| 0.5 | 1 | DeepSeek | default | ordinary least squares on the donor mean, slope 1.27 | donor average regression | shifted donor mean | 0.277 | no |
| 0.5 | 2 | gpt-oss-120b | default | equal weight donor mean | regression on the 29 donors | shifted donor mean | 0.049 | no |
| 0.5 | 2 | GLM-5.3 | default | donor sums for each period, cut before any value | no reply | no answer | no answer | no answer |
| 0.5 | 2 | DeepSeek | default | ordinary least squares on the donor mean, slope 0.945 | pre-disaster relationship with the donor pool | shifted donor mean | 0.006 | no |
| 0.2 | 1 | gpt-oss-120b | default | equal weight donor mean | none | shifted donor mean | 0.036 | no |
| 0.2 | 1 | GLM-5.3 | default | regression on four donors, cut before any value | no reply | no answer | no answer | no answer |
| 0.2 | 1 | DeepSeek | default | least squares on three donors with an intercept | none | README synthetic control | 0.484 | no |
| 0.2 | 2 | gpt-oss-120b | default | equal weight donor mean | synthetic control with weights that sum to one | shifted donor mean | 0.017 | no |
| 0.2 | 2 | GLM-5.3 | default | equal weights on four donors, cut before any value | no reply | no answer | no answer | no answer |
| 0.2 | 2 | DeepSeek | default | ordinary least squares on the donor sum | none | shifted donor mean | 0.079 | no |

### At σ = 1.0 in DGP 1, GLM-5.3 and DeepSeek regressed the target on the donor mean.

**gpt-oss-120b, default effort.** The chain names a synthetic control and an ordinary least squares fit of the target on the donors, calls that fit too heavy, and averages the 29 donors in each period. The reply calls the result an ordinary least squares synthetic control. The parser did not accept its answer line.

> Chain. we used simple average as proxy.

> Reply. using the 29 donor sites as a synthetic control

**gpt-oss-120b, low effort.** The chain sets out to regress the target on the donors, calls that too heavy, averages the donors at period 41, and decides on that donor mean for every period. The reply describes constrained least squares weights and prints a different value in each period, starting at the donor mean of period 41. The reading finds no step in the chain that gives the later values. The nearest rule is the pre-period mean.

> Chain. I'll output using donor mean approximated as 0.7 for all periods (simplify).

> Reply. the regression coefficients were constrained to be non‑negative and to sum to

**GLM-5.3, default effort.** The chain names a synthetic control with constrained weights, ridge and ordinary least squares. It regresses the target on the equal weight donor mean over periods 1 to 40, checks factor loadings and a slope shrunk toward one, and keeps the regression. The shifted donor mean moves the counterfactual one for one with the donor mean, while the model scales the donor mean by its fitted slope.

> Chain. DECISION: Go with Ŷ_t = -0.05 + 1.27·A_t.

> Reply. Since the donors are unaffected by the disaster, I built a synthetic‑control‑style predictor from the donor pool

**DeepSeek, default effort.** The chain names a synthetic control, a factor model and matrix completion, and regresses the target on the donor mean over periods 1 to 40. It applies that line to the donor mean of periods 41 to 60. Its fitted slope differs from the one for one move of the shifted donor mean.

> Chain. So counterfactual = -0.05 +1.253*donor_avg_post.

> Reply. Using a donor-average regression calibrated on periods 1–40

### At σ = 1.0 in DGP 2, DeepSeek computed the shifted donor mean and then chose a regression.

**gpt-oss-120b, default effort.** The chain names a synthetic control with weights that are nonnegative and sum to one, calls that fit too heavy, and averages the 29 donors in each period. The reply calls the result a weighted combination of the donors. The shifted donor mean adds the pre-period gap between the target and the donors to the donor mean, and the model's values are the donor mean without that gap.

> Chain. Now we have counterfactual series (average donors).

> Reply. fitting a weighted combination of the 29 donor sites

**GLM-5.3, default effort.** The chain calls a full constrained regression infeasible by hand, picks five donors, and computes each weight as a covariance over a variance. It starts shifting those weights so that they sum to one and stops there.

> Chain. Doing a full constrained regression (weights sum to 1, nonnegative) by hand is infeasible.

**DeepSeek, default effort.** The chain names a synthetic control, a factor model and matrix completion, then computes the donor mean shifted by the pre-period gap, which is the fourth rule. It then regresses the target on the donor mean, finds a slope near 0.31, and chooses the regression over the shifted donor mean. With a slope that low, its path stays close to the target's mean, and the nearest rule is the pre-period mean.

> Chain. I'll use the donor-average difference-in-differences.

> Chain. I'll choose regression on donor average because it's a proper regression forecast.

### At σ = 0.5 in DGP 1, the gpt-oss-120b reply describes weights its chain never computed.

**gpt-oss-120b, default effort.** The chain names a synthetic control fit, calls it too large, and averages the 29 donors in each period. The parser did not accept its answer line.

> Chain. Use average of donors for each period as counterfactual.

> Reply. (estimated as the average of the 29 donor sites for each period)

**gpt-oss-120b, low effort.** The chain sets out to fit a synthetic control, rejects the regression as too heavy, and switches to the donor mean. It finds averaging the donors by eye too much and ends on a placeholder. The reply describes ridge regression weights on the donors. The reading finds no step in the chain that gives the printed values. The nearest rule is the pre-period mean.

> Chain. Might just state using synthetic control but not compute exact numbers.

> Reply. A ridge‑regularised linear regression (no intercept) was run

**GLM-5.3, default effort.** The chain names a synthetic control, ordinary least squares, ridge, a factor model and matrix completion. It estimates the target's loading on the donor mean and then computes sums for each donor to rank donors by correlation. It stops at donor 17, in the product for period 40.

> Chain. So target has loading 1.27 on common factor estimated by donor average.

**DeepSeek, default effort.** The chain names a synthetic control, a factor model and matrix completion. It regresses the target on the sum of the 29 donors over periods 1 to 40 and evaluates the line at the donor sums of periods 41 to 60. Its fitted slope differs from the one for one move of the shifted donor mean.

> Chain. Donor average regression slope 1.27.

> Reply. Using a donor-average regression fit on periods 1–40

### At σ = 0.5 in DGP 2, the slope of DeepSeek's regression is near one and its answer lies close to the shifted donor mean.

**gpt-oss-120b, default effort.** The chain names a synthetic control regression, calls it too large, and averages the 29 donors in each period. The reply calls the result a regression of the target on the 29 donors. The model's values are the donor mean without the pre-period gap that the nearest rule adds.

> Chain. Could produce synthetic control using simple average of donors as counterfactual.

> Reply. target series is regressed on the 29 donor series

**GLM-5.3, default effort.** The chain names a classic synthetic control and starts summing the donors in each period to compare their mean with the target. It stops in period 6.

> Chain. This is classic synthetic control.

**DeepSeek, default effort.** The chain regresses the target on the donor mean and finds a slope of 0.945. It tries nonnegative weights on two and then three donors, finds they cannot match the pre-period highs and lows at once, and returns to the regression. With a slope near one, its path lies close to the shifted donor mean, at a distance of 0.006.

> Chain. So regression: target ≈ 0.0512 + 0.945 * donor_avg.

> Reply. Using the pre-disaster relationship between the target and the donor pool

### At σ = 0.2 in DGP 1, DeepSeek fitted three donors and lies nearest the README synthetic control.

**gpt-oss-120b, default effort.** The chain names a synthetic control regression, calls it too heavy, and averages the 29 donors in each period by hand. The reply names no method. The model's values are the donor mean without the pre-period gap that the nearest rule adds.

> Chain. Simpler: compute mean of donors for each period as counterfactual.

**GLM-5.3, default effort.** The chain names a synthetic control and ridge, fits three donors by regression, and adds a fourth with a negative slope. It starts adding a fifth donor and stops while checking that donor's values.

> Chain. Ridge or synthetic control (nonnegative weights summing to 1) is typical.

**DeepSeek, default effort.** The chain names a synthetic control and a factor model and fits three donors by least squares. It also fits a line on four donors with one negative weight and drops it because synthetic control weights are nonnegative. It applies the three donor line to periods 41 to 60. The README synthetic control chooses weights over all 29 donors that are nonnegative and sum to one, with no intercept, while the model's line uses three donors and an intercept. The distance to it is 0.484.

> Chain. Synthetic control weights are nonnegative.

> Chain. So use 3-donor.

### At σ = 0.2 in DGP 2, the gpt-oss-120b reply calls the donor mean a synthetic control.

**gpt-oss-120b, default effort.** The chain names a synthetic control with weights that are nonnegative and sum to one, calls it too large, and averages the 29 donors in each period. The reply says the values come from synthetic control weights that sum to one. The model's values are the donor mean without the pre-period gap that the nearest rule adds.

> Chain. Thus average donor series yields counterfactual values.

> Reply. Using the 29 donor sites as a synthetic control

**GLM-5.3, default effort.** The chain names a standard synthetic control, regresses the target on two donors, and then tests equal weights on four donors. It stops while comparing that average with the target in period 17.

> Chain. possibly constrained to be nonnegative and sum to 1, as in standard SCM

**DeepSeek, default effort.** The chain names a synthetic control with weights that are nonnegative and sum to one and calls a full fit impossible by hand. It regresses the target on the sum of the donors over periods 1 to 40 and keeps that regression. The nearest rule is the shifted donor mean.

> Chain. I'll use regression-adjusted donor mean because it's a standard counterfactual estimate.
