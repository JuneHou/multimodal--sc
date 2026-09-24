# Synthetic historical control on the monthly latent panel: method, implementation, first results

**Progress report for the meeting of 10 September 2026.** Sources for the method: Chen, Yang & Yang, *Synthetic Historical Control for Policy Evaluation*, SSRN 4995085; the collaborator's slides *Synthetic Historical Control for Policy Evaluation: Evidence from Satellite Data* (4 September 2026), whose walkthrough §2 follows; the decisions of the 4 September meeting. The synthetic-historical-control results of the 3 September report were withdrawn on 4 September. Scripts and tables: §7.

> **Corrected 16 September 2026.** The evaluation in §3–§5 originally reported the prediction error of the counterfactual built from the *smoothed* forward rows. That compares a denoised prediction against a noisy observation; the paper's eq 27 applies the fitted weights to the *observed* forward rows, so that both sides carry observation noise. §3's primary/secondary designation, the prediction-error columns of §4, and all of §5 now report the eq-27 evaluation, and the superseded values are labelled where they are kept. The conclusion changes: on Sentinel-1 the synthetic counterfactual does not beat the site's own mean at any $n$ (originally reported as 11 of 12 sites at $n = 2$), and on Sentinel-2 it beats it only at $n = 2$. §6 is unaffected — the band-feature pipeline always predicted from observed forward rows — but §6.1 now states which variance denominator its table uses, since that column is not directly comparable to the band pipeline's own published per-site numbers.

---

## 1. Data: the monthly composites

**Summary.** Monthly median composites over 21 months; 27 treated sites, each with 10 matched counterfactual sites; P11 (September 2024) is the treatment period; the outcome is the 980-value TerraMind latent of each composite, under two cloud fills.

- Composites: calendar-month medians of the daily Sentinel-1 and Sentinel-2 acquisitions, P01 (November 2023) to P21 (July 2025), 101 × 101 pixels at 10 m, bands and channel-first layout as the biweekly set.
- Periods: P01–P10 pre-hurricane; P11, September 2024 (landfall 27 September), the treatment period; P12 onward post. The Sentinel-2 September composite (2, 7 and 22 September) contains no post-landfall image; the Sentinel-1 composite includes 28 September. The collaborator's band pipeline excludes P11 as a transition month.

| | Sentinel-1 | Sentinel-2 |
|---|---|---|
| treated sites | 12 (sites 0015–0027) | 15 (sites 0001–0015) |
| matched counterfactual sites (10 per treated site) | 120 | 150 |
| months | 21 | 21 |

- Cloud fill, two arms (TerraMind cannot encode a masked pixel). *Chip-mean fill*: a masked pixel takes the chip's band mean. *Historical fill*: the per-pixel median of the site's clean pre-hurricane months P01–P10 (at least three months with every band finite), then the one-month template, then the chip mean; 98.1 % of the masked pixels at the Sentinel-2 treated sites come from the ten-month median. Masked-pixel share by month at the Sentinel-2 treated sites:

| P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | P17 | P18 | P19 | P20 | P21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.08 | 0.23 | 0.18 | 0.17 | 0.02 | 0.00 | 0.00 | 0.53 | 0.27 | 0.06 | 0.00 | 0.00 | 0.17 | 0.44 | 0.00 | 0.00 |

- Outcome: the TerraMind tokenizer latent, 5 channels × 14 × 14 parcels = 980 FSQ codes on [0, 1] (101 → 224 bilinear resize, TerraMind v1 standardization). The channels share units, so no cross-coordinate standardization. A site's parcels are registered across months, which is what makes the site's own past usable as a donor.

---

## 2. The method, in the source's order

**Summary.** One unit's own history. Stage 1 estimates the smooth untreated component $\ell_t$ of the pre-period; stage 2 rebuilds the post-period $\ell_t$ from the unit's own earlier blocks with convex weights; the effect is the observed post-period outcome minus that counterfactual. Presented as in the collaborator's slides, with the paper's equation numbers.

**Model (eq 2).**

$$ y_t = x_t^{\top}\beta + \ell_t + \delta_t d_t + \varepsilon_t, \qquad d_t = \mathbb{1}[t > T_0], $$

$x_t$ observed covariates, $\ell_t$ an unobserved smooth function of time (what would have happened without the intervention), $\delta_t$ the time-varying effect, $\varepsilon_t$ noise. Before $T_0$ the residual $y_t - x_t^{\top}\beta$ is $\ell_t + \varepsilon_t$ and $\ell_t$ is learnable; after $T_0$ it is $\ell_t + \delta_t + \varepsilon_t$ and the two cannot be separated from that series alone.

**Stage 1: the latent component (eqs 18–21; slide 6).** With covariates: local-linear kernel estimates $\hat{\mu}_{y,t} \approx E(y_t \mid t)$ and $\hat{\mu}_{x,t} \approx E(x_t \mid t)$; residualize, $\tilde y_t = y_t - \hat{\mu}_{y,t}$, $\tilde x_t = x_t - \hat{\mu}_{x,t}$; $\hat\beta = (\sum \tilde x_t \tilde x_t^{\top})^{-1} \sum \tilde x_t \tilde y_t$; $\hat u_t = y_t - x_t^{\top}\hat\beta$; then the local-linear fit at each target month,

$$ (\hat a_t, \hat b_t) = \arg\min_{a,b} \sum_{\tau \le T_0} K\!\left(\frac{\tau - t}{h T_0}\right) \left[\hat u_\tau - a - b\,\frac{\tau - t}{T_0}\right]^2, \qquad \hat\ell_t := \hat a_t, $$

Gaussian $K$, bandwidth $h$ by leave-one-out cross-validation on the pre-period. The smoothing removes $\varepsilon_t$; $\ell_t$ and $\delta_t$ cannot be separated from each other after $T_0$, but both can be separated from the noise. *Covariate-free case* (decision of 4 September): calendar-month sine and cosine are functions of $t$ already inside $\ell_t$, so no covariate is carried, $\hat u_t = y_t$, and the outcome itself is smoothed.

**The pseudo-panel (eq 7).** With $m$ pre-periods and $n$ post-periods the latent sequence becomes an $(m + n) \times (N + 1)$ matrix: the first column is the treated block, pre $[T_0 - m + 1, T_0]$ and post $[T_0 + 1, T_0 + n]$; column $i + 1$ is historical block $i$, the same window shifted back by $z_i = n + i - 1$, $i = 1, \dots, N$, $N = T_0 - n - (m - 1)$:

$$ L = \begin{bmatrix} \ell_{\mathrm{post}} & L_{\mathrm{post}} \\ \ell_{\mathrm{pre}} & L_{\mathrm{pre}} \end{bmatrix}. $$

With $T_0 = 10$, $m = 4$, $n = 2$ (the slides' example): treated pre P07–P10, post P11–P12; block 1 pre P05–P08, post P09–P10; … block 5 pre P01–P04, post P05–P06. The blocks are contiguous windows, and their post rows all lie before $T_0$: they are estimated latent components, not effects.

**Stage 2: the weights (eqs 22–24).** $\hat w$ minimises $\lVert \hat\ell_{\mathrm{pre}} - \hat L_{\mathrm{pre}} w \rVert^2$ over $w \ge 0$, $\sum_j w_j = 1$. Where $\hat L_{\mathrm{pre}}^{\top}\hat L_{\mathrm{pre}}$ is singular the paper adds $\lVert \varsigma^{1/2} C_2^{\top} w \rVert^2$, $C_2$ the eigenvectors of the zero eigenvalues, $\varsigma$ small (eq 24).

**The counterfactual and the effect (eqs 11–13, 25–30; slides 11 and 14).** The weighted post rows of the historical blocks give the counterfactual latent component, $\hat\ell^{(0)}_{T_0+k} = \sum_j \hat w_j\, \hat\ell_{T_0 + k - z_j}$, and the effect is $\hat\delta_{T_0+k} = \hat u_{T_0+k} - \hat\ell^{(0)}_{T_0+k}$, without covariates $y_{T_0+k} - \hat\ell^{(0)}_{T_0+k}$. Proposition 1 bounds the bias by $b_\epsilon(H, k) = 2\epsilon |k|^{H+1}/(H+1)!$, growing with the horizon $k$, which is why a small $n$ is preferred (slide 8).

**Choosing m and the donor pool (§2.4, eqs 31–33).** $\mathrm{MSE}_{\mathrm{pre}}(w) = \frac{1}{m}\sum_{t = T_0 - m + 1}^{T_0} (\hat\ell_t - \hat\ell_t(w))^2$; rule (a) takes the largest $m$ with $\mathrm{MSE}_{\mathrm{pre}} \le \nu$, rule (b) the $m$ with the smallest validation error. Stepwise matching: the best single block, then the block whose addition lowers $\mathrm{MSE}_{\mathrm{pre}}$ most, accepted while the improvement exceeds $\zeta$. The paper sets $\nu$ = 0.1 % and $\zeta$ = 0.001 % of the outcome's sample variance.

---

## 3. How the fit maps onto it

**Summary.** The paper's estimator on the 980-value latent as a vector outcome, no covariates, stage 1 as specified; the only choices beyond the paper are the bandwidth grid (26 values, 0.03 to 10 in units of $T_0$) and a minimum of two finite neighbours per target month. Stage 1 is the acknowledged challenge of this design.

| paper | this fit |
|---|---|
| outcome, scalar $y_t$ | the 980-d latent; one weight vector per site × sensor over all 980 coordinates, stacked as $m \cdot 980$ rows of the design |
| $x_t$, $\beta$ | none (§2), $\hat u_t = y_t$ |
| stage 1 | local-linear, Gaussian kernel, regressor $(\tau - t)/T_0$, weights $K((\tau - t)/(h T_0))$, coordinate by coordinate along time; one $T_0 \times T_0$ operator per site; missing months weight 0; $h$ by leave-one-out CV pooled over the 980 coordinates on a log grid, one $h$ per site × sensor, curve and flag saved |
| $T_0$, post | $T_0$ = P10; post P11–P12 ($n = 2$) and P11–P13 ($n = 3$) |
| $m$ | 2 … $T_0 - n$, both rules reported |
| eq 24 ridge | applied only when the Gram is singular; never needed ($m \cdot 980$ rows against at most 7 columns) |
| donor pool | stepwise as in the paper, $\zeta$ = 0.001 % of the site's pre-period latent variance; single-best and all-in fits recorded alongside |
| counterfactual | the paper's raw forward rows, eq 27 (reported): the fitted weights applied to the observed forward rows of the selected blocks. $\hat\ell^{(0)}$ from the smoothed forward rows is also computed, but it compares a denoised prediction against a noisy observation and is not the evaluation reported here (correction of 17 September) |
| validation | temporal validation: the identical pipeline on P01–P(10 − n) predicting P(11 − n)–P10, scored against the observed latent, with the site's own P01–P(10 − n) mean as reference |

---

## 4. Stage 1 and the donor selection

**Summary.** The stepwise rule keeps the most recent block alone at 698 of 702 treated fits; the latent MSE rises with the block's age. Because the selected block's forward rows do not depend on $m$, the counterfactual and the prediction error are the same at every $m$; $n = 2$ has the smaller latent MSE and prediction error and is the primary setting.

- Grid: 2 fills × 2 sensors × $n \in \{2, 3\}$ × $m = 2 \dots 10 - n$ over 286 site × sensor series, 7,722 fits.
- Sentinel-1 latents are identical under both fills (no cloud mask), so Sentinel-1 is shown once.
- Blocks are numbered as in eq 7: block 1 the most recent (pre P05–P08, forward rows P09–P10 at $m = 4$, $n = 2$), block 5 the oldest (P01–P04, P05–P06). The slides number from the oldest: block 1 here is D5 there.

**Latent MSE (eq 31) of each block used alone**, the collaborator's "trial MSE" per candidate donor and "joint latent MSE" for the kept pool,, $m = 4$, $n = 2$, mean over the treated sites:

| sensor | fill | block 1 | block 2 | block 3 | block 4 | block 5 |
|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 0.0041 | 0.0089 | 0.0153 | 0.0233 | 0.0327 |
| Sentinel-2 | chip-mean fill | 0.0086 | 0.0185 | 0.0314 | 0.0469 | 0.0652 |
| Sentinel-2 | historical fill | 0.0075 | 0.0162 | 0.0277 | 0.0416 | 0.0579 |

Block 1 alone at 698 of 702 treated fits; a second block accepted in 4 of 594 attempts, never a third. Rule (a) never selects ($\nu \approx 10^{-4}$ against 0.004–0.008); rule (b) is flat in $m$. The tables below are at $m = 4$, the slides' baseline, and hold for every $m$.

**Block layout per $n$**, in the form of slide 18 (historical fill; block 1 is selected alone at every row; the chip-mean fill differs only in four Sentinel-2 fits that add a second block with a small weight). Block 1 is the treated pre-window shifted back by $n$: it lengthens with $m$ while its forward rows stay the same. *Latent MSE* = eq 31 of the selected block in the run with $T_0$ = P10 (at least one block at every row), mean over treated sites; it selects the blocks and is not a prediction error. *Prediction error* = the validation of §5 (the same pipeline on P01–P(10 − n), the synthetic counterfactual against the observed held-out months); that window has $10 - 2n - (m - 1)$ blocks, so windows with none ($m \ge 7$ at $n = 2$, $m \ge 5$ at $n = 3$) are left out.

$n = 2$ (post P11–P12):

| m | treated pre | blocks N | selected block (block 1), pre | selected block, forward rows | latent MSE, Sentinel-1 | latent MSE, Sentinel-2 | prediction error, Sentinel-1 | prediction error, Sentinel-2 |
|---|---|---|---|---|---|---|---|---|
| 2 | P09–P10 | 7 | P07–P08 | P09–P10 | 0.0043 | 0.0078 | 0.2666 | 0.2413 |
| 3 | P08–P10 | 6 | P06–P08 | P09–P10 | 0.0042 | 0.0076 | 0.2666 | 0.2413 |
| 4 | P07–P10 | 5 | P05–P08 | P09–P10 | 0.0041 | 0.0075 | 0.2666 | 0.2413 |
| 5 | P06–P10 | 4 | P04–P08 | P09–P10 | 0.0040 | 0.0074 | 0.2666 | 0.2413 |
| 6 | P05–P10 | 3 | P03–P08 | P09–P10 | 0.0039 | 0.0073 | 0.2666 | 0.2413 |

$n = 3$ (post P11–P13):

| m | treated pre | blocks N | selected block (block 1), pre | selected block, forward rows | latent MSE, Sentinel-1 | latent MSE, Sentinel-2 | prediction error, Sentinel-1 | prediction error, Sentinel-2 |
|---|---|---|---|---|---|---|---|---|
| 2 | P09–P10 | 6 | P06–P07 | P08–P10 | 0.0094 | 0.0167 | 0.3022 | 0.2971 |
| 3 | P08–P10 | 5 | P05–P07 | P08–P10 | 0.0091 | 0.0165 | 0.3022 | 0.2971 |
| 4 | P07–P10 | 4 | P04–P07 | P08–P10 | 0.0089 | 0.0162 | 0.3022 | 0.2971 |

The $n = 2$ rows coincide with the collaborator's slide 18 (D7 … D3, pseudo-post P09–P10, step-1 best in 27 of 27). At $n = 3$ the validation predicts P08–P10 from a block that ends at P07, one horizon further, and the error grows with the horizon as Proposition 1 says (slide 8).

---

## 5. Temporal validation: predicting P09–P10 from P01–P08

**Summary.** Under the corrected evaluation (the fitted weights applied to the observed forward rows, eq 27), the synthetic counterfactual never beats the site's own mean on Sentinel-1, and beats it on Sentinel-2 only at $n = 2$ (10 of 15 sites under chip-mean fill, 8 of 15 under historical fill); at $n = 3$ it loses on both sensors. The error grows with the horizon.

The whole pipeline on P01–P(10 − n), $m = 4$. *Prediction error* = RMSE over the 980 coordinates between a held-out month's observed latent and its predictor, mean over the $n$ months and the treated sites. Reported predictor: the fitted weights applied to the **observed (raw) forward rows** of the selected blocks, eq 27 — both sides of the comparison then carry observation noise. Reference: the site's own mean of P01–P(10 − n), with nothing estimated. The smoothed-forward-row column is retained for the record only; it compares a denoised prediction against a noisy observation, and the counts drawn from it in the original 10 September text (11 / 12 on Sentinel-1, 14–15 / 15 on Sentinel-2) are superseded by the counts below.

| sensor | fill | n | sites | prediction error, synthetic (raw forward rows, eq 27 — reported) | prediction error, own mean | sites where synthetic beats own mean | prediction error, smoothed forward rows (superseded) |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 2 | 12 | 0.2666 | 0.2360 | 0 / 12 | 0.2303 |
| Sentinel-1 | chip-mean fill | 3 | 12 | 0.3022 | 0.2452 | 0 / 12 | 0.2478 |
| Sentinel-2 | chip-mean fill | 2 | 15 | 0.2267 | 0.2415 | 10 / 15 | 0.2080 |
| Sentinel-2 | chip-mean fill | 3 | 15 | 0.2969 | 0.2540 | 0 / 15 | 0.2330 |
| Sentinel-2 | historical fill | 2 | 15 | 0.2413 | 0.2439 | 8 / 15 | 0.2173 |
| Sentinel-2 | historical fill | 3 | 15 | 0.2971 | 0.2525 | 0 / 15 | 0.2371 |

Per month, historical fill. These are **smoothed-forward-row** values, kept only for the horizon pattern: the per-month split was never stored for the raw forward rows, so it cannot be restated under the corrected evaluation without re-running the pipeline. The means in the last column are therefore the superseded column of the table above, not the reported one.

| sensor | n | P08 | P09 | P10 | mean (smoothed, superseded) |
|---|---|---|---|---|---|
| Sentinel-1 | 2 | | 0.2229 | 0.2377 | 0.2303 |
| Sentinel-1 | 3 | 0.2396 | 0.2442 | 0.2595 | 0.2478 |
| Sentinel-2 | 2 | | 0.2093 | 0.2252 | 0.2173 |
| Sentinel-2 | 3 | 0.2232 | 0.2394 | 0.2488 | 0.2371 |

P09 and P10 are predicted worse at $n = 3$, where the fit ends at P07 and those months sit at horizons two and three. Against the own mean, under the reported evaluation: on Sentinel-2 at $n = 2$ the synthetic counterfactual's error is 6.1 % below it under chip-mean fill and 1.1 % below it under historical fill, and at $n = 3$ it is 17–18 % above; on Sentinel-1 it is 13.0 % above at $n = 2$ and 23.2 % above at $n = 3$.

---

## 6. Language models as the weight solver on the band-feature designs

**Summary.** On the band-space pipeline's own SHC design and metric, the two language models that answer beat the band-space SHC on Sentinel-1 and roughly tie it on Sentinel-2; a three-month mean beats both, and the models state that this average is the rule they used. The SCM design is incomplete with these models. The reference that carries forward is the three-month mean, not persistence.

**Question.** Given the band-mean tables of the band-space pipeline, does a general language model predict a held-out pre-hurricane month as well as that pipeline's own two estimators?

- Task: predict P10 (August 2024) from information up to P09; 27 target cells (12 Sentinel-1, 15 Sentinel-2 sites); nothing treated.
- Outcome: the band means over the parcel (Sentinel-2 B2, B3, B4, B8, B11, B12, NDVI, NDWI; Sentinel-1 VV, VH, VV−VH) on the monthly composites of §1.
- SHC design: the target site's own P01–P09. SCM design: the target's P01–P09 and its five matched donor sites (control rank 1–5) at P01–P10.
- Prompt: a one-paragraph guide to the bands, the table, a one-line answer format; no word for the hurricane, the landslides or a treatment.
- Settings: temperature 0, fixed seed, output limit at the service ceiling of 8,000 tokens, one attempt per cell, a retry only on a rate-limit response.
- Models: gpt-oss-120b at default and low reasoning; Kimi-K3 at low reasoning (its default exceeds the service's 360-second gateway limit). GLM-5.3 and DeepSeek-V4-Flash spend the whole token budget on hidden reasoning and return no text in nearly every attempt; not reported.
- Metrics, the band-space pipeline's: SHC design, that pipeline's own joint validation NMSE — per band the squared error at P10 over the band's P01–P09 variance, mean over bands; SCM design, standardized validation RMSE at P10 (each band's error over the pooled P01–P09 standard deviation of the targets and donors). References on the same features and split: the band-space held-out SHC re-run (sine–cosine partialled out, two-month kernel, per-band standardization, four-month blocks, stepwise selection, simplex weights, prediction from the selected blocks' observed forward month; reproduces that pipeline's stored per-site values), the band-space SCM solver (reproduces its stored weights), and the site's own P01–P09 mean.

### 6.1 SHC design: all 27 cells answered by every reported model

Outcome metric, the band pipeline's joint validation NMSE (variance over P01–P09), per site and mean / median over sites, lower is better.

**On reading the `band-space SHC` column against that pipeline's own published numbers.** The per-site values here use that pipeline's own denominator, the variance over P01–P09, so they are directly comparable with its stored per-site values: the re-run matches all 27 to 4.2e-07 and the sensor means equal its stored 0.4925 and 0.7867. An earlier version of this table used the variance over P01–P10 and was not comparable with them; every figure in §6 has been recomputed.

| sensor | site | Kimi-K3 (low reasoning) | gpt-oss-120b (low reasoning) | gpt-oss-120b (default reasoning) | band-space SHC | own mean P01–P09 |
|---|---|---|---|---|---|---|
| Sentinel-1 | 0015 | 0.446 | 1.055 | 0.263 | 1.182 | 0.412 |
| Sentinel-1 | 0016 | 0.172 | 0.084 | 0.047 | 0.081 | 0.594 |
| Sentinel-1 | 0017 | 0.152 | 0.224 | 0.332 | 0.293 | 0.816 |
| Sentinel-1 | 0018 | 0.123 | 0.383 | 0.145 | 0.265 | 0.711 |
| Sentinel-1 | 0019 | 0.351 | 0.507 | 0.314 | 0.484 | 1.039 |
| Sentinel-1 | 0020 | 0.073 | 0.069 | 0.063 | 0.165 | 0.390 |
| Sentinel-1 | 0021 | 0.558 | 0.548 | 1.013 | 0.482 | 0.181 |
| Sentinel-1 | 0022 | 0.700 | 0.948 | 0.067 | 0.440 | 0.228 |
| Sentinel-1 | 0023 | 1.141 | 1.679 | 1.685 | 1.184 | 1.081 |
| Sentinel-1 | 0024 | 0.348 | 0.483 | 0.313 | 0.397 | 1.304 |
| Sentinel-1 | 0025 | 0.334 | 0.421 | 0.310 | 0.521 | 0.919 |
| Sentinel-1 | 0027 | — | 0.392 | 0.326 | 0.415 | 0.440 |
| Sentinel-2 | 0001 | 0.346 | 0.118 | 0.255 | 0.153 | 0.216 |
| Sentinel-2 | 0002 | 1.779 | 1.512 | 1.488 | 1.675 | 3.027 |
| Sentinel-2 | 0003 | 1.194 | 1.112 | 1.594 | 1.164 | 1.713 |
| Sentinel-2 | 0004 | 6.270 | 1.154 | 2.974 | 3.232 | 0.576 |
| Sentinel-2 | 0005 | 2.966 | 1.759 | 3.646 | 2.167 | 1.760 |
| Sentinel-2 | 0006 | 1.251 | 0.820 | 0.444 | 0.614 | 0.807 |
| Sentinel-2 | 0007 | 0.488 | 0.130 | 0.169 | 0.184 | 0.790 |
| Sentinel-2 | 0008 | 0.429 | 0.201 | 0.267 | 0.240 | 1.175 |
| Sentinel-2 | 0009 | 0.581 | 0.198 | 0.270 | 0.207 | 1.069 |
| Sentinel-2 | 0010 | 1.338 | 0.717 | 2.018 | 0.899 | 1.276 |
| Sentinel-2 | 0011 | 0.178 | 0.124 | 0.166 | 0.108 | 0.559 |
| Sentinel-2 | 0012 | 1.719 | 0.622 | 0.394 | 0.917 | 1.094 |
| Sentinel-2 | 0013 | 0.192 | 0.116 | 0.174 | 0.154 | 0.800 |
| Sentinel-2 | 0014 | 0.071 | 0.056 | 0.054 | 0.036 | 0.562 |
| Sentinel-2 | 0015 | 0.048 | 0.132 | 0.255 | 0.051 | 1.321 |
| **Sentinel-1, mean / median** | | 0.400 / 0.348 | 0.566 / 0.452 | 0.407 / 0.311 | 0.492 / 0.427 | 0.676 / 0.653 |
| **Sentinel-2, mean / median** | | 1.257 / 0.581 | 0.585 / 0.201 | 0.945 / 0.270 | 0.787 / 0.240 | 1.116 / 1.069 |
| **sites where the model beats the band-space SHC** | | S1 8 / 11, S2 1 / 15 | S1 5 / 12, S2 11 / 15 | S1 9 / 12, S2 5 / 15 | | |

- Sentinel-1: gpt-oss-120b at default reasoning beats the band-space SHC at 9 of 12 sites and Kimi-K3 at 8 of 11; gpt-oss-120b at low reasoning at 5 of 12. All three beat the own mean on mean and median.
- Sentinel-2: the band-space SHC has the lowest mean and median; the model means are set by sites 0004, 0005 and 0010, where the band-space SHC is itself at 3.23, 2.17 and 0.90; on the medians gpt-oss-120b sits below it at either setting and Kimi-K3 above.
- The band-space SHC's forecast is P09 carried forward at 26 of 27 cells (block 1 alone at weight 1; site 0014 adds a second block at 0.10). What the models add on Sentinel-1 is a trend or seasonal reading of nine smooth values; on Sentinel-2, where reflectance jumps with residual cloud, carrying P09 forward is as good.
- gpt-oss-120b at low and default reasoning are close (low better at 16 of 27 cells).

### 6.2 SCM design: incomplete, reported as it stands

The SCM prompt is three to four times longer (six parcels, ten months) and the models reason longer on it. Answered cells, with the two failure modes (hidden reasoning fills the 8,000-token ceiling; the gateway closes the request at 360 s):

| model | Sentinel-1 answered | Sentinel-2 answered | output-ceiling failures | gateway timeouts |
|---|---|---|---|---|
| Kimi-K3 (low reasoning) | 9 / 12 | 10 / 15 | 4 | 4 |
| gpt-oss-120b (low reasoning) | 12 / 12 | 15 / 15 | 0 | 0 |
| gpt-oss-120b (default reasoning) | 11 / 12 | 2 / 15 | 14 | 0 |

Only gpt-oss-120b at low reasoning answers every cell; each other row is compared with the band-space SCM on the same cells (on all cells the band-space SCM is 0.3631 / 0.3149 on Sentinel-1 and 0.3416 / 0.3153 on Sentinel-2).

| sensor | model | cells | model, mean / median | band-space SCM on the same cells | cells where the model is better |
|---|---|---|---|---|---|
| Sentinel-1 | Kimi-K3 (low reasoning) | 9 | 0.2309 / 0.2192 | 0.4195 / 0.4049 | 6 of 9 |
| Sentinel-1 | gpt-oss-120b (low reasoning) | 12 | 0.3607 / 0.2519 | 0.3631 / 0.3149 | 4 of 12 |
| Sentinel-1 | gpt-oss-120b (default reasoning) | 11 | 0.3056 / 0.2393 | 0.3685 / 0.3257 | 5 of 11 |
| Sentinel-2 | Kimi-K3 (low reasoning) | 10 | 0.3532 / 0.3013 | 0.3195 / 0.2638 | 4 of 10 |
| Sentinel-2 | gpt-oss-120b (low reasoning) | 15 | 0.5720 / 0.5978 | 0.3416 / 0.3153 | 5 of 15 |
| Sentinel-2 | gpt-oss-120b (default reasoning) | 2 | 0.4417 / 0.4417 | 0.3110 / 0.3110 | 0 of 2 |

- With every cell answered, gpt-oss-120b ties the band-space SCM on Sentinel-1 (same mean, better at 4 of 12 sites) and is clearly worse on Sentinel-2.
- The Kimi-K3 advantage on Sentinel-1 rests on nine cells where the band-space SCM itself does badly (0.42 there against 0.36 on all twelve).

### 6.3 What the models do: plain rules, a diagnosis of the predictions, prompt ablations

Three checks on the same 27 cells and metric (variance over P01–P09).

**Plain forecasting rules** of P10 from the target's own P01–P09, per band: P09 carried forward; the mean of the last three months; the own mean; a least-squares line through nine months, and through the last four, extrapolated one step; damped trend exponential smoothing (Gardner & McKenzie 1985; parameters chosen on P01–P09); the mean of the two models' answers.

| method | Sentinel-1, mean / median | below the band-space SHC | Sentinel-2, mean / median | below the band-space SHC |
|---|---|---|---|---|
| mean of the last three months | 0.265 / 0.196 | 11 / 12 | 0.541 / 0.350 | 7 / 15 |
| damped trend exponential smoothing | 0.279 / 0.192 | 10 / 12 | 0.686 / 0.408 | 10 / 15 |
| linear trend on nine months | 0.362 / 0.230 | 9 / 12 | 1.141 / 0.927 | 2 / 15 |
| Kimi-K3 (low reasoning) | 0.400 / 0.348 | 8 / 11 | 1.257 / 0.581 | 1 / 15 |
| gpt-oss-120b (default reasoning) | 0.407 / 0.311 | 9 / 12 | 0.945 / 0.270 | 5 / 15 |
| mean of the two models | 0.482 / 0.412 | 6 / 11 | 0.850 / 0.347 | 5 / 15 |
| band-space SHC (= P09 carried forward) | 0.492 / 0.427 |  | 0.787 / 0.240 |  |
| linear trend on the last four months | 0.560 / 0.449 | 4 / 12 | 1.195 / 1.321 | 4 / 15 |
| gpt-oss-120b (low reasoning) | 0.566 / 0.452 | 5 / 12 | 0.585 / 0.201 | 11 / 15 |
| own mean P01–P09 | 0.676 / 0.653 | 4 / 12 | 1.116 / 1.069 | 2 / 15 |

**Which rule the models track.** Each prediction as its move from P09 in units of the band's P01–P09 standard deviation, regressed on the move of the nine-month linear trend (slope 1 = the trend, 0 = P09 carried forward, negative = against the trend); "direction right" = share of cells × bands where the move has the sign of the observed change; "nearest rule" = the plain rule the model's value is closest to.

| sensor | model | slope on the linear-trend move | R² | direction right: model | direction right: damped trend exponential smoothing | direction right: own mean | nearest rule (cells × bands) |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | Kimi-K3 (low) | 0.47 | 0.27 | 75 % | 81 % | 58 % | mean of the last three months 16, P09 carried forward 6 |
| Sentinel-1 | gpt-oss-120b (low) | 0.52 | 0.29 | 69 % | 81 % | 58 % | mean of the last three months 14, linear trend on nine months 7 |
| Sentinel-1 | gpt-oss-120b (default) | 0.69 | 0.40 | 69 % | 81 % | 58 % | mean of the last three months 14, P09 carried forward 9 |
| Sentinel-2 | Kimi-K3 (low) | -0.61 | 0.56 | 42 % | 68 % | 62 % | P09 carried forward 54, damped trend exponential smoothing 27 |
| Sentinel-2 | gpt-oss-120b (low) | -0.12 | 0.10 | 55 % | 68 % | 62 % | P09 carried forward 43, damped trend exponential smoothing 30 |
| Sentinel-2 | gpt-oss-120b (default) | -0.09 | 0.02 | 50 % | 68 % | 62 % | P09 carried forward 40, mean of the last three months 32 |

At site 0004 the P09 composite is a bright outlier in the visible and shortwave bands (B2 0.045 → 0.086, residual cloud) and P10 returns to 0.054; P09 carried forward keeps the outlier, both models push it further (B2 0.100 and 0.090), every averaging rule reverts. The models read a cloud jump as a trend.

**Prompt ablations** (27 cells per arm, one deterministic call each; one Kimi-K3 cell of the anonymous arm unanswered after a gateway timeout):

| prompt | Kimi-K3, Sentinel-1 | Kimi-K3, Sentinel-2 | gpt-oss-120b, Sentinel-1 | gpt-oss-120b, Sentinel-2 |
|---|---|---|---|---|
| prompt of §6.1 | 0.380 / 0.293 | 1.309 / 0.396 | 0.457 / 0.336 | 0.781 / 0.364 |
| no band guide | 0.426 / 0.346 | 1.044 / 0.383 | 0.353 / 0.259 | 0.810 / 0.365 |
| anonymous quantities x1…xk, no sensor, no guide | 0.389 / 0.348 | 0.697 / 0.263 | 0.585 / 0.335 | 0.696 / 0.298 |
| no calendar dates | 0.359 / 0.295 | 1.189 / 0.678 | 0.594 / 0.479 | 0.951 / 0.334 |
| prompt of §6.1 plus a second line naming the rule used | 0.388 / 0.266 | 1.221 / 0.373 | 0.393 / 0.315 | 0.864 / 0.364 |
| mean of the last three months, for reference | 0.283 / 0.210 | 0.503 / 0.364 | | |

**Rules the models state for themselves** (second line of the reply in the last arm). Sentinel-1: Kimi-K3 names an average of the last three months at 9 of 12 sites, a trend at 3; gpt-oss-120b an average at 9, a trend at 2, P09 carried forward at 1. Sentinel-2: Kimi-K3 a seasonal story at 14 of 15 (a post-peak decline extrapolated at 9, the summer plateau persisted at 5), a trend at 1; gpt-oss-120b a trend at 10, an average at 5.

- The Sentinel-1 advantage of §6.1 is smoothing over persistence: the mean of the last three months is below the band-space SHC at 11 of 12 sites and below every model, and the models say that average is what they computed.
- On Sentinel-2 the models extrapolate a seasonal decline, right on the median site and wrong wherever P09 carries residual cloud; Kimi-K3 moves against the linear trend (slope −0.61) and has the direction right at 42 % of cells × bands, against 68 % for damped trend exponential smoothing.
- Removing every domain word improves both models on Sentinel-2 (Kimi-K3's mean 1.31 → 0.70), so the band guide and the sensor name trigger the seasonal story; removing the calendar dates hurts, so the season is used and useful when right.
- Across the five wordings one model's Sentinel-1 mean ranges from 0.35 to 0.59, the whole spread between models; single deterministic calls on 12 sites cannot rank models. No prompt brings either model to the three-month mean.

**Decision.** The SHC design carries forward (short prompts, every reported model answers every cell); the SCM design is left as it stands. The models contribute a hand-written rule, a three-month average on Sentinel-1 and a seasonal extrapolation on Sentinel-2, which is better run directly. The reference that carries forward is the mean of the last three months: it beats P09 carried forward, which is what the band-space SHC reduces to here, at 11 of 12 Sentinel-1 and 7 of 15 Sentinel-2 sites. On Sentinel-2 a one-month departure is inside the size of a residual-cloud jump for every method; an effect read from band means there needs a cloud screen or a longer post-period window.

---

## 7. Reproducibility

Every number in §1 and §4–§6 is produced by a script and read from its output table. The panel index and the encoder are in `Satellite/notebooks/ts_SCM_ASCM/`, the synthetic historical control in `Satellite/notebooks/shc/` (moved there on 11 September; it imports the panel modules from the first folder), and §6 in `Satellite/notebooks/llm_band_sc/`. The 3 September SHC results rested on a design that departed from the method (last pre-period as the only target, no post-treatment period in any block, weights matched on the raw outcome, all blocks as donors); their scripts and tables are removed; the shared grid and comparison tables regenerated without them are unchanged (maximum difference 0).

| script / artifact | produces | read by |
|---|---|---|
| `panel_monthly.py` | `panel_monthly_index.csv` (site × sensor × month index, usability) | §1 |
| `panel_monthly_encode.py` | `data/embeddings_tok_panel/latents_monthly_long.npz`, `latents_monthly_long_histfill.npz` (+ `.fillcounts.csv`, manifests) | §1 |
| `panel_shc.py`, `panel_shc_tests.py` | the estimator and its consistency tests (the block construction reproduces the slides' $T_0 = 10$, $m = 4$, $n = 2$ example and the $z_i = n + i - 1$ rule; the smoother returns the series at $h \to 0$, the least-squares line at $h \to \infty$ and any linear series unchanged; a planted convex combination of two blocks is recovered by the stepwise rule) | §3 |
| `panel_shc_run.py` | `panel_shc_fits.csv`, `panel_shc_effects.csv`, `panel_shc_bandwidth.csv` (CV curves), `panel_shc_path.csv` (selection paths and every candidate), `panel_shc_summary.csv` | §4–§5 |
| `22_shc_monthly.ipynb` | tables and the bandwidth-curve figure | §4 |
| `Docs/Synthetic Historical Control_Sep 4.pptx`, meeting transcript of 4 September 2026 | the method walkthrough and the design decisions | §2–§3 |
| `lb_features.py`, `lb_prompts.py` | `lb_features_monthly.csv` (band means P01–P10, targets and donors), `lb_prompts_shc.jsonl`, `lb_prompts_scm.jsonl` (full prompt text and withheld truth) | §6 |
| `lb_run.py` | `lb_replies_<design>_<model>_<effort>.jsonl` (every reply, token usage, finish reason, errors) | §6 |
| `lb_shc.py`, `lb_lib.py`, `lb_tests.py` | the band-space SHC re-run, the band-space scaler and solver, the reproduction checks (the 27 stored SHC values to 4 × 10⁻⁷, the stored biweekly weights to 6 × 10⁻⁸, the band means to 5 × 10⁻⁷) | §6 |
| `lb_baselines.py`, `lb_prompts_ablation.py` | the plain rules (scored inside `lb_scores_shc.csv`), `lb_diag_models.csv`, `lb_diag_summary.csv`, `lb_diag_s2_sites.csv`; `lb_prompts_abl_*.jsonl` and `lb_replies_abl_*_<model>_low.jsonl` (the four ablation arms) | §6.3 |
| `lb_score.py`, `41_llm_band_sc.ipynb` | `lb_scores_shc.csv`, `lb_summary_shc.csv`, `lb_summary_shc_band.csv`, `lb_scores_scm.csv`, `lb_summary_scm.csv`; the notebook prints one full prompt per design and sensor | §6 |
