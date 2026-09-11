# Synthetic historical control on the monthly latent panel: method, implementation, first results

**Progress report for the meeting of 10 September 2026.** The synthetic-historical-control results of the 3 September report were withdrawn on 4 September and are not carried forward. Sources for the method: Chen, Yang & Yang, *Synthetic Historical Control for Policy Evaluation*, SSRN 4995085; the collaborator's slides *Synthetic Historical Control for Policy Evaluation: Evidence from Satellite Data* (4 September 2026), whose walkthrough §2 follows; and the decisions of the 4 September meeting. Scripts, caches and tables are listed in §8.

**Summary.** The estimator is rebuilt from the paper as settled on 4 September: the treatment period is P11 (September 2024) and the pre-period P01–P10; the latent component is estimated first by local-linear kernel smoothing of the outcome along time; the weights are fitted on the smoothed pre-segments of the site's own shifted blocks, selected stepwise; the counterfactual is the weighted forward rows of the smoothed component and the effect is the observed post-period latent minus it. The outcome is the 980-dimensional TerraMind tokenizer latent of the monthly median composites, under two cloud fills. Four findings. (i) At every treated site, both sensors, both fills and every window length, the stepwise rule keeps the most recent block alone (698 of 702 fits), so the counterfactual is the smoothed latent of the last pre-hurricane months whatever $m$ is. (ii) In the temporal validation (fit on P01–P08, predict P09–P10) the smoothed counterfactual beats the site's own mean on Sentinel-2 (15 of 15 sites) and narrowly on Sentinel-1 (11 of 12). (iii) In the effect run the treated sites' post-period departure equals that of their matched control sites and the validation error: no Helene effect is detectable in the latent under this design. Putting the covariate stage back as a mapping of land cover, elevation and slope learned across sites (§6.1) changes nothing on Sentinel-1 and lowers the Sentinel-2 validation error by 3–5 % without separating the treated sites from their controls. (iv) In a companion experiment on the band-space pipeline's own designs and metrics (§7), language models used as the weight solver beat the band-space SHC on Sentinel-1 and roughly tie it on Sentinel-2; a three-month mean beats both, and the models state that this average is the rule they applied. The reference that carries forward is the three-month mean, not persistence. The leave-one-out bandwidth is large at every site (kernel scale five to seven months on a ten-month window); that stage is the first to rethink.

---

## 1. Data: the monthly composites

Calendar-month median composites of the daily Sentinel-1 and Sentinel-2 acquisitions, 21 months P01 (November 2023) to P21 (July 2025), 101 × 101 pixels at 10 m, the same bands and channel-first layout as the biweekly set. P01–P10 (November 2023 to August 2024) are pre-hurricane; **P11, September 2024, is the treatment period** (landfall 27 September); P12 onward are post. The collaborator's band pipeline excludes P11 as a transition month; this report keeps it as the first treated period, with one caveat carried in the tables: the Sentinel-2 September composite is built from the 2, 7 and 22 September acquisitions, so it contains no post-landfall image, while the Sentinel-1 composite includes 28 September.

| | Sentinel-1 | Sentinel-2 |
|---|---|---|
| treated sites | 12 (sites 0015–0027) | 15 (sites 0001–0015) |
| matched counterfactual sites (10 per treated site) | 120 | 150 |
| months | 21 | 21 |

Of the 6,237 site × sensor × month rows, 6,055 composites are usable; 165 months have no acquisition (all in P01) and 17 Sentinel-2 counterfactual composites are entirely cloud-masked and are treated as missing. The two sensors cover different treated sites; only site 0015 is in both.

**Cloud fill.** TerraMind cannot encode a masked pixel, so every latent is post-fill, and both fills are run as arms. *Chip-mean fill*: each masked pixel takes the chip's own band mean. *Historical fill*: each masked pixel takes the per-pixel median of the site's own clean pre-hurricane months P01–P10 (a pixel counts when every band is finite, at least three months), then the one-month template, then the chip mean. At the Sentinel-2 treated sites 98.1% of masked pixels are filled from the ten-month median. Masked-pixel share by month at the Sentinel-2 treated sites:

| P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | P17 | P18 | P19 | P20 | P21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.08 | 0.23 | 0.18 | 0.17 | 0.02 | 0.00 | 0.00 | 0.53 | 0.27 | 0.06 | 0.00 | 0.00 | 0.17 | 0.44 | 0.00 | 0.00 |

**Outcome.** The TerraMind tokenizer latent of each composite, 5 channels × 14 × 14 parcels = 980 FSQ codes on [0, 1], encoded as the biweekly panel (101 → 224 bilinear resize, TerraMind v1 standardization). The five channels share units, so no cross-coordinate standardization is applied. The same site's parcels are registered across months, which is the property that suits SHC to imagery: a coordinate-wise convex combination of the site's own past is meaningful where a cross-site one is not.

---

## 2. The method, in the source's order

Presented as in the collaborator's slides, with the paper's equation numbers.

**Model and the identification problem (eq 2).** The outcome of one unit follows

$$ y_t = x_t^{\top}\beta + \ell_t + \delta_t d_t + \varepsilon_t, \qquad d_t = \mathbb{1}[t > T_0], $$

with $x_t$ observed covariates, $\ell_t$ an unobserved smooth function of time (the latent component, what would have happened without the intervention), $\delta_t$ the time-varying intervention effect and $\varepsilon_t$ noise. Before $T_0$ the residual $y_t - x_t^{\top}\beta$ equals $\ell_t + \varepsilon_t$ and $\ell_t$ is learnable; after $T_0$ it equals $\ell_t + \delta_t + \varepsilon_t$ and the two are not separable from that series alone. SHC reconstructs the post-period $\ell_t$ from the unit's own history.

**Stage 1: the latent component (eqs 18–21; slide 6).** With covariates: local-linear kernel estimates $\hat{\mu}_{y,t} \approx E(y_t \mid t)$ and $\hat{\mu}_{x,t} \approx E(x_t \mid t)$; residualize, $\tilde y_t = y_t - \hat{\mu}_{y,t}$, $\tilde x_t = x_t - \hat{\mu}_{x,t}$; $\hat\beta = (\sum \tilde x_t \tilde x_t^{\top})^{-1} \sum \tilde x_t \tilde y_t$; $\hat u_t = y_t - x_t^{\top}\hat\beta$; then the local-linear fit at each target month,

$$ (\hat a_t, \hat b_t) = \arg\min_{a,b} \sum_{\tau \le T_0} K\!\left(\frac{\tau - t}{h T_0}\right) \left[\hat u_\tau - a - b\,\frac{\tau - t}{T_0}\right]^2, \qquad \hat\ell_t := \hat a_t, $$

Gaussian $K$, bandwidth $h$ by leave-one-out cross-validation, pre-period only. The smoothing removes $\varepsilon_t$: $\ell_t$ and $\delta_t$ cannot be separated from each other after $T_0$, but both can be separated from the noise. **Covariate-free case.** By the decision of 4 September the first run carries no covariates: calendar-month sine and cosine are deterministic functions of $t$, already inside $\ell_t$, and residualizing them against time leaves nothing to identify $\beta$. Then $\hat u_t = y_t$ and the outcome itself is smoothed.

**The pseudo-panel (eq 7).** With $m$ pre-periods and $n$ post-periods, the latent sequence is rearranged into an $(m + n) \times (N + 1)$ matrix whose first column is the treated block, pre $[T_0 - m + 1, T_0]$ and post $[T_0 + 1, T_0 + n]$, and whose column $i + 1$ is historical block $i$, the same window shifted back by $z_i = n + i - 1$, $i = 1, \dots, N$, $N = T_0 - n - (m - 1)$:

$$ L = \begin{bmatrix} \ell_{\mathrm{post}} & L_{\mathrm{post}} \\ \ell_{\mathrm{pre}} & L_{\mathrm{pre}} \end{bmatrix}. $$

With $T_0 = 10$, $m = 4$, $n = 2$ (the slides' example): treated pre P07–P10, post P11–P12; block 1 pre P05–P08, post P09–P10; block 2 pre P04–P07, post P08–P09; … block 5 pre P01–P04, post P05–P06. The historical blocks are contiguous windows, so the temporal dependence inside each is kept; their post rows all lie before $T_0$ and are therefore estimated latent components, not effects.

**Stage 2: the weights (eqs 22–24).** $\hat w$ minimises $\lVert \hat\ell_{\mathrm{pre}} - \hat L_{\mathrm{pre}} w \rVert^2$ over $w \ge 0$, $\sum_j w_j = 1$. Where $\hat L_{\mathrm{pre}}^{\top}\hat L_{\mathrm{pre}}$ is singular ($N > m$ in the scalar case) the paper replaces it by its nearest positive-definite matrix, adding $\lVert \varsigma^{1/2} C_2^{\top} w \rVert^2$ with $C_2$ the eigenvectors of the zero eigenvalues and $\varsigma$ small (eq 24).

**The counterfactual and the effect (eqs 11–13, 25–30; slides 11 and 14).** Slide 11's flow ends with "apply $\hat w$ to the historical post segments" of the pseudo-panel $\hat L$ built from $\hat\ell_t$, and slide 14 fills the post rows of $L$ with the latent components $\ell$; the weighted post rows of the historical blocks therefore give the counterfactual latent component, $\hat\ell^{(0)}_{T_0+k}(\hat w) = \sum_j \hat w_j\, \hat\ell_{T_0 + k - z_j}$, and the effect is the observed residual minus it, $\hat\delta_{T_0+k} = \hat u_{T_0+k} - \hat\ell^{(0)}_{T_0+k}$; without covariates $\hat\delta_{T_0+k} = y_{T_0+k} - \hat\ell^{(0)}_{T_0+k}$. Proposition 1 bounds the bias by $b_\epsilon(H, k) = 2\epsilon |k|^{H+1}/(H+1)!$, growing with the horizon $k$, which is why a small $n$ is preferred (slide 8).

**Choosing m and the donor pool (§2.4, eqs 31–33).** Matching quality is $\mathrm{MSE}_{\mathrm{pre}}(w) = \frac{1}{m}\sum_{t = T_0 - m + 1}^{T_0} (\hat\ell_t - \hat\ell_t(w))^2$; rule (a) takes the largest $m$ with $\mathrm{MSE}_{\mathrm{pre}} \le \nu$, rule (b) the $m$ with the smallest validation error. The donor pool is built stepwise: the best single block; then, among the remaining blocks, the one whose addition lowers $\mathrm{MSE}_{\mathrm{pre}}$ most, accepted while the improvement exceeds $\zeta$. The paper's applications set $\nu$ to 0.1 % and $\zeta$ to 0.001 % of the outcome's sample variance.

---

## 3. How the fit maps onto it

| paper | this fit |
|---|---|
| outcome, scalar $y_t$ | the 980-d latent; one weight vector per site × sensor over all 980 coordinates, stacked as $m \cdot 980$ rows of the design |
| $x_t$, $\beta$ | none (§2), $\hat u_t = y_t$; a learned mapping is added in §6.1 |
| stage 1 | local-linear, Gaussian kernel, regressor $(\tau - t)/T_0$, weights $K((\tau - t)/(h T_0))$, coordinate by coordinate along time; one $T_0 \times T_0$ operator per site; missing months weight 0; $h$ by leave-one-out CV pooled over the 980 coordinates on a log grid, one $h$ per site × sensor, curve and flag saved |
| $T_0$, post | $T_0$ = P10; post P11–P12 ($n = 2$) and P11–P13 ($n = 3$) |
| $m$ | 2 … $T_0 - n$, both rules reported |
| eq 24 ridge | applied only when the Gram is singular; never needed ($m \cdot 980$ rows against at most 7 columns) |
| donor pool | stepwise as in the paper, $\zeta$ = 0.001 % of the site's pre-period latent variance; single-best and all-in fits recorded alongside |
| counterfactual | $\hat\ell^{(0)}$ from the smoothed forward rows (primary); the paper's raw forward rows (eq 27) as a secondary column |
| validation | temporal validation: the identical pipeline on P01–P(10 − n) predicting P(11 − n)–P10, scored against the observed latent, with the site's own P01–P(10 − n) mean as reference; negative controls: the identical effect run on the 10 matched counterfactual sites of each treated site, and the rank of the treated site's post-period departure among them |
| standardization | none: the five FSQ channels share units on [0, 1] |

**Stage 1 is the acknowledged challenge of this design.** It is implemented as specified above, with no departure; the one choice beyond the paper is the bandwidth grid, 26 values from 0.03 to 10 in units of $T_0$, and the requirement of at least two finite neighbours per target month.

---

## 4. Stage 1 and the donor selection

The grid is 2 fills × 2 sensors × $n \in \{2, 3\}$ × $m = 2 \dots 10 - n$ over the 286 site × sensor series (7,722 fits). Sentinel-1 latents are identical under both fills (the radar chips have no cloud mask), so Sentinel-1 is shown once in every table.

**Bandwidths.** Leave-one-out CV on the effect window P01–P10, treated sites:

| sensor | fill | sites | h median | h min | h max | interior | at grid floor | at grid ceiling |
|---|---|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 12 | 0.71 | 0.50 | 2.00 | 12 | 0 | 0 |
| Sentinel-2 | chip-mean fill | 15 | 0.50 | 0.16 | 1.26 | 15 | 0 | 0 |
| Sentinel-2 | historical fill | 15 | 0.50 | 0.16 | 1.00 | 15 | 0 | 0 |

Units are $T_0$ = 10 months, so a bandwidth of 0.5–0.7 is a kernel scale of five to seven months on a ten-month window: the smoothed component is close to a line through the pre-period.

**Which block is chosen.** Step-1 matching error of each single block, $m = 4$, $n = 2$, mean over the treated sites (block 1 is the most recent, shifted by $n$; block 5 the oldest):

| sensor | fill | block 1 | block 2 | block 3 | block 4 | block 5 |
|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 0.0041 | 0.0089 | 0.0153 | 0.0233 | 0.0327 |
| Sentinel-2 | chip-mean fill | 0.0086 | 0.0185 | 0.0314 | 0.0469 | 0.0652 |
| Sentinel-2 | historical fill | 0.0075 | 0.0162 | 0.0277 | 0.0416 | 0.0579 |

The error rises monotonically with the block's age. The stepwise rule keeps block 1 alone at 698 of 702 treated fits; a second block is accepted in 4 of 594 attempts and never a third. The same holds at every control site. Because block 1's forward rows are P09–P10 for $n = 2$ (P08–P10 for $n = 3$) whatever $m$ is, the counterfactual, the validation error and the effect are identical across $m$; $m$ only changes the pre-period matching error, which falls slowly with $m$. Rule (a) never selects: $\nu$ is about $10^{-4}$ against matching errors of 0.004 (Sentinel-1) and 0.008 (Sentinel-2). Rule (b) is flat in $m$ for the same reason. The tables below are therefore shown at $m = 4$, the slides' baseline, and hold for every $m$.

---

## 5. Temporal validation: predicting P09–P10 from P01–P08

The whole pipeline (smoothing, selection, weights) on P01–P08 only, $m = 4$; RMSE over the 980 coordinates between the observed latent and the counterfactual, mean over the treated sites. *Reference* = the site's own P01–P08 mean, with nothing estimated. The second SHC column applies the same weights to the raw forward rows (the paper's eq 27).

| sensor | fill | n | sites | SHC (smoothed forward rows) | SHC (raw forward rows) | reference: own mean | sites SHC < reference |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 2 | 12 | 0.2303 | 0.2666 | 0.2360 | 11 / 12 |
| Sentinel-1 | chip-mean fill | 3 | 12 | 0.2478 | 0.3022 | 0.2452 | 2 / 12 |
| Sentinel-2 | chip-mean fill | 2 | 15 | 0.2080 | 0.2267 | 0.2415 | 15 / 15 |
| Sentinel-2 | chip-mean fill | 3 | 15 | 0.2330 | 0.2969 | 0.2540 | 14 / 15 |
| Sentinel-2 | historical fill | 2 | 15 | 0.2173 | 0.2413 | 0.2439 | 14 / 15 |
| Sentinel-2 | historical fill | 3 | 15 | 0.2371 | 0.2971 | 0.2525 | 10 / 15 |

On Sentinel-2 the smoothed counterfactual is 6–14 % below the own-history mean at most sites; on Sentinel-1 it is 2 % below at $n = 2$ and above it at $n = 3$. The raw forward rows are worse than the smoothed ones everywhere, which is the measured value of stage 1 on this panel.

---

## 6. Effect run: post-period departure of the treated sites against their controls

$T_0$ = P10, the same pipeline; $\lVert \hat\delta_{T_0+k} \rVert / \sqrt{980}$ per post month, mean over sites. *Controls* are the matched counterfactual sites run identically. $m = 4$.

| sensor | fill | n | post month | treated | controls |
|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 2 | P11 | 0.2086 | 0.2134 |
| Sentinel-1 | chip-mean fill | 2 | P12 | 0.2648 | 0.2638 |
| Sentinel-1 | chip-mean fill | 3 | P11 | 0.2037 | 0.2080 |
| Sentinel-1 | chip-mean fill | 3 | P12 | 0.2528 | 0.2528 |
| Sentinel-1 | chip-mean fill | 3 | P13 | 0.2644 | 0.2582 |
| Sentinel-2 | chip-mean fill | 2 | P11 | 0.2067 | 0.1987 |
| Sentinel-2 | chip-mean fill | 2 | P12 | 0.2581 | 0.2499 |
| Sentinel-2 | chip-mean fill | 3 | P11 | 0.2064 | 0.2002 |
| Sentinel-2 | chip-mean fill | 3 | P12 | 0.2399 | 0.2326 |
| Sentinel-2 | chip-mean fill | 3 | P13 | 0.3084 | 0.3254 |
| Sentinel-2 | historical fill | 2 | P11 | 0.2040 | 0.1979 |
| Sentinel-2 | historical fill | 2 | P12 | 0.2453 | 0.2468 |
| Sentinel-2 | historical fill | 3 | P11 | 0.2057 | 0.1997 |
| Sentinel-2 | historical fill | 3 | P12 | 0.2293 | 0.2292 |
| Sentinel-2 | historical fill | 3 | P13 | 0.3020 | 0.3080 |

Per treated site, $n = 2$, historical fill: the departure at P11 and P12, the median departure of its ten controls, the site's rank among the eleven (1 = largest), and its validation error against the own-mean reference.

| sensor | site | h | P11 | P12 | controls, median | rank | validation SHC | validation reference |
|---|---|---|---|---|---|---|---|---|
| Sentinel-1 | 0015 | 0.79 | 0.246 | 0.289 | 0.245 | 3 / 11 | 0.253 | 0.256 |
| Sentinel-1 | 0016 | 0.50 | 0.211 | 0.278 | 0.246 | 6 / 11 | 0.251 | 0.249 |
| Sentinel-1 | 0017 | 0.63 | 0.200 | 0.253 | 0.243 | 11 / 11 | 0.224 | 0.233 |
| Sentinel-1 | 0018 | 0.79 | 0.194 | 0.255 | 0.227 | 9 / 11 | 0.214 | 0.229 |
| Sentinel-1 | 0019 | 0.79 | 0.206 | 0.266 | 0.245 | 10 / 11 | 0.231 | 0.236 |
| Sentinel-1 | 0020 | 0.63 | 0.213 | 0.269 | 0.239 | 5 / 11 | 0.235 | 0.240 |
| Sentinel-1 | 0021 | 0.63 | 0.214 | 0.275 | 0.245 | 7 / 11 | 0.237 | 0.237 |
| Sentinel-1 | 0022 | 0.63 | 0.221 | 0.278 | 0.228 | 3 / 11 | 0.237 | 0.241 |
| Sentinel-1 | 0023 | 2.00 | 0.195 | 0.237 | 0.197 | 2 / 11 | 0.221 | 0.231 |
| Sentinel-1 | 0024 | 1.00 | 0.223 | 0.275 | 0.263 | 9 / 11 | 0.233 | 0.243 |
| Sentinel-1 | 0025 | 1.00 | 0.208 | 0.265 | 0.252 | 10 / 11 | 0.225 | 0.231 |
| Sentinel-1 | 0027 | 0.50 | 0.172 | 0.238 | 0.217 | 9 / 11 | 0.203 | 0.206 |
| Sentinel-2 | 0001 | 0.50 | 0.227 | 0.254 | 0.226 | 2 / 11 | 0.226 | 0.241 |
| Sentinel-2 | 0002 | 0.63 | 0.221 | 0.240 | 0.193 | 2 / 11 | 0.223 | 0.233 |
| Sentinel-2 | 0003 | 0.63 | 0.176 | 0.263 | 0.230 | 8 / 11 | 0.209 | 0.247 |
| Sentinel-2 | 0004 | 0.40 | 0.218 | 0.197 | 0.224 | 10 / 11 | 0.245 | 0.234 |
| Sentinel-2 | 0005 | 0.40 | 0.209 | 0.226 | 0.229 | 9 / 11 | 0.224 | 0.235 |
| Sentinel-2 | 0006 | 0.50 | 0.175 | 0.266 | 0.228 | 8 / 11 | 0.202 | 0.249 |
| Sentinel-2 | 0007 | 0.50 | 0.173 | 0.257 | 0.235 | 7 / 11 | 0.233 | 0.267 |
| Sentinel-2 | 0008 | 0.40 | 0.189 | 0.246 | 0.238 | 9 / 11 | 0.241 | 0.249 |
| Sentinel-2 | 0009 | 0.40 | 0.187 | 0.245 | 0.232 | 10 / 11 | 0.236 | 0.244 |
| Sentinel-2 | 0010 | 0.40 | 0.221 | 0.230 | 0.233 | 9 / 11 | 0.262 | 0.267 |
| Sentinel-2 | 0011 | 1.00 | 0.210 | 0.250 | 0.222 | 5 / 11 | 0.197 | 0.242 |
| Sentinel-2 | 0012 | 1.00 | 0.196 | 0.245 | 0.217 | 6 / 11 | 0.196 | 0.254 |
| Sentinel-2 | 0013 | 0.16 | 0.232 | 0.223 | 0.224 | 6 / 11 | 0.168 | 0.229 |
| Sentinel-2 | 0014 | 0.79 | 0.246 | 0.273 | 0.213 | 1 / 11 | 0.179 | 0.243 |
| Sentinel-2 | 0015 | 0.32 | 0.179 | 0.264 | 0.193 | 2 / 11 | 0.217 | 0.225 |

The treated sites' departure equals their controls' and equals the validation error. A treated site has the largest departure of its eleven at 0 of 12 Sentinel-1 sites and 1 of 15 Sentinel-2 sites. The per-channel mean departure is shared by treated and control sites:

| sensor | post month | group | ch 1 | ch 2 | ch 3 | ch 4 | ch 5 |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | P11 | controls | -0.009 | +0.008 | +0.010 | +0.018 | -0.004 |
| Sentinel-1 | P11 | treated | -0.000 | +0.008 | +0.013 | +0.019 | -0.006 |
| Sentinel-1 | P12 | controls | -0.011 | +0.008 | +0.030 | -0.039 | +0.015 |
| Sentinel-1 | P12 | treated | -0.006 | +0.000 | +0.036 | -0.040 | +0.027 |
| Sentinel-2 | P11 | controls | +0.030 | +0.093 | +0.025 | +0.014 | -0.050 |
| Sentinel-2 | P11 | treated | +0.041 | +0.106 | +0.043 | +0.034 | -0.043 |
| Sentinel-2 | P12 | controls | +0.025 | +0.087 | -0.026 | -0.021 | -0.071 |
| Sentinel-2 | P12 | treated | +0.040 | +0.105 | -0.006 | -0.008 | -0.072 |

The Sentinel-2 shift in channel 2 at both post months is the July–August counterfactual meeting September–October imagery: the seasonal mismatch anticipated on 4 September when the donor is the most recent block. It is common to every site and is not an effect.

**What is established.** With the method as the paper defines it, on the 980-d latent of monthly composites, the smoothed own-history counterfactual predicts a clean pre-hurricane month better than the own-history mean on Sentinel-2, and the post-hurricane departure of the treated sites is indistinguishable from that of untouched sites. **Not established:** that the latent carries no Helene signal. The counterfactual for September–October is the smoothed July–August latent, so a seasonal change of the size of the validation error masks any effect of that size, and the stage-1 bandwidth smooths ten months almost to a line. Candidates for the next step are same-season historical blocks from earlier years, which need the longer series, and a statement of what an image-level effect is before a vector difference is read causally.

### 6.1 A learned covariate mapping

**What was added.** The model of §2 carries a covariate term x_t'β that the fit of §3 dropped. Following a suggestion of 10 September, that term is put back as a learned mapping g(x) of the site descriptors, fitted across sites, and nothing else in the pipeline changes: the kernel stage, the blocks, the stepwise selection, the weights, the counterfactual and the effect are run unchanged on the residual y_t − g(x_t), and the validation and the effect are scored on the observed latent against g + ℓ̂⁽⁰⁾, so every number below has the definition of §4–§6.

**The mapping.** x = the site's NLCD land-cover class (7 classes, class 41 as reference), elevation and slope, from the matching table; there is no time-varying covariate in the data. For each sensor, cloud fill and period, g is a multi-output ridge regression of the 980-d latent on these descriptors with an unpenalised intercept, fitted on the untreated rows of that period: every usable site for P01–P10 (61–165 sites), the control sites only for P11–P21 (120–150 sites); a treated site's post-hurricane months never enter any fit. The value subtracted from a site is the leave-one-site-out prediction, so no site's own latent enters its own covariate stage at any period. The ridge penalty is chosen by the leave-one-site-out error over P01–P10 on a log grid: 1000 for Sentinel-1 and 100 for Sentinel-2 under both fills, interior of the grid.

**What the mapping captures.** The leave-one-site-out R² of the descriptors is at or below 0.012 at every period and channel and negative at most (−0.034 to 0.012 over P01–P10): land cover, elevation and slope explain none of the cross-site variation of the latent within a month, and the penalty shrinks them to nothing. In effect g is the untreated cross-site mean latent of each period, learned from all sites before the hurricane and from the control sites after it. That period mean accounts for 0.7 % of the untreated latent variance on Sentinel-1 and 8 % on Sentinel-2 (chip-mean fill).

**Validation** (fit on P01–P(10 − n), predict the last n pre-hurricane months; RMSE over the 980 coordinates between the observed latent and the counterfactual, mean over treated sites; m = 4, as in §5):

| sensor | fill | n | validation error, without g | with g | own-mean reference | sites below the reference, without → with | most recent block alone, without / with |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 2 | 0.2303 | 0.2310 | 0.2360 | 11 → 10 / 12 | 12 / 12 |
| Sentinel-1 | chip-mean fill | 3 | 0.2478 | 0.2480 | 0.2452 | 2 → 2 / 12 | 12 / 12 |
| Sentinel-2 | chip-mean fill | 2 | 0.2080 | 0.2016 | 0.2415 | 15 → 14 / 15 | 15 / 15 |
| Sentinel-2 | chip-mean fill | 3 | 0.2330 | 0.2204 | 0.2540 | 14 → 14 / 15 | 15 / 15 |
| Sentinel-2 | historical fill | 2 | 0.2173 | 0.2113 | 0.2439 | 14 → 14 / 15 | 15 / 15 |
| Sentinel-2 | historical fill | 3 | 0.2371 | 0.2267 | 0.2525 | 10 → 13 / 15 | 15 / 15 |

**Effect run** (‖δ̂‖/√980 per post month, mean over sites, m = 4, as in §6):

| sensor | fill | n | post month | treated, without g | with g | controls, without g | with g |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | chip-mean fill | 2 | P11 | 0.2086 | 0.2090 | 0.2134 | 0.2142 |
| Sentinel-1 | chip-mean fill | 2 | P12 | 0.2648 | 0.2643 | 0.2638 | 0.2641 |
| Sentinel-1 | chip-mean fill | 3 | P11 | 0.2037 | 0.2041 | 0.2080 | 0.2088 |
| Sentinel-1 | chip-mean fill | 3 | P12 | 0.2528 | 0.2525 | 0.2528 | 0.2531 |
| Sentinel-1 | chip-mean fill | 3 | P13 | 0.2644 | 0.2632 | 0.2582 | 0.2579 |
| Sentinel-2 | chip-mean fill | 2 | P11 | 0.2067 | 0.1973 | 0.1987 | 0.1918 |
| Sentinel-2 | chip-mean fill | 2 | P12 | 0.2581 | 0.2490 | 0.2499 | 0.2428 |
| Sentinel-2 | chip-mean fill | 3 | P11 | 0.2064 | 0.1948 | 0.2002 | 0.1914 |
| Sentinel-2 | chip-mean fill | 3 | P12 | 0.2399 | 0.2307 | 0.2326 | 0.2244 |
| Sentinel-2 | chip-mean fill | 3 | P13 | 0.3084 | 0.2805 | 0.3254 | 0.3007 |
| Sentinel-2 | historical fill | 2 | P11 | 0.2040 | 0.1958 | 0.1979 | 0.1915 |
| Sentinel-2 | historical fill | 2 | P12 | 0.2453 | 0.2386 | 0.2468 | 0.2400 |
| Sentinel-2 | historical fill | 3 | P11 | 0.2057 | 0.1949 | 0.1997 | 0.1912 |
| Sentinel-2 | historical fill | 3 | P12 | 0.2293 | 0.2217 | 0.2292 | 0.2216 |
| Sentinel-2 | historical fill | 3 | P13 | 0.3020 | 0.2913 | 0.3080 | 0.2951 |

A treated site has the largest departure of its eleven at 0 of 12 Sentinel-1 sites with and without g, and at 1 of 15 Sentinel-2 sites in every cell but one (chip-mean fill, n = 2: 1 without g, 2 with g). The per-site ranks move by at most two places.

**Observations.** On Sentinel-1 the covariate stage changes nothing to three decimals. On Sentinel-2 it lowers the validation error by 3 to 5 % and lowers the post-period departure of treated and control sites by the same proportion, because the common September–October shift of §6 is now removed by the period mean learned from the control sites. The treated sites' departure still equals their controls'. The mapping does what the sine–cosine term does in the band-space pipeline, remove a shared seasonal level, and with these descriptors nothing beyond that: no effect was hiding behind a missing covariate of this kind.

---

## 7. Language models as the weight solver on the band-feature designs

**Question.** Given the band-mean tables of the band-space pipeline, does a general language model predict a held-out pre-hurricane month as well as that pipeline's own two estimators? This is the band-space pipeline's problem, not the latent one of §1–§6: the outcome is the spatial mean of each band over the parcel, Sentinel-2 B2, B3, B4, B8, B11, B12, NDVI, NDWI and Sentinel-1 VV, VH, VV−VH, on the monthly composites of §1.

**Designs.** Validation only, P10 (August 2024) predicted from information up to P09, 27 target cells (15 Sentinel-2 sites, 12 Sentinel-1 sites), nothing treated. *SHC design*: the target site's own P01–P09. *SCM design*: the target site's P01–P09 and the five matched donor sites (control rank 1–5, the donor rule of the band-space SCM) at P01–P10, so the donors are observed at the predicted month, as in every synthetic-control fit of this project. Each prompt opens with a one-paragraph guide to what the bands measure, then the table, then a one-line answer format; the model returns the band means at P10. No word for the hurricane, the landslides or a treatment appears in any prompt. Settings: temperature 0, fixed seed, output limit at the service ceiling of 8,000 tokens, one attempt per cell, a retry only on a rate-limit response.

**Models.** The four models hosted by the university service: gpt-oss-120b, GLM-5.3, Kimi-K3, DeepSeek-V4-Flash. GLM-5.3 and DeepSeek-V4-Flash spend the whole 8,000-token budget on hidden reasoning and return no text in nearly every attempt at either reasoning setting (GLM-5.3 answered 2 of 27 cells at its default setting and none at low; DeepSeek-V4-Flash none at either) and are not reported. Kimi-K3 answers only in its low-reasoning variant (the default exceeds the service's 360-second gateway limit). gpt-oss-120b answers at both settings and is reported at both.

**Metrics, exactly the band-space pipeline's.** SHC design: per band, squared error at P10 divided by the band's variance (one degree of freedom lost) over the site's P01–P10, then the mean over bands, the outcome metric of the band-space slides; the held-out variant with the P01–P09 variance is checked against that pipeline's stored values (§7.1). SCM design: standardized validation RMSE at P10, the error in each band divided by the pooled standard deviation over the targets and their donors at P01–P09, root mean square over bands. Two reference rows on the same features and split: the band-space pipeline's held-out SHC (seasonal sine–cosine partialled out, local-linear Gaussian kernel with a two-month bandwidth, per-band standardization, blocks of four months, stepwise block selection, simplex weights, prediction from the selected blocks' observed forward month), re-run and reproducing that pipeline's stored per-site values; and the band-space SCM solver (simplex on the flattened standardized P01–P09 matrix), which reproduces that pipeline's stored weights. The site's own P01–P09 mean is the no-estimation reference.

### 7.1 SHC design: all 27 cells answered by every reported model

Outcome metric (variance over P01–P10), per site and mean / median over sites, lower is better. On the held-out variant (variance over P01–P09) the band-space SHC row equals that pipeline's stored 0.4925 (Sentinel-1) and 0.7867 (Sentinel-2) and the ordering of the columns on the mean is unchanged.

| sensor | site | Kimi-K3 (low reasoning) | gpt-oss-120b (low reasoning) | gpt-oss-120b (default reasoning) | band-space SHC | own mean P01–P09 |
|---|---|---|---|---|---|---|
| Sentinel-1 | 0015 | 0.258 | 0.527 | 0.288 | 1.288 | 0.440 |
| Sentinel-1 | 0016 | 0.216 | 0.030 | 0.050 | 0.087 | 0.615 |
| Sentinel-1 | 0017 | 0.166 | 0.238 | 0.351 | 0.314 | 0.793 |
| Sentinel-1 | 0018 | 0.158 | 0.126 | 0.158 | 0.291 | 0.713 |
| Sentinel-1 | 0019 | 0.380 | 0.414 | 0.335 | 0.526 | 1.004 |
| Sentinel-1 | 0020 | 0.081 | 0.070 | 0.070 | 0.184 | 0.407 |
| Sentinel-1 | 0021 | 0.727 | 0.139 | 1.120 | 0.532 | 0.197 |
| Sentinel-1 | 0022 | 0.327 | 0.971 | 0.074 | 0.485 | 0.246 |
| Sentinel-1 | 0023 | 1.217 | 1.793 | 1.801 | 1.261 | 1.034 |
| Sentinel-1 | 0024 | 0.442 | 0.489 | 0.321 | 0.412 | 1.212 |
| Sentinel-1 | 0025 | 0.332 | 0.428 | 0.335 | 0.568 | 0.901 |
| Sentinel-1 | 0027 | 0.259 | 0.259 | 0.355 | 0.451 | 0.469 |
| Sentinel-2 | 0001 | 0.376 | 0.116 | 0.284 | 0.170 | 0.234 |
| Sentinel-2 | 0002 | 1.313 | 1.327 | 1.270 | 1.425 | 2.107 |
| Sentinel-2 | 0003 | 0.927 | 0.758 | 1.372 | 0.996 | 1.382 |
| Sentinel-2 | 0004 | 9.516 | 4.510 | 3.140 | 3.421 | 0.588 |
| Sentinel-2 | 0005 | 2.571 | 1.508 | 3.118 | 1.890 | 1.458 |
| Sentinel-2 | 0006 | 0.851 | 0.861 | 0.464 | 0.643 | 0.822 |
| Sentinel-2 | 0007 | 0.148 | 0.140 | 0.174 | 0.191 | 0.788 |
| Sentinel-2 | 0008 | 0.260 | 0.194 | 0.265 | 0.232 | 1.095 |
| Sentinel-2 | 0009 | 0.396 | 0.364 | 0.276 | 0.211 | 1.012 |
| Sentinel-2 | 0010 | 1.329 | 0.844 | 1.876 | 0.845 | 1.164 |
| Sentinel-2 | 0011 | 0.231 | 0.117 | 0.182 | 0.119 | 0.566 |
| Sentinel-2 | 0012 | 1.228 | 0.636 | 0.398 | 0.941 | 1.037 |
| Sentinel-2 | 0013 | 0.272 | 0.121 | 0.180 | 0.158 | 0.788 |
| Sentinel-2 | 0014 | 0.079 | 0.074 | 0.059 | 0.039 | 0.572 |
| Sentinel-2 | 0015 | 0.131 | 0.136 | 0.258 | 0.056 | 1.213 |
| **Sentinel-1, mean / median** | | 0.380 / 0.293 | 0.457 / 0.336 | 0.438 / 0.328 | 0.533 / 0.468 | 0.669 / 0.664 |
| **Sentinel-2, mean / median** | | 1.309 / 0.396 | 0.781 / 0.364 | 0.888 / 0.284 | 0.756 / 0.232 | 0.989 / 1.012 |
| **sites where the model beats the band-space SHC** | | S1 9 / 12, S2 3 / 15 | S1 9 / 12, S2 10 / 15 | S1 9 / 12, S2 5 / 15 | | |

**Observations.** (1) On Sentinel-1 every model beats the band-space SHC: Kimi-K3 at 9 of 12 sites, gpt-oss-120b at 9 of 12 at either setting, and all three beat the own-mean reference on mean and median. (2) On Sentinel-2 the band-space SHC has the lowest mean and median; gpt-oss-120b beats it at 10 of 15 sites at low reasoning and 5 of 15 at default, Kimi-K3 at 3 of 15. The model means are set by three sites (0004, 0005, 0010) where the models are far above 1 and the band-space SHC is itself at 3.42, 1.89 and 0.85; on the medians the models are within 0.05–0.16 of it. (3) The band-space SHC's forecast is the P09 value carried forward at 26 of the 27 cells: the stepwise rule keeps the most recent block alone at weight 1, and only Sentinel-2 site 0014 adds a second block (weight 0.10). What the models add on Sentinel-1 is therefore a trend or seasonal reading of nine smooth backscatter values; on Sentinel-2, where reflectance jumps with residual cloud, a model cannot tell a jump from a trend and carrying P09 forward is as good. (4) gpt-oss-120b at low and default reasoning are close (low is better at 16 of 27 cells); the reasoning setting is not what separates a model from the band-space SHC.

### 7.2 SCM design: incomplete, reported as it stands

The SCM prompt is three to four times longer (six parcels, ten months: 1,650–1,750 tokens for Sentinel-1 and 3,650–3,950 for Sentinel-2, against 480 and 869 for the SHC design) and the models reason longer on it. Answered cells per model, with the two failure modes (hidden reasoning fills the 8,000-token ceiling and no text is returned; the service gateway closes the request at 360 s):

| model | Sentinel-1 answered | Sentinel-2 answered | output-ceiling failures | gateway timeouts |
|---|---|---|---|---|
| Kimi-K3 (low reasoning) | 9 / 12 | 10 / 15 | 4 | 4 |
| gpt-oss-120b (low reasoning) | 12 / 12 | 15 / 15 | 0 | 0 |
| gpt-oss-120b (default reasoning) | 11 / 12 | 2 / 15 | 14 | 0 |

Only gpt-oss-120b at low reasoning answers every cell. Because the other model rows are subsets, each is compared with the band-space SCM on the same cells; the band-space SCM on all cells is 0.3631 / 0.3149 (Sentinel-1) and 0.3416 / 0.3153 (Sentinel-2).

| sensor | model | cells | model, mean / median | band-space SCM on the same cells | cells where the model is better |
|---|---|---|---|---|---|
| Sentinel-1 | Kimi-K3 (low reasoning) | 9 | 0.2309 / 0.2192 | 0.4195 / 0.4049 | 6 of 9 |
| Sentinel-1 | gpt-oss-120b (low reasoning) | 12 | 0.3607 / 0.2519 | 0.3631 / 0.3149 | 4 of 12 |
| Sentinel-1 | gpt-oss-120b (default reasoning) | 11 | 0.3056 / 0.2393 | 0.3685 / 0.3257 | 5 of 11 |
| Sentinel-2 | Kimi-K3 (low reasoning) | 10 | 0.3532 / 0.3013 | 0.3195 / 0.2638 | 4 of 10 |
| Sentinel-2 | gpt-oss-120b (low reasoning) | 15 | 0.5720 / 0.5978 | 0.3416 / 0.3153 | 5 of 15 |
| Sentinel-2 | gpt-oss-120b (default reasoning) | 2 | 0.4417 / 0.4417 | 0.3110 / 0.3110 | 0 of 2 |

**Observations.** (1) With every cell answered, gpt-oss-120b ties the band-space SCM on Sentinel-1 (the same mean, better at 4 of 12 sites) and is clearly worse on Sentinel-2. (2) The Kimi-K3 advantage on Sentinel-1 rests on nine cells on which the band-space SCM itself does badly (0.42 there against 0.36 on all twelve); the subset flatters the model. (3) Two of the four models cannot be used at all on this service, and a third loses 8 of 27 cells on this design.

### 7.3 What the models do: plain rules, a diagnosis of the predictions, prompt ablations

The band-space SHC forecast is P09 carried forward, so the question behind §7.1 is whether the models add anything that a one-line rule does not. Three checks, all on the same 27 cells and the same metric (variance over P01–P10).

**Plain forecasting rules** of P10 from the target's own P01–P09, per band: P09 carried forward; the mean of the last three months (P07–P09); the own mean; a least-squares line through the nine months, and through the last four, extrapolated one step; additive Holt smoothing with a damped trend whose three parameters are chosen on P01–P09 alone; and the mean of the two models' answers.

| method | Sentinel-1, mean / median | below the band-space SHC | Sentinel-2, mean / median | below the band-space SHC |
|---|---|---|---|---|
| mean of the last three months | 0.283 / 0.210 | 11 / 12 | 0.503 / 0.364 | 7 / 15 |
| damped Holt smoothing | 0.298 / 0.199 | 10 / 12 | 0.650 / 0.413 | 10 / 15 |
| Kimi-K3 (low reasoning) | 0.380 / 0.293 | 9 / 12 | 1.309 / 0.396 | 3 / 15 |
| linear trend on nine months | 0.389 / 0.251 | 9 / 12 | 1.122 / 1.009 | 2 / 15 |
| mean of the two models | 0.402 / 0.372 | 9 / 12 | 1.002 / 0.343 | 6 / 15 |
| gpt-oss-120b (default reasoning) | 0.438 / 0.328 | 9 / 12 | 0.888 / 0.284 | 5 / 15 |
| gpt-oss-120b (low reasoning) | 0.457 / 0.336 | 9 / 12 | 0.781 / 0.364 | 10 / 15 |
| band-space SHC (= P09 carried forward) | 0.533 / 0.468 |  | 0.756 / 0.232 |  |
| linear trend on the last four months | 0.590 / 0.482 | 4 / 12 | 1.210 / 1.367 | 4 / 15 |
| own mean P01–P09 | 0.669 / 0.664 | 4 / 12 | 0.988 / 1.012 | 2 / 15 |

**Which rule the models track.** Every prediction is expressed as its move from P09 in units of the band's P01–P09 standard deviation. The model's move is regressed on the move of the nine-month linear trend (slope 1 = the trend, 0 = P09 carried forward, negative = against the trend); "direction right" is the share of cells × bands where the move has the sign of the observed change; "nearest rule" counts which plain rule the model's value is closest to.

| sensor | model | slope on the linear-trend move | R² | direction right: model | direction right: damped Holt | direction right: own mean | nearest rule (cells × bands) |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | Kimi-K3 (low) | 0.47 | 0.27 | 75 % | 81 % | 58 % | mean of the last three months 16, P09 carried forward 6 |
| Sentinel-1 | gpt-oss-120b (low) | 0.52 | 0.29 | 69 % | 81 % | 58 % | mean of the last three months 14, linear trend on nine months 7 |
| Sentinel-1 | gpt-oss-120b (default) | 0.69 | 0.40 | 69 % | 81 % | 58 % | mean of the last three months 14, P09 carried forward 9 |
| Sentinel-2 | Kimi-K3 (low) | -0.61 | 0.56 | 42 % | 68 % | 62 % | P09 carried forward 54, damped Holt smoothing 27 |
| Sentinel-2 | gpt-oss-120b (low) | -0.12 | 0.10 | 55 % | 68 % | 62 % | P09 carried forward 43, damped Holt smoothing 30 |
| Sentinel-2 | gpt-oss-120b (default) | -0.09 | 0.02 | 50 % | 68 % | 62 % | P09 carried forward 40, mean of the last three months 32 |

The three Sentinel-2 sites that set every model's mean (0004, 0005, 0010) tell one story. At site 0004 the P09 composite is a bright outlier in the visible and shortwave bands (B2 0.045 → 0.086, B4 0.052 → 0.094, residual cloud), and P10 returns to 0.054 and 0.057. P09 carried forward keeps the outlier; both models push it further (B2 0.100 and 0.090); every averaging rule reverts. The models read a cloud jump as a trend.

**Prompt ablations** (27 cells per arm, one deterministic call each; the anonymous arm has one Kimi-K3 cell unanswered after a gateway timeout):

| prompt | Kimi-K3, Sentinel-1 | Kimi-K3, Sentinel-2 | gpt-oss-120b, Sentinel-1 | gpt-oss-120b, Sentinel-2 |
|---|---|---|---|---|
| prompt of §7.1 | 0.380 / 0.293 | 1.309 / 0.396 | 0.457 / 0.336 | 0.781 / 0.364 |
| no band guide | 0.426 / 0.346 | 1.044 / 0.383 | 0.353 / 0.259 | 0.810 / 0.365 |
| anonymous quantities x1…xk, no sensor, no guide | 0.389 / 0.348 | 0.697 / 0.263 | 0.585 / 0.335 | 0.696 / 0.298 |
| no calendar dates | 0.359 / 0.295 | 1.189 / 0.678 | 0.594 / 0.479 | 0.951 / 0.334 |
| prompt of §7.1 plus a second line naming the rule used | 0.388 / 0.266 | 1.221 / 0.373 | 0.393 / 0.315 | 0.864 / 0.364 |
| mean of the last three months, for reference | 0.283 / 0.210 | 0.503 / 0.364 | | |

**The rules the models state for themselves** (the second line of the reply in the last arm). Sentinel-1: Kimi-K3 names an average of the last three months at 9 of 12 sites and a trend at 3; gpt-oss-120b an average at 9, a trend at 2, P09 carried forward at 1. Sentinel-2: Kimi-K3 names a seasonal story at 14 of 15 sites (a post-peak senescence decline extrapolated at 9, the summer plateau persisted at 5) and a trend at 1; gpt-oss-120b a trend extrapolation at 10 and an average at 5.

**Observations.** (1) The Sentinel-1 advantage of §7.1 is the advantage of smoothing over persistence: the mean of the last three months is below the band-space SHC at 11 of 12 sites and below every model on mean and median, and the models say that this average is what they computed. (2) On Sentinel-2 the models extrapolate a seasonal decline that they know August should bring, which is right on the median site and wrong wherever P09 carries residual cloud; Kimi-K3 moves against the linear trend (slope −0.61) and has the direction right at 42 % of cells × bands, against 68 % for damped Holt. (3) Removing every domain word (anonymous quantities) improves both models on Sentinel-2 — Kimi-K3's mean from 1.31 to 0.70 — so the band guide and the sensor name are what trigger the seasonal story; removing the calendar dates hurts instead, so the season is used and is useful when it is right. (4) Across the five prompt wordings one model's Sentinel-1 mean ranges from 0.35 to 0.59, the whole spread between models in §7.1; single deterministic calls on 12 sites cannot rank models. (5) No prompt brings either model to the three-month mean.

**Decision.** The SHC design carries forward for the model-based comparison: prompts are short and every reported model answers every cell; the SCM design is not reportable at the full sample with these models and is left as it stands. What the language models contribute on this design is a hand-written rule, a three-month average on Sentinel-1 and a seasonal extrapolation on Sentinel-2, and the rule they approximate is better run directly. The reference that carries forward is the mean of the last three months: it beats P09 carried forward, which is what the band-space SHC reduces to on this panel, at 11 of 12 Sentinel-1 sites and 7 of 15 Sentinel-2 sites, so the latent SHC of §5, the effect run and the picture experiment are to be compared against that mean rather than against persistence. On Sentinel-2 a one-month departure is inside the size of a residual-cloud jump for every method; an effect read from band means there needs a cloud screen or a longer post-period window.

---

## 8. Reproducibility

Every number in §1 and §4–§7 is produced by a script and read from its output table. Paths are relative to `Satellite/notebooks/ts_SCM_ASCM/`; §7 lives in `Satellite/notebooks/llm_band_sc/`. The synthetic-historical-control results of the 3 September report rested on a design that departed from the method (last pre-period as the only target, no post-treatment period in any block, weights matched on the raw outcome, all blocks as donors); their scripts and tables are removed, and the shared grid and comparison tables regenerated without them are unchanged in every retained row (maximum difference 0).

| script / artifact | produces | read by |
|---|---|---|
| `panel_monthly.py` | `panel_monthly_index.csv` (site × sensor × month index, usability) | §1 |
| `panel_monthly_encode.py` | `data/embeddings_tok_panel/latents_monthly_long.npz`, `latents_monthly_long_histfill.npz` (+ `.fillcounts.csv`, manifests) | §1 |
| `panel_shc.py`, `panel_shc_tests.py` | the estimator and its consistency tests (the block construction reproduces the slides' $T_0 = 10$, $m = 4$, $n = 2$ example and the $z_i = n + i - 1$ rule; the smoother returns the series at $h \to 0$, the least-squares line at $h \to \infty$ and any linear series unchanged; a planted convex combination of two blocks is recovered by the stepwise rule) | §3 |
| `panel_shc_run.py` | `panel_shc_fits.csv`, `panel_shc_effects.csv`, `panel_shc_bandwidth.csv` (CV curves), `panel_shc_path.csv` (selection paths and every candidate), `panel_shc_summary.csv` | §4–§6 |
| `22_shc_monthly.ipynb` | tables and the bandwidth-curve figure | §4 |
| `panel_covariate.py`, `panel_shc_run.py --covariate site` | `panel_covariate.csv` (penalty, leave-one-site-out R², coefficients), `panel_shc_*_covariate.csv` (the same five tables on y − g) | §6.1 |
| `23_shc_covariate.ipynb`, `panel_shc_tests.py` (gates 7–12: coefficient recovery, leave-one-site-out, treated post rows absent, the driver without the option reproduces the 4 September fits) | the side-by-side tables | §6.1 |
| `Docs/Synthetic Historical Control_Sep 4.pptx`, meeting transcript of 4 September 2026 | the method walkthrough and the design decisions | §2–§3 |
| `lb_features.py`, `lb_prompts.py` | `lb_features_monthly.csv` (band means P01–P10, targets and donors), `lb_prompts_shc.jsonl`, `lb_prompts_scm.jsonl` (full prompt text and withheld truth) | §7 |
| `lb_run.py` | `lb_replies_<design>_<model>_<effort>.jsonl` (every reply, token usage, finish reason, errors) | §7 |
| `lb_shc.py`, `lb_lib.py`, `lb_tests.py` | the band-space SHC re-run, the band-space scaler and solver, the reproduction checks (the 27 stored SHC values to 4 × 10⁻⁷, the stored biweekly weights to 6 × 10⁻⁸, the band means to 5 × 10⁻⁷) | §7 |
| `lb_baselines.py`, `lb_prompts_ablation.py` | the plain rules (scored inside `lb_scores_shc.csv`), `lb_diag_models.csv`, `lb_diag_summary.csv`, `lb_diag_s2_sites.csv`; `lb_prompts_abl_*.jsonl` and `lb_replies_abl_*_<model>_low.jsonl` (the four ablation arms) | §7.3 |
| `lb_score.py`, `41_llm_band_sc.ipynb` | `lb_scores_shc.csv`, `lb_summary_shc.csv`, `lb_summary_shc_band.csv`, `lb_scores_scm.csv`, `lb_summary_scm.csv`; the notebook prints one full prompt per design and sensor | §7 |
