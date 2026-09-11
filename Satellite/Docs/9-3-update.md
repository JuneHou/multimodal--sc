# Position, history, and multiple outcomes: synthetic control on TerraMind latents

**Progress report for the meeting of 3 September 2026.** Follows the 27 August report
(`8-27-update.md`). Sources for the method sections:
Tian, Lee & Panchenko, *Synthetic Controls with Multiple Outcomes*, arXiv:2304.02272v2 (26 Jul 2024).
The scripts, tables and latent sets behind every number are listed in §13.

**Summary.** The 27 August report left two requests: keep the image whole, so that a donor
parcel stands in only for a treated position of the same land type, and let the site's own
history serve as the donor pool. The first is implemented and evaluated here on six
representations of the TerraMind tokenizer latent, each with the estimator it was built for,
under four cloud-fill strategies, with P10 as the held-out period (§6–§7). The second enters
only as the own-history mean reference: its synthetic-historical-control implementation was
withdrawn on 4 September 2026 and is redesigned in a separate study.

- *Position.* Same-position land-cover agreement between a treated chip and its donors averages
  0.469, so raw parcel-to-parcel comparison is structurally wrong. Class-and-history alignment
  cuts whole-image error by 26 % on both sensors (latent980: Sentinel-1 0.3288 → 0.2438,
  Sentinel-2 0.2885 → 0.2135), and the history term, not the class constraint, produces the gain
  (§5.2, §7).
- *Own history.* On Sentinel-1 the plain mean of the site's own nine periods beats every
  donor-based fit in five of the six representations: each period is the site's fixed image
  plus independent speckle (§6).
- *Settled on the way.* One scaler-free metric for every experiment (§1); the FSC arm, which the
  27 August report had mislabelled, is now the augmented estimator with a tuned penalty and is
  worth 0–6 % on this panel (§9); the cloud fill is the largest single effect on Sentinel-2 in
  every pooled representation (§7).
- *Method fidelity.* Deriving each estimator from its source (§5) uncovered one step of a
  published method that the code had omitted, the cell standardization of multi-outcome SC. It
  is now implemented and measured: standardization is the largest improvement to the
  shared-weight arm on Sentinel-2 (0.0289 → 0.0267 under historical fill).

**Scope.** This report is about the *embedding*: six representations of the TerraMind tokenizer
latent, each evaluated with the estimator it was built for, under four cloud-fill strategies. The
band-mean pipeline (per-band SCM, multi-outcome SC on 8 bands) is the companion analysis and is
not re-run here. It reproduces on our copy of the data (Exp 1 0.0282 against the reported 0.0279,
Exp 2 0.0206 against 0.0186, in metric M1) and is cited only as that.

**Method sections.** Each of the two arms that carries a result is derived in §5 from its
source: the published equations in the source's own numbering, the dimensions of our fit against
the source's own application, and the implementation of each step. Two points of the
implementation are confirmed there, the demeaning of Tian–Lee–Panchenko's eq (6) and the
weighting matrix V of the band-space specification; the one departure is measured in §5.1.

---

## 1. The two metrics

Let `ŷ` be an arm's predicted P10 vector for one treated site and `y` the observed one, both of
dimension M.

### M1 — band-space RMSE, the companion pipeline's metric

Raw band-mean RMSE in reflectance (S2) or dB (S1): per band the RMS over sites, headline the RMS
over all site × band cells, as used in the band-space scripts. This report quotes it only for the
band-space results and for the decode check (§10), which live natively in band space. M1 and M2 are never
placed in one column.

### M2 — our metric, for every one of our experiments

**Plain RMSE over the outcome dimensions, in native units, no scaler:**

$$
\text{rmse}_i \;=\; \frac{\lVert \hat{y}_i - y_i \rVert_2}{\sqrt{M}}
\qquad\text{headline}\;=\;\frac{1}{10}\sum_{i=1}^{10}\text{rmse}_i
$$

The metric has no scaler because each earlier inconsistency in this project (the historical-fill
denominator, the cross-family comparison after dividing by a standard deviation, the 2×2-block
arm) came from a scaler that did not match the object being scored. Relative L₂
($\lVert \hat{y}-y\rVert /\lVert y\rVert$) was considered: on every block it equals RMSE ÷ rms(y_i) per site and ranks
the estimators identically within every representation, but its denominator is mostly the fixed
offset of the FSQ codes (targets sit at rms 0.33–0.76 on [0, 1]), so it says less than the plain number.
It is retained as a secondary column in the source tables only.

**Units per block:** latent value for chip mean, quantile, 2×2 block and latent980; squared
latent value for the Gram. **What M2 licenses:** down a column within one representation the
comparison is exact — same object, same target, same sites. Across representations neither M2
nor any other metric gives a ranking: the units differ, and the aggregation differs — a
chip mean averages 196 parcels, so its per-dimension error is small by construction.
Cross-representation rows are context.

**Pass flags.** Two per site, both in M2 and in the same fill column: **C2** — the prediction
beats the site's own P01–P09 mean; **C3** — the fitted weights beat the equal-weight average of
the same donors. Both references are averages with nothing to estimate; they are listed in the
tables as references, not as estimators. The earlier criterion C1 (test ≤ 1.5 × train) is not
reported: with four free weights it passes automatically and carries no information. Train RMSE
is still reported as a number.

### The data, and what is unmodified

Source: `Satellite/data/biweekly_datasets/*.tif`, unaltered. Masked (cloudy / nodata) pixels are
NaN. TerraMind cannot encode NaN, so **every latent is post-fill**; chip-mean fill and historical
fill are two choices among mandatory ones, not modified versus unmodified. That is why the fill
appears as a design variable in every table below.

**Why P09 is not a test period.** P09 Sentinel-2 targets are heavily cloud-masked and therefore
themselves filled (up to 94 % of pixels at site 0010). Scoring against them asks the counterfactual
to reproduce a *cloudy* image and is circular wherever the fill supplied the target. Test period is
**P10 only** (98 % valid; every treated chip has all 196 parcels valid), trained on P01–P09.

---

## 2. The representations, and what each was built for

Every arm below is an embedding representation of the same object: the TerraMind
tokenizer latent of one chip, 5 channels × 14 × 14 parcels (980 numbers, FSQ codes on
[0, 1]). Band means belong to the band-space analysis and are not re-run here.

| representation | input dim M | construction | estimator it was built for |
|---|---|---|---|
| chip mean | 5 | mean of each channel over the 196 parcels | plain SCM (the identity embedding); multi-outcome SC over the 5 channels (notebook 16) |
| Gram | 15 | upper triangle incl. diagonal of A Aᵀ/196, column-major (their `covvec`) — what fires together at the same parcel | FSC, Okano–Kurisu example 3 (PSD cone; `nearPD` projection) |
| quantile | 100 | per channel, the quantile function on 20 grid points (0.01 … 0.99), concatenated — composition without position | FSC, Okano–Kurisu example 2 (2-Wasserstein; `modif` projection) |
| combined | 115 | quantile ⊕ Gram, each block scaled once to unit mean square (our direct sum) | FSC |
| 2×2 block | 245 | 5 × 7 × 7 block means of the parcel grid | whole-image SCM, alignment (Exp 1) |
| latent980 | 980 | the parcel grid itself | whole-image SCM, alignment (Exp 1) |

The 2×2 block is the parcel grid at half resolution: each 2 × 2 group of parcels is averaged
(5 × 7 × 7 = 245), which keeps position while averaging out the per-parcel lattice
quantization (the FSQ codes take 5–8 values per channel). It is the same object as latent980
at a coarser resolution; both are run on every spatial estimator and neither is preferred by
construction. The chip mean is the other end of the same scale, the grid averaged to one number
per channel.

Cells outside this mapping are not run. A full estimator × representation cross is not
reported: an estimator applied to a representation it was not built for has no interpretation.

**What the outcome is in multi-outcome SC.** The K = 5 channel means of one chip are treated
as K related outcomes of the same unit that share one simplex weight vector: the fit stacks
the K × T₀ pre-period values and, following the band-space multi-outcome implementation, first
removes each unit's own pre-period mean per outcome (within-site × outcome demeaning). The
contrast arm, one fit per channel, fits K separate weight vectors. The K = 8 band version
belongs to the band-space analysis and is referenced, not re-run. **§5.1 derives this from the source**
(the factor model, the matching condition, the bias rate, the demeaning step).

## 3. Cloud fill — the four strategies tested

Masked (cloudy / nodata) pixels are NaN in the shipped GeoTIFFs. TerraMind cannot encode NaN,
so every latent is post-fill; the fill is a design variable of the latent path and is listed
as such. Four strategies, two mechanisms:

| strategy | mechanism | applies to |
|---|---|---|
| chip-mean fill | masked pixels replaced by the chip's band means before encoding (latent set `latents_biweekly.npz`) | both sensors, all representations |
| historical fill | masked pixels replaced by the per-pixel median of the same site's clean P01–P08 chips, re-encoded (`latents_biweekly_histfill.npz`); no target informs its own fill | both sensors, all representations |
| masked pooling | chip-mean latents pooled over the **valid** parcels only (parcel ≥ 50 % finite pixels; chip needs ≥ 10 valid parcels, else it falls back to all-196 pooling) | Sentinel-2, pooled representations only |
| masked drop | masked pooling, and a chip below 10 valid parcels is treated as **missing**: that period is dropped from the fit for the whole donor group | Sentinel-2, pooled representations only |

Masked pooling and masked drop cannot apply to the 2×2 block or latent980 (dropping parcels
changes the vector), and Sentinel-1 has 0.09 % masked pixels, so those cells are blank.
Sentinel-1 is reported under both latent sets; the two columns are identical in every cell
because only 1 of its 1,147 latents differs between the sets.

## 4. Pipeline

| step | what happens | what it rests on |
|---|---|---|
| 0 · panel | per treated site: the site + its 5 covariate-matched controls (the collaborator's notebook-14 design), P01–P09 fit, P10 test | one treated unit, untreated donors, no interference |
| 1 · fill | one of the four strategies above, upstream of the encoder or at pooling time | the fill is a choice among mandatory ones, never "unfilled" |
| 2 · Ψ | chip → representation vector (table in §2) | isometric embedding into a Hilbert space for the FSC representations; the parcel grid for the whole-image arms |
| 3 · weights | simplex SCM (SLSQP), multi-outcome SC with demeaning (and with the source's cell standardization), augmented FSC (their R code, λ by their CV) | treated ≈ convex combination of donors — derived in §5 |
| 4 · evaluation | M2 at P10 and on the fit rows; C2 and C3 per site | a prediction that beats neither reference has no claim |

Dimensions per fit:

| quantity | value |
|---|---|
| donors J | 5 |
| free simplex weights J − 1 | **4** |
| fit periods T₀ | 9 (fewer where a group has missing periods: S1 site 07, and masked drop) |
| outcome dim M | 5 / 15 / 100 / 115 / 245 / 980 |
| stacked values M · T₀ | 45 – 8,820 |

Against the FSC paper's own applications — fertility 20 donors (19 free weights, M·T₀ = 704),
mortality 17 (16; 2,100), trade covariance 22 (21; 1,305) — we fit 4 free weights. That is
the first-order limitation of every arm here: M · T₀ is not the number of independent
constraints, and more outcome dimensions do not add weights.


---

## 5. Method walkthroughs

Two arms carry the results, and each is derived here from its source: the original
equations in the source's own numbering, the dimensions of our fit against the dimensions
of the source's own application, and the implementation of each step. Where the implementation
differs from the published method the difference is named and measured.

Notation is the source's throughout, with one harmonisation: the source's outcome scalar
becomes our M-vector (the representation of §2), and the extra M coordinates are stacked
inside the design exactly as the extra periods are.

### 5.1 Multi-outcome SC — Tian, Lee & Panchenko (arXiv:2304.02272v2)

The paper the band-space multi-outcome analysis follows. Six of our arms are its estimators:
shared weights, plain, demeaned, and demeaned + standardized (`scm`, `scm_demeaned`,
`scm_demeaned_std`), and the same three with one fit per channel (`perdim_scm`,
`perdim_scm_demeaned`, `perdim_scm_demeaned_std`).

**The model.** For J + 1 units, T periods, and K outcomes in a *domain* — "a collection of
related outcomes driven by the same set of observed and unobserved predictors" — the
untreated potential outcome follows the interactive fixed-effects model, eq (2):

$$
Y^{0}_{it,k} \;=\; \delta_{t,k} \;+\; Z_i'\,\theta_{t,k} \;+\; \mu_i'\,\lambda_{t,k} \;+\; \varepsilon_{it,k}
\tag{2}
$$

$\delta_{t,k}$ is the time trend of outcome k, `Z_i` the observed and $\mu_i$ the unobserved
predictors of unit i, with outcome-specific loadings $\theta_{t,k}$, $\lambda_{t,k}$. **The key
assumption is that $\mu_i$ is common to all K outcomes in the domain** — the paper states
that if the outcomes load on different $\mu_{i,k}$, "we lose the benefit of matching on
multiple related outcomes".

*Our reading.* Unit i = site, t = biweekly period, k = one of the 5 FSQ tokenizer channels.
$\mu_i$ is never estimated, in the paper or here: it is whatever site-specific, time-invariant
factor drives all K outcomes at once. **Our interpretation, not a data finding**, is that for a
chip this is the surface itself — land-cover composition, terrain, moisture regime — by analogy
with the paper's own list for economic outcomes ("infrastructure, technology, natural resources").
The 5 channels are 5 detectors of one frozen tokenizer reading the *same* 14 × 14 parcel
grid of the *same* chip, so a shared $\mu_i$ with channel-specific loadings $\lambda_{t,k}$ is the
natural description, not an added assumption. It is the same argument the band-space analysis
makes for its K = 8 bands ("all driven by vegetation, moisture & soil structure").

**The matching condition.** Conditions 1 and 2 are regularity (shocks independent across
i, t, k with zero conditional mean; the smallest eigenvalue of $(KT_0)^{-1} \sum_k\sum_t \lambda_{t,k}\lambda'_{t,k}$
bounded below). Condition 3, eq (3), is the perfect-fit condition — there exist weights
with $\hat{w}_j \ge 0$, $\sum_{j=2}^{J+1} \hat{w}_j = 1$ such that

$$
\sum_{j=2}^{J+1}\hat{w}_j Z_j = Z_1
\qquad\text{and}\qquad
\sum_{j=2}^{J+1}\hat{w}_j Y_{jt,k} = Y_{1t,k}
\quad\text{for all } t\le T_0,\; k\in\mathcal{K}
\tag{3}
$$

which in practice "may hold only approximately and the SC weights are obtained by
minimizing a weighted sum of the squared distance between the left-hand and the right-hand
sides of (3)" (p. 7).

*Our implementation.* The weights solve the simplex-constrained least-squares problem on the
stacked K·T₀ vector (SLSQP, bounds [0, 1], equality $\sum w = 1$, tolerance 10⁻¹², periods with
any missing entry dropped).

**The covariate half of (3) is not in the objective.** In eq (3) the weighted donors must
reproduce the treated unit's covariates $Z$ (left half) and its pre-period outcomes $Y$ (right
half). We match only the right half; the covariates (NLCD class, elevation, slope, distance,
from the site-matching table) built the donor pool. Proposition 1 is stated under
Conditions 1–3, and Condition 3 includes the $Z$ equality, so a paper cannot claim it verbatim.
Two positions are defensible: treat the surface descriptors as part of the unobserved $\mu_i$, so
the model has no observed predictors and eq (3) is the outcome equality alone; or keep $Z$ and
cite the paper's own remark (p. 8) that Botosaru and Ferman (2019) relax the perfect-fit condition
for the observed predictors. Either way the paper must say covariates built the pool and were not
matched in the fit.

**The two estimators, and what separates them.** eq (4), the multiple-outcome estimator,
and eq (5), the conventional single-outcome one:

$$
\hat{\tau}_{1t,k} = Y_{1t,k} - \sum_{j=2}^{J+1}\hat{w}_j\,Y_{jt,k}
\qquad\text{one }\hat{w}\text{ shared by all }K\text{ outcomes}\tag{4}$$
$$
\tilde{\tau}_{1t,k} = Y_{1t,k} - \sum_{j=2}^{J+1}\tilde{w}^{(k)}_j\,Y_{jt,k}
\qquad K\text{ separate weight vectors}\tag{5}
$$

Eq (4) is our shared-weight arm; eq (5) is our one-fit-per-channel arm, which solves K
separate simplex problems. Same donors, same periods, same target, same solver — the *only*
difference is whether the weights are shared. So the S1 chip-mean pair **0.0204 (shared)
against 0.0177 (separate)** is the paper's own contrast, measured on our data.

**Why sharing is supposed to help.** Proposition 1, with J fixed and T₀, K increasing:

$$
\bigl|\,\mathbb{E}(\tilde{\tau}_{1t,k}) - \tau_{1t,k}\,\bigr| = O\!\left(\frac{1}{\sqrt{T_0}}\right)
\qquad
\bigl|\,\mathbb{E}(\hat{\tau}_{1t,k}) - \tau_{1t,k}\,\bigr| = O\!\left(\frac{1}{\sqrt{K\,T_0}}\right)
$$

For us K = 5, T₀ = 9: 45 cells of matching mass against 9. What improves is the *bias
rate*; nothing here adds parameters — J = 5 donors give 4 free weights under either
estimator (§4). That is why the multi-outcome gain is real but bounded.

**Does sharing help here? Measured with the corrections held equal.** Proposition 1 is a
statement about bias rates, not a guarantee at K = 5, T₀ = 9, J = 5. The demeaning of eq (6) and
the standardization of footnote 5 can be applied to either estimator, so the full 2 × 3 grid is
run: sharing (eq 4) against one fit per channel (eq 5), each with none, one, or both corrections,
everything else held fixed (chip mean, M2 at P10, mean over 10 sites):

| chip mean, M2 at P10 | S1 | S2 chip-mean fill | S2 historical fill | S2 masked pooling | S2 masked drop |
|---|---|---|---|---|---|
| one fit per channel, eq (5) | 0.0177 | 0.0389 | 0.0324 | **0.0349** | 0.0325 |
| one fit per channel + demeaned, eq (5) + eq (6) | 0.0132 | 0.0391 | 0.0300 | 0.0363 | **0.0324** |
| one fit per channel + demeaned + standardized, eq (5) + eq (6) + fn 5 | 0.0124 | **0.0366** | 0.0296 | 0.0365 | 0.0327 |
| shared weights, eq (4) | 0.0204 | 0.0462 | 0.0285 | 0.0425 | 0.0348 |
| shared + demeaned, eq (4) + eq (6) | **0.0114** | 0.0413 | 0.0289 | 0.0387 | 0.0336 |
| shared + demeaned + standardized, eq (4) + eq (6) + fn 5 | 0.0115 | 0.0377 | **0.0267** | 0.0384 | 0.0330 |

Three things are separable now. **Demeaning is the larger effect and it helps both families**: on
Sentinel-1 it takes one fit per channel from 0.0177 to 0.0132 and the shared fit from 0.0204 to
0.0114. **Sharing on its own loses** to one fit per channel in four of five columns (S1 0.0204
against 0.0177), so Proposition 1's rate does not simply materialise at this K, T₀ and J. **Sharing
on top of the same corrections** wins the two columns whose training rows are cleanest —
Sentinel-1 (0.0114 against 0.0124) and Sentinel-2 under historical fill (0.0267 against 0.0296) —
and loses the other three by at most 0.002 (S2 chip-mean fill 0.0377 against 0.0366, the two
masked arms 0.0384/0.0330 against 0.0365/0.0327). Every margin between the two families is
≤ 0.003, against a demeaning effect of 0.005–0.009 on Sentinel-1.

The reading is that at this K the pooling of matching mass matters less than the corrections that
put the treated site inside the donors' convex hull in the first place — which is exactly the
motivation the paper gives in §2.2 and footnote 5 — and that it pays only where the training
panel is clean enough for the pooled cells to agree with each other.

**Demeaning.** Where units differ by a stable offset, the paper (§2.2, following Ferman &
Pinto 2021 and Abadie 2021) replaces the outcomes in (3) with pre-period demeaned ones —
eq (6), for T₀ ≥ 2:

$$
\dot{Y}_{it,k} = Y_{it,k} - \frac{1}{T_0}\sum_{s=1}^{T_0} Y_{is,k}
\qquad
\hat{\tau}^{\,\mathrm{DM}}_{1t,k} = \dot{Y}_{1t,k} - \sum_{j=2}^{J+1}\hat{w}_j\,\dot{Y}_{jt,k}
\tag{6}
$$

and Corollary 1 keeps the $O(1/\sqrt{KT_0})$ rate. The purpose is stated plainly: it "enables
the SC to track the dynamics in the outcome of the treated unit over time, while allowing
the levels to differ by a constant amount, which is similar to the 'parallel trends'
assumption".

*Our implementation is eq (6) exactly*: the per-unit × per-outcome pre-period means are
subtracted from the treated site and from every donor, the weights are fitted on the demeaned
design, and the prediction adds the treated site's own pre-period mean back. That restore is
algebraically identical to the band-space code's step (synthetic series minus its pre-period
mean plus the treated pre-period mean), because the demeaned donor combination has zero
pre-period mean by construction. Demeaning is the single largest effect in this report:
S1 chip mean **0.0204 → 0.0114**.

**Dimensions — this study against the paper's and the band-space applications.**

| | this study (chip mean) | her Sentinel-2 Exp 2 | COVID, health domain | Germany |
|---|---|---|---|---|
| donors J | 5 | 5 | 25 | 16 |
| outcomes K | 5 tokenizer channels | 8 bands | 3 | 9 |
| pre-periods T₀ | 9 | 7 | 6 quarters | 1 (1989) |
| stacked cells K·T₀ | **45** | 56 | 18 | 9 |
| free weights J − 1 | **4** | 4 | 24 | 15 |
| pre-period mean handling | own pre-mean per site × channel, eq (6) | donor-pool mean & SD per band, training periods only | own pre-mean per country × outcome | none |

**In our data.** The K = 5 outcomes are the 5 TerraMind tokenizer channels of the chip mean;
the J = 5 donors are the site's own `counterfactual_XXXX_01…05`. What eq (6) subtracts is one
number per unit × channel, the average of that channel's chip mean over P01–P09; the demeaned
design is the deviation from it, negative when a period sits below the unit's own mean. Every
treated site has its own 5 donors and its own fit, so there are 10 weight vectors of length 5.

**Where our code differed from the paper, and what it cost.** Footnote 5 of the paper says the outcomes "may have
different scales or volatilities so that the transitory shocks may be clustered at the outcome
level", and that this "can be dealt with by standardizing each outcome in each period before
matching"; the band-space COVID implementation does it as its step 2, dividing each cell by the SD
taken **across the donors** at that outcome × period. Our demeaned arm did not. The
demeaned + standardized arm now does: after demeaning, each cell is divided by the cross-donor SD
(ddof = 1, as R's `sd`), zero-SD cells are dropped, and `V = diag(1/M_k)` is applied explicitly.

It behaves exactly as the paper's reasoning predicts. On **Sentinel-1** it changes nothing
(0.0114 → 0.0115): the 5 FSQ channels are the same quantity on the same [0, 1] scale, so there is
no scale clustering to remove. On **Sentinel-2** it is the largest single improvement to the
shared-weight arm in this report — 0.0413 → **0.0377** under chip-mean fill and 0.0289 → **0.0267**
under historical fill (the best chip-mean number in this report; under chip-mean fill the
per-channel version of the same corrections is lower still at 0.0366) — because there the channels'
*volatilities* differ, the cloud fill having disturbed some far more than others. Omitting it was a
real omission, not a stylistic one.

**On the V matrix, which is *not* a difference.** ("Difference" throughout §5 means a place
where our implementation does something other than what the paper or the band-space code does.)
Step 3 of the band-space implementation sets `v_{tk} = 1/M_k` with `M_k` the usable cells of
outcome k, and our objective weights every cell equally. These are the same estimator whenever
`M_k` is equal across k, since a constant factor does not move an argmin — and `M_k` *is* equal
here, because a missing period enters as an all-NaN vector and the solver drops the whole period
for the group. The per-site results confirm it: the number of fitted periods varies by site
(Sentinel-1 site 07 fits on 5 periods, masked drop on 4–7) but never by channel within a site.
The algebra is part of the pipeline's consistency checks (§13).

### 5.2 Aligned SCM — our construction

Not from a paper: this is our formulation of the request after the 27 August report that a
donor parcel should stand in only for a treated position of the same land type. It is stated
here because it is one of the two arms that move the number (S1 2×2 block 0.1724 → 0.1344;
S2 980-d 0.2822 → 0.2082 under historical fill).

**The problem.** The latent is position-indexed: coordinate `(c, r, col)` means "channel c of
the parcel at row r, column col **of this chip**". Matching it across sites therefore matches
one site's pasture to another's parking lot. *Same-position class agreement*: for a pair, read
the NLCD class at each of the 196 grid positions in both chips, count positions where the classes
are equal, divide by 196. Mean over the 50 pairs is **0.469** (range 0.066–0.939); for `treatment_0001` and its first donor 70 of 196
agree, **35.7 %**:

| NLCD class | treated 0001 | donor CF-01 |
|---|---|---|
| deciduous | 75 | 105 |
| pasture | 83 | 47 |
| mixed | 15 | 3 |
| dev-low | 10 | 7 |
| dev-open | 7 | 8 |
| evergreen | 4 | 0 |
| dev-med | 2 | 9 |
| dev-high | 0 | 6 |
| wood-wet | 0 | 8 |
| herb-wet | 0 | 3 |

**The assignment problem.** For treated position p and donor parcel q, with the NLCD class
a **hard** constraint:

$$
C[p,q] \;=\; \mathrm{BIG}\cdot\mathbf{1}\{\mathrm{cls}_t(p)\neq\mathrm{cls}_j(q)\}
\;+\; \alpha\, d^{2}_{\mathrm{phys}}(p,q)
\;+\; \beta\, d^{2}_{\mathrm{hist}}(p,q)
\;+\; \gamma\, d^{2}_{\mathrm{ctx}}(p,q)
$$

where `d²_phys` is the mean squared standardised difference in elevation and slope
(plus aspect sin/cos in the `full` variant), `d²_hist` the same in the parcel's mean
TerraMind history over the clean P01–P08 periods (plus its SD in `full`), and `d²_ctx` the
3 × 3 neighbourhood class histogram. `BIG = 1e6`. A permutation is then

$$
\pi \;=\; \arg\min_{\pi}\; \sum_{p} C[p,\pi(p)]
$$

with, per class c, $\max(0, n_t(c) - n_j(c))$ dummy columns added so that only *unavoidable*
positions go unmatched; any assignment still costing ≥ BIG is unmatched too. Parcels with
NLCD nodata or fewer than 3 clean pre-periods are excluded, and a donor with fewer than 60
matched positions is dropped for that site. The donor's 196 parcels are then re-ordered into
the treated layout (NaN at unmatched positions), and ordinary simplex SCM runs on the permuted
vector — **the outcome
is still the latent; the descriptors choose only the permutation**, and the permutation is
computed from P01–P08 alone, so it cannot see P10.

**What alignment does.** The latent is position-indexed, so coordinate p of the donor is "the
parcel at grid position p of the donor's chip". Unaligned SCM compares treated position p with
donor position p, whatever land types those are. Alignment first solves an assignment: for every
treated position p, pick one donor parcel q of the same class, minimizing terrain and history
distance, each donor parcel used at most once. Then the donor's 980-vector is rebuilt in the
treated layout: at position p we place the donor's parcel q, and NaN where no donor parcel could
be assigned. SCM then runs on those rebuilt donor vectors.

For the pair above all five arms match **141 of 196** positions: the class constraint is hard, and
the treated site has 55 parcels in classes the donor cannot supply (36 pasture, 12 mixed, 4
evergreen, 3 dev-low), so every arm leaves those 55 unmatched and fills the other 141. The arms
differ in which donor parcel goes to each position, not in how many are placed. Per site, matched
positions of 196 for donors CF-01 … CF-05 (identical across the five arms; Sentinel-1 / Sentinel-2,
the two differing through the three-clean-periods rule); a donor under 60 matched positions,
in bold, is dropped for that site and sensor:

| site | matched positions of 196, CF-01 … CF-05 (S1 / S2) | dropped |
|---|---|---|
| site 0001 | 141 / 141 · 104 / 104 · 101 / 101 · 142 / 142 · 133 / 133 | — |
| site 0002 | 110 / 98 · 94 / 82 · 103 / 77 · 83 / **57** · 125 / 99 | CF-04 (S2) |
| site 0003 | 176 / 171 · 182 / 182 · 174 / 167 · 176 / 176 · 122 / 122 | — |
| site 0004 | **59** / **58** · 150 / 108 · **54** / **54** · 134 / 133 · 61 / 60 | CF-01, CF-03 |
| site 0005 | **58** / **56** · 121 / 120 · 111 / 109 · 136 / 128 · 157 / 149 | CF-01 |
| site 0006 | 171 / 171 · 155 / 155 · 147 / 147 · 177 / 177 · 157 / 157 | — |
| site 0007 | 147 / 146 · 191 / 190 · 174 / 173 · 158 / 158 · 95 / 94 | — |
| site 0008 | 150 / 148 · 123 / 122 · 147 / 146 · 91 / 90 · 121 / 120 | — |
| site 0009 | 160 / 154 · 181 / 176 · 149 / 144 · 149 / 143 · 150 / 144 | — |
| site 0010 | 174 / 154 · 177 / 153 · 130 / 117 · 118 / 102 · 90 / 77 | — |

**The arms.**

| arm | α (physical) | β (TerraMind history) | γ (context) | variant |
|---|---|---|---|---|
| B | 0 | 0 | 0 | class constraint only |
| C | 1 | 0 | 0 | minimal |
| D_min | 1 | 1 | 0 | minimal — **the reported one** |
| D_full | 1 | 1 | 0 | full (adds aspect, history SD) |
| E | 1 | 1 | 1 | full + 3 × 3 class context |

Arm B isolates the hard class constraint: it does nothing on its own (S1 2×2 block 0.1757 vs
0.1724 unaligned). The gain comes from $\beta$ — the TerraMind history term, arm D_min.

---

## 6. Results — Sentinel-1, M2 at P10

**Read down a column, never across representations.** The scale of every number below is set by
the representation before any estimator is applied: the same own-history reference is 0.0183 on
chip mean, 0.1055 on the 2×2 block and 0.2097 on latent980, because a chip mean averages 196
parcels and a parcel's FSQ code moves in steps of 0.14–0.25 (§1, §2).

Cell = test RMSE (train RMSE in parentheses), P01–P09 → P10. Sentinel-1 site 07 fits on 5 of the 9
periods (three of its donors lack P02, P03, P05, P06). Both latent sets give identical Sentinel-1
numbers (1 of 1,147 latents differs), so the historical-fill column is shown once and not read.

| representation | estimator | chip-mean fill | historical fill |
|---|---|---|---|
| **chip mean (5)** | *reference — equal-weight donor average* | 0.0275 (0.0259) | 0.0275 (0.0259) |
|  | *reference — own-history mean* | 0.0183 (0.0165) | 0.0183 (0.0165) |
|  | cross-sectional SCM | 0.0204 (0.0169) | 0.0204 (0.0169) |
|  | one fit per channel (single-outcome SC, their eq 5) | 0.0177 (0.0124) | 0.0177 (0.0124) |
|  | one fit per channel + demeaning (their eq 5 + eq 6) | 0.0132 (0.0098) | 0.0132 (0.0098) |
|  | one fit per channel + demeaning + cell standardization (eq 5 + eq 6 + fn 5) | 0.0124 (0.0105) | 0.0124 (0.0105) |
|  | multi-outcome SC + demeaning (their eq 6) | 0.0114 (0.0113) | 0.0114 (0.0113) |
|  | multi-outcome SC + demeaning + cell standardization (their fn 5) | 0.0115 (0.0116) | 0.0115 (0.0116) |
| **Gram (15)** | *reference — equal-weight donor average* | 0.0233 (0.0230) | 0.0233 (0.0230) |
|  | *reference — own-history mean* | 0.0163 (0.0154) | 0.0163 (0.0154) |
|  | cross-sectional SCM | 0.0182 (0.0164) | 0.0182 (0.0164) |
|  | FSC (augmented, tuned λ) | 0.0178 (0.0162) λ=0.1 | 0.0178 (0.0162) λ=0.1 |
| **quantile (100)** | *reference — equal-weight donor average* | 0.0600 (0.0590) | 0.0600 (0.0590) |
|  | *reference — own-history mean* | 0.0491 (0.0447) | 0.0491 (0.0447) |
|  | cross-sectional SCM | 0.0572 (0.0546) | 0.0572 (0.0546) |
|  | FSC (augmented, tuned λ) | 0.0571 (0.0545) λ=1e+08↑ | 0.0571 (0.0545) λ=1e+08↑ |
| **combined (115)** | *reference — equal-weight donor average* | 0.0941 (0.0924) | 0.0941 (0.0924) |
|  | *reference — own-history mean* | 0.0761 (0.0693) | 0.0761 (0.0693) |
|  | cross-sectional SCM | 0.0887 (0.0845) | 0.0887 (0.0845) |
|  | FSC (augmented, tuned λ) | 0.0885 (0.0843) λ=100 | 0.0885 (0.0843) λ=100 |
| **2×2 block (245)** | *reference — equal-weight donor average* | 0.1745 (0.1790) | 0.1745 (0.1790) |
|  | *reference — own-history mean* | 0.1055 (0.0957) | 0.1055 (0.0957) |
|  | cross-sectional SCM | 0.1724 (0.1752) | 0.1724 (0.1752) |
|  | aligned SCM (D_min) | 0.1344 (0.1276) | 0.1344 (0.1276) |
| **latent980 (980)** | *reference — equal-weight donor average* | 0.3292 (0.3337) | 0.3292 (0.3337) |
|  | *reference — own-history mean* | 0.2097 (0.1904) | 0.2097 (0.1904) |
|  | cross-sectional SCM | 0.3288 (0.3305) | 0.3288 (0.3305) |
|  | aligned SCM (D_min) | 0.2438 (0.2271) | 0.2438 (0.2271) |

**Pass counts, Sentinel-1** (C2 = beats own history / C3 = beats equal-weight donors, sites of 10):

| representation | estimator | chip-mean fill C2 / C3 | historical fill C2 / C3 |
|---|---|---|---|
| **chip mean (5)** | cross-sectional SCM | 4 / 9 of 10 | 4 / 9 of 10 |
|  | one fit per channel (single-outcome SC, their eq 5) | 6 / 8 of 10 | 6 / 8 of 10 |
|  | one fit per channel + demeaning (their eq 5 + eq 6) | 8 / 9 of 10 | 8 / 9 of 10 |
|  | one fit per channel + demeaning + cell standardization (eq 5 + eq 6 + fn 5) | 8 / 9 of 10 | 8 / 9 of 10 |
|  | multi-outcome SC + demeaning (their eq 6) | 9 / 9 of 10 | 9 / 9 of 10 |
|  | multi-outcome SC + demeaning + cell standardization (their fn 5) | 9 / 9 of 10 | 9 / 9 of 10 |
| **Gram (15)** | cross-sectional SCM | 4 / 10 of 10 | 4 / 10 of 10 |
|  | FSC (augmented, tuned λ) | 4 / 9 of 10 | 4 / 9 of 10 |
| **quantile (100)** | cross-sectional SCM | 2 / 8 of 10 | 2 / 8 of 10 |
|  | FSC (augmented, tuned λ) | 2 / 8 of 10 | 2 / 8 of 10 |
| **combined (115)** | cross-sectional SCM | 2 / 8 of 10 | 2 / 8 of 10 |
|  | FSC (augmented, tuned λ) | 2 / 8 of 10 | 2 / 8 of 10 |
| **2×2 block (245)** | cross-sectional SCM | 0 / 10 of 10 | 0 / 10 of 10 |
|  | aligned SCM (D_min) | 0 / 10 of 10 | 0 / 10 of 10 |
| **latent980 (980)** | cross-sectional SCM | 0 / 8 of 10 | 0 / 8 of 10 |
|  | aligned SCM (D_min) | 0 / 10 of 10 | 0 / 10 of 10 |

Observations.

1. **Sentinel-1 is a site constant plus noise — measured, and it explains the C2 column.** On
   latent980 the RMSE between any two pre-periods of the same site is 0.2857, which under
   "constant + independent noise" implies a per-entry noise σ of 0.202, an own-history-mean error
   of $\sigma\sqrt{1 + 1/9} = 0.2129$ and a P09-alone error of $\sigma\sqrt{2} = 0.2857$; the observed values are 0.2097 and
   0.2775, within 3 %. So each fortnight's latent is the site's fixed image plus speckle that is
   independent across fortnights, no other fortnight of any site carries information about P10's
   speckle, and no predictor can go below ≈ 0.202. The own-history mean sits 4 % above that
   floor; every fitted arm adds bias on top (donor mismatch, or weights fitted to noise) and lands
   above the reference on almost every site. Hence C2 ≈ 0 for every S1 arm in the spatial
   representations, and own history beats every cross-sectional fit everywhere (chip mean
   0.0183 vs SCM 0.0204, Gram 0.0163 vs 0.0182, latent980 0.2097 vs 0.3288). The chip mean shows
   the flip side: averaging 196 parcels removes most speckle and leaves a shared fortnight
   component — consecutive periods are closer (0.0158) than independence predicts (0.0248) — which
   only concurrent donors can supply.
2. **Demeaning is the only cross-sectional construction that beats own history on S1.**
   Multi-outcome SC with within-site demeaning on the 5 channels reaches 0.0114 and passes C2 on
   9 of 10: it is the own-history correction plus a same-fortnight donor term.
3. **Alignment helps the whole-image latent but does not reach own history.** D_min cuts latent980
   from 0.3288 to 0.2438 and the block from 0.1724 to 0.1344; own history is 0.2097 / 0.1055.
4. **The tuned FSC changes nothing on S1.** With an interior penalty the augmented estimator
   improves on plain SCM by 0.0004 (Gram) and 0.0002 (combined); on the quantile the CV switches
   the augmentation off (λ at the ceiling), and the two rows coincide.

## 7. Results — Sentinel-2, M2 at P10

Same layout; four fill columns. Masked pooling and masked drop are defined only for the pooled
representations. Under masked drop a group keeps 4–7 of the 9 fit periods (mean 5.6).

| representation | estimator | chip-mean fill | historical fill | masked pooling | masked drop |
|---|---|---|---|---|---|
| **chip mean (5)** | *reference — equal-weight donor average* | 0.0490 (0.1172) | 0.0483 (0.0453) | 0.0477 (0.1135) | 0.0477 (0.0641) |
|  | *reference — own-history mean* | 0.0758 (0.1400) | 0.0665 (0.0310) | 0.0909 (0.1322) | 0.0767 (0.0569) |
|  | cross-sectional SCM | 0.0462 (0.0840) | 0.0285 (0.0322) | 0.0425 (0.0861) | 0.0348 (0.0497) |
|  | one fit per channel (single-outcome SC, their eq 5) | 0.0389 (0.0706) | 0.0324 (0.0277) | 0.0349 (0.0745) | 0.0325 (0.0418) |
|  | one fit per channel + demeaning (their eq 5 + eq 6) | 0.0391 (0.0649) | 0.0300 (0.0201) | 0.0363 (0.0690) | 0.0324 (0.0349) |
|  | one fit per channel + demeaning + cell standardization (eq 5 + eq 6 + fn 5) | 0.0366 (0.0729) | 0.0296 (0.0224) | 0.0365 (0.0760) | 0.0327 (0.0388) |
|  | multi-outcome SC + demeaning (their eq 6) | 0.0413 (0.0758) | 0.0289 (0.0231) | 0.0387 (0.0779) | 0.0336 (0.0393) |
|  | multi-outcome SC + demeaning + cell standardization (their fn 5) | 0.0377 (0.0965) | 0.0267 (0.0240) | 0.0384 (0.0898) | 0.0330 (0.0420) |
| **Gram (15)** | *reference — equal-weight donor average* | 0.0548 (0.1112) | 0.0535 (0.0480) | 0.0527 (0.1051) | 0.0527 (0.0683) |
|  | *reference — own-history mean* | 0.1003 (0.1338) | 0.0772 (0.0368) | 0.1159 (0.1229) | 0.0946 (0.0646) |
|  | cross-sectional SCM | 0.0522 (0.0871) | 0.0332 (0.0355) | 0.0484 (0.0845) | 0.0411 (0.0532) |
|  | FSC (augmented, tuned λ) | 0.0522 (0.0871) λ=1e+08↑ | 0.0311 (0.0337) λ=0.1 | 0.0484 (0.0845) λ=1e+08↑ | 0.0411 (0.0532) λ=1e+08↑ |
| **quantile (100)** | *reference — equal-weight donor average* | 0.0859 (0.1493) | 0.0844 (0.0851) | 0.0845 (0.1468) | 0.0845 (0.1048) |
|  | *reference — own-history mean* | 0.1511 (0.1689) | 0.1139 (0.0595) | 0.1412 (0.1659) | 0.1187 (0.0910) |
|  | cross-sectional SCM | 0.0857 (0.1270) | 0.0709 (0.0754) | 0.0818 (0.1282) | 0.0745 (0.0925) |
|  | FSC (augmented, tuned λ) | 0.0856 (0.1270) λ=1e+08↑ | 0.0699 (0.0748) λ=1 | 0.0817 (0.1281) λ=1e+08↑ | 0.0745 (0.0924) λ=1e+08↑ |
| **combined (115)** | *reference — equal-weight donor average* | 0.1228 (0.2174) | 0.1177 (0.1169) | 0.1224 (0.2157) | 0.1195 (0.1484) |
|  | *reference — own-history mean* | 0.2172 (0.2483) | 0.1600 (0.0825) | 0.2126 (0.2450) | 0.1735 (0.1304) |
|  | cross-sectional SCM | 0.1220 (0.1836) | 0.0965 (0.1025) | 0.1176 (0.1870) | 0.1041 (0.1298) |
|  | FSC (augmented, tuned λ) | 0.1219 (0.1835) λ=1e+08↑ | 0.0951 (0.1017) λ=10 | 0.1175 (0.1870) λ=1e+08↑ | 0.1040 (0.1297) λ=1e+08↑ |
| **2×2 block (245)** | *reference — equal-weight donor average* | 0.1860 (0.1788) | 0.1851 (0.1686) | — | — |
|  | *reference — own-history mean* | 0.1536 (0.1738) | 0.1384 (0.0855) | — | — |
|  | cross-sectional SCM | 0.1893 (0.1647) | 0.1814 (0.1633) | — | — |
|  | aligned SCM (D_min) | 0.1392 (0.1544) | 0.1355 (0.1141) | — | — |
| **latent980 (980)** | *reference — equal-weight donor average* | 0.2840 (0.2283) | 0.2834 (0.2428) | — | — |
|  | *reference — own-history mean* | 0.2250 (0.2048) | 0.2076 (0.1242) | — | — |
|  | cross-sectional SCM | 0.2885 (0.2202) | 0.2822 (0.2400) | — | — |
|  | aligned SCM (D_min) | 0.2135 (0.1955) | 0.2082 (0.1539) | — | — |

**Pass counts, Sentinel-2:**

| representation | estimator | chip-mean fill C2 / C3 | historical fill C2 / C3 | masked pooling C2 / C3 | masked drop C2 / C3 |
|---|---|---|---|---|---|
| **chip mean (5)** | cross-sectional SCM | 8 / 4 of 10 | 9 / 9 of 10 | 8 / 4 of 10 | 10 / 7 of 10 |
|  | one fit per channel (single-outcome SC, their eq 5) | 9 / 7 of 10 | 9 / 7 of 10 | 10 / 7 of 10 | 10 / 7 of 10 |
|  | one fit per channel + demeaning (their eq 5 + eq 6) | 9 / 7 of 10 | 10 / 7 of 10 | 10 / 6 of 10 | 10 / 7 of 10 |
|  | one fit per channel + demeaning + cell standardization (eq 5 + eq 6 + fn 5) | 9 / 8 of 10 | 10 / 7 of 10 | 10 / 7 of 10 | 10 / 6 of 10 |
|  | multi-outcome SC + demeaning (their eq 6) | 9 / 7 of 10 | 10 / 7 of 10 | 10 / 6 of 10 | 10 / 6 of 10 |
|  | multi-outcome SC + demeaning + cell standardization (their fn 5) | 10 / 6 of 10 | 10 / 7 of 10 | 10 / 6 of 10 | 10 / 7 of 10 |
| **Gram (15)** | cross-sectional SCM | 8 / 4 of 10 | 9 / 9 of 10 | 9 / 3 of 10 | 10 / 8 of 10 |
|  | FSC (augmented, tuned λ) | 8 / 4 of 10 | 9 / 9 of 10 | 9 / 3 of 10 | 10 / 8 of 10 |
| **quantile (100)** | cross-sectional SCM | 10 / 3 of 10 | 10 / 7 of 10 | 10 / 4 of 10 | 10 / 8 of 10 |
|  | FSC (augmented, tuned λ) | 10 / 3 of 10 | 10 / 7 of 10 | 10 / 4 of 10 | 10 / 8 of 10 |
| **combined (115)** | cross-sectional SCM | 10 / 3 of 10 | 9 / 7 of 10 | 10 / 4 of 10 | 10 / 8 of 10 |
|  | FSC (augmented, tuned λ) | 10 / 3 of 10 | 9 / 7 of 10 | 10 / 4 of 10 | 10 / 8 of 10 |
| **2×2 block (245)** | cross-sectional SCM | 2 / 4 of 10 | 2 / 8 of 10 | — | — |
|  | aligned SCM (D_min) | 9 / 8 of 10 | 7 / 9 of 10 | — | — |
| **latent980 (980)** | cross-sectional SCM | 0 / 3 of 10 | 0 / 8 of 10 | — | — |
|  | aligned SCM (D_min) | 7 / 9 of 10 | 7 / 10 of 10 | — | — |

Observations.

1. **The fill is the largest single effect on Sentinel-2, in every pooled representation.**
   Chip-mean → historical fill: SCM on chip mean 0.0462 → 0.0285, Gram 0.0522 → 0.0332, quantile
   0.0857 → 0.0709, combined 0.1220 → 0.0965. C3 on chip mean and Gram goes from 4 of 10 to 9 of
   10: with a clean history the fitted weights finally beat the equal-weight average. Train error
   falls further than test (0.0840 → 0.0322 on chip mean), which locates the damage: chip-mean
   fill corrupts the *cloudy training fortnights*, and a fit on corrupted rows learns weights that
   do not transfer.
2. **Masked drop is the second-best fill; masked pooling barely helps.** Dropping the near-fully
   cloudy chips from the fit gives SCM 0.0348 on chip mean and 0.0411 on Gram (C3 7–8 of 10);
   pooling over valid parcels only, with the cloudy chips kept, gives 0.0425 / 0.0484 (C3 3–4 of
   10). The cloudy chips do damage as *rows*, not as *pixels*.
3. **The FSC augmentation is chosen only under historical fill.** There the CV returns an
   interior penalty (Gram λ = 0.1, quantile 1, combined 10) and the augmented estimator improves
   on plain SCM by 6 % (Gram 0.0332 → 0.0311), 1.4 % (quantile) and 1.5 % (combined). Under
   chip-mean fill, masked pooling and masked drop the CV puts λ at the ceiling of a 16-decade
   grid, i.e. it switches the augmentation off, and the FSC row equals the SCM row (§9 records
   what this supersedes).
4. **Whole-image S2: alignment and own history converge under historical fill.** On chip-mean
   latents aligned donors win (latent980 0.2135 vs own history 0.2250); with historical fill
   own history 0.2076 and aligned donors 0.2082 are within 1 % of each other.
5. **Alignment ablation** (chip-mean fill; D_min is the reported arm): class-only (B) and
   class + terrain (C) do nothing on S1 (0.3309 / 0.3272 vs 0.3288 unaligned) and cut S2 by
   6–8 % (0.2669 / 0.2704 vs 0.2885); adding the TerraMind history term (D_min) is what produces
   the gain (S1 0.2438, S2 0.2135); the richer descriptors D_full and E add nothing
   (S1 0.2540 / 0.2569, S2 0.2269 / 0.2240). Within-class *history similarity* drives the
   alignment, not the class constraint.

## 8. Per-site detail

Best by-design estimator of each representation (chosen on the chip-mean column), chip-mean /
historical fill, M2 at P10. Sentinel-2 valid-parcel count of the P10 target alongside (all 196:
P10 is the clean fortnight, so the per-site spread is not cloud in the *target*).

**Sentinel-1**

| site | chip mean (5)<br>multi-outcome SC + demeaning<br>chip / hist | Gram (15)<br>FSC (augmented, tuned λ)<br>chip / hist | quantile (100)<br>FSC (augmented, tuned λ)<br>chip / hist | combined (115)<br>FSC (augmented, tuned λ)<br>chip / hist | 2×2 block (245)<br>aligned SCM (D_min)<br>chip / hist | latent980 (980)<br>aligned SCM (D_min)<br>chip / hist |
|---|---|---|---|---|---|---|
| site 0001 | 0.012 / 0.012 | 0.022 / 0.022 | 0.061 / 0.061 | 0.096 / 0.096 | 0.149 / 0.149 | 0.270 / 0.270 |
| site 0002 | 0.010 / 0.010 | 0.016 / 0.016 | 0.053 / 0.053 | 0.083 / 0.083 | 0.147 / 0.147 | 0.278 / 0.278 |
| site 0003 | 0.008 / 0.008 | 0.009 / 0.009 | 0.047 / 0.047 | 0.072 / 0.072 | 0.130 / 0.130 | 0.253 / 0.253 |
| site 0004 | 0.020 / 0.020 | 0.026 / 0.026 | 0.064 / 0.064 | 0.101 / 0.101 | 0.142 / 0.142 | 0.222 / 0.222 |
| site 0005 | 0.017 / 0.017 | 0.014 / 0.014 | 0.055 / 0.055 | 0.084 / 0.084 | 0.135 / 0.135 | 0.250 / 0.250 |
| site 0006 | 0.009 / 0.009 | 0.021 / 0.021 | 0.066 / 0.066 | 0.102 / 0.102 | 0.138 / 0.138 | 0.275 / 0.275 |
| site 0007 | 0.006 / 0.006 | 0.018 / 0.018 | 0.064 / 0.064 | 0.099 / 0.099 | 0.116 / 0.116 | 0.220 / 0.220 |
| site 0008 | 0.013 / 0.013 | 0.025 / 0.025 | 0.059 / 0.059 | 0.093 / 0.093 | 0.132 / 0.132 | 0.233 / 0.233 |
| site 0009 | 0.010 / 0.010 | 0.013 / 0.013 | 0.052 / 0.052 | 0.080 / 0.080 | 0.126 / 0.126 | 0.225 / 0.225 |
| site 0010 | 0.008 / 0.008 | 0.015 / 0.015 | 0.049 / 0.049 | 0.077 / 0.077 | 0.129 / 0.129 | 0.211 / 0.211 |

**Sentinel-2**

| site | chip mean (5)<br>one fit per channel + demeaning + cell standardization<br>chip / hist | Gram (15)<br>FSC (augmented, tuned λ)<br>chip / hist | quantile (100)<br>FSC (augmented, tuned λ)<br>chip / hist | combined (115)<br>FSC (augmented, tuned λ)<br>chip / hist | 2×2 block (245)<br>aligned SCM (D_min)<br>chip / hist | latent980 (980)<br>aligned SCM (D_min)<br>chip / hist | valid parcels P10 |
|---|---|---|---|---|---|---|---|
| site 0001 | 0.032 / 0.022 | 0.048 / 0.038 | 0.087 / 0.078 | 0.122 / 0.105 | 0.156 / 0.159 | 0.222 / 0.223 | 196 |
| site 0002 | 0.044 / 0.034 | 0.076 / 0.047 | 0.090 / 0.077 | 0.133 / 0.108 | 0.142 / 0.141 | 0.221 / 0.216 | 196 |
| site 0003 | 0.018 / 0.027 | 0.014 / 0.018 | 0.059 / 0.051 | 0.080 / 0.069 | 0.096 / 0.092 | 0.173 / 0.170 | 196 |
| site 0004 | 0.030 / 0.030 | 0.096 / 0.055 | 0.115 / 0.090 | 0.171 / 0.126 | 0.171 / 0.174 | 0.242 / 0.240 | 196 |
| site 0005 | 0.068 / 0.035 | 0.099 / 0.034 | 0.133 / 0.084 | 0.194 / 0.117 | 0.143 / 0.134 | 0.211 / 0.195 | 196 |
| site 0006 | 0.014 / 0.022 | 0.041 / 0.012 | 0.074 / 0.057 | 0.104 / 0.075 | 0.137 / 0.118 | 0.198 / 0.185 | 196 |
| site 0007 | 0.046 / 0.040 | 0.052 / 0.036 | 0.071 / 0.069 | 0.102 / 0.095 | 0.123 / 0.122 | 0.205 / 0.204 | 196 |
| site 0008 | 0.033 / 0.030 | 0.029 / 0.026 | 0.074 / 0.067 | 0.102 / 0.090 | 0.137 / 0.135 | 0.214 / 0.203 | 196 |
| site 0009 | 0.039 / 0.037 | 0.048 / 0.017 | 0.084 / 0.059 | 0.117 / 0.078 | 0.147 / 0.142 | 0.239 / 0.240 | 196 |
| site 0010 | 0.043 / 0.019 | 0.019 / 0.027 | 0.070 / 0.067 | 0.095 / 0.089 | 0.140 / 0.138 | 0.209 / 0.207 | 196 |

Historical fill improves 8 of the 10 sites on chip mean, 8 on Gram and all 10 on quantile and
combined; the exceptions are sites 03 and 06 on chip mean and sites 03 and 10 on Gram. Site 05 is
the largest beneficiary in all four pooled representations (chip mean 0.068 → 0.035, Gram
0.099 → 0.034, quantile 0.133 → 0.084, combined 0.194 → 0.117). On Sentinel-1 the per-site
spread is about 3× in the pooled representations (chip mean 0.006–0.020) and under 1.3× in the
spatial ones (latent980 0.211–0.278).

## 9. Changes since the 27 August report

**The FSC label.** The rows labelled "FSC on Gram / histogram / hist+gram" in the 27 August
report were plain simplex SCM: Okano–Kurisu's un-augmented `FSCM()` solves the same constrained
least squares, and the stored weights from their R code equal ours on every stored group to
1.1e-4 (their `round(w, 4)` plus a 1e-6 ridge; §13). Those rows are named *cross-sectional SCM*
here. The estimator that carries the FSC label is the **augmented** one (`FSCM_aug_covmat`),
which that report had run with its penalty pinned at 0.99993 on a (0, 1) search interval,
truncated rather than tuned (the audit is in `8-27-fsc-fidelity.md`). It is rerun here with
their own CV objective on a log grid from 10⁻⁸ to 10⁸, extended downward when the optimum sits
on the floor; every FSC cell in §6–§7 carries its λ and a ceiling mark where the CV switched the
augmentation off. The augmented-FSC numbers of the 27 August report are superseded by §6–§7.

**The three-family tie.** The closing claim of the 27 August report, that three estimator
families tie and the remedy is therefore a richer donor pool or representation rather than a
better estimator, no longer holds. Two of the three families were the same optimizer under two
names, and the third ran untuned. The re-test with the tuned estimator (§6–§7, items 3–4) is
that the augmentation is worth 0–6 % on this panel and is switched off by the CV wherever the
training rows are cloud-corrupted; that is a measured statement about the estimator, not a
tautology.

**One implementation departure, found in the derivation of §5 and now measured.** Deriving the
arm from its source turned up a step of the published method that our implementation did not
carry out: *multi-outcome SC without the cell standardization.* Tian–Lee–Panchenko's footnote 5
and step 2 of the band-space COVID implementation divide each outcome × period cell by its
cross-donor SD before matching; our demeaned arm did not. It is added as the demeaned +
standardized arm (§5.1), run over the whole grid and reported as its own row. It was not a silent
bug in the numbers already published, but neither was the published method.

Two things the same derivation *confirmed*: the demeaned arm is their eq (6) exactly, pre-period
mean added back included; and the weighting matrix V is **not** a departure. The band-space V = diag(1/M_k) is
proportional to our implicit identity whenever every outcome has the same usable-cell count,
which holds here because a missing period is an all-NaN vector and the solver drops the whole
period for the group (the number of fitted periods varies by site but never by channel within a
site). Both are consistency checks of the pipeline (§13).

**Other changes.** The class-conditional mean-shift of donor latents ("moving pixels in
embedding space", §6 of the 28 August draft) was scored in the old standardized metric and is
not recomputed; its verdict, inert without alignment and redundant with it, is inherited.
Experiment 1's per-parcel arm (K = 10 parcels, ridge λ = 1) was dropped from the by-design grid.
The combined representation is now scaled per block as its definition requires (each block to
unit mean square, once over the set), so its SCM numbers differ from the 31 August draft by
≤ 0.003; every other chip-mean number reproduces that draft's relative-L₂ column to machine
precision.

## 10. What can be decoded: the reconstruction floor (M1)

The latent is FSQ-quantized and TerraMind can decode it, so the decisive question for any
latent-space causal claim is what survives the round trip. Decoding the **observed** treated
latent and comparing its band means to the true chip gives a floor no latent-space method can
beat. Reported in **M1**, since decoding produces band means:

| method (decoded → band means, P10) | S2 | S1 |
|---|---|---|
| **reconstruction floor** (observed latent decoded back) | **0.0758** | **0.9004** |
| synthetic, weights from band means | 0.0846 | 0.5796 |
| synthetic, weights from TerraMind 5 ch | 0.0823 | 0.6679 |
| synthetic, weights from latent980 | 0.0846 | 0.9445 |
| *(reference)* direct band-mean SCM, no decode | *0.0206* | *0.1463* |

The floor is **3.7× the direct band SCM error on Sentinel-2** and **6× on Sentinel-1**.
Consequence: **no latent-space synthetic control can produce competitive band-space
counterfactuals through the decoder.** The latent is usable as an outcome space, not as a route
back to physical units. One observation should not be over-read: on S1 some synthetics decode
better than the floor, because averaging donor latents smooths FSQ and speckle noise. A weighted
average of quantized codes is not itself a valid code, so decoder behaviour on it carries no
guarantee.

## 11. What is established and what is not

**Established.**
1. *Position must be handled, and one mechanism explains all of it.* Same-position class
   agreement between a treated site and its donors averages 0.469 over the 50 pairs (range
   0.066–0.939), so raw parcel-to-parcel comparison is structurally wrong. Supplying the site's own mean image — explicitly (the own-history
   mean), implicitly (within-site demeaning), or approximately (class-and-history alignment) — is
   what recovers performance in every arm, on both sensors.
2. *Alignment works, and history similarity is what drives it.* D_min cuts whole-image error by
   26 % on both sensors (S1 0.3288 → 0.2438, S2 0.2885 → 0.2135); the class constraint alone does
   nothing.
3. *Historical cloud fill is the largest Sentinel-2 effect in every pooled representation,* and
   on Sentinel-2 it is the only one of the four fills under which the FSC augmentation is
   selected by its own CV (λ = 0.1 / 1 / 10 for Gram / quantile / combined; every other
   Sentinel-2 cell rails at the grid ceiling). On Sentinel-1 the CV selects it for Gram
   (λ = 0.1) and combined (λ = 100) regardless of fill, and it is worth ≤ 0.0004 there.
4. *Sentinel-1 carries almost no cross-sectional donor signal:* in five of the six
   representations — Gram, quantile, combined, 2×2 block, latent980 — the own-history mean beats
   **every** donor-based fit, aligned and unaligned alike. The one exception is chip mean, where
   five arms beat it: one fit per channel (0.0177, C2 6/10), the same with demeaning (0.0132) and
   with demeaning + standardization (0.0124, both C2 8/10), and the two shared-weight arms that
   remove the site's own pre-period mean, demeaned (0.0114, C2 9/10) and demeaned + standardized
   (0.0115, C2 9/10). Those arms are own history *plus* a donor term.
**Not established.**
1. *No representation is demonstrably better than another.* Differences of a few per cent across
   10 sites are unresolvable without a paired test or a placebo distribution; the pass counts are
   descriptive.
2. *Latent-space accuracy does not imply better physical counterfactuals* (§10).
3. *No post-period effect has been shown to exceed the placebo floor*, unchanged since the 27 August report.
4. *Cross-representation rows are not a ranking.*

## 12. Proposed next steps

1. Paired significance test across the 10 sites, plus a placebo distribution over control sites,
   for the within-representation differences in §6–§7 (in particular: FSC vs SCM under
   historical fill; masked drop vs chip-mean fill).
2. Effect-detection comparison on P11–P20: which representation best separates treated sites
   from placebos — the question pre-period fit cannot answer.
3. Re-run the aligned arm without the 60-parcel donor-drop rule, to separate alignment's effect
   from the donor attrition it causes.
4. FSC prediction intervals (the interval procedure of the FSC reference code), still open.
5. The cell standardization helps Sentinel-2 chip mean and does nothing on Sentinel-1; it has
   only been run on the chip-mean representation, where multi-outcome SC is defined — whether the
   same correction helps the Gram and quantile arms is untested.
6. Out of scope for the present plan: stage-A donor re-selection from the 75k grid (needs new
   chip downloads), soft OT, VLM descriptors.

## 13. Reproducibility

Every number in §5–§9 is produced by a script and read from its output table; nothing is typed
by hand. Paths are relative to `Satellite/notebooks/ts_SCM_ASCM/` unless stated.

| script / artifact | produces | read by |
|---|---|---|
| `panel_metric2.py` | `panel_metric2_p10.csv` (means), `panel_metric2_sites.csv` (per site), `panel_metric2_afsc_lambda.csv` (FSC penalty search) | §5.1 tables, §6–§8 |
| `panel_align.py` | `panel_align_agreement.csv` (same-position class agreement), the permutations and matched-position counts of §5.2 | §5.2 |
| `panel_band_compare.py` | reproduction of the band-space pipeline on our copy of the data | Scope |
| notebooks `12`–`14`, `16` | alignment descriptors and aligned SCM (12, 13), historical-fill cache (14), multi-outcome SC on the 5 channels (16) | §2, §5 |
| `data/embeddings_tok_panel/latents_biweekly.npz`, `latents_biweekly_histfill.npz` | the chip-mean-fill and historical-fill latent sets | §3 |
| `Docs/8-27-fsc-fidelity.md` | the FSC audit trail | §9 |

Consistency checks run inside the scripts and stop the run on failure:

1. The multi-outcome demeaning equals Tian–Lee–Panchenko's eq (6) with the pre-period mean added
   back, and the V-matrix algebra of §5.1 holds (equal usable-cell counts across channels).
2. The stored weights of the FSC reference R code equal ours on every stored group to 1.1e-4.
