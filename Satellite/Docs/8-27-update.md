# Synthetic control in TerraMind latent space: from position-matched failure to functional SC

**Report, 2026-08-26** (restructured from the 8-25/26 working log; superseded working
notes are in git history). Companion documents: `8-27-fsc-fidelity.md` (fidelity audit of
the FSC implementation), notebooks `ts_SCM_ASCM/01–09`, result tables
`panel_fsc_prediction.csv`, `panel_by_model_ratio.csv`, `panel_fsc_lambda.csv`.

**What this report shows.** Synthetic control fitted directly on TerraMind's 980-d
spatial latent fails because the comparison is parcel-to-parcel across sites — a
representation problem, not an estimator problem. Pooling each chip into a
position-free object (a token histogram or a channel Gram matrix) and running Okano &
Kurisu's functional synthetic control (their unmodified R code) fixes the representation
failure: pooled models beat the site's own history on Sentinel-2 where no 980-d model
does. But decomposing the value chain shows the predictive power comes from observing
untreated sites *in the same fortnight* — the weight optimization, the covariate
matching, and every augmentation scheme add approximately nothing — and no post-period
hurricane effect exceeds the placebo noise floor. Generation-based cloud fill, helpful
for image-space tasks, *damages* the pooled fit.

---

## 1. Why not parcel-to-parcel comparison

**The dimension chain** (definitions used throughout):

| object | what it is | size |
|---|---|---|
| **site** | one location — 1 km × 1 km of ground, a 101 × 101 chip of 10 m pixels | 60 sites (10 treated, 50 controls) |
| **chip** | one image: site × sensor × 14-day period | ≤ 2,400; 2,322–2,357 exist |
| **parcel** | TerraMind resizes the chip to 224 × 224 and cuts it into a **14 × 14 grid of 196 patches**; each patch covers ≈ **72 m × 72 m** of ground | 196 per chip |
| **latent** | **5 numbers per parcel** (the tokenizer's channels) → 5 × 196 = **980** per chip; coordinate (c, r, col) = "channel c of the parcel at row r, column col *of this chip*" | 980-d |

Both stages of the original pipeline compare parcel (r, col) of one site with parcel
(r, col) of another: donor selection subtracts flattened 980-vectors elementwise
(`panel_lib.knn_donors`), and the weight fit minimizes the elementwise residual over the
same flat index (`Panel.design` + `scm_fit`). The Euclidean distance on this embedding is
position-matched *because the embedding is a spatial grid*.

That is the wrong comparison, for a measured reason: the token is a strong **local**
descriptor of its own parcel — on clean chips, 65–82 % of within-chip variation in
NDVI/NDWI/NDBI is linearly recoverable from the 5 numbers at that parcel (§4.1). So
matching coordinate (3,7) across sites genuinely matches one site's parking lot against
another site's forest. The measured cost: Sentinel-2's apparent in-sample fit is
**alignment overfitting** — destroying donor parcel arrangement while preserving
composition (a fixed random permutation of each donor's 196 parcels) moves the training
error from 0.820 to 1.114, i.e. the entire in-sample advantage over the no-fit baseline
is spatial alignment, and it does not survive the P09/P10 holdout (1.05). Sentinel-1 has
no alignment advantage to lose (+0.037). This settles the design question: the latent
must be pooled before any cross-site comparison.

## 2. Why the multiple-outcome SC framework does not apply

Tian, Lee & Panchenko (*Synthetic Controls with Multiple Outcomes*, Econometrics Journal
2026, [arXiv:2304.02272](https://arxiv.org/abs/2304.02272)) stack K outcomes × T₀
pre-periods with one shared simplex weight vector, improving bias from O(1/√T₀) to
O(1/√(KT₀)). Our 980-d estimator has exactly that *shape* (stacked coordinates, shared
simplex weights) — but the shape is not the method:

| their requirement | our situation | |
|---|---|---|
| simplex weights, one shared w, stacked over k × t, weighted least squares (their Condition 3) | `scm_fit`, identity weighting | ✓ |
| Condition 3 *also* matches predictors, Σⱼwⱼ Zⱼ = Z₁ | no covariates in the objective; donors from latent kNN only | ✗ half |
| demeaning per unit per outcome over pre-periods (their §2.2) | pooled z-scoring across all 60 sites | ✗ different |
| **Condition 1**: shocks ε_itk independent across i, t **and k** | fails: the latent covariance has effective dimension ≈ 28 (S2) / ≈ 132 (S1) of the nominal 980, so the promised O(1/√(KT₀)) reduction was never available — and pooling to 15–35 dimensions (§3) sacrifices little of what is actually there | ✗ |
| **Condition 3**: perfect pre-treatment fit — treated inside the donors' convex hull | fails: S1 training RMSPE 1.058 (the donors cannot fit the target even in-sample); a 5-donor hull is a 4-simplex inside ℝ⁷⁸⁴⁰ | ✗ |
| asymptotics: fixed J, **large T₀ and K** | T₀ = 8, fixed | ✗ |
| loadings μᵢ **common across outcomes** — a "domain" is "a collection of related outcomes driven by the same set of observed and unobserved predictors" (their example: GDP, industrial production, retail sales, CPI) | our 980 coordinates are grid positions of one image, each loading site-specific geography — not a domain of related series | ✗ |

Their Proposition 1 was therefore never in force; no guarantee was violated because none
applied. What the paper contributes here is the *language* for the failure (outcome-
specific loadings destroy the stacking gain) and the checklist that motivated pooling.

## 3. The pooled representations: histogram and Gram

Both are built **from the tokenizer output** — drop the encoder and we would simply
reproduce the collaborator's feature pipeline. Input per chip: the (5 × 196) array `A`.
Both belong to the *orderless pooling* family of computer vision (bag-of-visual-words /
codeword histograms; bilinear and covariance pooling), and each is a metric-space
outcome in Okano & Kurisu's functional SC ([arXiv:2601.07539](https://arxiv.org/abs/2601.07539)).

### 3.1 The histogram — *how much of each thing is present*

The channels are quantized to fixed FSQ lattices of **8, 8, 8, 6, 5 levels** on [0, 1]
(§4.1), so each channel's distribution over its 196 parcels is discrete on ≤ 8 atoms.
Per channel, count how many parcels take each level. Real chip
(`treatment_0001`, S2, P01):

```
channel 0:  0.14→  2   0.29→ 18   0.43→ 34   0.57→ 85   0.71→ 44   0.86→ 12   1.00→  1
channel 1:  0.00→  1   0.14→  1   0.29→  9   0.43→ 32   0.57→ 65   0.71→ 40   0.86→ 25   1.00→ 23
channel 2:  0.00→  2   0.29→  3   0.43→ 10   0.57→ 37   0.71→ 63   0.86→ 46   1.00→ 35
channel 3:  0.00→  2   0.20→ 16   0.40→ 29   0.60→ 62   0.80→ 75   1.00→ 12
channel 4:  0.00→  3   0.25→ 24   0.50→ 77   0.75→ 66   1.00→ 26          (each sums to 196)
```

**35 numbers**; parcel positions discarded — the point. Two sites with different maps
but the same mix of ground cover become comparable.

*FSC mapping:* this is Okano–Kurisu **example 2** — 1-D distributions embedded
isometrically into L²([0,1]) via the quantile function Ψ(μ) = F⁻¹ under the 2-Wasserstein
metric. Because the FSQ lattice makes each quantile function a step function with ≤ 8
jumps, **the quantile representation and the codeword histogram are the same object for
this encoder**; the pipeline evaluates the quantile function on a grid (20
points/channel, convergence-checked against their 100-point grid, differences ≤ 0.03).
The load-bearing assumption: monotone functions form a **convex cone**, so a
simplex-weighted average of quantile functions is itself a quantile function — the
Wasserstein barycenter. *Honest caveats:* concatenating the 5 channels is a direct sum
of Hilbert embeddings (valid isometry under the product metric) but treats the 5
marginals, not the joint 5-D distribution — a *sliced* Wasserstein treatment, our
extension, not in their code.

### 3.2 The Gram matrix — *what fires together at the same parcel*

$$G[a,b] = \frac{1}{196}\sum_{p=1}^{196}(\text{channel } a \text{ at parcel } p)\times(\text{channel } b \text{ at parcel } p)$$

Same chip:

```
          ch0    ch1    ch2    ch3    ch4
  ch0   0.346  0.370  0.408  0.371  0.336
  ch1   0.370  0.457  0.491  0.430  0.388
  ch2   0.408  0.491  0.587  0.466  0.464
  ch3   0.371  0.430  0.466  0.447  0.370
  ch4   0.336  0.388  0.464  0.370  0.428
```

**15 unique numbers** (symmetric, stored as the upper triangle including the diagonal in
column-major order — exactly the `covvec` convention of their `service.R`). The diagonal
is overall channel activity; the off-diagonal says whether two channels fire **on the
same parcels**.

*FSC mapping:* Okano–Kurisu **example 3** — symmetric PSD matrices, a closed convex
cone, so a simplex average is a valid covariance; augmented (negative-weight) outcomes
are projected back with `Matrix::nearPD`. *Honest caveat:* the unweighted covvec L²
metric counts off-diagonals once where true Frobenius counts them twice — faithful to
**their own code**, not to a textbook Frobenius isometry.

### 3.3 What each captures, what each misses

- **Same histogram, different Gram:** a *marsh* (every parcel green **and** wet) vs a
  *forest beside a lake* (half green-dry, half bare-wet) — identical totals, so identical
  histograms; G[green, wet] high for the marsh, near zero for the forest. Flooding puts
  water where vegetation was: the co-occurrence entry moves even when totals barely do.
- **Same Gram, different histogram:** half the site destroyed (0.8) + half untouched (0)
  vs the whole site moderately damaged (0.566) — identical second moments (0.32), but the
  histogram separates "one neighbourhood wiped out" from "everything lightly hit".
- Neither says *where*. Both discard geometry entirely.

**Measured payoff** (5-fold CV over 40 splits, 130 clean S2 chips, predicting chip-level
spectral quantities from each summary): on *levels* the 5-number channel mean is already
enough (NDVI mean R² 0.916 vs 0.907 histogram); on *within-chip dispersion* the richer
summaries win decisively (NDVI sd 0.544 → **0.840** histogram; NDWI sd 0.527 → 0.740;
NDVI p10 0.779 → 0.890). The extra content of the pooled representations lives in
heterogeneity, not levels — a fact §8 returns to.

## 4. The FSC pipeline and technical details

### 4.1 Input data: the TerraMind tokenizer latent

The 5 channels are **not chosen by us** — they are the latent dimension of TerraMind's
tokenizers (Jakubik et al., [arXiv:2504.11171](https://arxiv.org/abs/2504.11171)). The
paper's own wording (v2, tokenizer sections): the model uses **"modality-specific
tokenizers"** — "autoencoder-based architectures with a quantization step in the
bottleneck for image-like modalities"; "tokenizer encoders process an input image and
generate a latent representation for each 16×16 patch, which is then discretized with
finite-scalar-quantization (FSQ)"; the architecture is a "Vision Transformer (ViT)
encoder and a patched UNet decoder"; **"the latent dimension was set to 5"** with a
"codebook size of 8-8-8-6-5, aiming to learn consistent and abstract representations
across image patches" (8·8·8·6·5 = 15,360 ≈ 16K codewords); FSQ was chosen because it
"improves training stability by using a fixed codebook instead of a learnable one"; each
token is "a compressed representation of a patch with compression factors of between
250× and 3000×". (Note: the paper's term is *modality-specific tokenizer*; it does not
use the phrase "in-domain encoder".) We use the frozen pretrained
`terramind_v1_tokenizer_s2l2a` / `_s1grd`; a 224 × 224 input with 16 × 16 patches gives
the 14 × 14 × 5 latent of §1.

What those 5 numbers *mean* is measured, not assumed. Regressing spectral-index maps
(pooled to the same 14 × 14 grid through the tokenizer's own geometry) on the 5 channels,
cross-validated:

| chips | within-chip (parcel level) R² | between-chip (chip means) R² |
|---|---|---|
| clean (100 % valid pixels, n = 130) | NDVI 0.71, NDWI 0.65, NDBI 0.80 | 0.92, 0.90, 0.95 |
| cloudy (chip-mean-filled) | 0.14–0.39 | 0.45–0.52 |

The token is a strong local descriptor **on clean imagery**, and the chip-mean cloud
fill destroys the local-descriptor property — directly relevant to the genfill results
in §6.3.

### 4.2 The estimator is their code

Nothing reimplements FSC. Okano & Kurisu's reference R (`/data/wang/junh/githubs/FSC`,
read-only) runs unmodified in the local R 4.3.3 environment, verified by reproducing
their own published `service.R` / `mortality.R` estimation paths on their own data.
`fsc_bridge.R` marshals our outcome blocks into their `func_vals_list` and calls
`FSCM`, `FSCM_aug` / `FSCM_aug_covmat`, `cross_val*`, `modif`, `nearPD`, `placebo*`.
Deviations are disclosed, not silent — the audit is `8-27-fsc-fidelity.md`.

### 4.3 Steps, each with the assumption it rests on

**Step 0 — panel.** Per treated site: the site + its 5 covariate-matched controls
(`control_rank ≤ 5`, the collaborator's notebook-14 design), treated first (their unit-1
convention). Only periods where every unit of the group has a latent (their list is
rectangular); exclusions recorded, never silent. *Assumes* one treated unit, untreated
donors, no interference — inherited from the matched design; donor overlap across the 10
groups mildly correlates their errors (matters for pooled inference, §4.3 step 7).

**Step 1 — Ψ (representation).** §3. *Assumes* an isometric embedding into a Hilbert
space with outcomes in a convex set — satisfied per channel; the 5-channel concatenation
and the covvec metric carry the two disclosed caveats of §3.

**Step 2 — marshalling.** Python block → R. *One real bug lived here:* Python's
`np.triu_indices` walks the upper triangle row-major, R's `upper.tri` fills column-major,
so the Gram vector was silently scrambled at the language boundary. Every L²-based
result was unaffected (one fixed permutation on both sides), but `nearPD` — the only
step reading the vector *as a matrix* — was "repairing" a mis-assembled matrix. Fixed
(`panel_repr.IU5`, column-major) and gated: Python round-trip = identity; the Python
vector fed through the verbatim `service.R` reconstruction in R returns the true Gram to
max|diff| = 0. *Lesson: a reshape convention is part of the data format and needs an
explicit cross-language gate.*

**Step 3 — FSC weights.** Their `FSCM`: min ‖Cw − d‖² s.t. Σw = 1, w ≥ 0 over the
stacked (M · T₀) pre-treatment outcomes, solved by quadprog with their +1e-6 stabiliser
and `round(w, 4)`. Algorithmically identical to the collaborator's SCM — FSC adds the
*interpretation* (the average is a Wasserstein barycenter / valid covariance), not a new
optimizer. *Assumes* the treated site is approximately a convex combination of donors;
with 4 free parameters against 120–800 constraints the fit is imperfect by construction
and their finite-sample bound degrades through the pre-fit term.

**Step 4 — λ and the augmented estimator.** Their scripts select λ by `optimise()` on a
linear interval tuned to *their* data scales and then hardcode it; on our scales that
put λ on a boundary in 20 of 28 configurations (audit Issue 1). Disclosed deviation:
evaluate **their unmodified CV objective** on a log grid (10⁻⁶…10⁶; λ = 0 excluded —
it makes their `solve()` singular). Finding: the CV is **monotonically decreasing in λ
over 12 orders of magnitude** on most configurations — the optimum is at infinity, where
the augmented weights collapse onto FSC. **The augmentation is inert on this panel**
(afsc ≡ fsc to ≤ 0.02), the same verdict reached earlier by ridge-ASCM (notebook 03)
and the unconstrained text-SC ridge (notebook 06). Three augmentation schemes, one
answer: no residual signal exists for a ridge correction to recover.

**Step 5 — projection back onto the outcome space.** Augmented weights can leave the
simplex, making the synthetic outcome an invalid object (a non-monotone "quantile
function", a non-PSD "covariance"). Their method projects back — `modif`
(truncate + monotone rearrange, per channel block in our concatenation) and `nearPD`.
Three purposes: (i) Ψ⁻¹ is only defined on valid objects; (ii) interpretation
(densities, eigendecompositions) requires validity; (iii) projection onto a convex set
containing the truth is a contraction, so it can only help the evaluation. *Measured on
our data it is a no-op* (shift exactly 0 for Gram — all synthetic matrices already PSD;
~3·10⁻⁴ for quantiles): plain-FSC outcomes are valid automatically by convexity, and the
augmented fits collapse onto FSC. Kept because it is their method, free, and required
the moment either condition changes.

**Step 6 — evaluation.** Prediction RMSE at P09 and P10 **per period** (pooling P09+P10
hid a bad P09 behind a good P10), under both schemes of notebooks 02–06: *frozen* (fit
P01–P08, score both) and *expanding* (fit P01–P08 → P09; refit P01–P09, λ re-selected →
P10). Raw RMSE is in each representation's own units; comparisons across arms use the
ratios of §6.1.

**Step 7 — inference.** Their in-space placebo: rotate each of the 6 units of a matched
group into the treated slot, refit, rank the true treated site's post-period gap among
the 6 (per-site p floor 1/6); aggregate the 10 groups as Binomial(10, 1/6) on the count
of rank-1 sites — 4/10 → p = 0.070, 5/10 → 0.015, 6/10 → 0.002 (an earlier draft
mis-stated this ladder one step too generously; the executed notebooks always used the
exact `binom.sf`). *Assumes* exchangeability under the sharp null —
what the matched design provides; donor overlap makes the binomial aggregation
approximate, so per-site ranks are always reported alongside.

## 5. Results

One common metric across every model in the project: **prediction error ÷ error of
predicting the site's own P01–P08 mean**, per site then averaged (expanding scheme;
below 1 = beats the site's own history). Source: `panel_by_model_ratio.csv`. Reading
guide: ASCM ≡ SCM here (its λ pinned at the CV grid boundary) and augmented FSC ≡ FSC
(CV pushes λ → ∞), so each pair is one row; "ridge" is the Wang et al. text-SC estimator
(notebook 06, unit-norm variant); "SCM on 5 latent chip-means" is FSC with the identity
embedding, i.e. plain SCM on a 5-number outcome. The 980-d rows' null is computed in
their pooled-z-scored fitting space, so cross-family comparison inherits that scaler;
within-family comparisons are exact.

**Pairing caveat (Sentinel-1 only):** the chip-mean cache is missing site 07's complete
P01–P10 S1 group, so its FSC rows average **9 sites** while the genfill FSC rows average
**10** — and within the chip-mean S1 table the 980-d rows still use all 10 (those models
tolerate incomplete windows; FSC requires a rectangular panel). The S1 cross-cache
comparison is therefore not perfectly paired. Restricting genfill S1 to the same 9
groups shifts its means by −0.08 to +0.004 (site 07 is a below-average genfill site:
e.g. Gram P09 0.992 → 0.958, P10 1.241 → 1.160) and changes no conclusion — S1 still
beats its own history nowhere at P10. Sentinel-2 is fully paired (10 groups in both
caches).

**Sentinel-2, chip-mean cache**

| model (outcome space) | P09 | P10 |
|---|---|---|
| SCM / ASCM (raw latent, 980-d) | 1.132 | 1.219 |
| ridge / text-SC (raw latent, 980-d) | 1.062 | 1.158 |
| SCM on 5 latent chip-means (= FSC, identity) | **0.702** | 0.665 |
| FSC on Gram (15) | **0.698** | 0.554 |
| FSC on histogram (100) | 0.825 | **0.548** |
| FSC on hist+gram (115) | 0.803 | **0.546** |

**Sentinel-2, genfill cache**

| model | P09 | P10 |
|---|---|---|
| SCM / ASCM (980-d) | 0.946 | 1.106 |
| ridge (980-d) | 0.897 | 1.047 |
| SCM on chip-means (5) | 1.357 | **0.709** |
| FSC on Gram (15) | 1.328 | 0.728 |
| FSC on histogram (100) | 1.122 | 0.871 |
| FSC on hist+gram (115) | 1.125 | 0.855 |

**Sentinel-1, chip-mean cache**

| model | P09 | P10 |
|---|---|---|
| SCM / ASCM (980-d) | 1.544 | 1.533 |
| ridge (980-d) | 1.440 | 1.432 |
| SCM on chip-means (5) | 0.993 | 1.146 |
| FSC on Gram (15) | **0.958** | 1.160 |
| FSC on histogram (100) | 1.062 | 1.163 |
| FSC on hist+gram (115) | 1.056 | 1.161 |

**Sentinel-1, genfill cache**

| model | P09 | P10 |
|---|---|---|
| SCM / ASCM (980-d) | 1.546 | 1.535 |
| ridge (980-d) | 1.441 | 1.432 |
| SCM on chip-means (5) | 1.033 | 1.219 |
| FSC on Gram (15) | **0.992** | 1.241 |
| FSC on histogram (100) | 1.085 | 1.160 |
| FSC on hist+gram (115) | 1.080 | 1.162 |

**Per-representation pass counts** (FSC, expanding; sites of 10 passing each of the
three comparisons defined in §6.1 — sanity / beats-own-history / beats-equal-weights):

| cache     | representation  | P09: C1 / C2 / C3 | P10: C1 / C2 / C3 |
| -----------| -----------------| -------------------| -------------------|
| chip-mean | chip-mean (5)   | 10 / 8 / 5        | 10 / 8 / 4        |
| chip-mean | Gram (15)       | 10 / 8 / 4        | 10 / 8 / 4        |
| chip-mean | histogram (100) | 10 / 8 / 4        | 10 / **10** / 3   |
| chip-mean | hist+gram (115) | 10 / 8 / 4        | 10 / **10** / 3   |
| genfill   | chip-mean (5)   | 10 / **4** / 6    | 8 / 9 / 6         |
| genfill   | Gram (15)       | 10 / **5** / 6    | 10 / 9 / 6        |
| genfill   | histogram (100) | 10 / **4** / 6    | 10 / 6 / 6        |
| genfill   | hist+gram (115) | 10 / **4** / 6    | 10 / 7 / 6        |

Placebo (step 7): across 20 cells (caches × sensors × post-periods × arms), **nothing
reaches significance** — best p = 0.070 (Gram, genfill S1, P11); the rest 0.18–1.00.

## 6. Analysis

### 6.1 Three comparisons, and what each one certifies

A prediction error is only readable next to a simpler alternative. Worked example
(site 03, Gram, chip-mean cache, P10):

| prediction method | error |
|---|---|
| SCM with fitted weights (the thing being tested) | 0.052 |
| the same weights scored on the training window | 0.078 |
| **no donors** — the site's own P01–P08 average | 0.108 |
| **donors without fitting** — plain average of the 5 matched controls | 0.055 |

**Comparison 1** — against training error (the collaborator's notebook-14 flag,
holdout/train ≤ 1.5). *Did the fit merely memorize?* 0.052/0.078 = 0.67 → pass. It
passes everywhere — with 4 free weights there is no capacity to memorize, so this is a
sanity check, not evidence of quality.

**Comparison 2** — against using no donors. *Do the donors add anything beyond the
site's own history?* 0.052 < 0.108 → pass. (The per-site version of the old
"standardized RMSPE < 1" flag, with the pooled-scaler leak fixed.)

**Comparison 3** — against donors without fitting. *Does the weight optimization earn
anything beyond the pool?* 0.052 < 0.055 → pass, barely — and across all sites and
representations this passes only 3–6 of 10, a coin flip.

### 6.2 Where the predictive power actually comes from

Comparison 2's two predictors differ in *two* factors at once — **who** is used (own
site vs donors) and **when** the information comes from (a stale pre-period average vs
the fortnight being predicted). Isolating them (chip-mean cache, S2, chip-mean
representation, P10; mean RMSE over sites):

| predictor | who | when | RMSE |
|---|---|---|---|
| own P01–P08 average | self | stale (data ends Aug 29) | 0.0815 |
| own most recent pre-period (P09) | self | fresher (ends Sep 12) | 0.0737 |
| donors' P01–P08 average | pool | stale | **0.0868 — worse than own history** |
| donors' average **during the target fortnight** (Sep 13–26) | pool | same fortnight | 0.0490 |
| **all 50 controls**, same fortnight | any pool | same fortnight | **0.0451** |

Holding time fixed, switching self → pool makes prediction *worse*. Holding the pool
fixed, switching stale → same-fortnight delivers the entire gain. And the plain average
of all 50 controls does as well as the 5 matched ones. The corrected decomposition
(same pattern for Gram and for the genfill cache):

| ingredient of the SC pipeline | measured contribution |
|---|---|
| **contemporaneity** — untreated sites observed in the same fortnight (the common time factor: that fortnight's weather, season, phenology) | **all of it** (error roughly halves) |
| covariate matching (matched 5 vs any 50) | ≈ 0 (± 0.004, mixed sign) |
| SCM/FSC weight optimization (fitted vs equal weights) | ≈ 0 (comparison 3) |

This is coherent with SC's own factor-model logic — donors exist to supply the common
time factor — but at this pooled outcome level the common factor dominates so thoroughly
that any untreated pool captures it, making matching and weighting redundant
refinements. (Notebook 06 showed the weighting can even *hurt*: the 980-d simplex was
worse than predicting nothing.) **The honest counterfactual on this panel is the average
of untreated sites in the same fortnight** — an event-study control mean; every richer
construction measured here reduces to it. Post-period this rests on the usual
no-interference assumption.

### 6.3 The genfill result: cloud fill and the fit

P09 is the cloudy fortnight (mean 71 % valid pixels; 28 % of sites below half); P10 is
nearly clean (98 %). Two consequences, visible in every §5 table:

1. **At cloudy P09, the genfill cache fails for every pooled model** (1.12–1.36) while
   the chip-mean cache passes (0.70–0.83). The genfill targets at P09 are
   generation-derived (radar-conditioned synthesis), and neither cache offers a clean
   target there.
2. **At clean P10 the caches contain nearly the same raw content, so prediction
   differences isolate the *weights*** — fitted on the differing P01–P08 histories.
   Chip-mean-fitted weights predict clean P10 at 0.48–0.57 of the own-mean null;
   genfill-fitted at 0.66–0.85. **Generation fill damages the pooled-representation
   fit**, consistent with §4.1's finding that fill methods degrade the token's
   local-descriptor property — while notebook 05's image-space gains (donor-selection
   stability, the S1 gap-fill) still stand. Use the chip-mean cache for pooled FSC.

### 6.4 Gram vs histogram

Gram wins more cells and is never the worst (S2-P09 chip-mean cache 0.698 vs 0.825;
S1-P09 0.958/0.992; S2-P10 genfill 0.728 vs 0.871); the histogram matches or edges it
only on clean imagery (S2-P10 chip-mean cache 0.548 vs 0.554, with a perfect 10/10 on
comparison 2). A plausible mechanism: second moments are less distorted by fill
contamination than the full distribution. Gram is also 15 numbers against 100 and is the
arm matching Okano–Kurisu's published application exactly. But the ranking is secondary:
neither clears comparison 3, so differences among pooled representations are differences
in how each carries the common fortnight factor, not weighting skill.

### 6.5 Comparison with the collaborator's feature-space validation (notebooks 16–17)

Pushed 2026-08-23 (`scripts/16_test_scm_single_period_validation.ipynb`,
`scripts/17_test_ascm_single_period_validation.ipynb`), merged here 2026-08-26. These
run SCM and ridge-ASCM (Ben-Michael–Feller–Rothstein) on **interpretable band means**
— Sentinel-1: mean VV, VH, VV−VH (3 features); Sentinel-2: mean B2, B3, B4, B8, B11,
B12, NDVI, NDWI (8 features) — for the same 10 treated sites with the same 5 matched
controls each, features z-scored on training periods only.

**Design convergence.** The earlier notebooks 14–15 froze weights on P01–P08 and
scored P09–P10 jointly. Notebooks 16–17 switch to **per-period expanding validation —
exactly our scheme** (fit P01–P08 → predict P09; refit P01–P09 → predict P10; the
holdout never touches standardization, weights, or λ). Both pipelines independently
arrived at the same correction of the pooled-holdout design, and their P09/P10 columns
are now design-aligned with our §5 tables. *Units are not aligned*: their RMSE is in
pooled z-scored feature units, ours in latent-statistic units — patterns and
within-pipeline ratios are comparable, absolute values are not, and their validation
reports no own-history null, so their numbers cannot be placed on our common
error-÷-own-history metric.

Their site-mean standardized test RMSE (train RMSE of the same design in parentheses):

| sensor | P09 (fit P01–P08) | P10 (fit P01–P09) |
|---|---|---|
| Sentinel-1 | 0.284 (0.286) | 0.313 (0.298) |
| Sentinel-2 | 0.839 (0.587) | 0.342 (0.657) |

Their notebook-14 flag (test/train ratio): P09 — S2 only 4 of 10 "good" (3 caution,
3 poor, worst ratio 4.28); P10 — S2 10 of 10 "good", S1 9 of 10.

**Convergent findings.**

1. **The P09/P10 asymmetry is in the data, not our representation.** Their
   feature-space S2 error more than doubles from P10 to P09 (0.342 → 0.839, mean
   test/train ratio ≈ 1.4) with 3 "poor" sites — independent evidence that the cloudy
   P09 fortnight (71 % valid pixels) is hard for *any* image-derived outcome, matching
   §6.3.
2. **Ridge augmentation is inert-to-marginal there too.** Their leave-one-period-out
   CV pins λ at the top of their grid (10⁴) in 9 of 20 site × sensor fits (P09 design)
   and 10 of 20 (P10 design) — at that λ the ASCM weights collapse onto SCM (weight
   distance ≤ 10⁻⁴), the same boundary behaviour as our three augmentation schemes
   (§4.3 step 4). Where λ is interior, gains are mixed: at P09 ASCM improves the S1
   features (+0.03 to +0.05 mean RMSE) but *worsens* most S2 features (B8 −0.19,
   B12 −0.11); at P10 the site-level split is 5/10 improved per sensor.
3. **Donor weights concentrate in both pipelines**: their effective number of donors
   is 1.0–4.1 of 5 (largest weight 0.36–1.00; two S1 sites are single-donor),
   comparable to our ~3 of 5 non-zero FSC weights.

**Divergent finding — Sentinel-1.** In their feature space S1 validates *decently*
(P09 test ≈ train, 0.284 vs 0.286; flags mostly "good"), while in our latent space S1
beats the site's own history nowhere (§5). Radar band means evidently carry
donor-trackable signal that the S1 TerraMind latent statistics do not preserve — which
argues the S1 failure is **representation-specific**, and favours "rethink the S1
representation" over "drop S1" in §8.

**What their validation does not yet certify.** Their good/caution/poor flag is the
test-to-training ratio — rung 1 of §6.1's ladder (the sanity check that passes our
pipeline everywhere too). No comparison against the site's own history (comparison 2),
no unweighted-donor-average null (comparison 3), and no placebo test have been run in
feature space. Those are exactly the rungs where our pooled models stall, so applying
§6.1's harness to their per-site outputs (saved under their Dropbox
`test/scm_validation/` and `test/ascm_validation/`) is the natural next joint step.

## 7. Takeaways and conclusion

1. **The 980-d failure was a representation failure, not an estimator failure.**
   Position-matched latents compare different pieces of ground; S2's in-sample fit was
   alignment overfitting. Pooling fixes it: on Sentinel-2, pooled models beat the site's
   own history where no 980-d model does (worth ~0.4–0.6 on the common ratio; switching
   estimators within a family is worth ~0.0–0.1).
2. **FSC's contribution here is the representation framework and the inference
   machinery, not estimation gains.** Its objects (barycenters, PSD averages) make the
   pooled counterfactual well-defined and interpretable, and its placebo test gave this
   project its first real p-values. Its weight optimization — like ridge-ASCM and the
   text-SC ridge before it — adds approximately nothing; its augmentation is inert
   (CV monotone in λ over 12 decades).
3. **The predictive power is contemporaneity.** Untreated sites observed in the same
   fortnight carry the common time factor; matching and weighting contribute ≈ 0 on top.
   The defensible counterfactual is the same-fortnight untreated average, and any effect
   claim must beat that baseline.
4. **No post-period hurricane effect exceeds the noise floor**: 20 placebo cells, best
   p = 0.070. A properly-inferred null — which neither the latent pipeline nor the
   feature pipeline previously had the machinery to state.
5. **Generation cloud fill is task-dependent**: it helps image-space work (notebook 05)
   and damages the pooled-representation fit (§6.3). Sentinel-1 carries no donor signal
   in any representation under any estimator.

## 8. Next steps

**Solved by this work:**
- *Why* latent-space SC failed (representation, measured), and what fixes it (pooling).
- What the FSC-vs-SCM comparison captures: FSC ≡ SCM as an optimizer; the gain is
  well-defined pooled objects + placebo inference; a reusable null-testing harness
  (three comparisons + placebo) now exists and applies to the feature pipeline too.
- A value-chain decomposition that any future effect claim can be audited against.

**Unsolved / next:**
- **Cloud noise needs a better strategy than full-image fill.** Both fills fail at the
  cloudy P09 in opposite ways (generation moves targets off the donor manifold;
  chip-mean fill destroys token locality). Candidates: compute pooled statistics from
  *valid parcels only* (masked histograms/Grams need no fill at all);
  quality-weighted period selection.
- **The heterogeneity claim is untested.** §3.4's measured advantage of
  histogram/Gram is in within-chip dispersion, which the P09/P10 level-prediction
  holdout does not isolate. A dispersion-valued outcome experiment would test it.
- **Their conformal prediction intervals** (`mortality.R` L278–298) remain unbuilt —
  the open item of the fidelity audit.
- **Sentinel-1** shows no donor signal in any configuration; drop it from the pooled
  pipeline or rethink its representation. §6.5 tilts this toward *rethink*: the
  collaborator's feature-space S1 (band means) validates decently, so the signal
  exists and the S1 latent statistics are losing it.
- **Notebooks 07–09 still contain the pre-fix pooled analysis**; the corrected
  per-period results live in `panel_fsc_prediction.csv` / `panel_by_model_ratio.csv`.
  Rebuilding the notebooks to match is a pending, separately-tracked task.
