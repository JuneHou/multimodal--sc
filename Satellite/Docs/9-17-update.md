# Stage 1 of the latent synthetic historical control learned by a neural network

**Progress report for the meeting of 17 September 2026.** Continues the 10 September report (`9-10-update.md`), whose kernel-stage results (§4–§5 there) are the comparison throughout. Method sources as there: Chen, Yang & Yang, SSRN 4995085; the collaborator's slides of 4 September 2026. Scripts and tables: §5.

---

## 1. What is replaced, and by what

**Summary.** Stage 1 of the paper (slide 6: the covariate step and the local-linear kernel smoothing of one site's own months, which gives the untreated latent component ℓ̂_t for t ≤ T0) is replaced by a neural network f(l_i, t) trained across sites. Stage 2 (blocks, stepwise matching, weights, counterfactual) is the paper's, unchanged, and runs on the network's ℓ̂ exactly as it ran on the kernel's.

**What f is, in fixed-effects terms.** Write the latent of site i at period t as y_{i,t} = α_i + γ_t + ε_{i,t}: a site fixed effect and a period fixed effect, both unknowns estimated from the y's and looked up by an index. The network is that model with three changes:

- the site effect is a vector of 32 numbers instead of one number; this is l_i, the site's state, the l of the advisor's f(l, t), one vector per site (132 on Sentinel-1, 165 on Sentinel-2), estimated from the site's own latents like α_i would be;
- the period effect is a vector of 8 numbers instead of one, one per period;
- the two are combined through two hidden layers instead of being added, so the site's look can depend on the period nonlinearly, and the output is the whole 980-value latent.

The only input in the data sense is the index (site, period). The network looks up the site's 32-number row and the period's 8-number row, joins them into one 40-number vector, passes it through the hidden layers, and outputs 980 numbers: ℓ̂_{i,t}, its estimate of the untreated latent of site i at period t. In training that output is compared with the observed latent at (i, t) and the weights and both tables are adjusted. The observed latents are only ever the target, never the input. The kernel stage, by contrast, takes the site's ten observed latents as its input and returns their smoothed version; the network takes no latent as input, what it knows about a site is absorbed into the site's row.

**One network, f(l, t) only.** There is no f(l, λ, t). The network is trained on untreated rows alone (every site's months up to the fit's T0, all before the hurricane), so no treated image enters it and it holds no λ. After stage 2, the paper's Δ_t is read off as the observed post-hurricane latent minus the counterfactual; that is the λ of §4. In the advisor's notation the treated image is f(l, t) + λ, with λ read off, not fitted.

**How ℓ̂ enters stage 2.** ℓ̂ is used on both sides of the weight fit: the treated block's pre-segment (ℓ̂ at P07–P10 for m = 4) is the target, the historical blocks' pre-segments are the regressors, and the weights are the convex combination of blocks that reproduces the treated pre-segment. For the evaluation the weights are applied to the blocks' forward rows of the observed latent y (the paper's eq 27), not to ℓ̂: the synthetic month is then an observed-scale quantity, with the same noise as the month it is compared with, and the comparison is observed y minus synthetic y, at the held-out months in the validation and at P11–P12 for λ. (Applying the weights to ℓ̂ instead gives a noise-free prediction against a noisy truth and counts the month's noise as error; that earlier evaluation, used in the 10 September report, is corrected here on 11 September.) Under this evaluation stage 1 enters the result only through the blocks it selects and the weights it gives them.

**Two arms.** The fully connected arm and the convolutional arm are separate networks, trained separately, with the same input and the same 980-value output; they differ only in the last layer.

- Inputs: the site's 32-number row and the period's 8-number row, joined (40 numbers).
- Hidden layers: two of 256, ReLU.
- Last layer, fully connected arm: linear 256 → 980, every output value with its own weights.
- Last layer, convolutional arm: linear 256 → 32 × 7 × 7; transposed convolution (kernel 4, stride 2) → 32 × 14 × 14; 3 × 3 convolution → 5 × 14 × 14, read in the latent's (channel, row, column) order; filters shared across parcel positions.
- Parameters: 0.33 M (fully connected), 0.50 M (convolutional).
- Loss: squared error on the 980 values + 1e-3 · ‖l_i‖².
- Optimizer: Adam, learning rate 1e-3, weight decay 1e-4 on the weights; batch 64 (the number of training rows per update); seed 0.
- Stopping: 10 % of the site-months held out at random; the epoch with the lowest held-out error is kept; training stops after 50 epochs without improvement (cap 5000).

**Three fits per arm and sensor.** The pipeline is evaluated in three settings with different last pre-periods, and a validation network must not have seen the months it predicts, so each has its own network: T0 = P10 for λ (predicts P11–P12), P08 for the validation at n = 2 (predicts P09–P10), P07 for n = 3 (predicts P08–P10). Each is trained on every site's usable months up to its T0, treated and control sites alike, and never asked for a later period; the kernel stage is refitted on the same shorter windows. Sentinel-1 latents are identical under both cloud fills and are fitted once.

---

## 2. What the networks learned

**Summary.** Held-out R² is 0.15 to 0.40 where a network trained. The convolutional arm sits at the data's mean for several hundred epochs before it learns, and at three of its nine fits the stopping rule fired inside that plateau; those fits are untrained and are marked. The rule was kept as set on 11 September.

Held-out = the 10 % of site-months left out of training; R² against the held-out cells' own mean per coordinate.

| arm | sensor | fill | T0 of the fit | training rows | best epoch | held-out MSE | held-out R² | status |
|---|---|---|---|---|---|---|---|---|
| fully connected arm | Sentinel-1 | both | P10 | 1133 | 1114 | 0.0543 | 0.38 | trained |
| fully connected arm | Sentinel-1 | both | P08 | 895 | 1012 | 0.0538 | 0.39 | trained |
| fully connected arm | Sentinel-1 | both | P07 | 777 | 1102 | 0.0524 | 0.40 | trained |
| fully connected arm | Sentinel-2 | chip-mean | P10 | 1391 | 1210 | 0.0555 | 0.15 | trained |
| fully connected arm | Sentinel-2 | chip-mean | P08 | 1093 | 1251 | 0.0561 | 0.17 | trained |
| fully connected arm | Sentinel-2 | chip-mean | P07 | 946 | 1812 | 0.0535 | 0.26 | trained |
| fully connected arm | Sentinel-2 | historical | P10 | 1391 | 1092 | 0.0567 | 0.16 | trained |
| fully connected arm | Sentinel-2 | historical | P08 | 1093 | 1254 | 0.0585 | 0.16 | trained |
| fully connected arm | Sentinel-2 | historical | P07 | 946 | 2303 | 0.0541 | 0.26 | trained |
| convolutional arm | Sentinel-1 | both | P10 | 1133 | 1352 | 0.0610 | 0.30 | trained |
| convolutional arm | Sentinel-1 | both | P08 | 895 | 1265 | 0.0574 | 0.35 | trained |
| convolutional arm | Sentinel-1 | both | P07 | 777 | 401 | 0.0869 | -0.00 | untrained (stopped on the plateau) |
| convolutional arm | Sentinel-2 | chip-mean | P10 | 1391 | 1074 | 0.0432 | 0.34 | trained |
| convolutional arm | Sentinel-2 | chip-mean | P08 | 1093 | 405 | 0.0619 | 0.08 | untrained (stopped on the plateau) |
| convolutional arm | Sentinel-2 | chip-mean | P07 | 946 | 834 | 0.0448 | 0.38 | trained |
| convolutional arm | Sentinel-2 | historical | P10 | 1391 | 1420 | 0.0448 | 0.33 | trained |
| convolutional arm | Sentinel-2 | historical | P08 | 1093 | 234 | 0.0645 | 0.07 | untrained (stopped on the plateau) |
| convolutional arm | Sentinel-2 | historical | P07 | 946 | 944 | 0.0466 | 0.37 | trained |

**How much ℓ̂ moves in time.** Standard deviation over the ten pre-period months, mean over the 980 coordinates and the treated sites (historical fill, T0 = P10):

| sensor | observed latent | kernel ℓ̂ | fully connected ℓ̂ | convolutional ℓ̂ |
|---|---|---|---|---|
| Sentinel-1 | 0.1812 | 0.0672 | 0.0035 | 0.0009 |
| Sentinel-2 | 0.1845 | 0.0904 | 0.0608 | 0.0822 |

On Sentinel-1 the network's ℓ̂ is nearly constant in time: the period rows carry almost nothing and what the network learned is each site's mean look. On Sentinel-2 it carries a seasonal pattern of the kernel's size.

---

## 3. Stage 2 on the network's ℓ̂: block selection and validation

**Summary.** On Sentinel-1 every historical block of a site is nearly identical to its treated block, so the stepwise choice has nothing to choose on; on Sentinel-2 block 1 is no longer the best block. Because the prediction is built from observed forward rows, the three arms coincide wherever each selects block 1 alone (every row at n = 3); where they differ, the differences are within 0.006 except the fully connected arm on Sentinel-1, which spread its weight over two or three blocks and is worse. No arm beats the own-mean prediction on Sentinel-1; on Sentinel-2 all three come in under it at n = 2 and none at n = 3.

**Block selection**, m = 4, historical fill, treated sites; latent MSE = the collaborator's trial MSE of the selected block (eq 31):

| sensor | n | block 1 alone, kernel | latent MSE, kernel | block 1 alone, fully connected | latent MSE, fully connected | block 1 alone, convolutional | latent MSE, convolutional |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | 2 | 12 / 12 | 4.06e-03 | 0 / 12 | 1.41e-05 | 12 / 12 | 1.36e-06 |
| Sentinel-1 | 3 | 12 / 12 | 8.87e-03 | 0 / 12 | 1.60e-05 | 12 / 12 | 1.54e-06 |
| Sentinel-2 | 2 | 15 / 15 | 7.51e-03 | 0 / 15 | 6.95e-03 | 1 / 15 | 1.46e-02 |
| Sentinel-2 | 3 | 15 / 15 | 1.62e-02 | 0 / 15 | 9.91e-03 | 1 / 15 | 2.22e-02 |

**Prediction error** (fit on P01–P(10 − n), predict the last n pre-hurricane months; the fitted weights applied to the observed forward rows of the selected blocks; RMSE over the 980 coordinates between the observed latent of the held-out month and that synthetic month, mean over months and treated sites; m = 4). Own mean = the site's own mean of P01–P(10 − n) used as the prediction. The convolutional arm's validations at n = 2 on Sentinel-2 and at n = 3 on Sentinel-1 rest on untrained networks (§2).

| sensor | fill | n | prediction error, kernel stage | fully connected arm | convolutional arm | own mean | sites where the synthetic beats the own mean (kernel / fully connected / convolutional) |
|---|---|---|---|---|---|---|---|
| Sentinel-1 | both | 2 | 0.2666 | 0.3080 | 0.2633 | 0.2360 | 0 / 0 / 0 of 12 |
| Sentinel-1 | both | 3 | 0.3022 | 0.3022 | 0.3022 | 0.2452 | 0 / 0 / 0 of 12 |
| Sentinel-2 | chip-mean | 2 | 0.2267 | 0.2325 | 0.2276 | 0.2415 | 10 / 10 / 12 of 15 |
| Sentinel-2 | chip-mean | 3 | 0.2969 | 0.2969 | 0.2969 | 0.2540 | 0 / 0 / 0 of 15 |
| Sentinel-2 | historical | 2 | 0.2413 | 0.2376 | 0.2359 | 0.2439 | 8 / 8 / 9 of 15 |
| Sentinel-2 | historical | 3 | 0.2971 | 0.2971 | 0.2971 | 0.2525 | 0 / 0 / 0 of 15 |

---

## 4. λ at the post-hurricane months

**Summary.** The post-hurricane departure of the treated sites is of the size the kernel stage gives; the fully connected arm's is smaller on both sensors, the convolutional arm's equals the kernel's on Sentinel-1, where both select block 1 alone.

λ = observed latent − the fitted weights applied to the observed forward rows of the selected blocks, ‖λ‖/√980 per month, mean over the treated sites; m = 4, n = 2, historical fill; the networks behind these columns (T0 = P10) are all trained.

| sensor | month | kernel stage | fully connected arm | convolutional arm |
|---|---|---|---|---|
| Sentinel-1 | P11 | 0.2595 | 0.2371 | 0.2595 |
| Sentinel-1 | P12 | 0.3007 | 0.2713 | 0.3007 |
| Sentinel-2 | P11 | 0.2555 | 0.2370 | 0.2425 |
| Sentinel-2 | P12 | 0.2401 | 0.2302 | 0.2298 |

---

## 5. Language models on the band-feature design: per site, against her stored number directly

**Summary.** This section is a different pipeline from §1–§4: two language models (gpt-oss-120b, Kimi-K3) used as the weight solver on band means (Sentinel-1 VV/VH/VV−VH, Sentinel-2 eight bands), not the site-embedding network on the 980-value latent. Her stored per-site number and our reproduction of her method agree to seven decimals, so this table's `her stored SHC` column is her number, not a re-implementation.

Outcome: squared error at P10 divided by the site's own P01–P09 variance (her scaling), mean over bands; predicted from the site's own P01–P09 band means, nothing treated. `her stored SHC` = her stored per-site value; `own mean` = the site's P01–P09 mean used as the prediction, with nothing estimated.

| sensor | site | Kimi-K3 (low reasoning) | gpt-oss-120b (low reasoning) | her stored SHC | own mean P01–P09 |
|---|---|---|---|---|---|
| Sentinel-1 | 0015 | 0.2383 | 0.4899 | 1.1816 | 0.4121 |
| Sentinel-1 | 0016 | 0.2030 | 0.0286 | 0.0811 | 0.5943 |
| Sentinel-1 | 0017 | 0.1574 | 0.2242 | 0.2934 | 0.8159 |
| Sentinel-1 | 0018 | 0.1440 | 0.1162 | 0.2653 | 0.7112 |
| Sentinel-1 | 0019 | 0.3512 | 0.3814 | 0.4842 | 1.0392 |
| Sentinel-1 | 0020 | 0.0730 | 0.0627 | 0.1650 | 0.3899 |
| Sentinel-1 | 0021 | 0.6553 | 0.1259 | 0.4820 | 0.1809 |
| Sentinel-1 | 0022 | 0.2966 | 0.8876 | 0.4400 | 0.2279 |
| Sentinel-1 | 0023 | 1.1411 | 1.6792 | 1.1838 | 1.0810 |
| Sentinel-1 | 0024 | 0.4316 | 0.4828 | 0.3970 | 1.3041 |
| Sentinel-1 | 0025 | 0.3051 | 0.3900 | 0.5210 | 0.9187 |
| Sentinel-1 | 0027 | 0.2374 | 0.2374 | 0.4150 | 0.4397 |
| Sentinel-2 | 0001 | 0.3389 | 0.1062 | 0.1527 | 0.2164 |
| Sentinel-2 | 0002 | 1.6095 | 1.5120 | 1.6750 | 3.0268 |
| Sentinel-2 | 0003 | 1.1218 | 0.8897 | 1.1643 | 1.7133 |
| Sentinel-2 | 0004 | 9.0719 | 4.2700 | 3.2316 | 0.5764 |
| Sentinel-2 | 0005 | 3.0602 | 1.7169 | 2.1669 | 1.7604 |
| Sentinel-2 | 0006 | 0.8103 | 0.8205 | 0.6140 | 0.8070 |
| Sentinel-2 | 0007 | 0.1456 | 0.1371 | 0.1843 | 0.7898 |
| Sentinel-2 | 0008 | 0.2714 | 0.2012 | 0.2396 | 1.1746 |
| Sentinel-2 | 0009 | 0.4023 | 0.3501 | 0.2066 | 1.0691 |
| Sentinel-2 | 0010 | 1.4273 | 0.8984 | 0.8987 | 1.2764 |
| Sentinel-2 | 0011 | 0.2117 | 0.1049 | 0.1083 | 0.5594 |
| Sentinel-2 | 0012 | 1.2390 | 0.6215 | 0.9166 | 1.0940 |
| Sentinel-2 | 0013 | 0.2701 | 0.1181 | 0.1540 | 0.8005 |
| Sentinel-2 | 0014 | 0.0718 | 0.0680 | 0.0364 | 0.5619 |
| Sentinel-2 | 0015 | 0.1249 | 0.1318 | 0.0515 | 1.3213 |

---

## 6. Reproducibility

§5 is produced in `Satellite/notebooks/llm_band_sc/`; the rest is relative to `Satellite/notebooks/shc/` (the synthetic historical control folder; the panel modules it imports stay in `Satellite/notebooks/ts_SCM_ASCM/`). The kernel columns are read from the 10 September run's tables and are not recomputed.

| script / artifact | produces | read by |
|---|---|---|
| `panel_generator.py` | the network, its training and the per-site ℓ̂ | §1 |
| `panel_shc.py` (`shc_fit(..., L_hat=)`) | stage 2 on a given ℓ̂; the kernel stage is skipped when ℓ̂ is supplied | §1, §3 |
| `panel_shc_run.py --stage1 mlp` / `--stage1 cnn` | `panel_generator_{mlp,cnn}.csv` (§2 table), `panel_generator_curves_*.csv`, `panel_generator_lhat_*.npz` (the ℓ̂ arrays), `panel_shc_{fits,effects,summary,path}_{mlp,cnn}.csv`, `panel_shc_lambda_*.npz` (λ per site and month) | §2–§4 |
| `panel_shc_tests.py`, gates 7–12 | stage 2 unchanged under a given ℓ̂ (max diff 0 against the kernel fit); training rows stop at the fit's T0; planted site + period structure recovered by both arms; the kernel arm reproduces the 4 September fits; output layout; λ recomputed from the stored files | §1–§4 |
| `24_shc_generator.ipynb` (`build_nb24.py`) | every table of §2–§4 with the kernel, fully connected and convolutional columns side by side, and the training curves | §2–§4 |
| `lb_score.py`, `lb_shc.py`, `lb_her_nb10_results.csv` (`Satellite/notebooks/llm_band_sc/`) | `lb_scores_shc.csv` (per site, per model, `her_stored_joint_nmse` = her file, unmodified) | §5 |
