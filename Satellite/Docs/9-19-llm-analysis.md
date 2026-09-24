# Language models as the weight solver: why the predictions are wrong

**Analysis for the meeting of 19 September 2026.** Continues the language-model section of the
10 September report (`9-10-update.md` §6). That report established *how much*
error each arm carries. This one asks *what kind of error it is*: of every prediction that missed,
what made it miss. Nothing in the earlier report is revised. Scripts and tables: §7.

---

## 1. What is being compared

**Method.** Five hosted language models used as the weight solver on the band-space synthetic
historical control design: given one site's own band means for nine consecutive months, predict the
tenth. The withheld month is P10 (August 2024); all ten months are pre-hurricane and nothing in this
validation is treated. Twenty-seven target cells, twelve on Sentinel-1 and fifteen on Sentinel-2.

**Two prompts,** carrying the same evidence and the same one-line answer format, neither containing
any word for hurricane, landslide or damage.

- The **direct prompt**: a band guide, the site's nine monthly rows, the instruction to predict the
  next month.
- The **step-by-step prompt**: the same, plus the band-space pipeline's procedure in prose — smooth
  each month against its neighbours, treat equal-length earlier stretches as candidate controls,
  score each against the most recent stretch, add candidates while each improves the match, weight
  them non-negatively to sum to one, apply those weights to the values following each stretch. The
  wording was validated clause by clause against the pipeline's notebook before any call was made.

**Metric.** The band-space pipeline's own held-out measure, which its notebook calls the joint
validation NMSE: per band the squared error at the withheld month divided by the variance of that
band across P01–P09, averaged over bands. Lower is better. That variance is the pipeline's own
`train_var`, taken over the periods its fit is allowed to see. Scored this way the re-run reproduces
its twenty-seven published per-site values to 4.2 × 10⁻⁷ and its sensor means of 0.4925 and 0.7867,
so every figure below is directly comparable with its published numbers. An earlier draft of this
report used the variance over P01–P10, which is not that pipeline's denominator; every number here
has been recomputed.

**Provenance.** Neither gateway is reproducible — the research gateway accepts a temperature of zero
and ignores it, and the gateway serving the GPT-5 family rejects the parameter outright. The direct
prompt has two runs per open-weight model and two for GPT-5.5; the step-by-step prompt has one of
each. GPT-5.5 was reached two ways and the name resolves to two different snapshots, dated 24 and
23 April 2026; both are reported, and every effect below appears on both.

---

## 2. How wrong, before why

**Method.** All five models, both prompts, scored as in §1. Matched sites only — those a model
answered under both prompts — since the counts differ.

| sensor | model | matched | direct | step-by-step | change |
|---|---|---|---|---|---|
| Sentinel-1 | gpt-oss-120b | 12 / 12 | 0.496 | 0.495 | −0.1 % |
| Sentinel-1 | Kimi-K3 | 11 / 12 | 0.348 | 0.448 | +28.7 % |
| Sentinel-1 | GPT-5.5, 24 April | 12 / 12 | 0.434 | 0.517 | +19.0 % |
| Sentinel-1 | GPT-5.5, 23 April | 12 / 12 | 0.446 | 0.509 | +14.1 % |
| Sentinel-2 | gpt-oss-120b | 15 / 15 | 0.691 | 0.769 | +11.3 % |
| Sentinel-2 | Kimi-K3 | 8 / 15 | 1.589 | 0.918 | −42.2 % |
| Sentinel-2 | GPT-5.5, 24 April | 15 / 15 | 1.605 | 0.777 | −51.6 % |
| Sentinel-2 | GPT-5.5, 23 April | 15 / 15 | 1.376 | 0.787 | −42.8 % |

Reference rows on the same cells: the band-space SHC scores 0.492 and 0.787; carrying P09 forward
scores 0.492 and 0.789 — **the same predictor to three decimals on Sentinel-1 and to two on
Sentinel-2**, as expected, since the stepwise donor selection keeps the most recent block at
essentially every site. The mean of the last three
months scores 0.265 and 0.541, beating every model and both prompts on both sensors. The site's own
nine-month mean scores 0.676 and 1.116.

The rest of this document is about the gap between those numbers and zero.

---

## 3. Why the predictions are wrong

Seven causes, separated so that each can be judged on its own evidence. They are not equally large:
§3.2 and §3.3 account for most of the error on Sentinel-1, §3.5 for much of the Sentinel-2 mean, and
§3.7 is small and included because it is a real defect that the headline metric hides.

### 3.1 No answer was produced at all

**Method.** Calls that never returned a parsable line, counted by the reason the gateway reported.

| model | prompt | Sentinel-1 | Sentinel-2 | reason |
|---|---|---|---|---|
| GLM-5.3 | direct | 12 | 12 | output budget spent on reasoning |
| GLM-5.3 | step-by-step | 12 | 15 | output budget spent on reasoning |
| Kimi-K3 | step-by-step | 1 | 6 + 1 | budget spent; one degenerate loop |
| DeepSeek-V4-Flash | either | all | all | the model name is not served |

GLM-5.3 does not terminate: its stored traces run past 21,000 characters without reaching an answer
line, on a task where gpt-oss-120b finishes in 156 tokens. This is the model, not the budget.
Kimi-K3's losses concentrate on Sentinel-2, where eight bands make the arithmetic longest. These
failures are not accuracy problems and do not appear in any mean; they are a separate, prior failure
to produce output at all.

### 3.2 The series reverses direction almost every month — the dominant cause

**Method.** For each band and site, whether the withheld month continued the direction of the
previous month's move or reversed it, with the error computed separately for each group.

| sensor | model | prompt | error when continued | error when reversed | ratio |
|---|---|---|---|---|---|
| Sentinel-1 | GPT-5.5, 23 April | step-by-step | 0.065 | 0.549 | **8.5×** |
| Sentinel-1 | gpt-oss-120b | direct | 0.069 | 0.535 | 7.8× |
| Sentinel-1 | Kimi-K3 | step-by-step | 0.065 | 0.486 | 7.5× |
| Sentinel-1 | GPT-5.5, 23 April | direct | 0.077 | 0.479 | 6.2× |
| Sentinel-1 | GPT-5.5, 24 April | direct | 0.119 | 0.463 | 3.9× |
| Sentinel-1 | Kimi-K3 | direct | 0.117 | 0.400 | 3.4× |
| Sentinel-2 | GPT-5.5, 24 April | direct | 0.655 | 2.382 | 3.6× |
| Sentinel-2 | Kimi-K3 | direct | 0.576 | 1.894 | 3.3× |
| Sentinel-2 | gpt-oss-120b | direct | 0.460 | 0.879 | 1.9× |

**On Sentinel-1, 33 of 36 band-months reversed; only 3 continued.** On Sentinel-2, 66 of 120
reversed. The monthly series is close to mean-reverting at this horizon, and every model — and the
band-space pipeline, and the three-month mean — is some form of forward projection of recent
behaviour. Such a predictor is wrong almost every time by construction, and the error is
concentrated there by a factor of three to nine.

This is not a defect of any model. It is the property of the data that sets the floor under all of
these numbers, and it is the single largest reason the predictions are wrong.

### 3.3 A systematic upward bias, not just noise

**Method.** Signed error, prediction minus truth, in units of each band's own pre-period standard
deviation.

| sensor | model | prompt | bias | spread | bias / spread |
|---|---|---|---|---|---|
| Sentinel-1 | gpt-oss-120b | direct | **+0.478** | 0.589 | **0.81** |
| Sentinel-1 | Kimi-K3 | direct | +0.382 | 0.497 | **0.77** |
| Sentinel-1 | GPT-5.5, 24 April | direct | +0.380 | 0.539 | 0.71 |
| Sentinel-1 | GPT-5.5, 23 April | direct | +0.357 | 0.572 | 0.62 |
| Sentinel-1 | GPT-5.5, 24 April | step-by-step | +0.385 | 0.616 | 0.63 |
| Sentinel-2 | GPT-5.5, 23 April | direct | +0.507 | 1.062 | 0.48 |
| Sentinel-2 | gpt-oss-120b | direct | +0.242 | 0.729 | 0.33 |
| Sentinel-2 | GLM-5.3 | direct | +1.089 | 1.509 | 0.72 |

**Every model, every prompt, both sensors predicts too high.** The bias is 62–81 % of the spread on
Sentinel-1, so a large share of that error is a constant offset rather than random miss. On
Sentinel-1 the bands are backscatter in decibels, so "too high" means the predicted return is too
strong; the withheld August month sits lower than every model expects.

**It is not a model fault.** The same signed error was computed for the no-model rules, which have no
reasoning to blame:

| sensor | repeat P09 | mean last 3 | slope from last 3 | own 9-month mean |
|---|---|---|---|---|
| Sentinel-1 | +0.436 | +0.279 | +0.485 | **−0.046** |
| Sentinel-2 | +0.367 | +0.347 | +0.342 | +0.282 |

Every rule that looks at recent months is too high by about as much as the models are. Only the
site's own nine-month mean escapes it on Sentinel-1, and only because it ignores where the series
recently was.

In plain units, what happened is that **August was a low month**. Averaged over the twelve
Sentinel-1 sites:

| | July (P09) | mean of P07–P09 | August, actual |
|---|---|---|---|
| VV | −8.92 dB | −9.07 dB | **−9.31 dB** |
| VH | −15.20 dB | −15.36 dB | **−15.66 dB** |

August came in 0.38 dB below July on VV, and **below July at all twelve sites**. A model that
predicted "about July's level" was therefore too high, and so was repeating July exactly (too high
by 0.38 dB) and so was averaging the last three months (too high by 0.24 dB).

So this is not the models leaning high. It is August sitting below everything that came before it,
and every method that reads the recent months inherits the miss. No prompt removes it. Subtracting
a constant would remove it, but the constant could only be chosen by looking at the answer.

### 3.4 The step-by-step procedure is degenerate: executed correctly, it returns the last month

**Method.** The same replies read as predictions rather than errors: the gap between each prediction
and the last observed month, in units of that band's pre-period standard deviation. Zero means the
previous month was returned unchanged.

| sensor | model | direct | step-by-step |
|---|---|---|---|
| Sentinel-2 | GPT-5.5, 23 April | 0.406 | **0.008** |
| Sentinel-2 | GPT-5.5, 24 April | 0.389 | **0.014** |
| Sentinel-2 | Kimi-K3 | 0.285 | **0.009** |
| Sentinel-2 | gpt-oss-120b | 0.204 | **0.020** |
| Sentinel-1 | Kimi-K3 | 0.150 | **0.026** |
| Sentinel-1 | GPT-5.5, 23 April | 0.152 | 0.070 |
| Sentinel-1 | gpt-oss-120b | 0.127 | 0.171 |

Checked a second, independent way — against the carry-forward reference scored separately — the
step-by-step arm sits 0.005–0.028 from it on Sentinel-2 with twelve or thirteen of fifteen sites
within 0.01, where the direct arm sits 0.25–0.83 away. The two checks agree, including on where the
collapse is only partial: Sentinel-1 gpt-oss-120b resists on both.

The clearest single piece of evidence is Sentinel-2 site 0003, where **three models from three
organisations, on two gateways, return the same eight numbers to three decimals — and those numbers
are row P09.** They are not producing different answers that happen to score alike; they are all
returning the same row.

**Two different routes reach that one answer, and they must not be conflated.** The traces were read
for whether the five windows are actually worked with:

| model | traces naming all five windows | stating the most recent window alone was selected |
|---|---|---|
| Kimi-K3 | 16 / 27 (26 of 27 name at least one) | **24 / 27** |
| GLM-5.3 | 23 / 27 | **20 / 27** |
| gpt-oss-120b | 4 / 27 (6 name any) | 1 / 27 |
| GPT-5.5 | 0 / 25 (7 name any) | 3 / 25 |

**Route one — the procedure executed correctly.** Kimi-K3 and GLM-5.3 do the block arithmetic and
reach the most recent window on their own. Kimi-K3, closing a trace:

> "…try each window alone against smoothed P06–P09 and keep best; then adding windows refitting
> weights; add if improves fit… **W5-only is plausible.**… Use given VV-VH 6.00 not recompute 5.98.
> **Raw P09** VV-VH given 6.00."

That is the procedure followed faithfully, including the stopping rule. And the most recent window
is P05–P08, whose forward month **is P09** — so a correct execution returns the previous month.

**The estimator is degenerate on this data, and that is not an artifact of a vague threshold.** The
reference implementation uses an improvement threshold of 1 × 10⁻⁴ — it accepts almost any gain at
all — and still keeps **one window at 26 of 27 sites** (the most recent; the twenty-seventh keeps two).
Loosening the rule cannot produce a blend, because there is nothing to add that improves the pre-period
fit. The procedure, correctly executed with the loosest admissible stopping rule, is P09 carried
forward.

**Route two — the procedure abandoned.** gpt-oss-120b barely engages with the windows at all (six
traces mention one, none carry per-window numbers) and says why:

> "Probably smoothing via 3-month moving average?… **I'll just take raw as smoothed (due time).**
> Then windows 4-month averages? **Probably too complex.** I'll guess slight increase trend…"

For GPT-5.5 the route cannot be determined: only summaries are available, and a summarizer may drop
the arithmetic whether or not it happened.

The distinction matters for what can be fixed. If the models were failing to execute, a more precise
instruction would help. Two of the four are executing correctly, and the procedure's own answer is
the last observed month — so a clearer instruction would move gpt-oss-120b and GPT-5.5 *towards* P09,
not away from it. This is why the step-by-step arm's sign differs by sensor: carrying P09 forward
scores 0.492 on Sentinel-1 and 0.789 on Sentinel-2, while the direct prompt scores better than that
on Sentinel-1 and worse on Sentinel-2.

### 3.5 The input month is contaminated

**Method.** The Sentinel-2 site that dominates the mean for every model, examined row by row.

At site 0004 the last observed month carries residual cloud: blue reflectance nearly doubles against
both neighbours (0.045 → 0.086) and the greenness index falls sharply (0.761 → 0.621). The withheld
month returns to the seasonal track (0.054, 0.725).

| predictor | error |
|---|---|
| Kimi-K3, direct | 6.270 |
| GPT-5.5, 23 April, direct | 5.267 |
| band-space SHC | 3.232 |
| gpt-oss-120b, direct | 1.154 |
| mean of the last three months | **0.257** |

Every method anchored on the last month is misled, the band-space pipeline included, and the models
compound it by projecting the contaminated direction further. Averaging over three months dilutes
the bad month and scores an order of magnitude better than any of them. **This is a data-quality
failure before it is a prediction failure**, and it is not addressable by prompting: the information
needed is not in the nine rows the model is given.

Masked-pixel share at the Sentinel-2 treated sites peaks at exactly these months, P07–P09.

### 3.6 The metric concentrates the score in the flattest bands

**Method.** Share of the Sentinel-2 squared error carried by each band, against each band's
denominator.

| band | B2 | B3 | B4 | B8 | B11 | B12 | NDVI | NDWI |
|---|---|---|---|---|---|---|---|---|
| share of total error, GPT-5.5 direct | 13.1 % | 8.8 % | 22.0 % | **1.3 %** | 22.1 % | 17.2 % | 5.3 % | 10.3 % |
| share of total error, gpt-oss direct | 24.2 % | 13.8 % | 21.2 % | **2.7 %** | 18.3 % | 10.4 % | 4.5 % | 4.9 % |
| the band's own SD (mean over sites) | 0.018 | 0.021 | 0.020 | **0.138** | 0.031 | 0.029 | 0.183 | 0.097 |

The near-infrared band carries 1–3 % of the error despite being the largest band and the one that
moves most across the season. The reason is the denominator: its standard deviation is roughly seven
times that of blue, so its variance is around forty-eight times larger, and the same proportional
miss scores far lower there than on a flat band. Red, blue and the shortwave-infrared pair — the
quiet bands — own three quarters of the score.

This is a property of the metric rather than of the models, and it is stated here because it governs
how every other number in this document should be read: **a method is rewarded for getting the flat
bands right, not the informative one.** It is the pipeline's own metric and has not been changed.

### 3.7 Imposing algebra the data does not obey

**Method.** Each prediction checked against the definitions of the quantities it reports, compared
with how far the observed data itself departs from them.

The Sentinel-1 difference band is an exact identity. The observed rows satisfy it to 0.0007
decibels, which is rounding. Predictions:

| model | direct | step-by-step |
|---|---|---|
| GPT-5.5 (both snapshots) | 0.0000 | 0.0142 |
| Kimi-K3 | 0.0017 | 0.0145 |
| gpt-oss-120b | **0.0600** | 0.0083 |

gpt-oss-120b's direct answers contradict themselves by 0.06 dB — it reports a difference that is not
the difference of the two values it just gave.

The Sentinel-2 indices are the opposite case. They are formed per day and each band is composited
independently, so **the observed data does not satisfy the textbook formulas**: the real rows depart
from them by 0.009–0.012, and the withheld month by 0.004.

| model | prompt | departure of the prediction | departure of the truth |
|---|---|---|---|
| GPT-5.5, 23 April | direct | **0.0004** | 0.0039 |
| GPT-5.5, 24 April | direct | **0.0003** | 0.0039 |
| gpt-oss-120b | direct | 0.0186 | 0.0039 |
| Kimi-K3 | direct | 0.0048 | 0.0039 |

GPT-5.5 computes its indices from its own predicted bands exactly, making its answer roughly ten
times more internally consistent than the data it is predicting. That consistency is an error: it
is imposing a relationship the compositing does not preserve. The magnitude is small — a few
thousandths of an index unit, against a standard deviation of 0.18 for greenness — so this is a
minor contributor, listed because it is a genuine defect that the joint metric cannot show.

---

## 4. What the reasoning says, and whether the answer matches it

**Method.** Every stored trace read, not sampled: 324 calls. Two things are extracted. *Said* — which
method the trace names, detected by phrase. *Did* — which of six simple rules built from the same
nine rows the answer is actually closest to, in units of each band's pre-period standard deviation:
repeat P09, mean of the last two, mean of the last three, the site's own nine-month mean, a slope
extrapolation from the last three, and a least-squares line through all nine.

### 4.1 What is available

| model | prompt | calls with text | median length | kind |
|---|---|---|---|---|
| GLM-5.3 | direct / steps | 25 / 27, 27 / 27 | 25,255 / 21,604 | own chain |
| Kimi-K3 | direct / steps | 26 / 27, 27 / 27 | 3,942 / 19,058 | own chain |
| gpt-oss-120b | direct / steps | 27 / 27, 27 / 27 | 280 / 168 | own chain |
| GPT-5.5, 23 April | direct / steps | 21 / 27, 25 / 27 | 488 / 960 | summary |
| GPT-5.5, 24 April | direct / steps | **0 / 27** | — | none returned |
| DeepSeek-V4-Flash | either | 0 / 27 | — | no reply at all |

The research gateway returns each model's own chain, including for calls that never reached an
answer — which is how GLM-5.3's non-termination is visible. The GPT-5 family returns no chain on any
endpoint; the summary column is written by a separate summarizer and is not evidence of what the
model computed. Six of the direct-prompt summaries came back empty.

### 4.2 Said: the method the trace names

Percentage of traces naming each method.

| model | prompt | average | trend / slope | seasonal | smoothing | weights | persistence |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | direct | 19 | **96** | 78 | 0 | 0 | 0 |
| Kimi-K3 | direct | 35 | **92** | **96** | 31 | 8 | 8 |
| GLM-5.3 | direct | 60 | **84** | **100** | 72 | 12 | 20 |
| GPT-5.5 | direct | 0 | 43 | **95** | 0 | 0 | 0 |
| gpt-oss-120b | steps | 33 | 33 | 4 | 33 | 11 | 52 |
| Kimi-K3 | steps | **100** | 89 | 59 | **100** | **100** | 89 |
| GLM-5.3 | steps | **100** | 93 | 30 | **100** | **100** | 96 |
| GPT-5.5 | steps | 76 | 8 | 8 | **88** | **76** | 12 |

Under the direct prompt the models talk about trend and season. Under the step-by-step prompt they
talk about smoothing and weights — that is, they describe the procedure they were given, in 76–100 %
of traces.

### 4.3 Did: the rule the answer actually matches

Counts of sites by nearest rule.

| sensor | model | prompt | repeat P09 | slope from last 3 | mean last 3 | other |
|---|---|---|---|---|---|---|
| Sentinel-2 | GPT-5.5, 23 April | direct | 4 | **9** | 0 | 2 |
| Sentinel-2 | Kimi-K3 | direct | 3 | **10** | 1 | 1 |
| Sentinel-2 | gpt-oss-120b | direct | **10** | 2 | 1 | 2 |
| Sentinel-2 | GPT-5.5, 23 April | steps | **15** | 0 | 0 | 0 |
| Sentinel-2 | gpt-oss-120b | steps | **15** | 0 | 0 | 0 |
| Sentinel-2 | Kimi-K3 | steps | **8** | 0 | 0 | 0 |
| Sentinel-1 | GPT-5.5, 23 April | direct | 3 | 2 | **5** | 2 |
| Sentinel-1 | Kimi-K3 | direct | 3 | 1 | **4** | 4 |
| Sentinel-1 | GPT-5.5, 23 April | steps | **11** | 0 | 0 | 1 |
| Sentinel-1 | Kimi-K3 | steps | **11** | 0 | 0 | 0 |

**Not one answer under the step-by-step prompt matches a weighted combination of anything.** Every
Sentinel-2 answer, for every model, matches repeating the previous month.

### 4.4 Said against did

Of the calls whose step-by-step trace names averaging, what the answer actually was:

| trace names averaging | repeat P09 | mean last 3 | own 9-month mean | line through all 9 |
|---|---|---|---|---|
| yes | **42** | 2 | 2 | 1 |
| no | 23 | 1 | 0 | 0 |

Forty-two of forty-seven traces that describe averaging produced the previous month unchanged.

The distance to the nearest rule separates the two prompts more sharply than any accuracy number in
this document:

| sensor | model | direct | step-by-step |
|---|---|---|---|
| Sentinel-2 | GPT-5.5, 23 April | 0.221 | **0.008** |
| Sentinel-2 | GPT-5.5, 24 April | 0.187 | **0.008** |
| Sentinel-2 | Kimi-K3 | 0.149 | **0.009** |
| Sentinel-2 | gpt-oss-120b | 0.126 | **0.009** |
| Sentinel-1 | Kimi-K3 | 0.040 | **0.005** |
| Sentinel-1 | GPT-5.5, 23 April | 0.059 | **0.007** |

Under the direct prompt the answers sit between the simple rules — a tenth to a third of a standard
deviation from the nearest one — which is what a genuine computation looks like. Under the
step-by-step prompt they sit *on* a rule, to eight thousandths of a standard deviation. The answer
has stopped being computed.

**The consequence is about evidence, not accuracy.** Under the direct prompt the traces are broadly
faithful: trend and season are named in 84–96 % of them, and a slope extrapolation is the rule the
Sentinel-2 answers match. Under the step-by-step prompt the trace describes smoothing, block
selection and convex weights while the answer is a copied row. **A reasoning trace from this arm
cannot be taken as an account of what was computed** — it reports the instructions back, not the
work. Every claim in §3 rests on the predictions rather than on the traces for this reason.

---

## 5. Eight cases

**Method.** One correct and one incorrect case for each sensor and each prompt, three models side by
side. Sentinel-1 site 0022 appears twice on purpose: the same series under both prompts.

Reasoning is quoted with its kind marked. The research gateway returns the model's own chain. The
GPT-5 family returns no chain on any endpoint — the vendor states that raw reasoning tokens are not
exposed — so what is quoted for GPT-5.5 is a summary written by a separate summarizer.

### 5.1 Sentinel-1, direct prompt — correct: site 0020

| | VV | VH | VV−VH | error |
|---|---|---|---|---|
| P07 | −8.38 | −14.82 | 6.45 | |
| P08 | −8.80 | −15.22 | 6.42 | |
| P09 | −8.45 | −14.91 | 6.46 | |
| **P10 truth** | **−8.71** | **−15.16** | **6.45** | |
| GPT-5.5 | −8.55 | −15.00 | 6.45 | **0.070** |
| Kimi-K3 | −8.54 | −14.98 | 6.44 | 0.081 |
| gpt-oss-120b | −8.54 | −15.00 | 6.44 | 0.077 |
| band-space SHC | | | | 0.184 |
| mean of last three months | | | | 0.079 |

All three land within 0.18 decibels of the withheld month, and at this site their answers coincide
with the mean of the last three months, which scores 0.079 on the same cells. gpt-oss-120b's chain
shows that arithmetic: *"Use simple average of last few months (May-Jul). VV avg =
(−8.38−8.80−8.45)/3 = −8.543."* This is one of the rare sites where the withheld month did not
reverse sharply.

### 5.2 Sentinel-1, direct prompt — wrong: site 0022 (cause §3.2)

| | VV | VH | VV−VH | error |
|---|---|---|---|---|
| P07 | −10.19 | −15.99 | 5.81 | |
| P08 | −10.12 | −15.82 | 5.71 | |
| P09 | −9.81 | −15.41 | 5.58 | |
| **P10 truth** | **−10.17** | **−15.96** | **5.79** | |
| GPT-5.5, 23 April | −9.70 | −15.16 | 5.46 | 0.903 |
| Kimi-K3 | −9.60 | −15.30 | 5.70 | 0.700 |
| gpt-oss-120b | −9.50 | −15.20 | 5.70 | 0.948 |
| band-space SHC | | | | 0.440 |
| mean of last three months | | | | **0.067** |

Three months of rise, then the withheld month gives it all back. Every model extrapolates —
gpt-oss-120b: *"Slight increase in VV (less negative) over time: from −11.56 Jan to −9.81 July.
Linear trend maybe +0.3 per month… So predict August −9.5 approx."* All three land above the truth,
which is §3.3's bias in one site. The three-month mean scores 0.074 because averaging does not
commit to the direction.

### 5.3 Sentinel-1, step-by-step prompt — correct: site 0022 (cause §3.4)

| | VV | VH | VV−VH | error | distance from P09 |
|---|---|---|---|---|---|
| **P10 truth** | **−10.17** | **−15.96** | **5.79** | | |
| GPT-5.5 | −9.81 | −15.41 | 5.58 | 0.442 | **0.008** |
| Kimi-K3 | −9.81 | −15.41 | 5.58 | 0.442 | **0.008** |
| gpt-oss-120b | −9.70 | −15.30 | 5.60 | 0.622 | 0.147 |
| band-space SHC | | | | 0.440 | |

The same series. Two of three return row P09 to the digit and score what the band-space pipeline
scores. The error halves because the model stopped predicting, not because it predicted better.

### 5.4 Sentinel-1, step-by-step prompt — wrong: site 0027 (cause §3.4)

| | VV | VH | VV−VH | error | distance from P09 |
|---|---|---|---|---|---|
| P09 | −9.27 | −15.60 | 6.35 | | |
| **P10 truth** | **−9.64** | **−16.00** | **6.36** | | |
| GPT-5.5, 23 April | −9.27 | −15.31 | 6.05 | 0.725 | 0.325 |
| GPT-5.5, 24 April | −9.27 | −15.60 | 6.35 | 0.420 | 0.005 |
| Kimi-K3 | −9.27 | −15.60 | 6.35 | 0.420 | 0.005 |
| gpt-oss-120b | −9.27 | −15.60 | 6.35 | 0.420 | 0.005 |
| band-space SHC | | | | 0.415 | |
| mean of last three months | | | | 0.239 | |

The truth moves away from P09, so the same collapse is punished. Under the direct prompt the
23 April snapshot scored 0.331 here and the 24 April snapshot 0.362; the instructions roughly
doubled the first and worsened the second.

### 5.5 Sentinel-2, direct prompt — correct: site 0014

A series with no cloud disturbance: GPT-5.5 0.114, gpt-oss-120b 0.062, the band-space pipeline 0.036.
All small. Sites like this are why the Sentinel-2 *medians* sit at 0.201–0.240 across methods while
the *means* diverge — the means are set by a handful of contaminated sites, of which the next is the
clearest.

### 5.6 Sentinel-2, direct prompt — wrong: site 0004 (cause §3.5)

| | B2 | B8 | NDVI | NDWI | error |
|---|---|---|---|---|---|
| P07 | 0.052 | 0.467 | 0.792 | −0.690 | |
| P08 | 0.045 | 0.384 | 0.761 | −0.676 | |
| **P09** | **0.086** | 0.396 | **0.621** | **−0.582** | |
| **P10 truth** | **0.054** | **0.360** | **0.725** | **−0.651** | |
| GPT-5.5, 23 April | 0.078 | 0.294 | 0.481 | −0.500 | 5.267 |
| Kimi-K3 | 0.080 | 0.320 | 0.471 | −0.524 | 6.270 |
| gpt-oss-120b | 0.072 | 0.382 | 0.560 | −0.600 | 1.154 |
| band-space SHC | | | | | 3.232 |
| mean of last three months | | | | | **0.257** |

Worked through in §3.5. Note the models are wrong in the same direction as the contamination, and
wrong by more than simply repeating it would have been.

### 5.7 Sentinel-2, step-by-step prompt — correct: site 0012 (cause §3.4)

| | B2 | B8 | NDVI | error |
|---|---|---|---|---|
| P09 | 0.059 | 0.440 | 0.814 | |
| **P10 truth** | **0.033** | **0.359** | **0.858** | |
| GPT-5.5 | 0.059 | 0.440 | 0.814 | 0.909 |
| gpt-oss-120b | 0.059 | 0.440 | 0.814 | 0.909 |
| Kimi-K3 | *no answer — degenerate loop* | | | |
| band-space SHC | | | | 0.917 |
| mean of last three months | | | | **0.393** |

A fall of two thirds — the 24 April snapshot goes 2.721 → 0.909 — consisting entirely of returning row P09
unchanged. The three-month mean would still have been more than twice as good. The largest single
fall in the comparison is at site 0004, on the metric's own terms: 9.178 → 3.235.

### 5.8 Sentinel-2, step-by-step prompt — wrong: site 0003 (cause §3.4)

| | B2 | B3 | B4 | B8 | B11 | B12 | NDVI | NDWI | error |
|---|---|---|---|---|---|---|---|---|---|
| P09 | 0.041 | 0.061 | 0.035 | 0.431 | 0.190 | 0.079 | 0.856 | −0.755 | |
| **P10 truth** | **0.074** | **0.089** | **0.060** | **0.453** | **0.202** | **0.097** | **0.774** | **−0.677** | |
| GPT-5.5 | 0.041 | 0.061 | 0.035 | 0.431 | 0.190 | 0.079 | 0.856 | −0.755 | 1.179 |
| Kimi-K3 | 0.041 | 0.061 | 0.035 | 0.431 | 0.190 | 0.079 | 0.856 | −0.755 | 1.179 |
| gpt-oss-120b | 0.041 | 0.061 | 0.035 | 0.431 | 0.190 | 0.079 | 0.856 | −0.755 | 1.179 |
| band-space SHC | | | | | | | | | 1.164 |

Three models, three organisations, two gateways, one identical answer across all eight bands — row
P09. The withheld month rises sharply on six of eight bands and all three are wrong together.

---

## 6. What the causes imply

Ordered by how much error each would remove, not by how easy it is.

1. **Reversal (§3.2) sets the floor and no prompt reaches it.** Any predictor that projects recent
   behaviour is wrong on the 33 of 36 Sentinel-1 band-months that reverse. What helps is not
   committing to a direction — which is exactly why the three-month mean leads on both sensors
   (0.265 and 0.541) while every model and the band-space pipeline sit above it.
2. **The upward bias (§3.3) is not a model fault and not a prompt target.** The no-model rules carry
   the same offset (+0.28 to +0.49), so it belongs to the withheld month. The one unbiased predictor
   is the site's own nine-month mean on Sentinel-1 (−0.046), which ignores the recent level
   entirely — the same lesson as §3.2.
3. **Contamination (§3.5) is a data problem.** The information needed to discount a clouded month
   is not in the nine rows given to the model. Passing a quality flag, or a cloud-fraction column,
   would be a change to the evidence rather than to the prompt.
4. **The step-by-step prompt (§3.4) cannot be fixed by clearer wording.** Two of the four models
   execute it correctly and reach the most recent window on their own; the procedure's own answer is
   the previous month. A more precise instruction would move the other two towards that answer, not
   away from it. Its apparent Sentinel-2 gain is the degenerate answer scoring well on that sensor,
   not better reasoning. Changing the estimator — the window set, the stopping rule — would change
   the result; changing the prose will not.
5. **Read every number here through §3.6.** The metric rewards the flat bands; the near-infrared
   band, which carries the seasonal signal, contributes 1–3 % of the score.

---

## 7. Sources

All figures are computed from replies already stored; no model was called for this document.

| what | where |
|---|---|
| aggregate tables of §2 | `notebooks/llm_band_sc/lb_arms.py` → `lb_arms_cells.csv`, `lb_arms_per_site.csv` |
| failure diagnostics of §3.1–3.3, §3.6, §3.7 | `notebooks/llm_band_sc/lb_why.py` |
| trace availability, said-against-did of §4 | `notebooks/llm_band_sc/lb_traces.py` → `lb_traces.csv` |
| collapse checks of §3.4 and the cases of §5 | `notebooks/llm_band_sc/lb_cases.py` |
| reference rows (carry-forward, three-month mean, own mean, band-space SHC) | `notebooks/llm_band_sc/lb_scores_shc.csv` |
| run-averaged per-site values | `notebooks/llm_band_sc/lb_runs.py` → `lb_summary_shc_runs.csv` |
| prompts as sent | `lb_prompts_shc.jsonl`, `lb_prompts_shc_steps.jsonl` |
| raw replies, one folder per run | `notebooks/llm_band_sc/runs/` |
| the band-space SHC re-run and its gates | `notebooks/llm_band_sc/lb_shc.py`, `lb_tests.py` |

Reproduce with `python lb_arms.py`, `python lb_why.py`, `python lb_traces.py` and `python lb_cases.py` in
`notebooks/llm_band_sc/`.
