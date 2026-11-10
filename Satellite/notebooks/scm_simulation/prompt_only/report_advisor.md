# Where language and vision language models could add to disaster effect estimation

## Language and vision language models are meant to add what numbers and statistics cannot.

During the meeting, we think we should saw the two as complementary. The work starts with numbers because an earlier attempt to work from images did not go through. A prompt on numbers is the simplest setting in which a model's own ability can be checked against a known truth, before images.

A statistical method works only on the numbers it is given. A model is meant to add four things it cannot.

- **Meaning.** A statistical method treats a series as numbers, while a language model reads what they stand for, the vegetation index of a site hit by a natural disaster. The meeting traced the weak results of earlier prompts on raw sensor bands to bands carrying no meaning for the model, and chose the vegetation index for that reason.
- **Text.** Information that is not a number, such as social media posts, can be read by a language model and not by a statistical method. The meeting saw this as a strong application.
- **Images.** Features built band by band lose part of what an image shows and cannot rebuild it. A description of the image by a vision language model may carry what they lose.
- **Explanation.** A statistical fit returns weights and an estimate. A language model also writes how it reached its answer, read after the prediction as an explanation.

## Step 1 tested the models on simulated data, with no method named, without and with donors.

Step 1 is the simplest setting in which a model's own answer can be checked.

- **Known truth.** The simulated disaster lowers the target by 3 in every period from 41 to 60, and the untreated path is known. Every answer can be scored exactly, which real data does not allow.
- **No method named.** The prompt never mentions synthetic control or any other method, so the answer shows what the model does on its own. The reasoning it writes shows how it decided.
- **Without and with donors.** Without donors the model has only the target's own past. With donors it also has untreated sites as a reference for what the target would have done, and comparing the two shows whether the model uses that reference.

The test used six panels from the collaborator's synthetic control simulation study, at three noise levels (σ 1.0, 0.5 and 0.2) for each of two data generating processes. DGP 1 has fixed covariates and DGP 2 time varying ones. Each panel holds one treated unit and 29 untreated donors over 60 periods.

- **Prompt.** The model saw the target's series for periods 1 to 40, described as the vegetation index of a site that a natural disaster hits at period 41, and was asked for the counterfactual state of each period from 41 to 60. The version with donors added the 29 donors for periods 1 to 60, described as sites with no disaster.
- **Models.** gpt-oss-120b, GLM-5.3 and DeepSeek-V4.1-Flash, one run per case at default reasoning effort and with no length limit.
- **Scoring.** The effect is calculated afterwards as the mean over periods 41 to 60 of the observed value minus the model's counterfactual, as the study's own code does. It is scored by its bias, the estimated effect minus the true effect of −3.
- **Reading.** Every reasoning chain was read in full by Grok 4.7, and each quote was checked against its record by exact search.

### Without donors the models forecast from the target's past, and with donors they worked from the donor average.

| method | model | σ 1.0, DGP 1 | σ 1.0, DGP 2 | σ 0.5, DGP 1 | σ 0.5, DGP 2 | σ 0.2, DGP 1 | σ 0.2, DGP 2 |
|---|---|---|---|---|---|---|---|
| collaborator's synthetic control, 1,000 replications | | −0.017 | +0.022 | −0.008 | +0.012 | −0.003 | +0.004 |
| prompt, target alone | gpt-oss-120b | −0.732 | −0.512 | +0.225 | −0.129 | −0.108 | −0.190 |
| prompt, target alone | GLM-5.3 | −0.150 | −0.798 | −0.067 | no answer | no answer | no answer |
| prompt, target alone | DeepSeek | −0.141 | −0.790 | −0.033 | −0.410 | +0.008 | −0.180 |
| prompt, with donors | gpt-oss-120b | −0.383 | −0.674 | −0.065 | −0.350 | +0.116 | −0.155 |
| prompt, with donors | GLM-5.3 | +0.050 | no answer | no answer | no answer | no answer | no answer |
| prompt, with donors | DeepSeek | +0.048 | −0.792 | +0.096 | −0.400 | +0.100 | −0.145 |
| skill, model sets the fitter's settings | gpt-oss-120b | +0.181 | −0.687 | +0.216 | −0.380 | +0.251 | −0.173 |
| skill, model sets the fitter's settings | GLM-5.3 | +0.326 | −0.045 | +0.151 | −0.193 | +0.014 | −0.117 |
| skill, model sets the fitter's settings | DeepSeek | no answer | no answer | no answer | no answer | +0.027 | no answer |
| skill, model sets settings and two design choices | gpt-oss-120b | +0.086 | −0.624 | +0.141 | −0.268 | +0.199 | −0.130 |
| skill, model sets settings and two design choices | GLM-5.3 | +0.326 | −0.159 | +0.151 | −0.170 | +0.014 | −0.110 |
| skill, model sets settings and two design choices | DeepSeek | no answer | no answer | no answer | no answer | no answer | no answer |
| skill, model makes four design choices | gpt-oss-120b | +0.165 | −0.516 | +0.023 | −0.247 | +0.007 | −0.098 |
| skill, model makes four design choices | GLM-5.3 | +0.165 | −0.336 | +0.059 | −0.247 | +0.007 | −0.102 |
| skill, model makes four design choices | DeepSeek | no answer | no answer | no answer | −0.247 | +0.007 | −0.209 |
| skill written by GPT-5.4 | all three | +0.165 | −0.516 | +0.059 | −0.247 | +0.007 | −0.102 |
| skill written by DeepSeek | all three | +0.075 | −0.317 | +0.118 | −0.505 | +0.145 | −0.284 |

Bias is the estimated effect minus the true effect of −3 for every row, where the effect is the mean over periods 41 to 60 of observed minus counterfactual. The collaborator's row is the mean bias over 1,000 replications, as printed in the study. Every other row is one run on the first replication, the panel shipped with the study. Prompt rows give the model the series and ask for the counterfactual with no method named. Skill rows come from an earlier experiment in which the same models ran a synthetic control fit through code, and each skill written by another model gave the same answer for all three models. GPT-5.5 skill runs are left out, as the prompt test did not run GPT-5.5. No answer marks a prompt reply the server closed at 30 minutes, or a skill run that reached its 12 step limit, and is counted as wrong. Prompt runs used default reasoning effort, except three gpt-oss-120b answers (σ 1.0 DGP 1 with donors, σ 0.5 DGP 1 alone and with donors) taken from a rerun at low effort, and skill runs used low effort.

**Without donors, the models forecast the target from its own past.** The chains fitted autoregressive models, the pre-period mean, a straight line or exponential smoothing, and most paths settle near the target's mean.

**With donors, every chain named synthetic control but none could fit its weights by hand.** All 20 reasoning chains with donors name synthetic control, though the prompt names no method, and each judges the weight fit too heavy to do by hand. The default gpt-oss-120b chains averaged the 29 donors with equal weight, GLM-5.3 and DeepSeek regressed the target on the donor mean, and one DeepSeek answer fitted three donors.

GLM-5.3 reasoned until the server closed the reply at 30 minutes in 8 of its 12 cases, each time partway through a hand calculation.

**The written reply does not always match the reasoning.** In four of the six default gpt-oss-120b replies with donors, the reply describes synthetic control, a regression or fitted weights, while the chain averaged the donors with equal weight. The two low effort gpt-oss-120b answers with donors trace to no step in their chains, so the account of how a model decided has to come from its reasoning chain.

**What step 1 shows.** On numbers alone, the models given donors tried to do the work of a statistical method by hand, and fitting the weights is the part that suits statistics. Whether the meaning of a real vegetation index changes what a model does is the question for step 2.

## Step 2 repeats the test on real vegetation index data.

- **Why.** A real vegetation index carries meaning the simulated numbers lack, such as the season and what a hurricane does to vegetation. This is where a model may add something statistics cannot.
- **Design.** The same two prompts, with no method named, on NDVI from the Hurricane Helene Sentinel-2 sites, without and with untreated sites as donors.
- **Reference.** Real data has no true untreated path, so synthetic control is the reference for the version with donors and historical synthetic control for the series alone.

## Later steps bring in forecasting models and images.

Step 3 compares time series foundation models, forecasting the same NDVI series zero shot, with the language models. After that come vision language models on the images, language models combined with statistics, and a predicted picture for real-time monitoring.

## A vision language model could enter the work in three ways.

1. **It gives the counterfactual itself.** The model receives target and donor images, reads their relationship before the disaster as context, and answers in text with the counterfactual state and the effect. The answer is scored in the same benchmark as the other methods.
2. **It supplies vectors for synthetic control.** A satellite foundation model turns each image into an embedding or a land cover vector. The vectors give the distances between target and donors, the change over time and the season, which synthetic control then uses.
3. **Its description adds information.** Features built band by band lose part of what an image shows. A description of the image may carry that part into the statistics or the prompt.

No language or vision language model generates an image, so a predicted picture would come from an encoder and decoder that rebuild the image from its embedding.

## Advice is sought on six questions.

### Language models

1. Should success for a prompt-only answer mean closeness to synthetic control, or closeness to the truth within a set margin?
2. Is one run per case enough, or should each case be repeated to measure how much answers vary?
3. On real NDVI with no true path, are synthetic control and historical synthetic control the right references, and should held-out periods before the disaster also be scored?

### Vision language models

4. Which of the three uses is the contribution, and in what order?
5. Which image should the model receive (an RGB preview or an NDVI image), and which model (a general vision language model, GeoChat or a satellite foundation model)?
6. How can the part of an answer that comes from the image be separated from the part that comes from the prompt text?
