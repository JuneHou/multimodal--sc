# Ranked literature outline for latent synthetic control and REAP

## How to read this outline

This ranking covers all **34 local PDFs** in `literatures/`. It ranks relevance to the project—not general scholarly quality—using two five-point scores:

- **Similarity**: overlap with the project's theory, estimator, satellite/disaster setting, or latent-mixture validation.
- **Contribution**: how directly the work can improve the active REAP proposal, empirical design, validation, or decision-support deliverables.

The active project priority is: **Objective 3 / Module 2** (counterfactual prediction with synthetic control), followed by **Objective 4** (validation), **Objective 2 / Module 1** (image features and embeddings), and **Objective 5 / Module 3** (decision support). The clinical-text preliminary is treated as a validation study for convex mixing in learned representations, not as evidence of general LLM ability.

## Proposal aim and active project priorities

The proposal, **“Causal AI for Real-Time Agricultural Shock Monitoring,”** aims to build a multimodal platform that detects an agricultural or environmental shock, measures the factual condition from rapidly available data, estimates what would have happened without the shock, calculates the causal impact, and translates that evidence into stakeholder-ready decisions. Its central motivation is timing: satellite imagery and related signals arrive within days, while yields, income, employment, and other official outcomes may arrive months or years later.

The full proposal begins with **Objective 1**, a reproducible multimodal pipeline joining remote sensing, weather, land use, geospatial context, and later administrative or economic data while defining treated and comparison areas. The current REAP Helene dataset largely supplies this foundation, so the active methodological priorities are:

1. **Objective 3 / Module 2 — counterfactual prediction with synthetic control.** Estimate the no-shock condition from comparable unaffected locations. Interpretable feature-space synthetic control is the transparent primary method; multimodal covariate adjustment and foundation-model latent/image-space imputation are extensions that must be validated rather than assumed.
2. **Objective 4 — causal impact estimation and validation.** Define impact as the factual condition minus the estimated no-shock counterfactual, then test credibility with pre-shock prediction, placebos, sensitivity analyses, leave-one-region-out evaluation, and later-released administrative outcomes. The clinical-text planted-mixture study contributes here as a necessary geometric validity test for learned representations, not as evidence that an LLM can recover real causal effects.
3. **Objective 2 / Module 1 — factual image features and embeddings.** Produce timely observed-state measures such as NDVI/EVI, moisture and flood proxies, crop or vegetation stress, land-cover change, temperature, and learned remote-sensing embeddings. These measurements describe what happened and become inputs to the causal model, but factual prediction alone does not answer what would have happened without the shock.
4. **Objective 5 / Module 3 — decision support and translation.** Convert factual-versus-counterfactual differences into interpretable indicators, affected-area estimates, maps, rankings, uncertainty intervals, alerts, dashboards, and plain-language summaries for Extension agents, emergency managers, lenders, insurers, businesses, and policymakers. The final deliverables include a prototype dashboard, methodological paper, stakeholder brief, and concept for a larger external grant.

## Tier A — Core methods and closest applied precedent

### 1. Ben-Michael, Feller, and Rothstein (2021), *The Augmented Synthetic Control Method*
**Scores:** Similarity 5/5 · Contribution 5/5  
**File:** `LLM and Econometrics/The Augmented Synthetic Control Method-Ben-Michael-2021.pdf`  
**Alignment:** Objective 3 / Module 2; Objective 4.

The paper bias-corrects ordinary simplex synthetic control with an outcome model—preferably ridge regression—when exact pre-treatment balance is impossible, using cross-validation and diagnostics to control the trade-off between fit and extrapolation. This is REAP's most actionable extension because the current single-composite, ten-donor design may not place each treated site inside the donor convex hull.

### 2. Athey, Bayati, Doudchenko, Imbens, and Khosravi (2021), *Matrix Completion Methods for Causal Panel Data Models*
**Scores:** Similarity 5/5 · Contribution 5/5  
**File:** `LLM and Econometrics/✅matrix_completion_Susan_Athey_2021.pdf`  
**Alignment:** Objective 3 / Module 2; Objective 4.

The method treats treated-post outcomes as missing cells in a panel and imputes them with nuclear-norm-regularized low-rank matrix completion plus unit and time fixed effects. Once monthly Sentinel panels are available, it should be the principal robustness benchmark against donor-weighted SC because it tests whether shared latent factors predict the counterfactual better than sparse convex weights.

### 3. Rho, Illick, Narasipura, Abadie, Hsu, and Misra (2026), *Time-Aware Synthetic Control*
**Scores:** Similarity 5/5 · Contribution 4/5  
**File:** `LLM and Econometrics/✅TIME-AWARE SYNTHETIC CONTROL-Saeyoung Rho-2026.pdf`  
**Alignment:** Objective 3 / Module 2; Objective 4.

TASC fits a low-rank linear-Gaussian state-space model using expectation-maximization, Kalman filtering, and Rauch–Tung–Striebel smoothing so temporal order and trends affect the counterfactual. It is a promising exploratory benchmark for a future monthly REAP panel, but it is a recent preprint and does not preserve interpretable simplex donor weights.

### 4. Elliott, Strobl, and Sun (2015), *The Local Impact of Typhoons on Economic Activity in China: A View from Outer Space*
**Scores:** Similarity 4/5 · Contribution 5/5  
**File:** `Satellite Image/The local impact of typhoons on economic activity in China- A view from outer space- Robert J.R. Elliott-2015.pdf`  
**Alignment:** Objectives 1–2; Objective 5 / Module 3.

The paper combines storm tracks, a physical wind-field/damage model, 1-km annual nightlights, cell and year fixed effects, and simulated storm tracks to estimate local economic losses and future risk. It is the closest local applied precedent for turning physically grounded disaster exposure and satellite outcomes into economic decision products, although it uses fixed effects rather than synthetic control.

### 5. Hatamyar, Kreif, Rocha, and Huber (2023), *Machine Learning for Staggered Difference-in-Differences and Dynamic Treatment Effect Heterogeneity*
**Scores:** Similarity 4/5 · Contribution 4/5  
**File:** `LLM and Econometrics/✅Machine Learning for Staggered DiD and Dynamic Treatment Effects.pdf`  
**Alignment:** Objective 3 / Module 2; Objective 4.

MLDID combines group-time staggered DiD, cross-fitted ML nuisance models, robust/orthogonal scores, and R-learning to estimate dynamic conditional treatment effects. It is not needed for one common Helene event, but it becomes important if REAP expands to multiple disasters, different exposure timings, or heterogeneous recovery trajectories.

### 6. Athey (2017), *Beyond Prediction: Using Big Data for Policy Problems*
**Scores:** Similarity 4/5 · Contribution 4/5  
**File:** `LLM and Econometrics/✅Beyond prediction- Using big data for policy problems -Susan Athey-2017.pdf`  
**Alignment:** Proposal-wide causal framing.

This essay explains why predictive accuracy alone cannot answer policy questions that require potential outcomes, treatment effects, and alternative assignment decisions. It provides the clearest justification for using foundation models only as measurement/representation tools while matched DiD, SC, and placebo tests carry REAP's causal claims.

### 7. Athey, Cole, Nath, and Zhu (2026), *Targeting, Personalization, and Engagement in an Agricultural Advisory Service*
**Scores:** Similarity 3/5 · Contribution 5/5  
**File:** `Information and Beliefs/Targeting, Personalization, And Engagement In An Agricultural Advisory Service .pdf`  
**Alignment:** Objective 5 / Module 3; Objective 4.

Across randomized call-time experiments serving over one million farmers, the paper compares off-policy predictions with prospective on-policy RCTs and uses transfer learning, flexible policy models, uncertainty penalties, and constrained optimization to handle temporal drift and equity/capacity goals. It is REAP's strongest guide for converting impact estimates into an Extension-facing service and for verifying that retrospective model gains survive real deployment.

### 8. Manski (2000), *Economic Analysis of Social Interactions*
**Scores:** Similarity 3/5 · Contribution 4/5  
**File:** `Information and Beliefs/Economic Analysis of Social Interactions.pdf`  
**Alignment:** Objective 4; causal-assumption audit.

Manski separates endogenous, contextual, and common-shock interactions and formalizes why correlated outcomes do not by themselves identify peer or spillover effects. This directly sharpens REAP's interference problem: treated and control sites shared the hurricane, so spatial dependence and common exposure must be separated from the effect of mapped landslide occurrence.

## Tier B — Validation, experimental design, and decision infrastructure

### 9. Haaland, Roth, and Wohlfart (2023), *Designing Information Provision Experiments*
**Scores:** Similarity 2/5 · Contribution 5/5  
**File:** `Information and Beliefs/🏷️Designing Information Provision Experiments.pdf`  
**Alignment:** Objective 5 / Module 3; Objective 4.

This methodological survey covers prior/posterior belief elicitation, randomized information treatments, active controls, demand effects, follow-up measurement, and statistical power. It is the best design reference for testing whether REAP maps, uncertainty intervals, and counterfactual explanations change stakeholder beliefs or decisions, but it contributes nothing to estimating the satellite counterfactual itself.

### 10. Wald (1949), *Statistical Decision Functions*
**Scores:** Similarity 2/5 · Contribution 4/5  
**File:** `Information and Beliefs/🏷️Statistical Decision Functions.pdf`  
**Alignment:** Objective 5 / Module 3.

Wald formalizes statistical decisions through states, actions, experiments, losses, Bayes rules, minimax rules, admissibility, and risk. Its project contribution is to require an explicit decision and asymmetric loss function—rather than treating smaller prediction error as the final goal—when REAP turns uncertain effects into alerts or resource priorities.

### 11. Agarwal et al. (2017), *Making Contextual Decisions with Low Technical Debt*
**Scores:** Similarity 2/5 · Contribution 4/5  
**File:** `ML/Making Contextual Decisions with Low Technical Debt.pdf`  
**Alignment:** Objective 5 / Module 3; Objective 4.

The paper builds a contextual-bandit Decision Service around explore–log–learn–deploy, randomized propensity logging, inverse-propensity policy evaluation, reproducible replay, monitoring, and safeguards. It offers a practical architecture for an auditable REAP decision platform, though its randomized online-action counterfactuals are not transferable to observational hurricane identification.

### 12. Ruhrberg Estévez et al. (2026), *Causal Inference and Digital Twins: A Roadmap for the Future of Clinical Trials*
**Scores:** Similarity 3/5 · Contribution 3/5  
**File:** `LLM and Econometrics/✅Causal inference and digital twins- a roadmap for the future of clinical trials.pdf`  
**Alignment:** Clinical preliminary; Objective 4.

The perspective combines estimand definition, heterogeneous-effect methods, transportability, mechanistic/data-driven twins, hybrid controls, calibration, and prospective validation under different regulatory contexts. It supports the project's distinction between simulation and identified counterfactual estimation, but it presents a roadmap rather than a new estimator and is clinically specific.

### 13. Manning, Zhu, and Horton (2024), *Automated Social Science: Language Models as Scientist and Subjects*
**Scores:** Similarity 3/5 · Contribution 3/5  
**File:** `Agent and Experiments/Automated Social Science_Language Models as Scientist and Subjects.pdf`  
**Alignment:** Clinical preliminary; Objective 4.

The system uses structural causal models to generate hypotheses, agent attributes, randomized in-silico experiments, and pre-specified analyses across several economic settings. Its useful contribution is DAG/estimand discipline and explicit simulation validation; its LLM-generated effect sizes do not establish transport to patients, satellite images, or real disasters.

### 14. Xie, Yuan, Mei, and Jackson (2025), *Using Language Models to Decipher the Motivation Behind Human Behaviors*
**Scores:** Similarity 3/5 · Contribution 3/5  
**File:** `Agent and Experiments/Using Language Models to Decipher the Motivation Behind Human Behaviors.pdf`  
**Alignment:** Clinical preliminary.

The paper learns thousands of behavioral prompts and fits convex mixtures of LLM-induced choice distributions by minimizing Wasserstein distance to large human experimental datasets, with embeddings used to organize motivation semantics. This is the closest LLM-side mathematical analogy to planted donor mixtures, but its fitted behavioral weights are not known-truth causal weights and do not validate latent synthetic control.

### 15. Ashokkumar, Hewitt, Ghezae, and Willer (2026), *Large Language Models Can Predict the Results of Social Science Experiments*
**Scores:** Similarity 2/5 · Contribution 3/5  
**File:** `Agent and Experiments/Large language models can predict the results of social science experiments.pdf`  
**Alignment:** Clinical preliminary; Objective 4.

The authors generate demographically weighted synthetic responses and compare predicted treatment effects with hundreds of preregistered survey and megastudy effects. High correlations on text surveys but inflated magnitudes and weaker field/non-text performance provide a useful warning: success on clinical text cannot be assumed to transfer to satellite imagery without new planted and real-data validation.

### 16. Manning and Horton (2026), *General Social Agents*
**Scores:** Similarity 2/5 · Contribution 3/5  
**File:** `Agent and Experiments/General Social Agents.pdf`  
**Alignment:** Clinical preliminary; Objective 4.

The paper constructs theory-guided mixtures of LLM personas from seed-game human data and validates them across a precommitted population of new economic games. Its cross-environment holdout design is a useful model for testing representation generalization, but predicting choices in a bounded game family is not causal counterfactual identification.

### 17. Park et al. (2024), *Generative Agent Simulations of 1,000 People*
**Scores:** Similarity 2/5 · Contribution 3/5  
**File:** `Agent and Experiments/Generative Agent Simulations of 1,000 People.pdf`  
**Alignment:** Clinical preliminary; Objective 4.

The study conditions GPT-4o agents on long interviews from 1,052 people and validates their responses on surveys, personality measures, economic games, and experimental replications. It provides a rich individualized-agent validation template, but matching a person's observed answers does not identify that person's untreated potential outcome.

### 18. Agrawal, Athey, Kanodia, Nath, and Palikot (2026), *The Economics of Algorithmic Personalization*
**Scores:** Similarity 2/5 · Contribution 3/5  
**File:** `Information and Beliefs/The Economics of Algorithmic Personalization Evidence from an Educational Technology Platform.pdf`  
**Alignment:** Objective 5 / Module 3; Objective 4.

The study evaluates collaborative-filtering recommendations with an RCT and then monitors scaled deployment through a sharp regression-discontinuity threshold. It offers a useful template for building deployment rules that preserve causal evaluation as a REAP decision product evolves, but its engagement setting and local RDD estimand are remote from satellite counterfactuals.

### 19. Manski (2004), *Measuring Expectations*
**Scores:** Similarity 2/5 · Contribution 3/5  
**File:** `Information and Beliefs/Measuring Expectations.pdf`  
**Alignment:** Objective 5 / Module 3.

Manski develops probabilistic survey elicitation because observed choices generally cannot separately identify preferences and expectations. REAP could use this approach to elicit stakeholders' probability distributions over damage or recovery thresholds and later test calibration, rather than asking only for confidence labels or point forecasts.

### 20. Bakshy, Eckles, and Bernstein (2014), *Designing and Deploying Online Field Experiments*
**Scores:** Similarity 2/5 · Contribution 3/5  
**File:** `Agent and Experiments/Designing and Deploying Online Field Experiments.pdf`  
**Alignment:** Objective 4; Objective 5 / Module 3.

The paper introduces PlanOut infrastructure for deterministic randomization, namespaces, factorial/multi-unit designs, exposure logging, and iterative online experiments. These practices could support reproducible trials of REAP alerts or dashboard designs, but they do not solve identification, matching, or interference in the observational Helene study.

### 21. Scott (2017), *Comparing Consensus Monte Carlo Strategies for Distributed Bayesian Computation*
**Scores:** Similarity 2/5 · Contribution 2/5  
**File:** `ML/Comparing consensus Monte Carlo strategies for distributed Bayesian computation Steven L. Scott.pdf`  
**Alignment:** Indirectly Objective 4.

The paper compares averaging, resampling, kernel-density, and Gaussian-mixture methods for combining shard-level Bayesian posteriors, using simulations with known truth and increasing dimensions. Its failure tests offer loose inspiration for high-dimensional mixture diagnostics, but subposterior combination is mathematically and inferentially different from synthetic-control donor mixing.

## Tier C — Supporting communication, belief, and LLM literature

### 22. Jakobsen (2026), *Coarse Bayesian Updating*
**Scores:** Similarity 2/5 · Contribution 2/5  
**File:** `Information and Beliefs/🏷️Coarse Bayesian Updating.pdf`  
**Alignment:** Objective 5 / Module 3.

The paper axiomatizes agents who replace a Bayesian posterior with a representative point from a convex partition of the belief simplex, producing path dependence and sometimes false convergence. It can model stakeholders who convert continuous effect estimates into coarse warning categories, but its convex geometry is unrelated to donor-weight identification.

### 23. Bénabou and Tirole (2016), *Mindful Economics: The Production, Consumption, and Value of Beliefs*
**Scores:** Similarity 1/5 · Contribution 2/5  
**File:** `Information and Beliefs/Mindful Economics- The Production, Consumption, and Value of Beliefs.pdf`  
**Alignment:** Objective 5 / Module 3.

The article synthesizes motivated-belief models and evidence on strategic ignorance, selective recall, asymmetric updating, identity, and group reinforcement. It warns that more precise satellite evidence may not automatically change decisions, motivating explicit testing of framing and actionability rather than a change to REAP's estimator.

### 24. Brand, Israeli, and Ngwe (2025), *Using LLMs for Market Research*
**Scores:** Similarity 1/5 · Contribution 2/5  
**File:** `AI in Marketing/Using LLMs for Market Research.pdf`  
**Alignment:** Possible future stakeholder research; Objective 4.

The authors repeatedly sample LLM survey responses, estimate conjoint willingness-to-pay, compare with human surveys, and fine-tune on earlier human choices. The work is useful only as a validation template and warning about weak cross-category and subgroup transfer; it does not test image embeddings or causal counterfactuals.

### 25. Rabin and Schrag (1999), *First Impressions Matter: A Model of Confirmatory Bias*
**Scores:** Similarity 1/5 · Contribution 2/5  
**File:** `Information and Beliefs/First Impressions Matter- A Model of Confirmatory Bias.pdf`  
**Alignment:** Objective 5 / Module 3.

In a repeated-signal model, agents probabilistically misread evidence that conflicts with their current favored hypothesis and then update as if the misperception were correct. It motivates recording priors, blinding initial labels, and varying presentation order when REAP findings are reviewed, but contributes no empirical or causal estimator.

### 26. Bikhchandani, Hirshleifer, Tamuz, and Welch (2021/2024), *Information Cascades and Social Learning*
**Scores:** Similarity 1/5 · Contribution 2/5  
**File:** `Information and Beliefs/Information Cascades and Social Learning.pdf`  
**Alignment:** Objective 5 / Module 3.

This review studies sequential learning when agents observe others' actions rather than private signals, showing how fragile incorrect cascades and early-influencer effects arise. It suggests collecting independent judgments before displaying peer choices in a stakeholder interface, but is unrelated to the satellite estimator.

### 27. Sims (2003), *Implications of Rational Inattention*
**Scores:** Similarity 1/5 · Contribution 2/5  
**File:** `Information and Beliefs/🏷️Implications of rational inattention.pdf`  
**Alignment:** Objective 5 / Module 3.

Sims adds a finite Shannon-information-capacity constraint to dynamic decision problems, yielding delayed and smoothed responses to signals. The implication for REAP is interface prioritization under limited attention, not a method for estimating disaster effects.

### 28. Farmer, Kochar, and Lee (2026), *The Alpha-Law of Observable Belief Revision in Large Language Model Inference*
**Scores:** Similarity 1/5 · Contribution 2/5  
**File:** `Information and Beliefs/🏷️The Alpha-Law of Observable Belief Revision in Large Language Model Inference.pdf`  
**Alignment:** Optional LLM auditing in Objective 5.

The paper elicits LLM probabilities before and after evidence and fits a multiplicative update exponent across benchmark tasks, with supplementary multi-step and token-logprob analyses. If an LLM later narrates REAP results, this logging design could test update stability, but it does not bear on latent-space SC and its proposed alpha measure is less established than conventional calibration.

### 29. *Social Tipping Points: Understanding Rapid Transitions in Socioeconomic Systems* (undated slide deck)
**Scores:** Similarity 1/5 · Contribution 1/5  
**File:** `Information and Beliefs/Social_Tipping_Points.pdf`  
**Alignment:** Objective 5 / Module 3.

This derivative presentation summarizes threshold, contagion, committed-minority, network, and targeted-intervention ideas for sustainability transitions. It may supply language for stakeholder adoption, but it contains no original data or reproducible method and its claimed thresholds should not be transferred to REAP.

### 30. *Green Tipping: Behavioural Shifts for a Sustainable Future* (undated slide deck)
**Scores:** Similarity 1/5 · Contribution 1/5  
**File:** `Information and Beliefs/Tipping_Social_Transformation.pdf`  
**Alignment:** Objective 5 / Module 3.

The deck repackages committed minorities, observability, infrastructure, information, and incentives as mechanisms for social transformation. It is redundant with the preceding presentation and contributes communication vocabulary rather than evidence or an estimation method.

### 31. Chen, Lu, and Hansen, *Large Language Model Agents as Machini Moralis* (undated slides; workbook lists 2025)
**Scores:** Similarity 1/5 · Contribution 1/5  
**File:** `Agent and Experiments/LLMs_as_Machini_Moralis_slides.pdf`  
**Alignment:** Peripheral to the clinical preliminary.

The slides fit preference parameters to LLM choices in economic games, fine-tune agents on small synthetic datasets, and test transfer to moral dilemmas and pricing games. This concerns preference alignment and behavioral transfer, not weight recovery, satellite representations, or causal counterfactuals.

### 32. Zhang and Zhang (2025), *Generative AI and Information Asymmetry: Impacts on Adverse Selection and Moral Hazard*
**Scores:** Similarity 1/5 · Contribution 1/5  
**File:** `AI in Marketing/Generative AI and Information Asymmetry_ Impacts on Adverse Selection and Moral Hazard.pdf`  
**Alignment:** Peripheral measurement caution.

The paper inserts noisy AI signals of type and effort into principal-agent models and studies outcomes with theoretical analysis and synthetic simulations. Its only transferable lesson is to audit learned signals for error and manipulation; it provides no externally validated AI model or usable causal method for REAP.

### 33. Morris and Shin (2001/2003), *Global Games: Theory and Applications*
**Scores:** Similarity 1/5 · Contribution 1/5  
**File:** `Information and Beliefs/Global games- Theory and applications.pdf`  
**Alignment:** Optional stakeholder-coordination theory.

Global-games models add noisy private signals to coordination games and use cutoff equilibria or iterated deletion to select outcomes. This matters only if REAP studies coordinated stakeholder adoption or public higher-order beliefs, not for the current matched DiD/SC design.

### 34. Liu et al. (2026), *WHO&WHEN PRO: Can LLMs Really Attribute Failures in AI Agents?*
**Scores:** Similarity 1/5 · Contribution 1/5  
**File:** `Agent and Experiments/✅WHO&WHEN PRO- Can LLMs Really Attribute Failures in AI Agents?.pdf`  
**Alignment:** Optional engineering diagnostics.

The benchmark injects controlled errors into replayed multi-agent trajectories and tests models on responsible-agent, decisive-step, and failure-mode attribution. Its replay discipline could help debug a future multimodal pipeline, but “failure attribution” here is not causal attribution in the potential-outcomes sense.

## Recommended reading order by project task

- **Build the causal estimator:** 1 → 2 → 3 → 5 → 8.
- **Position the satellite/disaster application:** 4 → 6.
- **Design validation and guard against overclaiming:** 1 → 2 → 6 → 12 → 13 → 15.
- **Build the stakeholder/Extension layer:** 7 → 9 → 10 → 11 → 19.
- **Understand why this is not primarily an LLM project:** 6 → 12 → 15 → 24 → 34.

## Coverage and workbook note

`CALS Lit Review .xlsx` is a metadata workbook, not a literature item, so it is not included in the 34-paper ranking. It contains one additional citation without a local PDF—Gavrilova, Langørgen, and Zoutman, *Difference-in-Difference Causal Forests With an Application to Payroll Tax Incidence in Norway*—which is worth obtaining as a heterogeneous-effect benchmark if REAP expands beyond 26 treated sites.
