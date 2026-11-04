# Statistics readout for the SCM report deck (nature-statistics, bounded audit)

## Statistics review scope

- Input reviewed: `results/table_4_1.csv`, `table_4_2.csv`, `table_4_3.csv` (the README's Section 4.1, 4.2 and 4.3 checks
  for every answer and the three reference rows), `results/luck_replications.csv`, `results/scores.csv`.
- Boundary: descriptive results only. No test, p value or interval is reported anywhere in the deck, and none is computed
  here, because the design below supports none.
- Study design readout: two simulated data-generating processes (DGP 1, Equation 1 of Abadie, Diamond and Hainmueller
  2010; DGP 2, Equation 5), one replication each, as shipped. Each replication has one treated unit, 29 donors, 60
  periods and a true effect of −3 in periods 41 to 60. Four skill formats (scm-configure, scm-design, two profile skills)
  and four solver models are applied to the same two replications; each format, model and replication is run once. The
  endpoints are the README's: counterfactual RMSE and ATT error against the ground truth (4.1), pre-treatment RMSPE and
  covariate balance (4.2), leave-one-out changes (4.3).
- Independent unit and replication readout: the independent unit is the replication, a dataset drawn from the
  generator. n = 1 per DGP. The 20 post-treatment periods of one replication are repeated measures on one treated unit
  (serially correlated by construction in DGP 2), not independent units. The four formats and four models are conditions
  applied to the same replication, so every comparison between them is paired on one dataset. Run-to-run variation of a
  model under one condition is not measured, since each condition was run once.

## Major statistical issues

- [P0] Issue: small-sample overclaim. Evidence: one replication per DGP; the differences between formats in ATT error
  on DGP 1 are tenths of a unit. Why it matters: for an estimator that knows every factor of DGP 1, the ATT error is the
  mean of the treated unit's own noise; over the shipped replication and ten new replications of the same generator it
  has standard deviation 0.252 and ranges from −0.567 to +0.402 (`luck_replications.csv`). A gap between two formats on
  one replication that is smaller than this spread cannot rank them. Fix: state the results as one replication's values;
  rank formats only after repeated replications.
- [P0] Issue: pseudoreplication risk. Evidence: tables list four models and 20 periods per answer. Why it matters:
  counting models, periods or both DGPs as n would overstate precision. Fix: no pooled statistic over models or periods
  is presented as a sample; the profile formats, whose four answers are identical, are shown once.
- [P1] Issue: analysis unit mismatch for future comparisons. Evidence: all conditions share one replication. Fix: with
  repeated replications, compare formats replication by replication (paired), so that each replication's own noise,
  common to every condition, cancels.
- [P1] Issue: one DGP cannot separate estimation quality. Evidence: in DGP 2 every unit is its own autoregression with
  its own shocks; over periods 1 to 40 the treated unit's path correlates 0.08 with the donor average. Fix: report DGP 2
  for answer rate, steps and whether a reported ATT follows from its counterfactual, not for which weights are better.
- [P2] Issue: many cells, no family. Evidence: four formats, four models, two DGPs and three checks. Fix: no comparison
  is tested; each slide reports values and the reference rows they sit beside.
- [P2] Issue: run-to-run variation unmeasured. Evidence: one run per condition. Fix: say so where answer rates are shown.

## Ready-to-paste text for the analysis slide

One replication per process is a single dataset, and the answers of every format and model are measured on that same
dataset. The treated unit's own noise is part of its true untreated path, so even a counterfactual that knows every factor
of the first process misses the true effect by the mean of that noise. Across eleven replications of the same generator
that miss has standard deviation 0.252, from −0.567 to +0.402. The gaps between formats on the shipped replication are
smaller than this, so they describe this dataset and do not rank the formats. In the second process no term is shared
across units, so the donors carry no information about the treated unit after period 40, and every set of weights lands
near the same error.

## AUTHOR_INPUT_NEEDED

- The number of further replications per DGP, and whether DGP 2 receives them, decide whether the formats can be ranked;
  both are open with Jun.

## Reviewer-risk note

- A reviewer may ask for run-to-run variation of each model and for repeated replications before any statement about
  which format is better; the deck makes no such statement.
