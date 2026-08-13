# Dataset background — n2c2 2018 Track 1 / i2b2 2014 longitudinal corpus

2026-08-11. One-pager for the advisor meeting: what the dataset is, where it
comes from, and why it fits the latent-recovery study. Verification details:
`latent/results/n2c2_verify.json`; selection record:
[trial_dataset_decision_2026-08-10.md](trial_dataset_decision_2026-08-10.md)
(reports 9–10).

## What it is

The **2014 i2b2/UTHealth longitudinal clinical-narrative corpus**, re-released
as **n2c2 2018 Track 1 (cohort selection for clinical trials)**. Harvard DBMI
describes it as *"the only annotated corpus of longitudinal clinical
narratives currently available for research to the general research
community."* One XML file per patient containing their chronological sequence
of de-identified clinical records, plus 13 trial-eligibility criteria labels
(met / not met) per patient.

| fact | value |
|---|---|
| patients | **288** (202 train + 86 gold test) |
| records | **1,264** — mean 4.4 per patient (3–5 for 287/288) |
| ≥3 records | **287/288 (99.7%)** · ≥4: 274/288 (95.1%) |
| record length | mean 618 tokens, median 528, max 2,985 |
| time span per patient | median **3.9 years** (up to 17) |
| record types | 881 clinic/consult notes · 251 discharge/admission · 132 physician letters |
| population | adult diabetic patients, stratified by coronary-artery-disease trajectory |
| labels | 13 eligibility criteria; ADVANCED-CAD balanced (170/118) — usable severity signal |
| integrity (verified locally) | 0 malformed, 0 duplicates, 0 train/test leakage |

Dates are surrogate-shifted (years appear as 2059–2173) per de-identification
convention; within-patient ordering and intervals are preserved.

## Where it comes from — and what it is NOT

- **Source: Partners HealthCare's Research Patient Data Repository (RPDR)** —
  the longitudinal EHR of the Mass General Brigham system (MGH, Brigham, and
  affiliated clinics). Records were drawn from real outpatient and inpatient
  care, de-identified with surrogate names/dates, and released for the i2b2
  2014 shared task (Stubbs & Uzuner; Kumar et al., J Biomed Inform 2015).
- **It is NOT MIMIC-derived.** MIMIC comes from a different hospital (Beth
  Israel Deaconess) and covers ICU/ED encounters. Some *other* n2c2 tracks do
  use MIMIC (e.g., 2018 Track 2 medication extraction), and the community
  annotation set "MIMICause" bundled on the same portal annotates MIMIC
  sentences — neither is this corpus and neither is used in this project.

## Why it has deep per-patient panels where MIMIC did not

1. **Different slice of care.** An ICU database records a patient only when
   critically ill — most patients appear once, which is why the MIMIC
   admission panel collapsed (~350 of 46k patients with ≥4 admissions). An
   integrated health system's repository accumulates *routine longitudinal
   care*: clinic visits, consults, referral letters, over years. Hence 881 of
   1,264 records here are outpatient clinic/consult notes and the median
   patient spans 3.9 years.
2. **Deliberate curation.** The 2014 organizers selected diabetic patients
   specifically having 3–5 substantial records (≥300 words) over time,
   stratified into three equal groups — CAD at baseline, CAD emerging during
   the record, never-CAD — expressly to support studying risk-factor
   *progression*. The longitudinal depth is by design.

Meeting-ready sentence: *the dataset problem was never "clinical data is too
short" — ICU-admission panels and trial cycle-logs are the wrong slice of
care for trajectory questions; longitudinal outpatient narratives are the
right one, and this is the one openly-annotated corpus of them.*

## Access and terms

Obtained under Jun's existing n2c2/DBMI registration (data-use agreement
signed at download; redistribution prohibited — collaborators need their own
registration; the DBMI portal currently lists new registration as
temporarily closed, which makes the in-hand copy the practical path).
Stored gitignored at `preliminary/latent/data/n2c2/`; the official
Track 1 evaluation script (Apache 2.0) is included in the upload.

## Why it was selected over the alternatives

Full comparison in the decision doc (report 9). In short: MedAlign (Stanford)
has deeper text per patient (median 101 notes) plus native OMOP structured
data, but 276 patients, a 7–10-day application, and an evaluation-only
license; CheXpert Plus has 64,725 patients with paired images (kept as the
designated dataset for the later multimodal stage) but averages only 2.9
studies per patient. n2c2 won on the criteria this experiment needs:
verified real prose, near-universal ≥3-timepoint coverage, built-in severity
labels, and zero access friction. Its one gap — no native structured codes —
is covered by UMLS concept extraction (QuickUMLS) for the cross-space test.

## How it maps onto the experiment

Units = patients; time = date-sorted records (t = 0,1,2,…); artifact = the
record text itself; latents = text embeddings (Qwen3-Embedding-8B 768-d
primary, BioLORD-2023 second arm); observable space for the cross-space test
= UMLS concept bags; severity for interpolation = ADVANCED-CAD. Hypotheses,
thresholds, and seeds are pre-registered in
[latent_prereg_2026-08-11.md](latent_prereg_2026-08-11.md) +
[addendum](latent_prereg_addendum_2026-08-11.md).
