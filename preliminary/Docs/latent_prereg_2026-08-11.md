# Pre-registration — latent-recovery preliminary on NCT00174655

2026-08-11, written **before any L1–L4 experiment has run**. Companion docs:
[trial_dataset_decision_2026-08-10.md](trial_dataset_decision_2026-08-10.md)
(dataset, observable-run results), [causal_direction_notes_2026-08-10.md](causal_direction_notes_2026-08-10.md).
Convention follows the repo's E1 pre-registration: hypotheses, metrics, and
success criteria are fixed here first; deviations must be recorded, not
silently made.

## Research question (Jun's, verbatim intent)

Do clinical-visit artifacts (rendered medication/adverse-event text; the
code bag is the ground-truth observable) admit a recoverable latent factor
structure that synthetic control's convex geometry survives? Four claims:
(1) latents recoverable; (2) decoded/generated artifacts are *sensible*
(perplexity/likelihood); (3) convex mixtures are semantically meaningful
(near-death + similar survivor → intermediate severity); (4) **weights are
identifiable** — if the true mixing weight is 0.9, the recovered weight is
≈ 0.9.

## Setup (fixed before running)

- Data: NCT00174655 demo release, 977 patients / 9,092 visits. State text:
  `"Medications: …. Adverse events: …."` — concepts alphabetized within
  section (order-sensitivity control), treatment codes excluded
  (intervention, not state), empty section → "none.".
- Encoders (two arms): `Qwen/Qwen3-Embedding-8B` MRL-truncated to 768-d
  (primary; 4096-d also stored), `FremyCompany/BioLORD-2023` 768-d
  (mean-pooled arm; `max_seq_length` set to 512). Unit-norm outputs.
- Solver: the existing, tested `solve_simplex_qp` (SLSQP, Gram form,
  analytic gradients). Seeds fixed at 0; bootstrap B=2000, seed 0, via
  `boot_continuous`.
- Generation/scoring LLM (L3–L4): Qwen2.5-7B-Instruct, temperature 0;
  `score()` = mean NLL in nats, perplexity = exp(NLL).

## H1 — Semi-synthetic weight recovery (L2, design 1)

Pseudo-target z* = Σ_j w*_j z_j (+ noise), donors sampled from real visit
latents; fit ŵ by simplex QP.

Cells: K ∈ {2, 5, 10} donors × w* ∈ {Dirichlet(1), spiky (0.9, rest
uniform)} × σ ∈ {0, residual-matched} × pool ∈ {oracle-K, m=50, m=all} ×
2 encoders. N = 500 pseudo-targets per cell, seeded.

Metrics: L1(ŵ, w*) (primary); top-1 donor recovery rate; |ŵ_max − 0.9| in
the spiky cells.

**Success criteria (set now):**
- Oracle sanity bound: pool = the true K donors, σ=0 → mean L1 ≤ 0.01
  (else the solver or the design is broken — hard gate, not a finding).
- **H1 supported** in an encoder arm if, at m=50 with σ=0: mean L1(ŵ, w*) ≤
  0.30 at K=5 AND spiky-cell |ŵ_max − 0.9| ≤ 0.10 AND top-1 recovery ≥ 90%.
- **H1 refuted** in that arm if spiky-cell |ŵ_max − 0.9| > 0.25 or top-1
  recovery < 60% at m=50, σ=0.
- Prediction on record: BioLORD (mean-pooled) outperforms Qwen3
  (last-token-pooled) on L1 recovery; Qwen3 outperforms on the neighbor
  smell test. If one arm wins both, it carries L3/L4.

## H2 — Cross-space weight consistency (L2, design 2)

Real patients, T0=1 (961 eligible): fit SC in bag space (observable
machinery, cosine donor schema, m=50) and in each latent space, same donor
pool. Compare per-patient weight vectors.

Metrics: mean cosine similarity of weight vectors (primary); top-1 donor
agreement rate; L1 distance; each vs a permuted-donor null (weights of a
random other target, same pool size).

**Success criteria:** H2 supported if mean weight-cosine exceeds the
permutation null's 95th percentile AND top-1 donor agreement ≥ 3× the null
rate. Refuted if weight-cosine is within the null's IQR. (No absolute
threshold — the null calibrates it.)

## H3 — Generative sensibility (L3; gated on score() validation)

Gate first: `score()` validated on ordered-likelihood strings (a real
sentence < shuffled words < random tokens, strictly increasing NLL) before
any perplexity number is quoted.

Decode: prompt-conditioned generation of the mixed patient's visit text from
K donor texts + weights (temp 0). Baseline: nearest-real-visit text.

Metrics: (a) perplexity distribution of generated vs real visit texts;
(b) cycle-consistency — re-embed generated text, cosine to z*, and rank of
z* among all 9,092 cohort latents; (c) Spearman corr(donor-latent distance,
generated-text embedding distance) over sampled pairs; (d) code recovery F1
of parsed generated text vs the mixture's expected code set.

**Success criteria:** H3 supported if median generated perplexity ≤ 1.5× the
median real perplexity AND median cycle-consistency rank of z* is within the
top 5% of the cohort AND (c) is positive with p < 0.05. Refuted if generated
perplexity exceeds 3× real median (not sensible) or the cycle rank is no
better than chance.

## H4 — Severity interpolation (L4; same gate)

~50 pairs: deceased (feature.csv death=1, or top-quartile serious-AE burden)
× nearest latent neighbor with death=0 / bottom-half burden. α ∈ {0, .25,
.5, .75, 1}; decode each mixture; severity by (i) out-of-fold logistic
severity model on latents, (ii) LLM-judge (fixed rubric, temp 0, 1–10).

**Success criteria:** H4 supported if mean per-pair Spearman(α, severity) ≥
0.7 for at least one severity scorer with the two scorers agreeing in sign,
AND perplexity across α stays within the H3 sensibility band (mixtures
remain realistic). Refuted if mean Spearman < 0.3 on both scorers.

## Interpretation rules (fixed now)

- H1 is the load-bearing claim; if H1 fails in both encoder arms, H2–H4
  results are exploratory only and the memo must say the latent space does
  not support convex weight identification as constructed.
- If H1 holds but H2 fails: the latent supports mixing, but the latent and
  observable records tell different mixing stories — report as a finding
  about representation choice, not as method failure.
- Encoder selection for L3/L4 = the arm winning H1 primary metric at K=5,
  m=50, σ=0; ties broken by H2 weight-cosine.
- Any deviation from the cells, thresholds, or seeds above is recorded in
  the results doc with a reason.

## Costs and caps (so scope creep is visible)

L0–L2: CPU + one A40 for embedding; no LLM. L3–L4: one A40 for vLLM;
generation capped at ~1,500 completions total (500 mixtures + 250 pairs ×
5 α, minus reuse); judge calls same order. If generation exceeds 2× the cap,
stop and revisit.
