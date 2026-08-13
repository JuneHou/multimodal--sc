# Decode from estimated weights — the DiD-analog experiment (post-hoc)

2026-08-12. **POST-HOC, NOT pre-registered** — same status as
`run_copy_diagnostic.py` and `run_uniqueness_check.py`; recorded here per the
prereg's own deviation rule. Motivated by Jun's question during the REAP
satellite DiD-vs-SC discussion: H3 decoded notes only from the **planted**
ground-truth weights w\* (and only at K=5) — what does generation look like
when the decoder is fed what the pipeline would actually have at inference
time, the **solver-estimated** weights ŵ from a realistic retrieval pool?

## Design

For each planted mixture (the exact H1 stream: `draw_replicates`, seed 0, so
every replicate is paired with its H1 parquet row), three decode arms:

| arm | donors in the prompt | weights in the prompt | isolates |
|---|---|---|---|
| `wstar` | the true planted donors | the true planted w\* | upper anchor (H3's setting, extended to K=2,10) |
| `what` | support of ŵ from the m=50 retrieved pool (>0.01, top-10, renormalized) | ŵ | the full estimation pipeline |
| `equalK` | top-K retrieved donors by cosine | 1/K each | the DiD analog: same retrieval, no weight estimation |

(`wstar` vs `what`) prices estimation error; (`what` vs `equalK`) prices
weight estimation given retrieval. Donor order inside every prompt is
randomized per replicate (position independent of weight — the L3 convention,
kept because H4 found a first-listed position bias).

Grid: K ∈ {2, 5, 10} × {dirichlet, spiky} × N=25 per cell × 3 arms = **450
completions** (running generation total 1,450 of the prereg's ~1,500 cap).
σ = 0, qwen3-768 encoder (the selected arm). Decoder and scorer identical to
H3: vLLM `Qwen/Qwen2.5-7B-Instruct`, temperature 0, `GEN_MAX_TOKENS=800`,
gated on `score_gate_report.json`; real-corpus perplexity references reused
from `latent/results/_real_ppl.json`.

Metrics per row: `gen_ppl` (mean-NLL exp), copy diagnostic (8-gram shingle
overlap, pooled over prompt donors and vs the top-weight donor), and the
paired solver quantities (`l1_what`, `n_true_in_pool`, `w_hat_max`).

## Files

- `run_estimated_weight_generation.py` — the experiment (two-stage
  generate/score, checkpointed; `--n-per-cell 25` default).
- `results/estimated_weight_generation.parquet` — 450 rows, one per decode.
- `results/estimated_weight_generation_summary.json` — cell medians, arm
  medians, Spearman(L1, ppl), and the two anchors below.
- `results/` is gitignored repo-wide; generated text derives from n2c2 (DUA)
  and must never be committed.

## Verification anchors

1. `wstar` K=5 cells vs H3's `l3_generation.parquet` on the shared first-25
   replicates (same stream, same prompts, temp 0).
2. Solver refits vs the H1 `n2c2_weight_recovery.parquet` (qwen3, m50, σ=0)
   cell means on matched replicates.

## Results (run of 2026-08-12, 450 completions, one A40, ~21 min)

**Anchors passed.**
- Solver: mean L1 of our refits equals the H1 parquet cell means on matched
  replicates to machine precision (e.g. K5-spiky 0.230279… identical) — the
  mixture stream is exactly the H1 stream.
- `wstar` K=5 medians vs H3's `l3_generation.parquet` (first 25 reps):
  dirichlet 9.46 vs 9.78, spiky 8.74 vs 9.26 — close but not identical
  because this run randomizes donor *order* in the prompt (L3 used draw
  order) and H4 showed order matters; treated as consistent.

**Median perplexity ratio vs the 1,264 real records (support bar 1.5).**
Each arm cell = median gen-ppl of that arm's notes ÷ median real ppl — a
fluency ratio, NOT a weight (weights sum to 1 in every arm; ratios may
exceed 1). L1 is the distance Σ|ŵ−w\*| ∈ [0, 2], also not a weight:

| K | family | ppl ratio, `wstar` arm | ppl ratio, `what` arm | ppl ratio, `equalK` arm | mean L1(ŵ,w\*) |
|---|---|---|---|---|---|
| 2 | dirichlet | 0.94 | 0.95 | 0.93 | 0.12 |
| 2 | spiky | 1.06 | 0.96 | 0.95 | 0.22 |
| 5 | dirichlet | 1.04 | 0.89 | 0.91 | 0.52 |
| 5 | spiky | 0.96 | 0.98 | 0.90 | 0.23 |
| 10 | dirichlet | 1.18 | **0.82** | 1.06 | **0.88** |
| 10 | spiky | 1.05 | 0.91 | 0.94 | 0.22 |

Arm medians overall: `wstar` 9.39, `what` 8.44, `equalK` 8.57 (real 9.12).

**Copy diagnostic stays high everywhere** (pooled median 0.63–0.97 across
cells): the stand-in decoder still copies rather than blends, in every arm.
Spiky cells copy the dominant donor (`copy_top` 0.58–0.80); dirichlet cells
copy *somebody else* (`copy_top` ≈ 0 at K=5,10) — the H3/H4 pattern intact.

**The headline: generation fluency is insensitive to weight quality.**
Spearman(L1(ŵ,w\*), gen_ppl) = −0.13 (p = 0.11); Spearman(L1, copy) = 0.05
(p = 0.52), within the `what` arm. The worst-estimated cell (K=10 dirichlet,
L1 0.88 — the weights are mostly wrong) produced the *most* fluent text in
the table (ratio 0.82). Estimated-ŵ decodes even beat true-w\* decodes on
median perplexity, most likely because ŵ's donors are retrieval-selected and
mutually similar (easy to blend/copy) while the planted donors are random
records (a heterogeneous set is harder to write through).

## Reading

1. **Perplexity cannot price weight fidelity with a prompted decoder.** The
   decoder performs selection-then-copy (H3/H4), so its output is fluent no
   matter how wrong the weights are — `equalK` ≈ `what` ≈ `wstar` on ppl.
   A fluency metric passes everything; it is a floor check, not a validity
   instrument. Weight validity continues to live in the geometric battery
   (L1 recovery, LP uniqueness), not in generated text.
2. **This strengthens, not weakens, the paper-2 thesis**: the three learned
   components (retrieval, latent, decoder) are needed precisely because the
   prompted stand-in cannot convert weight information into text — a decoder
   *trained* for cycle-consistency and attribute monotonicity is the
   prerequisite for any generation-based evaluation of ŵ.
3. **Satellite translation** (what prompted this experiment): the collaborator's
   equal-weight pixel mean vs our estimated-weight SC will likewise NOT be
   separable by "does the counterfactual image look plausible" — plausibility
   is the wrong instrument there too. The comparison must be made on
   pre-period fit, placebo distributions, and planted-mixture recovery, which
   is exactly what `REAP/notebooks/01_dataset_analysis.ipynb` §8–9 measures.

## Deviation record

Post-hoc extension, not in `latent_prereg_2026-08-11.md`. New generation
budget: 450 completions; running project total 1,450 of the prereg's ~1,500
cap. Design choices made before seeing any output: N=25/cell, W_MIN=0.01,
top-10 donor cap, randomized donor order, σ=0, qwen3 arm only.
