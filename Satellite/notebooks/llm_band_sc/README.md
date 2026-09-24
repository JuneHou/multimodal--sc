# llm_band_sc — LLMs as the weight solver on the collaborator's band-feature designs

Opened 2026-09-09 (plan `quirky-growing-wombat`, third version). Jun: with the band table in a prompt the
picture is redundant, so test the idea on its own — the four ARC-hosted models on the collaborator's
designs, her band features, her metrics. **Monthly data for both designs; SHC first.** Evaluation = her
formulas exactly; results go into `Docs/` only on Jun's call.

## The two designs (validation only, P10 predicted from inputs up to P09)
| | SHC (her notebook 10) | SCM (her notebook 14 method on the monthly data) |
|---|---|---|
| cells | 27 target sites × sensor (S2 1–15, S1 15–27) | the same 27 |
| prompt input | the target site's own P01–P09 band means | target P01–P09 + its 5 matched donor sites (control rank 1–5) P01–P10 |
| asked | the target's band means at P10, one line | the same |
| metric | her outcome metric (slide 16 step 10, `data/shc/10`): per band squared error ÷ the target's outcome variance over P01–P10 (ddof 1); joint = mean over bands. Her held-out variant (variance over P01–P09) is kept alongside | her pooled scaler (mean/SD ddof 1 over targets + donors at P01–P09, per sensor × feature); per-cell standardized RMSE over features at P10; per-feature |error|; cross-site mean/median/min/max/SD |
| references | `shc` = OUR run of her held-out SHC (`lb_shc.py`, her notebook-10 pipeline step by step: seasonal sin/cos + trend OLS, kernel bandwidth 2, per-band standardization, m = 4 / n = 1 blocks on P01–P09, stepwise donors, SLSQP weights, P10 = weighted forward month) — gate: reproduces her 27 stored per-site values to 4e-7; `own_mean` = P01–P09 mean | `scm` = her SLSQP simplex solver on the same standardized matrix (training RMSE, ratio, flag, largest weight, effective donors) |

Wording: all of P01–P10 are pre-hurricane for every site; nothing is treated in validation; prompts say
TARGET parcel and DONOR parcels and never mention the hurricane. Every prompt opens with a short band guide.

## Pipeline
| step | file | writes |
|---|---|---|
| 1 | `lb_features.py` | `lb_features_monthly.csv` (1,620 rows: 27 targets + 5 donors each × P01–P10, her float64 nanmean extraction; 80 donor rows without a usable composite are NaN → "missing"); `lb_her_nb10_results.csv`, `lb_her_nb10_tables.csv`, `lb_her_nb10_feature_rows.csv` (her notebook 10 output parsed) |
| 1b | `lb_shc.py` | our run of her held-out SHC (gate against her stored values) |
| 2 | `lb_prompts.py` | `lb_prompts_shc.jsonl` (27), `lb_prompts_scm.jsonl` (27): full text, withheld truth, P01–P09 values |
| 3 | `lb_tests.py` | gates (all passed 2026-09-09): her notebook-14 weights and validation RMSE reproduced on her shipped biweekly table (max Δw 5.6e-8, ΔRMSE 7.6e-9); monthly extraction = her printed values (4.6e-7); her 27 per-site values parsed, sensor means = her table (2e-7); parser; no forbidden word |
| 4 | `lb_run.py --experiment shc` then `--experiment scm` | `lb_replies_<exp>_<model>_<effort>.jsonl` (raw reply, parsed values, usage, finish reason, seconds) |
| 5 | `lb_score.py` | `lb_scores_shc.csv`, `lb_summary_shc.csv`, `lb_summary_shc_band.csv`, `lb_scores_scm.csv`, `lb_summary_scm.csv`, `lb_summary_scm_feature.csv` |
| 6 | `build_nb41.py` → `41_llm_band_sc.ipynb` | the comparison tables (S1 before S2), per-site tables, failure log, one prompt per design and sensor |

## What do the models actually do? (plan 2026-09-10: ladder, diagnosis, ablations)
| step | file | writes |
|---|---|---|
| A | `lb_baselines.py` (rules, used by `lb_score.py`) | extra method rows in `lb_scores_shc.csv` / `lb_summary_shc.csv`: `persistence` (P09; = `shc` at the 26 single-block cells), `last3_mean` (P07–P09), `linear9` / `linear4` (OLS line on P01–P09 / P06–P09, one step ahead), `holt_damped` (additive damped Holt, α/β/φ by in-sample one-step SSE on P01–P09, grid), `ensemble` (mean of Kimi-low and gpt-oss-low) |
| B | `python lb_baselines.py` | `lb_diag_models.csv` (per model × cell × band: moves from P09 in P01–P09-SD units for the model, the truth and every rule; nearest rule; direction agreement), `lb_diag_summary.csv` (slope/R² of the model move on the linear-trend move, direction shares, nearest-rule counts), `lb_diag_s2_sites.csv` (series and predictions at S2 sites 0004/0005/0010) |
| C | `lb_prompts_ablation.py` → `lb_run.py --experiment abl_noguide\|abl_anon\|abl_nocal\|abl_rule` | `lb_prompts_abl_*.jsonl` (27 each; `labels` = x1..xk for anon), `lb_replies_abl_*_<model>_<effort>.jsonl`; scored by `lb_score.py` as `<model> [noguide\|anon\|nocal\|rule]`; `lb_lib.rule_text` extracts the stated rule |
Gates 6–9 in `lb_tests.py`: the default prompt builder reproduces the 27 stored SHC prompts byte for byte; rules exact on a planted line and Holt reductions; persistence = our run of her SHC at the single-block cells (diff 0); anon labels round-trip; ablation files carry the intended edits and no forbidden word. Results in `41_llm_band_sc.ipynb` §4.

## Fixed knobs
Models `gpt-oss-120b`, `GLM-5.3`, `Kimi-K3-thinking-low`, `DeepSeek-V4-Flash-thinking-low`, every request with `reasoning_effort: low` (Jun, 2026-09-09: Kimi-K3 and GLM-5.3 at default effort spent the whole 8,000-token ceiling on reasoning and returned empty content; at default effort GLM-5.3 answered 2 of 27 and Kimi-K3 0 of 20 — those files were removed; gpt-oss-120b's complete default-effort run is kept as `lb_replies_shc_gpt-oss-120b_default.jsonl` and labelled "(default effort)") (ARC `llm-api.arc.vt.edu/api/v1`, bearer key from
`$ARC_LLM_API_KEY` or `~/.config/arc_llm_key`, never in the repo); temperature 0, seed 0, max_tokens 8000 (the ARC ceiling per request), one attempt per prompt (temperature 0); the call is repeated only on an ARC limit reply (HTTP 429/503), up to 5 times, 60 s apart. Decimals in prompts: S2 3, S1 2. Calendar from `nb25_monthly_long_period_definitions.csv`.

## Provenance / gates
Data read-only (`data/monthly_long_datasets`, index `panel_monthly_index.csv`). Her notebook-14 shipped outputs
(`test/scm_validation/`) are used only as the solver gate — her biweekly numbers are not comparable with the
monthly runs. Nothing in `ts_SCM_ASCM/`, `vlm_generation/`, `test/` or `Docs/` is modified.

## Spec arms — can a model specify and run SHC? (plan 2026-09-23)

**Her code, not a re-implementation.** `lb_extract_her.py` copies the SHC functions out of
`data/shc/10_shc_monthly_multiple_outcomes.ipynb` by `ast` line span into `lb_her_nb10.py` (generated, never edited;
`--check` is gate G1, with the notebook's sha256 and a per-block manifest). `lb_her_panel.py` melts our wide CSV into her
long `feature_panel` and applies her cell-7 statements verbatim. `lb_her_run.py` drives her
`heldout_validation_for_sensor`; `--drift` compares with the frozen condensed copy `lb_shc_condensed.py` (all 27 cells:
joint error and every prediction differ by 0, `latent_mse` by at most 7.6e-15 from pandas vs numpy `std`);
`--freeze` writes `lb_shc_parity.csv` (full precision; read it with `lb_her_run.read_parity`, pandas' default float
parser is not round-trip exact) and `lb_shc_candidates.csv` (her per-candidate diagnostics, 246 rows). `lb_shc.py` is
now a thin adapter with its previous surface; the only change in `lb_scores_shc.csv` is one ulp on the two-block cell.

**The option layer.** `lb_spec.py`: Class 1 = `m, n, kernel_bandwidth, zeta, beta_trend_degree` (her code's own
arguments, applied by rebinding her module names inside `lb_her_run.her_globals`; the matching-window length is `m`
by construction, not a field) and Class 2 = `weight_constraint, intercept, band_weighting, fit_space, readd_seasonal,
selection` (branches her code does not have, in `lb_fit.py`). No defaults: a missing field is an error naming it;
the cross-field rules in `lb_spec.CROSS_FIELD_RULES` are shown to the model verbatim. `HER_SPEC` reproduces her
numbers bit-for-bit through `lb_fit` (G4) and through the forced general path (G5, difference 0.0).

**Withdrawn 2026-09-24.** The three prompt-based arms built on 2026-09-23 (shc_tool1, shc_tool2, shc_code: instruction
text in the request, a JSON spec filled by the model, the fitter run by the host), their prompt files, runner, scorer and
all their replies were deleted at Jun's instruction: Jun asked for skills (SKILL.md plus bundled scripts that the model
itself reads and runs) and the build was not that, and the change was never declared. What stands and will serve the
skill build: her extracted code (lb_her_nb10.py), the parity reference, the option layer (lb_spec.py, lb_fit.py), the
guard (lb_guard.py), the sandbox (lb_sandbox.py) and the gates (lb_spec_tests.py, without the prompt gate).
DeepSeek-V4.1-Flash-thinking-low's direct and step-by-step replies (lb_run.py, 2026-09-23) are kept: they are baselines.

## Skill arms (approved 2026-09-24; replaces the withdrawn prompt arms)

Three skills, generated from the approved texts in `lb_skills_build.py` into `skills/` (hashes in `skills/MANIFEST.json`,
gate SK1): `shc-configure` (A: the model decides the five settings, the six choices are fixed inside `scripts/fit.py`),
`shc-design` (B: settings and choices), `shc-implement` (C: no fitter; the model writes `predict.py`). Each holds
`SKILL.md`, `references/method.md` (identical across the three), `references/settings.md` (A, B), `scripts/show_data.py`,
and for A and B `scripts/fit.py` with verbatim copies of the package modules it needs plus an `lb_lib` shim.
A model operates one skill per cell through `lb_skill_loop.py`: the working directory holds `data.csv` (nine periods)
with the skill mounted read-only over it inside bubblewrap; one command per turn (`cat`, `ls`, `python <script>`,
heredoc write); the data enters only when the model runs `scripts/show_data.py` or reads `data.csv`; the cell ends at a
valid `answer.json` or the step cap (12). The submitted answer is scored (`lb_skill_score.py`, `lb.her_nmse`); every
fitter run, every turn's raw response, and the program are stored. Runner: `lb_skill_run.py --skill <name>`;
`--show <reply file>` prints a transcript. Gates SK1-SK6 in `lb_skill_tests.py` (parity through the sandbox, refusals,
nine periods + canary, dry runs, step cap), run by `lb_tests.py`.
