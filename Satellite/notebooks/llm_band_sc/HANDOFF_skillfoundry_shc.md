# Handoff for the SHC skills in the SkillFoundry profile

Branch `skillfoundry-shc`. Written 2026-10-13 for a Claude session that did not see the planning conversation.
Read the whole file before any action. Paths are relative to the repository root.

## Where the work stops

The work stops when the two skill drafts and the review page are committed and pushed to this branch. Nothing
past that point is built until Jun approves the texts in writing. No model is called by this session. No API key
file is read, sourced, grepped or printed.

## The approved plan

Jun approved this plan on 2026-10-12 and chose SHC only as the scope.

**Context.** The existing skill experiment ran her seasonal monthly synthetic historical control (SHC, her notebook
10) in three arms, `shc-configure`, `shc-design` and `shc-implement`. It used four models (gpt-oss-120b, GLM-5.3,
DeepSeek-V4.1-Flash-thinking-low, GPT-5.5) on 27 target cells. The model sees the target's P01 to P09 and predicts
P10. The question for this round is whether the SkillFoundry skill profile changes how the models operate the
same method on the same task. Their synthetic-data tests check that a script consumes controlled inputs and
cannot validate a causal estimator on real data. The validation stays her held-out P10 and the parity gates.

**Departures already approved.**

1. Two formats, code provided and code written. `shc-provided` opens the same settings and choices as
   `shc-design`, over the same bundled code. `shc-written` matches `shc-implement`. The settings-only arm has no
   profile counterpart and its rows stay as they are.
2. SkillFoundry is used as format, not pipeline. Their compilation of skills by a model is not used. The texts
   are written from her notebook and the SHC paper.
3. The synthetic example in `examples/` is visible to the model. `tests/` is not mounted in the sandbox, because
   its parity fixtures hold her predictions for three real cells.

**The one table.** Existing and new rows sit in one table, written by `lb_skill_score.py` to
`lb_summary_skill.csv`. Every row is scored from its reply file with the same truth and the same metric. The
reference rows come from `lb_summary_shc.csv`.

| format | skill style | skill | models |
|---|---|---|---|
| settings only | simple | `shc-configure` | four, existing |
| code provided | simple | `shc-design` | four, existing |
| code provided | profile | `shc-provided` | four, new run |
| code written | simple | `shc-implement` | four, existing |
| code written | profile | `shc-written` | four, new run |
| reference | | her SHC, last month, three-month mean | existing |

Fixed across every row: the 27 cells and truth from `lb_prompts_shc.jsonl`, the metric `lb.METRIC`, effort low,
step cap 12, token cap 16,000 (32,000 for DeepSeek), the same system message, task text and loop.

**Which paper and script for which skill.**

| skill | paper | her script | bundled module |
|---|---|---|---|
| `shc-provided` | Chen, Yang and Yang, Synthetic Historical Control, SSRN 4995085 (not in the repository). Equations also in `Satellite/Docs/Synthetic Historical Control_Sep 4.pptx` and `Satellite/Docs/9-10-update.md` §2 | her notebook 10, cells 4, 7, 11, 13, 15, 17, 19, 21, 25 (the notebook is not in git; `lb_her_nb10.py` is its verbatim extract) | `lb_her_nb10.py`, `lb_her_panel.py`, `lb_her_run.py`, `lb_spec.py`, `lb_fit.py`, copied verbatim as in `skills/shc-design/scripts/` |
| `shc-written` | same | same, cited in `refs.md` only | none, the model writes `predict.py` |

## What to borrow from SkillFoundry

Repository `https://github.com/ma-compbio-lab/SkillFoundry`, commit `4a8cc0cd9f923cde47a7f542d4a84b92bbabc59f`,
Apache License 2.0. Any file copied or ported from it carries the license notice and the commit hash in its header.

- **Their generator is a model prompt.** `skillfoundry/prompts.py` (`build_design_skill_prompt`) is sent to Codex
  through `skillfoundry/codex_exec.py`. Departure 2 says this is not run. Record in the review page that the
  generator was read and not run, and why.
- **The mapping is borrowed verbatim.** `SKILL_QUALITY_STANDARDS` in `skillfoundry/prompts.py` (from line 59) is the
  checklist both drafts must satisfy, items 1 to 6, including the nine `SKILL.md` sections.
- **The structure check is ported.** `validate_skills` in `scripts/validate_repository.py` requires `SKILL.md`,
  `metadata.yaml`, `refs.md`, `scripts`, `examples`, `tests` and `assets` in every skill. The plan omitted
  `assets/`, so it is added. `metadata.yaml` is JSON content in their repository (`load_yaml_compatible_json`), and
  ours follows that. Its `status` must be one of their `SKILL_STATUSES`.
- **The model to imitate** is
  `skills/statistical-and-machine-learning-foundations-for-science/dowhy-average-treatment-effect-starter/`. Read its
  `SKILL.md`, `metadata.yaml`, `refs.md`, `assets/`, `examples/` and `tests/` in full before drafting.
- **Conflicts with our harness are departures for Jun, not silent fixes.** Known so far: their items 4 and 5 want
  every path as a command-line argument with `--help` and a JSON summary, while our scripts read `data.csv` in the
  working directory. Their test commands assume the repository root, while our model runs inside the skill folder.

## What to produce

1. `Satellite/notebooks/llm_band_sc/skill_drafts/shc-provided/` and `.../skill_drafts/shc-written/`, each with
   `SKILL.md`, `metadata.yaml`, `refs.md`, `references/method.md`, `assets/`, `examples/` (one synthetic cell with
   known block weights and its expected output), `tests/` (execution, planted weights recovered to 1e-6, parity of
   `fit.py` at her settings on three cells against `lb_shc_parity.csv`, guard), and for `shc-provided` the new
   `scripts/fit.py` and `scripts/show_data.py`. Drafts live outside `skills/` so that no loader or manifest gate sees
   them.
2. A review page `Satellite/Docs/10-13-skillfoundry-shc-review.md` that shows every text a model will see, side by
   side with the existing `shc-design` and `shc-implement` texts, opens with the heading
   **Departures from your instruction** listing every difference from this plan with its reason, and states which
   SkillFoundry checklist item each section answers.
3. A commit on `skillfoundry-shc` with only these new files and this handoff, pushed.

Do not edit `skills/`, `skills/MANIFEST.json`, any `lb_replies_*.jsonl`, any score or summary CSV, or
`lb_her_nb10.py`. Do not change `lb_skill_loop.py`, `lb_skill_run.py` or `lb_skill_score.py` yet. Those changes
come after Jun approves the texts.

## Environment notes for a cloud session

- Install `numpy`, `pandas` and `scipy` if missing. The server's conda environment is not available.
- Her notebook 10 is not in git, so gate G1 in `lb_spec_tests.py` cannot run. Do not change paths to make it run.
- The bubblewrap sandbox used by `lb_skill_loop.py` may be unavailable. Sandbox gates SK2 to SK7 run later on the
  server. Report every gate that was not run, by name.
- The expected output of the synthetic example can be computed by calling `lb_fit.run_cell` directly, since the
  data is synthetic.

## Binding rules from Jun

These override any default behavior.

- **Fidelity.** Build exactly what this file says. Every difference of any kind is listed under **Departures from
  your instruction** with its reason. Nothing is labelled recommended.
- **Skill means skill.** The deliverable is `SKILL.md` plus scripts the model reads and runs in a shell, never a
  prompt with the steps pasted in.
- **Show first.** Every text a model will see is shown to Jun before any model call.
- **Metric.** Only `lb.METRIC`, per band squared error at P10 over that band's P01 to P09 variance, mean over bands.
  Never the P01 to P10 variant.
- **Her code.** `lb_her_nb10.py` is a generated verbatim extract and is never edited.
- **Keys.** Never touch an API key file. Jun runs every model call in his own terminal.
- **Evidence.** A claim not checked in this session is written as unverified. Report what was run and what it
  returned, and name every step that was skipped.
- **Records.** Records invalidated by a bug are deleted, never kept where a loader globs.
- **Writing for Jun.** No colon, semicolon or dash in body prose. Counts up to ten in words, 11 and above in
  numerals. No defensive sentences. Captions describe only what is shown. Headings are complete sentences. Code
  names stay out of prose. Never the word level for a unit or step. Never the words persistence or reversal in
  prose; say the last month or the P09 value. Standard terminology only, no coined names.

## After the cloud session pushes

On the server, run `git pull` on `skillfoundry-shc` before any local work. The server session then runs the
sandbox gates, shows the review page to Jun, and builds the rest of the plan only after his written approval.
