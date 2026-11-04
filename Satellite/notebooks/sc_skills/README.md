# Skill-making pipelines for synthetic control methods

This folder turns the way the SHC skills were made into three pipelines that work for any synthetic control method, given its paper, its code and its data. The configure pipeline fixes the reference code and lets the model set its settings. The design pipeline also lets the model pick the method's choices. The profile pipeline has SkillFoundry write the skill from the paper and the code. All three share one run stage and one stage that compares each answer's steps with the reference code's. The plan is `~/.claude/plans/quirky-growing-wombat.md`.

The SHC folders `../llm_band_sc/` and `../skillfoundry_profile/` are imported and never edited. The two SHC packages in `methods/` point at their files, and the tests check that the pipelines give back the existing SHC skills, request and tables.

## Jun's instructions define the pipelines

| date | Jun's words |
|---|---|
| 2026-10-30 | "i said test different method to generate different skill, config, design, skill profile... correct?" |
| 2026-10-30 | "your first step will be generalize our SHC skill development methods into different pipelines, for which we can adopt for what ever the causal synthetic SC method we want to do with given paper, code or other required inputs." |
| 2026-10-30 | formats "configure, design, profile"; scope "Build, run, score, compare steps"; folder "New folder, old unchanged" |
| 2026-10-30 | "for c, how we did before the same now. for f, same as before, for g, same as before, collaborator's code, if not available then a." |
| 2026-10-30 | "run 1 first with few sample to see whether there has error or problem. if not use 1, if cannot use or introduce erro ruse 2"; "what i want to test is in real data points whether R rynth works." |
| 2026-10-30 | "evaluation will be determined by using. not in generalized pipeline."; "both P10 and helene post band are DATA DATA INPUT DATA" |
| 2026-10-15 | "design is model design, config is her code model give config" |

## A method package holds every input of one method

`methods/<method>/method.py` defines `PACKAGE` (`scs_method.MethodPackage`).

| field | what it holds |
|---|---|
| paper | the paper, local PDF |
| reference_code | the collaborator's code, else the paper's package; its language; how it is copied unchanged; what it must reproduce (`parity`) or, for R, the R trial |
| data | the input data as shipped |
| spec | the settings (the reference code's own arguments) and the choices (its hard-coded choices), each with kind, allowed values and meaning; the reference code's values; the rules that tie the fields, as text for the model and as checks |
| engine | how the copied code runs, and the native engine for options the reference code lacks |
| texts | every text the model reads: the SKILL.md sections, `references/method.md`, the scripts, and the profile request's method sentence |
| skill_modules | files copied unchanged into each configure and design skill |

## How each pipeline runs

**Configure** (`scs_copy.py`, `scs_check.py`, `scs_spec.py`, `scs_build.py`)

1. Copy the reference code unchanged, character for character, with a sha256 per file.
2. Check the copy works on the data. The collaborator's code, its output written the way the collaborator's code prints it, must reproduce what the collaborator's notebook printed. A package written in R is run by us inside the box on a few cases of the real data, with no model. If it runs without an error and returns donor weights and a prediction, the skill uses the original R code. Otherwise its steps are rewritten in Python and must give the same weights and prediction as R on those cases.
3. The settings layer takes the settings only, with no defaults. A choice field is refused, and each rule is shown to the model word for word.
4. Build the skill folder from `templates/configure_SKILL.md` and the texts.

**Design** (adds `scs_fit.py`)

1. Steps 1 and 2 of configure.
2. The settings and the choices, with no defaults. The fitter runs the copied code when every choice equals the reference code's value, and the native engine otherwise.
3. Build the skill folder from `templates/design_SKILL.md`.

**Profile** (`scs_profile.py`, `scs_box.sh`)

1. The request comes from `templates/profile_request.md`, filled with the method's sentence and the use's description of the data and the answer.
2. `prepare` makes a SkillFoundry working copy at commit 4a8cc0c with the paper in `paper/` and the reference code in `code/`.
3. Jun runs `scs_box.sh probe` and then `scs_box.sh run` in Jun's own terminal with the key exported.
4. `collect` keeps what SkillFoundry wrote and builds the code provided and code written formats, with the rules of the SHC profile runs.

**Shared stages**

- Tests before any model call (`scs_tests.py`).
- Run (`scs_loop.py`, `scs_run.py`). Each solver model works one case in a folder holding the use's data files, with the skill mounted read-only. It sends one bash command per turn, and every turn, fit and answer is recorded. Models, token caps, effort and step cap are those of the SHC runs.
- Compare steps (`scs_steps.py`). For configure and design this covers the fit behind each answer, its settings and choices against the reference values, its donors and weights beside the reference code's on the same case, whether reasoning is logged, and the reasoning when the answer was made. For profile it is a table that gives, for each step of the method, what the generator's scripts do and what each solver's code does. For SHC a block number means a different stretch of time when the window length differs, so donor numbers are comparable only at equal settings.

## What is decided when a pipeline is used

A use (`scs_loop.Use`, in `USES` of the method module) names the cases and the files each case folder holds, the first message to the model, the check of `answer.json`, extra read-only mounts (an R installation), the output limit, the profile request's description of the data and the answer, and how a fit's predictions map onto the answer. How answers are evaluated is chosen with the use and is not part of the pipelines.

## How to add a method

1. Write `methods/<method>/method.py` with `PACKAGE` and at least one use in `USES`.
2. Run `python scs_copy.py <method> --write`, then `python scs_check.py <method>`.
3. Draft the choices from the reference code's hard-coded choices and show them to Jun before building.
4. Write the texts and show every text the model reads to Jun before any model call.
5. Build the skills with `python scs_build.py <method> methods/<method>/skills`.
6. Run `python scs_tests.py --method <method>`. Every test must pass with no model call.
7. Jun runs the models with `python scs_run.py --method <method> --use <use> --skill <skill folder>`.
8. For the profile skill, run `python scs_profile.py prepare`. Jun then runs `scs_box.sh` for the probe and the run, and after that `python scs_profile.py collect`.
9. Compare steps with `python scs_steps.py <method> <out folder>`.

## When two results count as the same

Two numbers count as the same when they differ by at most 0.0001. Jun set this on 2026-10-31 with "when you check whether same, just 4 decimal is enough". It holds in every check that our numbers reproduce a reference, which covers the printout check, the fits, the fit script in the box, the step tables and a Python port against R. Text, file bytes, hashes, donor ids and counts are compared exactly. Each test reports how many results are identical and how many are the same within 0.0001.

## Tests before any model call

| test | what it checks |
|---|---|
| copy | the copy of the reference code equals a fresh copy |
| check | the copied code reproduces what the collaborator's notebook printed, or the R trial passes |
| refusals | every field left out or given a wrong value is refused with a message naming it |
| build | the skill files equal a fresh build |
| box_fit | the fit script in the box at the reference values gives the copied code's output on every case, in both skills |
| box_refusal | a wrong settings file run through the fit script in the box is refused naming the field |
| isolation | inside the box the home folder, the project and the network are absent and the skill files are read-only |
| dry_run | a scripted stand-in for a model reads the skill, runs the fit script once and ends with its answer |
| loop_rules | the step cap, an empty reply, an empty command output and resuming behave as in the SHC runs |
| profile_request | the request text is complete |
| profile_collect | collecting stored generator runs again gives the hashes they recorded |
| profile_probe | a working copy prepared by script passes the generator box probe |

A package that had an earlier implementation (the SHC packages, through `OLD`) also runs `same_refusals`, `same_dispatch`, `same_fits` and `same_steps` against it. The step tables are compared row by row, matched on data, skill, model, sensor, site and prompt, and column by column by name. The existing tables were built for monthly and daily data together and then split, so each holds the other data set's setting columns, and the test requires those columns to be empty.
