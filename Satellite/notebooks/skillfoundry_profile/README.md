# SkillFoundry profile experiment

The SkillFoundry profile experiment has SkillFoundry's own skill generator write a skill for the synthetic historical
control (SHC) method from the SHC paper (Chen, Yang and Yang, SSRN 4995085). The solver models then use that skill on
the same 27 cells as the simple SHC skills in `../llm_band_sc/`. Each generated skill is tested in two formats. In the
code provided format the model runs the generator's scripts. In the code written format the scripts are removed and
the model writes its own code. The results sit side by side with the simple skills in one table, scored by
`joint_nmse`. That score is the squared error at P10 divided by the band's P01 to P09 variance, averaged over bands.

Everything this experiment produces stays in this folder. The simple skills and their records stay in
`../llm_band_sc/`. The second GPT-5.4 run produced a skill. One decision remains open.

## The experiment follows SkillFoundry's purpose and Jun's purpose

SkillFoundry "converts such resources into validated agent skills", the resources being "repositories, APIs, scripts,
notebooks, documentation, databases, and papers", and it "can design new task-specific skills on demand for concrete
scientific objectives" (paper abstract). Jun's purpose for this experiment is a skill made by SkillFoundry from the SHC
paper ("the skill foudnry input is the paper. the original paper and corresponding code provided") whose results sit
in one table with the simple skills ("i want the experiment comparable with existing simple skill version. which means
they can be in one table"). Every setting follows SkillFoundry's general path for that purpose. Runs that SkillFoundry
made for other purposes supply no setting.

## Jun's instructions define the experiment

| date | Jun's words |
|---|---|
| 2026-10-12 | "we can borrow their skill profile, for what field include and how to convert code and paper into skill ... for each skill we will test on 2 format, pure prompt will be skipped" |
| 2026-10-12 | "code provided vs code written" |
| 2026-10-12 | "for which sc question, i want the experiment comparable with existing simple skill version. which means they can be in one table" |
| 2026-10-12 | "SHC only" |
| 2026-10-12 | "I need them sit in same table" |
| 2026-10-12 | "make sure borrow the profile generator or mapping from their github" |
| 2026-10-13 | "1. A default rule. agree" |
| 2026-10-13 | "It is either the nine-section checklist from their generator prompt" |
| 2026-10-13 | "i said remove, and we will write new one. nothing will be revise" |
| 2026-10-13 | "1. if skill foundry doesn't have scripts integration step thenyou can use what in simple skill, if they have, do the code generation follow their step using the paper's code. the corresponding sc method's paper." |
| 2026-10-13 | "2. i call this experiment skill foundry profile experiment, so you should follow their setting, use their generator, use their pipeline, use their procedure, use their prompt template, use their skill profile sections." |
| 2026-10-13 | "5. why we cannot have if X then Y? please get out of the notebook... the skill foudnry input is the paper. the original paper and corresponding code provided, in our case likely you need convert R to python." |
| 2026-10-13 | "13. 32000 tokens" |
| 2026-10-13 | "15. side by side table is fine." |
| 2026-10-13 | "1. comprable model first, i guess it is deepseek" / "2. explain what is this file andhow it works." / "3. b" / "4. what is skill foundry pipeline? follow." / "5. a" / "6. a" / "7. what is skill foundry pipeline? follow." / "8. what is skill foundry pipeline? follow." / "9. a" |
| 2026-10-13 | "should we try with open router? we will test both gpt-5.4 at high effort and deep seek which ever works. probably litellm proxy easier. 2. a. 4b. a. 7. a. and we can compare the model generated code and skillfoundry prompt generated code. 10. i don't understand the difference. location problem? 11. b. i like the table, please document all we discussed. all profile experiment should be in another folder rather then togather with simple skill." |
| 2026-10-13 | "1r. b. 1e. follow skill foundry. 1v. a. 7b. b. 10. don't understand. 10. follow skill foundry. for 12. what is this? this is part of paper pipeline? 13. same as 12" |
| 2026-10-13 | "12. a. 14. a. 15. do what the paper did" |
| 2026-10-13 | "16. follow the paper. 17. b. 18. yes" |
| 2026-10-13 | "any other setting we discussed is the special case? then you ask me how i should follow 17 special case?" |
| 2026-10-13 | "19 follow" and "10 follow how the paper did", each followed by the passage on the 267 library skills and their sources |
| 2026-10-13 | "wri in your plan model to avoid special case, do what the paper did for the purpose or the problem they want to solve or why i want to do the plan. you understand zero my intention of implement it result in you consider special cases. please think where you did wrong and make sure never happen again." |
| 2026-10-13 | "drop all consideratioin for old skill to new skill, this is not our consideration and we not experiment this" |
| 2026-10-13 | "just provide local pdf paper, i am worry about the ssrn robot check" |
| 2026-10-13 | "20. a. follow skillfoundry" |
| 2026-10-14 | "why you never inform me this potential issue you wated all my tokens" |
| 2026-10-14 | "b" and "delete all wrong data and code", answering whether to raise the 6,000 character output limit for the profile runs |
| 2026-10-14 | "OL. C. for 15b i don't understand where your 0.006 and 0.0006 comes from. i need exactly how the collaborator evaluate" |
| 2026-10-14 | "you should not ask 15b. it should follow the evaluation of collaboration script" |
| 2026-10-14 | "revise your evaluation result of the simple skill, should be exactly same as collaborator's evaluation, no 0.006 0.0006.0.000006" |
| 2026-10-14 | "you added this tolerance secretly without let me know. please re read the global memory to understand what you have to follow when report  to me." |
| 2026-10-15 | "all finished", after the repeated monthly solver runs |

## SkillFoundry builds a requested skill in one generator run followed by a check

SkillFoundry is described in arXiv 2604.03964 and released at `github.com/ma-compbio-lab/SkillFoundry`. Its code was
read at commit 4a8cc0c. The paper PDF inside the repository is a blank conference template, so the arXiv copy was
read instead.

SkillFoundry has two modes. The library mode grows a collection of scientific skills through five stages. These are
tree check, resource search, skill build, skill test and refresh. Each skill is then repaired, benchmarked and checked
for novelty. The task mode builds one skill that a user asks for, with the command `design-skill`. This experiment
uses the task mode, the path the paper gives for task-specific skills.

In the task mode the user's task prompt is placed inside a fixed template. The template tells the generator to follow
the workflow contract in `experiments.md`. It then asks the generator to search for papers, code and documentation,
to build a single skill path, to add repository tests and a Slurm path, and to update the registries, reports and
`experiments.md`. The template also carries their skill quality standards, items 1 to 6, which define the nine
SKILL.md sections.

The generator is the Codex command line agent, run as `codex exec` in the repository root with the model, the
reasoning effort and an output schema. Their code adds the flag `--full-auto` only when asked. Codex works inside a
copy of their repository. It searches, writes the skill with its scripts and tests, and returns a short JSON report.
Each run keeps a run folder with the manifest, the prompt, the output schema, the final message and the attempts. A check then runs at the chosen verification mode. Mode `none` runs nothing. Mode `validate`
runs the structure check `scripts/validate_repository.py`. That check requires SKILL.md, metadata.yaml, refs.md,
scripts, examples, tests and assets for every skill entered in `registry/skills.jsonl`. Mode `standard` adds the site
build `scripts/build_site.py`. Mode `full` adds the unit tests.

Their code sets GPT-5.4 at high reasoning effort for `design-skill`. Their paper, in appendix A.1.6, reports GPT-5.4
at medium effort for the same stage.

Every one of the 267 library skills lists its sources. Together they cite 326 documentation pages, 50 code
repositories, 14 tutorials, 5 package pages and 4 papers. The task mode writes a skill to the same quality standards.

Packages a skill needs are installed into a dedicated environment folder `slurm/envs/<family>`. Of their 267 skills,
69 name such a folder in SKILL.md.

## The workflow contract file is named in every prompt and was never published

`experiments.md` is the rulebook their repository gives the generator. Every stage prompt says "Follow the workflow
contract in `experiments.md`". Codex opens the file inside the repository and follows it while it works, as a project
instruction file. The generator also writes to it. Resource search records where each resource came from, and
design-skill updates the file when the repository changes. Their change log describes its contents as the repository
layout, a layout rule for their website, and a policy for adding resources without adding skills.

No commit in the public repository has ever touched the file, so its wording is unknown. The planning files
`task_plan.md`, `findings.md` and `progress.md` and the folders `ref/`, `experiments/` and `slurm/` are also absent
from the public tree. Jun chose to run their prompt as published without the file (decision G2).

## Each decision carries Jun's words and their date

Departures were proposals that differ from Jun's instruction. Jun answered no to departures 1 to 3, and departure 4 awaits an answer.

| item | proposal | Jun's answer |
|---|---|---|
| departure 1 | build the two formats from the two simple skills | no, "1. if skill foundry doesn't have scripts integration step thenyou can use what in simple skill, if they have, do the code generation follow their step using the paper's code. the corresponding sc method's paper." (2026-10-13) |
| departure 2 | Claude writes the skill texts from notebook 10 and the SHC paper | no, "2. i call this experiment skill foundry profile experiment, so you should follow their setting, use their generator, use their pipeline, use their procedure, use their prompt template, use their skill profile sections." (2026-10-13) |
| departure 3 | leave out metadata.yaml, refs.md, tests, examples and assets | no, "5. a" (2026-10-13), keep every file the generator writes |
| departure 4 | a change to a prompt template that is no longer used | withdrawn when the answer to G19 replaced the template |

| row | decision | value | Jun's words (date) |
|---|---|---|---|
| profile | meaning of profile | SkillFoundry quality standards items 1 to 6, with the nine SKILL.md sections | "It is either the nine-section checklist from their generator prompt" (2026-10-13) |
| items 4 to 12 | example values, decision logic, usage example, library usage, environment setup, script interface, troubleshooting, related skills, skill names | written by SkillFoundry's generator from the paper, with the generator inputs fixed in rows G1 to G15 | "4. realistic value migh in the paper, please check." to "12. if you can, follow skill foudry." (2026-10-13) |
| item 13 | token cap | 32,000 for DeepSeek, 16,000 for the other solver models | "13. 32000 tokens" (2026-10-13) |
| item 14 | empty output bug in the simple skill loop | fixed and tested, records made under the bug deleted, the affected DeepSeek design cells rerun | "14. this is not what i should decide. fix your bug first." (2026-10-13) |
| item 15 | table layout | simple and profile side by side, each scored as the collaborator's notebook 10 scores its held-out P10 (section below) | "you should not ask 15b. it should follow the evaluation of collaboration script" (2026-10-14); "15. side by side table is fine." (2026-10-13) |
| OL | output limit of the profile runs | none; the model sees each command's output whole | "OL. C." (2026-10-14) |
| G1 | generator models | two generators, each run separately, GPT-5.4 at high effort and DeepSeek | "we will test both gpt-5.4 at high effort and deep seek which ever works" (2026-10-13) |
| G1r | route from Codex to each generator model | GPT-5.4 through OpenAI with Jun's key; DeepSeek through OpenRouter as `deepseek/deepseek-v4.1-flash`, with no proxy | "1r. b." (2026-10-13) |
| G1e | DeepSeek reasoning effort as generator | high, the effort their code sets for design-skill and the effort GPT-5.4 runs at; their paper reports medium | "1e. follow skill foundry." (2026-10-13) |
| G1v | Codex release | 0.116.0, the latest release from 19 March 2026; their library skills were built from 11 to 22 March 2026 | "1v. a." (2026-10-13) |
| G2 | workflow contract file | their prompt runs as published, with no `experiments.md` written or supplied | "2. a." (2026-10-13) |
| G3 | task prompt | the SHC method for one parcel's own history of several bands, predicting every band at the time point after that history; no time unit, no number of observations and no dataset's columns; the agent reads `data.csv` and writes `answer.json`, with no network; the text is shown to Jun before any run | "3. b" (2026-10-13); the question on monthly and daily data and "these question is not a setting question ... they are not hyperparameters for the pipeline" (2026-10-13) |
| G4 | resources | the request names the paper by its authors and title and points to the local copy only, with no SSRN link or number; the search instruction is the one their general design-skill prompt carries for papers, repositories, docs, notebooks, workflows and package references | "just provide local pdf paper, i am worry about the ssrn robot check" (2026-10-13) |
| G4b | local copy of the paper | the SHC paper PDF is placed at `paper/ssrn-4995085.pdf` in each working copy and the request names that path as the paper | "just provide local pdf paper, i am worry about the ssrn robot check" (2026-10-13) |
| G5 | files the generator writes | every file is kept, and the solver's sandbox mounts the whole generated skill folder | "5. a" (2026-10-13) |
| G6 | check after generation | their `standard` mode, the structure check then the site build; the evaluate-skills mode is not run | "6. a" (2026-10-13) |
| G7 | code written format | the generated skill with its scripts removed, the way shc-implement is shc-design without its fitter, its five modules and settings.md; the code the solver models write is compared with the generator's scripts | "7. a. and we can compare the model generated code and skillfoundry prompt generated code." (2026-10-13) |
| G7b | SKILL.md, tests, examples and references of the code written skill, which name the removed scripts | tests and examples that run the removed scripts are removed; SKILL.md and references are kept as generated | "7b. b." (2026-10-13) |
| G8 | packages the skill needs | the generator installs them into `slurm/envs/<family>` in its working copy, as 69 library skills do, and the solver's sandbox runs that environment's Python | "8. what is skill foundry pipeline? follow." and "10 follow how the paper did" (2026-10-13) |
| G9 | estimator | the paper's SHC | "9. a" (2026-10-13) |
| G10 | where the generator writes and whether the skill is registered | into their library under `skills/<domain>/<slug>/`, entered in their registry, as the 267 library skills are and as their general design-skill prompt says; the G6 check then examines the skill | "10 follow how the paper did" the passage on the 267 library skills (2026-10-13) |
| G11 | generator runs | one design-skill run per generator, their on-demand path for a task-specific skill; no refine run | "do what the paper did for the purpose or the problem they want to solve or why i want to do the plan" (2026-10-13) |
| G12 | how the solver models' code is compared with the generator's scripts | as the paper does, a table that states for each SHC step what the generator's scripts do and what each solver model's code does, next to the scores | "12. a." (2026-10-13) |
| G13 | folder | this folder holds the record, the generated skills, the run records, the scores and the side by side table | "all profile experiment should be in another folder rather then togather with simple skill" (2026-10-13) |
| G14 | what of the generator's working copy is kept in the repository | what their pipeline keeps, each run folder and every file the generator writes or changes, copied into `generated/<generator>/` with the diff against commit 4a8cc0c; the working copy itself stays out of git | "14. a." (2026-10-13) |
| G15 | what Codex may do while it generates | `--full-auto` with network access on and live web search; the paper describes the generator retrieving documentation, repositories, package references, notebooks and papers, and running each skill under execution tests, which needs all three | "15. do what the paper did" (2026-10-13) |
| G15t | timeouts | their defaults, 3600 s per generator run and 3600 s for the check | "follow their setting" in departure 2's answer (2026-10-13) |
| G16 | how the solver's task message names a generated skill | no name; the paper gives every setting the same prompt and varies only whether the skill is there, so the message is the simple skills' task text without its skill line, the same for both generators and both formats | "16. follow the paper." (2026-10-13) |
| G17 | focus term | "Causal inference", the leaf of their knowledge tree where SHC belongs; with no term the prompt lists eight leaves chosen for library coverage, unrelated to SHC | "17. b." (2026-10-13) |
| G18 | the generator's environment | `/data/wang/junh/envs/skillfoundry`, Python 3.11.17, Node.js 20.20.2 and Codex 0.116.0, installed with the two approved commands | "18. yes" (2026-10-13) |
| G19 | how the skill is generated | their design-skill command with its general prompt; the request is written in their documented form, "Design a skill for ...", and carries G3, G4 and G4b; the generator writes the skill from the paper and the documentation and code it finds, to the same quality standards as the library | "19 follow" the passage on the 267 library skills (2026-10-13) |
| G20 | Codex helper agents | Codex's default, helper agents allowed and their model chosen by the generator, as SkillFoundry's code leaves it | "20. a. follow skillfoundry" (2026-10-13) |
| G8d | the solver's Python when the generated skill names no environment | the interpreter the simple skills ran with, `/data/wang/junh/envs/satellite/bin/python`, with the simple skills' PATH; the skill's script worked under both candidate Pythons, and one Python across the table leaves the skill as the only difference | "you can use what ever environment works" (2026-10-13) |
| G14d | whether `sf_collect.py` also keeps each generator's Codex session file | yes, copied to `generated/<generator>/codex_session.jsonl` with its source path and sha256 in the manifest; a run with more than one session file stops the collection and goes to Jun | "a" (2026-10-13) |
| G15r | what the generator can read | its working copy with the paper, its own Codex home, the generator environment and system files, inside bubblewrap with the network; nothing else on the disk | "the skill foundry input is the paper. the original paper and corresponding code provided" (2026-10-13); the leak Jun asked about (2026-10-13) |

## Codex reaches GPT-5.4 through OpenAI and DeepSeek through OpenRouter

| check | result | source |
|---|---|---|
| ARC Responses endpoint | a POST to `llm-api.arc.vt.edu/api/v1/responses` with DeepSeek-V4.1-Flash-thinking-low returned HTTP 405 | Jun's terminal, 2026-10-13 |
| GPT-5.4 on Jun's OpenAI key | the model list shows gpt-5.4 with its dated, mini, nano and pro versions | Jun's terminal, 2026-10-13 |
| Codex and chat completions | Codex deprecated its chat completions mode on 9 December 2025, and its maintainers said support would end on 1 February 2026; the current configuration reference accepts only the Responses protocol | openai/codex discussion 7782, Codex configuration reference |
| Codex releases | 0.116.0 was the latest release from 19 March 2026, 0.125.0 the last before their commit of 26 April 2026, 0.162.1 the latest on 13 October 2026 | npm registry |
| LiteLLM bridge | the proxy converts Responses requests into chat completions; for an OpenAI compatible endpoint the bridge is forced with `use_chat_completions_api`; release 1.101.6 of 8 October 2026 fixed Codex tool calls through the bridge; the latest release is 1.105.0 | LiteLLM docs and release notes, PyPI |
| OpenRouter | Codex connects to `openrouter.ai/api/v1` with the Responses protocol; `openai/gpt-5.4` and `deepseek/deepseek-v4.1-flash` are listed with tool calls and reasoning effort | OpenRouter Codex tutorial, OpenRouter model list |
| SkillFoundry's Codex call | `codex exec` with `--cd`, `--output-schema`, `-o`, `-m` and the reasoning effort; `--full-auto` only when asked; their README example of design-skill passes none | `skillfoundry/codex_exec.py`, `skillfoundry/cli.py`, `README.md` at 4a8cc0c |
| Codex sandbox | `codex exec` runs in a read-only sandbox by default; `--full-auto` is now a deprecated compatibility flag and `--sandbox workspace-write` the preferred one; network access in the writable sandbox is a separate setting with no default stated; web search defaults to a cached index kept by OpenAI with no external web access | Codex non-interactive guide, Codex configuration reference |
| this host | Node.js 20.20.2 and Codex 0.116.0 now in `/data/wang/junh/envs/skillfoundry`; the system conda only runs with `PYTHONNOUSERSITE=1`, because a cryptography 50.0.2 package in the user's `~/.local` shadows conda's own | install log |
| Codex 0.116.0 settings | `--full-auto`, `--output-schema`, `-o` and `--cd` exist; `web_search="live"` and `sandbox_workspace_write.network_access=true` load, and an unknown web search value is refused; the program reads `CODEX_API_KEY` | `codex exec --help` and two settings checks with no model call |

## The generator's request is written in SkillFoundry's documented form

`prompts/design_task.md` is the request for the design run, written as their documented usage writes one ("Design a
skill for ..."). It carries the SHC method, the task, and the paper as a local copy with no SSRN link. SkillFoundry's own
prompt builder wraps it in the general design-skill prompt with the focus term "Causal inference", and the result is
`prompts/rendered/design.txt`. Each generator gets one design run.

## The experiment runs in five steps from this folder

The generator runs and the solver runs need keys exported in Jun's terminal. A generator run takes up to an hour. Collecting and testing need no key.

| step | command | what it does |
|---|---|---|
| generate | `bash run_generator.sh gpt-5.4` with `OPENAI_API_KEY` exported | SkillFoundry's design-skill command in `work/gpt-5.4/`, logged to `generated/gpt-5.4/run.log` |
| generate | `bash run_generator.sh deepseek` with `OPENROUTER_API_KEY` exported | the same command in `work/deepseek/` with DeepSeek through OpenRouter |
| collect | `python sf_collect.py <generator>` | keeps what the run wrote in `generated/<generator>/` and builds `skills/<generator>/code_provided/` and `code_written/` |
| test | `/data/wang/junh/envs/satellite/bin/python sf_tests.py` | gates PF1 to PF9 of the solver runner, the generator box, the daily harness and the output limit, with no model call |
| solve, monthly | `bash run_solvers.sh monthly` with `ARC_LLM_API_KEY` and `OPENAI_API_KEY` exported | `sf_run.py` under the satellite Python for both generators and both formats, the 27 monthly cells, records in `runs/`, printed output also in `runs/solvers_monthly.log` |
| solve, daily | `bash run_solvers.sh daily` with the same two keys exported | `sf_daily.py` the same way on the 52 daily cells, printed output also in `runs/solvers_daily.log`; `python sf_daily.py --cells` lists the cells |
| score | `/data/wang/junh/envs/satellite/bin/python sf_table.py [--generators gpt-5.4 deepseek]` | the side by side tables of row 15 on the monthly cells in `results/`, scored as notebook 10 scores its held-out P10, with a number registry row for every number of the tables |

The working copies `work/gpt-5.4/` and `work/deepseek/` are clones of SkillFoundry at commit 4a8cc0c with the paper at
`paper/ssrn-4995085.pdf`, and git ignores them. `codex_home/config.toml` defines the OpenRouter provider and holds no
key. Each run copies it into its own Codex home, `codex_home/<generator>/`.

## The first GPT-5.4 run failed inside Codex's sandbox

Codex 0.116.0 runs every shell command through bubblewrap with the option `--argv0`, which this host's bubblewrap
0.6.1 does not have. Every command of the first GPT-5.4 run returned "bwrap: Unknown option --argv0", so the
generator could neither read nor write the repository. It searched the web twice and started five helper agents on
gpt-5.1-codex-mini, which failed the same way. The run was interrupted after about seven minutes. Its six Codex
sessions used 2,635,859 tokens, of which 2,023,808 were cached input and 48,424 were output. It wrote nothing outside
its own run bookkeeping. Its records were deleted and the working copy was cloned again.

`run_generator.sh` now switches Codex to its own Landlock sandbox (`features.use_legacy_landlock=true`). Run under that
switch with no model call, a command runs, writes inside its folder, is refused a write outside it and reaches the
network, which is the policy G15 sets.

## Codex keeps its default for helper agents

Codex 0.116.0 lets the generator start helper agents and choose their model (`multi_agent`, on by default), and
SkillFoundry's code leaves the setting alone. In the failed run GPT-5.4 chose gpt-5.1-codex-mini for its five helpers.
Jun chose the default (G20).

## The second GPT-5.4 run wrote a standard-library SHC skill

This run and the four sections after it concern the skill made from the month-only request. That request was replaced the same day, and the run's records were deleted, as the last section explains.

The second GPT-5.4 run took about nine minutes. It made 39 model requests and used 3,853,337 tokens, of which
3,643,648 were cached input and 44,480 output, which is about $1.99 at OpenAI's published GPT-5.4 rates. It searched
the web zero times and started no helper agents. It wrote the skill
`skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecasting/`, entered it in
their registry with the paper as its source resource, and added a smoke test and a Makefile target. Their structure
check and site build both passed, with 268 skills and 376 resources. The skill applies a one-step synthetic
historical control to each band separately, chooses a matching window of four to seven months, and can remove a
linear trend first. Its script uses only the Python standard library and needs Python 3.10 or later.

The skill names no package environment, so the first collection stopped as row G8c requires. Jun answered G8d with
"you can use what ever environment works". In the solver's sandbox the script ran on its own example and on one real
cell, Sentinel-1 parcel treatment_0015, under the satellite environment's Python 3.11.15 and under the generator's
Python 3.11.17, and gave the same forecast under both. The skill's unittest failed under both, because the test finds
the script through the path of the SkillFoundry repository, which the solver's working directory does not have. The
solver models run on the Python the simple skills ran with, `/data/wang/junh/envs/satellite/bin/python`, so the table
compares the skills on one Python.

## The solver reads ten files in the code-provided format and eight in the code-written format

`sf_collect.py gpt-5.4` then kept the same 38 files with the same contents as the stopped attempt and built the two
formats in `skills/gpt-5.4/`. Every file below is bound read-only into the solver's working directory beside
`data.csv`.

| file | lines | code provided | code written |
|---|---|---|---|
| `SKILL.md` | 373 | yes | yes |
| `metadata.yaml` | 53 | yes | yes |
| `refs.md` | 5 | yes | yes |
| `assets/README.md` | 8 | yes | yes |
| `assets/toy_answer.json` | 8 | yes | yes |
| `assets/toy_diagnostics.json` | 232 | yes | yes |
| `examples/README.md` | 5 | yes | yes |
| `examples/toy_data.csv` | 28 | yes | yes |
| `scripts/run_shc_satellite_band_forecast.py` | 676 | yes | removed by G7c |
| `tests/test_run_shc_satellite_band_forecast.py` | 94 | yes | removed by G7c, since it names the script |

G7c keeps `SKILL.md` as generated in both formats. In the code-written format its text names the removed script five
times.

## The skill takes its procedure from the paper and its settings from the generator

The request in `prompts/design_task.md` names no window, tolerance or troubleshooting content. The skill's way of
choosing the matching window and the donor blocks is the paper's (Section 2.4). The numbers it uses are the
generator's own, and the paper's numbers sit in a passage the generator never read.

| setting | the paper | the generated skill | the generator read the paper's value |
|---|---|---|---|
| matching window m | the largest m between two preset bounds whose matching MSE is at most ν | the same rule | the rule, yes |
| bounds of m | 10 and 40, for quarterly UK GDP | 4 and 7, with a floor of 4 in the code | no |
| ν | 0.1% of the sample variance of the outcome | 0.25 of the variance of the smoothed band series | no |
| stepwise donor selection | add blocks until the gain in matching MSE is at most ζ | the same rule, the gain divided by the variance | the rule, yes |
| ζ | 0.001% of the same sample variance | 0.01 | no |
| largest donor pool | all N blocks | four blocks | not in the paper |

The generator turned the PDF into text and read lines 1 to 260 and 470 to 1,385 of it, which hold the introduction,
Section 2 apart from lines 260 to 470, and the start of the simulations in Section 3. Its two reads of lines 260 to 980 were
sent in the same batch as the conversion and ran before the text file was written, so they returned nothing. A keyword
search covered the whole text. The paper sets ν, ζ and
the bounds of m in footnotes 16 and 17 of its Brexit application in Section 4.1, at lines 1,744 to 1,751, which none of
these reads reached. Codex stores the generator's reasoning encrypted, so the log does not show
why it chose 0.25, 0.01 and 4 to 7, and the skill states no reason. With nine months the code allows windows up to
seven months.

The Troubleshooting section is required by SkillFoundry's prompt, which asks for the three to five most common failure
modes with their symptoms and remedies. The generator wrote its five entries from its own script's error messages and
from the task the request describes, namely the four column names, the periods P01 to P09 and the missing network.

## The generator's run is logged in three places

| log | path | what it holds | kept by `sf_collect.py` |
|---|---|---|---|
| SkillFoundry's stage record | `generated/gpt-5.4/files/reports/framework-runs/20261013-135123-shc-gpt-5.4/stages/01-design-skill/` | `prompt.md` is the full prompt; `stderr.txt` is Codex's printed transcript with every command, its output and the final message; `last_message.json` and `attempts.json` are their bookkeeping | yes, by G14c |
| the generator's text of the paper | `generated/gpt-5.4/files/scratch/ssrn-4995085.txt` and `scratch/tmp/ssrn-4995085.txt` | the `pdftotext` output whose lines it read | yes, by G14c |
| Codex's session file | `codex_home/sessions/2026/10/13/rollout-2026-10-13T13-51-23-01a13663-f841-76b0-bc12-0f27b7b54a25.jsonl`, copied to `generated/gpt-5.4/codex_session.jsonl` | every request with its timestamp and token counts, every command and the output the model received, and the reasoning in encrypted form; the header of `stderr.txt` reads "reasoning summaries: none" | yes, by G14d |

Neither log file holds a string shaped like an API key. The DeepSeek run's session file sits in the same folder.

Jun chose to keep it (G14d), so `sf_collect.py` copies the session file to `generated/<generator>/codex_session.jsonl`.

## The generated SHC code differs from the collaborator's SHC in seven steps

The comparison is between the generated script `run_shc_satellite_band_forecast.py` and the collaborator's held-out
SHC of notebook 10 (`data/shc/10_shc_monthly_multiple_outcomes.ipynb`, extracted byte for byte into
`llm_band_sc/lb_her_nb10.py`), the reference the simple skills are scored beside. Both predict P10 from P01 to P09,
both build the same historical blocks, and both forecast P10 as the weighted P10 analogue of the chosen blocks.

| step | collaborator, notebook 10 held-out | generated script |
|---|---|---|
| seasonality | removes an annual sine and cosine fitted after a linear trend is partialled out (`estimate_seasonal_beta_and_latent`) | none; instead it may remove a straight-line trend and add the trend back at P10 (`choose_transform`) |
| smoothing (stage 1) | local-linear Gaussian kernel, bandwidth fixed at 2 months | the same smoother, bandwidth chosen per band by leave-one-out error from 1, 1.5, 2, 2.5 and 3 months |
| bands | one weight vector shared by all bands of a sensor, fitted on the bands' standardized latents stacked together | each band fitted alone, with its own window, blocks and weights |
| window m | fixed at 4; the largest m with standardized MSE at most 1.0 is reported in a sensitivity block and not used | chosen per band from 4 to 7, the largest whose MSE over the variance is at most 0.25 |
| stepwise donors | adds the block that lowers the joint MSE most while the drop exceeds 0.0001; up to all five blocks | the same greedy rule, while the drop over the variance of the matched window exceeds 0.01; at most four blocks |
| weights | nonnegative and summing to one, solved by SLSQP in scipy | nonnegative and summing to one, solved by projected gradient descent in plain Python |
| forecast | weights times the observed values of the blocks' P10 analogues | the same, plus the trend at P10 when detrended; a negative forecast of a nonnegative band is set to zero, and a flat band copies P09 |

## The request now holds for monthly and daily data alike

Jun asked on 13 October 2026, after the collaborator moved the data to daily visits (notebook 14,
`data/scripts/scripts/validation/shc/14_shc_daily_sentinel1_sentinel2.ipynb`, and
`data/daily_Sep2020_Sep2025_datasets/`), "for the later on run with skill, i want to test on old monthly data and this. then there is a problem, seems like we should not emphasize the month or time window, the paper should be general to different time window or time unit?"

The paper is written for any time unit. Its model is "a semi-parametric time-series regression for {yt}" whose latent
part is "a non-stochastic, smooth function of time" (Section 2.1), m and n count periods, and both of its applications
use quarterly GDP. Notebook 14 applies the same method to satellite visits. Its time axis is calendar days, so the
smoother respects the uneven spacing between visits, and m and n count visits. It removes no season, picks m by the
paper's rule with ν set to 0.1 of each band's variance, and holds no forecast of a held-out visit. Its validation is
four checks of the fitted weights and fit.

| | monthly data (the simple skills' cells) | daily data |
|---|---|---|
| observations | composites P01 to P10, November 2023 to August 2024 | single visits, `period_id` D0001 onward, dates from September 2020 |
| sites | 12 Sentinel-1 and 15 Sentinel-2 treatment sites, 27 cells | 26 treatment sites per sensor, all of which pass notebook 14's filter of 60 visits before Helene and eight after |
| visits before Helene per site | nine months given to the solver | 117 to 220 for Sentinel-1, 155 to 242 for Sentinel-2 |

The request both generator runs used names one month ahead, nine consecutive calendar months, P01 to P09 and P10.
Both skills are tied to it. The GPT-5.4 script refuses anything but nine observations with P-numbered periods and
consecutive months, and the DeepSeek skill says "One parcel has exactly nine consecutive monthly observations per band".
Neither can take the daily data. The DeepSeek run took about half an hour, wrote 268 token count records and used
12,990,334 tokens, of which 12,173,696 were cached input and 160,600 output, about $0.20 at OpenRouter's listed DeepSeek
rates. The later DeepSeek run writes two such records per model request, so the month run most likely made 134
requests. Its log is deleted, so this cannot be checked.
Both of their checks passed, and its final reply did not parse as JSON.

Both runs' records were deleted on 13 October and both working copies were cloned again at 4a8cc0c with the paper, as
the failed first run was handled and as Jun's rule of 28 September on superseded records requires. The findings in
the sections above stay recorded here.

The request in `prompts/design_task.md` now reads

> Design a skill for forecasting a land parcel's satellite band values with the synthetic historical control method of Chen, Yang and Yang. The paper, Synthetic Historical Control for Policy Evaluation, is at `paper/ssrn-4995085.pdf`. The agent that uses the skill works in a directory holding `data.csv`, one parcel's own history of several bands, one row per band and time point. It predicts every band's value at the time point after that history and writes `answer.json` as {"prediction": {"<band>": <number>, ...}, "rationale": "<text>"}. The agent has a bash shell and no network access.

It names no time unit, no number of observations and no dataset's columns. `prompts/rendered/design.txt` is SkillFoundry's
design-skill prompt with this request in place of the old one, and nothing else changed.

## Each test dataset reaches the solver as the collaborator's code and the paper set it

Jun said "these question is not a setting question, this is what you think too narrow before, you should not number
them in general setting. they are not hyperparameters for the pipeline" and "you becomes lazy, not thinking, just raise
question for what the answer already exist in paper or in the collaborator's code". The test harness below is
therefore not part of the numbered settings, and each line follows its source.

| item | monthly data | daily data | source |
|---|---|---|---|
| cells | the 27 cells of the simple skills | every treatment site that passes notebook 14's filter, 26 per sensor, 52 cells | `llm_band_sc/lb_prompts_shc.jsonl`; notebook 14 cell 2 (`SITE_PREFIX`, `MIN_PRE_VISITS`, `MIN_POST_VISITS`) |
| what the solver gets and predicts | P01 to P09, predicts P10 | every clean visit before 27 September 2024 but the last, predicts the last | notebook 10 holds out the last pre-treatment period and trains on the ones before it (`lb_her_nb10.py` lines 56 to 59); notebook 14's `INTERVENTION_DATE` |
| `data.csv` columns | `band, period_id, month, mean_value` | `band, period_id, date, y` | each panel's own names for the band, period, time and value |
| score | joint_nmse with each band's P01 to P09 variance | joint_nmse with each band's variance over the visits given | notebook 10's `train_var` over the training periods (`lb_her_nb10.py` lines 1338 to 1360) |
| task message | G16 | G16 with the visit in place of the month, "for the next visit, <period_id> = <date>, from its own past visits" | G16, "16. follow the paper"; notebook 14 cuts the blocks on the visit order, so m and n count visits, and smooths on days since the first visit (cell 5, `visit_index` and `t`); the daily `period_id` is the calendar day number, D0001 for 1 September 2020, so its numbers skip the days without a visit |

## The month-only DeepSeek run read the held-out answers, so the generator now runs in a closed box

Jun saw the DeepSeek run's final reply name "the GPT-5.4 profile skill", the "code-provided vs code-written comparison
harness", "joint_nmse" and "the simple SHC skills", none of which is in the request or in SkillFoundry's prompt. Codex's
workspace-write sandbox limits writes only. Every file on the disk stayed readable, and the working copy sat inside
this folder. Scripts the run left in `/tmp` show what it read.

| file in `/tmp` | what it did |
|---|---|
| `eval_real.py` | read `llm_band_sc/lb_features_monthly.csv`, wrote each of the 27 cells' P01 to P09 file, ran its skill and scored the forecast against the true P10 with joint_nmse, beside persistence |
| `patch_joint.py` | rewrote its script so that one weight vector is shared by every band, the collaborator's design; its file time is after `eval_real.py` |
| `inspect_cell.py` | printed one cell's true P10 beside its forecast |
| `eval_cmp.py` | scored all cells again and compared each with the collaborator's `her_joint_validation_nmse` in `llm_band_sc/lb_her_nb10_results.csv` |

The Codex log that would show how it found these files was deleted with the run's records before the leak was seen.
The command list of the second GPT-5.4 run, printed before its log was deleted, named only paths inside its working
copy and `/tmp`, and its longer inline scripts were printed cut short.

Jun named the generator's input, "the skill foundry input is the paper. the original paper and corresponding code
provided". `run_generator.sh` therefore runs SkillFoundry's command inside bubblewrap (row G15r). The box shows the
working copy with the paper, the generator's own Codex home under `codex_home/<generator>/`, the generator environment,
system files and the network. `/tmp` and the home directory are empty, and the environment is cleared except for the
one key the run needs. `bash run_generator.sh --probe <generator>` checks this with no model call, and gate PF7 runs it
for both generators. Both pass. The experiment record, the simple skills, the collaborator's data, the other
working copy, the home directory and the host `/tmp` are absent inside the box. The network answers, and Codex's own
sandbox still writes inside the working copy and not outside it. The leftover Codex state of the month runs was
deleted. The `/tmp` files above stay until Jun has seen them.

## Both generators wrote their skill from the general request inside the box

Jun ran both generators on 13 October with the general request and the box of row G15r. After the generator, SkillFoundry's
design-skill command runs its own two checks (row G6), the structure check of the whole library and the site build, and
both passed after each run. Each skill names no package environment, so both run under G8d. Jun asked on 13 October to log
the table below and these two checks.

| | GPT-5.4 | DeepSeek |
|---|---|---|
| duration | about nine minutes | about 22 minutes |
| model requests | 44 | 140 |
| tokens | 4,157,659 | 10,379,768 |
| of which cached input | 3,857,152 | 9,662,976 |
| of which output | 44,312 | 154,332 |
| cost | about $2.27 at OpenAI's GPT-5.4 rates | about $0.17 at OpenRouter's listed rates |
| tool calls | 72 | 193 |
| skill folder in their library | `skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecast/` | `skills/statistical-and-machine-learning-foundations-for-science/synthetic-historical-control-band-forecast/` |
| structure check, `python3 scripts/validate_repository.py` | "Repository validation passed: 376 resources, 268 skills.", exit code 0 | "Repository validation passed: 377 resources, 268 skills.", exit code 0 |
| site build, `python3 scripts/build_site.py` | "Generated site data for 268 skills, 376 resources, and 0 framework runs.", exit code 0 | "Generated site data for 268 skills, 377 resources, and 0 framework runs.", exit code 0 |
| their verdict | "Design-skill finished with 0 verification failure(s)." | "Design-skill finished with 0 verification failure(s)." |
| their library status line | `Resources=376 Skills=268 CoveredLeaves=254 FrontierLeaves=0 TodoLeaves=0` | `Resources=377 Skills=268 CoveredLeaves=254 FrontierLeaves=0 TodoLeaves=0` |
| final reply | parsed as JSON | did not parse as JSON |
| files kept by `sf_collect.py` | 34 | 295 |
| files the solver sees, code provided and code written | 10 and 6 | 13 and 9 |

The "tokens used" line Codex prints, 300,507 and 716,792, counts the uncached input and the output. DeepSeek's session
file holds two token count records for each request, 280 records in all. Of the DeepSeek files kept, 221 are the
outputs of their benchmark tests and 33 of their skill review tests, which the generator ran inside its working copy.

Both Codex session files were read in full before the collection. Every command of the GPT-5.4 run ran inside its
working copy. Beyond it, the run read only Codex's own guide to writing skills in its Codex home. It searched the web
zero times and downloaded nothing. The DeepSeek run listed `/data/wang/junh/envs/`, where the box shows only the
generator environment, and the folders above its working copy, where it saw only the path down to it. It installed
numpy into the box's empty `/tmp`, which ended with the box. It downloaded one file from the web, the module
`mlsynth/utils/shc_helpers/orchestration.py` of the mlsynth package on GitHub (user jgreathouse9), and printed its first
60 lines. These are a docstring that lists the package's SHC steps and the module's imports. The package implements the
paper and is not the paper authors' code. The DeepSeek skill names it as a second source in `metadata.yaml` and
`refs.md`. Row G15 gives the generator network access and live web search.

GPT-5.4 read lines 1 to 260, 430 to 1,315 and 1,450 to 1,585 of the paper's `pdftotext` text. DeepSeek read lines 1 to
120 and 281 to 1,205 of the layout version. Their keyword searches covered the whole text. None of these reads reached
the footnotes that set ν, ζ and the bounds of m, at lines 1,744 to 1,751 and 1,965 to 1,968 of the first text and 1,690
to 1,696 and 2,028 to 2,034 of the second.

## In the code-written format the solver loses each skill's scripts and the files that name them

`sf_collect.py` built `skills/<generator>/code_provided/` and `code_written/` from each skill folder as generated. Every
file below is bound read-only into the solver's working directory beside `data.csv`. Both skills use only the Python
standard library.

| skill | file | lines | code written |
|---|---|---|---|
| GPT-5.4 | `SKILL.md` | 322 | kept |
| GPT-5.4 | `metadata.yaml` | 32 | kept |
| GPT-5.4 | `refs.md` | 11 | kept |
| GPT-5.4 | `assets/README.md` | 3 | kept |
| GPT-5.4 | `examples/README.md` | 4 | kept |
| GPT-5.4 | `examples/toy_satellite_history.csv` | 37 | kept |
| GPT-5.4 | `scripts/run_shc_satellite_forecast.py` | 709 | removed |
| GPT-5.4 | `scripts/validate_answer.py` | 83 | removed |
| GPT-5.4 | `tests/README.md` | 5 | removed, it names the script |
| GPT-5.4 | `tests/test_run_shc_satellite_forecast.py` | 104 | removed, it names the script |
| DeepSeek | `SKILL.md` | 313 | kept |
| DeepSeek | `metadata.yaml` | 24 | kept |
| DeepSeek | `refs.md` | 39 | kept |
| DeepSeek | `assets/README.md` | 19 | kept |
| DeepSeek | `assets/example_answer.json` | 9 | kept |
| DeepSeek | `assets/example_summary.json` | 170 | kept |
| DeepSeek | `assets/example_truth.json` | 6 | kept |
| DeepSeek | `examples/README.md` | 27 | removed, it names the script |
| DeepSeek | `examples/toy_parcel_data.csv` | 241 | kept |
| DeepSeek | `examples/toy_parcel_truth.json` | 6 | kept |
| DeepSeek | `scripts/run_shc_band_forecast.py` | 787 | removed |
| DeepSeek | `scripts/validate_answer.py` | 130 | removed |
| DeepSeek | `tests/test_run_shc_band_forecast.py` | 113 | removed, it names the script |

## The two scripts follow the paper's steps with settings of their own

| step | the paper | GPT-5.4 script | DeepSeek script |
|---|---|---|---|
| columns read | none | `band`, `time`, `value` or listed aliases, among them `date` for time | `band`, `time`, `value` or listed aliases, among them `date` for time and `y` for value; also two wide layouts |
| trend | detrend or difference the whole series first when it trends (lines 608 to 609) | runs each band on its values and on its first differences and keeps one that did not fall back to least squares, then the closer match | removes a fitted straight line when its rise is at least 0.2 of the band's standard deviation |
| controls | the vector of control variables in the semi-parametric model (Section 2.1), which may include lagged outcomes (line 269) | every band at the previous observation with an intercept, fitted by least squares; the match runs on the residual | none by default; the band's own lags as an option |
| time axis | a smooth function of time (Section 2.1) | the observation's position in the file | the observation's position in the file |
| smoothing | Gaussian kernel, bandwidth by leave-one-out cross validation (line 923) | local-linear Gaussian, bandwidth 0.18, 0.25, 0.33, 0.5 or 0.67 of the series length by leave-one-out | local-linear Gaussian, bandwidth one of ten fractions from 0.05 to 1.0 of the series length by leave-one-out |
| window m | the largest m between two bounds whose matching MSE is at most ν; bounds 10 and 40, ν 0.1% of the sample variance in both applications | 4 to 10, the longest whose RMSE over the window's standard deviation is at most 0.35, else the closest | 4 to 24, the largest whose MSE on the standardized band is at most 0.05, else the closest |
| donor blocks | stepwise until the gain is at most ζ, 0.001% of the sample variance | stepwise, at most four blocks, while the RMSE gain exceeds 0.02 of the window's standard deviation | all blocks up to the latest 120; stepwise and single block as options |
| weights | nonnegative, summing to one | the same, by projected gradient | the same, by Frank-Wolfe |
| forecast | the weights applied to the blocks' next values | the least squares prediction plus the weighted next residual of the blocks; the least squares prediction alone when the normalized RMSE exceeds 0.6125 | the weighted next value of the blocks, back on the band's scale with the trend added |

Notebook 14 smooths on days since the first visit, so the uneven spacing of visits enters its kernel. Both scripts
smooth on the observation's position, so daily visits are treated as evenly spaced.

## Neither script reads the monthly file as shipped

Each shipped script ran once with no model on two files. The first is the monthly cell of Sentinel-1 parcel
treatment_0015 (P01 to P09). The second is a daily file of Sentinel-2 parcel treatment_0001 cut for this check from its
164 visits before 27 September 2024 other than the last, since `sf_daily.py` is not built yet. No forecast was scored.
Both skill folders hash as collected afterwards.

| file | GPT-5.4 script | DeepSeek script |
|---|---|---|
| monthly, `band, period_id, month, mean_value` | stops with "Could not find a time column" | stops; every band "has fewer than 2 points" and no band is forecast |
| daily, `band, period_id, date, y` | stops with "Could not find a value column" | runs and forecasts all eight bands |

In the code-provided format the solver must therefore rename the columns before either script runs on a monthly cell,
and before the GPT-5.4 script runs on a daily cell. Each `SKILL.md` gives a renaming step in its procedure or its
troubleshooting. The request names no columns (row G3).


## The profile runs use the simple skills' loop with the changes the generated skills require

Jun asked on 13 October, "for monthly, why we cannot use the same run as the simple skill experiment? we only changed
skills?" `sf_run.py` runs the simple skills' loop itself. It imports `run_cell`, the system text, the sandbox, the step
cap, the answer check, the model calls with their retries and the record shape from `llm_band_sc/` and copies none of
them. It changes only what the generated skills require. Each change was tried on both skills with the simple loop and
no model.

| what differs | simple runner | why the generated skills need the change |
|---|---|---|
| which skill | `lb_skill_run.py --skill` accepts only shc-configure, shc-design and shc-implement, and nothing in `llm_band_sc/` is edited | the generated skills sit in `skills/<generator>/<format>/` |
| what the sandbox mounts | `SKILL.md`, `references/` and `scripts/` | neither generated skill has `references/`, and the simple loop's sandbox stops with "Can't find source path .../references" for all four folders; G5 keeps every file the generator wrote, so `metadata.yaml`, `refs.md`, `assets/`, `examples/` and `tests/` are mounted too |
| the task message | names the skill from the front matter of `SKILL.md` | the DeepSeek `SKILL.md` has no front matter, and the simple task text stops on it; G16 drops the skill line for every arm |
| the solver's Python | the satellite Python | unchanged, by G8d, since neither skill names an environment |
| output shown to the model | the first 6,000 characters of each command's output | each generated `SKILL.md` is longer than 6,000 characters (section on the deleted runs below); decision OL shows the whole output, and the simple loop's limit is restored after each profile cell |

## The daily cells reach the solver through `sf_daily.py`

`sf_daily.py` builds the 52 daily cells of the test harness above and runs them through the same profile loop. It
changes only `data.csv` and the first sentence of the task message. Jun asked on 13 October to "first run monthly, then
while i run it you build daily", so `sf_run.py`, `sf_collect.py` and `llm_band_sc/` stay unchanged while the monthly runs
are going. A cell's `data.csv` holds every clean visit before 27 September 2024 that all bands of the sensor share,
except the last, which is held out. The bands are named as in the monthly cells. The Sentinel-1 cells give 116 to 219
visits and hold out 16 or 21 September 2024. The Sentinel-2 cells give 154 to 241 visits and hold out 22 September 2024.
`python sf_daily.py --cells` lists them. Gate PF8 checks the cells against the panels, the working directory, the task
message and a canned dry run. All eight gates pass under the satellite Python.

## The first monthly solver runs were deleted because the loop cut each generated SKILL.md

The profile runs use the simple skills' loop, which shows the model at most 6,000 characters of each command's output
(`OUT_LIMIT` in `llm_band_sc/lb_skill_loop.py`). The GPT-5.4 `SKILL.md` holds 13,523 characters and the DeepSeek
`SKILL.md` 16,024, so every `cat SKILL.md` showed the model the first 6,000 only. The DeepSeek skill's Procedure section
starts at character 6,443. The simple skills' `SKILL.md` files hold 1,608 to 2,906 bytes and were never cut. The limit
was not among the parameters declared to Jun before the runs, and the mismatch was reported only after all 16 monthly
reply files had run. Jun asked on 14 October, "why you never inform me this potential issue you wated all my tokens".

Jun then chose to raise the limit for the profile runs ("b", 14 October) and asked to "delete all wrong data and code".
The 16 monthly reply files, their log and the three files `sf_table.py` had built from them were deleted the same day.
No code file was deleted. Jun chose no limit ("OL. C.", 14 October). `sf_run.py` sets it in the profile loop and
`sf_daily.py` in the daily loop, and both restore the simple loop's 6,000 characters after each cell. Gate PF9 checks
that a monthly and a daily profile cell show the model a long output whole. The monthly runs are then repeated.

## The table scores every answer as the collaborator's notebook 10 scores its held-out P10

Jun asked on 14 October that the table "should follow the evaluation of collaboration script". `sf_table.py` scores
the simple and the profile skills with one implementation, the functions of `llm_band_sc/lb_skill_score.py`, which
Jun asked the same day to make "exactly same as collaborator's evaluation". No comparison uses a tolerance. It builds
for each cell the rows of her `heldout_validation_for_sensor` in
`data/scripts/scripts/validation/shc/10_shc_monthly_multiple_outcomes.ipynb` (cell 25). For each band it takes the
squared error at P10 and divides it by the band's P01 to P09 variance with one degree of freedom. The variance is
missing when it falls below her tolerance. The mean over bands is the cell's joint_validation_nmse. A cell with no
valid answer counts as a skipped site, as a site whose validation fails is skipped in her cell 26. The tables give
her four held-out summaries. These are the processed and skipped sites per sensor (cell 34) and the mean
standardized P10 squared error per sensor (cell 30). They add the number, mean and median of joint_validation_nmse
over the processed sites of both sensors (cell 28) and the mean squared and standardized squared error per sensor and
band (cell 30). Applied to her SHC as `llm_band_sc/lb_shc.py` runs it, the tables reproduce every number her notebook
printed for the held-out P10, to six decimals. These include 0.492461 for Sentinel-1, 0.786712 for Sentinel-2, and a
mean of 0.655934 and a median of 0.415021 over the 27 sites. The tables are `results/table_monthly.md` and
`results/table_monthly_bands.md`, and every number in them has a row in `results/numbers_table_monthly.csv`.

## The repeated monthly runs finished on 15 October with no output cut

Jun ran `bash run_solvers.sh monthly` once, from 14 to 15 October. It wrote all 16 reply files with one record for
each of the 27 cells. No record holds the loop's truncation mark. Of the 432 cells, 381 ended with an answer and 51 at
the step cap of 12 steps. Every cell that ended with an answer is a processed site in the table.

| generator and format | gpt-oss-120b | GLM-5.3 | DeepSeek-V4.1-Flash-thinking-low | openai:gpt-5.5 |
|---|---|---|---|---|
| gpt-5.4 code provided | 3 | 0 | 0 | 0 |
| gpt-5.4 code written | 4 | 3 | 3 | 0 |
| deepseek code provided | 1 | 0 | 0 | 0 |
| deepseek code written | 11 | 0 | 26 | 0 |

Cells that ended at the step cap, of 27 in each file.

In the code-provided format every solver model ran the shipped script on all 27 cells. On the GPT-5.4 skill, GLM-5.3,
DeepSeek and GPT-5.5 report the same value in all 156 cell and band rows, and so the same summaries. On the DeepSeek
skill the three agree in 148 of the 156 rows. The GPT-5.4 script first regresses each band on an intercept and on every
band at the previous month (`build_level_design`, lines 428 to 444, with `ridge = 1e-6` at line 174). A Sentinel-2
cell holds eight bands over nine months, so that regression fits nine coefficients to eight observations, or to seven
on first differences. The forecast is the regression's prediction plus the matched residual, or the prediction alone
when the match is poor (lines 489 to 493). At five Sentinel-2 sites its forecast for B2, B3, B4 or B12 is below zero.

Nine profile rows cover all 27 sites, the sites of the reference rows. Each of the nine has a higher mean
joint_validation_nmse than her SHC (0.655934) and a higher mean standardized P10 squared error on each sensor. One of
the nine has a lower median than her SHC, GPT-5.5 on the DeepSeek skill in the code-written format (0.372070 against
0.415021). GLM-5.3 and GPT-5.5 also answered all 27 sites with shc-design, and the shc-design mean of each is lower
than the mean of each of its profile rows. The other seven profile rows processed fewer than 27 sites.

## Sources

- SkillFoundry paper, `https://arxiv.org/abs/2604.03964`
- SkillFoundry code, `https://github.com/ma-compbio-lab/SkillFoundry`, commit 4a8cc0c
- History of `experiments.md`, `https://api.github.com/repos/ma-compbio-lab/SkillFoundry/commits?path=experiments.md`
- Codex configuration reference, `https://learn.chatgpt.com/docs/config-file/config-reference`
- Codex non-interactive guide, `https://learn.chatgpt.com/docs/non-interactive-mode`
- Codex chat completions deprecation, `https://github.com/openai/codex/discussions/7782`
- Codex releases, `https://registry.npmjs.org/@openai/codex`
- LiteLLM Responses API, `https://docs.litellm.ai/docs/response_api`
- LiteLLM release notes, `https://docs.litellm.ai/release_notes/v1.101.6/v1-101-6.md`
- OpenRouter Codex tutorial, `https://openrouter.ai/blog/tutorials/codex-cli-openrouter/`
- OpenRouter model list, `https://openrouter.ai/api/v1/models`
