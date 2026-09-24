# Three ways to give a language model a method, and what each one changed

**Design note for the meeting of 22 September 2026.** Continues the language-model section of the
19 September analysis (`9-19-llm-analysis.md`). That analysis asked what kind of error the models
make when the synthetic historical control procedure is written into the prompt. This note
separates three different mechanisms for handing a model a method — a prompt description, a skill,
and a tool — establishes what each one actually is, and reports what the project has measured for
each. Only the first has been run. The other two are described as designs, and every statement
about them is marked as such.

---

## 1. The three mechanisms are not three sizes of the same thing

**Method.** The band-space synthetic historical control design, with hosted language models used as
the weight solver.

The three differ on one axis: **who holds the procedure, and who executes it.**

| | Prompt description | Skill | Tool |
|---|---|---|---|
| What it is | text in the request | a directory of files: instructions, reference documents, executable scripts | a callable interface declared to the model |
| Where the procedure lives | in the request, every call | on a filesystem, read when relevant | split — the model chooses arguments, the host owns the code |
| Who executes the arithmetic | the model, in its own reasoning | a script the model runs itself | the host, outside the model |
| What the model sees of the code | there is none | the source, and it may edit or extend it | nothing — only the returned result |
| Cost per call | the full text, always | about 100 tokens until triggered, then the instruction body | the schema, always |
| Can exist alone | yes | **no** | yes |

The last row is the one that settles the question of how a skill differs from a tool. A skill is not
a peer of a tool; it is built on one. The published specification is explicit that skills on the
Claude API require the code execution tool, *"whose container Skills run in"*
(fetched 21 September 2026 from the Agent Skills overview). The instructions are read with a shell
command; the bundled scripts are run with a shell command. Remove the shell and a skill is an
unreadable folder.

A working statement of the distinction:

> A **tool** adds a verb the model did not have. A **skill** adds knowledge of what to do,
> and may ship code the model runs through a verb it already had. A **prompt description** adds
> neither, only the request that the model do it in its head.

### The parts of a skill, and which part is the new one

A skill is read in three stages, each entering the request at a different time:

1. **Name and description** — always present, about 100 tokens, used only to decide whether the
   skill is relevant at all.
2. **The instruction body** — the procedure in prose, under 5,000 tokens, read when the skill is
   triggered.
3. **Bundled files and scripts** — read or run only when the instructions call for them. A script's
   source never enters the request; only its output does.

Two skills installed on this machine show the shape in practice. The spreadsheet skill is a
99-line instruction file plus one 11 KB recalculation script; the instructions do not explain how to
recalculate a workbook, they say to run the script and refuse to ship while it reports errors. The
PDF skill is 314 lines plus eight scripts and two reference documents that are read only when
needed.

**The consequence for this project is the third stage.** Stages one and two are prose — the same
material as a prompt description, delivered from disk instead of from the request. Our step-by-step
prompt is already an instruction body without frontmatter. A skill that ships no executable script
would test nothing the 19 September analysis has not tested.

---

## 2. Only the prompt description has been run, and it is the arm that is measured

**Method.** The same design, two prompts carrying identical evidence and the same one-line answer
format. The direct prompt gives a band guide, the site's nine monthly rows, and the instruction to
predict the tenth month. The step-by-step prompt adds the procedure from the collaborator's paper
in prose — smoothing, five overlapping four-month windows, stepwise selection of windows on fit,
and a weighted combination of the months that followed the kept windows. The step block is a fixed
1,161 characters.

Nothing in the runner supports the other two mechanisms. The request body built at
`notebooks/llm_band_sc/lb_lib.py:310` carries the model name, the messages, the temperature, a
token cap and a seed. There is no tool declaration and no second turn, so no model in this study
has ever been offered a callable or a filesystem.

### The step block costs 279 tokens of request and changes how much the models think

Median tokens per call, from the stored usage records in the reply files:

| Model | Arm | Calls | Request tokens | Reasoning and answer tokens |
|---|---|---|---|---|
| gpt-oss-120b | direct | 54 | 869 | 471 |
| gpt-oss-120b | step-by-step | 27 | 1,148 | **102** |
| Kimi-K3 | direct | 26 | 901 | 1,482 |
| Kimi-K3 | step-by-step | 19 | 803 | **5,714** |

The gpt-oss-120b rows are the clean comparison: the same 27 sites in both arms, so the 279-token
rise is the step block itself. The same added prose moves the two models in opposite directions.
Given the procedure, gpt-oss-120b produces about a fifth of the reasoning it produced without it.
Kimi-K3 produces close to four times as much. Neither arrives anywhere better. The Kimi-K3 rows sit
on different site sets — eight of its step-by-step calls failed — so the request-token figures for
that model are not comparable across arms and no inference is drawn from them.

### Neither model was made more accurate by being told the procedure

Mean over sites of the squared error at P10 divided by the variance of that band across P01–P09,
averaged over bands — the collaborator's held-out measure. Lower is better.

| Arm | Sentinel-1 | Sentinel-2 |
|---|---|---|
| Kimi-K3, direct | 0.400 (n=11) | 1.257 (n=15) |
| Kimi-K3, step-by-step | 0.448 (n=11) | 0.918 (n=8) |
| gpt-oss-120b, direct | 0.566 (n=12) | 0.585 (n=15) |
| gpt-oss-120b, step-by-step | 0.495 (n=12) | 0.769 (n=15) |
| Reference implementation of the design | 0.492 (n=12) | 0.787 (n=15) |
| Repeating the last observed month | 0.492 (n=12) | 0.789 (n=15) |
| Mean of the last three months | **0.265** (n=12) | **0.541** (n=15) |

The sign of the change is different for each model on each sensor, and the largest movement rests
on eight sites. The arithmetic mean of the last three months is better than every language-model
arm and better than the reference implementation, on both sensors.

### The reasoning traces show the description was read, and not executed

Classification of every stored trace on the step-by-step arm, from
`notebooks/llm_band_sc/lb_why_p09.csv`:

| | Traces | Compared the five windows | Derived why the last window wins | Objected to the result |
|---|---|---|---|---|
| Kimi-K3 | 19 | 19 | 19 | 7 |
| gpt-oss-120b | 27 | 3 | 0 | 0 |
| GPT-5.5 (summaries) | 25 | 9 | 2 | 1 |

Kimi-K3 read the description, worked out from it that the nearest window is the treated window moved
back one month, said in seven traces that the procedure was therefore trivial, and returned the
answer anyway. No weight was ever fitted and no fit was ever computed. The GPT-5.5 column is
written by a separate summarizer rather than by the model, and is never pooled with the other two.

**What the prompt description achieved, stated exactly:** the models can restate the procedure and
reason about its structure. None of them ran it.

---

## 3. What a skill would change, and what it would confound

**Not yet built. Every statement in this section is a design expectation, not a result.**

A skill for this design would be an instruction file plus a fitting script in a sandbox the model
can run. The model would read the instructions, write its own code around the script, and call it.

What it would add over the prompt description is the third stage only — the executable script. The
prose is what we already have.

Two obstacles make it the wrong instrument for the immediate question.

- **It measures two things at once.** A model that fails would have failed either at the design or
  at writing the code around it, and the transcript will not separate them. The question from the
  19 September analysis is specifically whether the models can set the problem up; a skill arm
  answers a broader and less decidable question.
- **It cannot be run across the four models.** A skill requires a filesystem and shell per model.
  The research gateway serving Kimi-K3, gpt-oss-120b and GLM-5.3 is called single-shot, and whether
  it passes tool calls through at all is unverified. Only GPT-5.5 could plausibly run a skill, and
  a single-model result cannot be compared with the table in §2.

---

## 4. What a tool would change, and why the interface has to be thin

**Not yet built. Every statement in this section is a design expectation, not a result.**

A tool arm declares a callable, the model chooses its arguments, and the host runs the fit. It
works over an ordinary chat request with a second turn, which keeps all four models on the same
footing and keeps the §2 table as the comparison.

The risk is that the argument list gives the method away. A callable declared as
`run_shc(donor_sites, blocks, pre_window)` puts the entire design into its own signature; a model
that fills three named slots has demonstrated nothing about the design, and the arm would measure
form-filling.

**The interface should therefore expose the arithmetic and nothing else** — a matrix and a target
vector in, a weight vector out, with no notion of sites, months or blocks in the signature. Then
constructing the matrix is the whole test: which columns are donors, which rows are the fitting
period, which month is withheld. That is precisely the part no trace in §2 performs, and the
fitting itself is arithmetic there is no reason to score.

A tool arm is decidable in a way the current arms are not. On the step-by-step prompt, following
the procedure correctly and skipping it entirely both produce the last observed month, so the two
cannot be told apart in the answer. With the fit handed over, the arguments are recorded, and what
the model chose is visible whatever it answers.

---

## 5. What this note establishes and what remains open

Established, from measurement: the prompt description raises request cost by 279 tokens, moves the
two models' reasoning volume in opposite directions by factors of roughly five and four, improves
neither model reliably, and produces traces in which the procedure is understood and not executed.

Established, from the specification and from the installed examples: a skill is instructions plus
bundled code that the model runs itself, requires a shell to exist at all, and differs from a prompt
description only in its executable stage; a tool is a callable the host executes and the model never
sees.

Open, and not answered here: whether either model, given a fitting routine it does not have to
implement, supplies a donor set, a fitting period and a withheld month that are correct — and
whether the result beats repeating the last observed month. No arm in this project addresses that,
and the runner cannot currently run one.

---

## 6. Sources

| Claim | Source | Checked |
|---|---|---|
| Skill structure, the three reading stages, the shell requirement | Agent Skills overview, platform.claude.com | 21 September 2026 |
| Instruction-file and script sizes of two installed skills | `~/.claude/skills/synced/.../xlsx`, `.../pdf` | 21 September 2026 |
| Request body carries no tool declaration | `notebooks/llm_band_sc/lb_lib.py:310` | 22 September 2026 |
| Step block length, 1,161 characters | `notebooks/llm_band_sc/lb_lib.py`, the step constant | 22 September 2026 |
| Token medians per model and arm | `notebooks/llm_band_sc/lb_replies_shc_*.jsonl`, stored usage records | 22 September 2026 |
| Outcome table | `notebooks/llm_band_sc/lb_summary_shc.csv` | 22 September 2026 |
| Trace classification | `notebooks/llm_band_sc/lb_why_p09.csv` | 22 September 2026 |

The outcome figures use the collaborator's held-out measure and no other: per band, the squared
error at P10 over the variance of that band across P01–P09, averaged over bands.
