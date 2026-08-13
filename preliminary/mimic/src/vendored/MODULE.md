# Module: vendored

## Purpose

Self-contained copies of four pieces of infrastructure lifted from
`/data/wang/junh/githubs/moa-clinical-rag` on 2026-08-01, so the latent
synthetic control study has a stable substrate that does not move when that repo
does. This is the scaffolding phase: none of it is synthetic-control logic
itself. It supplies the four capabilities the headline experiment needs —
turning clinical notes into vectors (`embedding`), turning structured codes into
a covariate matrix plus a predictive baseline to beat (`features`), putting
bootstrap confidence intervals on any forecast metric (`stats`), and running a
local LLM for both generation and perplexity scoring of notes (`llm`). The
donor-weight fitting, the pre/post visit split and the forecasting experiment
live elsewhere (`src/synthctl/`, `src/datasets/`) and are not part of this
module.

The code was **copied, not imported**, deliberately. The source repo is under
active edits and at least one of the source files does not import at all:
`src/metrics/tjur_readmission_full.py` reads `REPO_ROOT` at module scope
(line 11) without ever defining or importing it, so `import` raises `NameError`
immediately. Only pure functions were lifted from it.

## Structure

```
src/vendored/
├── __init__.py     re-exports embedding, features, stats. NOT llm (importing
│                   it pulls in vllm, which is slow and GPU-adjacent).
├── embedding.py    HF encoder -> cached, L2-normalised vectors
├── features.py     code bags -> binary design matrix + CV logistic baseline
├── stats.py        AUROC / Tjur's D / bootstrap CIs
├── llm.py          vLLM wrapper: batched generate + NLL scoring
└── MODULE.md       this file
```

No file imports any other file in the package; there is no internal call path.
`torch`/`transformers` (in `embedding`) and `vllm` (in `llm`) are imported
lazily inside the functions that need them, so importing the package is cheap
and does not touch a GPU.

Public API:

```python
# embedding.py
sha(s: Optional[str]) -> str
embed(model_name: str, texts: Iterable[str], cache_path, batch: int = 32,
      maxlen: int = 512, device: Optional[str] = None,
      verbose: bool = True) -> Dict[str, np.ndarray]

# features.py
_flatten(field: Any) -> List[str]
_bag(rec: Dict[str, Any], prefix: bool = True) -> Set[str]
design_matrix(bags, vocab: Optional[Dict[str, int]] = None,
              min_df: int = 5) -> Tuple[np.ndarray, Dict[str, int]]
cv_auroc(bags, y, folds: int = 5, seed: int = 0, min_df: int = 5,
         C: float = 1.0, max_iter: int = 2000) -> Dict[str, Any]

# stats.py
auc_mw(y, p) -> float
tjur(y, p) -> float
boot(y, p, fn, B: int = 2000, seed: int = 0,
     alpha: float = 0.05) -> Tuple[float, float]
boot_continuous(v, fn=None, B: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> Tuple[float, float]   # NEW, not vendored
star(lo: float, hi: float, null: float) -> str

# llm.py
class QwenIntegrator:
    __init__(model_name, gpu_id=None, gpu_memory_utilization=0.85,
             tensor_parallel_size=1, trust_remote_code=True,
             enforce_eager=True, lazy=True, verbose=True)
    generate(prompts: Union[str, Sequence[str]], max_tokens=32768,
             temperature=0.7, top_p=0.9, repetition_penalty=1.2,
             stop=None) -> Union[str, List[str]]
    __call__(prompt: str, max_tokens=32768, temperature=0.7) -> str
    score(texts: List[str]) -> List[float]
    shutdown() / config()
```

### Additions after vendoring

`stats.boot_continuous` is **new code, not vendored** (added 2026-08-01 by the
Phase 2 methodology pass). Nothing was removed and no vendored function
changed behaviour. See the `stats.py` rows in Hyperparameters and the
`boot` / `boot_continuous` note under Known limitations.

### Patches applied after vendoring

The files are copies, but they are **not frozen**. Two defects in `features.py`
were found by agent B (Phase 1), worked around rather than fixed, and then
**patched in place on 2026-08-01 by agent D (Phase 2)**. Both are recorded as
changes #8 and #9 in `features.py`'s own header. **Both were independently
re-confirmed on 2026-08-01** — see "Re-verification" below the table.

| # | function | was | now | why it mattered |
|---|---|---|---|---|
| 8 | `_flatten` | `for v in field or []` | `if field is None: field = []` then `for v in field` | `field or []` evaluates `bool(field)`, which raises `ValueError: The truth value of an array with more than one element is ambiguous` on a numpy array of length > 1. pandas materialises this project's Phase 0 parquet `list<str>` columns as object ndarrays, so **`features._bag` could not consume Phase 0's output at all** — the two modules in the repo did not compose. Phase 1 worked around it in `state._codes`; that workaround is now redundant (but harmless, and `state.py` is not to be modified). The nested-element test was widened from `isinstance(v, list)` to `(list, tuple, np.ndarray)` in the same edit, so a genuinely nested ndarray flattens instead of being `str()`-ed into `"[ 'a' 'b' ]"`. |
| 9 | `design_matrix` | `kept = [code for code, n in c.items() if n >= min_df]` | `kept = sorted(code for code, n in c.items() if n >= min_df)` | Callers pass `set`-valued bags; `Counter.update(set)` therefore inserts in `set` iteration order, which for `str` keys depends on `PYTHONHASHSEED`. The vocabulary's **column order — and hence `X` — was not reproducible across interpreter processes.** Also affected `cv_auroc`, which rebuilds a vocab per fold. Sorting makes the matrix a pure function of its inputs. |

**Verification (as run, 2026-08-01).** The same matrix was built from the Phase 0
parquet in three separate interpreter processes at `PYTHONHASHSEED` = 1, 987654
and 42, via `_bag` applied *directly* to parquet rows (the composition that used
to raise):

```
shape (9582, 585)   nnz 426240
sha1(X)     = 18ae4ebcc38fbbad89aee9ab0f62930b1a549a56
sha1(vocab) = 230eb3ff90b37bf6c395157fc5361a4ab72ea244    identical across all three
```

The pre-patch `_flatten` was re-created inline and confirmed to raise
`ValueError` on the very first parquet cell. `_flatten` was re-checked on `None`,
an empty ndarray, a flat list, a nested list and a nested object ndarray.

**Re-verification (independent, 2026-08-01, `scripts/smoke_phase2_fixes.py`).**
Both patches were checked again from scratch because they had been *claimed* by
an agent that was halted before it reported, and a claim is not a verification.
Result: **both are genuinely present.**

- Source-level, via an **AST round-trip that strips docstrings first** — a plain
  substring scan false-positives, because `_flatten`'s own docstring quotes the
  old `for v in field or []` in its patch note. In executable code, `field or []`
  is absent, `if field is None` is present, and the nested check names
  `tuple` and `np.ndarray`. `design_matrix` contains `kept = sorted(`.
- Behaviourally: `_flatten(np.array(["a","b","c"], dtype=object))` returns
  `["a","b","c"]` — the exact call that raised before — and `None`, an empty
  ndarray, a nested list and a nested object ndarray all behave.
- Cross-process column order: the **same** bags were built in three separate
  interpreter processes at `PYTHONHASHSEED` = 1, 987654 and 42.
  `sha1(X.tobytes())` and `sha1(vocab)` are identical across all three:

  ```
  shape (300, 48)
  sha1(X)     = 4ac83047eeb9b1a97271a626e35d1e8a6db1f5af
  sha1(vocab) = 84fd722549e05c5446143c10cf1b351b3019af0d
  ```

  **These are SYNTHETIC bags** (300 random code sets over a fixed 48-code
  vocabulary, `random.Random(1234)`), *not* the Phase 0 parquet. The
  MIMIC-derived digests recorded above (`18ae4ebc…` / `230eb3ff…`, shape
  `(9582, 585)`) were **not** re-derived, because that task was run under a hard
  constraint against touching MIMIC data. So the *mechanism* is confirmed
  hash-seed independent; the specific MIMIC artifact's digest was taken on
  trust from the earlier record.

**Consequence for existing artifacts — no rebuild needed.** `design_matrix`
change #9 is a *column permutation*; no cell value changes. The already-built
`data/derived/states_bag.npz` was written under the old (hash-dependent) order,
so its columns are **not** in sorted order — but the `vocab` array saved
alongside it is in that same old order, so the file is internally consistent.
Checked explicitly: after aligning rows and applying the vocab permutation, the
stored `X` is **bit-identical** to a freshly built one, and the stored vocab is
exactly a permutation of the sorted vocab. Phase 2 therefore consumes the
existing npz unchanged and always reads column names from the npz's own `vocab`,
never from a fresh `design_matrix` call. Anyone who *re-runs*
`run_phase1_states.py` will get a differently-permuted (and from then on stable)
column order; any consumer that hardcoded column indices would break, and none
does.

Provenance, per file (full detail in each file's header):

| target | source path | source lines |
|---|---|---|
| `embedding.py` | `src/metrics/build_embeddings.py` | 62-85 (`embed`), 32 (`sha`) |
| `features.py` | `src/metrics/feature_baseline.py` | 30-89 |
| `stats.py` | `src/metrics/tjur_readmission_full.py` | 29-63 (`tjur`, `boot`, `star`) |
| `stats.py` | `src/metrics/threshold_sweep.py` | 104-120 (`auc_mw`) |
| `llm.py` | `src/agents/run_condition_D.py` | 45-92 (`QwenIntegrator`) |

## Hyperparameters

Values **as used**, i.e. what the code will actually do if you call it without
arguments. "inherited" means the value came from the source repo and was kept
deliberately so behaviour matches the original.

| name | value | set where | why this value |
|---|---|---|---|
| `batch` (embedding) | 32 | `embedding.embed` arg default | Inherited. Never tuned; at maxlen 512 and fp16 this fits comfortably on one GPU. Not re-benchmarked here. |
| `maxlen` | 512 | `embedding.embed` arg default | Inherited. MedCPT is a BERT-family encoder with a 512-token limit, so this is the ceiling, not a choice. **Clinical notes routinely exceed it and are silently truncated** — see Known limitations. |
| CLS pooling index | 0 | `embedding.embed` body | Inherited, and required: MedCPT was trained with CLS pooling, so mean pooling would not match its retrieval geometry. |
| L2 norm epsilon | 1e-12 | `embedding.embed` body | Inherited. Guards a zero vector; small enough not to perturb real norms. |
| checkpoint interval | every 20 batches | `embedding.embed` body | Inherited. At batch 32 that is a 640-text loss window on a crash. |
| model dtype | fp16 on cuda, fp32 on cpu | `embedding.embed` body | Inherited. Halves memory; encoder inference is not precision-critical. Means **GPU and CPU runs are not bit-identical**. |
| hash | sha1 of utf-8, `errors='ignore'` | `embedding.sha` | Inherited. Content addressing, not security — collision risk is irrelevant here. |
| `min_df` | 5 | `features.design_matrix` / `cv_auroc` | Inherited. Drops codes appearing in <5 training units, controlling the long tail of rare ICD codes. Now reachable via `cv_auroc` (it was hardcoded and unreachable in the source). |
| `folds` | 5 | `features.cv_auroc` | Inherited. Stratified, so class balance is preserved per fold. |
| `seed` (CV) | 0 | `features.cv_auroc` | Inherited. Fixed for reproducibility. Only ONE seed is run — the reported std is across folds, not across seeds. |
| `C` | 1.0 | `features.cv_auroc` | Inherited sklearn default. **Not tuned.** No inner CV over C, so the baseline is un-tuned and may be beatable. |
| `max_iter` | 2000 | `features.cv_auroc` | Inherited. Well above the lbfgs default of 100 to avoid convergence warnings on wide sparse matrices. |
| `class_weight` | `'balanced'` | `features.cv_auroc`, hardcoded | Inherited, deliberately still hardcoded. Matters for imbalanced outcomes; note it changes the meaning of the predicted probabilities, so `tjur` on these outputs is not on a natural-rate scale. |
| feature coding | binary presence | `features.design_matrix` | Inherited. A code repeated across visits counts once — counts and visit timing are discarded. |
| section prefixes | `c:`/`p:`/`d:` | `features._bag`, `prefix=True` | Inherited. Keeps the same raw code distinct across conditions/procedures/drugs. |
| `B` (bootstrap) | 2000 | `stats.boot`, `stats.boot_continuous` | Inherited. Enough to stabilise a 95% percentile interval; the 2.5th percentile of 2000 draws is the ~50th order statistic. In `boot` the *effective* B is below nominal whenever a resample is single-class; in `boot_continuous` **all B are kept**, because there is no rejection rule. |
| `seed` (bootstrap) | 0 | `stats.boot`, `stats.boot_continuous` | Inherited. Fixed so published CIs are reproducible. `boot_continuous` uses the identical draw sequence (`default_rng(seed)`, then `rng.integers(0, n, n)` per replicate, in the same order), so its bounds are bit-identical to `boot`'s wherever `boot` discarded nothing. |
| `alpha` | 0.05 | `stats.boot`, `stats.boot_continuous` | New argument on `boot`; default reproduces the source's hardcoded `[2.5, 97.5]`. Same default and same percentile call in `boot_continuous`. |
| `fn` (`boot_continuous`) | **the mean** when omitted | `stats.boot_continuous` | Called as `fn(v_resampled)` — one argument, not `boot`'s `(y, p)`. The mean is the default because every current caller (`evaluate.boot_mean`, `evaluate.paired_boot`) wants a mean of a per-patient quantity, so making it explicit at each call site would be noise. |
| class guard (`boot_continuous`) | **none** | `stats.boot_continuous` | The whole reason the function exists. `boot`'s guard discards any resample that is single-class in `y`; for a per-patient RMSE, F1 or paired difference there is no `y`, so the guard is meaningless and it silently lowers the effective resample count. Phase 2 previously fabricated an alternating 0/1 dummy to get past it. Removing the guard also means `boot_continuous` cannot raise `RuntimeError`. |
| interval type | plain percentile, **not BCa** | both | Unchanged and inherited. No bias correction, so both are somewhat anti-conservative at small n or for a skewed statistic. `boot_continuous` does not improve on this; it only removes an inappropriate guard. |
| `null` for `star` | caller-supplied | `stats.star` | 0.5 for AUROC, 0.0 for Tjur's D. No multiple-comparison correction anywhere. |
| `gpu_memory_utilization` | 0.85 | `llm.QwenIntegrator.__init__` | Inherited. Leaves headroom on a shared box. **Never exercised here** — no engine was started. |
| `tensor_parallel_size` | 1 | same | Inherited. Single-GPU. |
| `enforce_eager` | True | same | Inherited. Disables CUDA graph capture: slower per token, lower memory, faster startup. |
| `trust_remote_code` | True | same | Inherited. Required by some Qwen revisions. Note vLLM 0.11.0 warns it is ignored for the `LLM` path. |
| `lazy` | True | same | **New.** Engine builds on first use, not at construction, so the class can be imported and introspected without a GPU. |
| `temperature` (generate) | 0.7 | `llm.generate` | Inherited. Sampling, not greedy — repeated calls will not agree. |
| `top_p` | 0.9 | `llm.generate` | Inherited. |
| `max_tokens` (generate) | 32768 | `llm.generate` | Inherited. Equals the model's full context, so this is effectively "no cap". |
| `repetition_penalty` | 1.2 | `llm.generate` | Inherited. Fairly aggressive; it biases the distribution and so **perturbs likelihoods** — a reason not to reuse generation params for scoring. |
| `stop` | `["<|im_end|>", "</s>"]` | `llm.generate` | Inherited. |
| `temperature` (score) | 0.0 | `llm.score` | **New.** Nothing should be sampled; the single generated token is discarded. |
| `max_tokens` (score) | 1 | `llm.score` | **New.** vLLM requires >=1; we only want the prompt's logprobs. |
| `prompt_logprobs` | 0 | `llm.score` | **New.** 0 = the actual prompt token's logprob with no extra top-k alternatives. Verified valid in vLLM 0.11.0 (`sampling_params.py:162`). |
| chat template in `score` | NOT applied | `llm.score` | **New, deliberate.** We want the likelihood of the text itself, not of the text wrapped in an instruction envelope. `generate` still applies it. |

## Inputs

| path | schema | provenance |
|---|---|---|
| (none at import) | — | The package reads no files on import and has no configured input paths. |
| `cache_path` arg to `embed` | pickle: `{sha1_hex: np.float32 vector}` | Written and re-read by `embed` itself. Created if absent. Content-addressed, so it is safe to share across runs and models **only if you use a distinct file per model** — the key is the text hash, not the model. |
| `bags`, `y` args to `features` | `Sequence[Set[str]]`, `np.ndarray` of 0/1 | Produced by `src/datasets/` (agent B) from MIMIC-III/IV. Not built here. |
| `texts`/`prompts` args to `llm` | `list[str]` | Clinical note text; PhysioNet-restricted. |
| model weights | HF id or local path | Resolved by transformers/vLLM from the local HF cache. No downloads were performed. |

The MIMIC roots (`mimic3_root`, `mimic4_root`, `mimic4_note_root`) live in
`preliminary/config/paths.yaml`. **Nothing in this module reads that file** —
paths are passed in by the caller.

## Outputs

| path | schema | who consumes it |
|---|---|---|
| `<cache_path>` (`embed`) | pickle `{sha1: float32 vector}`, L2-normalised, dim = model hidden size | The synthetic-control code in `src/synthctl/`, which needs note vectors per patient-visit. Look up with `cache[sha(text)]`. Gitignored (`*.pkl`, `.emb_cache/`). |
| return of `embed` | same dict, in memory | Caller. Includes pre-existing entries, not just newly computed ones. |
| return of `design_matrix` | `(X float32 (n, |vocab|) binary, vocab {code: col})` | Covariate matrix for donor-weight fitting; also the baseline's features. |
| return of `cv_auroc` | dict: `auroc_mean/std`, `auprc_mean/std`, `n_pos`, `n`, `n_feat_last_fold`, `folds`, `seed` | The baseline row the synthetic-control forecast is compared against. |
| return of `boot` | `(lo, hi)` floats | The outcome-AUROC CI in `src/synthctl/evaluate.py`, paired with `star`. |
| return of `boot_continuous` | `(lo, hi)` floats | Every per-patient CI in `src/synthctl/evaluate.py` — `boot_mean` and `paired_boot`, i.e. the code-set F1, Brier, RMSE and cosine intervals and every paired comparison. |
| return of `score` | `list[float]` mean NLL in nats, `nan` where nothing was scoreable | Perplexity comparisons: `exp(nll)`. |
| — | — | This module writes **no** result files itself. Anything under `results/`, `figures/`, `data/derived/` is written by callers and is gitignored as PhysioNet-derived. |

## How to run

The smoke test is the only executable entry point in this module:

```bash
cd /data/wang/junh/githubs/latent-synthetic-control
/data/wang/junh/envs/medrag/bin/python preliminary/mimic/scripts/smoke_vendored.py
```

Exits 0 iff all four modules pass. No network, no GPU, no model download.

To use the library, put `preliminary/mimic/src` on the path:

```python
import sys; sys.path.insert(0, "/data/wang/junh/githubs/latent-synthetic-control/preliminary/mimic/src")
from vendored.stats import auc_mw, boot, star
from vendored.features import design_matrix, cv_auroc
from vendored.embedding import embed, sha
from vendored.llm import QwenIntegrator     # import separately; pulls in vllm
```

The default `python` on PATH has none of the dependencies. Use the absolute
interpreter path above.

## Validation performed

Everything below is from the actual smoke-test run recorded above, on
2026-08-01, with `/data/wang/junh/envs/medrag/bin/python`. Final result:
**4/4 PASS, exit 0.**

```
PASS  embedding    |v|=1.000000 dim=32 cache 3->4 idempotent dedup ok
PASS  features     signal AUROC=1.000 noise AUROC=0.536 nfeat=7
PASS  stats        auc_mw==sklearn (tied too) CI=[0.793,0.875] null=[0.448,0.553] star ok
PASS  llm          gpu_mem_util=0.85 lazy=True sigs ok (engine NOT started)
```

**embedding** — checked against a locally constructed 2-layer random BERT
(hidden 32, vocab 212), never a downloaded model. `sha` is deterministic,
collision-free on near-identical inputs, and maps `None` to the empty-string
digest. Output vectors are float32, dim 32, and L2 norm measured **1.000000**
(tolerance 1e-4). Cache went 3 entries → unchanged on a repeat run (idempotent,
vectors bit-identical) → 4 after adding one text, with the pre-existing vectors
unchanged. Three identical input strings collapsed to **1** cache entry,
confirming content addressing.

**features** — `_flatten` handles nested lists, `None`, and non-string codes.
`_bag` prefixes sections correctly and keeps the same raw code distinct across
sections. `min_df=2` dropped the singleton code as expected; the matrix is
strictly binary and the column contents were checked cell-by-cell. Passing a
train vocab to a test fold drops unseen codes rather than erroring, and the
unseen code did not leak into the matrix. On 120 synthetic units with one
perfectly predictive code, 5-fold CV AUROC was **1.000**; on pure noise with the
identical labels it was **0.536**, i.e. near chance — this is the check that
would have caught label leakage through the fold split. Re-running at seed 0
reproduced the AUROC exactly.

**stats** — `auc_mw` returns exactly 1.0 / 0.0 on perfect and inverted
separation, exactly 0.5 on all-tied scores, and `nan` on single-class input.
Agreement with `sklearn.metrics.roc_auc_score` on 400 random points is within
**1e-12**, *and also within 1e-12 after rounding scores to 1 decimal to force
heavy ties* — the tie correction is genuinely right, not just right on distinct
scores. `tjur` hits 1.0 at perfect calibrated separation, 0.0 on a constant
predictor, and shrinks when scores are squashed at fixed ranking, confirming it
is calibration-sensitive. `boot` produced CI **[0.793, 0.875]** on signal
(brackets the point estimate, clears 0.5, `star` fires) and **[0.448, 0.553]**
on null data (straddles 0.5, `star` blank). Fixed seed reproduces the interval
exactly; a different seed changes it; `alpha=0.10` gives an interval no wider
than the 95% one. Single-class input raises `RuntimeError` rather than silently
returning `nan`.

**llm** — signatures only. `score` is `(self, texts)` returning `List[float]`;
`generate` takes `prompts` annotated `Union[str, Sequence[str]]` and returns a
`Union`, confirming batching; the inherited defaults (0.7 / 0.9 / 32768 / 1.2 /
0.85 / eager) are all present. Construction leaves `llm is None`. Prompt
formatting applies the Qwen template for a Qwen model name and passes text
through unchanged otherwise. Blank-input fast paths return without building an
engine. The anti-label fallback is verified absent from **executable code**, by
stripping comments and docstrings via an AST round-trip — a plain substring scan
false-positives on `llm.py`'s own header, which names the pattern in order to
record that it was dropped.

### NOT checked — read this before trusting anything above

- **`score()` has never been executed.** No vLLM engine was started anywhere in
  this work (GPUs in use, and explicitly out of scope). Its correctness rests on
  reading vLLM 0.11.0's source, not on running it. Specifically verified by
  reading, not by execution: `prompt_logprobs` is a valid `SamplingParams` field
  (`sampling_params.py:162`); the return type is
  `list[Optional[dict[int, Logprob]]]` (`logprobs.py:26`) so the first element
  being `None` is the documented shape; `Logprob.logprob` is the float field
  (`logprobs.py:11-21`); and output order matches input order because
  `LLM.generate` sorts by request id (`entrypoints/llm.py:1629`). **The
  off-by-one in the token/logprob alignment is the obvious failure mode and it
  is untested.** Validate on a short string with a known model before using any
  perplexity number.
- **Batched `generate()` has never been executed.** The batching change is
  untested beyond its signature and its blank-input fast path.
- **`embed()` has never been run against MedCPT** or any real encoder, only a
  random tiny BERT. Shapes, caching and normalisation are verified; nothing
  about embedding *quality* or MedCPT's real 768-dim output is.
- **No MIMIC data was touched at all.** Every input in the smoke test is
  synthetic. `cv_auroc` has never seen a real code bag, and the design matrix
  has never been built at realistic vocabulary size.
- **`boot` was smoke-tested at B=200, not the default B=2000**, to keep the test
  fast. Nominal coverage of the percentile interval was never simulated — it is
  not bias-corrected (not BCa) and will be anti-conservative for small n.
- **`boot_continuous`: what was and was not checked** (2026-08-01,
  `scripts/smoke_phase2_fixes.py`, 500 synthetic values). Checked: bounds are
  bit-identical (1e-12) to a plain unguarded percentile bootstrap written
  independently in the test; reproducible under a fixed seed and different under
  a changed one; `alpha=0.10` is no wider than `alpha=0.05`; a custom `fn`
  (median) is honoured; empty input raises `ValueError`; and `boot` is confirmed
  *unchanged* by showing it still raises `RuntimeError` on single-class `y` where
  `boot_continuous` on the same continuous data returns a finite interval.
  **Not checked:** nominal coverage (never simulated, same as `boot`); behaviour
  at very small n; any non-mean statistic beyond the median spot-check; and the
  claim that it is bit-identical to the *old* dummy-label path on real data —
  that equivalence follows from the identical draw sequence and was argued, not
  re-measured against the purged run.
- GPU/fp16 paths in `embed` are unexercised; only the CPU/fp32 path ran.
- No test of `embed`'s checkpoint-on-crash behaviour (the 20-batch interval was
  never reached — the test runs 2 batches).
- `shutdown()` is untested.

### Failures encountered and fixed during this work

Recorded because they were real, not hypothetical:

1. `generate("")` did **not** hit the intended empty-input fast path. The guard
   was `if not plist`, but a single empty string becomes `[""]`, which is a
   non-empty list — so it fell through and **attempted to start a vLLM engine**,
   contrary to the no-vLLM constraint. It failed only incidentally, on
   `RuntimeError: Cannot re-initialize CUDA in forked subprocess`. Fixed by
   guarding on `not any(p and p.strip() ...)` in both `generate` and `score`;
   the smoke test now asserts blank calls leave `llm is None`. No GPU memory was
   allocated (the failure occurred during engine-core init, before weight
   loading) and no stray process survived.
2. Two smoke-test bugs of my own: a `NameError` on an unimported `List` (moot
   anyway — `llm.py` uses `from __future__ import annotations`, so annotations
   are strings), and the anti-label scan false-positiving on documentation, now
   AST-based.

## Known limitations

- **Truncation at 512 tokens is the big one.** MedCPT cannot see past 512
  tokens, and MIMIC discharge summaries are routinely far longer. `embed`
  silently keeps the first 512 tokens and discards the rest. For a study whose
  premise is that notes are the high-dimensional artifact, this is a material
  modelling decision inherited by default, not chosen. Chunking-and-pooling was
  not implemented.
- The embedding cache is keyed on text hash **only**, not on model identity or
  maxlen. Point two different models at one cache file and you will silently
  read the wrong vectors. Use a separate file per model.
- `features` discards all temporal structure — binary presence, flattened
  across visits. For a design where visits *are* time, callers will need
  per-visit bags rather than `_bag`'s flattened set.
- `cv_auroc` reports `n_feat_last_fold`, which is the final fold's vocabulary
  size, not an average. Inherited quirk, now at least named honestly.
- The logistic baseline is un-tuned (`C=1.0`, no inner CV), so "the baseline"
  is a floor, not a strong opponent.
- `star` is an interval-exclusion marker, not a test, and applies no
  multiple-comparison correction — with 4 modules × several cells, some stars
  will be noise.
- `boot` resamples units i.i.d. If the same patient contributes multiple visits,
  that assumption is violated and intervals will be too narrow; a cluster
  bootstrap over patients would be needed.
- `generate` defaults to `temperature=0.7` with `repetition_penalty=1.2`, so it
  is stochastic and its likelihoods are distorted. Do not infer perplexity from
  generation; use `score`.
- `score` must run as a **separate pass** from generation: setting
  `prompt_logprobs` makes vLLM skip prefix caching for the request
  (`vllm/v1/core/kv_cache_manager.py:167-172`), so interleaving the two costs
  the KV cache and gains nothing.
- The source repo's `run_condition_D.py:308` writes `1 - int(sample['label'])`
  as the prediction on a parse failure — the anti-label. It is deliberately not
  reproduced. If you vendor anything further from those runners, check for it;
  any metric computed over records containing those rows is wrong.
- `cvxpy` is deliberately **not** a dependency and is not installed. Convex
  donor-weight fitting must be solved some other way (projected gradient or
  scipy). Phase 2 uses `scipy.optimize.minimize(method='SLSQP')`.
- `design_matrix`'s vocabulary order is now sorted and stable, but the
  **document-frequency counts still depend on the caller passing deduplicated
  bags**. A caller that passes a list with repeats inflates `min_df` counts.
  `set`-valued bags (the documented input) are safe.
- `boot` is written for a *binary-label* statistic: it discards any resample
  that is single-class in `y`. For a continuous paired statistic (e.g. a
  per-patient RMSE difference) that guard is meaningless. Phase 2 used to work
  around it by passing an alternating 0/1 dummy as `y`; that hack is **gone**,
  replaced by `boot_continuous`. `boot` itself is untouched and is still the
  right function for the outcome AUROC.
- **`boot` and `boot_continuous` now coexist and are easy to confuse.** The rule
  is: binary-label statistic (`auc_mw`, `tjur`) → `boot(y, p, fn)`; continuous
  per-unit statistic (a per-patient RMSE, F1, or paired difference) →
  `boot_continuous(v, fn)`. Nothing in the code enforces it, and calling `boot`
  with a fabricated `y` will still "work" while quietly discarding resamples.
- Both are i.i.d. over units. If the same patient contributes multiple rows,
  neither is correct and a cluster bootstrap is needed; `boot_continuous`
  inherits this limitation unchanged.
