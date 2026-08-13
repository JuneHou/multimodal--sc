"""
vLLM wrapper: batched generation + per-token NLL scoring for a local Qwen model.

VENDORED CODE -- copied, not imported.
  source repo:  /data/wang/junh/githubs/moa-clinical-rag
  source path:  src/agents/run_condition_D.py
  source lines: 45-92 (class `QwenIntegrator`)
  vendored on:  2026-08-01

Why D is canonical: conditions D, E and F each carry a copy of this class. The
task brief called them byte-identical; they are NOT (verified by diff), but the
differences are entirely in comment and docstring text -- D's docstrings carry
"- EXACT copy from mortality_single_agent_cot.py" annotations that E and F
dropped, and the class begins on a different line in each file (D:45, E:43,
F:44). The executable code is identical across all three. D is used here.

Changes from the source:
  1. **ADDED `score()`** -- mean per-token negative log-likelihood per text.
     No equivalent exists anywhere in the source repo; this is new code, not
     vendored. See its docstring for the vLLM mechanics.
  2. **`generate()` now batches.** The original `__call__` took ONE prompt and
     issued a one-element `self.llm.generate([...])` per call, which serialises
     what vLLM would otherwise run as a continuous batch. `generate()` accepts
     either a single string (returns a single string, so old call sites keep
     working) or a list of strings (returns a list, batched into one
     `llm.generate` call). `__call__` is retained as a thin alias so the
     original single-prompt call convention still works verbatim.
  3. Model loading is lazy. The original built the `LLM` inside `__init__`, so
     merely constructing the object seized a GPU. Construction is now free and
     the engine is built on first use, or eagerly with `lazy=False`. This is
     what lets the smoke test import and introspect the class without a GPU.
  4. `SamplingParams` values (temperature, top_p, max_tokens, stop,
     repetition_penalty) were hardcoded inside the original's generate body;
     they are now constructor/method arguments defaulting to the original's
     values. `gpu_memory_utilization` (0.85), `tensor_parallel_size` (1),
     `trust_remote_code` (True) and `enforce_eager` (True) are likewise now
     arguments with the original's values as defaults.
  5. Prompt formatting (`<|im_start|>user ... <|im_end|>` when 'qwen' is in the
     model name, raw otherwise) is unchanged, factored into `_format`.
  6. The original's CUDA_VISIBLE_DEVICES side effect in `__init__` -- it
     mutated os.environ at construction time -- now happens at engine-build
     time and only when `gpu_id` is not None. The "don't clobber an externally
     set value" behaviour is preserved.

  DELIBERATELY NOT COPIED: run_condition_D.py:306-308 has a parse-failure
  fallback that writes `result['prediction'] = 1 - int(sample['label'])`, i.e.
  it stores the ANTI-LABEL as the model's prediction whenever parsing fails.
  That silently injects guaranteed-wrong rows into the output records and
  poisons any downstream metric. It is outside the vendored line range and is
  not reproduced here in any form. If a parse fails, record the failure.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union


class QwenIntegrator:
    """Thin wrapper over a local vLLM engine.

    Two independent passes, which must NOT be interleaved (see `score`):
      - `generate(prompts)` -- sampled text.
      - `score(texts)`      -- mean per-token NLL, for perplexity.
    """

    def __init__(
        self,
        model_name: str,
        gpu_id: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        enforce_eager: bool = True,
        lazy: bool = True,
        verbose: bool = True,
    ):
        """Configure the wrapper. Does not touch the GPU unless lazy=False.

        Args:
            model_name: HF id or local path of the model.
            gpu_id: value for CUDA_VISIBLE_DEVICES, applied at engine-build
                time and only if the variable is not already set externally.
                None leaves the environment alone.
            gpu_memory_utilization: vLLM KV-cache fraction. 0.85 is the
                source repo's value.
            tensor_parallel_size: GPUs per engine.
            trust_remote_code: required by some Qwen revisions.
            enforce_eager: True disables CUDA graph capture -- slower per
                token, but lower memory and faster startup. The source repo's
                value; kept so behaviour matches.
            lazy: if True the engine is built on first generate/score call.
            verbose: print engine setup messages.
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.verbose = verbose
        self.llm = None
        if not lazy:
            self._ensure_llm()

    # ------------------------------------------------------------------ setup

    def _ensure_llm(self):
        """Build the vLLM engine once, on first use."""
        if self.llm is not None:
            return self.llm
        from vllm import LLM

        if self.gpu_id is not None:
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
                if self.verbose:
                    print(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id}")
            elif self.verbose:
                print(
                    "CUDA_VISIBLE_DEVICES already set to: "
                    f"{os.environ['CUDA_VISIBLE_DEVICES']}"
                )

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
        )
        if self.verbose:
            print(f"VLLM initialized for {self.model_name}")
        return self.llm

    def _format(self, prompt: str) -> str:
        """Apply the Qwen chat template. Unchanged from the source."""
        if "qwen" in self.model_name.lower():
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    # ------------------------------------------------------------- generation

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        stop: Optional[List[str]] = None,
    ) -> Union[str, List[str]]:
        """Generate completions.

        Accepts a single prompt (returns a single string) or a list of prompts
        (returns a list of strings, one per prompt, in input order). Batching a
        list into one `llm.generate` call is the change from the source, which
        issued one call per prompt.

        Order is guaranteed: vLLM's LLM.generate sorts its outputs by integer
        request id before returning (vllm/entrypoints/llm.py:1629), so
        `out[i]` corresponds to `prompts[i]`.
        """
        from vllm import SamplingParams

        single = isinstance(prompts, str)
        plist: List[str] = [prompts] if single else list(prompts)
        # Fast path: no prompts, or nothing but blanks. Returns without
        # building the engine -- there is nothing for the model to condition
        # on, and an accidental blank call should not seize a GPU.
        if not plist or not any(p and p.strip() for p in plist):
            return "" if single else ["" for _ in plist]

        llm = self._ensure_llm()
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop if stop is not None else ["<|im_end|>", "</s>"],
            repetition_penalty=repetition_penalty,
        )
        outputs = llm.generate([self._format(p) for p in plist], sampling_params)
        texts = [o.outputs[0].text for o in outputs]
        return texts[0] if single else texts

    def __call__(self, prompt: str, max_tokens: int = 32768,
                 temperature: float = 0.7) -> str:
        """Single-prompt alias, preserving the source's calling convention."""
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # ---------------------------------------------------------------- scoring

    def score(self, texts: List[str]) -> List[float]:
        """Mean per-token negative log-likelihood for each text.

        `exp()` the result to get perplexity.

        NEW CODE -- no counterpart in the source repo.

        Mechanics: we ask vLLM for `prompt_logprobs=0`, which makes it return
        the logprob of each token that actually appears in the prompt (0 = no
        extra top-k alternatives beyond the real token). `max_tokens=1` and
        `temperature=0.0` because we want nothing sampled; the one generated
        token is discarded. Each output's `prompt_logprobs` is a list aligned
        to `prompt_token_ids` (vllm/logprobs.py:26,
        `list[Optional[dict[int, Logprob]]]`) whose FIRST element is None --
        there is no conditional logprob for the first token, nothing precedes
        it -- so we skip index 0. Each remaining element is a
        {token_id: Logprob} dict; we look up the token that was actually there.

        IMPORTANT -- run scoring as a SEPARATE PASS from generation. Setting
        `prompt_logprobs` disables prefix caching for the request: vLLM bails
        out of the prefix-cache lookup whenever `sampling_params.prompt_logprobs
        is not None` (vllm/v1/core/kv_cache_manager.py:167-172, verified in
        vllm 0.11.0). Mixing scored and generated requests in one engine pass
        therefore costs you the KV cache on the scoring requests and gains
        nothing. Batch all scoring together.

        Note the text is scored RAW -- the Qwen chat template from `_format` is
        deliberately NOT applied, because we want the likelihood of the text
        itself, not of the text wrapped in an instruction envelope.

        Args:
            texts: raw strings to score. Empty strings and single-token texts
                have no scoreable positions and yield nan.

        Returns:
            List of mean NLL in nats, one per input, in input order. nan where
            no token position was scoreable.
        """
        from vllm import SamplingParams

        texts = list(texts)
        # Fast path: nothing scoreable. An all-blank batch has no token
        # positions to score, so return nan without building the engine.
        if not texts:
            return []
        if not any(t and t.strip() for t in texts):
            return [float("nan") for _ in texts]

        llm = self._ensure_llm()
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=0,
        )
        outputs = llm.generate(list(texts), sampling_params)

        out: List[float] = []
        for o in outputs:
            plp = getattr(o, "prompt_logprobs", None)
            if not plp:
                out.append(float("nan"))
                continue
            tok_ids = list(o.prompt_token_ids or [])
            total, n = 0.0, 0
            # Index 0 is None (no logprob for the first token) -- skip it.
            for i, d in enumerate(plp):
                if i == 0 or not d:
                    continue
                lp = None
                if i < len(tok_ids) and tok_ids[i] in d:
                    lp = d[tok_ids[i]].logprob
                elif len(d) == 1:
                    # prompt_logprobs=0 gives exactly the real token; if the id
                    # lookup missed, the single entry is still the right one.
                    lp = next(iter(d.values())).logprob
                if lp is None:
                    continue
                total += lp
                n += 1
            out.append(-total / n if n else float("nan"))
        return out

    # ------------------------------------------------------------------ misc

    def shutdown(self):
        """Drop the engine and free its GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def config(self) -> Dict[str, Any]:
        """Effective engine settings, for provenance in result records."""
        return {
            "model_name": self.model_name,
            "gpu_id": self.gpu_id,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "loaded": self.llm is not None,
        }
