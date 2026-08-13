"""HARD GATE -- validate `mimic/src/vendored/llm.py::QwenIntegrator.score()`.

`score()` had never been executed (vendored/MODULE.md: "The off-by-one in the
token/logprob alignment is the obvious failure mode and it is untested").  No
L3/L4 perplexity number may be quoted until this script exits 0.

Checks, all on Qwen/Qwen2.5-7B-Instruct
---------------------------------------
1.  ORDERING.  Three texts of known likelihood ordering:
      (a) a fluent English clinical sentence,
      (b) the same words randomly shuffled (`random.Random(0)`),
      (c) a matched-length string of random vocabulary tokens
          (`np.random.default_rng(0)`, special/added ids excluded).
    Requires strictly increasing mean NLL a < b < c with a margin.
2.  MAGNITUDE.  Fluent English prose must land in ~1-4 nats/token.
3.  DETERMINISM.  A repeat `score()` call reproduces (a)'s NLL to 1e-9.
4.  BATCHING.  A two-element batch returns per-text values equal to the
    single-text calls (1e-6).
5.  INDEPENDENT REFERENCE -- the check that actually catches an off-by-one.
    The same mean NLL is recomputed with plain `transformers` on a SECOND GPU
    (`--mode hf-ref`, run first, in its own process so no CUDA context is
    created in the vLLM process), using the same tokenisation, and compared.
    The reference also computes two DELIBERATELY MISALIGNED variants (logits
    shifted the wrong way, and the alignment that skips the last token instead
    of the first); the gate additionally requires that vLLM's number is closer
    to the correct reference than to either misaligned one, so the comparison
    is shown to have the power to detect the failure mode it is testing.
6.  TOKENISATION.  vLLM's `prompt_token_ids` are compared to the HF
    tokeniser's ids for the same string.

Output: latent/results/score_gate_report.json  (written whether it passes or
fails), exit 0 iff every check passes.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_score_gate.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
sys.path.insert(0, str(_PRELIM / "mimic"))

RESULTS = _LATENT / "results"
REPORT = RESULTS / "score_gate_report.json"
REF_JSON = RESULTS / "_score_gate_hf_ref.json"

MODEL = "Qwen/Qwen2.5-7B-Instruct"

#: (a) fluent English clinical prose.  Written for this test -- not a corpus
#: record, so the gate report carries no n2c2 text.
FLUENT = (
    "The patient is a 62-year-old man with a history of hypertension and "
    "type 2 diabetes who presents today for routine follow-up. He reports "
    "no chest pain, shortness of breath, or palpitations since his last "
    "visit. Blood pressure is well controlled on lisinopril, and his "
    "hemoglobin A1c has improved to 7.1 percent. We will continue the "
    "current regimen and recheck laboratory studies in three months."
)

MIN_MARGIN = 0.20      # nats/token, required gap between a<b and b<c
PPL_LO, PPL_HI = 1.0, 4.0
REF_TOL = 0.05         # nats/token, vLLM vs transformers on the same text


def pick_gpus(n: int) -> List[str]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(x) for x in ln.split(",")) for ln in out.strip().split("\n")]
    rows.sort(key=lambda r: (r[1], r[0]))
    return [str(r[0]) for r in rows[:n]]


def build_texts() -> Dict[str, Any]:
    """The three gate strings + their token ids.  Identical in both modes."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids_a = tok(FLUENT, add_special_tokens=True)["input_ids"]

    words = FLUENT.split()
    rnd = random.Random(0)
    rnd.shuffle(words)
    shuffled = " ".join(words)

    # matched-length random vocabulary tokens: sample ids uniformly from the
    # base vocabulary, excluding every special / added token.
    import numpy as np

    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
    vocab_n = int(tok.vocab_size)
    rng = np.random.default_rng(0)
    picked: List[int] = []
    while len(picked) < len(ids_a):
        cand = int(rng.integers(0, vocab_n))
        if cand in special:
            continue
        picked.append(cand)
    random_tokens = tok.decode(picked)

    texts = {"a_fluent": FLUENT, "b_shuffled": shuffled, "c_random": random_tokens}
    ids = {k: tok(v, add_special_tokens=True)["input_ids"] for k, v in texts.items()}
    return {"texts": texts, "ids": ids,
            "n_tokens": {k: len(v) for k, v in ids.items()}}


# ---------------------------------------------------------------- hf reference


def run_hf_ref(gpu: str) -> int:
    """Mean NLL by plain transformers, plus two misaligned variants."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from transformers import AutoModelForCausalLM

    built = build_texts()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=None).to("cuda").eval()

    out: Dict[str, Any] = {"gpu": gpu, "model": MODEL,
                           "dtype": "bfloat16",
                           "ids": built["ids"], "n_tokens": built["n_tokens"],
                           "correct": {}, "misaligned_shift_wrong_way": {},
                           "misaligned_skip_last": {}}
    for key, ids in built["ids"].items():
        t = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            logits = model(t).logits.float()
        logp = torch.log_softmax(logits, dim=-1)[0]           # (n, V)
        n = len(ids)
        tgt = torch.tensor(ids, device="cuda")
        # CORRECT: logits at position i predict token i+1.  Mean over the
        # n-1 scoreable positions -- the same set `score()` averages over.
        lp = logp[:-1].gather(1, tgt[1:, None]).squeeze(1)
        out["correct"][key] = float(-lp.mean())
        # MISALIGNED #1: logits at position i scored against token i (a model
        # cannot see its own token; this is the classic off-by-one).
        lp1 = logp[1:].gather(1, tgt[1:, None]).squeeze(1)
        out["misaligned_shift_wrong_way"][key] = float(-lp1.mean())
        # MISALIGNED #2: skip the LAST token instead of the first, i.e. treat
        # prompt_logprobs as if its final element were the None one.
        lp2 = logp[:-1].gather(1, tgt[:-1, None]).squeeze(1)
        out["misaligned_skip_last"][key] = float(-lp2.mean())
        out.setdefault("n_scored_positions", {})[key] = int(n - 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    REF_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "ids"}, indent=2))
    return 0


# ------------------------------------------------------------------ main gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gate", choices=("gate", "hf-ref"))
    ap.add_argument("--gpu", default="")
    ap.add_argument("--ref-gpu", default="")
    args = ap.parse_args()

    if args.mode == "hf-ref":
        return run_hf_ref(args.gpu or pick_gpus(1)[0])

    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    gpus = pick_gpus(2)
    gpu = args.gpu or gpus[0]
    ref_gpu = args.ref_gpu or (gpus[1] if gpus[1] != gpu else gpus[0])

    # -- reference first, in its own process (no CUDA context here yet) ------
    print(f"[gate] hf reference on GPU {ref_gpu} ...", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, __file__, "--mode", "hf-ref",
                        "--gpu", ref_gpu], cwd=str(_LATENT))
    if r.returncode != 0:
        raise SystemExit(f"hf reference failed with code {r.returncode}")
    ref = json.loads(REF_JSON.read_text())
    ref_seconds = round(time.time() - t0, 1)

    built = build_texts()
    texts = built["texts"]
    keys = ["a_fluent", "b_shuffled", "c_random"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from src.vendored.llm import QwenIntegrator

    llm = QwenIntegrator(MODEL, gpu_id=None, lazy=True)
    t0 = time.time()
    nll = llm.score([texts[k] for k in keys])
    load_and_score_seconds = round(time.time() - t0, 1)
    nll = {k: float(v) for k, v in zip(keys, nll)}

    # 3. determinism: repeat call on (a) alone
    nll_a_repeat = float(llm.score([texts["a_fluent"]])[0])
    # 4. batching: two-element batch vs the single calls
    batch2 = [float(x) for x in llm.score([texts["a_fluent"], texts["c_random"]])]
    single_c = float(llm.score([texts["c_random"]])[0])

    # 6. tokenisation: vLLM's own prompt_token_ids for text (a)
    from vllm import SamplingParams
    vout = llm.llm.generate([texts["a_fluent"]],
                            SamplingParams(temperature=0.0, max_tokens=1,
                                           prompt_logprobs=0))
    vllm_ids = list(vout[0].prompt_token_ids)
    plp = vout[0].prompt_logprobs
    plp_len = len(plp)
    plp_first_is_none = plp[0] is None

    checks: Dict[str, Any] = {}
    checks["ordering"] = {
        "nll": nll,
        "perplexity": {k: float(__import__("math").exp(v)) for k, v in nll.items()},
        "margin_b_minus_a": nll["b_shuffled"] - nll["a_fluent"],
        "margin_c_minus_b": nll["c_random"] - nll["b_shuffled"],
        "required_margin": MIN_MARGIN,
        "passed": bool(nll["a_fluent"] + MIN_MARGIN < nll["b_shuffled"]
                       and nll["b_shuffled"] + MIN_MARGIN < nll["c_random"]),
    }
    checks["magnitude"] = {
        "fluent_nll_nats_per_token": nll["a_fluent"],
        "band": [PPL_LO, PPL_HI],
        "passed": bool(PPL_LO <= nll["a_fluent"] <= PPL_HI),
    }
    checks["determinism"] = {
        "first": nll["a_fluent"], "repeat": nll_a_repeat,
        "abs_diff": abs(nll_a_repeat - nll["a_fluent"]),
        "tolerance": 1e-9,
        "passed": bool(abs(nll_a_repeat - nll["a_fluent"]) <= 1e-9),
    }
    checks["batching"] = {
        "batch2_a": batch2[0], "single_a": nll["a_fluent"],
        "batch2_c": batch2[1], "single_c": single_c,
        "max_abs_diff": max(abs(batch2[0] - nll["a_fluent"]),
                            abs(batch2[1] - single_c)),
        "tolerance": 1e-6,
        "passed": bool(max(abs(batch2[0] - nll["a_fluent"]),
                           abs(batch2[1] - single_c)) <= 1e-6),
    }
    diffs = {k: abs(nll[k] - ref["correct"][k]) for k in keys}
    d_bad1 = {k: abs(nll[k] - ref["misaligned_shift_wrong_way"][k]) for k in keys}
    d_bad2 = {k: abs(nll[k] - ref["misaligned_skip_last"][k]) for k in keys}
    checks["independent_reference"] = {
        "vllm_score": nll,
        "transformers_correct_alignment": ref["correct"],
        "abs_diff": diffs,
        "max_abs_diff": max(diffs.values()),
        "tolerance": REF_TOL,
        "transformers_misaligned_shift_wrong_way": ref["misaligned_shift_wrong_way"],
        "transformers_misaligned_skip_last": ref["misaligned_skip_last"],
        "abs_diff_vs_misaligned_shift_wrong_way": d_bad1,
        "abs_diff_vs_misaligned_skip_last": d_bad2,
        "closer_to_correct_than_to_either_misalignment": bool(
            all(diffs[k] < d_bad1[k] and diffs[k] < d_bad2[k] for k in keys)),
        "ref_gpu": ref["gpu"], "ref_seconds": ref_seconds,
        "passed": bool(max(diffs.values()) <= REF_TOL
                       and all(diffs[k] < d_bad1[k] and diffs[k] < d_bad2[k]
                               for k in keys)),
    }
    checks["tokenisation"] = {
        "n_tokens_hf": built["n_tokens"]["a_fluent"],
        "n_tokens_vllm": len(vllm_ids),
        "ids_identical": bool(vllm_ids == list(built["ids"]["a_fluent"])),
        "prompt_logprobs_len": plp_len,
        "prompt_logprobs_first_is_none": bool(plp_first_is_none),
        "n_scored_positions_expected": len(vllm_ids) - 1,
        "passed": bool(vllm_ids == list(built["ids"]["a_fluent"])
                       and plp_len == len(vllm_ids) and plp_first_is_none),
    }

    passed = all(c["passed"] for c in checks.values())
    report = {
        "stage": "L3/L4 hard gate: QwenIntegrator.score()",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "gpu_vllm": gpu, "gpu_hf_reference": ref_gpu,
        "vllm_version": __import__("vllm").__version__,
        "engine_config": llm.config(),
        "texts": {k: (v[:300] + ("..." if len(v) > 300 else ""))
                  for k, v in texts.items()},
        "n_tokens": built["n_tokens"],
        "checks": checks,
        "passed": bool(passed),
        "load_and_first_score_seconds": load_and_score_seconds,
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "patch_applied_to_vendored_llm_py": None,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[gate] {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
