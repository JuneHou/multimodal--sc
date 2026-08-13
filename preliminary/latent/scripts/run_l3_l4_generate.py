"""L3 + L4 GENERATION stage -- everything that needs the vLLM engine.

Gated on `latent/results/score_gate_report.json` reporting `passed: true`; the
script refuses to run otherwise.

TWO ENGINE PROCESSES, and why
-----------------------------
`--stage generate` decodes; `--stage score` computes perplexity.  They are
separate processes because they need different engine memory settings:

  * generation runs at the vendored default `gpu_memory_utilization=0.85`;
  * scoring does NOT fit there.  `score()` sets `prompt_logprobs=0`, which
    makes vLLM materialise logits for EVERY prompt token in the scheduled
    chunk (8,192 tokens x 151,936 vocab), and the first attempt died with
    `torch.OutOfMemoryError: Tried to allocate 2.70 GiB ... 1.74 GiB is free`
    on a 46 GB A40 partway through scoring the 1,264 real records.  The
    scoring engine therefore runs at `gpu_memory_utilization=0.50`, leaving
    ~20 GB for those activations.  Nothing about what is computed changes --
    only the KV-cache budget of a prefill-only pass.

Both stages checkpoint to jsonl after every chunk, because the first run lost
20 minutes of completed generation to that crash.

What it does (temperature 0 throughout):
  1. L3 decode -- 250 Dirichlet + 250 spiky-0.9 K=5 mixtures (L2's own
     `draw_replicates`, seed 0), retrieval-conditioned decoding from the K
     donor texts + weights.
  2. L4 decode -- 50 (ADVANCED-CAD met, nearest not-met) pairs x
     alpha in {0,.25,.5,.75,1}, the same decode mechanism on 2 records.
  3. LLM judge -- fixed rubric, 1-10 coronary-severity rating of every L4
     record.
  4. Perplexity -- `score()` over all generated records, all 1,264 real
     records, and (secondary) the first-800-token prefix of each real record
     as a length-matched reference.  A separate pass from generation, as the
     vendored module requires.

Writes (the embedding-based columns are added by `run_l3_l4_analyse.py`):
    latent/results/l3_generation.parquet
    latent/results/l4_interpolation.parquet
    latent/results/_gen_stage_meta.json
    latent/results/_real_ppl.json

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_l3_l4_generate.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "src"), str(_LATENT / "scripts"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = _LATENT / "results"
GATE = RESULTS / "score_gate_report.json"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

GEN_CHUNK = 50          # prompts per llm.generate call (progress + checkpoint)
SCORE_CHUNK = 32        # texts per score() call
REAL_PREFIX_TOKENS = 800
GEN_UTIL = 0.85         # vendored default
SCORE_UTIL = 0.50       # see the header: prompt_logprobs activations

CK_L3 = RESULTS / "_ck_l3_text.jsonl"
CK_L4 = RESULTS / "_ck_l4_text.jsonl"
CK_JUDGE = RESULTS / "_ck_judge.jsonl"
CK_SCORE = RESULTS / "_ck_score.jsonl"
META = RESULTS / "_gen_stage_meta.json"


def pick_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(x) for x in ln.split(",")) for ln in out.strip().split("\n")]
    rows.sort(key=lambda r: (r[1], r[0]))
    return str(rows[0][0])


def _load_ck(path: Path) -> List[Any]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:      # torn final line
                break
    return out


def _append_ck(path: Path, items: List[Any]) -> None:
    with path.open("a") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ==========================================================================
# spec: the mixtures, the pairs, and every prompt.  Identical in both stages.
# ==========================================================================


def build_spec(n_l3: int, n_pairs: int) -> Dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    import genmix as gm

    tok = AutoTokenizer.from_pretrained(MODEL)
    C = gm.load_corpus(_LATENT)
    df, Z = C["df"], C["Z"]
    assert np.array_equal(df.index.to_numpy(), np.arange(len(df)))
    texts_real: List[str] = df["text"].tolist()

    mixes = gm.l3_mixtures(len(df), Z, n_per_family=n_l3, K=5, seed=0)
    l3_prompts, n_trunc_l3 = [], 0
    for m in mixes:
        p, nt = gm.decode_prompt([texts_real[i] for i in m["donor_rows"]],
                                 m["weights"], tok)
        l3_prompts.append(p)
        n_trunc_l3 += nt

    pairs = gm.l4_pairs(df, Z, n_pairs=n_pairs, seed=0)
    l4_rows, l4_prompts, n_trunc_l4 = [], [], 0
    for pr in pairs:
        for a in gm.ALPHAS:
            # fixed order: RECORD 1 = not-met, RECORD 2 = met; alpha weights met
            p, nt = gm.decode_prompt(
                [texts_real[pr["notmet_row"]], texts_real[pr["met_row"]]],
                [1.0 - a, a], tok)
            n_trunc_l4 += nt
            l4_prompts.append(p)
            l4_rows.append({**pr, "alpha": float(a),
                            "z_alpha": (1.0 - a) * Z[pr["notmet_row"]]
                            + a * Z[pr["met_row"]]})

    ptok = [len(x) for x in
            tok(l3_prompts + l4_prompts, add_special_tokens=False)["input_ids"]]
    return {"tok": tok, "df": df, "Z": Z, "texts_real": texts_real,
            "mixes": mixes, "l3_prompts": l3_prompts, "pairs": pairs,
            "l4_rows": l4_rows, "l4_prompts": l4_prompts,
            "prompt_tokens": ptok, "n_trunc_l3": n_trunc_l3,
            "n_trunc_l4": n_trunc_l4}


# ==========================================================================
# stage: generate
# ==========================================================================


def stage_generate(spec: Dict[str, Any], gpu: str) -> Dict[str, Any]:
    import numpy as np

    import genmix as gm
    from src.vendored.llm import QwenIntegrator

    llm = QwenIntegrator(MODEL, gpu_id=None, lazy=True,
                         gpu_memory_utilization=GEN_UTIL)

    def gen(prompts: List[str], ck: Path, tag: str, max_tokens: int) -> List[str]:
        done = _load_ck(ck)
        if len(done) >= len(prompts):
            print(f"[gen:{tag}] {len(done)} completions from checkpoint",
                  flush=True)
            return done[:len(prompts)]
        t0 = time.time()
        for i in range(len(done), len(prompts), GEN_CHUNK):
            out = llm.generate(prompts[i:i + GEN_CHUNK], max_tokens=max_tokens,
                               temperature=0.0, top_p=1.0,
                               repetition_penalty=1.0)
            _append_ck(ck, out)
            done.extend(out)
            print(f"[gen:{tag}] {len(done)}/{len(prompts)} "
                  f"{time.time()-t0:.0f}s", flush=True)
        return done

    timings: Dict[str, Any] = {}
    t0 = time.time()
    l3_text = gen(spec["l3_prompts"], CK_L3, "l3", gm.GEN_MAX_TOKENS)
    timings["l3_generate_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    l4_text = gen(spec["l4_prompts"], CK_L4, "l4", gm.GEN_MAX_TOKENS)
    timings["l4_generate_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    judge_raw = gen([gm.JUDGE_PROMPT.format(record=t) for t in l4_text],
                    CK_JUDGE, "judge", 8)
    timings["judge_seconds"] = round(time.time() - t0, 1)

    meta = {
        "stage": "L3/L4 generation",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": gpu, "model": MODEL,
        "vllm_version": __import__("vllm").__version__,
        "gate": {"path": str(GATE), "passed": True},
        "decoding": {"temperature": 0.0, "top_p": 1.0,
                     "repetition_penalty": 1.0,
                     "max_tokens": gm.GEN_MAX_TOKENS,
                     "gpu_memory_utilization": GEN_UTIL,
                     "chat_template": "vendored QwenIntegrator._format "
                                      "(<|im_start|>user ... assistant, no "
                                      "system message)"},
        "judge": {"max_tokens": 8, "temperature": 0.0,
                  "n_parse_failures": int(sum(
                      1 for x in judge_raw if gm.parse_judge(x) != gm.parse_judge(x))),
                  "prompt": gm.JUDGE_PROMPT},
        "decode_instruction": gm.DECODE_INSTRUCTION,
        "donor_max_tokens": gm.DONOR_MAX_TOKENS,
        "n_donor_blocks_truncated": {
            "l3": spec["n_trunc_l3"], "l4": spec["n_trunc_l4"],
            "l3_total_blocks": len(spec["mixes"]) * 5,
            "l4_total_blocks": len(spec["l4_rows"]) * 2},
        "prompt_tokens": {"mean": float(np.mean(spec["prompt_tokens"])),
                          "max": int(max(spec["prompt_tokens"]))},
        "completions": {"l3": len(l3_text), "l4": len(l4_text),
                        "judge": len(judge_raw),
                        "total": len(l3_text) + len(l4_text) + len(judge_raw)},
        "pairs": spec["pairs"],
        "timings": timings,
    }
    return meta


# ==========================================================================
# stage: score
# ==========================================================================


def stage_score(spec: Dict[str, Any], gpu: str) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    import genmix as gm
    from src.vendored.llm import QwenIntegrator

    tok = spec["tok"]
    l3_text = _load_ck(CK_L3)[:len(spec["l3_prompts"])]
    l4_text = _load_ck(CK_L4)[:len(spec["l4_prompts"])]
    judge_raw = _load_ck(CK_JUDGE)[:len(spec["l4_prompts"])]
    assert len(l3_text) == len(spec["l3_prompts"]), "generation incomplete"
    assert len(l4_text) == len(judge_raw) == len(spec["l4_prompts"])

    prefix = [tok.decode(tok(t, add_special_tokens=False)["input_ids"]
                         [:REAL_PREFIX_TOKENS]) for t in spec["texts_real"]]
    blocks = [("real", spec["texts_real"]), ("real_prefix", prefix),
              ("l3", l3_text), ("l4", l4_text)]
    todo: List[tuple] = [(tag, i, t) for tag, ts in blocks
                         for i, t in enumerate(ts)]

    llm = QwenIntegrator(MODEL, gpu_id=None, lazy=True,
                         gpu_memory_utilization=SCORE_UTIL)
    done = {(d["tag"], d["i"]): d["nll"] for d in _load_ck(CK_SCORE)}
    todo = [x for x in todo if (x[0], x[1]) not in done]
    print(f"[score] {len(done)} from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), SCORE_CHUNK):
        ck = todo[i:i + SCORE_CHUNK]
        vals = llm.score([x[2] for x in ck])
        recs = [{"tag": t, "i": j, "nll": float(v)}
                for (t, j, _), v in zip(ck, vals)]
        _append_ck(CK_SCORE, recs)
        for r in recs:
            done[(r["tag"], r["i"])] = r["nll"]
        if (i // SCORE_CHUNK) % 10 == 0:
            print(f"[score] {i + len(ck)}/{len(todo)} {time.time()-t0:.0f}s",
                  flush=True)
    score_seconds = round(time.time() - t0, 1)

    def col(tag: str, n: int) -> np.ndarray:
        return np.array([done[(tag, i)] for i in range(n)], dtype=float)

    nll_real = col("real", len(spec["texts_real"]))
    nll_real_pref = col("real_prefix", len(prefix))
    nll_l3 = col("l3", len(l3_text))
    nll_l4 = col("l4", len(l4_text))

    df = spec["df"]
    mixes, ptok = spec["mixes"], spec["prompt_tokens"]
    l3_df = pd.DataFrame([{
        "mixture_id": m["mixture_id"], "family": m["family"], "rep": m["rep"],
        "K": m["K"],
        "donor_rows": [int(x) for x in m["donor_rows"]],
        "donor_patients": [str(df.loc[int(x), "patient_id"])
                           for x in m["donor_rows"]],
        "weights": [float(x) for x in m["weights"]],
        "top_donor_row": m["top_donor_row"], "top_weight": m["top_weight"],
        "z_star": m["z_star"].astype(np.float32),
        "generated_text": g, "generated_chars": len(g),
        "generated_tokens": len(tok(g, add_special_tokens=False)["input_ids"]),
        "prompt_tokens": pt,
        "gen_nll": float(nv), "gen_ppl": float(np.exp(nv)),
        "baseline_row": m["top_donor_row"],
        "baseline_nll": float(nll_real[m["top_donor_row"]]),
        "baseline_ppl": float(np.exp(nll_real[m["top_donor_row"]])),
    } for m, g, nv, pt in zip(mixes, l3_text, nll_l3, ptok[:len(mixes)])])

    l4_df = pd.DataFrame([{
        "pair_id": r["pair_id"], "alpha": r["alpha"],
        "met_row": r["met_row"], "notmet_row": r["notmet_row"],
        "met_patient": r["met_patient"], "notmet_patient": r["notmet_patient"],
        "neighbour_cosine": r["neighbour_cosine"],
        "z_alpha": r["z_alpha"].astype(np.float32),
        "generated_text": g,
        "generated_tokens": len(tok(g, add_special_tokens=False)["input_ids"]),
        "prompt_tokens": pt,
        "gen_nll": float(nv), "gen_ppl": float(np.exp(nv)),
        "judge_raw": jr, "judge_score": gm.parse_judge(jr),
    } for r, g, nv, pt, jr in zip(spec["l4_rows"], l4_text, nll_l4,
                                  ptok[len(mixes):], judge_raw)])

    l3_df.to_parquet(RESULTS / "l3_generation.parquet", index=False)
    l4_df.to_parquet(RESULTS / "l4_interpolation.parquet", index=False)
    (RESULTS / "_real_ppl.json").write_text(json.dumps({
        "nll_real": [float(x) for x in nll_real],
        "nll_real_prefix_800tok": [float(x) for x in nll_real_pref]}))
    return {"score_calls": {"real": len(nll_real),
                            "real_prefix": len(nll_real_pref),
                            "generated": len(nll_l3) + len(nll_l4)},
            "scoring": {"gpu_memory_utilization": SCORE_UTIL,
                        "chunk": SCORE_CHUNK,
                        "seconds": score_seconds,
                        "separate_pass_from_generation": True},
            "real_prefix_tokens": REAL_PREFIX_TOKENS}


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "generate", "score"))
    ap.add_argument("--gpu", default="")
    ap.add_argument("--n-l3", type=int, default=250, help="per w* family")
    ap.add_argument("--n-pairs", type=int, default=50)
    args = ap.parse_args()

    gate = json.loads(GATE.read_text())
    if not gate.get("passed"):
        raise SystemExit("score() gate did not pass -- L3/L4 must not run")

    gpu = args.gpu or pick_gpu()
    if args.stage == "all":
        # each stage needs its own engine memory budget -> its own process
        for st in ("generate", "score"):
            r = subprocess.run([sys.executable, __file__, "--stage", st,
                                "--gpu", gpu, "--n-l3", str(args.n_l3),
                                "--n-pairs", str(args.n_pairs)])
            if r.returncode != 0:
                return r.returncode
        return 0

    t_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = build_spec(args.n_l3, args.n_pairs)
    print(f"[{args.stage}] gpu={gpu} l3={len(spec['l3_prompts'])} "
          f"l4={len(spec['l4_prompts'])} prompt tokens "
          f"mean={sum(spec['prompt_tokens'])/len(spec['prompt_tokens']):.0f} "
          f"max={max(spec['prompt_tokens'])}", flush=True)

    if args.stage == "generate":
        meta = stage_generate(spec, gpu)
        meta["wall_clock_seconds_generate"] = round(time.time() - t_start, 1)
        META.write_text(json.dumps(meta, indent=2))
    else:
        meta = json.loads(META.read_text()) if META.exists() else {}
        meta.update(stage_score(spec, gpu))
        meta["wall_clock_seconds_score"] = round(time.time() - t_start, 1)
        meta["wall_clock_seconds"] = round(
            meta.get("wall_clock_seconds_generate", 0.0)
            + meta["wall_clock_seconds_score"], 1)
        META.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("pairs", "decode_instruction", "judge")},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
