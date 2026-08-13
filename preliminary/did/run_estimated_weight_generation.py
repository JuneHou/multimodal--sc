"""Decode-from-ESTIMATED-weights experiment (POST-HOC, NOT pre-registered).

Jun's question (2026-08-12, satellite DiD-vs-SC discussion): H3 generated
notes only from the PLANTED ground-truth weights w*, and only at K=5.  What
happens when the decoder is fed the weights the pipeline would actually have
at inference time -- the solver's estimate w_hat from a realistic retrieval
pool -- across K in {2, 5, 10}?

Three arms per planted mixture, isolating one step each:

  wstar  : true donors, true planted weights          (upper anchor; H3 at K=5)
  what   : m=50 retrieved pool, solver-estimated w    (the full pipeline)
  equalK : top-K retrieved donors, equal 1/K weights  (the DiD analog: same
           retrieval, no weight estimation)

(wstar vs what) prices estimation error; (what vs equalK) prices weight
estimation given retrieval.  Mixtures reuse `draw_replicates` seed 0, so every
replicate is PAIRED with its H1 parquet row (same donors, same w*), and the
per-replicate L1(w_hat, w*) can be joined to generation quality.

Grid: K in {2,5,10} x {dirichlet, spiky} x N=25 per cell x 3 arms = 450
completions.  Prereg caps generation at ~1,500 total; H3/H4 used 1,000, so
the running total stays at 1,450.  sigma=0, qwen3 arm only.

Same engine discipline as run_l3_l4_generate.py: generate and score are
separate processes (0.85 vs 0.50 gpu_memory_utilization), both checkpoint to
jsonl after every chunk, and the script refuses to run unless the score()
gate passed.  Real-corpus perplexity references are reused from
latent/results/_real_ppl.json (not re-scored).

    /data/wang/junh/envs/medrag/bin/python preliminary/did/run_estimated_weight_generation.py
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

_DID = Path(__file__).resolve().parent
_PRELIM = _DID.parent
_LATENT = _PRELIM / "latent"
for p in (str(_LATENT / "src"), str(_LATENT / "scripts"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = _DID / "results"
GATE = _LATENT / "results" / "score_gate_report.json"
REAL_PPL = _LATENT / "results" / "_real_ppl.json"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

K_GRID = (2, 5, 10)
FAMILIES = ("dirichlet", "spiky")
ARMS = ("wstar", "what", "equalK")
POOL_M = 50            # H1's realistic retrieval setting
W_MIN = 0.01           # what-arm support threshold
MAX_PROMPT_DONORS = 10  # prompt-length control (DONOR_MAX_TOKENS budget)

GEN_CHUNK = 50
SCORE_CHUNK = 32
GEN_UTIL = 0.85
SCORE_UTIL = 0.50

CK_GEN = RESULTS / "_ck_gen_text.jsonl"
CK_SCORE = RESULTS / "_ck_score.jsonl"
META = RESULTS / "_stage_meta.json"


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
# spec: mixtures, solver fits, and every prompt.  Identical in both stages.
# ==========================================================================


def build_spec(n_per_cell: int) -> Dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    import genmix as gm
    import run_weight_recovery as wr

    tok = AutoTokenizer.from_pretrained(MODEL)
    C = gm.load_corpus(_LATENT)
    df, Z = C["df"], C["Z"]
    texts_real: List[str] = df["text"].tolist()
    n_corpus = len(df)

    rows: List[Dict[str, Any]] = []
    prompts: List[str] = []
    n_trunc = 0
    for K in K_GRID:
        for family in FAMILIES:
            fam_code = wr.WSTAR_GRID.index(family)
            D, W = wr.draw_replicates(n_corpus, K, family, n_per_cell, seed=0)
            for rep in range(n_per_cell):
                donors = D[rep].astype(int)
                w_star = W[rep].astype(float)
                z_star = w_star @ Z[donors]

                # H1's m=50 retrieval pool + solver fit (sigma = 0)
                sims = Z @ z_star
                pool = np.argsort(-sims, kind="stable")[:POOL_M]
                sol = wr.solve_simplex_qp(Z[pool], z_star)
                w_hat = sol.w
                full_hat = np.zeros(n_corpus)
                full_hat[pool] = w_hat
                full_star = np.zeros(n_corpus)
                full_star[donors] = w_star
                l1 = float(np.abs(full_hat - full_star).sum())
                n_true_in_pool = int(np.isin(donors, pool).sum())

                # what arm: support of w_hat, largest first, capped, renormed
                order = np.argsort(-w_hat, kind="stable")
                keep = [i for i in order if w_hat[i] > W_MIN][:MAX_PROMPT_DONORS]
                if not keep:                     # degenerate: keep the top donor
                    keep = [int(order[0])]
                hat_donors = pool[keep]
                hat_w = w_hat[keep] / w_hat[keep].sum()

                arm_spec = {
                    "wstar": (donors, w_star),
                    "what": (hat_donors, hat_w),
                    "equalK": (pool[:K], np.full(K, 1.0 / K)),
                }
                for arm_code, arm in enumerate(ARMS):
                    a_donors, a_w = arm_spec[arm]
                    # donor position independent of weight, as in L3's draw
                    # order convention (H4 found a position bias)
                    perm = np.random.default_rng(
                        [7, K, fam_code, rep, arm_code]
                    ).permutation(len(a_donors))
                    p_donors = np.asarray(a_donors)[perm]
                    p_w = np.asarray(a_w, dtype=float)[perm]
                    prompt, nt = gm.decode_prompt(
                        [texts_real[i] for i in p_donors], p_w, tok)
                    n_trunc += nt
                    prompts.append(prompt)
                    rows.append({
                        "K": int(K), "family": family, "rep": int(rep),
                        "arm": arm,
                        "true_donor_rows": [int(x) for x in donors],
                        "true_weights": [float(x) for x in w_star],
                        "prompt_donor_rows": [int(x) for x in p_donors],
                        "prompt_weights": [float(x) for x in p_w],
                        "n_prompt_donors": int(len(p_donors)),
                        "l1_what": l1,
                        "n_true_in_pool": n_true_in_pool,
                        "w_hat_max": float(w_hat.max()),
                        "what_support_size": int((w_hat > W_MIN).sum()),
                        "solver_sse": float(sol.sse),
                        "solver_status": int(sol.status),
                    })

    ptok = [len(x) for x in
            tok(prompts, add_special_tokens=False)["input_ids"]]
    return {"tok": tok, "df": df, "texts_real": texts_real, "rows": rows,
            "prompts": prompts, "prompt_tokens": ptok, "n_trunc": n_trunc}


# ==========================================================================
# stage: generate
# ==========================================================================


def stage_generate(spec: Dict[str, Any], gpu: str) -> Dict[str, Any]:
    import numpy as np

    import genmix as gm
    from src.vendored.llm import QwenIntegrator

    llm = QwenIntegrator(MODEL, gpu_id=None, lazy=True,
                         gpu_memory_utilization=GEN_UTIL)
    prompts = spec["prompts"]
    done = _load_ck(CK_GEN)
    t0 = time.time()
    for i in range(len(done), len(prompts), GEN_CHUNK):
        out = llm.generate(prompts[i:i + GEN_CHUNK],
                           max_tokens=gm.GEN_MAX_TOKENS,
                           temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        _append_ck(CK_GEN, out)
        done.extend(out)
        print(f"[gen] {len(done)}/{len(prompts)} {time.time()-t0:.0f}s",
              flush=True)

    return {
        "stage": "estimated-weight generation (post-hoc)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": gpu, "model": MODEL,
        "vllm_version": __import__("vllm").__version__,
        "gate": {"path": str(GATE), "passed": True},
        "decoding": {"temperature": 0.0, "top_p": 1.0,
                     "repetition_penalty": 1.0,
                     "max_tokens": gm.GEN_MAX_TOKENS,
                     "gpu_memory_utilization": GEN_UTIL},
        "grid": {"K": list(K_GRID), "families": list(FAMILIES),
                 "arms": list(ARMS), "pool_m": POOL_M, "sigma": 0.0,
                 "w_min": W_MIN, "max_prompt_donors": MAX_PROMPT_DONORS},
        "completions": {"total": len(done)},
        "n_donor_blocks_truncated": spec["n_trunc"],
        "prompt_tokens": {"mean": float(np.mean(spec["prompt_tokens"])),
                          "max": int(max(spec["prompt_tokens"]))},
        "l3_l4_completions_prior": 1000,
        "prereg_cap_note": "running total 1450 of the ~1500 generation cap",
    }


# ==========================================================================
# stage: score  (+ copy diagnostic + summary; all CPU work lives here too)
# ==========================================================================


def stage_score(spec: Dict[str, Any], gpu: str) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    from run_copy_diagnostic import frac, shingles
    from src.vendored.llm import QwenIntegrator

    texts = _load_ck(CK_GEN)
    assert len(texts) == len(spec["prompts"]), "generation incomplete"

    llm = QwenIntegrator(MODEL, gpu_id=None, lazy=True,
                         gpu_memory_utilization=SCORE_UTIL)
    done = {d["i"]: d["nll"] for d in _load_ck(CK_SCORE)}
    todo = [i for i in range(len(texts)) if i not in done]
    print(f"[score] {len(done)} from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), SCORE_CHUNK):
        idx = todo[i:i + SCORE_CHUNK]
        vals = llm.score([texts[j] for j in idx])
        recs = [{"i": j, "nll": float(v)} for j, v in zip(idx, vals)]
        _append_ck(CK_SCORE, recs)
        done.update({r["i"]: r["nll"] for r in recs})
        print(f"[score] {i + len(idx)}/{len(todo)} {time.time()-t0:.0f}s",
              flush=True)

    nll_real = np.asarray(json.loads(REAL_PPL.read_text())["nll_real"], float)
    real_median_ppl = float(np.median(np.exp(nll_real)))

    texts_real = spec["texts_real"]
    recs = []
    for i, (row, g) in enumerate(zip(spec["rows"], texts)):
        gsh = shingles(g)
        donor_sh = [shingles(texts_real[d]) for d in row["prompt_donor_rows"]]
        pooled = set().union(*donor_sh) if donor_sh else set()
        top_j = int(np.argmax(row["prompt_weights"]))
        nll = done[i]
        recs.append({**row,
                     "generated_text": g,
                     "gen_nll": float(nll),
                     "gen_ppl": float(np.exp(nll)),
                     "copy_pooled": frac(gsh, pooled),
                     "copy_top_weight_donor": frac(gsh, donor_sh[top_j])})
    out = pd.DataFrame(recs)
    out.to_parquet(RESULTS / "estimated_weight_generation.parquet", index=False)

    cells = (out.groupby(["arm", "K", "family"])
             .agg(n=("gen_ppl", "size"),
                  median_ppl=("gen_ppl", "median"),
                  median_copy_pooled=("copy_pooled", "median"),
                  median_copy_top=("copy_top_weight_donor", "median"),
                  mean_l1_what=("l1_what", "mean"),
                  mean_true_in_pool=("n_true_in_pool", "mean"))
             .reset_index())
    cells["ppl_ratio_vs_real"] = cells["median_ppl"] / real_median_ppl

    what = out[out["arm"] == "what"]
    rho_ppl = spearmanr(what["l1_what"], what["gen_ppl"])
    rho_copy = spearmanr(what["l1_what"], what["copy_pooled"])

    summary = {
        "post_hoc": True,
        "real_median_ppl": real_median_ppl,
        "cells": cells.to_dict(orient="records"),
        "arm_medians_ppl": {a: float(out.loc[out["arm"] == a, "gen_ppl"]
                                     .median()) for a in ARMS},
        "spearman_l1_vs_gen_ppl_what": {"rho": float(rho_ppl.statistic),
                                        "p": float(rho_ppl.pvalue)},
        "spearman_l1_vs_copy_pooled_what": {"rho": float(rho_copy.statistic),
                                            "p": float(rho_copy.pvalue)},
    }

    # anchor 1: wstar K=5 cells vs H3's l3_generation (same stream, first reps)
    l3_path = _LATENT / "results" / "l3_generation.parquet"
    if l3_path.exists():
        l3 = pd.read_parquet(l3_path, columns=["family", "rep", "gen_ppl"])
        n_rep = int(out["rep"].max()) + 1
        l3 = l3[l3["rep"] < n_rep]
        ours = out[(out["arm"] == "wstar") & (out["K"] == 5)]
        summary["anchor_wstar_k5_vs_l3"] = {
            fam: {"ours_median_ppl": float(
                      ours.loc[ours["family"] == fam, "gen_ppl"].median()),
                  "l3_median_ppl": float(
                      l3.loc[l3["family"] == fam, "gen_ppl"].median())}
            for fam in FAMILIES}

    # anchor 2: solver refits vs the H1 parquet cell means (paired stream)
    h1_path = _LATENT / "results" / "n2c2_weight_recovery.parquet"
    if h1_path.exists():
        h1 = pd.read_parquet(h1_path)
        h1 = h1[(h1["arm"] == "qwen3") & (h1["pool"] == "m50")
                & (h1["sigma"] == 0.0)]
        n_rep = int(out["rep"].max()) + 1
        ours_l1 = (out[out["arm"] == "what"]
                   .groupby(["K", "family"])["l1_what"].mean())
        summary["anchor_l1_vs_h1_m50_sigma0"] = {
            f"K{K}_{fam}": {
                "ours_mean_l1_first_reps": float(ours_l1.loc[(K, fam)]),
                "h1_mean_l1_matched_reps": float(
                    h1[(h1["K"] == K) & (h1["wstar"] == fam)
                       & (h1["rep"] < n_rep)]["l1"].mean())}
            for K in K_GRID for fam in FAMILIES}

    (RESULTS / "estimated_weight_generation_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"},
                     indent=2), flush=True)
    return {"score_calls": len(texts), "summary_written": True}


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "generate", "score"))
    ap.add_argument("--gpu", default="")
    ap.add_argument("--n-per-cell", type=int, default=25)
    args = ap.parse_args()

    gate = json.loads(GATE.read_text())
    if not gate.get("passed"):
        raise SystemExit("score() gate did not pass -- must not run")

    gpu = args.gpu or pick_gpu()
    if args.stage == "all":
        for st in ("generate", "score"):
            r = subprocess.run([sys.executable, __file__, "--stage", st,
                                "--gpu", gpu,
                                "--n-per-cell", str(args.n_per_cell)])
            if r.returncode != 0:
                return r.returncode
        return 0

    t_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = build_spec(args.n_per_cell)
    print(f"[{args.stage}] gpu={gpu} prompts={len(spec['prompts'])} "
          f"tokens mean={sum(spec['prompt_tokens'])/len(spec['prompt_tokens']):.0f} "
          f"max={max(spec['prompt_tokens'])}", flush=True)

    if args.stage == "generate":
        meta = stage_generate(spec, gpu)
        meta["wall_clock_seconds_generate"] = round(time.time() - t_start, 1)
        META.write_text(json.dumps(meta, indent=2))
    else:
        meta = json.loads(META.read_text()) if META.exists() else {}
        meta.update(stage_score(spec, gpu))
        meta["wall_clock_seconds_score"] = round(time.time() - t_start, 1)
        META.write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
