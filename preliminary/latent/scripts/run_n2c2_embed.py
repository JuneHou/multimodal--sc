"""
L1 runner: embed `latent/data/n2c2_corpus.parquet` with both pre-registered arms.

Writes
    latent/data/states_n2c2_qwen3_4096.npz   native Qwen3-Embedding-8B output
    latent/data/states_n2c2_qwen3.npz        MRL-truncated 768-d  (PRIMARY)
    latent/data/states_n2c2_biolord.npz      BioLORD-2023 768-d (chunk+mean-pool)
    latent/results/n2c2_l1_report.json       every pre-registered check

Difference from `run_embed.py` (the invalid NCT00174655 run): records reach
2,985 tokens, so the BioLORD arm goes through `encode.encode_texts_chunked`
(<=510-token chunks on sentence/whitespace boundaries, mean-pooled,
re-normalised) as the addendum pre-registers.  Qwen3's 32k window needs none.
The neighbour smell test now prints real-prose excerpts, because this is the
first run whose inputs are prose at all.

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_n2c2_embed.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_LATENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LATENT / "src"))


def _pick_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(x) for x in ln.split(",")) for ln in out.strip().split("\n")]
    rows.sort(key=lambda r: (r[1], r[0]))
    return str(rows[0][0])


GPU = os.environ.get("CUDA_VISIBLE_DEVICES") or _pick_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import encode as enc  # noqa: E402

IN_PARQUET = _LATENT / "data" / "n2c2_corpus.parquet"
OUT = {
    "qwen3_4096": _LATENT / "data" / "states_n2c2_qwen3_4096.npz",
    "qwen3": _LATENT / "data" / "states_n2c2_qwen3.npz",
    "biolord": _LATENT / "data" / "states_n2c2_biolord.npz",
}
REPORT = _LATENT / "results" / "n2c2_l1_report.json"

N_SMELL_PATIENTS = 5
SMELL_SEED = 0
TOPK = 3
EXCERPT = 150
BATCH = {"qwen3": 8, "biolord": 128}

_STOP = set("""the a an and or of to in for with without on at by from as is are was were be been
this that these those his her their our your not no none had has have will would can could may
patient patients dr mr mrs pt he she they we you i it its there here also then than about into
out up down over under after before during per each any all both other same such only own more
most some few many much very just now new old year years old day days week weeks month months
record date am pm""".split())
_WORD = re.compile(r"[a-z]{4,}")


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _content_words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _excerpt(text: str, n: int = EXCERPT) -> str:
    """`n` chars of the record's first prose, whitespace-collapsed.

    The first ~10 lines of an n2c2 record are the date header, the clinic
    address and the patient/physician block -- de-identification surrogates that
    say nothing about clinical content.  The excerpt therefore starts at the
    first line with >= 10 words, which is where narrative begins.
    """
    lines = text.split("\n")
    start = 0
    for i, ln in enumerate(lines):
        if len(ln.split()) >= 10:
            start = i
            break
    body = re.sub(r"\s+", " ", "\n".join(lines[start:])).strip()
    return (body or re.sub(r"\s+", " ", text).strip())[:n]


def _head(text: str, n: int = 100) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-states", action="store_true",
                    help="skip encoding: reuse the saved .npz states and the "
                         "encoder section of the existing report, recompute "
                         "only the smell test / overlap / MRL sections")
    args = ap.parse_args()

    t_start = time.time()
    df = pd.read_parquet(IN_PARQUET)
    texts: List[str] = df["text"].tolist()
    pid = df["patient_idx"].to_numpy(dtype=np.int32)
    tt = df["t"].to_numpy(dtype=np.int32)
    n = len(texts)

    report: Dict[str, Any] = {
        "stage": "L1 (n2c2 rerun)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": n,
        "n_patients": int(df["patient_id"].nunique()),
        "input_parquet": str(IN_PARQUET),
        "arms": {},
    }

    raw: Dict[str, np.ndarray] = {}
    if args.from_states:
        prev = json.loads(REPORT.read_text())
        report["arms"] = prev["arms"]
        report["encoding_reused_from"] = {"report": str(REPORT),
                                          "generated": prev["generated"]}
        X_q4096 = enc.load_states(OUT["qwen3_4096"])["X"]
        raw["biolord"] = enc.load_states(OUT["biolord"])["X"]
        s = enc.load_states(OUT["qwen3"])
        assert np.array_equal(s["patient_idx"], pid) and np.array_equal(s["t"], tt)
        X_q768 = s["X"]
        report["outputs"] = {k: str(v) for k, v in OUT.items()}
        return _analyse(report, df, texts, pid, tt, n, X_q4096, X_q768,
                        raw["biolord"], t_start)

    import torch

    report["gpu_visible"] = GPU
    report["gpu_name"] = torch.cuda.get_device_name(0)

    for arm in ("qwen3", "biolord"):
        t0 = time.time()
        model = enc.load_encoder(arm, device="cuda")
        tok = model.tokenizer
        lens = np.array([len(x) for x in
                         tok(texts, add_special_tokens=True)["input_ids"]])
        max_seq = int(model.max_seq_length)
        n_over = int((lens > max_seq).sum())

        info: Dict[str, Any] = {
            "model": enc.QWEN3_REPO if arm == "qwen3" else enc.BIOLORD_REPO,
            "dtype": str(next(model.parameters()).dtype),
            "max_seq_length_used": max_seq,
            "batch_size": BATCH[arm],
            "token_length": {"max": int(lens.max()), "mean": float(lens.mean()),
                             "p95": float(np.percentile(lens, 95)),
                             "n_over_max_seq_length": n_over},
        }

        if arm == "qwen3":
            assert n_over == 0, (
                f"qwen3: {n_over} texts exceed max_seq_length={max_seq}")
            Xn = enc.l2_normalise(enc.encode_texts(model, texts,
                                                   batch_size=BATCH[arm]))
            info["chunking"] = "none (32k window covers the longest record)"
        else:
            # ---- pre-registered chunk + mean-pool ------------------------
            Xn, counts = enc.encode_texts_chunked(
                model, texts, max_tokens=enc.CHUNK_MAX_TOKENS,
                batch_size=BATCH[arm])
            # single-chunk texts must be bit-identical to direct encoding:
            # same input list -> same batching -> same bytes.
            single = [i for i in range(n) if counts[i] == 1]
            rng_s = np.random.default_rng(SMELL_SEED)
            samp = sorted(rng_s.choice(single, size=min(64, len(single)),
                                       replace=False).tolist())
            s_txt = [texts[i] for i in samp]
            a_direct = enc.l2_normalise(enc.encode_texts(
                model, s_txt, batch_size=BATCH[arm], show_progress=False))
            a_chunk, c2 = enc.encode_texts_chunked(
                model, s_txt, max_tokens=enc.CHUNK_MAX_TOKENS,
                batch_size=BATCH[arm], show_progress=False)
            assert int(c2.max()) == 1
            identical = _sha(a_direct) == _sha(a_chunk)
            assert identical, (
                "single-chunk rows differ from direct encoding: max abs diff "
                f"{float(np.abs(a_direct - a_chunk).max())}")
            info["chunking"] = {
                "max_tokens_per_chunk": enc.CHUNK_MAX_TOKENS,
                "boundaries": "sentence, then whitespace if a sentence overflows",
                "pooling": "mean over chunk vectors, then re-L2-normalise",
                "n_texts_chunked": int((counts > 1).sum()),
                "frac_texts_chunked": float((counts > 1).mean()),
                "chunks_total": int(counts.sum()),
                "chunks_per_text_max": int(counts.max()),
                "single_chunk_bit_identical_to_direct": bool(identical),
                "single_chunk_check_n": len(samp),
                "single_chunk_check_sha256": _sha(a_direct),
            }
            info["token_length"]["n_over_max_seq_length_note"] = (
                f"{n_over} records exceed the 512-token window; chunk+mean-pool "
                "is why this is not truncation")
        t_enc = time.time() - t0

        # byte-determinism: the same 100 rows encoded twice, own call each time
        head = texts[:100]
        if arm == "qwen3":
            a = enc.l2_normalise(enc.encode_texts(model, head,
                                                  batch_size=BATCH[arm],
                                                  show_progress=False))
            b = enc.l2_normalise(enc.encode_texts(model, head,
                                                  batch_size=BATCH[arm],
                                                  show_progress=False))
        else:
            a, _ = enc.encode_texts_chunked(model, head,
                                            batch_size=BATCH[arm],
                                            show_progress=False)
            b, _ = enc.encode_texts_chunked(model, head,
                                            batch_size=BATCH[arm],
                                            show_progress=False)
        info["determinism_100row_reencode_byte_identical"] = bool(
            _sha(a) == _sha(b))
        info["determinism_sha256"] = _sha(a)
        info["reencode_vs_full_corpus_pass_max_abs_diff"] = float(
            np.abs(a - Xn[:100]).max())

        norms = np.linalg.norm(Xn, axis=1)
        info["native_dim"] = int(Xn.shape[1])
        info["unit_norm"] = {"min": float(norms.min()), "max": float(norms.max()),
                             "max_abs_dev_from_1": float(np.abs(norms - 1).max())}
        info["encode_seconds"] = round(t_enc, 1)
        report["arms"][arm] = info
        raw[arm] = Xn
        del model
        torch.cuda.empty_cache()
        print(f"[l1] {arm} done in {t_enc:.1f}s", flush=True)

    # ---- save ------------------------------------------------------------
    meta_common = {"corpus": str(IN_PARQUET), "n_rows": n,
                   "row_order": "n2c2_corpus.parquet order (patient asc, t asc)"}
    X_q4096 = raw["qwen3"]
    X_q768 = enc.mrl_truncate(X_q4096, enc.MRL_DIM)
    enc.save_states(OUT["qwen3_4096"], X_q4096, pid, tt,
                    {**meta_common, "arm": "qwen3", "dim": 4096,
                     "model": enc.QWEN3_REPO, "instruction_prefix": None})
    enc.save_states(OUT["qwen3"], X_q768, pid, tt,
                    {**meta_common, "arm": "qwen3", "dim": enc.MRL_DIM,
                     "model": enc.QWEN3_REPO, "instruction_prefix": None,
                     "reduction": "MRL truncation to first 768 dims, re-L2-normalised",
                     "primary": True})
    enc.save_states(OUT["biolord"], raw["biolord"], pid, tt,
                    {**meta_common, "arm": "biolord", "dim": 768,
                     "model": enc.BIOLORD_REPO,
                     "max_seq_length": enc.BIOLORD_MAX_SEQ,
                     "chunked": True})
    report["outputs"] = {k: str(v) for k, v in OUT.items()}
    return _analyse(report, df, texts, pid, tt, n, X_q4096, X_q768,
                    raw["biolord"], t_start)


def _analyse(report, df, texts, pid, tt, n, X_q4096, X_q768, X_bl, t_start) -> int:
    """Smell test, neighbour concentration, MRL agreement, report write-out."""
    # ---- neighbour smell test -------------------------------------------
    spaces = {"qwen3_768": X_q768, "biolord": X_bl, "qwen3_4096": X_q4096}
    cw = [_content_words(t) for t in texts]
    rng = np.random.default_rng(SMELL_SEED)
    patients = np.sort(rng.choice(np.unique(pid), size=N_SMELL_PATIENTS,
                                  replace=False))
    smell: List[Dict[str, Any]] = []
    for p in patients:
        q_row = int(np.where((pid == p) & (tt == 0))[0][0])
        entry: Dict[str, Any] = {
            "patient_idx": int(p), "t": 0, "row": q_row,
            "query_head": _head(texts[q_row]),
            "query_excerpt": _excerpt(texts[q_row]),
            "query_tokens": int(df["n_ws_tokens"].iloc[q_row]),
            "arms": {}}
        for space, X in spaces.items():
            sims = X @ X[q_row]
            sims[pid == p] = -np.inf          # other patients only
            top = np.argsort(-sims, kind="stable")[:TOPK]
            entry["arms"][space] = [{
                "row": int(r), "patient_idx": int(pid[r]), "t": int(tt[r]),
                "cosine": float(sims[r]),
                "head": _head(texts[r]),
                "excerpt": _excerpt(texts[r]),
                "content_word_jaccard": _jaccard(cw[q_row], cw[r]),
            } for r in top]
        smell.append(entry)
    report["neighbour_smell_test"] = {
        "seed": SMELL_SEED, "n_patients": N_SMELL_PATIENTS,
        "query": "each patient's t=0 record", "top_k": TOPK,
        "patients": [int(p) for p in patients], "results": smell}

    # corpus-wide concentration + a lexical-overlap number behind the eyeball
    agg: Dict[str, Any] = {}
    for space, X in spaces.items():
        S = X @ X.T
        cos, jac = [], []
        for r0 in range(n):
            s = S[r0].copy()
            s[pid == pid[r0]] = -np.inf
            top = np.argsort(-s, kind="stable")[:TOPK]
            for r in top:
                cos.append(float(s[r]))
                j = _jaccard(cw[r0], cw[r])
                if j == j:
                    jac.append(j)
        off = S[~np.eye(n, dtype=bool)]
        agg[space] = {"n_queries": n,
                      "mean_top3_cosine": float(np.mean(cos)),
                      "median_top3_cosine": float(np.median(cos)),
                      "mean_top3_content_word_jaccard": float(np.mean(jac)),
                      "mean_offdiag_cosine": float(off.mean()),
                      "p99_offdiag_cosine": float(np.percentile(off, 99)),
                      "min_offdiag_cosine": float(off.min())}
    rng2 = np.random.default_rng(SMELL_SEED)
    pairs = rng2.integers(0, n, size=(5000, 2))
    rj = [_jaccard(cw[i], cw[j]) for i, j in pairs if pid[i] != pid[j]]
    agg["random_pairs_baseline"] = {
        "n": len(rj), "mean_content_word_jaccard": float(np.mean(rj))}
    agg["comparison_to_invalid_digit_string_run"] = {
        "qwen3_768_mean_top3_cosine": 0.9908326092438821,
        "biolord_mean_top3_cosine": 0.995879988355951,
        "source": str(_LATENT / "results" / "l1_report.json")}
    report["neighbour_profile_overlap"] = agg

    # top-1 neighbour agreement between the 4096-d and the 768-d MRL space
    S4, S7 = X_q4096 @ X_q4096.T, X_q768 @ X_q768.T
    t1_4096, t1_768 = [], []
    for r0 in range(n):
        for S, acc in ((S4, t1_4096), (S7, t1_768)):
            s = S[r0].copy()
            s[pid == pid[r0]] = -np.inf
            acc.append(int(np.argmax(s)))
    iu = np.triu_indices(n, 1)
    report["mrl_truncation"] = {
        "dim": enc.MRL_DIM,
        "neighbour_top1_agreement_4096_vs_768": float(
            np.mean(np.array(t1_4096) == np.array(t1_768))),
        "pairwise_cosine_pearson_4096_vs_768": float(
            np.corrcoef(S4[iu], S7[iu])[0, 1]),
    }

    report["wall_clock_seconds"] = round(time.time() - t_start, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "neighbour_smell_test"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
