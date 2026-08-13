#!/usr/bin/env python3
"""
Smoke test for src/vendored/. Tiny synthetic inputs, PASS/FAIL per module.

Run:
    /data/wang/junh/envs/medrag/bin/python preliminary/mimic/scripts/smoke_vendored.py

No network, no model downloads, no GPU. `embedding` is exercised against a
LOCALLY CONSTRUCTED tiny random BERT (transformers config, random weights) so
the cache/pooling/normalisation path runs for real without hitting the hub.
`llm` is checked by import and signature introspection ONLY -- it is not
instantiated and no vLLM engine is started, because the GPUs are in use.
"""
from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

RESULTS = []


def check(name, fn):
    """Run one module's checks, capturing any failure."""
    try:
        detail = fn()
        RESULTS.append((name, True, detail))
        print(f"PASS  {name:12} {detail}")
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name:12} {type(e).__name__}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------- embedding
def t_embedding():
    from vendored.embedding import embed, sha

    assert sha("abc") == sha("abc"), "sha not deterministic"
    assert sha("abc") != sha("abd"), "sha collision on near-identical input"
    assert sha(None) == sha(""), "None should hash as empty"
    assert len(sha("x")) == 40, "sha1 hex digest should be 40 chars"

    # Build a tiny random BERT locally: no download, no network.
    import torch
    from transformers import AutoTokenizer, BertConfig, BertModel

    tmp = Path(tempfile.mkdtemp(prefix="smoke_emb_"))
    mdir = tmp / "tiny_bert"
    cfg = BertConfig(
        vocab_size=1000, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(mdir)
    # Minimal WordPiece vocab so we do not need a hub tokenizer either.
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        f"tok{i}" for i in range(200)
    ] + ["patient", "sepsis", "renal", "failure", "stable", "the", "a"]
    (tmp / "vocab.txt").write_text("\n".join(vocab))
    from transformers import BertTokenizerFast

    BertTokenizerFast(vocab_file=str(tmp / "vocab.txt")).save_pretrained(mdir)
    AutoTokenizer.from_pretrained(mdir)  # confirm it round-trips

    texts = ["patient with sepsis", "renal failure", "stable patient"]
    cpath = tmp / "cache.pkl"

    c1 = embed(str(mdir), texts, cpath, batch=2, maxlen=16,
               device="cpu", verbose=False)
    assert len(c1) == 3, f"expected 3 cached vectors, got {len(c1)}"
    v = c1[sha(texts[0])]
    assert v.dtype == np.float32, f"expected float32, got {v.dtype}"
    assert v.shape == (32,), f"expected hidden_size 32, got {v.shape}"
    nrm = float(np.linalg.norm(v))
    assert abs(nrm - 1.0) < 1e-4, f"not L2-normalised: |v|={nrm}"

    # Idempotence: re-running adds nothing and returns identical vectors.
    c2 = embed(str(mdir), texts, cpath, batch=2, maxlen=16,
               device="cpu", verbose=False)
    assert len(c2) == 3, "cache grew on a repeat run"
    assert np.array_equal(c1[sha(texts[0])], c2[sha(texts[0])]), "vector changed"

    # Incremental: one new text is added, old ones untouched.
    c3 = embed(str(mdir), texts + ["a new note"], cpath, batch=2, maxlen=16,
               device="cpu", verbose=False)
    assert len(c3) == 4, f"expected 4 after adding one text, got {len(c3)}"
    assert np.array_equal(c1[sha(texts[1])], c3[sha(texts[1])]), "old vec changed"

    # Duplicate inputs collapse to one cache entry (content addressing).
    c4 = embed(str(mdir), ["dup", "dup", "dup"], tmp / "c2.pkl", batch=2,
               maxlen=16, device="cpu", verbose=False)
    assert len(c4) == 1, f"3 identical texts -> {len(c4)} entries, expected 1"

    return f"|v|={nrm:.6f} dim=32 cache 3->4 idempotent dedup ok"


# ---------------------------------------------------------------- features
def t_features():
    from vendored.features import _bag, _flatten, cv_auroc, design_matrix

    assert _flatten([["a", "b"], ["c"]]) == ["a", "b", "c"], "nested flatten"
    assert _flatten(None) == [], "None should flatten to []"
    assert _flatten(["a", 1]) == ["a", "1"], "should stringify"

    rec = {"conditions": [["I10"]], "procedures": [["P1"]], "drugs": [["D1", "D1"]]}
    b = _bag(rec)
    assert b == {"c:I10", "p:P1", "d:D1"}, f"unexpected bag {b}"
    assert _bag({"conditions": [["X"]]}, prefix=False) == {"X"}, "prefix=False"
    # Same raw code in two sections must stay distinct under prefixing.
    assert _bag({"conditions": [["Z"]], "drugs": [["Z"]]}) == {"c:Z", "d:Z"}

    # min_df filtering: 'rare' appears once, must be dropped at min_df=2.
    bags = [{"a", "b"}, {"a", "b"}, {"a", "rare"}]
    X, vocab = design_matrix(bags, min_df=2)
    assert set(vocab) == {"a", "b"}, f"min_df=2 kept {set(vocab)}"
    assert X.shape == (3, 2), f"shape {X.shape}"
    assert X.dtype == np.float32, X.dtype
    assert X[:, vocab["a"]].tolist() == [1.0, 1.0, 1.0], "column a"
    assert X[:, vocab["b"]].tolist() == [1.0, 1.0, 0.0], "column b"
    assert set(X.ravel().tolist()) <= {0.0, 1.0}, "matrix must be binary"

    # Reusing a train vocab drops unseen codes rather than erroring.
    Xte, v2 = design_matrix([{"a", "never_seen"}], vocab=vocab)
    assert v2 is vocab and Xte.shape == (1, 2), "vocab reuse"
    assert Xte[0, vocab["a"]] == 1.0 and Xte.sum() == 1.0, "unseen code leaked"

    # Learnable signal: code 'sig' present iff y==1 -> AUROC should be high.
    rng = np.random.default_rng(0)
    n = 120
    y = np.array([i % 2 for i in range(n)])
    sig_bags = []
    for i in range(n):
        codes = {f"noise{rng.integers(0, 6)}", f"noise{rng.integers(0, 6)}"}
        if y[i] == 1:
            codes.add("sig")
        sig_bags.append(codes)
    r = cv_auroc(sig_bags, y, folds=5, seed=0, min_df=5)
    assert r["n"] == n and r["n_pos"] == n // 2, r
    assert r["auroc_mean"] > 0.9, f"separable data gave AUROC {r['auroc_mean']}"

    # Pure noise -> AUROC near chance. Guards against label leakage.
    noise_bags = [{f"n{rng.integers(0, 8)}", f"n{rng.integers(0, 8)}"}
                  for _ in range(n)]
    r0 = cv_auroc(noise_bags, y, folds=5, seed=0)
    assert 0.3 < r0["auroc_mean"] < 0.7, f"noise gave AUROC {r0['auroc_mean']}"

    # Determinism at fixed seed.
    r_again = cv_auroc(sig_bags, y, folds=5, seed=0, min_df=5)
    assert r_again["auroc_mean"] == r["auroc_mean"], "not deterministic at seed=0"

    return (f"signal AUROC={r['auroc_mean']:.3f} noise AUROC={r0['auroc_mean']:.3f} "
            f"nfeat={r['n_feat_last_fold']}")


# ------------------------------------------------------------------- stats
def t_stats():
    from vendored.stats import auc_mw, boot, star, tjur

    # Perfect and inverted separation.
    y = np.array([0, 0, 1, 1])
    assert auc_mw(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0, "perfect sep"
    assert auc_mw(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0, "inverted sep"
    # All-tied scores -> exactly 0.5 (this is the tie correction working).
    assert auc_mw(y, np.array([0.5] * 4)) == 0.5, "tie correction"
    assert np.isnan(auc_mw(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))), \
        "single-class should be nan"

    # Agreement with sklearn on random data -- the real check on auc_mw.
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    yr = rng.integers(0, 2, 400)
    pr = np.clip(0.5 + 0.25 * yr + rng.normal(0, 0.2, 400), 0, 1)
    mine, ref = auc_mw(yr, pr), roc_auc_score(yr, pr)
    assert abs(mine - ref) < 1e-12, f"auc_mw={mine} vs sklearn={ref}"
    # And with heavy ties, where naive implementations diverge.
    pt = np.round(pr, 1)
    mt, rt = auc_mw(yr, pt), roc_auc_score(yr, pt)
    assert abs(mt - rt) < 1e-12, f"tied: auc_mw={mt} vs sklearn={rt}"

    # Tjur's D.
    assert abs(tjur(y, np.array([0.0, 0.0, 1.0, 1.0])) - 1.0) < 1e-12, "tjur max"
    assert abs(tjur(y, np.array([0.5] * 4))) < 1e-12, "tjur of constant = 0"
    # Calibration sensitivity: same ranking, squashed scale -> smaller D.
    d_wide = tjur(y, np.array([0.1, 0.2, 0.8, 0.9]))
    d_narrow = tjur(y, np.array([0.49, 0.495, 0.505, 0.51]))
    assert d_narrow < d_wide, "tjur should be calibration-sensitive"

    # Bootstrap: CI must bracket the point estimate and exclude 0.5 for signal.
    lo, hi = boot(yr, pr, auc_mw, B=200, seed=0)
    assert lo < ref < hi, f"CI [{lo},{hi}] excludes point estimate {ref}"
    assert lo > 0.5, f"strong signal CI [{lo},{hi}] should clear 0.5"
    assert star(lo, hi, 0.5) == "*", "star should fire"

    # Null data: CI should straddle 0.5, star blank.
    pn = rng.random(400)
    lo0, hi0 = boot(yr, pn, auc_mw, B=200, seed=0)
    assert lo0 < 0.5 < hi0, f"null CI [{lo0},{hi0}] should contain 0.5"
    assert star(lo0, hi0, 0.5) == " ", "star should not fire on null"

    # Reproducibility at fixed seed; different seed moves the CI.
    assert boot(yr, pr, auc_mw, B=200, seed=0) == (lo, hi), "not reproducible"
    assert boot(yr, pr, auc_mw, B=200, seed=1) != (lo, hi), "seed had no effect"

    # Wider alpha -> wider interval.
    lo90, hi90 = boot(yr, pr, auc_mw, B=200, seed=0, alpha=0.10)
    assert (hi90 - lo90) <= (hi - lo) + 1e-12, "90% CI wider than 95%"

    # Single-class input must raise, not silently return nan (our change).
    try:
        boot(np.ones(20, dtype=int), rng.random(20), auc_mw, B=50, seed=0)
        raise AssertionError("single-class boot should have raised")
    except RuntimeError:
        pass

    return (f"auc_mw==sklearn (tied too) CI=[{lo:.3f},{hi:.3f}] "
            f"null=[{lo0:.3f},{hi0:.3f}] star ok")


# --------------------------------------------------------------------- llm
def t_llm():
    """Import + signature introspection only. No model load, no GPU, no vLLM engine."""
    from vendored.llm import QwenIntegrator

    assert inspect.isclass(QwenIntegrator)
    for m in ("score", "generate", "__call__", "_format", "_ensure_llm"):
        assert callable(getattr(QwenIntegrator, m, None)), f"missing {m}"

    # score(self, texts) -> list[float].
    # llm.py uses `from __future__ import annotations`, so annotations are
    # strings, not objects -- compare as strings.
    s = inspect.signature(QwenIntegrator.score)
    assert list(s.parameters) == ["self", "texts"], list(s.parameters)
    assert str(s.return_annotation) in ("List[float]", "list[float]"), \
        s.return_annotation
    assert "List[str]" in str(s.parameters["texts"].annotation), \
        s.parameters["texts"].annotation

    # generate accepts a list (Union input) and returns Union.
    g = inspect.signature(QwenIntegrator.generate)
    assert "prompts" in g.parameters, "generate should take `prompts`, not `prompt`"
    ann = str(g.parameters["prompts"].annotation)
    assert "Union" in ann and "Sequence" in ann, f"generate not batched: {ann}"
    assert "Union" in str(g.return_annotation), g.return_annotation
    # Original hardcoded defaults must be preserved.
    assert g.parameters["temperature"].default == 0.7
    assert g.parameters["top_p"].default == 0.9
    assert g.parameters["max_tokens"].default == 32768
    assert g.parameters["repetition_penalty"].default == 1.2

    # Construction must NOT touch the GPU (lazy=True default).
    q = QwenIntegrator("Qwen/Qwen2.5-7B-Instruct", lazy=True, verbose=False)
    assert q.llm is None, "constructor built an engine despite lazy=True"
    assert q.config()["gpu_memory_utilization"] == 0.85
    assert q.config()["enforce_eager"] is True
    assert q.config()["loaded"] is False

    # Prompt formatting, no engine needed.
    f = q._format("hi")
    assert f.startswith("<|im_start|>user\nhi<|im_end|>"), repr(f)
    assert QwenIntegrator("meta-llama/x", verbose=False)._format("hi") == "hi"

    # Empty/blank-input fast paths must not build an engine. This is a real
    # guard, not a formality: an earlier revision of llm.py let generate("")
    # fall through and tried to start vLLM.
    assert q.score([]) == []
    assert q.generate([]) == []
    assert q.generate("") == ""
    assert q.generate(["", "   "]) == ["", ""]
    assert all(np.isnan(v) for v in q.score(["", "  "]))
    assert q.llm is None, "a blank call built a vLLM engine"

    # The source repo's anti-label parse-failure fallback
    # (run_condition_D.py:308, `result['prediction'] = 1 - int(sample['label'])`)
    # must not have been carried over. Checked against EXECUTABLE code only:
    # llm.py's header legitimately names the pattern to record that it was
    # deliberately dropped, so a plain substring scan over the file would
    # false-positive on our own documentation. Strip comments and docstrings by
    # round-tripping through the AST.
    import ast

    tree = ast.parse(Path(inspect.getfile(QwenIntegrator)).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]  # drop the docstring statement
    code = ast.unparse(tree)  # comments are not preserved by the AST
    for pat in ("1 - int(", "1 - label", "1 - y", "1 - self.label"):
        assert pat not in code, f"anti-label fallback pattern present: {pat}"

    ck = inspect.signature(QwenIntegrator.__init__)
    return (f"gpu_mem_util={ck.parameters['gpu_memory_utilization'].default} "
            f"lazy={ck.parameters['lazy'].default} sigs ok (engine NOT started)")


if __name__ == "__main__":
    print("smoke_vendored: exercising src/vendored/ on synthetic inputs")
    print(f"python: {sys.executable}")
    print("-" * 72)
    check("embedding", t_embedding)
    check("features", t_features)
    check("stats", t_stats)
    check("llm", t_llm)
    print("-" * 72)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{n_pass}/{len(RESULTS)} modules PASS")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
