"""lb_gate0 — probe each gateway once for the three facts the spec arms depend on (plan 2026-09-23, gate 0):
  1. does the chat endpoint pass a `tools` declaration through and return `tool_calls`?
  2. which `reasoning_effort` values does it accept (low / medium / high / none)?
  3. does the 8,000-token cap lift under `stream: true`?
Every raw response is stored, nothing is inferred from a status code alone. Keys are read from the
environment only (lb_lib.api_key); a gateway whose key is not exported is skipped and said so.
Run in a terminal where the keys are exported:
    export ARC_LLM_API_KEY=...   # and/or HOKIE_AI_KEY, OPENAI_API_KEY
    python lb_gate0.py
Writes lb_gate0.jsonl (one record per probe, raw response included) and prints one table.
"""
import json
import os
import sys
import time

import requests

import lb_lib as lb

ARC_MODELS = ["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low"]   # per `--list` on 2026-09-23
GATEWAYS = {"arc": ARC_MODELS, "hokie": ["gpt-5.5"], "openai": ["gpt-5.5"]}
TIMEOUT = 180
OUT = lb.LB / "lb_gate0.jsonl"

TOOL = {"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                   "required": ["a", "b"]}}}


def endpoint(prov, key):
    if prov == "arc":
        return f"{lb.BASE}/chat/completions", {"Authorization": f"Bearer {key}"}
    if prov == "hokie":
        return f"{lb.HOKIE_BASE}/chat/completions", {"api-key": key}
    return f"{lb.OPENAI_BASE}/chat/completions", {"Authorization": f"Bearer {key}"}


def post(prov, key, body, stream=False):
    url, auth = endpoint(prov, key)
    t0 = time.time()
    try:
        r = requests.post(url, json=body, timeout=TIMEOUT, stream=stream,
                          headers={**auth, "Content-Type": "application/json"})
        if stream:
            first = next(r.iter_lines(), b"")
            raw = {"status": r.status_code, "first_line": first.decode("utf-8", "replace")[:500]}
            r.close()
        else:
            try:
                raw = r.json()
            except ValueError:
                raw = {"non_json_body": r.text[:2000]}
            raw = {"status": r.status_code, "body": raw}
    except Exception as e:                       # network / timeout: record, do not raise
        raw = {"status": None, "exception": repr(e)}
    raw["seconds"] = round(time.time() - t0, 1)
    return raw


def base_body(prov, model, content, cap):
    msgs = [{"role": "user", "content": content}]
    if prov == "arc":
        return {"model": model, "messages": msgs, "temperature": 0, "max_tokens": cap, "seed": 0}
    return {"model": model, "messages": msgs, "seed": 0, "max_completion_tokens": cap}


def walk_keys(obj, hits, path=""):
    """Every key anywhere in the response whose name mentions reasoning / thinking, with whether it is set."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(w in k.lower() for w in ("reason", "think", "thought")):
                hits[f"{path}.{k}"] = bool(v) if not isinstance(v, (int, float)) else v
            walk_keys(v, hits, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_keys(v, hits, f"{path}[{i}]")
    return hits


def list_models():
    """GET /models on each gateway whose key is exported; prints the ids (Kimi-K3 answered 'Model not found' on 2026-09-23)."""
    for prov in GATEWAYS:
        try:
            key = lb.api_key(prov)
        except Exception:
            print(f"[{prov}] key not exported — skipped"); continue
        url, auth = endpoint(prov, key)
        url = url.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            r = requests.get(url, headers=auth, timeout=60)
            ids = [m.get("id") for m in (r.json().get("data") or [])] if r.status_code == 200 else None
        except Exception as e:
            ids = None; r = None
        print(f"[{prov}] {url} -> {getattr(r, 'status_code', None)}: {ids if ids is not None else (r.text[:300] if r is not None else 'request failed')}")


def main():
    if "--list" in sys.argv:
        return list_models()
    records, rows = [], []
    for prov, models in GATEWAYS.items():
        try:
            key = lb.api_key(prov)
        except Exception:
            print(f"[{prov}] key {lb.KEY_ENV[prov]} not exported — skipped")
            continue
        for model in models:
            spec = f"{prov}:{model}"
            # 1. tool call
            body = base_body(prov, model, "Use the add tool to add 2 and 3. Do not answer in words.", 256)
            body["tools"] = [TOOL]; body["tool_choice"] = "auto"
            raw = post(prov, key, body)
            msg = (((raw.get("body") or {}).get("choices") or [{}])[0]).get("message", {}) if raw.get("status") == 200 else {}
            tool_calls = bool(msg.get("tool_calls"))
            records.append({"probe": "tools", "gateway": prov, "model": model, "raw": raw})
            # 2. effort levels
            effort_ok = {}
            for eff in ("low", "medium", "high", None):
                body = base_body(prov, model, "Reply with the single word OK.", 64)
                if eff:
                    body["reasoning_effort"] = eff
                raw = post(prov, key, body)
                hits = walk_keys(raw.get("body"), {}) if raw.get("status") == 200 else {}
                effort_ok[eff or "none"] = {"status": raw.get("status"), "reasoning_keys": hits}
                records.append({"probe": "effort", "effort": eff, "gateway": prov, "model": model, "raw": raw})
            # 3. cap: 16000 non-streaming, then streaming
            cap_ns = post(prov, key, base_body(prov, model, "Reply with the single word OK.", 16000))
            body = base_body(prov, model, "Reply with the single word OK.", 16000); body["stream"] = True
            cap_st = post(prov, key, body, stream=True)
            records.append({"probe": "cap_nonstream_16000", "gateway": prov, "model": model, "raw": cap_ns})
            records.append({"probe": "cap_stream_16000", "gateway": prov, "model": model, "raw": cap_st})
            rows.append((spec, tool_calls,
                         " ".join(f"{k}:{v['status']}" for k, v in effort_ok.items()),
                         cap_ns.get("status"), cap_st.get("status"),
                         sorted({kk for v in effort_ok.values() for kk in v["reasoning_keys"]})))
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n{'model':38s} tool_calls  effort status (low/medium/high/none)    cap16k  stream16k  reasoning keys seen")
    for spec, tc, eff, cns, cst, keys in rows:
        print(f"{spec:38s} {str(tc):10s}  {eff:36s} {str(cns):6s}  {str(cst):9s}  {keys}")
    print(f"\n{len(records)} raw responses -> {OUT}")


if __name__ == "__main__":
    main()
