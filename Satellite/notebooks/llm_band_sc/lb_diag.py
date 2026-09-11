"""lb_diag — one call to one model with the run settings, printing status, elapsed time, the server's error
body if any, finish reason and token usage.   python lb_diag.py Kimi-K3-thinking-low [max_tokens] [reasoning_effort|none]"""
import json
import sys
import time

import requests

import lb_lib as lb

model = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
effort = (sys.argv[3] if len(sys.argv) > 3 else lb.REASONING_EFFORT)
effort = None if effort == "none" else effort
p = [json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")][0]
body = {"model": model, "messages": [{"role": "user", "content": p["prompt"]}], "temperature": 0, "max_tokens": max_tokens, "seed": 0}
if effort:
    body["reasoning_effort"] = effort
t0 = time.time()
r = requests.post(f"{lb.BASE}/chat/completions", json=body, timeout=900,
                  headers={"Authorization": f"Bearer {lb.api_key()}", "Content-Type": "application/json"})
print(f"{model} max_tokens={max_tokens} reasoning_effort={effort}: HTTP {r.status_code} after {time.time() - t0:.0f}s")
if r.status_code != 200:
    print("server says:", r.text[:800])
else:
    j = r.json(); ch = j["choices"][0]
    print("finish:", ch.get("finish_reason"), "| usage:", j.get("usage"))
    print("content:", repr((ch["message"].get("content") or "")[:200]))
    extra = {k: str(v)[:80] for k, v in ch["message"].items() if k not in ("content", "role") and v}
    print("other message fields:", extra)
