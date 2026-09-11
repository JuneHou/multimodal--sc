"""lb_run — send the prompts to the ARC-hosted models and store every reply.
    python lb_run.py --experiment shc|scm|abl_noguide|abl_anon|abl_nocal|abl_rule [--models gpt-oss-120b GLM-5.3 Kimi-K3-thinking-low DeepSeek-V4-Flash-thinking-low] [--limit 2] [--seed 0] [--effort low|default]
Resumable per (experiment, model): lb_replies_<experiment>_<model>.jsonl, one line per prompt (a failed prompt is
tried again only if the script is run again; the scorer keeps the last line per prompt) with the raw
reply, the parsed values, token usage, finish reason and seconds. One attempt per prompt (at temperature 0 a repeat
would return the same reply); the call is repeated only when the ARC service limit is hit (HTTP 429 or 503),
up to RATE_LIMIT_RETRIES times, RATE_LIMIT_WAIT seconds apart (Jun, 2026-09-09).
"""
import argparse
import json
import time

import requests

import lb_lib as lb

RATE_LIMIT_CODES = (429, 503)   # ARC concurrency / rate limit reached, or service busy
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_WAIT = 60            # seconds between retries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["shc", "scm", "abl_noguide", "abl_anon", "abl_nocal", "abl_rule"], required=True)
    ap.add_argument("--models", nargs="+", default=lb.MODELS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default=lb.REASONING_EFFORT, help="reasoning_effort sent with each request; 'default' = send nothing")
    a = ap.parse_args()
    effort = None if a.effort == "default" else a.effort
    tag = a.effort
    key = lb.api_key()
    P = [json.loads(l) for l in open(lb.LB / f"lb_prompts_{a.experiment}.jsonl")]
    if a.limit:
        first = {}
        for p in P:
            first.setdefault(p["sensor"], p)
        P = list(first.values())[:a.limit]
    for model in a.models:
        mf = lb.LB / f"lb_replies_{a.experiment}_{model}_{tag}.jsonl"
        done = {json.loads(l)["prompt_id"] for l in open(mf) if json.loads(l)["ok"]} if mf.exists() else set()   # failed ones are retried
        todo = [p for p in P if p["prompt_id"] not in done]
        print(f"{model} / {a.experiment}: {len(todo)} to do, {len(done)} done", flush=True)
        with open(mf, "a") as f:
            for i, p in enumerate(todo):
                rec = {"prompt_id": p["prompt_id"], "experiment": a.experiment, "model": model, "seed": a.seed,
                       "reasoning_effort": tag,
                       "attempts": [], "values": None, "ok": False}
                t0 = time.time()
                for attempt in range(1, 1 + RATE_LIMIT_RETRIES + 1):
                    try:
                        text, usage = lb.call(model, p["prompt"], seed=a.seed, key=key, reasoning_effort=effort)
                        vals = lb.parse(p["sensor"], text, p.get("labels"))
                        rec["attempts"].append({"attempt": attempt, "reply": text, "usage": usage, "parsed": vals is not None})
                        if vals is not None:
                            rec.update(values=vals, ok=True, reply=text, usage=usage)
                        break                                           # one real attempt: temperature 0, a repeat would repeat
                    except requests.HTTPError as e:                     # retry ONLY when the ARC service limit is hit
                        code = e.response.status_code if e.response is not None else None
                        body = (e.response.text[:600] if e.response is not None else "")
                        rec["attempts"].append({"attempt": attempt, "error": f"HTTP {code}: {e}", "body": body})
                        if code in RATE_LIMIT_CODES and attempt <= RATE_LIMIT_RETRIES:
                            time.sleep(RATE_LIMIT_WAIT); continue
                        break
                    except Exception as e:
                        rec["attempts"].append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"})
                        break
                rec["seconds"] = round(time.time() - t0, 1)
                f.write(json.dumps(rec) + "\n"); f.flush()
                last = rec["attempts"][-1]
                print(f"[{i + 1}/{len(todo)}] {p['prompt_id']} {'ok' if rec['ok'] else 'FAILED'} {rec['seconds']}s "
                      f"{str(last.get('reply', last.get('error', '')))[:100]!r}", flush=True)


if __name__ == "__main__":
    main()
