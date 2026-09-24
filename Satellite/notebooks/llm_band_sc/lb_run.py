"""lb_run — send the prompts to the hosted models and store every reply.
    python lb_run.py --experiment shc|scm|abl_noguide|abl_anon|abl_nocal|abl_rule|shc_steps [--models gpt-oss-120b GLM-5.3 Kimi-K3-thinking-low openai:gpt-5.5] [--limit 2] [--seed 0] [--effort low|default] [--run-label ...] [--reasoning-summary]
A model is named `gateway:model` -- arc:, hokie: or openai: -- because the gateway is part of its
identity: `gpt-5.5` is an alias that Hokie AI and OpenAI resolve independently, so the same string on
two gateways is two serving paths whose replies must not be pooled. The gateway goes into the reply
FILENAME, so one can never append into the other's file, and into every record as `gateway` and
`usage.served_model` (the snapshot that actually answered). A bare name keeps its historical default
(GPT-5 family -> hokie, else arc), so pre-2026-09-19 commands and files still mean what they meant.
Each gateway has its own key: ARC_LLM_API_KEY, HOKIE_AI_KEY, OPENAI_API_KEY. None is ever read from a
file by this script -- export it in your own shell.
--run-label writes into runs/<label>/ so a repeat run sits beside the previous one instead of resuming it;
this matters because NEITHER gateway is reproducible (ARC ignores temperature 0 / seed 0, and Hokie AI
rejects temperature outright for the GPT-5 family), so a single run does not pin down a model.
Resumable per (experiment, model): lb_replies_<experiment>_<model>.jsonl, one line per prompt (a failed prompt is
tried again only if the script is run again; the scorer keeps the last line per prompt) with the raw
reply, the parsed values, token usage, finish reason and seconds. One attempt per prompt -- NOT because a repeat
would reproduce it (it does not: see --run-label above), but so that a run costs one call per prompt and
repeats are taken deliberately as whole labelled runs; the call is repeated when the ARC service limit is hit (HTTP 429 or 503), or when
a 400 response's body names a transient backend condition (2026-09-15: "read timeout", "queue is full" seen on
the shc_steps run) rather than a permanent one ("Model not found" is never retried), up to RATE_LIMIT_RETRIES
times, RATE_LIMIT_WAIT seconds apart (Jun, 2026-09-09 / 2026-09-15).
"""
import argparse
import json
import time

import requests

import lb_lib as lb

RATE_LIMIT_CODES = (429, 503)   # ARC concurrency / rate limit reached, or service busy
TRANSIENT_400_PHRASES = ("timeout", "queue is full", "queue full")   # a 400 with one of these is worth retrying
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_WAIT = 60            # seconds between retries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["shc", "scm", "abl_noguide", "abl_anon", "abl_nocal", "abl_rule", "shc_steps"], required=True)
    ap.add_argument("--models", nargs="+", default=lb.MODELS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default=lb.REASONING_EFFORT, help="reasoning_effort sent with each request; 'default' = send nothing")
    ap.add_argument("--run-label", default="", help="write replies to runs/<label>/ instead of the main folder, "
                                                    "so a repeat run does not overwrite the previous one (lb_runs.py reads runs/*/)")
    ap.add_argument("--reasoning-summary", action="store_true",
                    help="GPT-5 family only: call the Responses API and store the reasoning SUMMARY "
                         "(not the raw chain, which OpenAI does not expose) alongside the answer")
    a = ap.parse_args()
    effort = None if a.effort == "default" else a.effort
    tag = a.effort
    out_dir = lb.LB / "runs" / a.run_label if a.run_label else lb.LB
    out_dir.mkdir(parents=True, exist_ok=True)
    P = [json.loads(l) for l in open(lb.LB / f"lb_prompts_{a.experiment}.jsonl")]
    if a.limit:
        first = {}
        for p in P:
            first.setdefault(p["sensor"], p)
        P = list(first.values())[:a.limit]
    for spec in a.models:
        prov, _, spec = lb.parse_model(spec)          # the gateway is part of the model's identity
        summary = a.reasoning_summary and prov != "arc"
        if a.reasoning_summary and not summary:
            print(f"{spec}: --reasoning-summary ignored, ARC already returns the raw chain", flush=True)
        key = lb.api_key(prov)                        # each gateway has its own key
        mf = out_dir / f"lb_replies_{a.experiment}_{lb.safe_name(spec)}_{tag}.jsonl"
        done = {json.loads(l)["prompt_id"] for l in open(mf) if json.loads(l)["ok"]} if mf.exists() else set()   # failed ones are retried
        todo = [p for p in P if p["prompt_id"] not in done]
        print(f"{spec} / {a.experiment} [{prov}{', responses+summary' if summary else ''}] -> "
              f"{mf.parent.name}/: {len(todo)} to do, {len(done)} done", flush=True)
        with open(mf, "a") as f:
            for i, p in enumerate(todo):
                rec = {"prompt_id": p["prompt_id"], "experiment": a.experiment, "model": spec,
                       "gateway": prov, "seed": a.seed,
                       "reasoning_effort": tag,
                       "attempts": [], "values": None, "ok": False}
                t0 = time.time()
                for attempt in range(1, 1 + RATE_LIMIT_RETRIES + 1):
                    try:
                        fn = lb.call_summary if summary else lb.call
                        text, usage = fn(spec, p["prompt"], seed=a.seed, key=key, reasoning_effort=effort)
                        vals = lb.parse(p["sensor"], text, p.get("labels"))
                        rec["attempts"].append({"attempt": attempt, "reply": text, "usage": usage, "parsed": vals is not None})
                        if vals is not None:
                            rec.update(values=vals, ok=True, reply=text, usage=usage)
                        break                                           # one real attempt; repeats are whole runs (--run-label)
                    except requests.HTTPError as e:                     # retry only on a transient condition
                        code = e.response.status_code if e.response is not None else None
                        body = (e.response.text[:600] if e.response is not None else "")
                        rec["attempts"].append({"attempt": attempt, "error": f"HTTP {code}: {e}", "body": body})
                        transient = code in RATE_LIMIT_CODES or (code == 400 and any(w in body.lower() for w in TRANSIENT_400_PHRASES))
                        if transient and attempt <= RATE_LIMIT_RETRIES:
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
