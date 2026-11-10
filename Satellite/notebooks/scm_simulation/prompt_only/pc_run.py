"""Run the prompt-only test for one method at one noise level. Jun runs it in Jun's terminal with ARC_LLM_API_KEY
exported; nothing here reads a key file.

    python prompt_only/pc_run.py --method without_donors --sigma 1.0 [--models ...] [--effort default|low] [--limit N]

Each prompt is one user message, no system message, temperature 0, seed 0, no length field (every ARC call streamed),
timeout 600 seconds, HTTP 429 and 503 retried five times 60 seconds apart (lb_skill_run.make_call). Effort default sends
no reasoning_effort field. One record per call goes to <method>/sigma_<s>/<model>_<effort>.jsonl.
A case is skipped when its last record at this effort holds a reply; a record that ended in a call error is run again.
With --effort low only the cases whose default record holds a reply without a valid answer line run (departure 3).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

import pc_lib as pc

sys.path.insert(0, str(pc.SIM.parent / "llm_band_sc"))
import lb_lib as lb             # noqa: E402
import lb_skill_run             # noqa: E402

SEED, TIMEOUT = 0, 600


def run(method, sigma, models, effort, limit):
    prompts = [json.loads(l) for l in pc.PROMPTS.read_text().splitlines()]
    prompts = [p for p in prompts if p["method"] == method and p["sigma"] == sigma]
    if limit:
        prompts = prompts[:limit]
    key = lb.api_key("arc")
    for model in models:
        out = pc.record_path(method, sigma, model, effort)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = pc.last_records(out)
        default = pc.last_records(pc.record_path(method, sigma, model, "default"))
        call = lb_skill_run.make_call(model, SEED, key, None if effort == "default" else effort, TIMEOUT, None)
        for p in prompts:
            pid = p["prompt_id"]
            r = done.get(pid)
            if r and not r.get("error"):
                print(f"{model} {pid} {effort}: has a reply, skipped")
                continue
            if effort == "low":
                d = default.get(pid)
                if d is None or d.get("error") or d.get("ok"):
                    print(f"{model} {pid} low: default record {'missing' if d is None else 'answered or call error'}, skipped")
                    continue
            rec = {"prompt_id": pid, "method": method, "sigma": sigma, "case": p["case"], "model": model,
                   "effort": effort, "prompt": p["prompt"], "max_tokens": None, "seed": SEED,
                   "time": datetime.now(timezone.utc).isoformat()}
            t0 = time.time()
            try:
                content, usage = call([{"role": "user", "content": p["prompt"]}])
                cf = pc.parse(content)
                rec.update(content=content, reasoning=usage.get("reasoning"), usage=usage, counterfactual=cf,
                           ok=cf is not None, error=None)
            except Exception as e:                      # a call error is recorded and run again next time
                rec.update(content=None, reasoning=None, usage=None, counterfactual=None, ok=False, error=repr(e)[:2000])
            rec["seconds"] = round(time.time() - t0, 1)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            u = rec["usage"] or {}
            print(f"{model} {pid} {effort}: ok={rec['ok']} finish={u.get('finish_reason')} "
                  f"streamed={u.get('streamed')} {rec['seconds']}s error={rec['error']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=pc.METHODS)
    ap.add_argument("--sigma", required=True, choices=pc.SIGMAS)
    ap.add_argument("--models", nargs="+", default=list(pc.MODELS))
    ap.add_argument("--effort", default="default", choices=pc.EFFORTS)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.method, a.sigma, a.models, a.effort, a.limit)


if __name__ == "__main__":
    main()
