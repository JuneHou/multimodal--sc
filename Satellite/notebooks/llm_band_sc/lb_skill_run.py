"""lb_skill_run — run one skill arm over the 27 cells (plan approved 2026-09-24).

    python lb_skill_run.py --skill shc-configure|shc-design|shc-implement [--models ...] [--effort low]
                           [--limit N] [--seed 0] [--run-label NAME] [--max-tokens 16000] [--step-cap 12]
    python lb_skill_run.py --show <reply file> [--cell <prompt_id>]        print a cell's transcript turn by turn

The model never receives the data in a message: it runs scripts/show_data.py (or reads data.csv) itself. Each
record is written in lb_run's shape (prompt_id, experiment, model, ..., values, ok) so lb_score.py can score it,
plus the whole transcript: every turn's reply, raw response, command and output; every fitter run; the program.
Keys are read from the environment only (lb_lib.api_key).
"""
import argparse
import json
import time

import requests

import lb_guard
import lb_lib as lb
import lb_skill_loop
import lb_skills_build

SKILLS = {"shc-configure": "shc_configure", "shc-design": "shc_design", "shc-implement": "shc_implement"}
RATE_LIMIT_CODES, RATE_LIMIT_RETRIES, RATE_LIMIT_WAIT = (429, 503), 5, 60


def make_call(spec_model, seed, key, effort, timeout, max_tokens):
    def call(messages):
        for k in range(RATE_LIMIT_RETRIES + 1):
            try:
                return lb.call_turns(spec_model, messages, seed=seed, key=key, reasoning_effort=effort,
                                     timeout=timeout, max_tokens=max_tokens)
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code in RATE_LIMIT_CODES and k < RATE_LIMIT_RETRIES:
                    print(f"    HTTP {code}; waiting {RATE_LIMIT_WAIT}s"); time.sleep(RATE_LIMIT_WAIT); continue
                body = e.response.text[:400] if e.response is not None else ""      # keep the gateway's reason
                raise requests.HTTPError(f"{e} :: {body}", response=e.response) from None
    return call


def show(path, cell=None):
    for l in open(path):
        r = json.loads(l)
        if cell and r["prompt_id"] != cell:
            continue
        print(f"\n######## {r['prompt_id']}  {r['model']}  skill={r['skill']}  ended={r['ended']} steps={r['n_steps']} ok={r['ok']}")
        for s in r["steps"]:
            u = s.get("usage") or {}
            rs = u.get("reasoning")
            c = s.get("command") or {}
            print(f"\n--- step {s['step']} | finish={u.get('finish_reason')} | reasoning {len(rs) if isinstance(rs, str) else 0} chars"
                  f" | blocks in reply: {c.get('n_blocks')} ---")
            if isinstance(rs, str) and rs:
                print("[thinking] " + rs[:1500] + (" …" if len(rs) > 1500 else ""))
            print("[executed] " + (c.get("text") or c.get("message") or "")[:1200])
            print("[output]   " + (s.get("output") or "")[:2500])
        print(f"\n[answer] {r['answer']}")
        if r.get("fit_runs"):
            print(f"[fit runs] {len(r['fit_runs'])}: " + "; ".join(f"{f['donor_ids']} w={[round(x, 3) for x in f['weights']]} mse={f['fit_mse']:.4g}" for f in r["fit_runs"]))
        if cell:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", choices=list(SKILLS))
    ap.add_argument("--models", nargs="+", default=["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low", "openai:gpt-5.5"])
    ap.add_argument("--effort", default=lb.REASONING_EFFORT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-label", default="")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--step-cap", type=int, default=lb_skill_loop.STEP_CAP)
    ap.add_argument("--show", default="")
    ap.add_argument("--cell", default="")
    a = ap.parse_args()
    if a.show:
        return show(a.show, a.cell or None)
    if not a.skill:
        ap.error("--skill is required")
    lb_skills_build.check()                                   # SK1: what is mounted is what was approved
    skill_dir = lb_skills_build.SKILLS / a.skill
    exp = SKILLS[a.skill]
    effort = None if a.effort == "default" else a.effort
    out_dir = lb.LB / "runs" / a.run_label if a.run_label else lb.LB
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")]
    if a.limit:
        keep, seen = [], {}
        for p in base:
            if seen.get(p["sensor"], 0) < a.limit:
                keep.append(p); seen[p["sensor"]] = seen.get(p["sensor"], 0) + 1
        base = keep
    panel9 = lb_guard.training_panel()
    import sys, numpy, scipy, pandas
    env = {"python": sys.executable, "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
           "max_tokens": a.max_tokens, "step_cap": a.step_cap, "skill_manifest": json.loads(lb_skills_build.MANIFEST.read_text())}
    for spec_model in a.models:
        prov, model, _ = lb.parse_model(spec_model)
        key = lb.api_key(prov)
        out = out_dir / f"lb_replies_{exp}_{lb.safe_name(spec_model)}_{a.effort}.jsonl"
        done = {json.loads(l)["prompt_id"] for l in open(out) if json.loads(l).get("ok")} if out.exists() else set()
        print(f"{spec_model} / {a.skill} (effort {a.effort}, step cap {a.step_cap}) -> {out.name}; {len(done)} done")
        call = make_call(spec_model, a.seed, key, effort, a.timeout, a.max_tokens)
        with open(out, "a") as f:
            for p in base:
                pid = p["prompt_id"][:-4] + "_" + exp[4:]
                if pid in done:
                    continue
                cell = lb_guard.cell_panel(panel9, p["sensor"], p["target_site"])
                rec = lb_skill_loop.run_cell(skill_dir, cell, p["sensor"], p["target_site"], call, step_cap=a.step_cap)
                last_usage = next((s["usage"] for s in reversed(rec["steps"]) if s.get("usage")), {})
                rec.update(prompt_id=pid, experiment=exp, model=spec_model, served_model=model, gateway=prov, seed=a.seed,
                           reasoning_effort=a.effort, env=env,
                           usage={k: v for k, v in last_usage.items() if k != "raw_response"},
                           reply=next((s["reply"] for s in reversed(rec["steps"]) if s.get("reply")), ""))
                f.write(json.dumps(rec) + "\n"); f.flush()
                fits = len(rec["fit_runs"])
                print(f"  {pid:36s} ok={rec['ok']!s:5s} steps={rec['n_steps']:2d} ended={rec['ended']:9s} "
                      f"fit_runs={fits} refs={len(rec['refs_read'])} data={rec['data_shown']}")


if __name__ == "__main__":
    main()
