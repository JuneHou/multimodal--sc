"""scs_run — run solver models on one skill folder over the cases of a use (the run stage of every pipeline).

    python scs_run.py --method <method> --use <use> --skill <skill folder> [--models ...] [--effort low]
                      [--seed 0] [--step-cap 12] [--limit N] [--label NAME] [--out FOLDER]

The skill folder is a configure or design skill built by scs_build.py, or a profile skill collected by scs_profile.py.
Models, effort and step cap are those of the SHC runs: gpt-oss-120b, GLM-5.3, DeepSeek-V4.1-Flash and GPT-5.5; low
effort; 12 steps. No reply-length limit is set for any model, so the reply length is the provider's default maximum
and every ARC call streams (Jun, ~/.claude/CLAUDE.md, 2026-11-04: "you will NEVER NEVER NEVER set token limit for what
ever experiment which ever model"); each record stores max_tokens null. Model calls go through the SHC
runs' own call function (llm_band_sc/lb_skill_run.make_call: retries on HTTP 429 and 503, 5 times, 60 s apart).
Keys are read from the environment only; Jun runs every model call in Jun's own terminal with the keys exported.

A case already holding a record that ended with an answer or at the step cap is never run again; a case whose only
record ended in a call error is run again. Records go to <out>/<skill>_<model>_<effort>.jsonl when --out is given, and
to methods/<method>/runs/<label>/ otherwise.
"""
import argparse
import dataclasses
import os
import json
import sys
from pathlib import Path

import scs_loop
import scs_method

import lb_lib as lb                    # noqa: E402   model names, gateways and keys of the SHC runs
import lb_skill_run                    # noqa: E402   make_call

DEFAULT_MODELS = ["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low", "openai:gpt-5.5"]


MAX_TOKENS = None                      # no reply-length limit for any model (see the module docstring)


def resume_set(path):
    """prompt_ids whose record ended with an answer or at the step cap; they are never run again."""
    path = Path(path)
    if not path.exists():
        return set()
    return {r["prompt_id"] for r in map(json.loads, open(path)) if r.get("ended") in ("answer", "step_cap")}


def env_prefixes(env_prefix):
    """The skill's environment prefix, and for a venv the base interpreter's prefix it points to (sf_run.env_binds)."""
    out = [os.path.realpath(env_prefix)]
    cfg = Path(env_prefix) / "pyvenv.cfg"
    if cfg.exists():
        home = next((l.split("=", 1)[1].strip() for l in cfg.read_text().splitlines() if l.strip().startswith("home")), None)
        if home:
            out.append(os.path.realpath(str(Path(home).parent)))
    return list(dict.fromkeys(out))


def run(method, use_name, skill_dir, models, effort="low", seed=0, step_cap=scs_loop.STEP_CAP, limit=0, label="", out=""):
    mod = scs_method.load_module(method)
    use = mod.USES[use_name]
    skill_dir = Path(skill_dir)
    solver = skill_dir.parent / "solver.json"           # written by scs_profile.collect for a profile skill
    env_prefix = json.loads(solver.read_text()).get("env_prefix") if solver.exists() else None
    if env_prefix:                                     # the skill's own environment, read-only, its bin first on PATH
        use = dataclasses.replace(use, mounts=[*use.mounts, *env_prefixes(env_prefix)],
                                  path_dirs=[Path(env_prefix) / "bin", *use.path_dirs])
    skill_dir = Path(skill_dir)
    cases = use.cases()[:limit] if limit else use.cases()
    out_dir = Path(out) if out else scs_method.METHODS / method / "runs" / (label or use_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir.parent / "MANIFEST.json"
    env = {"python": sys.executable, "step_cap": step_cap, "skill_env_prefix": env_prefix,
           "skill_manifest": json.loads(manifest.read_text()) if manifest.exists() else None}
    for spec_model in models:
        prov, model, _ = lb.parse_model(spec_model)
        key = lb.api_key(prov)
        out = out_dir / f"{skill_dir.name}_{lb.safe_name(spec_model)}_{effort}.jsonl"
        done = resume_set(out)
        print(f"{spec_model} / {skill_dir.name} / {use_name} -> {out}; {len(done)} done")
        call = lb_skill_run.make_call(spec_model, seed, key, None if effort == "default" else effort, 600, MAX_TOKENS)
        with open(out, "a") as f:
            for case in cases:
                pid = f"{case['prompt_id']}_{skill_dir.name}"
                if pid in done:
                    continue
                rec = scs_loop.run_case(skill_dir, case, use, call, step_cap=step_cap)
                last_usage = next((s["usage"] for s in reversed(rec["steps"]) if s.get("usage")), {})
                rec.update(prompt_id=pid, method=method, use=use_name, model=spec_model, served_model=model,
                           gateway=prov, seed=seed, reasoning_effort=effort, max_tokens=MAX_TOKENS,
                           env=env, usage={k: v for k, v in last_usage.items() if k != "raw_response"})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  {pid:40s} ok={rec['ok']!s:5s} steps={rec['n_steps']:2d} ended={rec['ended']:9s} "
                      f"fit_runs={len(rec['fit_runs'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--use", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-cap", type=int, default=scs_loop.STEP_CAP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="", help="folder for the records; without it, methods/<method>/runs/<label>")
    a = ap.parse_args()
    run(a.method, a.use, a.skill, a.models, a.effort, a.seed, a.step_cap, a.limit, a.label, a.out)


if __name__ == "__main__":
    main()
