"""sf_run — the solver models on a SkillFoundry-generated SHC skill, over the 27 cells of the simple skill arms.

    python sf_run.py --generator gpt-5.4|deepseek --format code_provided|code_written [--models ...] [--effort low]
                     [--limit N] [--seed 0] [--max-tokens N] [--step-cap 12]
    python sf_run.py --show <reply file> [--cell <prompt_id>]

The loop is llm_band_sc's run_cell, unchanged: the same SYSTEM text, the same bash sandbox, step cap, placeholders and
answer check. Three things differ, each set by the plan:
  G5   every top-level file and folder of the generated skill is bound read-only into the working directory
       (the simple loop binds SKILL.md, references/ and scripts/ only);
  G8c  the generated environment's prefix is bound read-only at its own path and its bin/ is first on PATH
       (the simple loop puts the satellite interpreter there);
  G8d  a skill that names no environment (env_prefix null) gets the simple loop's own interpreter and PATH, unchanged;
  G16  the task message has no skill line, the same for both generators and both formats;
  OL   the model sees each command's output whole (the simple loop shows 6,000 characters), departure 5.
Token caps follow row 13: 32,000 for DeepSeek, 16,000 for the other models. Records go to runs/.
"""
import argparse
import contextlib
import functools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LB_DIR = HERE.parent / "llm_band_sc"
sys.path.insert(0, str(LB_DIR))

import lb_guard                      # noqa: E402
import lb_lib as lb                  # noqa: E402
import lb_sandbox                    # noqa: E402
import lb_skill_loop as L            # noqa: E402
import lb_skill_run as R             # noqa: E402
from sf_collect import tree_sha     # noqa: E402

GENERATORS = ("gpt-5.4", "deepseek")
FORMATS = ("code_provided", "code_written")
DEFAULT_MODELS = ["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low", "openai:gpt-5.5"]
RESERVED = ("data.csv", "answer.json")
OUT_LIMIT = sys.maxsize          # decision OL, Jun "OL. C." (2026-10-14): no limit on the output shown to the model


def max_tokens_for(model):
    """Row 13: 32,000 for DeepSeek, 16,000 for the other solver models."""
    return 32000 if "deepseek" in model.lower() else 16000


def task_text_profile(skill_dir):
    """G16: the simple loop's task text without its skill line."""
    return (f"Predict this land parcel's mean band values for the next month, P{lb.TARGET:02d} = {lb.CALENDAR[lb.TARGET]}, "
            "from its own past months.\nThe working directory holds a skill for this task. "
            "Its instructions are in SKILL.md; read that file first and follow it. Reply with one ```bash block per turn; "
            "it is run by bash in the working directory and its output comes back in the next message. "
            "Finish by writing answer.json as the skill describes.")


def env_binds(env_prefix):
    """The environment prefix, and for a venv the base interpreter's prefix it points to, read-only at their paths."""
    prefixes = [os.path.realpath(env_prefix)]
    cfg = Path(env_prefix) / "pyvenv.cfg"
    if cfg.exists():
        home = next((l.split("=", 1)[1].strip() for l in cfg.read_text().splitlines() if l.strip().startswith("home")), None)
        if home:
            prefixes.append(os.path.realpath(str(Path(home).parent)))
    out = []
    for p in dict.fromkeys(prefixes):
        if not any(p == t or p.startswith(t + os.sep) for t in ("/usr", "/lib", "/lib64", "/etc", "/bin")):
            out += ["--ro-bind", p, p]
    return out


class ProfileCell(L.Cell):
    """L.Cell with the G5 mounts and the G8c environment (G8d: none); everything else is the parent's."""

    def __init__(self, skill_dir, cell_panel, sensor, site, env_prefix):
        self.skill = Path(skill_dir)
        self.entries = sorted(p.name for p in self.skill.iterdir() if p.name != "__pycache__")
        clash = [n for n in RESERVED if n in self.entries]
        assert not clash, f"skill folder holds {clash} at its top level"
        self.env_prefix = Path(env_prefix) if env_prefix else None
        self.env_bin = str(self.env_prefix / "bin") if env_prefix else L.PY_DIR
        self.sensor, self.site = sensor, site
        self.bands = [lb.disp(b) for b in lb.BANDS[sensor]]
        self.dir = Path(tempfile.mkdtemp(prefix="sf_skill_"))
        assert "P10" not in set(cell_panel["period_id"]) and cell_panel["period_id"].nunique() == 9
        d = cell_panel.assign(band=[lb.disp(b) for b in cell_panel["band"]], seq=cell_panel["period_id"].str[1:].astype(int))
        d = d.sort_values(["seq", "band"])[["band", "period_id", "month", "mean_value"]]
        d.to_csv(self.dir / "data.csv", index=False)
        for n in self.entries:                          # mount points for the read-only binds
            (self.dir / n).mkdir() if (self.skill / n).is_dir() else (self.dir / n).touch()
        self.ref_re = re.compile(r"\bSKILL\.md\b|(?<![\w./-])(?:" + "|".join(re.escape(n) for n in self.entries) + r")(?:/[\w./-]+)?")
        self.refs_read, self.data_shown, self.python_runs, self.fit_records = [], False, 0, []

    def do_shell(self, script):
        """L.Cell.do_shell with the G5 binds and the G8c PATH."""
        for m in self.ref_re.finditer(script):
            self.refs_read.append(m.group(0))
        if "show_data.py" in script or "data.csv" in script:
            self.data_shown = True
        if re.search(r"\bpython3?\b", script):
            self.python_runs += 1
        binds = [a for n in self.entries for a in ("--ro-bind", str(self.skill / n), f"/work/{n}")]
        extra = env_binds(self.env_prefix) if self.env_prefix else []
        cmd = [lb_sandbox.BWRAP, *lb_sandbox.MOUNTS, *extra, "--bind", str(self.dir), "/work", *binds,
               "--chdir", "/work", "--unshare-all", "--die-with-parent", "--new-session", "--",
               lb_sandbox.BASH, "-c", script]
        env = {"PATH": f"{self.env_bin}:/usr/bin:/bin", "HOME": "/work", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=self.dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=L._limits())
        try:
            out, err = p.communicate(timeout=L.CMD_TIMEOUT)
            code = p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), L.signal.SIGKILL)
            out, err = p.communicate()
            code, err = -9, err + f"\n[killed: exceeded {L.CMD_TIMEOUT} s]"
        if err.strip().startswith("bwrap:"):
            raise lb_sandbox.SandboxInfraError(err.strip().splitlines()[0])
        shown = []
        for line in out.splitlines():
            if line.startswith(L.FIT_MARK):
                try:
                    self.fit_records.append(json.loads(line[len(L.FIT_MARK):]))
                except json.JSONDecodeError:
                    pass
            else:
                shown.append(line)
        text = "\n".join(shown) + (("\n" + err) if err.strip() else "")
        if code != 0:
            text += f"\n[exit status {code}]"
        return text.strip("\n"), {"exit": code, "seconds": round(time.time() - t0, 1)}

    def files(self):
        return {n: h for n, h in super().files().items() if n not in self.entries}


@contextlib.contextmanager
def profile_loop(env_prefix):
    """Run L.run_cell with ProfileCell, the G16 task text and OUT_LIMIT; the simple loop's names are restored afterwards."""
    saved = L.Cell, L.task_text, L.OUT_LIMIT
    L.Cell, L.task_text, L.OUT_LIMIT = functools.partial(ProfileCell, env_prefix=env_prefix), task_text_profile, OUT_LIMIT
    try:
        yield
    finally:
        L.Cell, L.task_text, L.OUT_LIMIT = saved


def run_cell(skill_dir, env_prefix, cell_panel, sensor, site, call_fn, step_cap=L.STEP_CAP):
    with profile_loop(env_prefix):
        return L.run_cell(skill_dir, cell_panel, sensor, site, call_fn, step_cap=step_cap)


def load_solver(g, fmt):
    """What sf_collect built for this generator; the mounted folder must still hash to what was collected."""
    sj = HERE / "skills" / g / "solver.json"
    if not sj.exists():
        raise SystemExit(f"{sj} is missing: run `python sf_collect.py {g}` first")
    s = json.loads(sj.read_text())
    skill_dir = HERE / "skills" / g / fmt
    got = tree_sha(skill_dir)
    if got != s[f"{fmt}_sha"]:
        raise SystemExit(f"{skill_dir} changed since sf_collect ({got[:12]} != {s[fmt + '_sha'][:12]})")
    return skill_dir, s["env_prefix"], got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=GENERATORS)
    ap.add_argument("--format", choices=FORMATS)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--effort", default=lb.REASONING_EFFORT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = row 13 (32,000 DeepSeek, 16,000 others)")
    ap.add_argument("--step-cap", type=int, default=L.STEP_CAP)
    ap.add_argument("--show", default="")
    ap.add_argument("--cell", default="")
    a = ap.parse_args()
    if a.show:
        return R.show(a.show, a.cell or None)
    if not (a.generator and a.format):
        ap.error("--generator and --format are required")
    skill_dir, env_prefix, skill_sha = load_solver(a.generator, a.format)
    exp = f"profile_{a.generator}_{a.format}"
    effort = None if a.effort == "default" else a.effort
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    base = [json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")]
    if a.limit:
        keep, seen = [], {}
        for p in base:
            if seen.get(p["sensor"], 0) < a.limit:
                keep.append(p); seen[p["sensor"]] = seen.get(p["sensor"], 0) + 1
        base = keep
    panel9 = lb_guard.training_panel()
    env_py = subprocess.run([str(Path(env_prefix) / "bin" / "python") if env_prefix else sys.executable, "-c",
                             "import sys, importlib.metadata as m; print(sys.version.split()[0]); "
                             "print({d.metadata['Name']: d.version for d in m.distributions()})"],
                            capture_output=True, text=True)
    for spec_model in a.models:
        prov, model, _ = lb.parse_model(spec_model)
        key = lb.api_key(prov)
        mt = a.max_tokens or max_tokens_for(spec_model)
        env = {"skill_dir": str(skill_dir), "skill_sha": skill_sha, "env_prefix": env_prefix,
               "python": str(Path(env_prefix) / "bin" / "python") if env_prefix else sys.executable,
               "env_python": env_py.stdout.strip(), "max_tokens": mt, "step_cap": a.step_cap}
        out = out_dir / f"replies_{exp}_{lb.safe_name(spec_model)}_{a.effort}.jsonl"
        done = R.resume_set(out)
        print(f"{spec_model} / {exp} (effort {a.effort}, max tokens {mt}, step cap {a.step_cap}) -> {out.name}; {len(done)} done")
        call = R.make_call(spec_model, a.seed, key, effort, a.timeout, mt)
        with open(out, "a") as f:
            for p in base:
                pid = p["prompt_id"][:-4] + "_" + exp
                if pid in done:
                    continue
                cell = lb_guard.cell_panel(panel9, p["sensor"], p["target_site"])
                rec = run_cell(skill_dir, env_prefix, cell, p["sensor"], p["target_site"], call, step_cap=a.step_cap)
                last_usage = next((s["usage"] for s in reversed(rec["steps"]) if s.get("usage")), {})
                rec.update(prompt_id=pid, experiment=exp, generator=a.generator, format=a.format, model=spec_model,
                           served_model=model, gateway=prov, seed=a.seed, reasoning_effort=a.effort, env=env,
                           usage={k: v for k, v in last_usage.items() if k != "raw_response"},
                           reply=next((s["reply"] for s in reversed(rec["steps"]) if s.get("reply")), ""))
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"  {pid:52s} ok={rec['ok']!s:5s} steps={rec['n_steps']:2d} ended={rec['ended']:9s} "
                      f"python_runs={rec['python_runs']} refs={len(rec['refs_read'])} data={rec['data_shown']}")


if __name__ == "__main__":
    main()
