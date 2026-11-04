"""scs_loop — one case: a model operates a skill folder through a bash shell (the run stage of every pipeline).

The loop of the SHC runs (llm_band_sc/lb_skill_loop.py: the same system text, command parsing, sandbox limits, the
fit marker the model never sees, the empty-reply and empty-output placeholders) with the case folder of the profile
runs (skillfoundry_profile/sf_run.py ProfileCell: every top-level entry of the skill is bound read-only). What the
model is given is the use's, decided when a pipeline is used (scs_loop.Use):

    files       the data files written into the case folder
    task_text   the first user message
    answer      the check of answer.json
    mounts      further read-only paths inside the box (an R installation), path_dirs first on PATH
    out_limit   characters of a command's output shown to the model (all of it is stored); none by default

    run_case(skill_dir, case, use, call_fn, step_cap=12) -> the full record of the case
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
LB = HERE.parent / "llm_band_sc"
if str(LB) not in sys.path:
    sys.path.insert(0, str(LB))

import lb_sandbox                      # noqa: E402   the box: bwrap, the system mounts, the interpreter's prefix
import lb_skill_loop as L              # noqa: E402   SYSTEM, parse_command, limits, FIT_MARK, frontmatter

STEP_CAP = L.STEP_CAP
SYSTEM = L.SYSTEM
FIT_MARK = L.FIT_MARK
RESERVED = ("answer.json",)
FITLOG_DIR = "/fitlog"     # inside the box: a folder of the harness, outside the case folder; FIT_LOG names its file


@dataclass
class Use:
    """How a pipeline is used: which cases, what each case folder holds, what the model is told, how its answer.json
    is read. cases() -> [{"prompt_id", "id" (fields copied into the record), "files" {name: text}, "ctx"}]."""
    name: str
    cases: Callable
    task_text: Callable                   # (skill_dir, case) -> first user message
    answer: Callable                      # (case, answer dict) -> (answer or None, problem or None)
    mounts: list = field(default_factory=list)
    path_dirs: list = field(default_factory=list)
    out_limit: int = sys.maxsize
    nofile: int = 64                      # open-file limit of each command in the box; R needs at least 167
    dry_answer: Optional[Callable] = None  # (case, fit record) -> answer dict, for the scripted dry run only
    profile_task: str = ""                # profile request: "Design a skill for <profile_task> with ..."
    profile_data: str = ""                # profile request: "... works in a directory holding <profile_data>"
    profile_answer: str = ""              # profile request: "It <profile_answer>"
    fit_values: Optional[Callable] = None  # fit record -> its predictions in the answer's shape (step comparison)


def _limits(nofile):
    """The command limits of lb_skill_loop._limits (CPU, memory, file size) with the use's open-file limit, whose
    default is lb_skill_loop's 64."""
    import resource

    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (L.CMD_TIMEOUT + 5, L.CMD_TIMEOUT + 5))
        resource.setrlimit(resource.RLIMIT_AS, (L.MEM_MB << 20, L.MEM_MB << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        resource.setrlimit(resource.RLIMIT_FSIZE, (L.FSIZE_MB << 20, L.FSIZE_MB << 20))
    return f


def _binds(paths):
    out = []
    for p in dict.fromkeys(os.path.realpath(str(x)) for x in paths):
        if not any(p == t or p.startswith(t + os.sep) for t in ("/usr", "/lib", "/lib64", "/etc", "/bin")):
            out += ["--ro-bind", p, p]
    return out


class Case:
    """The case folder and the shell on it."""

    def __init__(self, skill_dir, case, use):
        self.skill, self.case, self.use = Path(skill_dir), case, use
        self.entries = sorted(p.name for p in self.skill.iterdir() if p.name != "__pycache__")
        clash = [n for n in list(case["files"]) + list(RESERVED) if n in self.entries]
        assert not clash, f"skill folder holds {clash} at its top level"
        self.dir = Path(tempfile.mkdtemp(prefix="scs_case_"))
        for name, text in case["files"].items():
            (self.dir / name).write_text(text)
        for n in self.entries:                          # mount points for the read-only binds
            (self.dir / n).mkdir() if (self.skill / n).is_dir() else (self.dir / n).touch()
        self.ref_re = re.compile(r"\bSKILL\.md\b|(?<![\w./-])(?:" + "|".join(re.escape(n) for n in self.entries)
                                 + r")(?:/[\w./-]+)?")
        self.data_re = re.compile("|".join(re.escape(n) for n in case["files"]) + r"|show_data")
        self.refs_read, self.data_shown, self.python_runs, self.r_runs, self.fit_records = [], False, 0, 0, []
        # Fit records reach the harness two ways: a FIT_MARK line on stdout, or a line appended to the file FIT_LOG
        # names. The file does not depend on what the command does with its output (a pipe, grep, a redirect).
        self.fitlog = Path(tempfile.mkdtemp(prefix="scs_fitlog_"))
        (self.fitlog / "fits.jsonl").touch()
        self.fitlog_read = 0

    def do_shell(self, script):
        """Run one bash block inside the box. Returns (text shown to the model, meta)."""
        for m in self.ref_re.finditer(script):
            self.refs_read.append(m.group(0))
        if self.data_re.search(script):
            self.data_shown = True
        if re.search(r"\bpython3?\b", script):
            self.python_runs += 1
        if re.search(r"\bRscript\b", script):
            self.r_runs += 1
        binds = [a for n in self.entries for a in ("--ro-bind", str(self.skill / n), f"/work/{n}")]
        cmd = [lb_sandbox.BWRAP, *lb_sandbox.MOUNTS, *_binds(self.use.mounts), "--bind", str(self.dir), "/work", *binds,
               "--bind", str(self.fitlog), FITLOG_DIR, "--chdir", "/work", "--unshare-all", "--die-with-parent", "--new-session", "--",
               lb_sandbox.BASH, "-c", script]
        path = ":".join([*map(str, self.use.path_dirs), L.PY_DIR, "/usr/bin", "/bin"])
        env = {"PATH": path, "HOME": "/work", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8", "FIT_LOG": f"{FITLOG_DIR}/fits.jsonl"}
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=self.dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=_limits(self.use.nofile))
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
            if line.startswith(FIT_MARK):
                try:
                    self.fit_records.append(json.loads(line[len(FIT_MARK):]))
                except json.JSONDecodeError:
                    pass
            else:
                shown.append(line)
        raw = (self.fitlog / "fits.jsonl").read_bytes() if (self.fitlog / "fits.jsonl").exists() else b""
        for line in raw[self.fitlog_read:].decode(errors="replace").splitlines():
            try:
                self.fit_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.fitlog_read = len(raw)
        text = "\n".join(shown) + (("\n" + err) if err.strip() else "")
        if code != 0:
            text += f"\n[exit status {code}]"
        return text.strip("\n"), {"exit": code, "seconds": round(time.time() - t0, 1)}

    def answer(self):
        p = self.dir / "answer.json"
        if not p.exists():
            return None, None
        try:
            a = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return None, f"answer.json is not valid JSON: {e}"
        return self.use.answer(self.case, a)

    def programs(self):
        """Every program the model wrote in the case folder (a .py or .R file), by name."""
        out = {}
        for p in sorted(self.dir.iterdir()):
            if p.is_file() and p.suffix in (".py", ".R") and p.name not in self.entries:
                try:
                    out[p.name] = p.read_text()
                except UnicodeDecodeError:
                    pass
        return out

    def files(self):
        out = {}
        for n in sorted(os.listdir(self.dir)):
            p = self.dir / n
            if p.is_file() and n not in self.entries:
                out[n] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    def close(self):
        L.shutil.rmtree(self.dir, ignore_errors=True)
        L.shutil.rmtree(self.fitlog, ignore_errors=True)


def run_case(skill_dir, case, use, call_fn, step_cap=STEP_CAP):
    """call_fn(messages) -> (text, usage). Returns the full record of the case."""
    c = Case(skill_dir, case, use)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": use.task_text(skill_dir, case)}]
    steps, ended, answer, t0 = [], "step_cap", None, time.time()
    try:
        for k in range(1, step_cap + 1):
            try:
                text, usage = call_fn(messages)
            except Exception as e:
                steps.append({"step": k, "error": repr(e)[:500]}); ended = "call_error"; break
            cmd = L.parse_command(text)
            meta = {}
            if cmd["kind"] == "error":
                output = cmd["message"]
            else:
                n_fits = len(c.fit_records)
                output, meta = c.do_shell(cmd["text"])
                for fr in c.fit_records[n_fits:]:          # the step each fit was run at
                    fr["step"] = k
                if cmd["n_blocks"] > 1:
                    output += f"\n[{cmd['n_blocks'] - 1} further command block(s) in your reply were ignored: one block per turn]"
            answer, problem = c.answer()
            if cmd["kind"] == "bash" and problem:
                output += "\n" + problem
            steps.append({"step": k, "reply": text, "usage": usage, "command": cmd, "output": output, **meta})
            if answer is not None:
                ended = "answer"; break
            shown = output[:use.out_limit] + ("\n[output truncated]" if len(output) > use.out_limit else "")
            messages += [{"role": "assistant", "content": text if text.strip() else "(empty reply)"},
                         {"role": "user", "content": shown if shown.strip() else "(no output)"}]
        return {"skill": Path(skill_dir).name, **case["id"], "steps": steps, "n_steps": len(steps), "ended": ended,
                "answer": answer, "values": answer["prediction"] if answer else None,
                "rationale": answer["rationale"] if answer else None, "ok": answer is not None,
                "fit_runs": c.fit_records, "program": c.programs().get("predict.py"), "programs": c.programs(),
                "refs_read": sorted(set(c.refs_read)), "data_shown": c.data_shown, "python_runs": c.python_runs,
                "r_runs": c.r_runs, "files": c.files(),
                "multi_block_replies": sum(1 for s in steps if s.get("command", {}).get("n_blocks", 0) > 1),
                "seconds": round(time.time() - t0, 1)}
    finally:
        c.close()
