"""lb_skill_loop — one cell of a skill arm: a model operates a skill directory through a real bash shell.

The working directory holds data.csv (this parcel's nine months, nothing else) and, read-only, the skill's
SKILL.md, references/ and scripts/. Each reply is one ```bash block, executed as written by bash inside
bubblewrap: only that directory and the interpreter are visible, the skill files cannot be written, there is
no network, and CPU, memory, file size and wall clock are capped. If a reply holds several blocks, the first
is executed and the model is told the rest were ignored. The cell ends when a valid answer.json exists, or at
the step cap. Every turn — reply text, full raw response, command, output — is recorded; every fit.py run is
captured from a marked stdout line the model does not see.
"""
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import lb_lib as lb
import lb_sandbox

STEP_CAP = 12
CMD_TIMEOUT = 120
MEM_MB = 2048
FSIZE_MB = 16
OUT_LIMIT = 6000                       # characters of a command's output returned to the model (all of it is stored)
FENCE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.S)
FIT_MARK = "@@FIT@@"
PY_DIR = str(Path(lb_sandbox.PY).parent)

SYSTEM = ("You are operating a bash shell in a working directory that contains a skill: files with instructions "
          "(SKILL.md), reference documents and scripts for one task. Each reply you send is executed as one bash "
          "command block, and its output is returned to you in the next message. Nothing outside the working "
          "directory is visible and there is no network. Read the skill's instructions and follow them.")


def frontmatter(skill_dir):
    text = (Path(skill_dir) / "SKILL.md").read_text()
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    fm, key, out = m.group(1), None, {}
    for line in fm.splitlines():
        if line.startswith((" ", "\t")) and key:
            out[key] += " " + line.strip()
        else:
            key, _, val = line.partition(":")
            key = key.strip(); out[key] = val.strip()
    return out


def task_text(skill_dir):
    fm = frontmatter(skill_dir)
    return (f"Predict this land parcel's mean band values for the next month, P{lb.TARGET:02d} = {lb.CALENDAR[lb.TARGET]}, "
            "from its own past months.\nThe working directory holds a skill for this task:\n"
            f"  {fm['name']} — {fm['description']}\n"
            "Its instructions are in SKILL.md; read that file first and follow it. Reply with one ```bash block per turn; "
            "it is run by bash in the working directory and its output comes back in the next message. "
            "Finish by writing answer.json as the skill describes.")


# Native tool-call syntaxes some models emit instead of a fence (2026-09-24 runs): DeepSeek's DSML invoke form and
# GLM's <tool_call> form. Both carry one bash command and are accepted as the block they are; the turn records which
# syntax was used. A fence left unclosed and followed by such tags is read up to the first tag.
DSML_CMD = re.compile(r'<｜DSML｜ parameter name="command"[^>]*>(.*?)</｜DSML｜ parameter>', re.S)
GLM_CMD = re.compile(r"<arg_value>(.*?)</arg_value>", re.S)
OPEN_FENCE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*)$", re.S)
NATIVE_TAGS = re.compile(r"</?(?:｜DSML｜[^>]*|tool_call|arg_key|arg_value)>")


def parse_command(text):
    blocks, syntax = FENCE.findall(text), "fence"
    if not blocks:
        blocks, syntax = DSML_CMD.findall(text), "dsml"
    if not blocks:
        blocks, syntax = GLM_CMD.findall(text), "glm_tool_call"
    if not blocks:
        m = OPEN_FENCE.search(text)
        if m:
            blocks, syntax = [NATIVE_TAGS.split(m.group(1))[0]], "unclosed_fence"
    if not blocks or not blocks[0].strip():
        return {"kind": "error", "n_blocks": 0, "syntax": "none",
                "message": "No ```bash block found. Reply with one ```bash block; it is run by bash in the working directory."}
    body = NATIVE_TAGS.sub("", blocks[0]).strip("\n")
    return {"kind": "bash", "n_blocks": len(blocks), "text": body, "syntax": syntax}


def _limits():
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (CMD_TIMEOUT + 5, CMD_TIMEOUT + 5))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_MB << 20, MEM_MB << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_MB << 20, FSIZE_MB << 20))
    return f


class Cell:
    """The working directory of one cell and the shell on it."""

    def __init__(self, skill_dir, cell_panel, sensor, site):
        self.skill = Path(skill_dir)
        self.sensor, self.site = sensor, site
        self.bands = [lb.disp(b) for b in lb.BANDS[sensor]]
        self.dir = Path(tempfile.mkdtemp(prefix="lb_skill_"))
        assert "P10" not in set(cell_panel["period_id"]) and cell_panel["period_id"].nunique() == 9
        d = cell_panel.assign(band=[lb.disp(b) for b in cell_panel["band"]], seq=cell_panel["period_id"].str[1:].astype(int))
        d = d.sort_values(["seq", "band"])[["band", "period_id", "month", "mean_value"]]
        d.to_csv(self.dir / "data.csv", index=False)
        (self.dir / "SKILL.md").touch()                    # mount points for the read-only binds
        (self.dir / "references").mkdir()
        (self.dir / "scripts").mkdir()
        self.refs_read, self.data_shown, self.python_runs, self.fit_records = [], False, 0, []

    def do_shell(self, script):
        """Run one bash block inside the sandbox. Returns (text shown to the model, meta)."""
        for m in re.finditer(r"\bSKILL\.md\b|\breferences/[\w./-]+", script):
            self.refs_read.append(m.group(0))
        if "show_data.py" in script or "data.csv" in script:
            self.data_shown = True
        if re.search(r"\bpython3?\b", script):
            self.python_runs += 1
        cmd = [lb_sandbox.BWRAP, *lb_sandbox.MOUNTS, "--bind", str(self.dir), "/work",
               "--ro-bind", str(self.skill / "SKILL.md"), "/work/SKILL.md",
               "--ro-bind", str(self.skill / "references"), "/work/references",
               "--ro-bind", str(self.skill / "scripts"), "/work/scripts",
               "--chdir", "/work", "--unshare-all", "--die-with-parent", "--new-session", "--",
               lb_sandbox.BASH, "-c", script]
        env = {"PATH": f"{PY_DIR}:/usr/bin:/bin", "HOME": "/work", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=self.dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=_limits())
        try:
            out, err = p.communicate(timeout=CMD_TIMEOUT)
            code = p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            out, err = p.communicate()
            code, err = -9, err + f"\n[killed: exceeded {CMD_TIMEOUT} s]"
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
        if not isinstance(a, dict) or not isinstance(a.get("prediction"), dict):
            return None, 'answer.json must be an object with a "prediction" object and a "rationale" string'
        pred = a["prediction"]
        if any(b not in pred for b in self.bands):         # never name the bands: a model that has not looked at
            return None, ("answer.json: prediction must give a number for every band of this parcel, "   # the data
                          "under the band names the parcel's data uses")                                # must not
        try:                                                                                            # learn them
            vals = {b: float(pred[b]) for b in self.bands}                                              # here
        except (TypeError, ValueError) as e:
            return None, f"answer.json: every prediction must be a number ({e})"
        if any(v != v for v in vals.values()):
            return None, "answer.json: a prediction is NaN"
        return {"prediction": vals, "rationale": str(a.get("rationale", ""))}, None

    def program(self):
        p = self.dir / "predict.py"
        try:
            return p.read_text() if p.exists() else None
        except UnicodeDecodeError:
            return None

    def files(self):
        out = {}
        for n in sorted(os.listdir(self.dir)):
            p = self.dir / n
            if p.is_file() and n != "SKILL.md":
                out[n] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def run_cell(skill_dir, cell_panel, sensor, site, call_fn, step_cap=STEP_CAP):
    """call_fn(messages) -> (text, usage). Returns the full record of the cell."""
    cell = Cell(skill_dir, cell_panel, sensor, site)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_text(skill_dir)}]
    steps, ended, answer, t0 = [], "step_cap", None, time.time()
    try:
        for k in range(1, step_cap + 1):
            try:
                text, usage = call_fn(messages)
            except Exception as e:
                steps.append({"step": k, "error": repr(e)[:500]}); ended = "call_error"; break
            cmd = parse_command(text)
            meta = {}
            if cmd["kind"] == "error":
                output = cmd["message"]
            else:
                output, meta = cell.do_shell(cmd["text"])
                if cmd["n_blocks"] > 1:
                    output += f"\n[{cmd['n_blocks'] - 1} further command block(s) in your reply were ignored: one block per turn]"
            answer, problem = cell.answer()
            if cmd["kind"] == "bash" and problem:
                output += "\n" + problem
            steps.append({"step": k, "reply": text, "usage": usage, "command": cmd, "output": output, **meta})
            if answer is not None:
                ended = "answer"; break
            # An empty reply (thinking exhausted the cap) is recorded as empty above, but the gateway refuses a
            # conversation containing an empty assistant message ("message content cannot be empty", the cause of
            # every DeepSeek 400 on 2026-09-24), so the turn is carried forward as a marked placeholder.
            messages += [{"role": "assistant", "content": text if text.strip() else "(empty reply)"},
                         {"role": "user", "content": output[:OUT_LIMIT] + ("\n[output truncated]" if len(output) > OUT_LIMIT else "")}]
        return {"skill": Path(skill_dir).name, "sensor": sensor, "site_id": site, "steps": steps, "n_steps": len(steps),
                "ended": ended, "answer": answer, "values": answer["prediction"] if answer else None,
                "rationale": answer["rationale"] if answer else None, "ok": answer is not None,
                "fit_runs": cell.fit_records, "program": cell.program(), "refs_read": sorted(set(cell.refs_read)),
                "data_shown": cell.data_shown, "python_runs": cell.python_runs, "files": cell.files(),
                "multi_block_replies": sum(1 for s in steps if s.get("command", {}).get("n_blocks", 0) > 1),
                "seconds": round(time.time() - t0, 1)}
    finally:
        cell.close()
