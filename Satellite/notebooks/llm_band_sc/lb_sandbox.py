"""lb_sandbox — run a model-written program (arm C) on one cell's nine months, isolated.

Isolation is bubblewrap (bwrap), mandatory: a fresh mount namespace with /usr, /lib, /lib64, /etc read-only,
/proc and /dev provided, an empty /tmp, and ONE writable directory holding panel.csv (nine periods of one cell),
harness.py (ours) and user.py (the model's source, byte for byte). Nothing else of the filesystem exists inside,
so no path reaches the repository or the full feature table. --unshare-all removes the network. Python runs with
-I (no PYTHONPATH, no user site). Resource limits: CPU, address space, open files, and file size 0 so the child
cannot create a non-empty file. A static check refuses disallowed imports, file access and path-like strings
before anything runs, and returns the reason so the model can repair. After the run the directory inventory
must equal what we wrote.

    static_check(source)                        -> None or SandboxError naming line and reason
    run_user_code(source, cell_panel, ...)      -> {ok, predictions, stdout, stderr, seconds, error, source_sha256}
"""
import ast
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import lb_lib as lb

PY = sys.executable
BWRAP = shutil.which("bwrap")
ALLOWED_ROOTS = {"numpy", "scipy", "pandas", "math", "statistics", "itertools", "functools", "collections",
                 "json", "typing", "dataclasses", "warnings", "operator", "random", "numbers", "decimal", "fractions"}
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "input", "breakpoint",
                "getattr", "setattr", "delattr", "vars", "memoryview"}
BANNED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "urllib", "requests", "http", "ctypes",
                  "importlib", "multiprocessing", "tempfile", "io", "builtins", "signal", "threading", "pickle",
                  "glob", "codecs", "mmap", "select", "asyncio", "concurrent"}
FILE_ATTRS = ("read_csv", "read_table", "read_excel", "read_json", "read_pickle", "read_parquet", "read_hdf",
              "read_feather", "loadtxt", "genfromtxt", "fromfile", "load", "memmap", "to_csv", "to_pickle",
              "to_json", "to_excel", "save", "savetxt", "savez")
PATH_EXTS = (".csv", ".tif", ".tiff", ".npy", ".npz", ".pkl", ".json", ".txt", ".parquet", ".h5", ".xlsx")
MOUNTS = ["--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
          "--ro-bind", "/bin", "/bin", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
BASH = shutil.which("bash")
# The running interpreter's own installation (a conda env may live anywhere, e.g. /data/.../envs/satellite) must
# exist inside the namespace or bwrap cannot exec it. It is bound read-only, and it must not contain the project.
_PREFIXES = sorted({os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})
_PROJECT = os.path.realpath(str(lb.LB))
for _p in _PREFIXES:
    assert not _PROJECT.startswith(_p + os.sep), f"interpreter prefix {_p} contains the project; refusing to bind it"
    if not any(_p.startswith(top + os.sep) or _p == top for top in ("/usr", "/lib", "/lib64", "/etc")):
        MOUNTS += ["--ro-bind", _p, _p]


class SandboxInfraError(RuntimeError):
    """The sandbox itself failed to start (not the model's program). Never returned to the model as a repair."""

HARNESS = r'''
import json, sys
import numpy, scipy, scipy.optimize, scipy.linalg, pandas as pd
ALLOWED = %(allowed)r
_loaded = {m.split(".")[0] for m in sys.modules}
import importlib.abc
class _Gate(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root not in ALLOWED and root not in _loaded:
            raise ImportError("import of %%s is not allowed in the sandbox" %% name)
        return None
sys.meta_path.insert(0, _Gate())
panel = pd.read_csv("panel.csv")
with open("user.py") as f:
    src = f.read()
ns = {"__name__": "user_program", "np": numpy, "pd": pd}
exec(compile(src, "user.py", "exec"), ns)
if "predict" not in ns or not callable(ns["predict"]):
    raise SystemExit("user.py defines no callable `predict`")
sensor, site = str(panel["sensor"].iloc[0]), str(panel["site_id"].iloc[0])
out = ns["predict"](panel.copy(), sensor, site)
try:
    out = {str(k): float(v) for k, v in dict(out).items()}
except Exception as e:
    raise SystemExit("predict must return a dict of band -> number: %%r" %% (e,))
print("@@RESULT@@" + json.dumps({"predictions": out}))
'''


class SandboxError(RuntimeError):
    pass


def static_check(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxError(f"syntax error at line {e.lineno}: {e.msg}")
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_ROOTS:
                    problems.append(f"line {node.lineno}: import {a.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_ROOTS:
                problems.append(f"line {node.lineno}: import from {node.module} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES | BANNED_MODULES:
            problems.append(f"line {node.lineno}: `{node.id}` is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"line {node.lineno}: attribute {node.attr} is not allowed")
            elif node.attr in FILE_ATTRS:
                problems.append(f"line {node.lineno}: file access `{node.attr}` is not allowed; the data is the `panel` argument")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s.startswith(("/", "./", "../", "~")) or ".." in s or s.lower().endswith(PATH_EXTS) or s.startswith("__"):
                problems.append(f"line {node.lineno}: path-like or dunder string {s!r} is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "predict" for n in tree.body):
        problems.append("no top-level `def predict(panel, sensor, site_id)` found")
    if problems:
        raise SandboxError("; ".join(problems[:12]))


def _limits(cpu_s, mem_mb):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    return f


def _inventory(d):
    return {n: hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() for n in sorted(os.listdir(d))}


def run_user_code(source, cell_panel, timeout_s=120, mem_mb=2048):
    """cell_panel: one cell's nine periods (lb_guard.cell_panel). Returns a record; never raises for the model's errors."""
    if BWRAP is None:
        raise SandboxError("bwrap is not available on this host; arm C requires it")
    rec = {"ok": False, "predictions": None, "stdout": "", "stderr": "", "seconds": 0.0, "error": None,
           "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "static_ok": False}
    try:
        static_check(source)
    except SandboxError as e:
        rec["error"] = f"static check: {e}"
        return rec
    rec["static_ok"] = True
    assert "P10" not in set(cell_panel["period_id"]) and len(cell_panel["period_id"].unique()) == 9
    d = tempfile.mkdtemp(prefix="lb_sbx_")
    try:
        csv = cell_panel[["site_id", "sensor", "band", "period_id", "month", "mean_value"]].copy()
        csv["band"] = [lb.disp(b) for b in csv["band"]]         # the names the prompt gives the model
        csv.to_csv(os.path.join(d, "panel.csv"), index=False)
        with open(os.path.join(d, "harness.py"), "w") as f:
            f.write(HARNESS % {"allowed": sorted(ALLOWED_ROOTS)})
        with open(os.path.join(d, "user.py"), "w") as f:
            f.write(source)
        before = _inventory(d)
        cmd = [BWRAP, *MOUNTS, "--bind", d, "/work", "--chdir", "/work", "--unshare-all", "--die-with-parent",
               "--new-session", "--", PY, "-I", "harness.py"]
        env = {"PATH": "/usr/bin:/bin", "HOME": "/work", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL",   # no OpenMP runtime: it wants a
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}              # shared-memory file, which FSIZE=0 forbids
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True,
                             preexec_fn=_limits(timeout_s + 5, mem_mb))
        try:
            out, err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            out, err = p.communicate()
            rec.update(stdout=out[-4000:], stderr=err[-4000:], seconds=round(time.time() - t0, 1),
                       error=f"timeout after {timeout_s}s")
            return rec
        rec.update(stdout=out[-4000:], stderr=err[-4000:], seconds=round(time.time() - t0, 1))
        after = _inventory(d)
        if after != before:
            rec["error"] = f"sandbox directory changed: {sorted(set(after) ^ set(before)) or 'contents rewritten'}"
            return rec
        if p.returncode != 0:
            last = err.strip().splitlines()[-1] if err.strip() else "no stderr"
            if last.startswith("bwrap:"):
                raise SandboxInfraError(f"sandbox failed to launch: {last}")
            rec["error"] = f"exit {p.returncode}: {last}"
            return rec
        line = next((l for l in out.splitlines() if l.startswith("@@RESULT@@")), None)
        if line is None:
            rec["error"] = "no result line on stdout"
            return rec
        rec["predictions"] = json.loads(line[len("@@RESULT@@"):])["predictions"]
        rec["ok"] = True
        return rec
    finally:
        shutil.rmtree(d, ignore_errors=True)
