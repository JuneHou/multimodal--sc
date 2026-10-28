"""sf_collect — after one generator run, keep what SkillFoundry's pipeline wrote and build the two formats.

    python sf_collect.py gpt-5.4|deepseek

G14c  generated/<g>/files/ holds every path that differs from commit 4a8cc0c in work/<g>/ (tracked changes, new
      files, new git-ignored files such as the run folder under reports/), whether Codex or their check wrote it;
      generated/<g>/diff_tracked.patch is `git diff 4a8cc0c`; paper/, __pycache__/ and the files inside new
      slurm/envs/<family> prefixes are left out, and each such prefix gets its package list in generated/<g>/envs/.
G8c   the skill folder is the one new directory under skills/ holding a SKILL.md; the environment is the one
      slurm/envs/<family> prefix named in that SKILL.md or metadata.yaml. Anything else stops the work.
G8d   a skill that names no prefix runs on the simple loop's own interpreter: solver.json records env_prefix null.
G14d  the Codex session file (under codex_home/<g>/) whose first line names work/<g> as its working directory is copied to
      generated/<g>/codex_session.jsonl; when there is not exactly one, the work stops.
G7c   skills/<g>/code_provided/ is that folder as generated; skills/<g>/code_written/ is the same without scripts/
      and without every file under tests/ or examples/ whose text contains the name of a file in scripts/.
Everything is recorded in generated/<g>/manifest.json; skills/<g>/solver.json tells sf_run.py what to mount.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "4a8cc0c"
CONDA = "/usr/local/anaconda3/bin/conda"
ENV_RE = re.compile(r"slurm/envs/([A-Za-z0-9._-]+)")
LEFT_OUT = ("paper/",)
SESSIONS = HERE / "codex_home"                   # codex_home/<generator>/sessions/... (G15r)


class Stop(SystemExit):
    """A condition the plan says stops the work and goes to Jun."""


def git(work, *args):
    return subprocess.run(["git", "-C", str(work), *args], check=True, capture_output=True, text=True).stdout


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_sha(d):
    """One hash over every file's relative path and content, in sorted order."""
    h = hashlib.sha256()
    for p in sorted(q for q in Path(d).rglob("*") if q.is_file()):
        h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def changed(work):
    """(changed or new paths, deleted paths) relative to BASE, including git-ignored new files."""
    diff = [l for l in git(work, "diff", "--name-status", "--no-renames", BASE).splitlines() if l]
    deleted = sorted(l.split("\t", 1)[1] for l in diff if l.startswith("D"))
    modified = {l.split("\t", 1)[1] for l in diff if not l.startswith("D")}
    new = set(git(work, "ls-files", "--others", "--exclude-standard").splitlines())
    new |= set(git(work, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines())
    return sorted(modified | new), deleted


def session_files(work, root=SESSIONS):
    """G14d: the Codex session files whose session_meta names this working copy as cwd."""
    out = []
    for f in sorted(Path(root).rglob("rollout-*.jsonl")):
        with open(f) as h:
            first = json.loads(h.readline() or "{}")
        if first.get("type") == "session_meta" and first.get("payload", {}).get("cwd") == str(work):
            out.append(f)
    return out


def env_prefixes(paths):
    return sorted({m.group(0) for p in paths for m in [re.match(r"slurm/envs/[^/]+", p)] if m})


def package_list(prefix):
    if (prefix / "conda-meta").is_dir():
        cmd = [CONDA, "list", "-p", str(prefix), "--explicit"]
    else:
        cmd = [str(prefix / "bin" / "python"), "-m", "pip", "freeze"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    return " ".join(cmd), r.returncode, r.stdout + r.stderr


def code_written(provided, written):
    """G7c: drop scripts/ and the tests/examples files that name a file in scripts/."""
    names = sorted({p.name for p in (provided / "scripts").rglob("*") if p.is_file()}) if (provided / "scripts").is_dir() else []
    shutil.copytree(provided, written)
    removed = []
    if (written / "scripts").exists():
        removed += [str(p.relative_to(written)) for p in sorted((written / "scripts").rglob("*")) if p.is_file()]
        shutil.rmtree(written / "scripts")
    for sub in ("tests", "examples"):
        for p in sorted((written / sub).rglob("*")) if (written / sub).is_dir() else []:
            if p.is_file() and any(n.encode() in p.read_bytes() for n in names):
                removed.append(str(p.relative_to(written))); p.unlink()
    return names, removed


def main(g):
    work, out, skills = HERE / "work" / g, HERE / "generated" / g, HERE / "skills" / g
    if not (out / "run.log").exists():
        raise Stop(f"{out / 'run.log'} is missing: run `bash run_generator.sh {g}` first")
    if not any(l.startswith("finished ") for l in (out / "run.log").read_text().splitlines()):
        raise Stop(f"{out / 'run.log'} has no 'finished' line: the generator run has not ended")
    if (out / "manifest.json").exists() or skills.exists():
        raise Stop(f"{g} is collected already ({out / 'manifest.json'}, {skills})")
    paths, deleted = changed(work)
    envs = env_prefixes(paths)
    kept = [p for p in paths if not p.startswith(LEFT_OUT) and "__pycache__/" not in p
            and not any(p.startswith(e + "/") for e in envs)]
    for p in kept:
        dst = out / "files" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / p, dst, follow_symlinks=False)
    (out / "diff_tracked.patch").write_text(git(work, "diff", "--no-renames", BASE))
    env_lists = {}
    for e in envs:
        cmd, code, text = package_list(work / e)
        (out / "envs").mkdir(parents=True, exist_ok=True)
        (out / "envs" / (Path(e).name + ".txt")).write_text(f"# {cmd}\n# exit {code}\n{text}")
        env_lists[e] = {"command": cmd, "exit": code, "file": f"envs/{Path(e).name}.txt"}
    runs = sorted({"/".join(p.split("/")[:3]) for p in kept if p.startswith("reports/framework-runs/")})

    # G8c: exactly one new skill folder and exactly one environment it names
    new_skill_dirs = sorted({str(Path(p).parent) for p in paths if re.fullmatch(r"skills/[^/]+/[^/]+/SKILL\.md", p)
                             and p not in git(work, "ls-tree", "-r", "--name-only", BASE, "skills").split()})
    problems = []
    if len(new_skill_dirs) != 1:
        problems.append(f"expected one new skill folder under skills/, found {new_skill_dirs}")
    named, env_prefix = [], None
    if len(new_skill_dirs) == 1:
        sk = work / new_skill_dirs[0]
        for f in ("SKILL.md", "metadata.yaml"):
            if (sk / f).exists():
                named += ENV_RE.findall((sk / f).read_text(errors="replace"))
        named = sorted(set(named))
        if len(named) > 1:
            problems.append(f"expected the skill to name at most one slurm/envs/<family> prefix, found {named}")
        elif named:
            env_prefix = work / "slurm" / "envs" / named[0]
            if not (env_prefix / "bin" / "python").exists():
                problems.append(f"{env_prefix}/bin/python does not exist")
        clash = [n for n in ("data.csv", "answer.json") if (sk / n).exists()]
        if clash:
            problems.append(f"the skill folder holds {clash} at its top level, which the solver's working directory uses")

    sessions = session_files(work)
    if len(sessions) != 1:
        problems.append(f"expected one Codex session file for {work}, found {[str(s) for s in sessions]}")

    manifest = {"generator": g, "base": BASE, "head": git(work, "rev-parse", "HEAD").strip(),
                "run_folders": runs, "changed": {p: sha256(work / p) for p in kept if (work / p).is_file()},
                "deleted": deleted, "left_out": [p for p in paths if p not in kept],
                "env_prefixes": env_lists, "new_skill_dirs": new_skill_dirs, "skill_names_env": named,
                "problems": problems}
    if problems:
        out.joinpath("manifest.json").write_text(json.dumps(manifest, indent=1))
        raise Stop("stop and tell Jun:\n  " + "\n  ".join(problems))

    shutil.copy2(sessions[0], out / "codex_session.jsonl")
    manifest["codex_session"] = {"source": str(sessions[0].relative_to(HERE)), "sha256": sha256(sessions[0]),
                                 "file": "codex_session.jsonl"}
    provided = skills / "code_provided"
    shutil.copytree(work / new_skill_dirs[0], provided, ignore=shutil.ignore_patterns("__pycache__"))
    names, removed = code_written(provided, skills / "code_written")
    env_prefix = str(env_prefix) if env_prefix else None
    manifest.update(skill_dir=new_skill_dirs[0], env_prefix=env_prefix,
                    code_provided_sha=tree_sha(provided), code_written_sha=tree_sha(skills / "code_written"),
                    script_names=names, code_written_removed=removed)
    out.joinpath("manifest.json").write_text(json.dumps(manifest, indent=1))
    skills.joinpath("solver.json").write_text(json.dumps(
        {"generator": g, "env_prefix": env_prefix, "code_provided_sha": manifest["code_provided_sha"],
         "code_written_sha": manifest["code_written_sha"]}, indent=1))
    print(f"{g}: {len(kept)} files kept, {len(deleted)} deleted, run folders {runs}, envs {list(env_lists)}")
    print(f"  skill {new_skill_dirs[0]} -> {provided}; env {env_prefix or 'none named: the simple loop interpreter (G8d)'}")
    print(f"  code_written removes {len(removed)} files: {removed}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
