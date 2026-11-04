"""scs_profile — the SkillFoundry profile pipeline for one method package (profile steps 1, 2 and 4).

    request(pkg, use)                         the request text, from templates/profile_request.md
    prepare(pkg, use, generator, root, repo)  a SkillFoundry working copy at commit 4a8cc0c with the paper in paper/
                                              and the reference code in code/, its Codex home and the request file
    collect(work, out, skills, ...)           after a generator run: keep what SkillFoundry wrote and build the two
                                              formats, with the rules of skillfoundry_profile/sf_collect.py

Step 3, the generator run, is scs_box.sh, run by Jun in Jun's own terminal with the key exported.

    python scs_profile.py request <method> <use>
    python scs_profile.py prepare <method> <use> <generator> <root> [<repo>]
    python scs_profile.py collect <work> <out> <skills> <run.log> <codex home root>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import scs_method

HERE = Path(__file__).resolve().parent
PROFILE = HERE.parent / "skillfoundry_profile"
sys.path.insert(0, str(PROFILE))
import sf_collect as C                 # noqa: E402   changed, session_files, env_prefixes, package_list, code_written, tree_sha

BASE = C.BASE                          # SkillFoundry commit 4a8cc0c (README row G1v)
LEFT_OUT = ("paper/", "code/")         # the paper and the reference code we put in; never collected as output
REPO = "https://github.com/ma-compbio-lab/SkillFoundry"
GENERATORS = ("gpt-5.4", "deepseek")


def request(pkg, use):
    t = pkg.texts
    return (HERE / "templates" / "profile_request.md").read_text().format(
        task=use.profile_task, method_name=t.method_name, authors=t.authors, paper_title=t.paper_title,
        paper_file=Path(pkg.paper).name, code=t.profile_code, data=use.profile_data, answer=use.profile_answer)


def prepare(pkg, use, generator, root, repo=REPO):
    """root/work/<generator>: SkillFoundry at BASE, paper/<paper file>, code/<reference code>; root/codex_home/<generator>
    with SkillFoundry profile's config.toml; root/request.md. Returns the paths."""
    assert generator in GENERATORS, generator
    root = Path(root)
    work, chome = root / "work" / generator, root / "codex_home" / generator
    if work.exists():
        raise SystemExit(f"{work} exists: this generator's working copy is prepared already")
    work.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "checkout", "--quiet", BASE], check=True)
    (work / "paper").mkdir()
    shutil.copy2(pkg.paper, work / "paper" / Path(pkg.paper).name)
    if pkg.texts.profile_code_paths:
        (work / "code").mkdir()
        for p in map(Path, pkg.texts.profile_code_paths):
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, work / "code" / p.name)
        with open(work / ".git" / "info" / "exclude", "a") as f:
            f.write("code/\n")                        # git status stays clean, as for paper/ (their .gitignore)
    chome.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROFILE / "codex_home" / "config.toml", chome / "config.toml")
    req = root / "request.md"
    req.write_text(request(pkg, use))
    return {"work": work, "codex_home": chome, "request": req, "label": f"{pkg.texts.skill_prefix}-{generator}",
            "focus": pkg.texts.profile_focus}


def collect(work, out, skills, run_log, codex_home_root):
    """sf_collect.main with the paths as arguments and code/ left out as paper/ is. Writes out/files/,
    out/diff_tracked.patch, out/codex_session.jsonl, out/manifest.json, skills/code_provided/, skills/code_written/ and
    skills/solver.json. Stops (SystemExit) on every condition sf_collect stops on."""
    work, out, skills, run_log = Path(work), Path(out), Path(skills), Path(run_log)
    # git prints a path holding a non-ASCII character quoted and escaped unless core.quotepath is off, and the filters
    # below compare raw path prefixes (the SCM paper's file name has a curly apostrophe); this applies to every git call
    # of this process and changes no file of the working copy
    import os
    os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="core.quotepath", GIT_CONFIG_VALUE_0="off")
    if not run_log.exists():
        raise C.Stop(f"{run_log} is missing: run scs_box.sh first")
    if not any(l.startswith("finished ") for l in run_log.read_text().splitlines()):
        raise C.Stop(f"{run_log} has no 'finished' line: the generator run has not ended")
    if (out / "manifest.json").exists() or skills.exists():
        raise C.Stop(f"collected already ({out / 'manifest.json'}, {skills})")
    paths, deleted = C.changed(work)
    envs = C.env_prefixes(paths)
    kept = [p for p in paths if not p.startswith(LEFT_OUT) and "__pycache__/" not in p
            and not any(p.startswith(e + "/") for e in envs)]
    for p in kept:
        dst = out / "files" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / p, dst, follow_symlinks=False)
    (out / "diff_tracked.patch").write_text(C.git(work, "diff", "--no-renames", BASE))
    env_lists = {}
    for e in envs:
        cmd, code, text = C.package_list(work / e)
        (out / "envs").mkdir(parents=True, exist_ok=True)
        (out / "envs" / (Path(e).name + ".txt")).write_text(f"# {cmd}\n# exit {code}\n{text}")
        env_lists[e] = {"command": cmd, "exit": code, "file": f"envs/{Path(e).name}.txt"}
    runs = sorted({"/".join(p.split("/")[:3]) for p in kept if p.startswith("reports/framework-runs/")})
    base_skills = C.git(work, "ls-tree", "-r", "--name-only", BASE, "skills").split()
    new_skill_dirs = sorted({str(Path(p).parent) for p in paths
                             if C.re.fullmatch(r"skills/[^/]+/[^/]+/SKILL\.md", p) and p not in base_skills})
    problems, named, env_prefix = [], [], None
    if len(new_skill_dirs) != 1:
        problems.append(f"expected one new skill folder under skills/, found {new_skill_dirs}")
    else:
        sk = work / new_skill_dirs[0]
        for f in ("SKILL.md", "metadata.yaml"):
            if (sk / f).exists():
                named += C.ENV_RE.findall((sk / f).read_text(errors="replace"))
        named = sorted(set(named))
        if len(named) > 1:
            problems.append(f"expected the skill to name at most one slurm/envs/<family> prefix, found {named}")
        elif named:
            env_prefix = work / "slurm" / "envs" / named[0]
            if not any((env_prefix / "bin" / x).exists() for x in ("python", "Rscript")):
                problems.append(f"{env_prefix}/bin holds neither python nor Rscript")
        clash = [n for n in ("data.csv", "answer.json") if (sk / n).exists()]
        if clash:
            problems.append(f"the skill folder holds {clash} at its top level, which the solver's working directory uses")
    sessions = C.session_files(work, root=codex_home_root)
    if len(sessions) != 1:
        problems.append(f"expected one Codex session file for {work}, found {[str(s) for s in sessions]}")
    manifest = {"base": BASE, "head": C.git(work, "rev-parse", "HEAD").strip(), "run_folders": runs,
                "changed": {p: C.sha256(work / p) for p in kept if (work / p).is_file()}, "deleted": deleted,
                "left_out": [p for p in paths if p not in kept], "env_prefixes": env_lists,
                "new_skill_dirs": new_skill_dirs, "skill_names_env": named, "problems": problems}
    out.mkdir(parents=True, exist_ok=True)
    if problems:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        raise C.Stop("stop and tell Jun:\n  " + "\n  ".join(problems))
    shutil.copy2(sessions[0], out / "codex_session.jsonl")
    manifest["codex_session"] = {"source": str(sessions[0]), "sha256": C.sha256(sessions[0]), "file": "codex_session.jsonl"}
    provided = skills / "code_provided"
    shutil.copytree(work / new_skill_dirs[0], provided, ignore=shutil.ignore_patterns("__pycache__"))
    names, removed = C.code_written(provided, skills / "code_written")
    env_prefix = str(env_prefix) if env_prefix else None
    manifest.update(skill_dir=new_skill_dirs[0], env_prefix=env_prefix, code_provided_sha=C.tree_sha(provided),
                    code_written_sha=C.tree_sha(skills / "code_written"), script_names=names,
                    code_written_removed=removed)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    solver = {"env_prefix": env_prefix, "code_provided_sha": manifest["code_provided_sha"],
              "code_written_sha": manifest["code_written_sha"]}
    (skills / "solver.json").write_text(json.dumps(solver, indent=1))
    return manifest


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "request":
        mod = scs_method.load_module(args[0])
        sys.stdout.write(request(mod.PACKAGE, mod.USES[args[1]]))
    elif cmd == "prepare":
        mod = scs_method.load_module(args[0])
        r = prepare(mod.PACKAGE, mod.USES[args[1]], args[2], args[3], *(args[4:5] or [REPO]))
        print(json.dumps({k: str(v) for k, v in r.items()}, indent=1))
        print(f"next, in Jun's terminal: bash scs_box.sh probe {args[2]} {r['work']} {r['codex_home']} "
              f"{HERE.parent} {Path.home()}\n"
              f"then: bash scs_box.sh run {args[2]} {r['work']} {r['codex_home']} <out folder> {r['request']} "
              f"{r['label']} \"{r['focus']}\"")
    elif cmd == "collect":
        m = collect(*args[:5])
        print(f"collected: {m['skill_dir']}; code_written removes {len(m['code_written_removed'])} files")
