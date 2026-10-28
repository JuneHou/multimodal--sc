"""sf_tests — gates for the profile runner and the collection step, on a fixture skill (no model call).
    PF1  the task message is the G16 text; the simple loop's Cell and task_text are restored after a profile cell
    PF2  G5 mounts: the working directory shows data.csv and exactly the skill's top-level entries, all read-only;
         no references/ or scripts/ appear when the skill has none; project, home and network are absent
    PF3  G8c: python resolves to the generated environment's bin/, for a conda prefix and for a venv;
         G8d: with no environment, python and python3 resolve to the simple loop's interpreter
    PF4  data.csv holds nine periods; a canary P10 value reaches no output
    PF5  run_cell through the profile loop: a canned dry run ends with a valid answer; the step cap ends a silent
         cell; an empty output is carried forward as "(no output)"
    PF6  G7c: code_written drops scripts/ and the tests/examples files that name a script, and keeps the rest;
         G14c: changed() sees tracked changes, new files and new git-ignored files; env prefixes are found;
         G14d: session_files() picks the Codex session files whose first line names the working copy
    PF7  G15r: inside run_generator.sh's box the generator sees its working copy and the paper and nothing of the
         experiment, the simple skills, the collaborator's data, the other working copy, home or host /tmp; no key
         variable passes; the network answers; Codex's own sandbox writes inside the working copy and not outside
    PF8  the daily harness of sf_daily.py: 52 cases, 26 per sensor; each data.csv equals the case's visits given in
         lb_cases_daily.csv; visits_to_predict.csv lists the visits to predict with no value; the task message is the
         plan's text; a canned dry run answers; the simple loop's names are restored
    PF9  decision OL: in a monthly and a daily profile cell the model gets a 20,000 character output whole, with no
         truncation mark, and the simple loop's 6,000 character limit is restored afterwards
    PF10 on both generated skills in both formats: the daily working directory shows data.csv, visits_to_predict.csv
         and exactly the skill's entries, read-only; a per-visit answer ends a canned case; one value per band is
         refused with the daily message
    PF11 sf_score_daily.py on canned answers equal to notebook 14's own predictions over the 52 cases gives notebook
         14's printed Table 2 by band and cell 12 effect summary, every number within 0.0001 (Jun "i don't care the
         0.0001", 2026-10-15); no statement of cell 12 raises
    python sf_tests.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import sf_run as S                 # puts llm_band_sc on sys.path
import sf_collect as C
import sf_daily as D
import lb_skill_run_daily as RD
import lb_guard
import lb_lib as lb
import lb_skill_loop as L

ENV = Path("/data/wang/junh/envs/skillfoundry")          # stand-in for a generated environment (Python 3.11)
PROJECT = str(lb.LB.parent.parent.parent)                # the repository
HOME = "/home/grads/junh"


def _fixture(root, with_scripts=True):
    sk = Path(root) / "shc-fixture"
    (sk / "tests").mkdir(parents=True); (sk / "examples").mkdir(); (sk / "assets").mkdir()
    (sk / "SKILL.md").write_text("# SHC fixture\n\nRun `python scripts/hello.py`, then write answer.json.\n")
    (sk / "refs.md").write_text("- fixture reference\n")
    (sk / "metadata.yaml").write_text('{"name": "SHC fixture", "test_commands": ["slurm/envs/shcfix/bin/python scripts/hello.py"]}\n')
    (sk / "assets" / "README.md").write_text("assets\n")
    (sk / "tests" / "test_hello.py").write_text("import subprocess\nsubprocess.run(['python', 'scripts/hello.py'])\n")
    (sk / "tests" / "test_other.py").write_text("assert 1 + 1 == 2\n")
    (sk / "examples" / "run.sh").write_text("python scripts/hello.py\n")
    (sk / "examples" / "notes.md").write_text("no script named here\n")
    if with_scripts:
        (sk / "scripts").mkdir()
        (sk / "scripts" / "hello.py").write_text("import sys\nprint('PREFIX', sys.prefix)\n")
    return sk


def _canned(script):
    it = iter(script)
    return lambda messages: (next(it), {"finish_reason": "stop", "reasoning": "canned", "gateway": "test"})


def _fence(cmd):
    return f"```bash\n{cmd}\n```"


def _cell(sk, env=ENV, sensor="sentinel1", site="treatment_0015", panel=None):
    panel = lb_guard.training_panel() if panel is None else panel
    return S.ProfileCell(sk, lb_guard.cell_panel(panel, sensor, site), sensor, site, env_prefix=env)


def gate_pf1_task_text():
    want = (f"Predict this land parcel's mean band values for the next month, P10 = {lb.CALENDAR[lb.TARGET]}, from its own "
            "past months.\nThe working directory holds a skill for this task. Its instructions are in SKILL.md; read that "
            "file first and follow it. Reply with one ```bash block per turn; it is run by bash in the working directory "
            "and its output comes back in the next message. Finish by writing answer.json as the skill describes.")
    assert S.task_text_profile(None) == want, S.task_text_profile(None)
    cell0, text0 = L.Cell, L.task_text
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        seen = []
        def call(messages):
            seen.append(messages[1]["content"]); return _fence("echo '{\"prediction\": {}}' > /dev/null"), {}
        S.run_cell(sk, ENV, lb_guard.cell_panel(lb_guard.training_panel(), "sentinel1", "treatment_0015"),
                   "sentinel1", "treatment_0015", call, step_cap=1)
    assert seen and seen[0] == want, seen[:1]
    assert L.Cell is cell0 and L.task_text is text0, "simple loop names not restored"
    print("PF1 ok: the task message is the G16 text; the simple loop's Cell and task_text are restored afterwards")


def gate_pf2_mounts_and_isolation():
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        c = _cell(sk)
        try:
            out, m = c.do_shell("ls -A"); assert m["exit"] == 0, out
            assert out.split() == sorted(["data.csv"] + [p.name for p in sk.iterdir()]), out
            out, m = c.do_shell("cat refs.md metadata.yaml assets/README.md tests/test_other.py examples/notes.md"); assert m["exit"] == 0, out
            for target in ("SKILL.md", "refs.md", "scripts/hello.py", "tests/test_other.py"):
                out, m = c.do_shell(f"echo x >> {target}"); assert m["exit"] != 0 and "Read-only" in out, (target, out)
            out, m = c.do_shell("rm scripts/hello.py"); assert m["exit"] != 0, out
            out, m = c.do_shell("touch notes.txt && ls notes.txt"); assert m["exit"] == 0, out
            out, m = c.do_shell(f"test -e {PROJECT} && echo VISIBLE || echo absent"); assert out.strip() == "absent", out
            out, m = c.do_shell(f"test -e {HOME} && echo VISIBLE || echo absent"); assert out.strip() == "absent", out
            out, m = c.do_shell("python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\""); assert m["exit"] != 0, out
            assert set(c.files()) == {"data.csv", "notes.txt"}, c.files()
        finally:
            c.close()
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t, with_scripts=False)
        c = _cell(sk)
        try:
            out, m = c.do_shell("ls -A"); assert "scripts" not in out.split() and "references" not in out.split(), out
        finally:
            c.close()
    print("PF2 ok: the working directory shows data.csv and exactly the skill's entries, all read-only; no scripts/ or "
          "references/ appear when the skill has none; project, home and network are absent")


def gate_pf3_environment():
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        c = _cell(sk)
        try:
            out, m = c.do_shell("command -v python; python scripts/hello.py"); assert m["exit"] == 0, out
            assert out.splitlines()[0] == f"{ENV}/bin/python" and f"PREFIX {ENV}" in out, out
        finally:
            c.close()
        venv = Path(t) / "venv"
        subprocess.run([str(ENV / "bin" / "python"), "-m", "venv", "--without-pip", str(venv)], check=True)
        c = _cell(sk, env=venv)
        try:
            out, m = c.do_shell("command -v python; python scripts/hello.py"); assert m["exit"] == 0, out
            assert out.splitlines()[0] == f"{venv}/bin/python" and f"PREFIX {venv}" in out, out
        finally:
            c.close()
        c = _cell(sk, env=None)
        try:
            out, m = c.do_shell("command -v python; command -v python3; python3 scripts/hello.py"); assert m["exit"] == 0, out
            lines = out.splitlines()
            assert lines[:2] == [f"{L.PY_DIR}/python", f"{L.PY_DIR}/python3"] and f"PREFIX {sys.prefix}" in out, out
        finally:
            c.close()
    print("PF3 ok: python resolves to the environment's bin/ and runs, for a conda prefix and for a venv; "
          "with no environment, python and python3 are the simple loop's interpreter")


def gate_pf4_nine_periods_and_canary():
    panel, sentinels = lb_guard.canary_panel()
    panel9 = panel[panel.period_num <= 9]
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        for sensor, site in (("sentinel1", "treatment_0015"), ("sentinel2", "treatment_0014")):
            c = _cell(sk, sensor=sensor, site=site, panel=panel9)
            try:
                data, _ = c.do_shell("cat data.csv")
                assert data.count("\n") == 9 * len(lb.BANDS[sensor]) and "P10" not in data
                out, _ = c.do_shell("ls -R; cat *.md; python scripts/hello.py")
                lb_guard.assert_no_sentinel(data + out, sentinels, f"profile cell {sensor} {site}")
            finally:
                c.close()
    print("PF4 ok: data.csv holds nine periods; a sentinel P10 value reaches no output")


def gate_pf5_loop():
    panel9 = lb_guard.training_panel()
    cellp = lb_guard.cell_panel(panel9, "sentinel1", "treatment_0015")
    bands = [lb.disp(b) for b in lb.BANDS["sentinel1"]]
    ans = json.dumps({"prediction": {b: 1.0 for b in bands}, "rationale": "fixture"})
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        rec = S.run_cell(sk, ENV, cellp, "sentinel1", "treatment_0015",
                         _canned([_fence("cat SKILL.md"), _fence("python scripts/hello.py"),
                                  _fence(f"cat > answer.json <<'EOF'\n{ans}\nEOF")]))
        assert rec["ended"] == "answer" and rec["ok"] and rec["n_steps"] == 3 and rec["values"] == {b: 1.0 for b in bands}, rec["ended"]
        assert "SKILL.md" in rec["refs_read"] and rec["python_runs"] == 1, (rec["refs_read"], rec["python_runs"])
        rec = S.run_cell(sk, ENV, cellp, "sentinel1", "treatment_0015", _canned([_fence("ls")] * 20), step_cap=4)
        assert rec["ended"] == "step_cap" and rec["n_steps"] == 4 and not rec["ok"]
        sent = []
        def call(messages):
            sent.append(messages[-1]["content"]); return (_fence("true") if len(sent) == 1 else _fence(f"cat > answer.json <<'EOF'\n{ans}\nEOF")), {}
        rec = S.run_cell(sk, ENV, cellp, "sentinel1", "treatment_0015", call)
        assert rec["ok"] and sent[1] == "(no output)", sent
    print("PF5 ok: the profile loop answers on a canned dry run, stops at the step cap, and sends \"(no output)\" for an empty output")


def gate_pf6_collect():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "sessions" / "2026" / "10" / "13"
        root.mkdir(parents=True)
        for name, cwd in (("rollout-a.jsonl", "/w/gpt-5.4"), ("rollout-b.jsonl", "/w/deepseek"), ("rollout-c.jsonl", "/w/gpt-5.4x")):
            (root / name).write_text(json.dumps({"type": "session_meta", "payload": {"cwd": cwd}}) + "\n{}\n")
        got = C.session_files(Path("/w/gpt-5.4"), root=Path(t) / "sessions")
        assert [f.name for f in got] == ["rollout-a.jsonl"], got
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        names, removed = C.code_written(sk, Path(t) / "written")
        w = Path(t) / "written"
        assert names == ["hello.py"], names
        assert sorted(removed) == ["examples/run.sh", "scripts/hello.py", "tests/test_hello.py"], removed
        assert not (w / "scripts").exists() and (w / "tests" / "test_other.py").exists() and (w / "examples" / "notes.md").exists()
        assert (w / "SKILL.md").read_text() == (sk / "SKILL.md").read_text() and (w / "refs.md").exists()
        repo = Path(t) / "repo"
        repo.mkdir()
        g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout
        g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
        (repo / ".gitignore").write_text("reports/\nslurm/\npaper/\n")
        (repo / "kept.txt").write_text("a\n"); (repo / "gone.txt").write_text("b\n")
        g("add", "-A"); g("commit", "-q", "-m", "base")
        base, C.BASE = C.BASE, g("rev-parse", "HEAD").strip()
        try:
            (repo / "kept.txt").write_text("a2\n"); (repo / "gone.txt").unlink()
            (repo / "skills" / "d" / "s").mkdir(parents=True); (repo / "skills" / "d" / "s" / "SKILL.md").write_text("x")
            (repo / "reports" / "framework-runs" / "r1").mkdir(parents=True); (repo / "reports" / "framework-runs" / "r1" / "manifest.json").write_text("{}")
            (repo / "slurm" / "envs" / "shcfix" / "bin").mkdir(parents=True); (repo / "slurm" / "envs" / "shcfix" / "bin" / "python").write_text("")
            (repo / "paper").mkdir(); (repo / "paper" / "p.pdf").write_text("pdf")
            paths, deleted = C.changed(repo)
        finally:
            C.BASE = base
        assert deleted == ["gone.txt"], deleted
        assert set(paths) == {"kept.txt", "skills/d/s/SKILL.md", "reports/framework-runs/r1/manifest.json",
                              "slurm/envs/shcfix/bin/python", "paper/p.pdf"}, paths
        assert C.env_prefixes(paths) == ["slurm/envs/shcfix"], C.env_prefixes(paths)
        assert C.ENV_RE.findall((sk / "metadata.yaml").read_text()) == ["shcfix"]
    print("PF6 ok: code_written drops scripts/, tests/test_hello.py and examples/run.sh and keeps the rest; changed() "
          "sees tracked edits, deletions, new and git-ignored files; the env prefix is found; the session file of the "
          "working copy is picked")


def gate_pf7_generator_box():
    expected = {"experiment README": "absent", "simple-skill folder": "absent", "collaborator data": "absent",
                "other working copy": "absent", "home directory": "absent", "tmp entries": "0", "paper": "readable",
                "env names with KEY": "0", "framework": "starts", "codex sandbox write inside": "ok",
                "codex sandbox write outside": "denied"}
    for g in ("gpt-5.4", "deepseek"):
        before = subprocess.run(["git", "-C", str(S.HERE / "work" / g), "status", "--porcelain"], capture_output=True, text=True).stdout
        r = subprocess.run(["bash", str(S.HERE / "run_generator.sh"), "--probe", g], capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        got = dict(l.split(": ", 1) for l in r.stdout.splitlines() if ": " in l)
        for k, v in expected.items():
            assert got.get(k) == v, (g, k, got.get(k))
        assert got.get("network", "").startswith("reachable"), (g, got.get("network"))
        after = subprocess.run(["git", "-C", str(S.HERE / "work" / g), "status", "--porcelain"], capture_output=True, text=True).stdout
        assert before == after, (g, "the probe changed the working copy")
    print("PF7 ok: each generator box shows its working copy and the paper only; no key variable; network reachable; "
          "Codex's sandbox writes inside and not outside")


DAILY_TASK = ("Predict this land parcel's mean band values at each visit listed in visits_to_predict.csv, from its own past "
              "visits.\nThe working directory holds a skill for this task. Its instructions are in SKILL.md; read that file "
              "first and follow it. Reply with one ```bash block per turn; it is run by bash in the working directory and its "
              "output comes back in the next message. Finish by writing answer.json as {\"prediction\": {\"<period_id>\": "
              "{\"<band>\": <value>, ...}, ...}, \"rationale\": \"<text>\"}.")


def _daily_answer(cases, sensor, site):
    _, to_predict = RD.case_parts(cases, sensor, site)
    bands = [lb.disp(b) for b in lb.BANDS[sensor]]
    return json.dumps({"prediction": {p: {b: 1.0 for b in bands} for p, _ in to_predict}, "rationale": "fixture"})


def gate_pf8_daily():
    import pandas as pd
    assert D.task_text_daily(None) == DAILY_TASK, D.task_text_daily(None)
    cases = RD.load_cases()
    pairs = D.case_pairs(cases)
    assert [sum(s == x for x, _ in pairs) for s in ("sentinel1", "sentinel2")] == [26, 26], len(pairs)
    cell0, text0, lim0 = L.Cell, L.task_text, L.OUT_LIMIT
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        for sensor, site in pairs:
            given, to_predict = RD.case_parts(cases, sensor, site)
            c = D.ProfileDailyCell(sk, (given, to_predict), sensor, site, env_prefix=None)
            try:
                out, m = c.do_shell("ls -A"); assert m["exit"] == 0, out
                assert out.split() == sorted(["data.csv", "visits_to_predict.csv"] + [p.name for p in sk.iterdir()]), out
                d = pd.read_csv(c.dir / "data.csv", dtype={"period_id": str}, float_precision="round_trip")
                want = given.assign(band=[lb.disp(b) for b in given["band"]], date=given["date"].dt.strftime("%Y-%m-%d"))
                want = want.sort_values(["date", "band"])[["band", "period_id", "date", "y"]].reset_index(drop=True)
                assert d.equals(want), (sensor, site)
                v = pd.read_csv(c.dir / "visits_to_predict.csv", dtype=str)
                assert list(v.columns) == ["period_id", "date"] and list(zip(v["period_id"], v["date"])) == to_predict
                assert d["date"].max() < "2024-09-27" <= v["date"].min() and not set(v["period_id"]) & set(d["period_id"])
                post = cases[cases["sensor"].eq(sensor) & cases["site_id"].eq(site) & cases["part"].eq("post")]
                assert set(post["period_id"]) == set(v["period_id"]) and len(v) == post["period_id"].nunique()
            finally:
                c.close()
        sensor, site = pairs[0]
        ans = _daily_answer(cases, sensor, site)
        seen = []
        def call(messages):
            seen.append(messages[1]["content"])
            return (_fence("head -3 data.csv; head -3 visits_to_predict.csv") if len(seen) == 1
                    else _fence(f"cat > answer.json <<'EOF'\n{ans}\nEOF")), {}
        rec = D.run_case(sk, None, cases, sensor, site, call)
        assert rec["ended"] == "answer" and rec["ok"] and rec["values"] == json.loads(ans)["prediction"], rec["ended"]
        assert seen[0] == DAILY_TASK and "band,period_id,date,y" in rec["steps"][0]["output"]
    assert (L.Cell, L.task_text, L.OUT_LIMIT) == (cell0, text0, lim0), "simple loop names not restored"
    print("PF8 ok: 52 daily cases, 26 per sensor; each data.csv equals the case's visits given in lb_cases_daily.csv and "
          "holds no date on or after 27 September 2024; visits_to_predict.csv lists the visits to predict with no value; "
          "the task message is the plan's text; a canned dry run answers; the simple loop's names are restored")


def gate_pf9_output_whole():
    simple_limit = L.OUT_LIMIT
    panel9 = lb_guard.training_panel()
    cellp = lb_guard.cell_panel(panel9, "sentinel1", "treatment_0015")
    cases = RD.load_cases()
    sensor, site = D.case_pairs(cases)[0]
    bands = [lb.disp(b) for b in lb.BANDS["sentinel1"]]
    monthly = json.dumps({"prediction": {b: 1.0 for b in bands}, "rationale": "fixture"})
    daily = _daily_answer(cases, sensor, site)
    with tempfile.TemporaryDirectory() as t:
        sk = _fixture(t)
        long_md = "# SHC fixture\n" + "".join(f"line {i:05d} of the procedure\n" for i in range(800))
        (sk / "SKILL.md").write_text(long_md)
        assert len(long_md) > 20000 > simple_limit
        for run, ans in ((lambda call: S.run_cell(sk, None, cellp, "sentinel1", "treatment_0015", call), monthly),
                         (lambda call: D.run_case(sk, None, cases, sensor, site, call), daily)):
            sent = []
            def call(messages):
                sent.append(messages[-1]["content"])
                return (_fence("cat SKILL.md") if len(sent) == 1 else _fence(f"cat > answer.json <<'EOF'\n{ans}\nEOF")), {}
            rec = run(call)
            assert rec["ok"] and sent[1] == long_md.strip("\n") and "[output truncated]" not in sent[1], len(sent[1])
            assert L.OUT_LIMIT == simple_limit == 6000, L.OUT_LIMIT
    print("PF9 ok: a monthly and a daily profile cell show the model a 20,000 character output whole; "
          "the simple loop's 6,000 character limit is restored afterwards")


def gate_pf10_generated_skills_daily():
    cases = RD.load_cases()
    sensor, site = D.case_pairs(cases)[0]
    good = _daily_answer(cases, sensor, site)
    flat = json.dumps({"prediction": {lb.disp(b): 1.0 for b in lb.BANDS[sensor]}, "rationale": "one value per band"})
    for g in S.GENERATORS:
        for fmt in S.FORMATS:
            skill_dir, env_prefix, _ = S.load_solver(g, fmt)
            c = D.ProfileDailyCell(skill_dir, RD.case_parts(cases, sensor, site), sensor, site, env_prefix=env_prefix)
            try:
                out, m = c.do_shell("ls -A"); assert m["exit"] == 0, out
                assert out.split() == sorted(["data.csv", "visits_to_predict.csv"] + c.entries), (g, fmt, out)
                out, m = c.do_shell("echo x >> SKILL.md"); assert m["exit"] != 0 and "Read-only" in out, (g, fmt, out)
            finally:
                c.close()
            rec = D.run_case(skill_dir, env_prefix, cases, sensor, site,
                             _canned([_fence("head -5 SKILL.md"), _fence(f"cat > answer.json <<'EOF'\n{good}\nEOF")]))
            assert rec["ok"] and rec["n_steps"] == 2 and rec["values"] == json.loads(good)["prediction"], (g, fmt, rec["ended"])
            rec = D.run_case(skill_dir, env_prefix, cases, sensor, site,
                             _canned([_fence(f"cat > answer.json <<'EOF'\n{flat}\nEOF")]), step_cap=1)
            assert not rec["ok"] and "for every visit listed in visits_to_predict.csv" in rec["steps"][0]["output"], (g, fmt)
    print("PF10 ok: for both generators and both formats the working directory shows data.csv, visits_to_predict.csv and "
          "exactly the skill's entries, read-only; a per-visit answer ends the case; one value per band is refused with "
          "the daily message")


def _pf11_body():
    """Run in a fresh interpreter that imported sf_score_daily first (one thread before numpy loads)."""
    from decimal import Decimal, InvalidOperation
    import pandas as pd
    import sf_score_daily as F
    eff = pd.read_csv(lb.LB / "lb_her14_effects.csv", float_precision="round_trip")
    cases = pd.read_csv(lb.LB / "lb_cases_daily.csv", float_precision="round_trip")
    post = cases[cases["part"] == "post"].drop_duplicates(["sensor", "site_id", "period_id"])
    pid = {(s, site, pd.Timestamp(d)): p for s, site, d, p in zip(post["sensor"], post["site_id"], post["date"], post["period_id"])}
    recs = {}
    for (site, sensor), g in eff.groupby(["site_id", "sensor"]):
        vals = {}
        for b, d, v in zip(g["band"], g["date"], g["y_star_synthetic"]):
            vals.setdefault(pid[(sensor, site, pd.Timestamp(d))], {})[lb.disp(b)] = float(v)
        recs[(site, sensor)] = {"ok": True, "values": vals}
    assert len(recs) == 52, len(recs)
    tables, ns, log = F.score_row("pf11", recs)
    assert not [e for e in log if e["cell"] == 12], [e for e in log if e["cell"] == 12]
    for name in ("table2_by_band", "effect_summary"):
        got = tables[name].reset_index(drop=True)
        want = pd.read_csv(lb.LB / f"lb_her14_printed_{name}.csv", dtype=str)
        assert list(got.columns) == list(want.columns) and len(got) == len(want), (name, list(got.columns), len(got))
        worst = Decimal(0)
        for c in want.columns:
            for a, b in zip(got[c].astype(str), want[c]):
                try:
                    worst = max(worst, abs(Decimal(a) - Decimal(b)))
                except InvalidOperation:
                    assert a == b, (name, c, a, b)
        assert worst <= Decimal("0.0001"), (name, worst)
        print(f"  {name}: {len(got)} rows, largest difference from notebook 14's printed table {worst}")


def gate_pf11_scorer():
    r = subprocess.run([sys.executable, "-c", "import sf_score_daily, sf_tests; sf_tests._pf11_body()"],
                       cwd=S.HERE, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    print(r.stdout.rstrip())
    print("PF11 ok: canned profile answers equal to notebook 14's own predictions give notebook 14's printed Table 2 by "
          "band and effect summary, every number within 0.0001; no statement of cell 12 raised")


def main():
    gate_pf1_task_text()
    gate_pf2_mounts_and_isolation()
    gate_pf3_environment()
    gate_pf4_nine_periods_and_canary()
    gate_pf5_loop()
    gate_pf6_collect()
    gate_pf7_generator_box()
    gate_pf8_daily()
    gate_pf9_output_whole()
    gate_pf10_generated_skills_daily()
    gate_pf11_scorer()
    print("all profile gates passed")


if __name__ == "__main__":
    main()
