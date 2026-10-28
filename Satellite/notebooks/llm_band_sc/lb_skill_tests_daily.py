"""lb_skill_tests_daily — gates of the daily simple skills (plan ~/.claude/plans/simple-skills-daily.md, Verification).
    DK1  python lb_extract_her14.py --check
    DK2  python lb_her14_run.py --parity
    DK3  her fit of the case with the most visits finishes inside the 120 s command limit in the skill sandbox
         with one thread; the one-case run reproduces her printed T0, n, m*, J* and R of that case
    DK4  lb_fit_daily.run_case at her settings and her Class 2 behaviour returns, for every band and post visit of
         all 52 cases, her y_star_synthetic of our run of her code (lb_her14_effects.csv) with difference 0, and her
         cell 10 arithmetic on it returns her counterfactual_y0 with difference 0
    DK5  a spec breaking each cross-field rule is refused with its field named; shc-configure refuses a Class 2
         field. No fit outside her code is run by any gate: Jun "you should not test with code deviate from her"
         (2026-10-15)
    DK6  python lb_skills_build_daily.py --check
    DK7  a case's working directory holds the visits given and the dates to predict and no value after them: every
         observed value of the visits to predict is replaced by a sentinel in the case table, and no sentinel reaches
         data.csv, visits_to_predict.csv, show_data.py, fit.py or the fit record; from inside the sandbox the project,
         the home directory and the network do not exist and the skill files cannot be written; shc-configure's
         fit.py refuses a choice field
    DK8  canned dry runs of shc-configure and shc-design end with a valid answer.json and no model call; the fit
         record at her settings equals her y_star_synthetic of our run (lb_her14_effects.csv) for that case
    DK9  a 20,000 character output reaches the model whole, and OUT_LIMIT is 6,000 again after the case
    DK10 the step cap ends a case that never answers; an empty reply and an empty output are carried forward as
         "(empty reply)" and "(no output)"; the runner re-runs only call-error cases
    DK11 the scorer (lb_skill_score_daily.py) at her settings with her own predictions over her 52 cases reproduces
         our run of her code with difference 0 (lb_her14_effects, effect_by_site_band, effect_summary, inference),
         its parsed printed tables equal her printed tables within 0.0001 (as DK2), and it draws 52 figures
    python lb_skill_tests_daily.py [dk3] [dk4] [dk5] [dk6] [dk7] [dk8] [dk9] [dk10] [dk11]
"""
import lb_one_thread  # noqa: F401  first import: one thread, as the skill sandbox runs
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lb_fit_daily as F
import lb_guard
import lb_her14_run
import lb_lib as lb
import lb_skill_run
import lb_skill_run_daily as RD
import lb_skills_build_daily as BD
import lb_spec_daily as S
import lb_sandbox
import lb_skill_loop as L

# Her cells 1 and 2 (imports and settings), then cells 4 to 8 on the one case's rows of her cell 3 feature_panel.
ONE_CASE = r'''
import json, time
t0 = time.time()
import pandas as pd
import lb_her_nb14 as her14
ns = {"__name__": "her_notebook_14"}
for i in (1, 2):
    exec(compile(her14.CELL_SRC[i], f"<notebook 14 cell {i}>", "exec"), ns)
fp = pd.read_csv("case.csv")
fp["date"] = pd.to_datetime(fp["date"])
ns["feature_panel"] = fp
for i in (4, 5, 6, 7, 8):
    exec(compile(her14.CELL_SRC[i], f"<notebook 14 cell {i}>", "exec"), ns)
(key, fit), = ns["final_fit_store"].items()
case = ns["stack_case_store"][key]
print(json.dumps({"seconds": time.time() - t0, "case": list(key), "T0": len(case["pre_dates"]),
                  "n": len(case["post_dates"]), "m_star": fit["m_star"], "J_star": len(fit["selected_cols"]),
                  "R": f"{fit['final_R']:.4f}"}))
'''


def _sandbox(workdir, script):
    """One bash command in the skill sandbox: the mounts, environment and limits of lb_skill_loop.Cell.do_shell."""
    cmd = [lb_sandbox.BWRAP, *lb_sandbox.MOUNTS, "--bind", str(workdir), "/work", "--chdir", "/work",
           "--unshare-all", "--die-with-parent", "--new-session", "--", lb_sandbox.BASH, "-c", script]
    env = {"PATH": f"{L.PY_DIR}:/usr/bin:/bin", "HOME": "/work", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=L._limits())
    try:
        out, err = p.communicate(timeout=L.CMD_TIMEOUT)
        code = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, err = p.communicate()
        code, err = -9, err + f"\n[killed: exceeded {L.CMD_TIMEOUT} s]"
    return code, out, err, time.time() - t0


def dk3():
    t1 = pd.read_csv(lb.LB / "lb_her14_printed_table1.csv")
    t1["visits"] = t1["T0_pre"] + t1["n_post"]
    big = t1.sort_values("visits", ascending=False).iloc[0]
    ns, _, _ = lb_her14_run.run_cells(cells=[1, 2, 3])
    fp = ns["feature_panel"]
    rows = fp.loc[fp["site_id"].eq(big["site_id"]) & fp["sensor"].eq(big["sensor"])]
    work = Path(tempfile.mkdtemp(prefix="lb_dk3_"))
    try:
        rows.to_csv(work / "case.csv", index=False)
        shutil.copy(lb.LB / "lb_her_nb14.py", work / "lb_her_nb14.py")
        (work / "one_case.py").write_text(ONE_CASE)
        code, out, err, wall = _sandbox(work, "python one_case.py")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if code != 0:
        print(err[-3000:])
        raise SystemExit(f"DK3 FAIL: exit {code} after {wall:.1f} s on {big['sensor']} {big['site_id']}")
    res = json.loads(out.strip().splitlines()[-1])
    hers = {"T0": int(big["T0_pre"]), "n": int(big["n_post"]), "m_star": int(big["m_star"]), "J_star": int(big["J_star"]),
            "R": f"{big['R_at_m_star']:.4f}"}
    ours = {k: res[k] for k in hers}
    print(f"{big['sensor']} {big['site_id']}: {big['visits']} visits; one-case run {ours}, her printed {hers}")
    print(f"seconds inside python {res['seconds']:.1f}, command wall time {wall:.1f}, limit {L.CMD_TIMEOUT}")
    ok = ours == hers and wall <= L.CMD_TIMEOUT
    print("DK3 " + ("ok" if ok else "FAIL"))
    return ok


# ---- DK4 and DK5 ----

def load_cases():
    """lb_cases_daily.csv read exactly (pandas' default float parser is not round-trip exact)."""
    return pd.read_csv(lb.LB / "lb_cases_daily.csv", parse_dates=["date"], float_precision="round_trip")


def case_inputs(cases, sensor, site):
    """-> (pre: band, date, y of the visits given; to_predict: [(period_id, 'YYYY-MM-DD')]; T0)."""
    g = cases.loc[cases["sensor"].eq(sensor) & cases["site_id"].eq(site)]
    pre = g.loc[g["part"].eq("pre"), ["band", "date", "y"]].reset_index(drop=True)
    post = g.loc[g["part"].eq("post")].drop_duplicates("period_id").sort_values("period_id")
    tp = [(p, d.strftime("%Y-%m-%d")) for p, d in zip(post["period_id"], post["date"])]
    return pre, tp, pre["date"].nunique()


def dk4():
    cases = load_cases()
    eff = pd.read_csv(lb.LB / "lb_her14_effects.csv", parse_dates=["date"], float_precision="round_trip")
    pairs = cases[["sensor", "site_id"]].drop_duplicates().itertuples(index=False, name=None)
    n_vals, worst, bad = 0, 0.0, []
    for sensor, site in pairs:
        pre, tp, T0 = case_inputs(cases, sensor, site)
        spec = S.validate_spec(S.HER_SPEC, sensor, T0, tp)
        out = F.run_case(pre, tp, sensor, spec, engine="her")
        ref = eff.loc[eff["sensor"].eq(sensor) & eff["site_id"].eq(site)]
        date_of = dict(tp)
        for _, r in ref.iterrows():
            pid = next(p for p, d in tp if d == r["date"].strftime("%Y-%m-%d"))
            pred = out["predictions"][lb.disp(r["band"])][pid]
            obs = r["observed_y"]
            cf = obs - (obs - pred)                     # her cell 10: gamma = y_sa - synth, y0 = y_sa - gamma
            d1 = abs(pred - r["y_star_synthetic"])
            same_cf = (cf == r["counterfactual_y0"]) or (np.isnan(cf) and np.isnan(r["counterfactual_y0"]))
            n_vals += 1
            worst = max(worst, d1)
            if d1 != 0 or not same_cf:
                bad.append((sensor, site, r["band"], date_of[pid], pred, r["y_star_synthetic"]))
        print(f"  {sensor} {site}: m* {out['m_star']}, J* {len(out['donor_ids'])}, "
              f"{len(ref)} values, {sum(1 for b in bad if b[1] == site)} differ")
    print(f"{n_vals:,} values over {cases[['sensor', 'site_id']].drop_duplicates().shape[0]} cases; "
          f"largest |difference| {worst:.3e}; {len(bad)} values differ")
    for b in bad[:10]:
        print("   ", b)
    ok = not bad and n_vals == len(eff)
    print("DK4 " + ("ok" if ok else "FAIL"))
    return ok


def _smallest_case(cases, sensor):
    sizes = cases.loc[cases["sensor"].eq(sensor)].groupby("site_id")["period_id"].nunique()
    return sizes.idxmin()


def dk5():
    cases = load_cases()
    ok = True
    sensor = "sentinel1"
    pre, tp, T0 = case_inputs(cases, sensor, _smallest_case(cases, sensor))
    n = len(tp)
    breaks = [
        ({"m_min": 20, "m_max": 10}, "'m_min','m_max'"),
        ({"m_max": T0 - n + 1}, "'m_max'"),
        ({"m_max": None, "m_min": T0 - n + 1}, "'m_min','m_max'"),
        ({"readd_seasonal": True, "fit_space": "raw"}, "'readd_seasonal','fit_space'"),
        ({"intercept": "fitted"}, "'intercept','fit_space','readd_seasonal'"),
        ({"weight_constraint": "nonneg"}, "'weight_constraint','fit_space','readd_seasonal'"),
        ({"band_subset": ["VV"]}, "'band_subset'"),
        ({"band_weighting": "subset", "band_subset": [tp[0][0]]}, f"refers to {tp[0][0]}"),
    ]
    for change, field in breaks:
        try:
            S.validate_spec({**S.HER_SPEC, **change}, sensor, T0, tp)
            good, msg = False, "accepted"
        except S.SpecError as e:
            good, msg = field in str(e), str(e)
        ok &= good
        print(f"  refuse {change}: {'ok' if good else 'FAIL'} ({msg[:160]})")
    try:
        S.class1_only({**{k: S.HER_SPEC[k] for k in S.CLASS1}, "fit_space": "raw"}, sensor, T0, tp)
        good = False
    except S.SpecError as e:
        good = "protocol violation" in str(e) and "fit_space" in str(e)
    ok &= good
    print(f"  shc-configure refuses a Class 2 field: {'ok' if good else 'FAIL'}")
    print("DK5 " + ("ok" if ok else "FAIL"))
    return ok


# ---- DK6 to DK10 ----

def _canned(script):
    it = iter(script)
    return lambda messages: (next(it), {"finish_reason": "stop", "reasoning": "canned", "gateway": "test",
                                        "raw_response": {"choices": [{"message": {"content": "…", "reasoning": "canned"}}]}})


def _fence(cmd):
    return f"```bash\n{cmd}\n```"


def _write(name, text):
    return _fence(f"cat > {name} <<'EOF'\n{text}\nEOF")


HER_SETTINGS = {k: S.HER_SPEC[k] for k in S.CLASS1}
DRY = ("sentinel1", "treatment_0001")


def dk6():
    BD.check()
    return True


def dk7():
    cases = load_cases()
    sentinels = {}
    for i, (sensor, site) in enumerate((("sentinel1", "treatment_0001"), ("sentinel2", "treatment_0001"))):
        mask = cases["sensor"].eq(sensor) & cases["site_id"].eq(site) & cases["part"].eq("post")
        for j, b in enumerate(lb.BANDS[sensor]):
            sentinels[f"{sensor} {b}"] = lb_guard.SENTINEL_BASE + 100 * i + j
            cases.loc[mask & cases["band"].eq(b), "y"] = sentinels[f"{sensor} {b}"]
    ok = True
    for sensor, site in (("sentinel1", "treatment_0001"), ("sentinel2", "treatment_0001")):
        given, tp = RD.case_parts(cases, sensor, site)
        cell = RD.DailyCell(BD.SKILLS / "shc-design", (given, tp), sensor, site)
        try:
            data, _ = cell.do_shell("cat data.csv")
            vis, _ = cell.do_shell("cat visits_to_predict.csv")
            n_rows = data.count("\n")
            good = (n_rows == len(given) and set(given["period_id"]).isdisjoint(p for p, _ in tp)
                    and vis.count("\n") == len(tp))
            cell.do_shell(f"cat > spec.json <<'EOF'\n{json.dumps(S.HER_SPEC)}\nEOF")
            out, meta = cell.do_shell("python scripts/fit.py spec.json")
            shown, _ = cell.do_shell("python scripts/show_data.py")
            good &= meta["exit"] == 0
            lb_guard.assert_no_sentinel(data + vis + out + shown + json.dumps(cell.fit_records), sentinels,
                                        f"daily cell {sensor} {site}")
            probe, _ = cell.do_shell(f"ls {lb.LB} 2>&1; ls /home/grads/junh 2>&1; "
                                     "python -c \"import socket; socket.create_connection(('8.8.8.8', 53), 2)\" 2>&1 | tail -1; "
                                     "echo x >> SKILL.md 2>&1; echo x > scripts/fit.py 2>&1; ls scripts")
            good &= probe.count("No such file or directory") >= 2 and "Read-only file system" in probe
            good &= ("Network is unreachable" in probe or "OSError" in probe)
            print(f"  {sensor} {site}: {n_rows} rows given, {len(tp)} visits to predict, no sentinel in any file or "
                  f"output; isolation {'ok' if good else 'FAIL'}")
            ok &= good
        finally:
            cell.close()
    given, tp = RD.case_parts(load_cases(), *DRY)
    cell = RD.DailyCell(BD.SKILLS / "shc-configure", (given, tp), *DRY)
    try:
        cell.do_shell(f"cat > settings.json <<'EOF'\n{json.dumps({**HER_SETTINGS, 'fit_space': 'raw'})}\nEOF")
        out, meta = cell.do_shell("python scripts/fit.py settings.json")
        good = meta["exit"] != 0 and "protocol violation" in out and "fit_space" in out
        print(f"  shc-configure fit.py refuses a choice field: {'ok' if good else 'FAIL'}")
        ok &= good
    finally:
        cell.close()
    print("DK7 " + ("ok" if ok else "FAIL"))
    return ok


def dk8():
    cases = load_cases()
    eff = pd.read_csv(lb.LB / "lb_her14_effects.csv", parse_dates=["date"], float_precision="round_trip")
    sensor, site = DRY
    ref = eff.loc[eff["sensor"].eq(sensor) & eff["site_id"].eq(site)]
    given, tp = RD.case_parts(cases, sensor, site)
    pid_of = {d: p for p, d in tp}
    expected = {}
    for _, r in ref.iterrows():
        expected.setdefault(pid_of[r["date"].strftime("%Y-%m-%d")], {})[lb.disp(r["band"])] = r["y_star_synthetic"]
    to_answer = ("python -c \"import json; r=[json.loads(l) for l in open('.fit_log.jsonl')][-1]; "
                 "p=r['predictions']; v=sorted({k for b in p.values() for k in b}); "
                 "json.dump({'prediction': {k: {b: p[b][k] for b in p} for k in v}, 'rationale': 'dry run'}, open('answer.json','w'))\"")
    scripts = {
        "shc-configure": [_fence("cat SKILL.md"), _fence("python scripts/show_data.py"), _fence("cat references/settings.md"),
                          _write("settings.json", json.dumps(HER_SETTINGS)), _fence("python scripts/fit.py settings.json"),
                          _fence(to_answer)],
        "shc-design": [_fence("cat SKILL.md"), _fence("ls -R"), _fence("head -n 5 data.csv"), _fence("cat references/method.md"),
                       _write("spec.json", json.dumps(S.HER_SPEC)), _fence("python scripts/fit.py spec.json"),
                       _fence(to_answer)],
    }
    ok = True
    for skill, steps in scripts.items():
        saved = L.OUT_LIMIT
        rec = RD.run_case(BD.SKILLS / skill, cases, sensor, site, _canned(steps), step_cap=12)
        good = rec["ok"] and rec["ended"] == "answer" and rec["n_steps"] == len(steps) and L.OUT_LIMIT == saved
        good &= rec["data_shown"] and "SKILL.md" in rec["refs_read"]
        fr = rec["fit_runs"][-1] if rec["fit_runs"] else None
        diffs = [abs(fr["predictions"][b][p] - v) for p, row in expected.items() for b, v in row.items()] if fr else [np.inf]
        same = good and max(diffs) == 0 and rec["values"] == expected
        print(f"  {skill}: ended {rec['ended']} after {rec['n_steps']} steps; fit record vs her y_star_synthetic: "
              f"{len(diffs)} values, largest |difference| {max(diffs):.3e}; answer equals it: {rec['values'] == expected}")
        ok &= same
    print("DK8 " + ("ok" if ok else "FAIL"))
    return ok


def dk9():
    cases = load_cases()
    seen = []
    script = iter([_fence("python -c \"print('x' * 20000)\""), _fence("true"),
                   _fence("echo '{}' > answer.json")])
    def call(messages):
        seen.append([m["content"] for m in messages])
        return next(script), {"finish_reason": "stop", "reasoning": "", "gateway": "test", "raw_response": {}}
    rec = RD.run_case(BD.SKILLS / "shc-design", cases, *DRY, call, step_cap=3)
    whole = seen[1][-1].count("x") >= 20000 and "[output truncated]" not in seen[1][-1]
    ok = whole and L.OUT_LIMIT == 6000 and rec["ended"] == "step_cap"
    print(f"  the model received {seen[1][-1].count('x'):,} of 20,000 characters; OUT_LIMIT after the case {L.OUT_LIMIT}")
    print("DK9 " + ("ok" if ok else "FAIL"))
    return ok


def dk10():
    import tempfile
    cases = load_cases()
    sensor, site = DRY
    bands = [lb.disp(b) for b in lb.BANDS[sensor]]
    _, tp = RD.case_parts(cases, sensor, site)
    valid = json.dumps({"prediction": {p: {b: 1 for b in bands} for p, _ in tp}, "rationale": "x"})
    ok = True
    rec = RD.run_case(BD.SKILLS / "shc-configure", cases, sensor, site, _canned([_fence("ls")] * 20), step_cap=5)
    good = rec["ended"] == "step_cap" and rec["n_steps"] == 5 and not rec["ok"]
    print(f"  step cap: {'ok' if good else 'FAIL'}"); ok &= good
    seen = []
    script = iter(["", "   \n", _fence("ls"), _fence("true"), _write("answer.json", valid)])
    def call(messages):
        seen.append([m["content"] for m in messages])
        assert all(m["content"].strip() for m in messages), "empty message content sent to the model"
        return next(script), {"finish_reason": "length", "reasoning": "…", "gateway": "test", "raw_response": {}}
    rec = RD.run_case(BD.SKILLS / "shc-configure", cases, sensor, site, call, step_cap=6)
    good = (rec["ok"] and rec["n_steps"] == 5 and rec["steps"][0]["reply"] == "" and "(empty reply)" in seen[1][2]
            and rec["steps"][3]["output"] == "" and seen[4][-1] == "(no output)")
    print(f"  empty reply and empty output carried forward, case still answers: {'ok' if good else 'FAIL'}"); ok &= good
    partial = json.dumps({"prediction": {tp[0][0]: {b: 1 for b in bands}}, "rationale": "x"})
    rec = RD.run_case(BD.SKILLS / "shc-configure", cases, sensor, site, _canned([_write("answer.json", partial)] * 2), step_cap=2)
    msg = rec["steps"][0]["output"]
    good = not rec["ok"] and "every visit listed" in msg and tp[1][0] not in msg and "VV" not in msg
    print(f"  an answer missing visits is refused without naming visits or bands: {'ok' if good else 'FAIL'}"); ok &= good
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "replies.jsonl"
        f.write_text("".join(json.dumps({"prompt_id": p, "ended": e}) + "\n" for p, e in
                             [("a", "answer"), ("b", "step_cap"), ("c", "call_error")]))
        good = lb_skill_run.resume_set(f) == {"a", "b"}
    print(f"  resume re-runs only call-error cases: {'ok' if good else 'FAIL'}"); ok &= good
    print("DK10 " + ("ok" if ok else "FAIL"))
    return ok


def _csv_frame(df):
    """A frame as its csv text reads back, so a frame in memory and a frame from a file compare on equal terms."""
    import io as _io
    return pd.read_csv(_io.StringIO(df.to_csv(index=False)), float_precision="round_trip")


def _sorted(df, keys):
    k = [c for c in keys if c in df.columns]
    return df.sort_values(k, kind="mergesort").reset_index(drop=True) if k else df.reset_index(drop=True)


def dk11():
    """The scorer at her settings with her own predictions over her 52 cases reproduces our run of her code with
    difference 0 and her printed tables within 0.0001 (as DK2), and draws her 52 figures."""
    import lb_skill_score_daily as SD
    import lb_her14_run as H
    stores = SD.run_jobs({(k[0], k[1], SD.spec_json(SD.S.HER_SPEC), tuple(SD.HER_CELLS)) for k in SD.CASE_KEYS})
    tables, ns, log = SD.her_row(stores)
    ok = not log
    print(f"  statements raised: {len(log)}")
    keys = ["site_id", "sensor", "band", "date"]
    for name, ref in (("daily_effects", "lb_her14_effects.csv"), ("effect_by_site_band", "lb_her14_effect_by_site_band.csv"),
                      ("effect_summary", "lb_her14_effect_summary.csv"), ("conformal_inference", "lb_her14_inference.csv")):
        a = _sorted(_csv_frame(ns[name]), keys)
        b = _sorted(pd.read_csv(lb.LB / ref, float_precision="round_trip"), keys)
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            same = True
        except AssertionError as e:
            same = False
            print(f"    {name}: {str(e).splitlines()[0][:200]}")
        print(f"  {name} vs {ref}: {len(a)} rows, {'difference 0' if same else 'FAIL'}"); ok &= same
    for name, df in tables.items():
        her = pd.read_csv(lb.LB / f"lb_her14_printed_{name}.csv", dtype=str, keep_default_na=False)
        ours = df.astype(str)
        tk = ["sensor", "site_id", "band", "check", "selection_mode"]
        good = list(ours.columns) == list(her.columns) and len(ours) == len(her)
        bad = 0
        if good:
            o, h = _sorted(ours, tk), _sorted(her, tk)
            bad = sum(not (x == y or H.within_0001(x, y)) for c in o.columns for x, y in zip(o[c], h[c]))
            good = bad == 0
        print(f"  printed {name}: {len(ours)} rows, {'ok' if good else f'FAIL ({bad} values beyond 0.0001)'}")
        ok &= good
    n_fig = len(list((SD.FIG_ROOT / "her").glob("*.png")))
    print(f"  figures written: {n_fig} of 52"); ok &= n_fig == 52
    print("DK11 " + ("ok" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    gates = {"dk3": dk3, "dk4": dk4, "dk5": dk5, "dk6": dk6, "dk7": dk7, "dk8": dk8, "dk9": dk9, "dk10": dk10,
             "dk11": dk11}
    names = sys.argv[1:] or list(gates)
    sys.exit(0 if all(gates[n]() for n in names) else 1)
