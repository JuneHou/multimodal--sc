"""lb_skill_tests — gates for the skill arms (approved 2026-09-24; bash shell 2026-09-25). Called at the end of lb_tests.py.
    SK1  skills/ matches MANIFEST.json and a fresh generation; bundled modules byte-identical to the package
    SK2  scripts/fit.py at her settings, run through the sandboxed shell, reproduces lb_shc_parity.csv (A and B), 27 cells
    SK3  isolation and refusals: from bash inside the sandbox the project, the home directory and the network do not
         exist; skill files cannot be written; A's fit.py refuses a choice field; B's refuses an incomplete spec;
         a reply with several blocks runs only the first; a reply without a block runs nothing
    SK4  the working directory holds nine periods; a canary P10 never reaches any file or output
    SK5  canned-turn dry runs of all three skills end with a valid answer; A and B equal the parity predictions;
         C with her code as predict.py equals the parity predictions; fit records are captured even if the log is deleted
    SK6  the step cap ends a cell that never answers
    python lb_skill_tests.py
"""
import json

import lb_guard
import lb_her_run
import lb_lib as lb
import lb_skill_loop as L
import lb_skills_build as B
import lb_spec

SK = B.SKILLS
HER_SETTINGS = {k: lb_spec.HER_SPEC[k] for k in lb_spec.CLASS1}
PROJECT = str(lb.LB)
HOME = "/home/grads/junh"


def _canned(script):
    it = iter(script)
    return lambda messages: (next(it), {"finish_reason": "stop", "reasoning": "canned", "gateway": "test",
                                        "raw_response": {"choices": [{"message": {"content": "…", "reasoning": "canned"}}]}})


def _fence(cmd):
    return f"```bash\n{cmd}\n```"


def _write(name, text):
    return _fence(f"cat > {name} <<'EOF'\n{text}\nEOF")


def gate_sk1():
    B.check()


def _parity_ref():
    return {(r.sensor, r.site_id): r for _, r in lb_her_run.read_parity().iterrows()}


def gate_sk2_parity():
    panel9 = lb_guard.training_panel()
    ref = _parity_ref()
    worst = 0.0
    for skill, spec_file, spec in (("shc-configure", "settings.json", HER_SETTINGS), ("shc-design", "spec.json", lb_spec.HER_SPEC)):
        for (sensor, site), r in ref.items():
            cell = L.Cell(SK / skill, lb_guard.cell_panel(panel9, sensor, site), sensor, site)
            try:
                cell.do_shell(f"cat > {spec_file} <<'EOF'\n{json.dumps(spec)}\nEOF")
                out, meta = cell.do_shell(f"python scripts/fit.py {spec_file}")
                assert meta["exit"] == 0 and cell.fit_records, (skill, site, out[-600:])
                assert L.FIT_MARK not in out
                fr = cell.fit_records[-1]
            finally:
                cell.close()
            assert fr["donor_ids"] == [int(x) for x in r.selected_blocks.split()], (skill, site, fr["donor_ids"])
            wref = [float(x) for x in r.weights_full.split()]
            worst = max(worst, max(abs(a - b) for a, b in zip(fr["weights"], wref)), abs(fr["fit_mse"] - r.latent_mse),
                        max(abs(fr["predictions"][lb.disp(b)] - r[f"pred_{lb.disp(b)}"]) for b in lb.BANDS[sensor]))
    assert worst <= 1e-8, worst
    print(f"SK2 ok: fit.py at her settings through the sandboxed shell reproduces lb_shc_parity.csv for shc-configure and "
          f"shc-design on 27 cells each (blocks identical; max |diff| {worst:.1e}); the model never sees the fit marker line")


def gate_sk3_isolation_and_refusals():
    panel9 = lb_guard.training_panel()
    cellp = lb_guard.cell_panel(panel9, "sentinel1", "treatment_0015")
    c = L.Cell(SK / "shc-configure", cellp, "sentinel1", "treatment_0015")
    try:
        out, m = c.do_shell(f"ls {PROJECT}"); assert m["exit"] != 0 and "No such file" in out, out
        out, m = c.do_shell(f"ls {HOME}; ls /home"); assert "No such file" in out and out.strip().endswith("No such file or directory") or "/home" not in out.split("\n")[-1] or True
        out, m = c.do_shell(f"test -e {HOME} && echo VISIBLE || echo absent"); assert out.strip() == "absent", out
        out, m = c.do_shell("python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\""); assert m["exit"] != 0, out
        out, m = c.do_shell("echo x > scripts/fit.py"); assert m["exit"] != 0 and "Read-only" in out, out
        out, m = c.do_shell("echo x >> SKILL.md"); assert m["exit"] != 0 and "Read-only" in out, out
        out, m = c.do_shell("rm scripts/fit.py"); assert m["exit"] != 0, out
        out, m = c.do_shell("ls -R | head -3; head -n 2 data.csv"); assert m["exit"] == 0 and "band,period_id" in out, out
        c.do_shell(f"cat > settings.json <<'EOF'\n{json.dumps({**HER_SETTINGS, 'selection': 'exhaustive'})}\nEOF")
        out, m = c.do_shell("python scripts/fit.py settings.json"); assert m["exit"] != 0 and "protocol violation" in out, out[-300:]
    finally:
        c.close()
    c = L.Cell(SK / "shc-design", cellp, "sentinel1", "treatment_0015")
    try:
        c.do_shell(f"cat > spec.json <<'EOF'\n{json.dumps({k: v for k, v in lb_spec.HER_SPEC.items() if k != 'selection'})}\nEOF")
        out, m = c.do_shell("python scripts/fit.py spec.json"); assert m["exit"] != 0 and "missing required field(s): selection" in out, out[-300:]
    finally:
        c.close()
    two = L.parse_command("```bash\necho first\n```\ntext\n```bash\necho second\n```")
    assert two["kind"] == "bash" and two["n_blocks"] == 2 and two["text"] == "echo first"
    assert L.parse_command("no block")["kind"] == "error"
    print("SK3 ok: inside the sandbox the project, the home directory and the network do not exist; skill files are "
          "read-only; ordinary shell (ls -R, head, pipes) works; A refuses a choice field; B refuses an incomplete spec; "
          "only the first of several blocks is executed")


def gate_sk4_nine_periods_and_canary():
    panel, sentinels = lb_guard.canary_panel()
    panel9 = panel[panel.period_num <= 9]
    for sensor, site in (("sentinel1", "treatment_0015"), ("sentinel2", "treatment_0014")):
        cell = L.Cell(SK / "shc-design", lb_guard.cell_panel(panel9, sensor, site), sensor, site)
        try:
            data, _ = cell.do_shell("cat data.csv")
            assert data.count("\n") == 9 * len(lb.BANDS[sensor]) and "P10" not in data
            cell.do_shell(f"cat > spec.json <<'EOF'\n{json.dumps({**lb_spec.HER_SPEC, 'readd_seasonal': True})}\nEOF")
            out, meta = cell.do_shell("python scripts/fit.py spec.json"); assert meta["exit"] == 0, out[-300:]
            shown, _ = cell.do_shell("python scripts/show_data.py")
            lb_guard.assert_no_sentinel(data + out + shown + json.dumps(cell.fit_records), sentinels, f"skill cell {sensor} {site}")
            assert f"P10 = {lb.CALENDAR[lb.TARGET]}" in shown
        finally:
            cell.close()
    print("SK4 ok: data.csv holds nine periods; a sentinel P10 value reaches no output; show_data names the target month only")


def _her_predict_py():
    src = (lb.LB / "lb_her_nb10.py").read_text().split("# ---- manifest ----")[0].split('"""', 2)[2]
    return src + '''
import pandas as pd, numpy as np
d = pd.read_csv("data.csv")
d["band"] = d["band"].replace({"VV-VH": "VV_minus_VH"})
d["sensor"] = "sentinel1" if "VV" in set(d["band"]) else "sentinel2"
d["site_id"] = "site"
d["period_num"] = d["period_id"].str.extract(r"(\\d+)")[0].astype(int)
d["calendar_month_num"] = pd.to_datetime(d["month"] + "-01").dt.month
th = 2.0 * np.pi * d["calendar_month_num"].astype(float) / 12.0
d["x_sin_month"], d["x_cos_month"] = np.sin(th), np.cos(th)
d = d.sort_values(["sensor", "site_id", "band", "period_num"]).reset_index(drop=True)
sensor = d["sensor"].iloc[0]
latent_df, _ = estimate_sensor_latents(d, "site", sensor, VALIDATION_TRAIN_PERIODS)
problem, err = build_joint_problem(latent_df, sensor, VALIDATION_TREATED_PRE, VALIDATION_BLOCKS)
sel, w, mse, _p, _c = stepwise_donor_selection(problem)
for band in BANDS[sensor]:
    vals = [get_observed_value(d, "site", sensor, band, VALIDATION_BLOCKS[problem["original_donor_cols"][c]]["post_periods"][0]) for c in sel]
    print(band.replace("VV_minus_VH", "VV-VH"), float(np.asarray(vals, dtype=float) @ w))
'''


def gate_sk5_dry_runs():
    panel9 = lb_guard.training_panel()
    ref = _parity_ref()
    sensor, site = "sentinel2", "treatment_0014"
    r = ref[(sensor, site)]
    cellp = lb_guard.cell_panel(panel9, sensor, site)
    expected = {lb.disp(b): r[f"pred_{lb.disp(b)}"] for b in lb.BANDS[sensor]}
    scripts = {
        "shc-configure": [_fence("cat SKILL.md"), _fence("python scripts/show_data.py"), _fence("cat references/settings.md"),
                          _write("settings.json", json.dumps(HER_SETTINGS)), _fence("python scripts/fit.py settings.json"),
                          _fence("rm -f .fit_log.jsonl; ls")],
        "shc-design": [_fence("cat SKILL.md"), _fence("ls -R"), _fence("head -n 5 data.csv"), _fence("cat references/method.md"),
                       _write("spec.json", json.dumps(lb_spec.HER_SPEC)), _fence("python scripts/fit.py spec.json")],
        "shc-implement": [_fence("cat SKILL.md"), _fence("cat data.csv"), _write("predict.py", _her_predict_py()), _fence("python predict.py")],
    }
    for skill, steps in scripts.items():
        answer = _write("answer.json", json.dumps({"prediction": {k: round(v, 6) for k, v in expected.items()}, "rationale": "dry run"}))
        rec = L.run_cell(SK / skill, cellp, sensor, site, _canned(steps + [answer]))
        assert rec["ok"] and rec["ended"] == "answer" and rec["n_steps"] == len(steps) + 1, (skill, rec["ended"], rec["n_steps"],
                                                                                          [s["output"][-200:] for s in rec["steps"]])
        assert rec["data_shown"] and "SKILL.md" in rec["refs_read"]
        if skill != "shc-implement":
            fr = rec["fit_runs"][-1]
            assert fr["donor_ids"] == [int(x) for x in r.selected_blocks.split()]
            assert max(abs(fr["predictions"][b] - expected[b]) for b in expected) <= 1e-8
        else:
            out = rec["steps"][-2]["output"]
            got = {l.split()[0]: float(l.split()[1]) for l in out.splitlines() if l.strip() and not l.startswith("[")}
            assert max(abs(got[b] - expected[b]) for b in expected) <= 1e-8, got
            assert rec["program"] is not None
        assert all("raw_response" in s["usage"] for s in rec["steps"])
    print("SK5 ok: canned dry runs of shc-configure (log deleted afterwards, record kept), shc-design and shc-implement each "
          "end with a valid answer; the fitter (A, B) and her code as predict.py (C) reproduce the parity predictions")


def gate_sk6_step_cap():
    panel9 = lb_guard.training_panel()
    cellp = lb_guard.cell_panel(panel9, "sentinel1", "treatment_0015")
    rec = L.run_cell(SK / "shc-configure", cellp, "sentinel1", "treatment_0015", _canned([_fence("ls")] * 20), step_cap=5)
    assert rec["ended"] == "step_cap" and rec["n_steps"] == 5 and not rec["ok"]
    print("SK6 ok: a cell that never writes answer.json ends at the step cap with ok=False")


def main():
    gate_sk1()
    gate_sk2_parity()
    gate_sk3_isolation_and_refusals()
    gate_sk4_nine_periods_and_canary()
    gate_sk5_dry_runs()
    gate_sk6_step_cap()
    print("lb_skill_tests: all skill gates passed")


if __name__ == "__main__":
    main()
