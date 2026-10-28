"""sf_daily — the solver models on a SkillFoundry-generated SHC skill over notebook 14's 52 daily cases, with the task
of the simple daily runs (plan ~/.claude/plans/profile-skills-daily.md; Jun "a", 2026-10-22).

    python sf_daily.py --generator gpt-5.4|deepseek --format code_provided|code_written [--models ...] [--effort low]
                       [--limit N] [--seed 0] [--max-tokens N] [--step-cap 12]
    python sf_daily.py --cells          lists the cases: sensor, site, visits given, visits to predict

Each line from its source:
  cases     the 52 cases of llm_band_sc/lb_cases_daily.csv, read from notebook 14's xlsx panels, in that file's order
            (lb_skill_run_daily.load_cases)
  data      data.csv (band, period_id, date, y: the visits given) and visits_to_predict.csv (period_id, date: no
            values), written as lb_skill_run_daily.DailyCell writes them
  answer    lb_skill_run_daily.DailyCell.answer: every band at every visit listed in visits_to_predict.csv
  message   G16, the simple daily task text without its skill line, with the answer form written out in place of
            "as the skill describes" (Jun "for 2, a", 2026-10-22)
The loop is sf_run's: ProfileCell's mounts (G5), PATH (G8c, G8d) and shell, L.run_cell, load_solver, the token caps of
row 13 and sf_run's OUT_LIMIT (decision OL: no limit). sf_run.py and llm_band_sc/ are not changed. Records go to
runs/replies_daily_<generator>_<format>_<model>_<effort>.jsonl.
"""
import argparse
import contextlib
import functools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import sf_run as S                   # puts llm_band_sc on sys.path
import lb_lib as lb                  # noqa: E402
import lb_skill_loop as L            # noqa: E402
import lb_skill_run as R             # noqa: E402
import lb_skill_run_daily as RD      # noqa: E402

HERE = S.HERE
RESERVED = S.RESERVED + ("visits_to_predict.csv",)
COLUMNS = ["band", "period_id", "date", "y"]


def task_text_daily(skill_dir=None):
    """G16 on the simple daily task text, with the answer form in place of "as the skill describes"."""
    return ("Predict this land parcel's mean band values at each visit listed in visits_to_predict.csv, "
            "from its own past visits.\nThe working directory holds a skill for this task. "
            "Its instructions are in SKILL.md; read that file first and follow it. Reply with one ```bash block per turn; "
            "it is run by bash in the working directory and its output comes back in the next message. "
            'Finish by writing answer.json as {"prediction": {"<period_id>": {"<band>": <value>, ...}, ...}, '
            '"rationale": "<text>"}.')


class ProfileDailyCell(S.ProfileCell):
    """ProfileCell's mounts, PATH and shell with the simple daily working directory and answer check."""

    answer = RD.DailyCell.answer

    def __init__(self, skill_dir, case, sensor, site, env_prefix):
        given, to_predict = case
        self.skill = Path(skill_dir)
        self.entries = sorted(p.name for p in self.skill.iterdir() if p.name != "__pycache__")
        clash = [n for n in RESERVED if n in self.entries]
        assert not clash, f"skill folder holds {clash} at its top level"
        self.env_prefix = Path(env_prefix) if env_prefix else None
        self.env_bin = str(self.env_prefix / "bin") if env_prefix else L.PY_DIR
        self.sensor, self.site = sensor, site
        self.bands = [lb.disp(b) for b in lb.BANDS[sensor]]
        self.pids = [p for p, _ in to_predict]
        self.dir = Path(tempfile.mkdtemp(prefix="sf_daily_"))
        assert set(given.columns) == set(COLUMNS), given.columns
        assert not set(self.pids) & set(given["period_id"]), "a visit to predict is among the visits given"
        d = given.assign(band=[lb.disp(b) for b in given["band"]], date=given["date"].dt.strftime("%Y-%m-%d"))
        d.sort_values(["date", "band"])[COLUMNS].to_csv(self.dir / "data.csv", index=False)
        pd.DataFrame(to_predict, columns=["period_id", "date"]).to_csv(self.dir / "visits_to_predict.csv", index=False)
        for n in self.entries:                          # mount points for the read-only binds
            (self.dir / n).mkdir() if (self.skill / n).is_dir() else (self.dir / n).touch()
        self.ref_re = re.compile(r"\bSKILL\.md\b|(?<![\w./-])(?:" + "|".join(re.escape(n) for n in self.entries) + r")(?:/[\w./-]+)?")
        self.refs_read, self.data_shown, self.python_runs, self.fit_records = [], False, 0, []


@contextlib.contextmanager
def daily_loop(env_prefix):
    """Run L.run_cell with ProfileDailyCell, the daily task text and sf_run's OUT_LIMIT; the loop's names are restored."""
    saved = L.Cell, L.task_text, L.OUT_LIMIT
    L.Cell = functools.partial(ProfileDailyCell, env_prefix=env_prefix)
    L.task_text = task_text_daily
    L.OUT_LIMIT = S.OUT_LIMIT
    try:
        yield
    finally:
        L.Cell, L.task_text, L.OUT_LIMIT = saved


def run_case(skill_dir, env_prefix, cases, sensor, site, call_fn, step_cap=L.STEP_CAP):
    with daily_loop(env_prefix):
        return L.run_cell(skill_dir, RD.case_parts(cases, sensor, site), sensor, site, call_fn, step_cap=step_cap)


def case_pairs(cases):
    return list(cases[["sensor", "site_id"]].drop_duplicates().itertuples(index=False, name=None))


def prompt_id(sensor, site, exp):
    return f"{lb.SHORT[sensor]}_{site}_{exp}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=S.GENERATORS)
    ap.add_argument("--format", choices=S.FORMATS)
    ap.add_argument("--models", nargs="+", default=S.DEFAULT_MODELS)
    ap.add_argument("--effort", default=lb.REASONING_EFFORT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = row 13 (32,000 DeepSeek, 16,000 others)")
    ap.add_argument("--step-cap", type=int, default=L.STEP_CAP)
    ap.add_argument("--cells", action="store_true")
    a = ap.parse_args()
    cases = RD.load_cases()
    pairs = case_pairs(cases)
    if a.cells:
        for sensor, site in pairs:
            given, to_predict = RD.case_parts(cases, sensor, site)
            print(f"{sensor:9s} {site}  given {given['period_id'].nunique():3d} visits  to predict {len(to_predict):2d}")
        print(f"{len(pairs)} cases: " + ", ".join(f"{s} {sum(p[0] == s for p in pairs)}" for s in lb.BANDS))
        return
    if not (a.generator and a.format):
        ap.error("--generator and --format are required")
    skill_dir, env_prefix, skill_sha = S.load_solver(a.generator, a.format)
    exp = f"daily_{a.generator}_{a.format}"
    effort = None if a.effort == "default" else a.effort
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    if a.limit:
        keep, seen = [], {}
        for s, site in pairs:
            if seen.get(s, 0) < a.limit:
                keep.append((s, site)); seen[s] = seen.get(s, 0) + 1
        pairs = keep
    env_py = subprocess.run([str(Path(env_prefix) / "bin" / "python") if env_prefix else sys.executable, "-c",
                             "import sys, importlib.metadata as m; print(sys.version.split()[0]); "
                             "print({d.metadata['Name']: d.version for d in m.distributions()})"],
                            capture_output=True, text=True)
    for spec_model in a.models:
        prov, model, _ = lb.parse_model(spec_model)
        key = lb.api_key(prov)
        mt = a.max_tokens or S.max_tokens_for(spec_model)
        env = {"skill_dir": str(skill_dir), "skill_sha": skill_sha, "env_prefix": env_prefix,
               "python": str(Path(env_prefix) / "bin" / "python") if env_prefix else sys.executable,
               "env_python": env_py.stdout.strip(), "max_tokens": mt, "step_cap": a.step_cap, "out_limit": "none"}
        out = out_dir / f"replies_{exp}_{lb.safe_name(spec_model)}_{a.effort}.jsonl"
        done = R.resume_set(out)
        print(f"{spec_model} / {exp} (effort {a.effort}, max tokens {mt}, step cap {a.step_cap}, no output limit) "
              f"-> {out.name}; {len(done)} done")
        call = R.make_call(spec_model, a.seed, key, effort, a.timeout, mt)
        with open(out, "a") as f:
            for sensor, site in pairs:
                pid = prompt_id(sensor, site, exp)
                if pid in done:
                    continue
                rec = run_case(skill_dir, env_prefix, cases, sensor, site, call, step_cap=a.step_cap)
                last_usage = next((s["usage"] for s in reversed(rec["steps"]) if s.get("usage")), {})
                rec.update(prompt_id=pid, experiment=exp, dataset="daily", generator=a.generator, format=a.format,
                           model=spec_model, served_model=model, gateway=prov, seed=a.seed, reasoning_effort=a.effort,
                           env=env, usage={k: v for k, v in last_usage.items() if k != "raw_response"},
                           reply=next((s["reply"] for s in reversed(rec["steps"]) if s.get("reply")), ""))
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"  {pid:52s} ok={rec['ok']!s:5s} steps={rec['n_steps']:2d} ended={rec['ended']:9s} "
                      f"python_runs={rec['python_runs']} refs={len(rec['refs_read'])} data={rec['data_shown']}")


if __name__ == "__main__":
    main()
