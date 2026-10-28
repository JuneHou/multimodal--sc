"""lb_skill_run_daily — run one daily skill arm over the 52 cases of notebook 14 (plan ~/.claude/plans/simple-skills-daily.md,
step 8).

    python lb_skill_run_daily.py --skill shc-configure|shc-design [--models ...] [--effort low] [--limit N] [--seed 0]
                                 [--step-cap 12]
    python lb_skill_run_daily.py --show <reply file> [--cell <prompt_id>]     print a case's transcript turn by turn

The loop is lb_skill_loop.run_cell, unchanged: the same SYSTEM text, bash sandbox, step cap, placeholders and resume
rule. Three names of the loop are set for each daily case and restored after it, as sf_run.py does:
  Cell       DailyCell: the working directory holds data.csv (the visits given: band, period_id, date, y) and
             visits_to_predict.csv (period_id, date), and the answer is checked in the daily form, every band at every
             visit to predict; no value after the visits given is passed to the cell
  task_text  the monthly task text with "for the next month, P10 = <month>, from its own past months" replaced by
             "at each visit listed in visits_to_predict.csv, from its own past visits"
  OUT_LIMIT  none: the model sees each command's output whole, Jun "no limit" (2026-10-15)
Token caps: 32,000 for DeepSeek, 16,000 for the other models. Records go to lb_replies_daily_<arm>_<model>_<effort>.jsonl.
Keys are read from the environment only (lb_lib.api_key).
"""
import argparse
import json
import sys

import pandas as pd

import lb_lib as lb
import lb_skill_loop as L
import lb_skill_run as R
import lb_skills_build_daily as B

SKILLS = {"shc-configure": "shc_configure", "shc-design": "shc_design"}
OUT_LIMIT = sys.maxsize


def max_tokens_for(model):
    """32,000 for DeepSeek, 16,000 for the other solver models (Jun "13. 32000 tokens", 2026-10-13)."""
    return 32000 if "deepseek" in model.lower() else 16000


def task_text_daily(skill_dir):
    fm = L.frontmatter(skill_dir)
    return ("Predict this land parcel's mean band values at each visit listed in visits_to_predict.csv, "
            "from its own past visits.\nThe working directory holds a skill for this task:\n"
            f"  {fm['name']} — {fm['description']}\n"
            "Its instructions are in SKILL.md; read that file first and follow it. Reply with one ```bash block per turn; "
            "it is run by bash in the working directory and its output comes back in the next message. "
            "Finish by writing answer.json as the skill describes.")


class DailyCell(L.Cell):
    """L.Cell with the daily working directory and the daily answer form; the shell is the parent's."""

    def __init__(self, skill_dir, case, sensor, site):
        given, to_predict = case
        self.skill = L.Path(skill_dir)
        self.sensor, self.site = sensor, site
        self.bands = [lb.disp(b) for b in lb.BANDS[sensor]]
        self.pids = [p for p, _ in to_predict]
        self.dir = L.Path(L.tempfile.mkdtemp(prefix="lb_skill_daily_"))
        assert set(given.columns) == {"band", "period_id", "date", "y"}, given.columns
        assert not set(self.pids) & set(given["period_id"]), "a visit to predict is among the visits given"
        d = given.assign(band=[lb.disp(b) for b in given["band"]], date=given["date"].dt.strftime("%Y-%m-%d"))
        d.sort_values(["date", "band"])[["band", "period_id", "date", "y"]].to_csv(self.dir / "data.csv", index=False)
        pd.DataFrame(to_predict, columns=["period_id", "date"]).to_csv(self.dir / "visits_to_predict.csv", index=False)
        (self.dir / "SKILL.md").touch()                    # mount points for the read-only binds
        (self.dir / "references").mkdir()
        (self.dir / "scripts").mkdir()
        self.refs_read, self.data_shown, self.python_runs, self.fit_records = [], False, 0, []

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
        if any(p_ not in pred or not isinstance(pred[p_], dict) or any(b not in pred[p_] for b in self.bands)
               for p_ in self.pids):              # never name the visits or the bands, as the monthly check
            return None, ("answer.json: prediction must give, for every visit listed in visits_to_predict.csv under "
                          "its period_id, a number for every band of this parcel under the band names the parcel's "
                          "data uses")
        try:
            vals = {p_: {b: float(pred[p_][b]) for b in self.bands} for p_ in self.pids}
        except (TypeError, ValueError) as e:
            return None, f"answer.json: every prediction must be a number ({e})"
        if any(v != v for row in vals.values() for v in row.values()):
            return None, "answer.json: a prediction is NaN"
        return {"prediction": vals, "rationale": str(a.get("rationale", ""))}, None


def load_cases():
    return pd.read_csv(lb.LB / "lb_cases_daily.csv", parse_dates=["date"], float_precision="round_trip")


def case_parts(cases, sensor, site):
    """-> (visits given: band, period_id, date, y; [(period_id, 'YYYY-MM-DD')] of the visits to predict). The
    observed values of the visits to predict are not returned."""
    g = cases.loc[cases["sensor"].eq(sensor) & cases["site_id"].eq(site)]
    given = g.loc[g["part"].eq("pre"), ["band", "period_id", "date", "y"]].reset_index(drop=True)
    post = g.loc[g["part"].eq("post")].drop_duplicates("period_id").sort_values("period_id")
    return given, [(p, d.strftime("%Y-%m-%d")) for p, d in zip(post["period_id"], post["date"])]


def run_case(skill_dir, cases, sensor, site, call, step_cap):
    """L.run_cell with DailyCell, the daily task text and no output limit; the loop's names are restored after."""
    saved = L.Cell, L.task_text, L.OUT_LIMIT
    L.Cell, L.task_text, L.OUT_LIMIT = DailyCell, task_text_daily, OUT_LIMIT
    try:
        return L.run_cell(skill_dir, case_parts(cases, sensor, site), sensor, site, call, step_cap=step_cap)
    finally:
        L.Cell, L.task_text, L.OUT_LIMIT = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", choices=list(SKILLS))
    ap.add_argument("--models", nargs="+", default=["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low", "openai:gpt-5.5"])
    ap.add_argument("--effort", default=lb.REASONING_EFFORT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--step-cap", type=int, default=L.STEP_CAP)
    ap.add_argument("--show", default="")
    ap.add_argument("--cell", default="")
    a = ap.parse_args()
    if a.show:
        return R.show(a.show, a.cell or None)
    if not a.skill:
        ap.error("--skill is required")
    B.check()                                                 # DK6: what is mounted is what was shown to Jun
    skill_dir = B.SKILLS / a.skill
    exp = SKILLS[a.skill]
    effort = None if a.effort == "default" else a.effort
    cases = load_cases()
    pairs = list(cases[["sensor", "site_id"]].drop_duplicates().itertuples(index=False, name=None))
    if a.limit:
        keep, seen = [], {}
        for s, site in pairs:
            if seen.get(s, 0) < a.limit:
                keep.append((s, site)); seen[s] = seen.get(s, 0) + 1
        pairs = keep
    import numpy, scipy
    for spec_model in a.models:
        prov, model, _ = lb.parse_model(spec_model)
        key = lb.api_key(prov)
        max_tokens = max_tokens_for(spec_model)
        env = {"python": sys.executable, "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
               "max_tokens": max_tokens, "step_cap": a.step_cap, "out_limit": "none",
               "skill_manifest": json.loads(B.MANIFEST.read_text())}
        out = lb.LB / f"lb_replies_daily_{exp}_{lb.safe_name(spec_model)}_{a.effort}.jsonl"
        done = R.resume_set(out)
        print(f"{spec_model} / {a.skill} (effort {a.effort}, step cap {a.step_cap}, max tokens {max_tokens}, "
              f"no output limit) -> {out.name}; {len(done)} done")
        call = R.make_call(spec_model, a.seed, key, effort, a.timeout, max_tokens)
        with open(out, "a") as f:
            for sensor, site in pairs:
                pid = f"daily_{lb.SHORT[sensor]}_{site}_{exp[4:]}"
                if pid in done:
                    continue
                rec = run_case(skill_dir, cases, sensor, site, call, a.step_cap)
                last_usage = next((s["usage"] for s in reversed(rec["steps"]) if s.get("usage")), {})
                rec.update(prompt_id=pid, experiment=f"daily_{exp}", model=spec_model, served_model=model, gateway=prov,
                           seed=a.seed, reasoning_effort=a.effort, env=env,
                           usage={k: v for k, v in last_usage.items() if k != "raw_response"},
                           reply=next((s["reply"] for s in reversed(rec["steps"]) if s.get("reply")), ""))
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"  {pid:40s} ok={rec['ok']!s:5s} steps={rec['n_steps']:2d} ended={rec['ended']:9s} "
                      f"fit_runs={len(rec['fit_runs'])} refs={len(rec['refs_read'])} data={rec['data_shown']}")


if __name__ == "__main__":
    main()
