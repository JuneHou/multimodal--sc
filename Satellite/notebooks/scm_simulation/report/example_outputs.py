"""Example outputs page: one scm-configure run per solver model (DGP 1, sigma = 1), every step as recorded, and the
skill each SkillFoundry generator wrote (the Purpose and Procedure sections of its SKILL.md and its full R script).

    python report/example_outputs.py      # from notebooks/scm_simulation/, writes report/example_outputs/index.html

Everything shown is read from the run records and the generated skill folders; the ATT of each counterfactual comes
from score.counterfactual_att, the collaborator's formula (mc_evaluation_r.R line 94). Light theme only (Jun
2026-11-06: "can you change the theme of the current artifact and the report to light?").
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import score  # noqa: E402

OUT = HERE / "report" / "example_outputs" / "index.html"
RUNS = [
    ("gpt-oss-120b", "scm-configure/sigma_1.0/scm-configure_gpt-oss-120b_low.jsonl"),
    ("GLM-5.3", "scm-configure/sigma_1.0/scm-configure_GLM-5.3_low.jsonl"),
    ("GPT-5.5", "scm-configure/sigma_1.0/scm-configure_openai_gpt-5.5_low.jsonl"),
]
GENERATORS = [
    ("GPT-5.4", "profile/skills/gpt-5.4/code_provided", "scripts/run_synth_single_treated_unit_scm.R"),
    ("DeepSeek-V4.1-Flash", "profile/skills/deepseek/code_provided", "scripts/estimate_synthetic_control.R"),
]
CASE, SIGMA, TRUE_ATT = "dgp1", "1.0", -3.0
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

e = html.escape


def count(n, noun):
    return f"{WORDS.get(n, n)} {noun}{'' if n == 1 else 's'}"


def signed(x, plus=False):
    """Three decimals with a typographic minus sign, and a plus sign when asked."""
    t = f"{abs(x):.3f}"
    return ("\u2212" if x < 0 else ("+" if plus else "")) + t


def run_record(path):
    for line in open(HERE / path):
        r = json.loads(line)
        if r["case"] == CASE:
            return r
    raise SystemExit(f"no {CASE} case in {path}")


def section(text, heading):
    """The level-2 section `## heading` of a Markdown file, verbatim, up to the next level-2 heading."""
    m = re.search(rf"^## {re.escape(heading)}\n.*?(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise SystemExit(f"no section {heading}")
    return m.group(0).rstrip()


def box(label, text, cls):
    return (f'<figure class="box {cls}"><figcaption>{e(label)}</figcaption>'
            f'<pre>{e(text) if text.strip() else "<span class=empty>(empty)</span>"}</pre></figure>')


def answer_text(r):
    files = r.get("files") or {}
    if isinstance(files, dict) and isinstance(files.get("answer.json"), str):
        return files["answer.json"]
    a = r["answer"]
    return json.dumps({"counterfactual": a["prediction"]["counterfactual"], "att": a["att"], "rationale": a["rationale"]}, indent=2)


def run_section(model, r):
    att_cf = score.counterfactual_att(CASE, r["answer"]["prediction"]["counterfactual"], SIGMA)
    steps = []
    for s in r["steps"]:
        u = s.get("usage") or {}
        cmd = s.get("command") or {}
        ran = cmd.get("kind") == "bash"
        meta = (f"exit code {s.get('exit')} · {s.get('seconds')} s" if ran else "no command ran")
        parts = [f'<header><span class="stepno">Step {s["step"]}</span><span class="meta">{e(meta)}</span></header>']
        if (u.get("reasoning") or "").strip():
            parts.append(box("Reasoning text in the record", u["reasoning"], "reason"))
        parts.append(box("Reply", s["reply"], "reply"))
        parts.append(box("Output returned to the model", s["output"], "output"))
        steps.append(f'<article class="step">{"".join(parts)}</article>')
    note = ("The record holds no reasoning text for this model." if not any(
        (s.get("usage") or {}).get("reasoning") for s in r["steps"]) else "")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    head = (f'<section class="run" id="{slug}"><h2>{e(model)} wrote its specification and its answer in '
            f'{count(r["n_steps"], "step")}</h2>'
            f'<p class="lede">{e(model)} under scm‑configure on the DGP 1 panel at σ = 1. '
            f'The fitter ran {WORDS.get(len(r["fit_runs"]), len(r["fit_runs"]))} '
            f'time{"" if len(r["fit_runs"]) == 1 else "s"}. {e(note)}</p>')
    final = box("Answer file at the end of the run", answer_text(r), "answer")
    return head + "".join(steps) + final + "</section>", (model, r, att_cf)


def summary_table(rows):
    body = []
    for model, r, att_cf in rows:
        body.append(
            f"<tr><td>{e(model)}</td><td class=num>{r['n_steps']}</td><td class=num>{len(r['fit_runs'])}</td>"
            f"<td class=num>{signed(r['answer']['att'])}</td><td class=num>{signed(att_cf)}</td>"
            f"<td class=num>{signed(att_cf - TRUE_ATT, plus=True)}</td></tr>")
    return ('<div class="tablewrap"><table><thead><tr><th>solver model</th><th class=num>steps</th>'
            '<th class=num>fitter runs</th><th class=num>ATT written in the answer</th>'
            '<th class=num>ATT of its counterfactual</th><th class=num>ATT error</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>'
            '<p class="caption">DGP 1, σ = 1, scm‑configure, one run per model. The ATT of a counterfactual is the mean '
            'over periods 41 to 60 of the treated unit’s outcome minus the counterfactual, and its error is that '
            'ATT minus the true effect of −3.</p>')


def generator_section(name, folder, script):
    skill = (HERE / folder / "SKILL.md").read_text()
    m = re.search(r"^name:\s*(\S+)", skill, re.M)
    skill_name = m.group(1) if m else Path(folder).parent.name
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (f'<section class="gen" id="{slug}"><h2>{e(name)} wrote the skill and the R script the profile runs '
            f'called</h2><p class="lede">The SKILL.md sections Purpose and Procedure, verbatim, and the full R script.</p>'
            + box("SKILL.md · Purpose", section(skill, "Purpose"), "reply")
            + box("SKILL.md · Procedure", section(skill, "Procedure"), "reply")
            + box(f"R script · {Path(script).name}", (HERE / folder / script).read_text(), "code tall")
            + "</section>")


CSS = """
/* Layout: one reading column, transcripts in bounded scroll boxes, the deck's palette and faces, light only */
:root { --bg:#F7F6F2; --surface:#FFFFFF; --tint:#EEF0EC; --ink:#17212B; --muted:#4A5560; --faint:#5B6670;
  --rule:#D3D8D2; --accent:#0E6B73; --warm:#A65300;
  --display:'Source Serif 4', Georgia, serif; --body:'IBM Plex Sans', Arial, sans-serif;
  --mono:'IBM Plex Mono', 'Courier New', monospace; color-scheme: light; }
body { background:var(--bg); color:var(--ink); font-family:var(--body); font-size:16px; line-height:1.55; }
.page { max-width:1040px; margin:0 auto; padding-block:48px 96px;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px));
  display:flex; flex-direction:column; gap:56px; }
.kicker { font-size:13px; font-weight:600; letter-spacing:2.5px; text-transform:uppercase; color:var(--accent); margin:0; }
h1 { font-family:var(--display); font-weight:600; font-size:clamp(30px, 5vw, 44px); line-height:1.12; margin:8px 0 16px;
  text-wrap:balance; }
h2 { font-family:var(--display); font-weight:600; font-size:clamp(22px, 3.4vw, 28px); line-height:1.2; margin:0;
  text-wrap:balance; }
p { margin:0; max-width:68ch; }
.lede { color:var(--muted); }
nav { display:flex; flex-wrap:wrap; gap:8px 20px; font-size:15px; }
nav a { color:var(--accent); text-decoration:none; border-bottom:1px solid var(--rule); }
nav a:hover, nav a:focus-visible { border-bottom-color:var(--accent); outline:none; }
header.top { display:flex; flex-direction:column; gap:14px; }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:15px; background:var(--surface); }
th, td { padding:8px 12px; border-bottom:1px solid var(--rule); text-align:left; }
th { background:var(--tint); font-weight:600; }
.num { text-align:right; }
.caption { color:var(--faint); font-size:14px; margin-top:8px; }
section { display:flex; flex-direction:column; gap:18px; min-width:0; }
.step { display:flex; flex-direction:column; gap:10px; padding:18px 20px; background:var(--surface);
  border:1px solid var(--rule); border-radius:8px; min-width:0; }
.step header { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; }
.stepno { font-weight:600; color:var(--accent); }
.meta { color:var(--faint); font-size:14px; font-variant-numeric:tabular-nums; }
.box { margin:0; display:flex; flex-direction:column; gap:4px; min-width:0; }
.box figcaption { font-size:12px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:var(--faint); }
.box pre { margin:0; font-family:var(--mono); font-size:13px; line-height:1.5; white-space:pre-wrap;
  overflow-wrap:anywhere; max-height:340px; overflow:auto; padding:12px 14px; border-radius:6px;
  border:1px solid var(--rule); background:var(--surface); color:var(--ink); }
.reason pre { background:var(--tint); color:var(--muted); font-style:italic; }
.output pre { background:#FBFBF9; }
.answer pre { border-color:var(--accent); }
.tall pre { max-height:560px; white-space:pre; overflow-wrap:normal; }
.empty { color:var(--faint); }
"""


def main():
    runs, rows = [], []
    for model, path in RUNS:
        sec, row = run_section(model, run_record(path))
        runs.append(sec)
        rows.append(row)
    gens = [generator_section(*g) for g in GENERATORS]
    nav = "".join(f'<a href="#{re.sub(r"[^a-z0-9]+", "-", m.lower()).strip("-")}">{e(m)}</a>'
                  for m in [r[0] for r in RUNS] + [g[0] for g in GENERATORS])
    page = f"""<title>SCM Example Outputs</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:wght@600&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="top">
<p class="kicker">Synthetic control skills · example outputs</p>
<h1>What each model wrote, step by step</h1>
<p class="lede">Each solver model ran scm‑configure on the DGP 1 panel at σ = 1. Every step below is shown as the record keeps it, with the reasoning text where the record has it, the reply with its bash command, and the output returned to the model. Each generator model wrote one profile skill. Its SKILL.md sections Purpose and Procedure and its full R script follow the runs.</p>
<nav>{nav}</nav>
</header>
<section><h2>The three runs reached different counterfactuals</h2>{summary_table(rows)}</section>
{"".join(runs)}
{"".join(gens)}
</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page):,} characters)")
    for model, r, att_cf in rows:
        print(f"  {model}: steps {r['n_steps']}, fitter runs {len(r['fit_runs'])}, att written {r['answer']['att']:.4f}, "
              f"att of counterfactual {att_cf:.4f}, error {att_cf - TRUE_ATT:+.4f}")


if __name__ == "__main__":
    main()
