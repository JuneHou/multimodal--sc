"""Collect every text a model reads in scm-four-design into review/model_texts_four_design.md, from the built skill
(plan ~/.claude/plans/use-nature-skills-analyze-wiggly-coral.md, step 6: shown to Jun before any model call).

    python four_design/make_review.py        (from notebooks/scm_simulation/, after scs_build.py)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM.parent / "sc_skills"))
import scs_loop     # noqa: E402
import scs_method   # noqa: E402

mod = scs_method.load_module("../../scm_simulation/four_design")
use = mod.USES["simulation"]
case = use.cases()[0]
sk = HERE / "skills" / "scm-four-design"
old = SIM / "skills" / "scm-design"
parts = ["# scm-four-design: every text a model reads\n\n",
         "Built from `notebooks/scm_simulation/four_design/method.py` into "
         "`notebooks/scm_simulation/four_design/skills/scm-four-design/`. The build also makes `scm-four-configure`, "
         "which is never run, so it is not shown.\n"]


def sec(title, text, lang=""):
    parts.append(f"\n## {title}\n\n```{lang}\n{text.rstrip()}\n```\n")


def same_as_old(rel):
    return (sk / rel).read_text() == (old / rel).read_text()


sec("1. System message (the pipeline's, unchanged)", scs_loop.SYSTEM)
sec("2. First message, case dgp1 (dgp2 differs only in the file name)", use.task_text(sk, case))
sec("3. SKILL.md", (sk / "SKILL.md").read_text(), "markdown")
sec("4. references/method.md" + (", identical to scm-design's" if same_as_old("references/method.md") else ""),
    (sk / "references/method.md").read_text(), "markdown")
sec("5. references/settings.md", (sk / "references/settings.md").read_text(), "markdown")
sec("6a. scripts/show_data.py" + (", identical to scm-design's" if same_as_old("scripts/show_data.py") else ""),
    (sk / "scripts/show_data.py").read_text(), "python")
sec("6b. scripts/fit.R", (sk / "scripts/fit.R").read_text(), "r")
for m in ("four_fields.R", "four_translate.R"):
    sec(f"6c. scripts/{m} (new, copied unchanged from notebooks/scm_simulation/four_design/)",
        (sk / f"scripts/{m}").read_text(), "r")
for m in ("synth_fit.R", "scm_native.R"):
    assert same_as_old(f"scripts/{m}"), m
    sec(f"6d. scripts/{m} (identical to scm-design's, copied unchanged from notebooks/scm_simulation/)",
        (sk / f"scripts/{m}").read_text(), "r")
out = SIM / "review" / "model_texts_four_design.md"
out.write_text("".join(parts))
print(f"wrote {out} ({sum(len(p) for p in parts)} characters)")
