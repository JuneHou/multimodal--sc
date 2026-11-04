"""Collect every text a model reads into review/model_texts.md, from the built skills (plan step 7: shown to Jun
before any model call).

    python review/make_review.py        (from notebooks/scm_simulation/, after scs_build.py)
"""
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "sc_skills"))
import scs_loop     # noqa: E402
import scs_method   # noqa: E402
import scs_profile  # noqa: E402

mod = scs_method.load_module("../../scm_simulation")
pkg, use = mod.PACKAGE, mod.USES["simulation"]
sk = HERE / "skills"
case = use.cases()[0]
parts = ["# SCM skills: every text a model reads\n\n",
         "Built from `notebooks/scm_simulation/method.py` into `notebooks/scm_simulation/skills/`. The solver models "
         "see sections 1 to 6. The two SkillFoundry generators see section 7, with the paper in `paper/` and Synth "
         "1.1-10's CRAN source in `code/Synth/`.\n"]


def sec(title, text, lang=""):
    parts.append(f"\n## {title}\n\n```{lang}\n{text.rstrip()}\n```\n")


sec("1. System message (the pipeline's, unchanged)", scs_loop.SYSTEM)
for tag, fmt in (("2a", "configure"), ("2b", "design")):
    sec(f"{tag}. First message, case dgp1, skill scm-{fmt} (dgp2 differs only in the file name)",
        use.task_text(sk / f"scm-{fmt}", case))
sec("2c. First message for a profile skill, case dgp1 (row G16 of the profile experiment: no skill line)",
    use.task_text(HERE / "profile" / "skills" / "gpt-5.4" / "code_provided", case))
for tag, fmt in (("3a", "configure"), ("3b", "design")):
    sec(f"{tag}. scm-{fmt}/SKILL.md", (sk / f"scm-{fmt}/SKILL.md").read_text(), "markdown")
sec("4. references/method.md (both skills)", (sk / "scm-configure/references/method.md").read_text(), "markdown")
for tag, fmt in (("5a", "configure"), ("5b", "design")):
    sec(f"{tag}. scm-{fmt}/references/settings.md", (sk / f"scm-{fmt}/references/settings.md").read_text(), "markdown")
sec("6a. scripts/show_data.py (both skills)", (sk / "scm-configure/scripts/show_data.py").read_text(), "python")
sec("6b. scm-configure/scripts/fit.R", (sk / "scm-configure/scripts/fit.R").read_text(), "r")
a = (sk / "scm-configure/scripts/fit.R").read_text().splitlines()
b = (sk / "scm-design/scripts/fit.R").read_text().splitlines()
sec("6c. scm-design/scripts/fit.R, its difference from 6b",
    "\n".join(difflib.unified_diff(a, b, "configure", "design", lineterm="", n=1)), "diff")
for m in ("scm_fields.R", "synth_fit.R", "scm_native.R"):
    sec(f"6d. scripts/{m} (both skills, copied unchanged from notebooks/scm_simulation/)",
        (sk / f"scm-configure/scripts/{m}").read_text(), "r")
sec("7. SkillFoundry request (both generators)", scs_profile.request(pkg, use))
out = HERE / "review" / "model_texts.md"
out.write_text("".join(parts))
print(f"wrote {out} ({sum(len(p) for p in parts)} characters)")
