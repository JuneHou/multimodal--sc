"""scs_build — build a method's configure and design skill folders from its package (configure step 4, design step 3).

Each skill folder holds SKILL.md (from templates/<format>_SKILL.md filled with the package's texts), references/method.md,
references/settings.md (generated from the package's settings and choices), the package's scripts, the fit script (the
package's template with this format's file name, fixed choices and validation filled in), and the skill modules copied
unchanged. MANIFEST.json at the root records the sha256 of every file.

configure  the model writes settings.json with the settings only; the choices are fixed at the reference code's values
design     the model writes spec.json with the settings and the choices

    python scs_build.py <method> <out folder>          build both skills
    python scs_build.py <method> <out folder> --check  compare <out folder> with a fresh build, file for file
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import scs_method
from scs_spec import KIND_LABEL

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
FORMATS = ("configure", "design")
SPEC_FILE = {"configure": "settings.json", "design": "spec.json"}
FIXED_WIDTH = 100                     # the FIXED line of the configure fit script wraps before this column


def skill_name(pkg, fmt):
    return f"{pkg.texts.skill_prefix}-{fmt}"


def skill_md(pkg, fmt):
    t = pkg.texts
    template = (TEMPLATES / f"{fmt}_SKILL.md").read_text()
    if fmt == "configure":
        return template.format(prefix=t.skill_prefix, description=t.configure_description, method_name=t.method_name,
                               intro=t.configure_intro, fixed_choices=t.fixed_choices,
                               n_settings=scs_method.number_word(len(pkg.spec.settings)),
                               settings_list=t.settings_list, settings_rules=t.settings_rules,
                               workflow=t.configure_workflow)
    return template.format(prefix=t.skill_prefix, description=t.design_description, method_name=t.method_name,
                           intro=t.design_intro, design_overview=t.design_overview, choices_list=t.choices_list,
                           design_rules=t.design_rules, workflow=t.design_workflow)


def _choice_shape(f):
    if f.kind == "option":                    # departure 2: "option" and each value, with the options that carry it
        out = {"option": "<" + "|".join(f.allowed) + ">"}
        for spec in f.allowed.values():
            for k, vf in spec["values"].items():
                if k not in out:
                    users = [a for a, sa in f.allowed.items() if k in sa["values"]]
                    out[k] = f"{vf.meaning.partition(': ')[0]}, only with {' or '.join(users)}"
        return out
    return "<true|false>" if f.kind == "bool" else "<" + "|".join(f.allowed) + ">"


def _choice_options(f):
    if f.kind == "option":
        return ", ".join(f"`{a}`" + (" with " + " and ".join(f"`{k}`" for k in s["values"]) if s["values"] else "")
                         for a, s in f.allowed.items())
    return ", ".join(f"`{a}`" for a in f.allowed) if f.kind == "enum" else "`true`, `false`"


def _option_values(S):
    """Rows of the values the options carry, for choices of the kind "option" (none for other packages)."""
    rows = []
    for name, f in S.choices.items():
        if f.kind != "option":
            continue
        for a, s in f.allowed.items():
            for k, vf in s["values"].items():
                rng, sep, why = vf.meaning.partition(": ")
                rows.append(f"| `{name}` | `{a}` | `{k}` | {rng} | {why} |")
    if not rows:
        return []
    return ["", "## Values the options carry", "", "| choice | option | value | type and range | meaning |",
            "|---|---|---|---|---|"] + rows


def settings_md(pkg, fmt):
    S = pkg.spec
    rows = ["| field | type and range | meaning |", "|---|---|---|"]
    for name, f in S.settings.items():
        rng, sep, why = f.meaning.partition(": ")
        rows.append(f"| `{name}` | {rng} | {why} |")
    out = [f"# Fields for {skill_name(pkg, fmt)}", "",
           f"`{SPEC_FILE[fmt]}` holds exactly the fields below; there are no defaults, and a file "
           "with a field missing, a field misspelled, or a value outside its range is refused with the field named.", "",
           "## Settings", ""] + rows
    if fmt == "design":
        out += ["", "## Choices", "", "| field | options | meaning |", "|---|---|---|"]
        for name, f in S.choices.items():
            out.append(f"| `{name}` | {_choice_options(f)} | {f.meaning} |")
        for name, (f, label, shape) in S.optional.items():
            out.append(f"| `{name}` | {label} | {f.meaning} |")
        out += _option_values(S)
        out += ["", "## Rules that tie the fields together", ""] + [f"- {r}" for r in S.rules]
        shape = {**{k: KIND_LABEL[f.kind] for k, f in S.settings.items()},
                 **{k: _choice_shape(f) for k, f in S.choices.items()},
                 **{k: shape for k, (f, label, shape) in S.optional.items()}}
    else:
        out += ["", "## Rules", ""] + [f"- {S.rules[i]}" for i in S.configure_rules]
        shape = {k: KIND_LABEL[f.kind] for k, f in S.settings.items()}
    out += ["", "## Shape of the file (every value is yours)", "", "```json", json.dumps(shape, indent=1), "```"]
    return "\n".join(out) + "\n"


def _literal(v):
    return json.dumps(v) if isinstance(v, str) else repr(v)


def fixed_code(pkg):
    """The configure fit script's FIXED: every choice at the reference code's value, wrapped before FIXED_WIDTH."""
    items = [f'"{k}": {_literal(pkg.spec.reference[k])}' for k in pkg.spec.choices]
    lines, cur = [], "FIXED = {"
    for i, it in enumerate(items):
        piece = it + ("}" if i == len(items) - 1 else ",")
        if cur.strip() not in ("FIXED = {", "") and len(cur) + 1 + len(piece) > FIXED_WIDTH:
            lines.append(cur)
            cur = " " * len("FIXED = {") + piece
        else:
            cur = cur + ("" if cur.endswith("{") else " ") + piece
    lines.append(cur)
    return pkg.texts.fixed_comment + "\n" + "\n".join(lines) + "\n"


def fit_script(pkg, fmt):
    t = pkg.texts
    configure = fmt == "configure"
    return t.fit_script % {
        "file": SPEC_FILE[fmt],
        "fixed_doc": (f"\nThe {scs_method.number_word(len(pkg.spec.choices))} choices of the method are fixed here; only "
                      f"the {scs_method.number_word(len(pkg.spec.settings))} settings are read.") if configure else "",
        "fixed_code": fixed_code(pkg) if configure else "",
        "validate": t.validate_configure if configure else t.validate_design}


def files(pkg):
    """{relative path: text} of both skills, without the manifest."""
    out = {}
    for fmt in FORMATS:
        name = skill_name(pkg, fmt)
        out[f"{name}/SKILL.md"] = skill_md(pkg, fmt)
        out[f"{name}/references/method.md"] = pkg.texts.method_md
        out[f"{name}/references/settings.md"] = settings_md(pkg, fmt)
        for fname, text in pkg.texts.scripts.items():
            out[f"{name}/scripts/{fname}"] = text
        out[f"{name}/scripts/{pkg.texts.fit_script_name}"] = fit_script(pkg, fmt)
        for mname, src in pkg.skill_modules:
            out[f"{name}/scripts/{mname}"] = Path(src).read_text()
    return out


def build(pkg, root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    manifest = {}
    for rel, text in files(pkg).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        manifest[rel] = hashlib.sha256(text.encode()).hexdigest()
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def compare(pkg, root, names=None):
    """Compare the skill folders under `root` with a fresh build, file for file by sha256. -> list of differences."""
    root, fresh = Path(root), files(pkg)
    names = names or [skill_name(pkg, f) for f in FORMATS]
    diff = []
    for rel, text in fresh.items():
        p = root / rel
        if not p.exists():
            diff.append(f"missing: {rel}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != hashlib.sha256(text.encode()).hexdigest():
            diff.append(f"differs: {rel}")
    for name in names:
        for p in sorted((root / name).rglob("*")):
            rel = str(p.relative_to(root))
            if p.is_file() and rel not in fresh:
                diff.append(f"extra: {rel}")
    return diff


if __name__ == "__main__":
    method, out = sys.argv[1], sys.argv[2]
    pkg = scs_method.load(method)
    if "--check" in sys.argv:
        d = compare(pkg, out)
        print(f"{method}: {len(files(pkg))} files compared with {out}; {len(d)} differences")
        for x in d:
            print("  ", x)
        raise SystemExit(1 if d else 0)
    m = build(pkg, out)
    print(f"built {len(m)} files under {out}")
