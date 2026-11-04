"""scs_method — the method package: every input one synthetic control method gives the three skill-making pipelines
(configure, design, SkillFoundry profile). A pipeline reads a package and nothing else names a method, sensor, band
or period. Plan ~/.claude/plans/quirky-growing-wombat.md, "Method package".

    paper           the paper, local PDF
    reference_code  the collaborator's code, else the paper's package: its language and how it is copied unchanged
    data            the input data as shipped
    spec            settings (the reference code's own arguments) and choices (its hard-coded choices), each field
                    with kind, allowed values and meaning, the reference code's values, and the rules that tie them
    engine          how the copied code runs, and the native engine for options the reference code lacks
    texts           every text the model reads: SKILL.md sections, references/method.md and the scripts
    skill_modules   files copied unchanged into each configure and design skill's scripts/ folder

What the model is given from the data, and how its answers are evaluated, are decided when a pipeline is used.

    python scs_method.py <method>      validate methods/<method>/method.py and print its fields
"""
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
METHODS = HERE / "methods"
KINDS = ("int", "float", "int_or_null", "positive_list", "enum", "bool",
         "name_list", "predictor_list", "positive_list_or_null")
# name_list              allowed = tuple of names; a list of distinct names from it (may be empty)
# predictor_list         allowed = {"variables": tuple, "ops": tuple}; a list of {"variable", "periods", "op"} objects,
#                        periods a non-empty list of whole numbers >= 1 (may be empty)
# positive_list_or_null  allowed = predicate on a list, as positive_list; or null
LANGUAGES = ("python", "R")
COPY_MODES = ("functions", "cells", "package")


@dataclass
class Field:
    """One field of the spec: kind (one of KINDS), allowed (predicate for numbers, options for enum, None for bool),
    meaning (the line settings.md shows; for a setting, "<range>: <meaning>")."""
    kind: str
    allowed: object
    meaning: str


@dataclass
class Check:
    """A rule over the spec. fn(given, out, ctx, errors) appends one message per violation. `configure` marks a rule
    that involves settings only, so it also applies when the choices are fixed."""
    fn: Callable
    configure: bool


@dataclass
class Spec:
    settings: dict                     # name -> Field
    choices: dict                      # name -> Field
    optional: dict                     # name -> (Field, settings.md type label, shape text)
    rules: list                        # rule texts, shown to the model word for word
    configure_rules: list              # indices of `rules` that involve settings only
    checks: list                       # Check, in the order their messages are reported
    reference: dict                    # the reference code's value of every setting and choice
    configure_label: str               # how a refusal names the configure skill


@dataclass
class CopyRule:
    """How the reference code becomes an unchanged copy.
    functions: blocks of (cell, kind, names) sliced by ast line span from a notebook; kind "def" or "assign", or a
               dict {"cell", "kind", "name", "const", "select"} for statements carried as one string
    cells:     whole notebook cells as strings; `skip` {cell: names} leaves out statements naming those names
    package:   an installed package, recorded by name, version and a hash of its installed files"""
    mode: str
    source: Path                       # notebook, or the package's installed folder
    out: Optional[Path] = None         # the file the copy is written to (functions, cells)
    blocks: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    skip: dict = field(default_factory=dict)
    header: Optional[Callable] = None  # header(notebook sha256) -> text at the top of the copy
    imports: str = ""                  # import lines after the header (functions mode)
    package: str = ""                  # package name (package mode)


@dataclass
class ReferenceCode:
    """parity() yields (case, quantity, ours, printed) for collaborator's code: the copied code run on the data at the
    reference values, written as the collaborator's code prints it, against what the collaborator's notebook printed
    (compared with scs_check.same). trial holds the R trial of a package written in R:
    {"script": the R script, "cases": case folders of the real data, "r_env": the R installation}."""
    origin: str                        # "collaborator" or "paper package"
    language: str                      # one of LANGUAGES
    copy: CopyRule
    parity: Optional[Callable] = None
    trial: Optional[dict] = None


@dataclass
class Engine:
    """reference(ctx, spec) runs the copied code at the spec's settings with every choice at the reference value.
    native(ctx, spec, forced) runs the native engine; forced=True runs it even on the reference choices."""
    reference: Callable
    native: Callable


@dataclass
class Texts:
    method_name: str                   # as in "Configure and run the <method_name>"
    skill_prefix: str                  # skills are <prefix>-configure and <prefix>-design
    method_md: str
    configure_description: str
    design_description: str
    configure_intro: str
    design_intro: str
    fixed_choices: str                 # after "The fitter makes these choices for you: "
    settings_list: str
    settings_rules: str
    configure_workflow: str
    design_overview: str
    choices_list: str
    design_rules: str
    design_workflow: str
    scripts: dict                      # file name -> text, identical in both skills (show_data, the shim)
    fit_script: str                    # fit script, a %-template with file, fixed_doc, fixed_code, validate
    fit_script_name: str               # fit.py or fit.R
    fixed_comment: str                 # comment line above FIXED in the configure fit script
    validate_configure: str            # validation lines of the configure fit script
    validate_design: str               # validation lines of the design fit script
    authors: str = ""                  # profile request: "the <method_name> method of <authors>"
    paper_title: str = ""              # profile request: "The paper, <paper_title>, is at `paper/<file>`."
    profile_code: str = ""             # profile request: the sentence naming the code in code/ ("" when none is given)
    profile_focus: str = ""            # SkillFoundry's focus term for the request
    profile_code_paths: list = field(default_factory=list)   # files and folders placed in code/ of the working copy


@dataclass
class MethodPackage:
    name: str
    paper: Path
    reference_code: ReferenceCode
    data: list                         # paths of the input data as shipped
    spec: Spec
    engine: Engine
    texts: Texts
    skill_modules: list                # [(name in scripts/, source path)], copied unchanged


def validate(pkg):
    """Every field present and well formed. Returns a list of problems (empty when the package is valid)."""
    bad = []
    if not pkg.paper.exists():
        bad.append(f"paper not found: {pkg.paper}")
    rc = pkg.reference_code
    if rc.language not in LANGUAGES:
        bad.append(f"reference_code.language {rc.language!r} not in {LANGUAGES}")
    if rc.copy.mode not in COPY_MODES:
        bad.append(f"reference_code.copy.mode {rc.copy.mode!r} not in {COPY_MODES}")
    if not rc.copy.source.exists():
        bad.append(f"reference code source not found: {rc.copy.source}")
    for p in pkg.data:
        if not Path(p).exists():
            bad.append(f"data not found: {p}")
    s = pkg.spec
    for group in ("settings", "choices"):
        for name, f in getattr(s, group).items():
            if f.kind not in KINDS:
                bad.append(f"{group}.{name}: kind {f.kind!r} not in {KINDS}")
            if f.kind in ("enum", "name_list") and not isinstance(f.allowed, (tuple, list)):
                bad.append(f"{group}.{name}: {f.kind} without options")
            if f.kind == "predictor_list" and not (isinstance(f.allowed, dict) and {"variables", "ops"} <= set(f.allowed)):
                bad.append(f"{group}.{name}: predictor_list without variables and ops")
    missing_ref = [k for k in list(s.settings) + list(s.choices) if k not in s.reference]
    if missing_ref:
        bad.append(f"spec.reference lacks {missing_ref}")
    if any(not 0 <= i < len(s.rules) for i in s.configure_rules):
        bad.append("spec.configure_rules points outside spec.rules")
    for name, src in pkg.skill_modules:
        if not Path(src).exists():
            bad.append(f"skill module not found: {src}")
    for k, v in vars(pkg.texts).items():
        if isinstance(v, str) and not v:
            bad.append(f"texts.{k} is empty")
    return bad


def load_module(method):
    """methods/<method>/method.py as a module (PACKAGE, and OLD when the method already had an implementation)."""
    name = f"scs_method_{method}"
    if name in sys.modules:
        return sys.modules[name]
    path = METHODS / method / "method.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load(method):
    """methods/<method>/method.py -> its PACKAGE."""
    return load_module(method).PACKAGE


def number_word(n):
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return words[n] if 0 <= n <= 10 else str(n)


if __name__ == "__main__":
    for method in sys.argv[1:]:
        pkg = load(method)
        problems = validate(pkg)
        print(f"{method}: {len(pkg.spec.settings)} settings, {len(pkg.spec.choices)} choices, "
              f"{len(pkg.skill_modules)} skill modules, reference code {pkg.reference_code.language} "
              f"({pkg.reference_code.copy.mode})")
        print("  valid" if not problems else "  INVALID: " + "; ".join(problems))
