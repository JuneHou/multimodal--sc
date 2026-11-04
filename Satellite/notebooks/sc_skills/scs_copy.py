"""scs_copy — the unchanged copy of a method's reference code (configure and design, step 1).

The copy is the reference code character for character: nothing is retyped, reformatted, dedented or renamed. Three
ways, set by the package's CopyRule (scs_method.py):

    functions  named definitions and assignments of a notebook, each sliced by its ast line span; a block given as a
               dict carries the statements its `select` picks as one string constant
    cells      whole notebook cells, each carried as one string CELL_SRC[i], leaving out the statements that name a
               name in `skip`
    package    an installed package (an R package for Synth): no text is copied, the record holds the package name,
               its version and the sha256 of every installed file, so a later change to the installed copy is caught

The copy file ends with a manifest (name or cell, line spans, sha256) and the notebook's sha256.

    python scs_copy.py <method>            print what would be copied
    python scs_copy.py <method> --write    write the copy to the package's CopyRule.out
    python scs_copy.py <method> --check    copy again in memory and byte-compare with CopyRule.out
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import scs_method


def notebook(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def cell_source(nb, i):
    c = nb["cells"][i]
    assert c["cell_type"] == "code", f"cell {i} is not a code cell"
    return "".join(c["source"])


def _target_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
           ({node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()) | \
           {a.asname or a.name for a in getattr(node, "names", []) if isinstance(a, ast.alias)}


# ----------------------------------------------------------------------------- functions
def function_blocks(rule):
    nb, sha = notebook(rule.source)
    blocks, missing = [], []
    for b in rule.blocks:
        if isinstance(b, dict):
            src = cell_source(nb, b["cell"])
            lines = src.splitlines(keepends=True)
            chosen = b["select"](ast.parse(src), lines)
            text = "".join("".join(lines[n.lineno - 1:n.end_lineno]) for n in chosen)
            blocks.append({"name": b["name"], "cell": b["cell"], "kind": b["kind"],
                           "lines": [chosen[0].lineno, chosen[-1].end_lineno], "text": text,
                           "sha256": hashlib.sha256(text.encode()).hexdigest(), "const": b["const"]})
            continue
        cell, kind, names = b
        src = cell_source(nb, cell)
        lines = src.splitlines(keepends=True)
        found = set()
        for n in ast.parse(src).body:
            name = n.name if (kind == "def" and isinstance(n, ast.FunctionDef)) else (_target_name(n) if kind == "assign" else None)
            if name in names:
                text = "".join(lines[n.lineno - 1:n.end_lineno])
                found.add(name)
                blocks.append({"name": name, "cell": cell, "kind": kind, "lines": [n.lineno, n.end_lineno],
                               "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
        missing += [f"cell {cell}: {n}" for n in names if n not in found]
    return sha, blocks, missing


def emit_functions(rule):
    sha, blocks, missing = function_blocks(rule)
    if missing:
        raise SystemExit("cannot copy; missing from the notebook: " + ", ".join(missing))
    parts = [rule.header(sha) + rule.imports]
    for b in blocks:
        parts.append(f"\n# ---- cell {b['cell']}: {b['name']} (notebook lines {b['lines'][0]}-{b['lines'][1]}) ----\n")
        if "const" in b:
            assert "'''" not in b["text"], f"{b['name']} contains a triple quote"
            parts.append(f"{b['const']} = r'''" + b["text"] + "'''\n")
        else:
            parts.append(b["text"] if b["text"].endswith("\n") else b["text"] + "\n")
    manifest = [{k: b[k] for k in ("name", "cell", "kind", "lines", "sha256")} for b in blocks]
    parts.append("\n# ---- manifest ----\nMANIFEST = " + json.dumps(manifest, indent=1) + "\n"
                 f"NOTEBOOK_SHA256 = {sha!r}\n")
    return "".join(parts)


# ----------------------------------------------------------------------------- cells
def cell_parts(rule):
    nb, sha = notebook(rule.source)
    cells = []
    for i in rule.cells:
        src = cell_source(nb, i)
        lines = src.splitlines(keepends=True)
        skip = rule.skip.get(i, set())
        stmts = [(n.lineno, n.end_lineno, "".join(lines[n.lineno - 1:n.end_lineno]), not (_names(n) & skip))
                 for n in ast.parse(src).body]
        text = "\n".join(s[2].rstrip("\n") for s in stmts if s[3]) + "\n"
        cells.append({"cell": i, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "kept": [[s[0], s[1]] for s in stmts if s[3]],
                      "skipped": [[s[0], s[1]] for s in stmts if not s[3]]})
    return sha, cells


def emit_cells(rule):
    sha, cells = cell_parts(rule)
    parts = [rule.header(sha), "CELL_SRC = {}\n"]
    for c in cells:
        assert "'''" not in c["text"], f"cell {c['cell']} contains a triple single quote"
        parts.append(f"\n# ---- cell {c['cell']} ----\nCELL_SRC[{c['cell']}] = r'''" + c["text"] + "'''\n")
    manifest = [{k: c[k] for k in ("cell", "kept", "skipped", "sha256")} for c in cells]
    parts.append("\n# ---- manifest ----\nMANIFEST = " + json.dumps(manifest, indent=1) + "\n"
                 f"NOTEBOOK_SHA256 = {sha!r}\n")
    return "".join(parts)


# ----------------------------------------------------------------------------- package
def emit_package(rule):
    """An installed package's record: name, version (from its DESCRIPTION file) and the sha256 of every file."""
    root = Path(rule.source)
    desc = (root / "DESCRIPTION").read_text() if (root / "DESCRIPTION").exists() else ""
    version = next((l.split(":", 1)[1].strip() for l in desc.splitlines() if l.startswith("Version:")), "")
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file()}
    return json.dumps({"package": rule.package, "version": version, "installed_at": str(root), "files": files},
                      indent=1) + "\n"


def emit(pkg):
    rule = pkg.reference_code.copy
    return {"functions": emit_functions, "cells": emit_cells, "package": emit_package}[rule.mode](rule)


def check(pkg):
    rule = pkg.reference_code.copy
    text, committed = emit(pkg), rule.out.read_text()
    same = text == committed
    print(f"{pkg.name}: copy of the reference code ({rule.mode}) "
          + (f"is byte-identical to {rule.out.name}" if same else
             f"DIFFERS from {rule.out.name} ({len(committed)} vs {len(text)} characters)"))
    return same


if __name__ == "__main__":
    method, argv = sys.argv[1], sys.argv[2:]
    pkg = scs_method.load(method)
    if "--check" in argv:
        raise SystemExit(0 if check(pkg) else 1)
    text = emit(pkg)
    if "--write" in argv:
        pkg.reference_code.copy.out.write_text(text)
        print(f"wrote {pkg.reference_code.copy.out}")
    else:
        print(f"{pkg.name}: {len(text)} characters would be written to {pkg.reference_code.copy.out}")
