"""lb_extract_her14 — copy the collaborator's daily SHC (notebook 14) out of her notebook, byte for byte.

Source: data/scripts/scripts/validation/shc/14_shc_daily_sentinel1_sentinel2.ipynb (plan
~/.claude/plans/simple-skills-daily.md, step 2). Every top-level statement of the cells listed in CELLS is located
with `ast` and sliced by its exact line span, so the emitted text is her text: nothing is retyped, reformatted or
renamed. Her cells build module-level state (series_store, stack_case_store, final_fit_store, daily_effects) that
her functions read as globals, so each cell is carried as one source string and lb_her14_run.py executes the strings
in order in one namespace, as her notebook does.

Statements not carried, each named in SKIP: in cell 1 the matplotlib import (her figures, cell 15, are not run);
in cell 2 her Dropbox paths, the output folders and the checks that read them. lb_her14_run.py binds PANEL_FILES to
our copies of her panels in their place, as lb_her_panel.py replaces the I/O half of notebook 10's cells 6 and 7.

Her evaluation cells go to a second file, lb_her_eval14.py (plan step 10, Decisions "her cells 11, 14 and 15 in
code"): the cell 1 statement that imports matplotlib, then cells 11 (validation), 14 (result tables) and 15 (figures)
whole. lb_her_nb14.py is bundled in both daily skills, which have run, so it is never changed by this.

    python lb_extract_her14.py            dry run: print what would be carried and skipped, write nothing
    python lb_extract_her14.py --write    (re)generate lb_her_nb14.py and lb_her_eval14.py
    python lb_extract_her14.py --check    re-extract in memory and byte-compare with both committed files (gate DK1)
"""
import ast
import hashlib
import json
import sys

import lb_lib as lb

NB = lb.ROOT / "Satellite" / "data" / "scripts" / "scripts" / "validation" / "shc" / "14_shc_daily_sentinel1_sentinel2.ipynb"
OUT = lb.LB / "lb_her_nb14.py"

# Her cells in run order. Cell 9 (the nu sensitivity table), cell 11 (validation report), cell 14 (result tables),
# cell 15 (figures) and cell 16 (saving) read the state but write nothing cells 10, 12 and 13 use.
CELLS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13]

# Top-level names whose statements are left out of a cell: a statement is skipped when it assigns, defines or
# reads any of these names.
SKIP = {
    1: {"plt"},
    2: {"CALS_DIR", "DAILY_DIR", "PANEL_FILE_CANDIDATES", "first_existing", "PANEL_FILES", "OUTPUT_DIR",
        "FIGURE_DIR", "required_file"},
}

# lb_her_eval14.py: her evaluation cells, run by lb_skill_score_daily.py after the cells above. From cell 1 only the
# statements naming a name in EVAL_ONLY are carried (the matplotlib import SKIP leaves out above); the other cells whole.
# FIGURE_DIR, which her cell 15 writes into, is bound by the scorer, as PANEL_FILES is bound by lb_her14_run.py.
EVAL_OUT = lb.LB / "lb_her_eval14.py"
EVAL_CELLS = [1, 11, 14, 15]
EVAL_ONLY = {1: {"plt"}}


def notebook():
    raw = NB.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def cell_source(nb, i):
    c = nb["cells"][i]
    assert c["cell_type"] == "code", f"cell {i} is not a code cell"
    return "".join(c["source"])


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
           ({node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()) | \
           {a.asname or a.name for a in getattr(node, "names", []) if isinstance(a, ast.alias)}


def cell_blocks(src, skip):
    """[(first line, last line, text, kept)] for every top-level statement of one cell, in order."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = []
    for n in tree.body:
        text = "".join(lines[n.lineno - 1:n.end_lineno])
        out.append((n.lineno, n.end_lineno, text, not (_names(n) & skip)))
    return out


def extract(cell_list=CELLS, only=None):
    """Kept statements per cell: all but those naming a SKIP name, or with `only`, just those naming an `only` name."""
    nb, sha = notebook()
    cells = []
    for i in cell_list:
        if only and i in only:
            blocks = [(a, b, t, not k) for a, b, t, k in cell_blocks(cell_source(nb, i), only[i])]
        else:
            blocks = cell_blocks(cell_source(nb, i), set() if only else SKIP.get(i, set()))
        text = "\n".join(b[2].rstrip("\n") for b in blocks if b[3]) + "\n"
        cells.append({"cell": i, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "kept": [[b[0], b[1]] for b in blocks if b[3]],
                      "skipped": [[b[0], b[1]] for b in blocks if not b[3]]})
    return sha, cells


def emit():
    return _emit(*extract(), 'the strings in order in one namespace. Regenerate: python lb_extract_her14.py --write.\n')


def emit_eval():
    return _emit(*extract(EVAL_CELLS, EVAL_ONLY),
                 'the strings in order in one namespace. Her evaluation cells: from cell 1 only the matplotlib import,\n'
                 'then cells 11, 14 and 15 whole; lb_skill_score_daily.py runs them after the cells of lb_her_nb14.py.\n'
                 'Regenerate: python lb_extract_her14.py --write.\n')


def _emit(sha, cells, run_line):
    parts = ['"""GENERATED by lb_extract_her14.py from '
             'data/scripts/scripts/validation/shc/14_shc_daily_sentinel1_sentinel2.ipynb - DO NOT EDIT.\n'
             f'notebook sha256: {sha}\n'
             'CELL_SRC[i] holds the top-level statements of her cell i, each a byte copy sliced by ast line span, joined in\n'
             'notebook order; the manifest at the foot gives the kept and skipped line spans. lb_her14_run.py executes\n'
             + run_line +
             'Verify: python lb_extract_her14.py --check (gate DK1).\n"""\n',
             "CELL_SRC = {}\n"]
    for c in cells:
        assert "'''" not in c["text"], f"cell {c['cell']} contains a triple single quote"
        parts.append(f"\n# ---- cell {c['cell']} ----\nCELL_SRC[{c['cell']}] = r'''" + c["text"] + "'''\n")
    manifest = [{k: c[k] for k in ("cell", "kept", "skipped", "sha256")} for c in cells]
    parts.append("\n# ---- manifest ----\nMANIFEST = " + json.dumps(manifest, indent=1) + "\n"
                 f"NOTEBOOK_SHA256 = {sha!r}\n")
    return "".join(parts)


def check():
    for out, text in ((OUT, emit()), (EVAL_OUT, emit_eval())):
        committed = out.read_text()
        if text != committed:
            raise SystemExit(f"DK1 FAIL: {out.name} differs from a fresh extraction ({len(committed)} vs {len(text)} chars)")
        print(f"DK1 ok: {out.name} is byte-identical to a fresh extraction; notebook sha256 "
              f"{hashlib.sha256(NB.read_bytes()).hexdigest()[:16]}...")


def main(argv):
    if "--check" in argv:
        check()
    elif "--write" in argv:
        OUT.write_text(emit())
        EVAL_OUT.write_text(emit_eval())
        print(f"wrote {OUT.name}, {EVAL_OUT.name}")
    else:
        for name, (sha, cells) in ((OUT.name, extract()), (EVAL_OUT.name, extract(EVAL_CELLS, EVAL_ONLY))):
            print(f"{name}: notebook sha256 {sha[:16]}...")
            for c in cells:
                print(f"  cell {c['cell']:2d}: {len(c['kept'])} statements kept, skipped line spans {c['skipped']}")


if __name__ == "__main__":
    main(sys.argv[1:])
