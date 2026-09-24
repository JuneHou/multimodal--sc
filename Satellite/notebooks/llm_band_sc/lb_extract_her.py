"""lb_extract_her — copy the collaborator's SHC functions out of her notebook, byte for byte.

Source: data/shc/10_shc_monthly_multiple_outcomes.ipynb (lb_lib.NB26). Each requested definition is
located with `ast` and sliced by its exact line span, so the emitted text is her text: nothing is
retyped, reformatted, dedented or renamed. Comments that sit BETWEEN statements are not carried
(they are outside every statement's span); comments inside a function body are.

    python lb_extract_her.py            dry run: print the manifest and any missing name, write nothing
    python lb_extract_her.py --write    (re)generate lb_her_nb10.py
    python lb_extract_her.py --check    re-extract in memory and byte-compare with the committed file (gate G1)

lb_her_nb10.py is generated and never hand-edited. Its header records the notebook's sha256; its foot
carries a manifest (name, cell, line span, sha256 of the slice) so every block can be traced to the
notebook by a diff.
"""
import ast
import hashlib
import json
import sys

import lb_lib as lb

NB = lb.NB26
OUT = lb.LB / "lb_her_nb10.py"

# Cell-4 constants that the SHC functions read. Her Dropbox paths (BASE_DIR, MONTHLY_DIR, OUTPUT_DIR,
# M_SENSITIVITY_DIR) and the print statements are not taken.
CONSTANTS = ["PRE_PERIODS", "HURRICANE_PERIOD", "POST_PERIODS", "ALL_PERIODS",
             "M", "N_POST", "TREATED_PRE", "TREATED_POST",
             "M_COMPARISON_VALUES", "M_SENSITIVITY_VALUES", "M_SELECTION_NU_STD",
             "VALIDATION_PERIOD", "VALIDATION_TRAIN_PERIODS", "VALIDATION_TREATED_PRE",
             "BANDS", "USE_MONTH_SIN_COS", "BETA_TREND_DEGREE", "KERNEL_BANDWIDTH",
             "ZETA", "SD_TOL", "OPT_TOL", "OPT_MAXITER"]

# (cell index, kind, names). "def" = FunctionDef by name; "assign" = top-level Assign to that name;
# "cell7" = the derived-column statements of cell 7, carried as one string (see below).
BLOCKS = [
    (4, "assign", CONSTANTS),
    (7, "cell7", None),
    (11, "def", ["make_historical_blocks_from_periods"]),
    (11, "assign", ["MAIN_BLOCKS", "VALIDATION_BLOCKS"]),
    (13, "def", ["gaussian_kernel", "local_linear_predict", "prepare_band_series",
                 "_trend_basis", "estimate_seasonal_beta_and_latent"]),
    (15, "def", ["estimate_sensor_latents"]),
    (17, "def", ["build_band_problem", "build_joint_problem"]),
    (19, "def", ["solve_convex_weights", "stepwise_donor_selection"]),
    (21, "def", ["evaluate_pre_fit"]),
    (25, "def", ["get_observed_value", "heldout_validation_for_sensor"]),
]


def notebook():
    raw = NB.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def cell_source(nb, i):
    c = nb["cells"][i]
    assert c["cell_type"] == "code", f"cell {i} is not a code cell"
    return "".join(c["source"])


def _target_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def slices(src, kind, names):
    """[(name, verbatim text, (lineno, end_lineno))] in notebook order; missing names reported by caller."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = []
    if kind == "cell7":
        # Her derived-column statements: every top-level Assign after the `if feature_panel.empty:` guard
        # and before the first print(). They are statements over a bound name `feature_panel`, so they
        # are carried as text and applied by lb_her_panel in a namespace where that name is bound.
        guard = next(n for n in tree.body if isinstance(n, ast.If) and "empty" in ast.dump(n.test))
        first_print = next(n.lineno for n in tree.body
                           if isinstance(n, ast.Expr) and n.lineno > guard.end_lineno
                           and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "print")
        chosen = [n for n in tree.body if isinstance(n, ast.Assign) and guard.end_lineno < n.lineno < first_print]
        text = "".join("".join(lines[n.lineno - 1:n.end_lineno]) for n in chosen)
        return [("CELL7_DERIVED", text, (chosen[0].lineno, chosen[-1].end_lineno))]
    wanted = set(names)
    for n in tree.body:
        name = n.name if (kind == "def" and isinstance(n, ast.FunctionDef)) else (_target_name(n) if kind == "assign" else None)
        if name in wanted:
            out.append((name, "".join(lines[n.lineno - 1:n.end_lineno]), (n.lineno, n.end_lineno)))
    return out


def extract():
    nb, sha = notebook()
    blocks, missing = [], []
    for cell, kind, names in BLOCKS:
        got = slices(cell_source(nb, cell), kind, names)
        found = {g[0] for g in got}
        if names:
            missing += [f"cell {cell}: {n}" for n in names if n not in found]
        for name, text, span in got:
            blocks.append({"name": name, "cell": cell, "kind": kind, "lines": list(span), "text": text,
                           "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return sha, blocks, missing


def emit():
    sha, blocks, missing = extract()
    if missing:
        raise SystemExit("cannot emit; missing from the notebook: " + ", ".join(missing))
    head = ('"""GENERATED by lb_extract_her.py from data/shc/10_shc_monthly_multiple_outcomes.ipynb - DO NOT EDIT.\n'
            f'notebook sha256: {sha}\n'
            'Each block is a byte copy of the named statement(s) of the named notebook cell, sliced by ast line span;\n'
            'comments between statements are not carried. Regenerate: python lb_extract_her.py --write.\n'
            'Verify: python lb_extract_her.py --check (gate G1). Manifest at the foot.\n"""\n'
            "import numpy as np\nimport pandas as pd\nfrom scipy.optimize import minimize\n")
    parts = [head]
    for b in blocks:
        parts.append(f"\n# ---- cell {b['cell']}: {b['name']} (notebook lines {b['lines'][0]}-{b['lines'][1]}) ----\n")
        if b["kind"] == "cell7":
            assert "'''" not in b["text"], "cell 7 text contains a triple quote"
            parts.append("CELL7_DERIVED_SRC = r'''" + b["text"] + "'''\n")
        else:
            parts.append(b["text"] if b["text"].endswith("\n") else b["text"] + "\n")
    manifest = [{k: b[k] for k in ("name", "cell", "kind", "lines", "sha256")} for b in blocks]
    parts.append("\n# ---- manifest ----\nMANIFEST = " + json.dumps(manifest, indent=1) + "\n"
                 f"NOTEBOOK_SHA256 = {sha!r}\n")
    return "".join(parts)


def check():
    text = emit()
    committed = OUT.read_text()
    if text != committed:
        raise SystemExit(f"G1 FAIL: {OUT.name} differs from a fresh extraction ({len(committed)} vs {len(text)} chars)")
    print(f"G1 ok: {OUT.name} is byte-identical to a fresh extraction; notebook sha256 {NB.name}: {hashlib.sha256(NB.read_bytes()).hexdigest()[:16]}...")


def main(argv):
    if "--check" in argv:
        return check()
    sha, blocks, missing = extract()
    for b in blocks:
        print(f"cell {b['cell']:2d}  {b['kind']:6s}  {b['name']:40s} lines {b['lines'][0]:4d}-{b['lines'][1]:4d}  {len(b['text']):6d} chars")
    print(f"{len(blocks)} blocks; notebook sha256 {sha[:16]}...")
    if missing:
        print("MISSING:", "; ".join(missing))
    if "--write" in argv:
        OUT.write_text(emit())
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
