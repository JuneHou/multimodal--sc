"""lb_skill_redo — set aside the records of cells that must be re-run, so lb_skill_run.py resumes exactly those.

Decided 2026-09-24: cells in which the model issued commands in its native tool-call syntax (refused by the parser
at the time) and DeepSeek's incomplete cells are re-run. Nothing is deleted: the removed records are appended to
runs/superseded_<date>/<same file name>, with the reason.

    python lb_skill_redo.py --model GLM-5.3                              (native-syntax cells)
    python lb_skill_redo.py --model DeepSeek-V4.1-Flash-thinking-low     (native-syntax or not-answered cells)
    python lb_skill_redo.py --model ... --dry-run                        (list only)
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import lb_lib as lb

EXPERIMENTS = ("shc_configure", "shc_design", "shc_implement")
NATIVE = ("<tool_call>", "</arg_value>", "DSML")


def select(r, only=None):
    if only == "call_error":                                   # 2026-09-28: gateway errors caused by the harness
        return "gateway error: empty assistant message sent by the loop" if r.get("ended") == "call_error" else None
    native = any(tag in (s.get("reply") or "") for s in r.get("steps", []) for tag in NATIVE)
    if native:
        return "native tool-call syntax refused by the parser"
    if r["model"].startswith("DeepSeek") and r.get("ended") != "answer":
        return f"not answered: {r.get('ended')}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["call_error"], default=None, help="set aside only cells that ended in a gateway error")
    a = ap.parse_args()
    stamp = datetime.date.today().isoformat()
    out_dir = lb.LB / "runs" / f"superseded_{stamp}"
    total = 0
    for exp in EXPERIMENTS:
        f = lb.LB / f"lb_replies_{exp}_{lb.safe_name(a.model)}_low.jsonl"
        if not f.exists():
            continue
        keep, moved = [], []
        for l in open(f):
            r = json.loads(l)
            why = select(r, a.only)
            if why:
                r["superseded_reason"] = why; r["superseded_on"] = stamp
                moved.append(json.dumps(r) + "\n")
            else:
                keep.append(l)
        print(f"{f.name}: {len(moved)} cell(s) set aside, {len(keep)} kept")
        for m in moved:
            r = json.loads(m); print(f"    {r['prompt_id']:34s} {r['superseded_reason']}")
        total += len(moved)
        if moved and not a.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f.name, "a") as g:
                g.writelines(moved)
            with open(f, "w") as g:
                g.writelines(keep)
    print(f"\n{total} cells to re-run" + ("" if a.dry_run else f"; superseded records in {out_dir}"))
    if total and not a.dry_run:
        print("run:\n" + "\n".join(f"  python lb_skill_run.py --skill {e.replace('shc_', 'shc-')} --models {a.model}" for e in EXPERIMENTS))


if __name__ == "__main__":
    main()
