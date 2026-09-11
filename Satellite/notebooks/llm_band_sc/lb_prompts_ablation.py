"""lb_prompts_ablation — the four SHC-design ablation prompt files (plan 2026-09-10, analysis C).
    abl_noguide  band guide removed
    abl_anon     guide, sensor and band names removed (x1..xk); calendar kept
    abl_nocal    guide kept, calendar dates removed (P01..P10 only)
    abl_rule     the 2026-09-09 prompt + a second line naming the rule used
Each file has the 27 cells with the same truth/train fields as lb_prompts_shc.jsonl; prompt_id ends in the arm
name; 'labels' carries the band names used in the prompt when they differ from the real ones.
    python lb_prompts_ablation.py
"""
import json

import pandas as pd

import lb_lib as lb
from lb_prompts import rows_of

ARMS = {"abl_noguide": dict(guide=False), "abl_anon": dict(anon=True), "abl_nocal": dict(calendar=False),
        "abl_rule": dict(ask_rule=True)}


def main():
    df = pd.read_csv(lb.FEATURES_CSV)
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    for arm, kw in ARMS.items():
        recs = []
        for pid, p in base.items():
            s, t = p["sensor"], p["target_site"]
            r = dict(p); r["prompt_id"] = pid[:-4] + "_" + arm[4:]; r["experiment"] = arm
            r["prompt"] = lb.prompt_shc(s, rows_of(df, t, s), **kw)
            r["labels"] = lb.anon_labels(s) if kw.get("anon") else None
            recs.append(r)
        with open(lb.LB / f"lb_prompts_{arm}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{arm}: {len(recs)} prompts")


if __name__ == "__main__":
    main()
