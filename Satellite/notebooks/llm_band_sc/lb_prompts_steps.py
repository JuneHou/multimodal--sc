"""lb_prompts_steps — the SHC-design step-by-step prompt arm (plan 2026-09-15, approved): the same
P01-P09 -> P10 SHC-design prompt as lb_prompts_shc.jsonl, with her paper's block/weight-fitting
procedure added in prose (lb_lib.SHC_STEPS) before the answer line, so the model's reasoning is
scaffolded. Data given to the model, and the answer format, are unchanged from lb_prompts_shc.jsonl.
    python lb_prompts_steps.py
"""
import json

import pandas as pd

import lb_lib as lb
from lb_prompts import rows_of


def main():
    df = pd.read_csv(lb.FEATURES_CSV)
    base = [json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")]
    recs = []
    for p in base:
        s, t = p["sensor"], p["target_site"]
        r = dict(p)
        r["prompt_id"] = p["prompt_id"][:-4] + "_steps"
        r["experiment"] = "shc_steps"
        r["prompt"] = lb.prompt_shc(s, rows_of(df, t, s), steps=True)
        recs.append(r)
    with open(lb.LB / "lb_prompts_shc_steps.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"shc_steps: {len(recs)} prompts")


if __name__ == "__main__":
    main()
