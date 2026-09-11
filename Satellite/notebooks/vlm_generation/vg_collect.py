"""vg_collect — after the outputs are copied back: join every manifest line to its prompt, check the
PNG opens, write vg_generated_index.csv and print a count table. No scoring here.
    python vg_collect.py
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image

import vg_lib as vg


def main():
    P = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(vg.PROMPTS_JSONL)}
    rows = []
    for mf in sorted(vg.VG.glob("vg_outputs_*.jsonl")):
        for line in open(mf):
            r = json.loads(line); p = P[r["prompt_id"]]
            row = {k: p[k] for k in ("prompt_id", "sensor", "arm", "band", "target", "target_site")}
            row.update(model=r["model"], layout=r["layout"], seed=r["seed"], ok=r.get("ok", False),
                       file=r.get("file"), seconds=r.get("seconds"), error=r.get("error", ""))
            if row["ok"]:
                try:
                    with Image.open(vg.VG / r["file"]) as im:
                        row["width"], row["height"] = im.size
                        row["readable"] = True
                except Exception as e:
                    row["readable"] = False; row["error"] = f"unreadable: {e}"
            rows.append(row)
    if not rows:
        print("no manifests yet (vg_outputs_*.jsonl); nothing generated"); return None
    df = pd.DataFrame(rows); df.to_csv(vg.VG / "vg_generated_index.csv", index=False)
    print(f"{len(df)} outputs, {int(df.ok.sum())} ok, {int((~df.ok).sum())} failed; "
          f"readable {int(df.get('readable', pd.Series(dtype=bool)).fillna(False).sum())}")
    print(df.groupby(["model", "layout", "sensor", "arm", "band", "target"]).agg(n=("ok", "size"), ok=("ok", "sum"),
          sec=("seconds", "mean")).to_string())
    return df


if __name__ == "__main__":
    main()
