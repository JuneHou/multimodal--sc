"""lb_prompts — build lb_prompts_shc.jsonl (27) and lb_prompts_scm.jsonl (27) from lb_features_monthly.csv.
Each line: prompt_id, experiment, sensor, target_site, donors, prompt, truth (the withheld P10 values,
display names), train (the target's P01-P09 values per band, for her notebook-26 variance).
    python lb_prompts.py
"""
import json

import numpy as np
import pandas as pd

import lb_lib as lb


def rows_of(df, site, sensor):
    d = df[(df.site_id == site) & (df.sensor == sensor)]
    return {int(r.seq): r._asdict() for r in d.itertuples(index=False)}


def main():
    df = pd.read_csv(lb.FEATURES_CSV)
    out = {"shc": [], "scm": []}
    for sensor in lb.SENSORS:
        targets = sorted(df[(df.sensor == sensor) & (df.role == "target")].site_id.unique())
        for t in targets:
            tr = rows_of(df, t, sensor)
            truth = {lb.disp(b): float(tr[lb.TARGET][f"mean_{b}"]) for b in lb.BANDS[sensor]}
            train = {lb.disp(b): [float(tr[q][f"mean_{b}"]) for q in lb.PRE] for b in lb.BANDS[sensor]}
            base = {"sensor": sensor, "target_site": t, "truth": truth, "train": train,
                    "target_usable": {f"P{q:02d}": bool(tr[q]["usable"]) for q in range(1, lb.TARGET + 1)}}
            out["shc"].append({"prompt_id": f"{lb.SHORT[sensor]}_{t}_shc", "experiment": "shc", **base, "donors": [],
                               "prompt": lb.prompt_shc(sensor, tr)})
            dsites = df[(df.sensor == sensor) & (df.target_site == t) & (df.role == "donor")].sort_values("control_rank").site_id.unique().tolist()
            drows = [rows_of(df, d, sensor) for d in dsites]
            out["scm"].append({"prompt_id": f"{lb.SHORT[sensor]}_{t}_scm", "experiment": "scm", **base, "donors": dsites,
                               "prompt": lb.prompt_scm(sensor, tr, drows)})
    for exp, recs in out.items():
        with open(lb.LB / f"lb_prompts_{exp}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{exp}: {len(recs)} prompts")


if __name__ == "__main__":
    main()
