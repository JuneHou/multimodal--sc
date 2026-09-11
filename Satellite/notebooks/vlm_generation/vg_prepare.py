"""vg_prepare — build every prompt of the first run: preview PNGs, contact sheets (both layouts),
band-feature lines, `vg_prompts.jsonl` and `vg_prompts_index.csv`. Runs here (satellite env);
nothing is generated here.

Grid: sensor {S2 treated 1-15, S1 treated 15-27} x arm {shc, scm} x band {0, 1} x target {P10, P11}.
The observed P10/P11 previews of every target site are also written (gallery only).
A prompt is skipped (and logged in vg_prompts_skipped.csv) when the target or any of its 5 similar
sites lacks a usable composite at a shown month.

    python vg_prepare.py
"""
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

import vg_lib as vg
import vg_render as vr
import panel_lib as pl
import panel_monthly as pm

ROW_LABELS = ["TARGET"] + [f"SIMILAR {k}" for k in range(1, vg.N_SIMILAR + 1)]


def preview_path(site, sensor, seq):
    return vg.PREVIEWS / f"{site}_{sensor}_P{seq:02d}.png"


def main():
    t0 = time.time()
    for d in (vg.PREVIEWS, vg.SHEETS, vg.SHEETS / "persite"):
        d.mkdir(parents=True, exist_ok=True)
    roster = pm.load_roster()
    idx = pd.read_csv(pm.INDEX_CSV)
    usable = {(r.site_id, r.sensor, int(r.seq)): bool(r.usable) for r in idx.itertuples()}
    tif = {(r.site_id, r.sensor, int(r.seq)): r.tif for r in idx.itertuples()}
    feats_cache, img_cache = {}, {}

    def picture(site, sensor, seq):
        """Preview PNG (written once) + band means, from the raw composite."""
        key = (site, sensor, seq)
        if key not in img_cache:
            chip = pm.read_chip(tif[key], sensor)               # (101, 101, C) raw units, NaN nodata
            arr = vr.PREVIEW[sensor](chip)
            assert arr is not None, key
            im = vr.to_image(arr)
            p = preview_path(*key)
            if not p.exists():
                im.save(p)
            img_cache[key] = im
            feats_cache[key] = pl.feats_from_raw(sensor, chip)
        return img_cache[key], feats_cache[key]

    prompts, skipped = [], []
    for sensor in vg.SENSORS:
        treated = sorted(idx[(idx.sensor == sensor) & (idx.group == "treatment")].site_id.unique())
        for site in treated:
            similar = vg.donors5(roster, site)
            for target in vg.TARGETS:
                for arm in vg.ARMS:
                    months = vg.months_shown(arm, target)
                    sites = [site] + (similar if arm == "scm" else [])
                    missing = [(s, q) for s in sites for q in months if not usable.get((s, sensor, q), False)]
                    if missing:
                        for band in (0, 1):
                            skipped.append({"sensor": sensor, "site": site, "arm": arm, "target": f"P{target:02d}",
                                            "band": band, "reason": "not usable: " + " ".join(f"{s}@P{q:02d}" for s, q in missing)})
                        continue
                    # pictures, band lines, sheets (shared by the two band arms)
                    rows, band_lines, persite_paths = [], [], []
                    for label, s in zip(ROW_LABELS, sites):
                        items = []
                        for q in months:
                            im, feats = picture(s, sensor, q)
                            items.append((im, f"P{q:02d} {vg.CALENDAR[q]}"))
                            band_lines.append(vg.band_line(sensor, label, q, feats))
                        rows.append((label, items))
                    stem = f"{vg.SENSOR_SHORT[sensor]}_{site}_{arm}_P{target:02d}"
                    sheet_path = vg.SHEETS / f"{stem}.png"
                    vr.contact_sheet(rows).save(sheet_path)
                    for label, items in rows:
                        p = vg.SHEETS / "persite" / f"{stem}_{label.replace(' ', '')}.png"
                        vr.contact_sheet([(label, items)]).save(p)
                        persite_paths.append(str(p.relative_to(vg.VG)))
                    for band in (0, 1):
                        pid = f"{stem}_band{band}"
                        rec = {"prompt_id": pid, "sensor": sensor, "arm": arm, "band": band,
                               "target": f"P{target:02d}", "target_site": site,
                               "similar_sites": similar if arm == "scm" else [],
                               "months": [f"P{q:02d}" for q in months],
                               "sheet_image": str(sheet_path.relative_to(vg.VG)),
                               "persite_images": persite_paths,
                               "prompt_sheet": vg.prompt_text(sensor, arm, target, "sheet", band_lines if band else None),
                               "prompt_persite": vg.prompt_text(sensor, arm, target, "persite", band_lines if band else None)}
                        prompts.append(rec)
            for target in vg.TARGETS:            # observed target-month previews, for the gallery only (never in a prompt)
                if usable.get((site, sensor, target), False):
                    picture(site, sensor, target)
            print(f"  {sensor} {site}: {sum(p['target_site'] == site and p['sensor'] == sensor for p in prompts)} prompts", flush=True)
    with open(vg.PROMPTS_JSONL, "w") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")
    cols = ["prompt_id", "sensor", "arm", "band", "target", "target_site", "similar_sites", "months", "sheet_image"]
    df = pd.DataFrame(prompts)[cols].copy()
    df["similar_sites"] = df["similar_sites"].map(" ".join); df["months"] = df["months"].map(" ".join)
    df.to_csv(vg.PROMPTS_CSV, index=False)
    pd.DataFrame(skipped, columns=["sensor", "site", "arm", "target", "band", "reason"]).to_csv(vg.VG / "vg_prompts_skipped.csv", index=False)
    print(f"{len(prompts)} prompts, {len(skipped)} skipped, {len(list(vg.PREVIEWS.glob('*.png')))} previews, "
          f"{time.time() - t0:.0f}s")
    print(df.groupby(["sensor", "arm", "target", "band"]).size().to_string())
    return df


if __name__ == "__main__":
    main()
