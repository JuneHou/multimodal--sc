"""vg_tests — hard checks for the prompt build.
1. Our preview arrays equal the collaborator's notebook 07/08 code run on the same file
   (their code pasted verbatim below, reading bands with tifffile in the same order rasterio
   band 1..3 gives).
2. donors5 = exactly the 5 control_rank 1..5 sites; every prompt has the right number of sites and months.
3. Sheet geometry matches the jsonl (rows = sites, columns = months).
4. Prints one full prompt per arm x band for inspection.
    python vg_tests.py
"""
import json
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

import vg_lib as vg
import vg_render as vr
import panel_monthly as pm


# ---- collaborator's code, verbatim (data/scripts/07 create_rgb_preview, 08 percentile_stretch /
#      create_sentinel1_preview), with rasterio `source.read(k, masked=True).filled(np.nan)`
#      replaced by the k-th plane of the channel-first tif.
def collab_s2(tif):
    a = tifffile.imread(tif).astype("float32"); a[~np.isfinite(a)] = np.nan
    blue, green, red = a[0], a[1], a[2]
    rgb = np.stack([red, green, blue], axis=-1)
    valid_values = rgb[np.isfinite(rgb)]
    lower = np.nanpercentile(valid_values, 2)
    upper = np.nanpercentile(valid_values, 98)
    if upper <= lower:
        upper = lower + 1e-6
    stretched = np.clip((rgb - lower) / (upper - lower), 0, 1)
    stretched[~np.isfinite(stretched)] = 0
    return stretched


def collab_stretch(array, lower_percentile=2, upper_percentile=98):
    array = array.astype("float32")
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros_like(array, dtype="float32")
    lower = np.nanpercentile(valid, lower_percentile)
    upper = np.nanpercentile(valid, upper_percentile)
    if upper <= lower:
        upper = lower + 1e-6
    stretched = np.clip((array - lower) / (upper - lower), 0, 1)
    stretched[~np.isfinite(stretched)] = 0
    return stretched


def collab_s1(tif):
    a = tifffile.imread(tif).astype("float32"); a[~np.isfinite(a)] = np.nan
    vv, vh, difference = a[0], a[1], a[2]
    return np.stack([collab_stretch(vv), collab_stretch(vh), collab_stretch(difference)], axis=-1)


def main():
    idx = pd.read_csv(pm.INDEX_CSV)
    rng = np.random.default_rng(0)
    worst = {}
    for sensor, ref in (("sentinel2", collab_s2), ("sentinel1", collab_s1)):
        rows = idx[(idx.sensor == sensor) & (idx.usable)].sample(20, random_state=0)
        w = 0.0
        for r in rows.itertuples():
            ours = vr.PREVIEW[sensor](pm.read_chip(r.tif, sensor))
            theirs = ref(r.tif)
            w = max(w, float(np.abs(ours - theirs).max()))
            png = np.asarray(Image.open(vg.PREVIEWS / f"{r.site_id}_{sensor}_P{r.seq:02d}.png")) \
                if (vg.PREVIEWS / f"{r.site_id}_{sensor}_P{r.seq:02d}.png").exists() else None
            if png is not None:
                assert np.abs(png.astype(float) - np.rint(theirs * 255)).max() <= 1, "png differs from collab preview"
        worst[sensor] = w
        assert w == 0.0, (sensor, w)
    print("1. preview == collaborator's notebook 07/08 arrays on 20 chips per sensor: max|diff| =", worst)

    roster = pm.load_roster()
    P = [json.loads(l) for l in open(vg.PROMPTS_JSONL)]
    for p in P:
        n_sites = 1 + (vg.N_SIMILAR if p["arm"] == "scm" else 0)
        months = vg.months_shown(p["arm"], int(p["target"][1:]))
        assert p["months"] == [f"P{q:02d}" for q in months], p["prompt_id"]
        assert months[-1] == int(p["target"][1:]) - 1, "every site stops the month before the target"
        if p["arm"] == "scm":
            assert p["similar_sites"] == vg.donors5(roster, p["target_site"])
            ranks = roster.set_index("site_id").loc[p["similar_sites"], "control_rank"].tolist()
            assert ranks == [1, 2, 3, 4, 5], ranks
        assert len(p["persite_images"]) == n_sites
        sheet = Image.open(vg.VG / p["sheet_image"])
        tw, th = 101 * 2, 101 * 2 + 22
        W = 20 + len(months) * tw + (len(months) - 1) * 8
        H = 20 + n_sites * th + (n_sites - 1) * 8
        assert sheet.size == (W, H), (p["prompt_id"], sheet.size, (W, H))
        assert "hurricane" not in p["prompt_sheet"].lower() and "weight" not in p["prompt_sheet"].lower()
    print(f"2-3. {len(P)} prompts: months, donor ranks 1-5, image counts and sheet geometry all consistent")
    seen = set()
    for p in P:
        k = (p["sensor"], p["arm"], p["band"])
        if k in seen or p["target"] != "P10":
            continue
        seen.add(k)
        print(f"\n===== {p['prompt_id']} (layout sheet) =====\n{p['prompt_sheet']}")
    print("\nvg_tests: all checks passed")


if __name__ == "__main__":
    main()
