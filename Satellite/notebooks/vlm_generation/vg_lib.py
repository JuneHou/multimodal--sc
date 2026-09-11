"""vg_lib — shared constants, roster helpers and prompt templates for the VLM direct-generation
direction (plan `quirky-growing-wombat`, approved 2026-09-08).

Design (Jun, 2026-09-08): the VLM is given PICTURES only and told which site is the target and
which sites are similar; no weights, no block labels. Two donor structures:
  shc  the target site's own earlier months (all of them)
  scm  the target site + its 5 covariate-matched controls (control_rank 1-5), the two most
       recent months per site
Two outputs generated independently: P10 from months up to P09, P11 from months up to P10.
Every site in a prompt stops at the same month. With / without band-feature text.
Evaluation is out of scope here.
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TS = HERE.parent / "ts_SCM_ASCM"
sys.path.insert(0, str(TS))
import panel_lib as pl            # noqa: E402
import panel_monthly as pm        # noqa: E402

VG = HERE
PREVIEWS = VG / "previews"
SHEETS = VG / "sheets"
OUTPUTS = VG / "outputs"
PROMPTS_JSONL = VG / "vg_prompts.jsonl"
PROMPTS_CSV = VG / "vg_prompts_index.csv"

SENSORS = ("sentinel1", "sentinel2")
SENSOR_SHORT = {"sentinel1": "s1", "sentinel2": "s2"}
MODELS = {"qwen": "Qwen/Qwen-Image-Edit-2511", "bagel": "ByteDance-Seed/BAGEL-7B-MoT"}
TARGETS = (10, 11)                # P10 <- months <= P09 ; P11 <- months <= P10
SCM_MONTHS = 2                    # Jun: "two months per donor and the target"
N_SIMILAR = 5                     # control_rank 1..5, the SCM donor set used throughout the repo
ARMS = ("shc", "scm")
LAYOUTS = ("sheet", "persite")

_cal = pd.read_csv(pm.MONTHLY / "nb25_monthly_long_period_definitions.csv")
CALENDAR = {int(p[1:]): m for p, m in zip(_cal["period_id"], _cal["month_label"])}   # seq -> "2024-08"

SENSOR_SENTENCE = {
    "sentinel2": ("Each picture is a Sentinel-2 optical satellite image of a land parcel about 1 km "
                  "across, shown in natural colour (red, green, blue bands). Black pixels are "
                  "cloud-masked, no data."),
    "sentinel1": ("Each picture is a Sentinel-1 radar satellite image of a land parcel about 1 km "
                  "across, shown in false colour (VV, VH and VV−VH backscatter as red, green, "
                  "blue). Black pixels are no data."),
}
BAND_UNITS = {
    "sentinel2": "reflectance 0-1 for B2 B3 B4 B8 B11 B12, plus NDVI and NDWI",
    "sentinel1": "dB for VV, VH, VV-VH",
}
BAND_DECIMALS = {"sentinel2": 3, "sentinel1": 2}


def plabel(seq):
    return f"P{seq:02d} = {CALENDAR[seq]}"


def months_shown(arm, target):
    """Months every site shows for (arm, target). SHC: all months before the target; SCM: the
    SCM_MONTHS most recent ones."""
    allm = list(range(1, target))
    return allm if arm == "shc" else allm[-SCM_MONTHS:]


def donors5(roster, site):
    """The site's 5 covariate-matched controls, control_rank 1..5 in rank order."""
    d = roster[(roster.matched_treatment_site_id == site) & (roster.group == "counterfactual")
               & (roster.control_rank >= 1) & (roster.control_rank <= N_SIMILAR)]
    d = d.sort_values("control_rank")["site_id"].tolist()
    assert len(d) == N_SIMILAR, (site, d)
    return d


def prompt_text(sensor, arm, target, layout, band_lines=None):
    """The entire prompt (plan section 'Prompts'). band_lines: list of strings or None."""
    months = months_shown(arm, target)
    mlist = ", ".join(plabel(q) for q in months)
    nxt = plabel(target)
    if arm == "shc":
        head = (f"You are given {len(months)} pictures of one land parcel, one picture per month, "
                f"in time order:\n{mlist}.")
        gen = (f"Generate the picture of the same parcel for the next month, {nxt}, with the same "
               f"viewpoint, scale and rendering as the given pictures. Output only that one picture.")
    else:
        head = (f"You are given pictures of six land parcels for the {len(months)} most recent months, "
                f"in time order:\n{mlist}. One picture per month per parcel.")
        if layout == "sheet":
            head += ("\nThe first row is the TARGET parcel. The other five rows are SIMILAR parcels: "
                     "parcels in the same region with the same land cover, elevation and slope as the target.")
        else:
            head += ("\nThe first image is the TARGET parcel. Images 2 to 6 are SIMILAR parcels: "
                     "parcels in the same region with the same land cover, elevation and slope as the "
                     "target. Each image shows one parcel's months in time order.")
        gen = (f"Generate the picture of the TARGET parcel for the next month, {nxt}, with the same "
               f"viewpoint, scale and rendering as the given pictures. Output only that one picture.")
    parts = [head, SENSOR_SENTENCE[sensor]]
    if band_lines:
        parts.append(f"Mean band values of each picture ({BAND_UNITS[sensor]}):\n" + "\n".join(band_lines))
    parts.append(gen)
    return "\n".join(parts)


def band_line(sensor, row_label, seq, feats):
    """'TARGET P01: B2 0.034, B3 0.052, ...' from panel_lib.feats_from_raw output."""
    d = BAND_DECIMALS[sensor]
    vals = ", ".join(f"{b.replace('_minus_', '-')} {feats[b]:.{d}f}" for b in pl.BANDS[sensor])
    return f"{row_label} P{seq:02d}: {vals}"
