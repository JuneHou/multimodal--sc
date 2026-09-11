"""vg_text_api — text-output arm on the ARC hosted vision models (llm-api.arc.vt.edu, OpenAI-compatible).
The model sees the same pictures as the image models (persite layout by default) and, instead of a picture,
writes the mean band values of the TARGET parcel at the next month, one line, same units as the band text.
Only the band-text prompts (band=1) are used: the model cannot measure band values from a picture, so the
band table of every input picture is always in the prompt (Jun, 2026-09-09). 108 prompts per model.

Key: read from $ARC_LLM_API_KEY, else from ~/.config/arc_llm_key (chmod 600). Never stored in the repo.
    python vg_text_api.py --models Kimi-K3 DeepSeek-V4-Flash --layout persite [--limit 4] [--seed 0]
Outputs: vg_text_outputs_<model>_<layout>.jsonl (raw reply, parsed values, usage, seconds; resumable),
         vg_text_outputs.csv (all models joined to the prompt fields, plus the observed band means of the
         output month for later use; no scoring here).
"""
import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

import vg_lib as vg
import panel_lib as pl
import panel_monthly as pm

BASE = "https://llm-api.arc.vt.edu/api/v1"
VISION_MODELS = ["Kimi-K3", "DeepSeek-V4-Flash"]
FORMAT_LINE = {
    "sentinel2": "B2 0.000, B3 0.000, B4 0.000, B8 0.000, B11 0.000, B12 0.000, NDVI 0.000, NDWI 0.000",
    "sentinel1": "VV 0.00, VH 0.00, VV-VH 0.00",
}


def api_key():
    k = os.environ.get("ARC_LLM_API_KEY")
    p = Path.home() / ".config" / "arc_llm_key"
    if not k and p.exists():
        k = p.read_text().strip()
    if not k:
        raise SystemExit("no API key: export ARC_LLM_API_KEY or write it to ~/.config/arc_llm_key")
    return k


BAND_MEANING = {
    "sentinel2": ("The bands are Sentinel-2 surface reflectance: B2 blue, B3 green, B4 red, B8 near-infrared, "
                  "B11 and B12 shortwave infrared, all on a 0-1 scale; NDVI is a vegetation index in [-1, 1] "
                  "(higher = greener vegetation) and NDWI a water index in [-1, 1]."),
    "sentinel1": ("The bands are Sentinel-1 radar backscatter in decibels: VV and VH polarisations "
                  "(more negative = weaker return) and VV-VH their difference."),
}


def text_prompt(p, layout):
    """Text-output prompt: the picture description, then the band table of EVERY input picture (the model
    cannot measure band values from a picture, so they are always supplied), then the ask."""
    assert p["band"] == 1, "text-output arm runs on band-text prompts only"
    base = p["prompt_sheet"] if layout == "sheet" else p["prompt_persite"]
    head = base.rsplit("Generate the picture", 1)[0].rstrip()          # description + band table
    who = "the same parcel" if p["arm"] == "shc" else "the TARGET parcel"
    nxt = vg.plabel(int(p["target"][1:]))
    return (f"{head}\n{BAND_MEANING[p['sensor']]}\n"
            f"Using the pictures and the band values above, predict the mean band values of {who} for the next "
            f"month, {nxt} ({vg.BAND_UNITS[p['sensor']]}). Reply with exactly one line in this format and "
            f"nothing else:\n{FORMAT_LINE[p['sensor']]}")


def b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def parse(sensor, text):
    names = [b.replace("_minus_", "-") for b in pl.BANDS[sensor]]
    out = {}
    for n in names:
        m = re.search(rf"(?<![A-Za-z0-9-]){re.escape(n)}\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text)
        if m:
            out[n] = float(m.group(1))
    return out if len(out) == len(names) else None


def call(key, model, prompt, images, seed):
    content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": b64(vg.VG / im)}} for im in images]
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 300, "seed": seed}
    r = requests.post(f"{BASE}/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json=body, timeout=300)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], j.get("usage", {})


def observed_bands(sensor, site, seq, idx):
    r = idx[(idx.site_id == site) & (idx.sensor == sensor) & (idx.seq == seq) & (idx.usable)]
    if not len(r):
        return {}
    f = pl.feats_from_raw(sensor, pm.read_chip(r.tif.iloc[0], sensor))
    return {f"observed_{b.replace('_minus_', '-')}": float(v) for b, v in f.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=VISION_MODELS)
    ap.add_argument("--layout", choices=["sheet", "persite"], default="persite")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    key = api_key()
    P = [json.loads(l) for l in open(vg.PROMPTS_JSONL) if json.loads(l)["band"] == 1]   # band table always present
    if a.limit:
        keep = {}
        for p in P:
            keep.setdefault((p["sensor"], p["arm"]), p)
        P = list(keep.values())[:a.limit]
    idx = pd.read_csv(pm.INDEX_CSV)
    for model in a.models:
        mf = vg.VG / f"vg_text_outputs_{model}_{a.layout}.jsonl"
        done = {json.loads(l)["prompt_id"] for l in open(mf)} if mf.exists() else set()
        todo = [p for p in P if p["prompt_id"] not in done]
        print(f"{model}: {len(todo)} to do ({len(done)} done)", flush=True)
        with open(mf, "a") as f:
            for i, p in enumerate(todo):
                images = [p["sheet_image"]] if a.layout == "sheet" else p["persite_images"]
                prompt = text_prompt(p, a.layout)
                t0 = time.time(); rec = {"prompt_id": p["prompt_id"], "model": model, "layout": a.layout, "seed": a.seed,
                                         "prompt": prompt, "n_images": len(images)}
                try:
                    text, usage = call(key, model, prompt, images, a.seed)
                    vals = parse(p["sensor"], text)
                    rec.update(ok=vals is not None, reply=text, values=vals, usage=usage)
                except Exception as e:
                    rec.update(ok=False, error=f"{type(e).__name__}: {e}")
                rec["seconds"] = round(time.time() - t0, 1)
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{i + 1}/{len(todo)}] {p['prompt_id']} {'ok' if rec['ok'] else 'FAILED'} {rec['seconds']}s "
                      f"{rec.get('reply', rec.get('error', ''))[:90]!r}", flush=True)
    # joined table
    rows = []
    PP = {p["prompt_id"]: p for p in (json.loads(l) for l in open(vg.PROMPTS_JSONL))}
    obs_cache = {}
    for mf in sorted(vg.VG.glob("vg_text_outputs_*.jsonl")):
        for line in open(mf):
            r = json.loads(line); p = PP[r["prompt_id"]]
            row = {k: p[k] for k in ("prompt_id", "sensor", "arm", "band", "target", "target_site")}
            row.update(model=r["model"], layout=r["layout"], ok=r["ok"], seconds=r["seconds"], reply=r.get("reply", r.get("error", "")))
            for k, v in (r.get("values") or {}).items():
                row[f"pred_{k}"] = v
            ok_ = (p["sensor"], p["target_site"], int(p["target"][1:]))
            if ok_ not in obs_cache:
                obs_cache[ok_] = observed_bands(*ok_, idx)
            row.update(obs_cache[ok_])
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows); df.to_csv(vg.VG / "vg_text_outputs.csv", index=False)
        print(df.groupby(["model", "layout", "sensor", "arm", "band", "target"]).agg(n=("ok", "size"), ok=("ok", "sum"), sec=("seconds", "mean")).to_string())


if __name__ == "__main__":
    main()
