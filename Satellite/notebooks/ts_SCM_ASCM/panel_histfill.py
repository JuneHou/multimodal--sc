"""panel_histfill — Experiment 2 cloud handling: fill masked pixels from the SAME site's
clean pre-treatment history, then re-encode with the frozen TerraMind tokenizers.

Rule (plan E2.2–E2.5, approved 2026-08-28): for every chip, a non-finite pixel is replaced
by the per-pixel MEDIAN of that site's P01–P08 chips where that pixel is finite
(>= MIN_CLEAN periods). If fewer periods are available the median over whatever is
available is used (>= 1), and only pixels never observed in P01–P08 fall back to the
chip mean of the filled chip inside `Tok.prep` (`fill_nan`). Every fill is counted.

History is P01–P08 for every period, including the P09/P10 targets and the post
periods, so the target chip's own pixels never inform their own fill and the fill never
sees a period beyond the training window. Same parcel, same class, same position by
construction — no alignment. Produces `latents_biweekly_histfill.npz` the way notebook 05
produced the generation-fill cache; the cache is scored by the notebook-11 machinery.
"""
import json

import numpy as np
import pandas as pd

from panel_lib import LATD, SENSORS, TRAIN8, read_chip_biweekly, save_latents

MIN_CLEAN = 3
CACHE = LATD / "latents_biweekly_histfill.npz"


def history_template(idx, site, sensor, periods=TRAIN8, min_clean=MIN_CLEAN):
    """(template (101,101,C) float32 with NaN where never observed, count (101,101),
    template_any) from the site's existing P01-P08 chips."""
    rows = idx.query("site_id == @site and sensor == @sensor and file_exists and seq in @periods")
    if not len(rows):
        return None, None, None
    stack = np.stack([read_chip_biweekly(r.tif, sensor) for r in rows.itertuples()])
    fin = np.isfinite(stack).all(axis=-1)                     # (T, H, W)
    count = fin.sum(axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)                     # NaN where count == 0
    tmpl = np.where((count >= min_clean)[..., None], med, np.nan).astype(np.float32)
    tmpl_any = np.where((count >= 1)[..., None], med, np.nan).astype(np.float32)
    return tmpl, count, tmpl_any


def histfill_chip(chip, tmpl, tmpl_any):
    """Return (filled chip, n_masked, n_filled_median, n_filled_any); pixels still NaN
    afterwards are chip-mean filled by `Tok.prep`."""
    out = chip.copy()
    m = ~np.isfinite(chip).all(axis=-1)
    n_masked = int(m.sum())
    if n_masked == 0 or tmpl is None:
        return out, n_masked, 0, 0
    f1 = m & np.isfinite(tmpl).all(axis=-1)
    out[f1] = tmpl[f1]
    f2 = m & ~f1 & np.isfinite(tmpl_any).all(axis=-1)
    out[f2] = tmpl_any[f2]
    return out, n_masked, int(f1.sum()), int(f2.sum())


def build_cache(idx, tokb, batch=64, cache=CACHE, log=print):
    """Encode every existing chip after historical fill. Returns (lat dict, fill-count
    DataFrame). Determinism gate as in notebooks 01/05 (re-encode first batch, exact)."""
    lat, counts = {}, []
    todo = idx.loc[idx["file_exists"]]
    for sensor in SENSORS:
        rows = [r for r in todo.itertuples() if r.sensor == sensor]
        tmpls = {}
        jobs = []
        for r in rows:
            if r.site_id not in tmpls:
                tmpls[r.site_id] = history_template(idx, r.site_id, sensor)
            tmpl, cnt, tmpl_any = tmpls[r.site_id]
            chip = read_chip_biweekly(r.tif, sensor)
            filled, nm, n1, n2 = histfill_chip(chip, tmpl, tmpl_any)
            counts.append({"site_id": r.site_id, "sensor": sensor, "period_id": r.period_id,
                           "seq": r.seq, "n_masked": nm, "n_filled_median": n1,
                           "n_filled_any": n2, "n_chipmean_fallback": nm - n1 - n2,
                           "history_periods": int(cnt.max()) if cnt is not None else 0})
            jobs.append((r, tokb.prep(sensor, filled)))
        for s0 in range(0, len(jobs), batch):
            chunk = jobs[s0:s0 + batch]
            q = tokb.encode_batch(sensor, [j[1] for j in chunk])
            for (r, _), qi in zip(chunk, q):
                lat[(r.site_id, sensor, r.period_id)] = qi.numpy()
            log(f"{sensor}: {min(s0 + batch, len(jobs))}/{len(jobs)}")
        first = jobs[:batch]
        q2 = tokb.encode_batch(sensor, [j[1] for j in first])
        g = max(float(np.abs(lat[(r.site_id, sensor, r.period_id)] - qi.numpy()).max())
                for (r, _), qi in zip(first, q2))
        log(f"encode determinism gate {sensor}: max|diff| = {g:.2e}")
        assert g < 1e-5
    return lat, pd.DataFrame(counts)


def write_cache(lat, counts, cache=CACHE):
    save_latents(lat, cache)
    counts.to_csv(cache.with_suffix(".fillcounts.csv"), index=False)
    manifest = {"date": "2026-08-28",
                "base": "biweekly chips (same 2322 images as latents_biweekly.npz)",
                "fill": "per-pixel median of the same site's P01-P08 chips where the pixel is "
                        f"finite in >= {MIN_CLEAN} periods; else median over >= 1 period; else "
                        "chip mean (Tok.prep fill_nan). History = P01-P08 for EVERY period.",
                "encoder": "terramind_v1_tokenizer_s1grd / s2l2a, batch 64, determinism gate",
                "n_latents": len(lat),
                "fill_totals": {k: int(counts[k].sum()) for k in
                                ("n_masked", "n_filled_median", "n_filled_any",
                                 "n_chipmean_fallback")}}
    cache.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
