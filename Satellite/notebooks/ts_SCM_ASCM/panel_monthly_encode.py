"""Encode every usable monthly composite (P01–P21, both sensors, treated + counterfactual
sites) into TerraMind tokenizer latents, under the two cloud fills:

    chip-mean fill   `Tok.prep` as is: NaN pixels -> the chip's own band mean, then encode.
                     -> data/embeddings_tok_panel/latents_monthly_long.npz
    historical fill  per-pixel median of the site's own clean PRE months (P01–P10, a pixel
                     counts when every band is finite, >= MIN_CLEAN months), applied to all
                     21 periods; pixels never clean in the history fall back to the 1-month
                     template, then to chip mean inside `Tok.prep`.
                     -> data/embeddings_tok_panel/latents_monthly_long_histfill.npz
                        + .fillcounts.csv

Same tokenizers, preprocessing and BATCH as the biweekly cache (notebook 01 / panel_histfill);
the determinism gate re-encodes the first chunk of each sensor at the ORIGINAL batch shape.

    BATCH=64 python panel_monthly_encode.py            # both arms
    python panel_monthly_encode.py chipmean|histfill   # one arm
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import panel_lib as pl, panel_monthly as pm
pl.pick_gpu()
import numpy as np, pandas as pd

BATCH = int(os.environ.get("BATCH", "64"))
MIN_CLEAN = 3
GATE_TOL = 1e-5


def encode_jobs(tokb, jobs, log=print):
    """jobs: list of (key, sensor, (H,W,C) chip). Returns {key: (5,14,14) float32}."""
    lat = {}
    for sensor in pm.SENSORS:
        js = [j for j in jobs if j[1] == sensor]
        first = None
        t0 = time.time()
        for i in range(0, len(js), BATCH):
            chunk = js[i:i + BATCH]
            tensors = [tokb.prep(sensor, chip) for _, _, chip in chunk]
            q = tokb.encode_batch(sensor, tensors)
            for (key, _, _), qi in zip(chunk, q):
                lat[key] = qi.numpy().astype(np.float32)
            if first is None:
                first = (tensors, q)
            if (i // BATCH) % 10 == 0:
                log(f"  {sensor}: {i + len(chunk)}/{len(js)} encoded ({time.time() - t0:.0f}s)")
        # determinism gate at the original batch shape
        if first is not None:
            g = float((tokb.encode_batch(sensor, first[0]) - first[1]).abs().max())
            assert g < GATE_TOL, f"{sensor}: re-encode differs by {g:.2e}"
            log(f"  {sensor}: determinism gate max|dq| = {g:.1e}")
    return lat


def history_template(chips_pre, min_clean=MIN_CLEAN):
    """chips_pre: list of (H,W,C) usable pre-period chips. Returns (tmpl, count, tmpl_any)."""
    stack = np.stack(chips_pre)                       # (T,H,W,C)
    fin = np.isfinite(stack).all(axis=-1)             # (T,H,W)
    count = fin.sum(0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(fin[..., None], stack, np.nan), axis=0)
    tmpl = np.where((count >= min_clean)[..., None], med, np.nan).astype(np.float32)
    tmpl_any = np.where((count >= 1)[..., None], med, np.nan).astype(np.float32)
    return tmpl, count, tmpl_any


def histfill_chip(chip, tmpl, tmpl_any):
    m = ~np.isfinite(chip).all(axis=-1)
    out = chip.copy()
    n_masked = int(m.sum())
    use_med = m & np.isfinite(tmpl).all(axis=-1)
    out[use_med] = tmpl[use_med]
    use_any = m & ~use_med & np.isfinite(tmpl_any).all(axis=-1)
    out[use_any] = tmpl_any[use_any]
    return out, n_masked, int(use_med.sum()), int(use_any.sum())


def main(arms=("chipmean", "histfill")):
    idx = pm.build_index()
    idx.to_csv(pm.INDEX_CSV, index=False)
    use = idx[idx.usable].reset_index(drop=True)
    print(f"index: {len(idx)} rows, {len(use)} usable composites", flush=True)
    tokb = pl.load_tokenizers()
    base = {"key_format": "site_id|sensor|Pnn", "latent_shape": [5, 14, 14], "batch": BATCH,
            "periods": "P01..P21 (P01-P10 pre, P11 = Sep 2024 treatment period)",
            "source": str(pm.MONTHLY), "preprocessing": "panel_lib.Tok.prep (101->224 bilinear, "
            "TerraMind v1 standardization; S2 6 reflectance bands x1e4 into the 12-band slots, "
            "S1 VV/VH dB)", "n_composites": int(len(use))}

    if "chipmean" in arms:
        print("== chip-mean fill", flush=True)
        jobs = [((r.site_id, r.sensor, r.period_id), r.sensor, pm.read_chip(r.tif, r.sensor))
                for r in use.itertuples()]
        lat = encode_jobs(tokb, jobs)
        assert len(lat) == len(use)
        pl.save_latents(lat, pm.CACHE_CHIPMEAN, manifest={**base, "fill": "chip-mean (Tok.prep fill_nan)"})
        print(f"saved {pm.CACHE_CHIPMEAN} ({len(lat)} latents)", flush=True)
        del jobs, lat

    if "histfill" in arms:
        print("== historical fill", flush=True)
        jobs, counts = [], []
        for (site, sensor), g in use.groupby(["site_id", "sensor"], sort=True):
            g = g.sort_values("seq")
            chips = {int(r.seq): pm.read_chip(r.tif, sensor) for r in g.itertuples()}
            pre = [chips[q] for q in pm.PRE if q in chips]
            if pre:
                tmpl, count, tmpl_any = history_template(pre)
            else:
                tmpl = tmpl_any = None
            for r in g.itertuples():
                chip = chips[int(r.seq)]
                if tmpl is None:
                    filled, nm, nmed, nany = chip, int((~np.isfinite(chip).all(-1)).sum()), 0, 0
                else:
                    filled, nm, nmed, nany = histfill_chip(chip, tmpl, tmpl_any)
                left = int((~np.isfinite(filled).all(-1)).sum())
                counts.append({"site_id": site, "sensor": sensor, "period_id": r.period_id, "seq": int(r.seq),
                               "n_masked": nm, "n_filled_median": nmed, "n_filled_any": nany,
                               "n_chipmean_fallback": left, "history_months": len(pre)})
                jobs.append(((site, sensor, r.period_id), sensor, filled))
        lat = encode_jobs(tokb, jobs)
        assert len(lat) == len(use)
        C = pd.DataFrame(counts)
        C.to_csv(str(pm.CACHE_HISTFILL).replace(".npz", ".fillcounts.csv"), index=False)
        pl.save_latents(lat, pm.CACHE_HISTFILL, manifest={**base, "fill": f"historical: per-pixel median of own "
                        f"clean P01-P10 months (>= {MIN_CLEAN}), then 1-month template, then chip mean"})
        print(f"saved {pm.CACHE_HISTFILL} ({len(lat)} latents); masked px filled from median "
              f"{C.n_filled_median.sum()}, from any {C.n_filled_any.sum()}, chip-mean fallback "
              f"{C.n_chipmean_fallback.sum()}", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("chipmean", "histfill"))
