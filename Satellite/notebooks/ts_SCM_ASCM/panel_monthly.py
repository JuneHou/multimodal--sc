"""panel_monthly — index, chip reader and latent panel for the MONTHLY composites.

Data: `Satellite/data/monthly_long_datasets/{sensor}/{treatment|counterfactual}/{before|after}/
{site}_{Pnn}_{start}_{end}_monthly_long_{sensor}.tif`, notebook-25 calendar-month nanmedian
composites, 21 periods P01 (Nov 2023) … P21 (Jul 2025). Channel-first (C, 101, 101) float32,
NaN nodata; S2 = B2,B3,B4,B8,B11,B12,NDVI,NDWI, S1 = VV,VH,VV-VH (same layout as the biweekly
tifs, so `panel_lib.read_chip_biweekly` does the transpose and the band-count assert).

Timeline (meeting 2026-09-04 + Jun): P01–P10 pre-hurricane, **P11 (Sep 2024) is the treatment
period**, P12… post. Site scope per sensor is whatever the collaborator's notebook 25 built:
Sentinel-2 treatment_0001–0015 + their matched counterfactuals, Sentinel-1 treatment_0015–0027
+ theirs (the two sensors cover different treated sites; only site 15 is in both).

Index rows come from the notebook-25 inventories; a site × period is *usable* when its file
exists and the build did not fail (the 17 "failed" S2 composites are 100 % NaN on disk) or
report no acquisition.

Latent cache key format: ``site_id|sensor|Pnn`` (no before_/after_ prefix; seq = nn).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import panel_lib as pl

MONTHLY = pl.ROOT / "Satellite" / "data" / "monthly_long_datasets"
MATCH_CSV = pl.ROOT / "Satellite" / "data" / "finals" / "site_matching_table.csv"
CACHE_CHIPMEAN = pl.LATD / "latents_monthly_long.npz"
CACHE_HISTFILL = pl.LATD / "latents_monthly_long_histfill.npz"
INDEX_CSV = pl.TS / "panel_monthly_index.csv"

PERIODS = list(range(1, 22))          # P01..P21
PRE = list(range(1, 11))              # P01..P10, T0 = 10
T0 = 10
TREAT_START = 11                      # P11 = September 2024, Helene landfall 27 Sep
SENSORS = pl.SENSORS


def pid(seq):
    return f"P{int(seq):02d}"


def load_roster():
    """One row per site: site_id, group, matched_treatment_site_id (own id for treated),
    control_rank (0 for treated). From the 26 × 10 matching table."""
    m = pd.read_csv(MATCH_CSV)
    t = pd.DataFrame({"site_id": sorted(m["treatment_site_id"].unique())})
    t["group"] = "treatment"; t["matched_treatment_site_id"] = t["site_id"]; t["control_rank"] = 0
    c = m[["counterfactual_site_id", "treatment_site_id", "control_rank"]].rename(
        columns={"counterfactual_site_id": "site_id", "treatment_site_id": "matched_treatment_site_id"})
    c["group"] = "counterfactual"
    r = pd.concat([t, c[t.columns]], ignore_index=True)
    assert r["site_id"].is_unique
    return r


def build_index(roster=None):
    """Every (site, sensor, period) the inventories know about, with the constructed tif
    path, `file_exists` and `usable`."""
    roster = load_roster() if roster is None else roster
    frames = []
    for sensor in SENSORS:
        inv = pd.read_csv(MONTHLY / f"nb25_monthly_long_{sensor}_inventory.csv")
        inv = inv[["site_id", "group", "period", "period_id", "period_start", "period_end",
                   "month_label", "acquisition_count", "build_action", "valid_pixel_fraction",
                   "quality_label", "validation_status"]].copy()
        inv["sensor"] = sensor
        frames.append(inv)
    idx = pd.concat(frames, ignore_index=True)
    idx["seq"] = idx["period_id"].str.extract(r"P(\d+)")[0].astype(int)
    idx["tif"] = [str(MONTHLY / r.sensor / r.group / r.period /
                      f"{r.site_id}_{r.period_id}_{r.period_start}_{r.period_end}_monthly_long_{r.sensor}.tif")
                  for r in idx.itertuples()]
    idx["file_exists"] = [Path(p).exists() for p in idx["tif"]]
    idx["usable"] = idx["file_exists"] & ~idx["build_action"].isin(["no_data", "failed"])
    idx = idx.merge(roster[["site_id", "matched_treatment_site_id", "control_rank"]],
                    on="site_id", how="left")
    assert idx["matched_treatment_site_id"].notna().all(), "site in inventory but not in the matching table"
    # the inventories list a file for every non-missing build: check the two agree
    built = idx["build_action"].isin(["created", "skipped_existing", "failed"])
    assert (idx["file_exists"] == built).all(), "inventory / disk disagreement"
    return idx.sort_values(["sensor", "site_id", "seq"]).reset_index(drop=True)


def read_chip(path, sensor):
    """(101, 101, C) float32 in native units, NaN where masked; the notebook-01 unit gates."""
    c = pl.read_chip_biweekly(path, sensor)
    assert c.shape[:2] == (101, 101), c.shape
    if sensor == "sentinel1":
        v = np.nanmean(c[..., 0]); assert -35.0 < v < 5.0, ("S1 VV not in dB?", path, v)
    else:
        if np.isfinite(c[..., 0]).any():
            v = np.nanmean(c[..., 0]); assert 0.0 < v < 0.5, ("S2 B2 not reflectance?", path, v)
            n = np.nanmean(c[..., 6]); assert -1.001 <= n <= 1.001, ("NDVI out of range", path, n)
    return c


class MonthlyPanel:
    """Latents keyed by (site, sensor, seq); seq 1..21."""

    def __init__(self, lat, idx, roster):
        self.idx, self.roster = idx, roster
        self.flat = {(s, sen, int(p[1:])): np.asarray(v, np.float64).ravel() for (s, sen, p), v in lat.items()}
        self.D = next(iter(self.flat.values())).size
        self.vfrac = {(r.site_id, r.sensor, r.seq): r.valid_pixel_fraction for r in idx.itertuples()}
        self.usable = {(r.site_id, r.sensor, r.seq): bool(r.usable) for r in idx.itertuples()}
        self.treat_of = dict(zip(roster["site_id"], roster["matched_treatment_site_id"]))
        self.group_of = dict(zip(roster["site_id"], roster["group"]))

    @classmethod
    def from_npz(cls, npz_path, idx=None, roster=None):
        roster = load_roster() if roster is None else roster
        idx = pd.read_csv(INDEX_CSV) if idx is None else idx
        return cls(pl.load_latents(npz_path), idx, roster)

    def sites(self, sensor, group=None):
        s = sorted({k[0] for k in self.flat if k[1] == sensor})
        return [x for x in s if group is None or self.group_of[x] == group]

    def treatments(self, sensor):
        return self.sites(sensor, "treatment")

    def controls_of(self, tid, sensor):
        return [s for s in self.sites(sensor, "counterfactual") if self.treat_of[s] == tid]

    def L(self, site, sensor, seq):
        """flattened float64 latent (D,) or None when the composite is not usable."""
        if not self.usable.get((site, sensor, int(seq)), False):
            return None
        return self.flat.get((site, sensor, int(seq)))

    def series(self, site, sensor, seqs):
        """(len(seqs), D) with all-NaN rows where a period is not usable."""
        nan = np.full(self.D, np.nan)
        return np.stack([self.L(site, sensor, q) if self.L(site, sensor, q) is not None else nan
                         for q in seqs])
