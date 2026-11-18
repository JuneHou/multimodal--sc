# HLS SHC / ASCM fitting scripts (2-year treatment history)

Pushed from the private `CALS` project repo (Hurricane Helene NDVI analysis).
Fitting code only -- no figure generation (the source repo's companion
`_figures.py` scripts, and the plotting code that lived inside these two
scripts, are both left out).

```
notebooks/HLS/
├── 04_shc_hls_2y.py        # per-site Synthetic Historical Control (SHC)
└── 02_ascm_hls_2y.R        # per-site (Augmented) Synthetic Control, CWZ conformal inference
```

Input data both scripts read lives at `Satellite/HLS/panels/` (a sibling of
`notebooks/`, not inside it -- see below).

## What input data each method actually needs

**SHC (`04_shc_hls_2y.py`)** has no donor sites at all -- a treated site's own
earlier periods are its "donors". Its only real input is that one treated
site's own NDVI history: a per-site, per-period panel of `(site, date,
phase, ndvi)`, phase being `pre`/`post` relative to 2024-09-27. The
**raw** source for that is the treated-site daily NDVI already in this repo
under `Satellite/HLS/treatment/2y_daily/` (`treatment_daily_2yr.csv` and
`per_site/*.csv`).

**ASCM (`02_ascm_hls_2y.R`)** fits each treated site against a donor pool via
(augmented) synthetic control, so it needs NDVI for the treated site **and**
its donor pool, plus which donors are assigned to which site. The raw source
for the treated side is the same `treatment/2y_daily/` data; the donor side
is the `conus_random10` pool's daily NDVI, already in this repo under
`Satellite/HLS/conus_random10/2y_daily/conus_random10_daily_2yr.csv`.

## Input data: `Satellite/HLS/panels/`

Both scripts read an intermediate **panel** build (one file per cadence:
weekly/biweekly/monthly) rather than the raw daily CSVs directly. That panel
build -- the actual output of a separate `00_build_panels.py` that aggregates
the raw daily data (`treatment/2y_daily/`, `conus_random10/2y_daily/`) into
per-site-per-period rows and assigns each treated site's donor pool -- is now
pushed here too, at `Satellite/HLS/panels/`:

- SHC reads `site_periods_full_<cadence>.csv` (falls back to
  `site_periods_<cadence>.csv` if that doesn't exist)
- ASCM reads `conus_random10_<cadence>_ndvi.csv`, `site_grids_<cadence>.csv`,
  `site_periods_<cadence>.csv`, `site_donors_<cadence>.csv`
- `panel_inventory.csv` is a small metadata index `00_build_panels.py` also writes

`00_build_panels.py` itself is not pushed -- only its output. Point each
script's panel path (`--panels` / `PANELS` in the R script) at
`Satellite/HLS/panels/` and, once the absolute-path issue below is fixed,
both should run against this repo's data as-is.

## Known portability issue

Both files hardcode the source machine's absolute path
(`CALS = Path("/Users/gaoyujuan/REAP Dropbox/Gao yujuan/Virginia Tech/CALS")`
in Python, the R equivalent) rather than resolving relative to the repo.
Update that constant at the top of each file before running them here.
