# HLS SHC / ASCM scripts (2-year treatment history)

Pushed from the private `CALS` project repo (Hurricane Helene NDVI analysis),
commit date 2026-11-18. Two method scripts plus their figure-generation
companions, each pair in its own subfolder mirroring the source repo's layout:

```
HLS/
├── hls_2y_shc/
│   ├── 04_shc_hls_2y.py        # per-site Synthetic Historical Control (SHC)
│   └── _figures.py             # figures built from 04's own output tables
└── hls_2y_random_control/
    ├── 02_ascm_hls_2y.R        # per-site (Augmented) Synthetic Control, CWZ conformal inference
    └── _figures.py             # figures + effect maps; called by 02 at the end of its run
```

## What each script needs to run

**`hls_2y_shc/04_shc_hls_2y.py`**
- `<panels>/site_periods_full_<cadence>.csv` (weekly/biweekly/monthly) -- the only
  real input; SHC has no donor panel.

**`hls_2y_shc/_figures.py`**
- `04`'s own output tables (`site_paths.csv`, `site_summary.csv`, `site_weights.csv`)
- `datasets/flooding/HLS/treatment/site_characteristics/treatment_site_landcover.csv`

**`hls_2y_random_control/02_ascm_hls_2y.R`**
- `<panels>/conus_random10_<cadence>_ndvi.csv`
- `<panels>/site_grids_<cadence>.csv`
- `<panels>/site_periods_<cadence>.csv`
- `<panels>/site_donors_<cadence>.csv`
- calls `_figures.py --method ascm --cadence <cadence>` itself at the end of each run

**`hls_2y_random_control/_figures.py`**
- `02`'s own output tables (`ROOT/<method>/<cadence>/tables/...`)
- `datasets/flooding/HLS/treatment/site_characteristics/treatment_site_landcover.csv`
  and `treatment_landcover_average.csv`
- `datasets/flooding/HLS/treatment/treatment_sites.csv`
- `datasets/flooding/HLS/conus_random10/conus_random10_control_sites.csv`
- `datasets/flooding/HLS/conus_random10/conus_random10_matched_pairs.csv`
- `datasets/flooding/HLS/_work/site_selection/...` (donor/candidate site-selection tables)

## Checked against `Satellite/HLS/` in this repo (commit `d3d49973`)

**Already present here**, same filenames:
- `Satellite/HLS/treatment/treatment_sites.csv`
- `Satellite/HLS/treatment/site_characteristics/treatment_site_landcover.csv`
- `Satellite/HLS/treatment/site_characteristics/treatment_landcover_average.csv`
- `Satellite/HLS/conus_random10/conus_random10_control_sites.csv`
- `Satellite/HLS/conus_random10/conus_random10_matched_pairs.csv`

**NOT present here** -- these are the actual primary inputs both method scripts read,
and they are *derived* files (built by a separate `00_build_panels.py` panel-builder
from the raw composites above, not pushed in this commit):
- `site_periods_full_<cadence>.csv`, `site_periods_<cadence>.csv`
- `conus_random10_<cadence>_ndvi.csv`
- `site_grids_<cadence>.csv`, `site_donors_<cadence>.csv`
- the `datasets/flooding/HLS/_work/site_selection/` donor/candidate tables

So these two scripts will not run against this repo's current `Satellite/HLS/` data
as-is -- the panel-building step (and the `_work/site_selection` tables) still needs
to be run or pushed separately.

## Known portability issue

Every file here hardcodes the source machine's absolute path
(`CALS = Path("/Users/gaoyujuan/REAP Dropbox/Gao yujuan/Virginia Tech/CALS")` in
Python, the equivalent in R) rather than resolving relative to the repo. Update that
constant at the top of each file before running them here.
