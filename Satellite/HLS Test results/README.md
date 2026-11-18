# HLS SHC / ASCM result tables (2-year treatment history)

Output of `Satellite/notebooks/HLS/04_shc_hls_2y.py` (SHC) and
`Satellite/notebooks/HLS/02_ascm_hls_2y.R` (ASCM), run against the panels at
`Satellite/HLS/panels/`. Tables only -- no figures or maps (neither script
produces them; see `notebooks/HLS/README.md`).

```
HLS Test results/
├── shc/{weekly,biweekly,monthly}/tables/
│   ├── site_summary.csv          -- one row per site: ATT, m, pre_rmse_raw/latent,
│   │                                 pre_mse_window, placebo_rank_p
│   ├── site_paths.csv            -- full observed/counterfactual/effect series per site
│   ├── site_weights.csv          -- per-site donor-block weights
│   ├── site_mscan.csv            -- matching-length (m) scan, latent_mse per candidate m
│   ├── bandwidth_loocv.csv       -- leave-one-out MSE per candidate smoothing bandwidth
│   ├── pretreatment_latent.csv   -- smoothed pre-period latent series
│   ├── season_match_feasibility.csv
│   ├── historical_placebo_comparison.csv / historical_placebo_audit.csv
│   ├── att_by_period.csv / att_summary.csv
│   ├── site_eligibility.csv / failed_sites.csv
│   └── run_config.json           -- exact CLI args + input/script sha256 for that run
└── ascm/{weekly,biweekly,monthly}/tables/        (all prefixed conus_random10_)
    ├── site_summary.csv          -- one row per site: SCM/ASCM ATT, RMSE, conformal interval/p
    ├── site_effects.csv          -- full observed/counterfactual/effect series, both estimators
    ├── site_donor_weights.csv / donor_weight_mean.csv
    ├── site_inference.csv
    ├── att_by_period.csv / att_inference.csv
    └── placebo_summary.csv
```

The headline result in both: a per-site ATT, the estimated NDVI effect of
Hurricane Helene -- SHC relative to that site's own pre-event history, ASCM
relative to a weighted donor pool of unaffected sites.
