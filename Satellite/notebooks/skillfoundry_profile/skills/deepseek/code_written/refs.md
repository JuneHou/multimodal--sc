# References

## Primary method

- Chen, Yi-Ting, Jui-Chung Yang, and Tzu-Ting Yang. *Synthetic Historical Control for Policy
  Evaluation.* SSRN working paper 4995085.
  - Local copy: `paper/ssrn-4995085.pdf` (repository root).
  - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4995085
  - Key ideas used by this skill: a semi-parametric time-series regression, a *treated block*
    matched against a donor pool of *historical blocks* drawn from the same series, and a
    simplex-constrained (non-negative, sum-to-one) matching weight vector reused on the
    historical pseudo-post windows to build the counterfactual / forecast.

## Reference implementations and related software

- `mlsynth` — Greathouse, J. *Machine-learning-style synthetic control estimators*, including
  the SHC helpers (`mlsynth/utils/shc_helpers/orchestration.py`) whose docstrings cite the
  Chen–Yang–Yang historical-control construction. https://github.com/jgreathouse9/mlsynth
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic Control Methods for Comparative
  Case Studies.* JASA. https://doi.org/10.1198/jasa.2009.ap08746
- Abadie, A. (2021). *Using Synthetic Controls: Feasibility, Data Requirements, and
  Methodological Aspects.* Journal of Economic Literature.
  https://doi.org/10.1257/jel.20191450

## Supporting methodology

- Robinson, P. M. (1988). *Root-N-Consistent Semiparametric Regression.* Econometrica.
  https://doi.org/10.2307/1912705
- Cleveland, W. S., & Devlin, S. J. (1988). *Locally Weighted Regression: An Approach to
  Regression Analysis by Local Fitting.* JASA. https://doi.org/10.1080/01621459.1988.10478639
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.),
  for the persistence baseline and decomposition/detrending conventions.
  https://otexts.com/fpp3/

## Domain data conventions

- Sentinel-2 / Landsat surface-reflectance band naming and scaling conventions (NDVI, NIR, Red,
  SWIR), useful when interpreting the band keys and reasonable value ranges in `data.csv`.
  https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi
