# Design Skill Report: Synthetic Historical Control Satellite Forecast

- Date: 2026-10-13
- Skill path: `skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecast`
- User task: forecast the next time point for each band in a single parcel's local `data.csv` and emit `answer.json`.
- Primary source: `Synthetic Historical Control for Policy Evaluation` by Chen, Yang, and Yang.
- Implementation notes: pure-Python offline runner, lagged multiband residualization, local-linear smoothing, stepwise donor selection on historical blocks, answer-contract validator, deterministic toy example, skill-local tests, and repository smoke coverage.
- Repository state after validation: 376 resources, 268 skills.
- Workflow note: `experiments.md` was referenced by the stage prompt but was not present in this checkout, so no direct update to that file was possible.
