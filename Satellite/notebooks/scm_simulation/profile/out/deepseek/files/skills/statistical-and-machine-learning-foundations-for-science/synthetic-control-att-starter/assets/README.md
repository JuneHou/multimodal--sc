# Assets

- `toy_answer.json`: the `answer.json` produced by fitting the bundled
  `examples/tobacco_panel.csv` for periods 1989-2000 with the Synth estimator.
- `toy_summary.json`: the richer diagnostics JSON (donor weights, predictor
  weights, pre-treatment MSPE, per-period gaps) for the same run.
- Regenerate both with the commands in `SKILL.md` and confirm consistency with
  `scripts/validate_answer.py`.

Do not store R libraries, package caches, or large panels here.
