# Tests

- Skill-local runtime: `python3 scripts/run_shc_satellite_forecast.py --input examples/toy_satellite_history.csv --out scratch/shc-answer.json --details-out scratch/shc-details.json`
- Answer contract: `python3 scripts/validate_answer.py --input examples/toy_satellite_history.csv --answer scratch/shc-answer.json`
- Unit tests: `python3 -m unittest discover -s tests -p 'test_*.py'`
