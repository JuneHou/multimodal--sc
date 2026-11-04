# Tests

`test_synthetic_control_att_starter.py` runs the full pipeline on a freshly
generated panel and checks ATT recovery, the `answer.json` contract, and the
validator's rejection path.

The tests require the `Rscript` binary and the `Synth` and `jsonlite` R
packages on `PATH` (see the Environment Setup section in `SKILL.md`). They fail
with a clear message if R is unavailable.
