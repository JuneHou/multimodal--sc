"""lb_one_thread — import first in every entry script of the daily simple skills, before numpy is loaded.

Sets the thread variables the skill sandbox sets (lb_skill_loop.Cell.do_shell): one thread for OpenBLAS, MKL and
OpenMP. Her notebook 14 fit depends on the BLAS thread count (2026-10-15: Sentinel-2 treatment_0004 at her settings
differs by 3.7e-4 between the default thread count and one thread), so every run of her code, the parity reference,
the gates and the scoring run with the thread count of the sandbox the models use.
"""
import os
import sys

VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
if "numpy" in sys.modules and any(os.environ.get(v) != "1" for v in VARS):
    raise RuntimeError("lb_one_thread must be imported before numpy")
for _v in VARS:
    os.environ[_v] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
