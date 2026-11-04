"""How much the ATT error of one replication moves by luck alone (the analysis slide of the SCM report deck).

For DGP 1 the treated unit's untreated outcome is its shared part delta_t + theta_t'Z_1 + lambda_t'mu_1 plus its own
noise eps_1t. A counterfactual that knows every factor makes an ATT error equal to the mean of eps_1t over periods 41 to
60, which no method can predict. The collaborator's generator (data/simulation/generate_simulation_data.py) is rebuilt
in memory, unchanged, for the shipped seed and for ten new replications under its own --seed rule (DGP 1 uses seed s);
nothing is written to data/.

    python report/luck.py      writes results/luck_replications.csv and prints it
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
GEN = HERE.parents[1] / "data" / "simulation" / "generate_simulation_data.py"


def generator():
    spec = importlib.util.spec_from_file_location("gen", GEN)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def known_factor_att_error(g, seed):
    """The draws of generate_equation1 in its own order; returns the mean of the treated unit's eps over 41..60."""
    rng = np.random.default_rng(seed)
    rng.uniform(-1.0, 1.0, size=(g.N_UNITS, g.N_COVARIATES))
    rng.uniform(-1.0, 1.0, size=(g.N_UNITS, g.N_FACTORS))
    rng.normal(size=g.T)
    rng.normal(size=(g.T, g.N_COVARIATES))
    rng.normal(size=(g.T, g.N_FACTORS))
    eps = rng.normal(size=(g.T, g.N_UNITS))
    return float(eps[g.T0:, 0].mean())


def main():
    g = generator()
    Y0, _ = g.generate_equation1(g.SEED_DGP1)
    truth = pd.read_csv(HERE.parents[1] / "data" / "simulation" / "simulation_data" / "ground_truth.csv")
    assert np.allclose(Y0[:, 0], truth[truth.dgp == "dgp1"].Y0_true.to_numpy()), "shipped replication not rebuilt"
    rows = [{"replication": "shipped", "seed": g.SEED_DGP1, "att_error_known_factors": known_factor_att_error(g, g.SEED_DGP1)}]
    rows += [{"replication": f"seed {s}", "seed": s, "att_error_known_factors": known_factor_att_error(g, s)}
             for s in range(1, 11)]
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "luck_replications.csv", index=False)
    v = df.att_error_known_factors
    print(df.to_string(index=False))
    print(f"\n{len(df)} replications: sd {v.std(ddof=1):.3f}, min {v.min():+.3f}, max {v.max():+.3f}")


if __name__ == "__main__":
    main()
