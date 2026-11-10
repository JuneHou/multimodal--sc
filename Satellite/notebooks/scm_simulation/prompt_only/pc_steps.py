"""Step analysis a, what the numbers match: the distance of every answer's counterfactual to every rule of its method,
in results/steps_rules.csv (plan decisions "a, what the numbers match" and "closeness to a rule").

Distance = root mean squared difference over periods 41 to 60 between the answer's counterfactual and the rule's.
Method 1 answers against three rules, method 2 answers against five (pc_lib.rules). The smallest distance names the
nearest rule.

    python prompt_only/pc_steps.py
"""
import pandas as pd

import pc_lib as pc

RULES = list(pc.RULE_NAMES)


def main():
    rows = []
    for method in pc.METHODS:
        for sigma in pc.SIGMAS:
            for dgp in pc.DGPS:
                rules = pc.rules(method, sigma, dgp)
                for model in pc.MODELS:
                    r, effort = pc.chosen(method, sigma, dgp, model)
                    if r is None:
                        continue
                    d = {k: pc.rmse(r["counterfactual"], v) for k, v in rules.items()}
                    near = min(d, key=d.get)
                    rows.append({"method": method, "sigma": sigma, "dgp": dgp, "model": model, "effort": effort,
                                 **{f"to_{k}": d.get(k) for k in RULES}, "nearest": near, "nearest_distance": d[near]})
    t = pd.DataFrame(rows)
    t.to_csv(pc.RESULTS / "steps_rules.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
