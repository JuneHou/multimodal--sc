"""Results tables of the prompt-only test, one per noise level and DGP, in results/sigma_<s>/table_<dgp>.csv.

Rows in the README's order: ATT bias, ATT RMSE, counterfactual RMSE, within ±0.25 and ±0.50 of the truth, then the
effort that answered. Columns: README Section 5 as the collaborator printed it, then the three models without donors,
then the three models with donors. An answer's cells come from its counterfactual on that panel, over periods 41 to 60:
the ATT is the mean of observed minus counterfactual (score.counterfactual_att), the ATT bias is ATT minus -3, the ATT
RMSE its absolute value, the within rows yes or no, the counterfactual RMSE against Y0_true. A case without an answer
reads "no answer" in every row (Jun "count all no answer as wrong"). A case whose reply the ARC side closed at 30 minutes
is named in results/sigma_<s>/notes_<dgp>.txt, printed under its table (Jun "1. c", 2026-11-07).

    python prompt_only/pc_score.py
"""
import pandas as pd

import pc_lib as pc

ROWS = pc.README_ROWS + ["answered at effort"]
# one run has no ATT RMSE over replications; the answers' cell is the absolute ATT error, labelled so (Jun "a", 2026-11-07)
LABEL = {"ATT RMSE": "ATT RMSE (README) / absolute ATT error (one run, answers)"}


def answer_column(sigma, dgp, method, model):
    r, effort = pc.chosen(method, sigma, dgp, model)
    if r is None:
        return {row: "no answer" for row in ROWS}
    s = pc.scored(sigma, dgp, r["counterfactual"])
    e = s["att_error"]
    return {"ATT bias": e, "ATT RMSE": abs(e), "counterfactual RMSE": s["cf_rmse"],
            "within ±0.25 of the truth": "yes" if abs(e) <= 0.25 else "no",
            "within ±0.50 of the truth": "yes" if abs(e) <= 0.50 else "no", "answered at effort": effort}


def cut_note(sigma, dgp, method, model):
    """The note for a default-effort reply closed by the ARC side at 30 minutes (no finish reason, no reply text), or
    None."""
    pid = pc.prompt_id(method, sigma, dgp)
    r = pc.last_records(pc.record_path(method, sigma, model, "default")).get(pid)
    if r is None or r.get("ok") or r.get("error"):
        return None
    u = r.get("usage") or {}
    if u.get("finish_reason") is None and not r.get("content") and r["seconds"] >= 1800:
        return (f"{model}, {method.replace('_', ' ')}: no answer; the ARC side closed the reply at {r['seconds']} s, "
                f"before any reply text, after {u.get('completion_tokens'):,} reasoning tokens")
    return None


def main():
    for sigma in pc.SIGMAS:
        out = pc.RESULTS / f"sigma_{sigma}"
        out.mkdir(parents=True, exist_ok=True)
        for dgp in pc.DGPS:
            cols = {"README Section 5": {**pc.readme_column(sigma, dgp), "answered at effort": "no model call"}}
            for method in pc.METHODS:
                for model in pc.MODELS:
                    cols[f"{model}, {method.replace('_', ' ')}"] = answer_column(sigma, dgp, method, model)
            t = pd.DataFrame(cols).reindex(ROWS).rename(index=LABEL)
            t.index.name = "row"
            t.to_csv(out / f"table_{dgp}.csv")
            print(f"\n== sigma {sigma} {dgp}\n" + t.to_string())
            notes = [n for m in pc.METHODS for mo in pc.MODELS if (n := cut_note(sigma, dgp, m, mo))]
            nf = out / f"notes_{dgp}.txt"
            if notes:
                nf.write_text("\n".join(notes) + "\n")
                print("notes:\n" + "\n".join(notes))
            elif nf.exists():
                nf.unlink()


if __name__ == "__main__":
    main()
