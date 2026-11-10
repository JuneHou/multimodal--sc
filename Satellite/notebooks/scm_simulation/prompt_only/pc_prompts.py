"""Write the 12 prompts (two methods, three noise levels, two DGPs) to prompts.jsonl and the review file Jun reads
before any model call, review/prompts.md.

    python prompt_only/pc_prompts.py
"""
import json

import pc_lib as pc


def table_edges(text, header_prefix):
    """First and last data line of the table whose header starts with header_prefix."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_prefix))
    rows = []
    for l in lines[i + 1:]:
        if not l.strip():
            break
        rows.append(l)
    return rows[0], rows[-1]


def main():
    rows = []
    for method in pc.METHODS:
        for sigma in pc.SIGMAS:
            for dgp in pc.DGPS:
                rows.append({"prompt_id": pc.prompt_id(method, sigma, dgp), "method": method, "sigma": sigma,
                             "case": dgp, "prompt": pc.prompt(method, sigma, dgp)})
    pc.PROMPTS.write_text("".join(json.dumps(r) + "\n" for r in rows))

    by = {r["prompt_id"]: r for r in rows}
    md = ["# Prompts of the prompt-only test", "",
          "Every prompt is one user message with no system message. The two prompts of σ = 1.0 DGP 1 are shown in full. "
          "The table lists every prompt with its length and the first and last line of each table it holds.", ""]
    for method in pc.METHODS:
        pid = pc.prompt_id(method, "1.0", "dgp1")
        md += [f"## {pid}", "", "```", by[pid]["prompt"], "```", ""]
    md += ["## All prompts", "", "| prompt_id | characters | target first | target last | donors first | donors last |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        tf, tl = table_edges(r["prompt"], "period,target")
        if r["method"] == "with_donors":
            df, dl = table_edges(r["prompt"], "period,donor_01")
            df, dl = df[:40] + " …", dl[:40] + " …"
        else:
            df = dl = "none"
        md.append(f"| {r['prompt_id']} | {len(r['prompt'])} | `{tf}` | `{tl}` | `{df}` | `{dl}` |")
    (pc.HERE / "review").mkdir(exist_ok=True)
    (pc.HERE / "review" / "prompts.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} prompts written to {pc.PROMPTS} and review/prompts.md")


if __name__ == "__main__":
    main()
