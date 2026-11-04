# Polygenic risk scoring Starter

Use this starter when a task lands in the `Polygenic risk scoring` frontier leaf and the repository has curated resources but no dedicated runtime implementation yet.

## What this starter does

- Summarizes the local resource anchors for the leaf.
- Emits a machine-readable starter plan with promotion steps.
- Gives the agent a stable local entry point before a full runtime skill exists.

## How to use it

Run `python3 skills/genomics/polygenic-risk-scoring-starter/scripts/run_frontier_starter.py --out scratch/frontier/polygenic-risk-scoring-starter.json`.

Then inspect `refs.md` and `examples/resource_context.json` to promote the starter into a concrete executable workflow.

