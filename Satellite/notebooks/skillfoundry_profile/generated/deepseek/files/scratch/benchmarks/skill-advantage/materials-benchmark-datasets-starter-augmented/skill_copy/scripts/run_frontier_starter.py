#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SKILL_ROOT / "assets" / "starter_summary.json")
    args = parser.parse_args()
    metadata = json.loads((SKILL_ROOT / "metadata.yaml").read_text(encoding="utf-8"))
    context = json.loads((SKILL_ROOT / "examples" / "resource_context.json").read_text(encoding="utf-8"))
    summary = {
        "skill_slug": metadata["slug"],
        "skill_name": metadata["name"],
        "status": metadata["status"],
        "leaf_slug": context["leaf_slug"],
        "leaf_name": context["leaf_name"],
        "domain_slug": context["domain_slug"],
        "source_resource_ids": context["source_resource_ids"],
        "starter_steps": context["starter_objectives"],
        "promotion_checklist": [
            "Add a runnable example or toy dataset.",
            "Add a repository-level smoke or integration test.",
            "Promote status to sandbox_verified after checks pass."
        ]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
