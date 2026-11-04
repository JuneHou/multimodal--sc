You are operating inside the SciSkillUniverse repository.
Follow the workflow contract in `experiments.md`.
This mode is for a user-specified task: search resources first when the request needs additional materials, then implement and test the resulting skill.

STAGE: design_skill
MODE: design_skill

USER_TASK_PROMPT:
Design a skill for estimating the effect of a treatment on one treated unit from panel data on that unit and untreated donor units with the synthetic control method of Abadie, Diamond and Hainmueller. The paper, Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program, is at `paper/Synthetic Control Methods for Comparative Case Studies- Estimating the Effect of California’s Tobacco Control Program.pdf`. The paper's own software, the Synth R package version 1.1-10, is at `code/Synth/`, and R 4.3.3 with Synth and jsonlite installed is available to the agent. The agent that uses the skill works in a directory holding one panel file `*_panel.csv` with the columns `unit_id`, `period`, `treatment_group`, `Z1`, `Z2` and `Y_observed`, one row per unit and period, for the treated unit `treated` and the untreated donor units. It estimates the treated unit's outcome without treatment in each treated period the task names and the average effect over those periods, and writes `answer.json` as {"counterfactual": {"<period>": <number>, ...}, "att": <number>, "rationale": "<text>"}. The agent has a bash shell and no network access.

CURRENT_REPOSITORY_SUMMARY_JSON:
{
  "covered_domain_count": 27,
  "covered_leaf_count": 254,
  "empty_domain_count": 0,
  "focus_leaves": [
    {
      "coverage_status": "covered",
      "domain_name": "Statistical and Machine Learning Foundations for Science",
      "domain_slug": "statistical-and-machine-learning-foundations-for-science",
      "leaf_name": "Causal inference",
      "resource_count": 1,
      "resource_ids": [
        "dowhy-docs"
      ],
      "skill_count": 1,
      "skill_slugs": [
        "dowhy-average-treatment-effect-starter"
      ],
      "taxonomy_key": "statistical_and_machine_learning_foundations_for_science",
      "topic_slug": "causal-inference",
      "verified_skill_count": 1
    }
  ],
  "frontier_leaf_count": 0,
  "resource_count": 375,
  "skill_count": 267,
  "taxonomy_domain_count": 27,
  "todo_leaf_count": 0
}

FOCUS_LEAVES_JSON:
[
  {
    "coverage_status": "covered",
    "domain_name": "Statistical and Machine Learning Foundations for Science",
    "domain_slug": "statistical-and-machine-learning-foundations-for-science",
    "leaf_name": "Causal inference",
    "resource_count": 1,
    "resource_ids": [
      "dowhy-docs"
    ],
    "skill_count": 1,
    "skill_slugs": [
      "dowhy-average-treatment-effect-starter"
    ],
    "taxonomy_key": "statistical_and_machine_learning_foundations_for_science",
    "topic_slug": "causal-inference",
    "verified_skill_count": 1
  }
]

REQUIRED_BEHAVIOR:
- Search for canonical papers, repositories, docs, notebooks, workflows, or package references if the task needs them.
- Build or refine a single concrete skill path for the user task.
- Add or update repository-level tests and, if relevant, a Slurm exercise path.
- Update registries, reports, and `experiments.md` if repository state changes.

SKILL_QUALITY_STANDARDS:
Every skill created or significantly updated in this stage MUST satisfy ALL of the following requirements.
These are not optional guidelines—they are hard requirements for skill acceptance.

1. ACTIONABLE RUNBOOK — SKILL.md must be an operator-grade execution playbook, not a high-level summary.
   - The Procedure section must contain numbered, concrete steps that an agent can execute directly.
   - Each step must specify:
       * The exact command or Python/shell snippet to run (not a placeholder like "run the script").
       * What intermediate file or output is produced and its key fields or shape.
       * How to inspect and validate the result at this checkpoint before proceeding to the next step.
       * The decision logic to apply based on the observed result ("if X then Y, else Z").
   - Include at least one realistic, non-trivial usage example that produces scientifically meaningful
     output—not just a smoke-test or toy placeholder.
   - If the skill wraps an external API, document: endpoint URL, authentication method,
     rate limits, and a concrete retry strategy with example code.
   - If the skill requires searching for data or references, provide explicit search guidance:
       * Which databases, search engines, or repositories to query (e.g., PubMed, OpenAlex, PyPI,
         GitHub, CELLxGENE, UniProt, NCBI, etc.).
       * How to formulate effective queries with concrete example query strings.
       * How to evaluate, rank, and select search results (what metadata to inspect, what criteria
         to prefer, when to stop searching).
       * How to download and verify the selected resource before using it.
   - If the skill uses specific tools or libraries, show the concrete API usage:
       * Import statements and initialization code.
       * Key function calls with realistic argument values, not just `func(args)` placeholders.
       * How to interpret the return values and handle edge cases (empty results, pagination, etc.).
   - Separate the real-run workflow clearly from any deterministic CI/smoke-test path.

2. ENVIRONMENT SETUP — dedicate a named section in SKILL.md titled "Environment Setup" (or "Requirements").
   - List every required package with its installation command:
       pip install <package>==<version>   # or conda equivalent
   - State the minimum runtime version (Python ≥ X.Y, R ≥ X.Y, etc.) and any known version conflicts.
   - Describe any environment variables or config files that must be set before running.
   - Provide a "pre-flight check" the agent can run to verify the environment is ready, e.g.:
       python3 -c "import scanpy; print(scanpy.__version__)"
   - When a required package has version constraints (e.g., Python < 3.13), provide a working
     environment-creation recipe:
       conda create -n skill-env python=3.11 && conda activate skill-env && pip install ...
   - Include a browser-based or offline fallback path when the primary package cannot be installed.
   - If the skill depends on external CLI tools (e.g., bcftools, GATK, samtools, nextflow),
     document how to install and verify them:
       which bcftools && bcftools --version

3. ERROR HANDLING — dedicate a named section in SKILL.md titled "Troubleshooting" or "Error Handling".
   - Cover the 3–5 most common failure modes (network timeout, missing package, bad input format,
     API auth failure, disk space, unexpected tool version, etc.).
   - For each failure mode state:
       * The recognizable symptom: exact error message, exit code, or observable behavior.
       * A concrete, step-by-step remediation procedure.
       * An alternate execution path or manual fallback when the primary tool is unavailable.
   - Scripts themselves must handle errors gracefully: catch exceptions, emit a descriptive message
     to stderr, and exit with a non-zero code—never silent failures or bare `except: pass`.

4. PATH PORTABILITY — no hardcoded absolute or repo-anchored paths anywhere in SKILL.md or scripts.
   - SKILL.md procedures and examples MUST NOT contain absolute paths (e.g., /Users/... /home/...).
   - SKILL.md MUST NOT embed paths anchored to this specific repository layout.
       BAD:  python3 skills/genomics/bcftools-variant-filtering-starter/scripts/run_bcftools.py
       BAD:  --input skills/genomics/bcftools-variant-filtering-starter/examples/toy.vcf
       GOOD: python3 scripts/run_bcftools.py              # documented as: run from the skill directory
       GOOD: python3 run_bcftools.py                      # or: add the scripts/ dir to PATH
       GOOD: python3 <skill_dir>/scripts/run_bcftools.py  # generic placeholder form
   - All scripts must accept every input/output path as a command-line argument
     (--input, --out, --workspace, etc.); never hardcode file paths inside script source code.
   - Acceptable exception: test_commands in metadata.yaml may use the canonical
     `skills/<domain>/<slug>/scripts/...` form relative to the repository root, since the
     framework always runs them from the repo root. This exception does NOT extend to SKILL.md.

5. SCRIPT DESIGN — every helper script in the skill's scripts/ directory must:
   - Be self-contained: import only standard-library or well-known third-party packages
     (declared in Environment Setup); do not import from other skills or internal framework modules.
   - Accept ALL file paths (input data, output files, working directory, config) via command-line
     arguments with argparse or equivalent. Provide sensible defaults only when portable
     (e.g., default output to current directory, not a hardcoded repo path).
   - Include a --help flag that documents every argument, its type, and a short usage example.
   - Print a JSON summary on stdout (or write to --out) so downstream automation can parse results.
   - Exit with code 0 on success, non-zero on failure; write diagnostics to stderr.
   - Handle common edge cases: empty input, missing network, oversized files, unexpected types.
   - Use an `if __name__ == "__main__":` guard so the script can also be imported as a module.

6. COMPLETENESS — verify SKILL.md contains ALL of these named sections before finishing:
   [ ] Purpose — 1–2 sentence functional description and primary use case
   [ ] When to use / When not to use — precise boundary conditions and routing to adjacent skills
   [ ] Environment Setup — packages, versions, pre-flight check, environment recipe if needed
   [ ] Inputs — each input with: name, type/format, whether required or optional, example value
   [ ] Outputs — each output with: filename pattern, format, key fields, and size/shape guidance
   [ ] Procedure — numbered steps with exact commands, expected artifacts, and decision checkpoints
   [ ] Troubleshooting / Error Handling — top failure modes with symptoms and remediation
   [ ] Validation — how to confirm a successful end-to-end run, including expected output shape
   [ ] Related Skills — cross-references to adjacent or prerequisite skills in the repository


ADDITIONAL_CONTEXT:
None.

RESPONSE_CONTRACT:
- Return only a JSON object that matches the provided schema.
