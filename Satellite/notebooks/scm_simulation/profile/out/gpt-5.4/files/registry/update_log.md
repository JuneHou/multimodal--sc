# Update Log

## 2026-03-22

- Added `skills/proteomics/protein-language-model-function-analysis-starter` as a concrete `sandbox_verified` bridge between protein embedding extraction and sequence-to-function triage, with a deterministic mock backend, optional ESM-2 or ProtT5 upgrade path, and a Slurm batch template.
- Added four canonical protein-language-model resources to `registry/resources.jsonl` and `registry/resources_dedup.jsonl`: Transformers ESM docs, the foundational ESM paper, the ProtTrans paper, and the ProtT5 model card.
- Added a repo-level smoke target and smoke test for the new protein language model starter, then committed bundled toy FASTA, labels, config, and starter outputs.
- Revalidated the updated surface with `python3 scripts/validate_repository.py`, `python3 -m unittest discover -s skills/proteomics/protein-language-model-function-analysis-starter/tests -p 'test_*.py'`, `python3 -m unittest tests.smoke.test_protein_language_model_function_analysis_starter`, `python3 skills/proteomics/protein-language-model-function-analysis-starter/scripts/run_frontier_starter.py --out scratch/protein-lm/frontier_summary.json`, `make smoke-protein-language-models`, and `python3 scripts/build_site.py`.

- Promoted `skills/genomics/gwas-starter` from a frontier-plan placeholder to a concrete `sandbox_verified` GWAS summary-statistics QC starter that reads tabular sumstats, writes a flagged TSV, and emits downstream interpretation recommendations.
- Added five canonical GWAS resources to `registry/resources.jsonl` and `registry/resources_dedup.jsonl`: the GIANT meta-analysis QC protocol, LDSC, FUMA, GWASLab, and GWAS Catalog summary-statistics documentation.
- Regenerated the bundled GWAS example outputs under `skills/genomics/gwas-starter/assets/` and kept the compatibility wrapper `run_frontier_starter.py` delegating to the new runnable QC path.
- Revalidated the updated surface with `python3 scripts/validate_repository.py`, `python3 -m unittest discover -s skills/genomics/gwas-starter/tests -p 'test_*.py'`, `python3 skills/genomics/gwas-starter/scripts/run_frontier_starter.py --out scratch/gwas/frontier_wrapper_summary.json`, `python3 scripts/build_site.py`, and `python3 -m unittest tests.integration.test_build_site`.

## 2026-03-21

- Re-ran the repository refresh surface and rebuilt the generated site outputs without changing the taxonomy, resource registry, or skill registry, keeping the repository at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the taxonomy is still fully skill-backed.
- Refreshed the shared review outputs with a clean `audit_skill_suite` pass, a `266/266` dry-run smoke-matrix mapping, and a six-case benchmark sample spanning GWAS, proteomics, privacy-preserving analysis, pseudobulk analysis, and interface analysis.
- Re-ran `python3 -m unittest tests.integration.test_build_site tests.integration.test_skill_review_scripts tests.integration.test_sciskill_framework`, which passed `52` tests and kept the framework-facing refresh surface green.
- Wrote a new dated per-run bundle under `reports/refresh-runs/20260321-075940-cycle/` and updated the planning files plus `registry/coverage_report.md` so the on-disk refresh snapshot matches the executed checks.

- Refreshed the shared stage-owned artifacts under `reports/refresh-runs/20260321-050747-cycle/` without changing the taxonomy or registries, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the taxonomy is still fully skill-backed.
- Revalidated the structural surface with `scripts/validate_repository.py` and `scripts/audit_skill_suite.py`, which stayed at `0` hard failures, `266/266` smoke-covered skills, and `253/266` skills with local tests.
- Refreshed `scripts/run_skill_smoke_matrix.py` in `--dry-run` mode, preserving an honest `266/266` target-mapping snapshot without claiming full-suite execution.
- Executed the current `12` focus-leaf starter smokes directly, covering the surfaced proteomics, geospatial, GWAS, and symbolic-discovery starters and writing fresh `scratch/frontier/*.json` artifacts.
- Ran a targeted `12`-case `scripts/benchmark_skill_advantage.py` suite spanning GWAS, proteomics, climate reanalysis, single-cell planning, and multiome integration; maintained skills matched baseline success while improving mean deliverable rate from `0.583` to `1.0`.
- Rebuilt site data through `scripts/build_site.py`, keeping the tree totals unchanged while refreshing `site/framework_runs.json` to surface `16` framework runs.
- Updated the planning files, `registry/coverage_report.md`, `registry/update_log.md`, and `reports/refresh-runs/20260321-050747-cycle/summary.md` so the on-disk refresh snapshot matches the executed checks.

- Re-ran the repository refresh surface after the current framework changes without expanding the registry, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the taxonomy is still fully skill-backed.
- Refreshed `reports/refresh-20260321-cycle/` with a clean `audit_skill_suite` pass, a `266/266` dry-run smoke-matrix mapping, and a representative `fastqc-multiqc-read-qc` benchmark that still favors the maintained wrapper on deliverable rate.
- Re-ran `python3 -m unittest tests.integration.test_sciskill_framework tests.integration.test_skill_review_scripts -v`, which passed `22` selected integration tests and confirmed that the refresh-owned script surface remains consistent with the new campaign-runner and evaluation changes.
- Rebuilt site data through `scripts/build_site.py`, keeping the tree totals unchanged while refreshing `site/framework_runs.json` to surface `12` framework runs, all still counted conservatively under `attention`.
- Kept the carry-forward implementation queue centered on `differential-proteomics`, `ms-proteomics-preprocessing`, `sequence-to-function-modeling`, `protein-embeddings`, `protein-identification-quantification`, `spatial-transcriptomics`, `pseudobulk-analysis`, `rna-velocity`, `gwas`, `fine-mapping`, `privacy-preserving-analysis`, and `regulatory-network-inference`.
- Updated the planning files, `registry/coverage_report.md`, `registry/update_log.md`, and `reports/refresh-20260321-cycle/summary.md` so the on-disk working state matches the current refresh snapshot.

- Re-ran the repository tree-check surface without expanding the registry, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the taxonomy is still fully skill-backed.
- Refreshed the review surfaces under `reports/tree-check-20260321-cycle/` with a clean `audit_skill_suite` pass, a `266/266` dry-run smoke-matrix mapping, a refreshed framework-status snapshot, and a representative `fastqc-multiqc-read-qc` benchmark that still favors the maintained skill surface.
- Recorded that `single-cell-rna-seq-preprocessing` and `single-cell-integration-batch-correction` are already verified, so the next transcriptomics queue should stay on `spatial-transcriptomics`, `pseudobulk-analysis`, and `rna-velocity`.
- Recorded that broad `legal` and `policy` focus terms still collapse onto `privacy-preserving-analysis`, so the next cycle should keep explicit privacy or regulatory terms rather than changing framework ranking during tree-check.
- Narrowed the carry-forward implementation queue to `differential-proteomics`, `ms-proteomics-preprocessing`, `sequence-to-function-modeling`, `protein-embeddings`, `protein-identification-quantification`, `spatial-transcriptomics`, `pseudobulk-analysis`, `rna-velocity`, `gwas`, `fine-mapping`, `privacy-preserving-analysis`, and `regulatory-network-inference`.
- Updated the planning files, `registry/coverage_report.md`, `registry/update_log.md`, and `reports/tree-check-20260321-cycle/summary.md` so the on-disk working state matches the new tree-check snapshot.

- Re-ran the repository refresh surface after the recent framework changes without expanding the registry, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the taxonomy is still fully skill-backed.
- Refreshed `reports/refresh-20260321-cycle/` with a clean `audit_skill_suite` pass, a `266/266` dry-run smoke-matrix mapping, and a representative `fastqc-multiqc-read-qc` benchmark that still favors the maintained wrapper on deliverable rate.
- Re-ran `python3 -m unittest tests.integration.test_sciskill_framework tests.integration.test_skill_review_scripts -v`, which passed `17` integration tests and confirmed that the refresh-owned script surface remains consistent with the framework orchestration changes.
- Rebuilt site data through `scripts/build_site.py`, keeping the tree totals unchanged while refreshing `site/framework_runs.json` to surface `8` framework runs.
- Updated the planning files, `registry/coverage_report.md`, `registry/update_log.md`, and `reports/refresh-20260321-cycle/summary.md` so the on-disk working state matches the current refresh snapshot.

- Re-ran the repository tree-check surface without expanding the registry, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the taxonomy is still fully skill-backed.
- Refreshed the review surfaces under `reports/tree-check-20260321-cycle/` with a clean `audit_skill_suite` pass, a `266/266` dry-run smoke-matrix mapping, a narrowed framework-status snapshot, and a representative `fastqc-multiqc-read-qc` benchmark that still favors the maintained skill surface.
- Recorded that `single-cell-rna-seq-preprocessing` and `single-cell-integration-batch-correction` are already verified, so the next transcriptomics queue should stay on `spatial-transcriptomics`, `pseudobulk-analysis`, and `rna-velocity`.
- Narrowed the carry-forward implementation queue to `differential-proteomics`, `ms-proteomics-preprocessing`, `sequence-to-function-modeling`, `protein-embeddings`, `protein-identification-quantification`, `spatial-transcriptomics`, `pseudobulk-analysis`, `rna-velocity`, `gwas`, `fine-mapping`, `privacy-preserving-analysis`, and `regulatory-network-inference`.
- Updated the planning files, `registry/coverage_report.md`, `registry/update_log.md`, and `reports/tree-check-20260321-cycle/summary.md` so the on-disk working state matches the new tree-check snapshot.

- Re-ran the refresh surface without changing the registries, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Re-ran `scripts/close_frontier_leaves.py`, which correctly stayed a no-op because the full taxonomy is already skill-backed.
- Refreshed the review surfaces under `reports/refresh-20260321-cycle/` with a clean `audit_skill_suite` pass, a `266/266` dry-run smoke-matrix mapping, and a representative `fastqc-multiqc-read-qc` benchmark that still favors the maintained skill on deliverable rate.
- Rebuilt site data through `scripts/build_site.py`, keeping the tree totals unchanged while refreshing `site/framework_runs.json` to surface `6` manifest-backed framework runs plus the current `latest_status` snapshot.
- Recorded that all currently surfaced framework runs remain in `attention` state for real reasons already present in the manifests, so the refresh pass preserved conservative website reporting instead of masking parser or novelty-review issues.
- Updated the planning files, `registry/coverage_report.md`, and `reports/refresh-20260321-cycle/summary.md` so the on-disk working state matches the refresh stage.

## 2026-03-20

- Re-ran the repository tree-check surface without expanding the registry, confirming the repository remains at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Rebuilt site data through `scripts/build_site.py` and refreshed the focused framework-status snapshot under `reports/tree-check-20260320-cycle/framework_status.json`.
- Revalidated the structural review surface with `scripts/audit_skill_suite.py`, which stayed at `0` hard failures, `266/266` smoke-covered skills, and `253/266` skills with local tests.
- Revalidated the smoke-matrix mapping with `scripts/run_skill_smoke_matrix.py --dry-run`, which still resolved `266/266` skill targets with no failures or missing mappings.
- Re-ran the `fastqc-multiqc-read-qc` representative benchmark and confirmed the maintained wrapper still outperforms the no-skill baseline on deliverable rate while preserving command success.
- Recorded that the current post-zero-TODO backlog is now the `146`-skill implemented queue, with `145` covered leaves still lacking any verified skill.
- Refined the next implementation queue toward `differential-proteomics`, `ms-proteomics-preprocessing`, `sequence-to-function-modeling`, `protein-embeddings`, `spatial-transcriptomics`, `pseudobulk-analysis`, `rna-velocity`, `gwas`, `fine-mapping`, `privacy-preserving-analysis`, `regulatory-network-inference`, and `protein-identification-quantification`.
- Noted a focus-term caveat for future cycles: broad terms like `spatial` and `legal` pull lexically valid but lower-priority leaves ahead of spatial omics and privacy/regulatory work, so the next cycle should narrow its focus terms instead of changing framework logic during tree-check.

## 2026-03-15

- Closed all remaining frontier-only leaves.
- Synced `11` previously implemented but unregistered skills into `registry/skills.jsonl`, including `metadata-harmonization-starter`, `synthetic-toy-dataset-generator-starter`, `cooler-hic-matrix-summary-starter`, `bcftools-variant-filtering-starter`, the Semantic Scholar / figure-table / dataset-link / benchmark scientific-knowledge starters, and `matminer-toy-property-prediction-starter`.
- Generated `145` deterministic implemented starter skills so every taxonomy leaf now has at least one mapped skill, lifting the repository to `266` skills while keeping the resource count at `366`.
- Repaired the blocked `bcftools` runtime by provisioning a dedicated self-consistent prefix at `slurm/envs/bcftools` and retargeting the skill wrapper there.
- Added new Phase 43 repository smoke coverage, a generic frontier-starter smoke surface, and site-tree assertions that now require `site/tree.json` to report `0` frontier-only leaves and `0` pure TODO leaves.
- Added a missing Make smoke target for `deepchem-molgraph-featurization`, which restored full skill-suite audit and dry-run smoke-matrix coverage at `266/266`.
- Rebuilt the site for `266` skills and `366` resources, producing `254` covered leaves, `0` frontier-only leaves, and `0` pure TODO leaves.
- Revalidated the repository after the frontier-zero closure wave with `make validate`, `make build-site`, targeted Phase 43 smoke/integration checks, and a full `make test` pass at `123` passing automated tests.
- Added `23` new canonical resources for a Phase 42 frontier-resource deepening pass: `openreview-py-github`, `semantic-scholar-recommendations-api`, `semantic-scholar-graph-api`, `paperswithcode-data-github`, `docling-github`, `grobid-github`, `kallisto-docs`, `deepvariant-github`, `bcftools-filtering-howto`, `nf-core-sarek-usage`, `cooler-docs`, `meme-suite-overview-docs`, `muon-docs`, `openms-docs`, `deepfri-github`, `coffea-hep-docs`, `jax-md-docs`, `agml-docs`, `alphasimr-introduction`, `apsim-docs`, `openmpi-docs`, `dask-jobqueue-docs`, and `apptainer-bind-paths-docs`.
- Deepened single-anchor frontier coverage across scientific knowledge extraction, genomics, epigenomics, proteomics, physics, agriculture, and HPC without changing the skill library.
- Rebuilt the site for `110` skills and `366` resources while keeping the tree at `100` covered leaves, `154` frontier-only leaves, and `0` pure TODO leaves.
- Revalidated the repository after the resource-only expansion with `make validate`, `make build-site`, and a full `make test` pass at `118` passing tests.

## 2026-03-14

- Added four targeted canonical resources for the Phase 40 frontier-closure batch: `openalex-work-object-docs`, `openalex-filter-works-docs`, `networkx-pagerank-docs`, and `openmm-forcefield-api-docs`.
- Added five new verified skills: `openalex-citation-chain-starter`, `reactome-pathway-analysis-starter`, `networkx-network-propagation-starter`, `rdkit-conformer-generation-starter`, and `openmm-forcefield-assignment-starter`.
- Fixed `scripts/build_site.py` so specific three-part alias mappings now override generic second-level leaf matches, which correctly restores `reactome-identifier-enrichment` and `gene-set-tooling-from-bioconductor` coverage in the generated tree.
- Added new command-level smoke targets plus repository smoke and regression coverage for the Phase 40 skills and the alias-resolution fix.
- Rebuilt the site for `101` skills and `340` resources, moving the tree to `91` covered leaves, `163` frontier-only leaves, and `0` pure TODO leaves.
- Added four targeted canonical resources for the next frontier-conversion wave: `chaospy-github`, `inspect-docs`, `autoprotocol-python-github`, and `dash-basic-callbacks`.
- Added four new verified skills: `chaospy-uncertainty-propagation-starter`, `inspect-evaluation-harness-starter`, `autoprotocol-experiment-plan-starter`, and `dash-scientific-dashboard-starter`.
- Reused existing dedicated prefixes by installing `chaospy 4.3.21` into `slurm/envs/numerics`, `autoprotocol 10.3.0` into `slurm/envs/instrumentation`, `inspect_ai 0.3.193` into `slurm/envs/agents`, and `dash 4.0.0` into `slurm/envs/reporting`.
- Restored `autoprotocol` compatibility by pinning `setuptools<81` in `slurm/envs/instrumentation`, which brought back the `pkg_resources` surface required by the current pinned package stack.
- Added four new command-level smoke targets plus repository smoke coverage for the Phase 38 skills, extended the site integration test for the new leaf mappings, and rebuilt the site for `96` skills and `336` resources.
- Hardened the Inspect harness so the current installed `sample.target` shape is handled correctly and repeated scratch reruns do not accumulate stale JSON logs into the summary payload.
- Revalidated the repository after Phase 38 with `make validate`, `make build-site`, `python3 -m unittest tests.smoke.test_phase38_uq_agents_robotics_dashboard_skills -v`, `python3 -m unittest tests.integration.test_build_site -v`, the four new smoke commands, and a full `make test` pass at `106` passing automated tests.

## 2026-03-11

- Bootstrapped the repository structure described in `experiments.md`.
- Added a full seed taxonomy in both Markdown and YAML-compatible JSON form.
- Seeded the registry with 11 canonical resources across literature APIs, workflow engines, HPC, and single-cell tooling.
- Added five initial skills with conservative statuses and provenance.
- Added validation and site-generation scripts plus a static browser entry point.
- Added smoke, regression, integration, and Slurm-oriented tests.
- Expanded the registry to 16 canonical resources and the skill library to 10 skills.
- Added Crossref, PubMed, Ensembl, RCSB PDB, and ClinicalTrials.gov API-backed skills and corresponding smoke tests.
- Added the RCSB entry-summary skill and the supporting RCSB Data API and NCBI developer landing-page resources.
- Installed dedicated `slurm/envs/nextflow-tools` and `slurm/envs/scanpy` prefixes for workflow-engine and Scanpy verification.
- Added verified Nextflow and nf-core skills, promoted Scanpy QC to runtime-verified local execution, and promoted the Slurm template to a real cluster-tested skill.
- Expanded the registry to 25 canonical resources and 13 skills, including new installation and API-reference resources for Nextflow, nf-core, Scanpy, and Slurm.
- Promoted `snakemake-toy-workflow-starter` to a runnable, sandbox-verified local workflow with deterministic output summaries and a dedicated `slurm/envs/snakemake` prefix.
- Added `reactome-event-summary` and `ncbi-gene-search` as new verified API-backed skills, and expanded the registry to 28 canonical resources and 15 skills.
- Hardened the NCBI wrappers with retry/backoff for transient `429` and `5xx` responses, and made the site integration test derive expected counts from `registry/skills.jsonl`.
- Reused the existing resource base to add `multi-source-literature-brief` as a composite literature aggregation skill without adding new external ecosystems.
- Upgraded `nextflow-hello-workflow` from local-only verification to a real Slurm-backed executor path with trace and `sacct` accounting capture.
- Expanded the skill library to 16 skills, raised the automated suite to 22 passing tests, and added new smoke targets for the literature brief and Slurm-backed Nextflow path.
- Expanded the taxonomy to 21 top-level AI-for-science domains with more actionable leaves across proteomics, computational chemistry, workflow languages, materials, and earth/climate branches.
- Added four new verified skills: `reactome-identifiers-enrichment`, `pride-project-search`, `ebi-proteins-entry-summary`, and `chembl-molecule-search`.
- Expanded the resource registry to 44 canonical resources, including Reactome Analysis Service, PRIDE, EBI Proteins, ChEMBL, CWL, OpenWDL, RDKit, OpenMM, Psi4, fgsea, and clusterProfiler references.
- Added repository-level smoke coverage for the new pathway, proteomics, and cheminformatics skills and raised the automated suite to 27 passing tests.
- Hardened the Slurm-backed Nextflow runner by waiting for published outputs and trace rows before writing summaries, which fixed an intermittent empty-output failure exposed by unique scratch directories.
- Expanded the taxonomy and registry into proteomics, cheminformatics, CWL, WDL, computational chemistry, and Bioconductor enrichment resources, bringing the canonical resource count to 44.
- Added `reactome-identifiers-enrichment`, `pride-project-search`, `ebi-proteins-entry-summary`, and `chembl-molecule-search`, bringing the verified skill count to 20.
- Added repository smoke coverage for pathway, proteomics, and cheminformatics APIs, rebuilt the site for 20 skills and 44 resources, and raised the automated suite to 27 passing tests.
- Hardened the Ensembl wrapper after a transient live smoke failure and reconciled concurrent Reactome edits into one canonical verified entry point.
- Installed dedicated `slurm/envs/workflow-languages`, `slurm/envs/chem-tools`, and `slurm/envs/bioconductor` prefixes for CWL/WDL, RDKit/OpenMM, and Bioconductor-backed skills.
- Added six new verified skills: `cwl-commandlinetool-starter`, `wdl-task-starter`, `rdkit-molecular-descriptors`, `openmm-system-minimization`, `fgsea-preranked-enrichment`, and `clusterprofiler-custom-enrichment`.
- Expanded the verified library to 26 skills, added workflow-language and scientific-compute smoke coverage, and rebuilt the site for 26 skills and 44 resources.

## 2026-03-13

- Added three new canonical resources for the next frontier-conversion wave: `rocrate-py-github`, `minimap2-manpage`, and `qcodes-measurement-tutorial`.
- Added four new verified skills: `frictionless-tabular-validation-starter`, `rocrate-metadata-bundle-starter`, `minimap2-read-mapping-starter`, and `qcodes-parameter-sweep-starter`.
- Repaired `slurm/envs/data-tools` by reinstalling `chardet 5.2.0`, which restored Frictionless encoding detection without reintroducing the earlier requests-compatibility warning state.
- Fixed the Frictionless wrapper so absolute input paths validate safely through `basepath` instead of tripping the resource-loader safety check.
- Corrected the `minimap2` toy FASTQ fixture after local testing showed one read was malformed and silently dropped.
- Added new Make smoke targets and repository smoke coverage for the four Phase 37 skills, then rebuilt the site for `92` skills and `332` resources.
- Reworked `site/index.html` to wrap long content, constrain grid children with `min-width: 0`, and degrade cleanly to single-column layouts on smaller screens.
- Updated `experiments.md` so the front-end protocol now explicitly requires responsive layout across phone, tablet, and desktop widths without horizontal page overflow.
- Revalidated the repository with `make validate`, `make build-site`, `python3 -m unittest tests.smoke.test_phase37_data_provenance_alignment_instrumentation_skills -v`, `python3 -m unittest tests.integration.test_build_site -v`, the four new smoke commands, a full `make test` pass at `102` passing tests, and direct `npx playwright screenshot` browser captures at `360x800`, `768x1024`, and `1440x900`.

## 2026-03-12

- Installed dedicated `slurm/envs/psi4` and `slurm/envs/deepchem` prefixes and promoted both stretch toolchains into verified starter skills.
- Added four new verified public-data and ontology skills: `encode-experiment-search`, `biosamples-sample-search`, `metabolights-study-search`, and `quickgo-term-search`.
- Added two new verified local scientific-compute skills: `psi4-single-point-energy` and `deepchem-circular-featurization`.
- Expanded the canonical resource frontier with `Psi4` installation docs plus new imaging, geospatial, and materials references for `scikit-image`, `xarray`, `GeoPandas`, and `matminer`.
- Added new smoke coverage and Make targets for QuickGO, DeepChem, and Psi4, then rebuilt the site for 32 skills and 57 resources.
- Revalidated the full repository with 39 passing automated tests.
- Installed dedicated `slurm/envs/geospatial` and `slurm/envs/statistics` prefixes for GeoPandas and PyMC/ArviZ-backed skills.
- Added two new verified local starters: `geopandas-spatial-join-starter` and `pymc-bayesian-linear-regression-starter`.
- Added the new canonical `arviz-docs` resource and expanded the registry to 59 canonical resources and 38 skills.
- Promoted previously indexed frontier packages into verified starters: `scanpy-ranked-genes-starter`, `skimage-otsu-segmentation-starter`, `xarray-climate-cube-starter`, and `matminer-composition-featurization` now also have generated asset payloads committed under `assets/`.
- Upgraded the site tree generator so `site/tree.json` and `site/index.html` render the full taxonomy with exact leaf-topic skill/resource mappings when available.
- Fixed the GeoPandas `PROJ` / `GDAL` data-path issue inside the skill wrapper and removed deprecation warnings from the `scikit-image` and `xarray` starters.
- Revalidated the frontier smoke surface plus the full repository, reaching 45 passing automated tests.
- Added seven new official cross-cutting resources: LangGraph, DSPy, Matplotlib, Plotly Python, Quarto, Lychee, and pre-commit documentation.
- Added three new verified starter skills: `skill-registry-router-starter`, `matplotlib-publication-plot-starter`, and `registry-link-audit-starter`.
- Fixed wrong repository-root resolution in the router and link-audit starters, tightened the router token normalization so marker-ranking queries land on the ranked-genes skill, and added retry/backoff to the link audit for transient `429` and `5xx` failures.
- Fixed two additional review-surfaced reliability issues: the Matplotlib starter now preserves non-integer residual labels, and the ChEMBL wrapper now retries transient failures and falls back to a curated `imatinib` asset when the live API times out during the canonical smoke path.
- Expanded the registry to 66 canonical resources and 41 skills, and closed the last three empty top-level taxonomy domains.
- Rebuilt the site for 41 skills and 66 resources and revalidated the full repository at 48 passing automated tests.
- Added four new canonical official docs resources: `pydeseq2-docs`, `sourmash-docs`, `dowhy-docs`, and `rasterio-docs`.
- Added six new verified local starters: `skimage-regionprops-feature-extraction`, `rdkit-molecule-standardization`, `arviz-posterior-diagnostics-starter`, `plotly-interactive-report-starter`, `precommit-regression-testing-starter`, and `matminer-property-regression-starter`.
- Installed `plotly` into `slurm/envs/statistics` and created `slurm/envs/maintenance` for deterministic `pre-commit` execution.
- Added new Make smoke targets and repository smoke coverage for the six new frontier starters, then rebuilt the site for 47 skills and 70 resources.
- Reviewed the new frontier skills, switched Plotly HTML generation to self-contained offline assets, and revalidated the repository at 54 passing automated tests.
- Created new repo-managed prefixes for `sourmash`, `DoWhy`, `PyDESeq2`, and Quarto-backed reporting under `slurm/envs/metagenomics`, `slurm/envs/causal`, `slurm/envs/transcriptomics`, and `slurm/envs/reporting`.
- Added five new verified starters: `sourmash-signature-compare-starter`, `rasterio-windowed-raster-preprocessing-starter`, `dowhy-average-treatment-effect-starter`, `pydeseq2-differential-expression-starter`, and `quarto-notebook-report-starter`.
- Promoted four previously frontier-only leaves to covered leaves: `metagenomics`, `remote-sensing-preprocessing`, `causal-inference`, and `notebook-to-report-conversion`.
- Hardened the Quarto runtime path by exporting the conda activation variables inside the skill wrapper so direct repo-local invocation works without shell activation.
- Added four new canonical official resources: `deepagents-docs`, `lifelines-docs`, `interpro-api`, and `networkx-docs`.
- Added four new verified starters: `langgraph-planning-execution-agent-starter`, `lifelines-kaplan-meier-starter`, `interpro-entry-summary`, and `networkx-graph-construction-starter`.
- Repaired `scripts/build_site.py` so legacy topic slugs now resolve onto current taxonomy leaves instead of undercounting site-tree coverage.
- Redesigned `site/index.html` into a richer coverage-first browser with a domain board, frontier panel, stronger taxonomy explorer, and improved filtered skill catalog.
- Verified the new starters through skill-local tests, repository smoke coverage, `make validate`, `make build-site`, and a full `make test` pass at 63 passing tests.
- Installed dedicated `slurm/envs/census` and `slurm/envs/genomics` prefixes for `cellxgene-census 1.17.0` and `pysam 0.23.3`.
- Added four new canonical resources: `scanpy-combat-docs`, `slurm-squeue`, `pysam-docs`, and `samtools-docs`.
- Added four new verified starters: `scanpy-combat-batch-correction-starter`, `cellxgene-census-atlas-query-starter`, `slurm-monitoring-accounting-starter`, and `pysam-sam-bam-summary-starter`.
- Closed the three frontier-only leaves exposed after the Phase 25 tree repair and pushed `site/tree.json` to `48` covered leaves, `0` frontier-only leaves, and `155` TODO leaves.
- Hardened `metabolights-study-search` and `reactome-event-summary` with retry/backoff plus canonical asset fallback so transient upstream failures no longer break repository-wide verification.
- Corrected the Phase 26 Scanpy ComBat smoke test to match the script's `pre_batch_mean_abs_diff` and `post_batch_mean_abs_diff` field names.
- Revalidated the repository at `78` canonical resources, `60` skills, and `67` passing automated tests, then passed explicit smoke targets for Scanpy ComBat, CELLxGENE Census atlas query, pysam SAM/BAM summary, and Slurm monitoring/accounting.
- Added four new canonical resources: `scanpy-score-genes-docs`, `scanpy-dpt-docs`, `reactome-content-service-docs`, and `slurm-job-arrays`.
- Added four new verified starters: `scanpy-cell-type-annotation-starter`, `scanpy-dpt-trajectory-starter`, `reactome-pathway-hierarchy-walk-starter`, and `slurm-job-array-starter`.
- Extended transcriptomics coverage into `cell-type-annotation` and `trajectory-inference`, systems biology into `pathway-traversal-and-hierarchy-walks`, and HPC into `job-arrays`.
- Rebuilt the site to `52` covered leaves, `0` frontier-only leaves, and `151` TODO leaves.
- Revalidated the repository at `82` canonical resources, `64` skills, and `71` passing automated tests, then passed explicit smoke targets for Scanpy annotation, Scanpy DPT, Reactome hierarchy traversal, and Slurm job arrays.
- Added three canonical resources: `rdkit-murcko-scaffold-docs`, `pymatgen-structure-docs`, and `string-api-docs`.
- Added three new verified starters: `rdkit-scaffold-analysis-starter`, `pymatgen-crystal-structure-parsing-starter`, and `string-interaction-partners-starter`.
- Fixed the STRING wrapper so canonical asset fallback only applies to live API failures, not invalid user input.
- Rebuilt the site to `55` covered leaves, `0` frontier-only leaves, and `148` TODO leaves.
- Revalidated the repository at `85` canonical resources, `67` skills, and `74` passing automated tests, then passed explicit smoke targets for RDKit scaffold analysis, pymatgen structure parsing, and STRING interaction partners.
- Expanded the top-level taxonomy from `21` to `27` domains by adding neuroscience, physics, ecology, agriculture, lab automation, and scientific computing branches.
- Refined `experiments.md`, `registry/taxonomy.yaml`, `registry/taxonomy.md`, and `scripts/build_site.py` so the TODO tree now renders the missing AI-for-Science domain families explicitly.
- Rebuilt the site tree to `27` top-level domains with `21` covered domains, `6` empty domains, `55` covered leaves, `0` frontier-only leaves, and `199` TODO leaves.
- Revalidated the affected integration surface with `make validate`, `make build-site`, and `python3 -m unittest tests.integration.test_build_site -v`.
- Added twelve canonical official resources across the new top-level domains: SciPy `solve_ivp`, GBIF APIs, NiBabel, Astropy FITS, PlantCV, Opentrons Protocol API, MNE-Python, fMRIPrep, FEniCSx, and scikit-bio.
- Added six new verified starters: `scipy-ode-simulation-starter`, `gbif-species-occurrence-search-starter`, `nibabel-nifti-summary-starter`, `astropy-fits-image-summary-starter`, `plantcv-plant-phenotyping-starter`, and `opentrons-liquid-handling-protocol-starter`.
- Installed four new repo-managed prefixes under `slurm/envs/`: `neuro`, `astronomy`, `plant-science`, and `automation`.
- Closed all six previously empty top-level domains by promoting them to covered domains, while also seeding five new frontier-only leaves with official docs for future waves.
- Rebuilt the site tree to `27` covered domains, `61` covered leaves, `5` frontier-only leaves, and `188` TODO leaves.
- Revalidated the repository at `97` canonical resources, `73` skills, and `80` passing repository tests, and passed the new explicit smoke targets for SciPy ODE simulation, GBIF occurrence search, NiBabel NIfTI summary, Astropy FITS summary, PlantCV phenotyping, and Opentrons protocol simulation.
- Added five new canonical resources: `mne-rawarray-docs`, `mne-filtering-docs`, `nilearn-signal-clean-docs`, `scikitbio-treenode-docs`, and `fipy-docs`.
- Added five new verified starters: `gbif-dataset-search-starter`, `scikitbio-tree-comparison-starter`, `mne-eeg-preprocessing-starter`, `nilearn-fmri-denoising-starter`, and `fipy-diffusion-pde-starter`.
- Installed or extended three repo-managed prefixes for the frontier conversions: `slurm/envs/neuro`, `slurm/envs/ecology`, and `slurm/envs/numerics`.
- Closed all five frontier-only leaves seeded in Phase 30, moving `site/tree.json` to `66` covered leaves, `0` frontier-only leaves, and `188` TODO leaves.
- Revalidated the repository at `102` canonical resources, `78` skills, and `85` passing repository tests, and passed the new explicit smoke targets for GBIF dataset search, scikit-bio tree comparison, MNE EEG preprocessing, Nilearn fMRI denoising, and FiPy diffusion PDE solving.
- Added `49` new canonical resources without adding new skills, expanding the registry to `225` resources while keeping the skill library at `78`.
- Seeded broad new official coverage across scientific knowledge, data handling, genomics, epigenomics, metabolomics, proteomics, clinical data science, scientific agents, physics, robotics, neuroscience, and scientific computing.
- Reopened the site tree intentionally into a breadth-first frontier state: `66` covered leaves, `87` frontier-only leaves, and `101` pure TODO leaves.
- Revalidated the repository after the resource-only pass with `make validate`, `make build-site`, and a full `make test` run at `85` passing automated tests.
- Updated `experiments.md`, the planning files, the coverage report, and the Phase 32 cycle report to treat resource-only breadth passes as a first-class follow-up once top-level domains are fully covered.
- Added five new verified skills: `fastqc-multiqc-read-qc-starter`, `macs3-peak-calling-starter`, `uniprot-sequence-feature-annotation-starter`, `mne-connectivity-graph-starter`, and `fairlearn-bias-audit-starter`.
- Converted five resource-backed frontier leaves into covered leaves, moving the site tree to `71` covered leaves, `82` frontier-only leaves, and `101` pure TODO leaves.
- Added new command-level smoke targets plus repository smoke coverage for the five Phase 33 starters and extended the site integration test so the generated tree asserts each new leaf mapping.
- Fixed a path-serialization bug in the FastQC and MACS3 wrappers that only surfaced when skill-local tests used system temporary directories outside the repo root.
- Revalidated the repository after Phase 33 with `make validate`, `make build-site`, targeted smoke and integration checks, and a full `make test` pass at `90` passing automated tests.
- Added `101` new canonical resources in a Phase 34 resource-only completion sweep across the remaining pure TODO leaves.
- Expanded the registry to `326` canonical resources while keeping the verified skill library at `83` skills.
- Rebuilt the site tree to `71` covered leaves, `183` frontier-only leaves, and `0` pure TODO leaves.
- Revalidated the repository after the zero-TODO registry expansion with `make validate`, `make build-site`, and a full `make test` pass at `90` passing automated tests.
- Added three targeted canonical docs for the next frontier-conversion wave: `optuna-docs`, `papermill-parameterize-docs`, and `github-actions-workflow-syntax-docs`.
- Installed `optuna 4.7.0` into `slurm/envs/statistics` and `papermill 2.7.0` into `slurm/envs/reporting`.
- Added five new verified starters: `zarr-chunked-array-store-starter`, `papermill-parameterized-notebook-starter`, `github-actions-scientific-ci-starter`, `openmm-langevin-dynamics-starter`, and `optuna-bayesian-optimization-starter`.
- Fixed an overly strict OpenMM test assumption after confirming that the short deterministic Langevin trajectory does not contract monotonically under the chosen seed and temperature.
- Rebuilt the site to `329` canonical resources, `88` skills, `76` covered leaves, `178` frontier-only leaves, and `0` pure TODO leaves.
- Revalidated the repository after the Phase 35 frontier-conversion wave with `make validate`, `make build-site`, targeted smoke and integration checks, and a full `make test` pass at `95` passing automated tests.
- Added three new canonical resources: `pyarrow-cookbook-docs`, `rapidfuzz-docs`, and `datasketch-docs`.
- Added nine new verified starters: `numcodecs-compression-decompression-starter`, `pyarrow-format-conversion-starter`, `scipy-statistical-testing-starter`, `umap-dimensionality-reduction-starter`, `pydoe3-experimental-design-starter`, `ase-geometry-optimization-starter`, `rapidfuzz-skill-deduplication-starter`, `datasketch-resource-deduplication-starter`, and `mkdocs-summary-catalog-starter`.
- Repaired the new-skill local test strategy so prefix-only dependencies are exercised through their dedicated interpreters instead of importing unavailable packages from the base unittest process.
- Rebuilt the site to `343` canonical resources, `110` skills, `100` covered leaves, `154` frontier-only leaves, and `0` pure TODO leaves.
- Revalidated the repository after the Phase 41 local frontier-conversion wave with `make validate`, `make build-site`, targeted smoke and integration checks, and a full `make test` pass at `118` passing automated tests.
- 2026-03-21 refresh pass: reran `scripts/validate_repository.py`, `scripts/close_frontier_leaves.py`, `scripts/audit_skill_suite.py`, `scripts/run_skill_smoke_matrix.py --dry-run`, and a focused `scripts/benchmark_skill_advantage.py` pass for GWAS plus interface-analysis starter cases.
- 2026-03-21 refresh pass: confirmed the repository stayed at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier leaves, and `0` TODO leaves with no new verification-label promotions.
- 2026-03-21 refresh pass: repaired the active framework-cycle report by writing `reports/framework-runs/20260321-042804-protein-spatial-sequence-law-gwas-single-cell-cycle-006/stages/05-refresh/result.json` and the top-level `manifest.json`, so the site can surface the run in `site/framework_runs.json`.
- 2026-03-21 tree-check pass: reran `scripts/validate_repository.py`, `scripts/build_site.py`, `scripts/close_frontier_leaves.py`, `scripts/audit_skill_suite.py --json-out scratch/tree_check_audit.json --markdown-out scratch/tree_check_audit.md`, `scripts/run_skill_smoke_matrix.py --dry-run --json-out scratch/tree_check_smoke_matrix.json --markdown-out scratch/tree_check_smoke_matrix.md`, `scripts/benchmark_skill_advantage.py --case gwas-starter-summary --case gwas-starter-checklist --case ms-proteomics-preprocessing-starter-summary --case ms-proteomics-preprocessing-starter-checklist --json-out scratch/tree_check_benchmark.json --markdown-out scratch/tree_check_benchmark.md`, and `python3 -m unittest tests.integration.test_build_site tests.integration.test_skill_review_scripts`.
- 2026-03-21 tree-check pass: confirmed the repository stayed at `366` deduplicated resources, `266` skills, `254` covered leaves, `0` frontier leaves, and `0` TODO leaves, with `close_frontier_leaves.py` remaining a no-op.
- 2026-03-21 tree-check pass: refreshed `task_plan.md`, `findings.md`, and `progress.md` onto the current audit cycle and narrowed the next-pass queue toward unverified proteomics, spatial omics, GWAS, and privacy/regulatory leaves.
- 2026-10-31 design-skill pass: added two canonical causal-inference resources, `synthetic-control-methods-jasa-2010` and `synth-cran-package`, to anchor an offline synthetic-control workflow around the Abadie, Diamond, and Hainmueller paper plus the maintained Synth R package.
- 2026-10-31 design-skill pass: added the verified local skill `synth-single-treated-unit-scm`, including a portable R entrypoint that auto-detects a single `*_panel.csv`, estimates treated-period counterfactuals with Synth, writes the required `answer.json`, and emits optional diagnostics.
- 2026-10-31 design-skill pass: extended repository coverage with a new causal-inference smoke path and build-site assertion, then rebuilt the site to `377` deduplicated resources and `268` skills and revalidated the affected surfaces with `scripts/validate_repository.py`, `scripts/build_site.py`, `python3 -m unittest discover -s skills/statistical-and-machine-learning-foundations-for-science/synth-single-treated-unit-scm/tests -p 'test_*.py'`, `python3 -m unittest tests.integration.test_build_site -v`, and `python3 -m unittest tests.smoke.test_local_skills.LocalSkillSmokeTests.test_synth_single_treated_unit_scm_runtime -v`.
