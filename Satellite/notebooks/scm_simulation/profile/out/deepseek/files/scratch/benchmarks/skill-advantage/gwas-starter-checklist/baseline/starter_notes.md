# GWAS starter notes

Leaf: GWAS
Leaf slug: gwas
Domain slug: genomics
Source resource ids: plink2-docs, gwas-meta-analysis-qc-protocol, ldsc-repo, fuma-docs, gwaslab-docs, gwas-catalog-summary-statistics-docs

Starter objectives:
- Normalize common summary-statistics headers into one local schema.
- Flag malformed, duplicate, low-information, and ambiguous variants before downstream analysis.
- Emit a flagged TSV and JSON summary with lead signals and interpretation recommendations.
- Route cleaned outputs toward PLINK2, LDSC, FUMA, and GWAS-SSF compatible follow-up.

Promotion note: add a runnable example, add a repository smoke test, then promote after verification.
Promotion status target: sandbox_verified after checks pass.
