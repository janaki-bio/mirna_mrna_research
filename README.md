# mRNA-to-miRNA Expression Prediction (TCGA-BRCA)

Practice project: can mRNA expression predict miRNA expression levels?
Uses TCGA-BRCA paired data, Random Forest regression + SHAP.
See project notes below as they develop.

## Data
TCGA-BRCA, processed mRNA (RNA-seq) and miRNA (miRNA-seq) expression
quantification files, GDC Data Portal (portal.gdc.cancer.gov).

## Results

### R² distribution
Random Forest regression was run per miRNA (mRNA features → miRNA
expression) across 100 miRNAs. 13/100 are predictable at R²>0.5, with a
mean R² of 0.31. The distribution looks bimodal, with a dip around
R²=0.45-0.5 separating a low-predictability group from a smaller
high-predictability group. See
[data/r2_distribution.png](data/r2_distribution.png).

### SHAP-derived biological clusters
SHAP analysis of the top-predicted miRNAs surfaces three recognizable
biological clusters among their top predictor genes:
- **Luminal FOXA1/SPDEF cluster** — mir-934, mir-577, mir-18a
- **Immune infiltration** — mir-150
- **Ciliogenesis** — mir-449a

### Shared predictor genes
SPDEF, CFAP99, and AOX2P each appear among the top predictors for 3 of
the 5 miRNAs examined in the SHAP analysis, suggesting a small set of
genes recur across otherwise distinct biological programs. See
[data/shared_genes.png](data/shared_genes.png).

### Novelty caveat
At least two papers from 2025 have already explored mRNA-to-miRNA
expression prediction in TCGA-BRCA along similar lines. This project
largely replicates known biology (luminal markers, immune infiltration,
ciliogenesis signatures) rather than surfacing a new finding, and should
be read as a practice/replication exercise rather than novel research.
