# Results tables

Unless stated otherwise, values are cross-validated means over 15 test folds on 1,053 TCGA-BRCA tumours, with folds that hold out whole sequencing plates (5 folds balanced by tumour count, 3 repeats). Columns ending in `_patient` use patient-grouped folds instead.

| File | Contents |
|---|---|
| `mirna_attribution.csv` | One row per miRNA precursor (337): R² of every model, ceiling, attribution category, robustness, flags, and patient-grouped comparisons |
| `label_uncertainty.csv` | Per-miRNA uncertainty of the difference that set its category (Nadeau–Bengio corrected resampled t, 95% interval) and the borderline flag |
| `sensitivity_settings.csv` | Category counts under alternative settings (10 / 50 components, ±250 kb / ±1 Mb locus window, components without host and local genes, thresholds 0.7 / 0.9) for plate-grouped, patient-grouped and plate-adjusted analyses |
| `cv_design_comparison.csv` | Mean R² of the full and 20-component models with folds grouped by patient, tissue source site and sequencing plate (per fold and pooled) |
| `pc_programmes_technical.csv` | The ten leading principal components: marker-programme correlations, PAM50 and purity association, and association with plate, site, library size and miRNA read depth |
| `plate_confounding.csv` | Cramér's V between mRNA plate, miRNA plate and tissue source site |
| `low_expression_mirnas.csv` | mir-934, mir-577, mir-449a and mir-449b, which fall below the expression threshold, evaluated with the same models (patient-grouped folds) |
| `model_comparison_top100.csv` | Random forest versus ridge regression for the 100 most variable miRNAs (all 1,087 tumours, patient-grouped folds) |
| `summary.json` | Cohort summary, category counts by design, median shares, robustness counts, clinical baselines and marker-gene programme definitions |

## Key columns in `mirna_attribution.csv`

| Column | Meaning |
|---|---|
| `r2_full`, `r2_full_sd` | Ridge regression on all 27,671 genes: mean and SD of test-fold R² |
| `r2_host` | Linear regression on the host gene(s); empty if no host gene or the host gene is not measurable (`host_status`) |
| `r2_state_k`, `r2_host_state_k` | Ridge regression on the leading k principal components (k = 5, 10, 20, 50), without and with the host gene(s) |
| `r2_locus_own`, `r2_locus_paralog`, `r2_locus_any` | Ridge regression on genes within ±500 kb of the miRNA, of its paralogs, or both |
| `r2_locus_state20`, `r2_host_locus_state20` | Locus genes + 20 components, without and with the host gene(s) |
| `r2_pam50_cpe`, `r2_pam50_cpe_lump` | PAM50 subtype + CPE purity; the same plus LUMP purity with a missingness indicator |
| `ceiling_model`, `r2_best` | The model with the highest mean R² (full model or a reference model) and its R² (the ceiling) |
| `r2_best_reference`, `residual_gap` | Highest R² among the reference models, and `r2_full` minus that value |
| `category`, `category_detail` | Attribution (see below); for tumour state, the most correlated marker programme and the sign of the correlation |
| `borderline`, `robust`, `robust_incl_components` | Uncertainty and robustness of the category (see the report, Methods) |
| `category_k10`, `category_k50`, `category_250kb`, `category_1Mb`, `category_pc_excl` | Category under alternative settings |
| `flags` | X, cross-mapped reads in > 50% of samples; P, paralog group; H, host gene not measurable |
| `r2_full_pooled_patient`, `_site`, `_plate` | Pooled out-of-fold R² of the full model under the three fold designs |
| `category_patient`, `category_plate_adjusted` | Category with patient-grouped folds, without and with plate adjustment of both data types |

Categories: poorly predicted if `r2_best` < 0.3; otherwise the first of host → tumour state → host + state → locus (own or paralog) whose reference model reaches at least 80% of `r2_best`; residual if `residual_gap` ≥ 0.10; partly explained otherwise. Categories describe predictive sufficiency, not causes. Use `robust` or `robust_incl_components` to restrict to reliable labels.
