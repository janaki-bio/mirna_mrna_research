# Analysis scripts

The scripts run in numeric order (`run_all.sh`). Each reads the outputs of the preceding steps from `analysis/results/` and `analysis/cache/` and writes its own; `export_results.py` runs last and writes the public tables in `results/` from `analysis/results/`.

| Script | What it does | Main outputs | Report section |
|---|---|---|---|
| `01_export_pam50_purity.R` | Exports PAM50 calls and CPE / LUMP purity from TCGAbiolinks | `external/tcgabiolinks/*.csv` | Methods: Cohort |
| `02_build_matrices.py` | Matches primary tumours with mRNA and miRNA files; builds the TPM and RPM matrices from `data/raw` | `cache/mrna_matrix.pkl`, `cache/mirna_matrix.pkl`, `results/00_data_checks.csv` | Methods: Data |
| `03_holdout_rf_r2.py` | Random forest on an 80/20 split for the 100 most variable miRNAs | `data/r2_results.csv` | Supplement: model choice |
| `04_select_analysis_set.py` | Drops FFPE, keeps one tumour per patient with PAM50 and CPE (1,053) | `results/audit_samples.csv`, `audit_sample_steps.json` | Methods: Cohort; Results: Study design and cohort |
| `05_fetch_plate_barcodes.py` | Fetches aliquot barcodes (plates) from the GDC API; the output is included in the repository, so `run_all.sh` skips this step unless `FETCH_GDC_API=1` | `external/gdc/file_aliquots.csv` | Methods: Batches, sites and cross-validation designs |
| `06_annotate_hosts.py` | Host genes, host type, ±50 kb neighbours, paralog groups and clusters from GENCODE v36 | `results/audit_hosts.csv`, `audit_mir_mapping.csv` | Methods: Host genes, loci and paralogs |
| `07_annotate_low_expression_hosts.py` | Same annotation for mir-934, mir-577, mir-449a and mir-449b | `results/audit_hosts_low_expression.csv` | Supplement: low-expression miRNAs |
| `08_cross_mapped_reads.py` | Cross-mapped read flags in the raw miRNA files | `results/10_e4_cross_mapped_all.csv` | Methods: Host genes, loci and paralogs (flag X) |
| `09_baseline_models.py` | Patient-grouped CV: full ridge model, host gene, PAM50 and purity | `results/audit_per_mirna.csv` | Methods: Models |
| `10_state_models.py` | Tumour-state models (k = 5–50 components), programme scores, low-expression miRNAs, site-grouped CV | `results/audit_per_mirna_v2.csv`, `audit_low_expression.csv`, `audit_pc_programmes.csv`, `audit_meta_v2.json` | Methods: Models, Programme scores; Results: leading components |
| `11_locus_models.py` | Own-locus and paralog-locus models | `results/audit_per_mirna_v3.csv` | Methods: Models (genomic locus) |
| `12_robustness_fits.py` | Alternative k, locus windows and components without host / local genes; fold uncertainty | `results/audit_sensitivity_settings.csv`, `audit_uncertainty.csv` | Methods: Robustness and uncertainty |
| `13_reference_model_fits.py` | Per-fold fits of every reference model for all 337 targets | `cache/audit_r2_fold_records.pkl` | Methods: Models, Attribution |
| `14_attribution_labels.py` | Attribution with the per-fold-maximum ceiling; host-model checks | `results/audit_per_mirna_r2.csv` (first version) | Methods: Attribution |
| `15_technical_pcs.py` | Top 10 components vs tissue source site, PAM50, purity, library size | `results/audit_technical_pcs.csv` | Methods: Programme scores and technical variables |
| `16_clinical_site_cv.py` | Clinical baselines; pooled R² under patient and site folds | `results/audit_clinical_r2.csv`, `audit_site_pooled.csv` | Results: Shared sequencing batches inflate cross-validation |
| `17_plate_vs_components.py` | Plate η² for the top 10 components | `results/audit_technical_pcs.csv` (plate columns) | Results: leading components |
| `18_ceiling_model.py` | Ceiling model by mean R²; final labels and uncertainty for patient folds | `results/audit_per_mirna_r2.csv`, `audit_uncertainty_r2.csv` | Methods: Attribution |
| `19_site_balanced_cv.py` | Site folds balanced by tumour count | `results/audit_site_pooled.csv` | Methods: Batches, sites and cross-validation designs |
| `20_plate_cv_checks.py` | Plate-grouped CV, Cramér's V, plate-adjusted components | `results/audit_plate_cv.csv`, `audit_plate_confounding.csv` | Results: Shared sequencing batches; Methods: Plate adjustment |
| `21_plate_cv_fits.py` | Per-fold fits of every model with plate-grouped folds (`--design plate`) and with plate-adjusted patient folds (`--design adj`) | `cache/audit_plate_fold_records.pkl`, `audit_plateadj_sym_fold_records.pkl` | Methods: Batches…; Plate adjustment |
| `22_plate_labels.py` | Main labels (plate-grouped CV), robustness, uncertainty, clinical baselines | `results/audit_per_mirna_plate.csv`, `audit_robustness_plate.csv`, `audit_uncertainty_plate.csv`, `audit_clinical_plate.csv`, `audit_plate_adjusted_sym.csv` | Results: Attribution; Robustness and uncertainty |
| `23_plate_meta.py` | Counts and settings of the plate-grouped analysis | `results/audit_meta_plate.json` | Results: Attribution |
| `24_repeated_cv_rf_ridge.py` | Random forest vs ridge, 100 miRNAs, 5 folds × 3 repeats | `results/06_f4_cv_folds.csv` | Supplement: model choice |
| `25_summarise_rf_ridge.py` | Per-miRNA summary of 24 | `results/07_f4_per_mirna.csv` | Supplement: model choice (Fig. S1A) |
| `26_shap_holdout.py` | Holdout random forest and TreeSHAP | `results/02_shap_top10.csv` | Supplement: model choice |
| `27_shap_stability.py` | Fold-to-fold overlap of the top-10 SHAP genes | `results/13_e8_shap_stability.csv` | Supplement: model choice (Fig. S1B) |
| `export_results.py` | Writes the public tables from `analysis/results/` | `../results/*.csv`, `summary.json` | all tables |
| `common.py`, `audit_common.py` | Paths, matrix loading, filters, ridge grid and seeds | | |

Report sections are those of `report/mirna_prediction_attribution.pdf`.

## Running

```bash
conda activate omics
RSCRIPT="conda run -n r-env Rscript" bash analysis/run_all.sh
```

Run it from the repository root (or anywhere). Requirements:

- the `omics` environment (`environment.yml`) for the Python steps and the `r-env` environment (`environment-r.yml`) for step 01; set `PY` and `RSCRIPT` to choose the interpreters;
- raw data in `data/raw` (step 0 of `run_all.sh` calls `gdc-client download`, about 5 GB, needs network);
- the GENCODE v36 annotation at `analysis/external/gencode_v36/gencode.v36.annotation.gtf.gz`;
- step 05 (GDC API query for the plate barcodes) is skipped by default and the included `external/gdc/file_aliquots.csv` is used; set `FETCH_GDC_API=1` to re-query the API, which needs network and overwrites the included file. Step 01 uses the data bundled with TCGAbiolinks and needs no network.

The full run takes roughly four hours on six cores. The slowest steps are 13 (~20 min), 21 (~22 min for each of the two designs) and 24 (~40 min). Fits are cached in `analysis/cache/`; delete a cache file to refit.

`export_results.py` runs last: it checks that the category rule reproduces the stored labels and then writes `results/` (documented in `results/README.md`). Figures are made afterwards with `python report/make_figures.py` from the repository root.

Notes:

- Steps 14 → 18 and 16 → 19 pass tables through a rewrite: `18_ceiling_model.py` and `19_site_balanced_cv.py` read copies of the step 14 and 16 outputs from `analysis/results/prev_r2/`; `run_all.sh` makes these copies between steps 17 and 18.
- Intermediate files keep their historical names (`audit_per_mirna_v3.csv`, `audit_meta_v2.json`, …); the final labels are in `audit_per_mirna_r2.csv` (patient-grouped folds) and `audit_per_mirna_plate.csv` (plate-grouped folds, main analysis).
- `analysis/results/`, `analysis/cache/`, `analysis/figures/` and `analysis/external/tcgabiolinks/` are generated and not tracked.
