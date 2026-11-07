#!/usr/bin/env bash
# Runs the whole analysis in dependency order, then writes the public tables (export_results.py runs last).
#
#   bash analysis/run_all.sh            # from the repository root or from anywhere
#
# Environments: the Python steps use the conda env "omics" (environment.yml), step 01 uses "r-env" (environment-r.yml).
# Override the interpreters if needed:  PY="conda run -n omics python" RSCRIPT=/path/to/Rscript bash analysis/run_all.sh
# Network access is needed for step 0 (gdc-client download, about 5 GB). Step 05 (GDC API query for the plate barcodes)
# is skipped by default and the included analysis/external/gdc/file_aliquots.csv is used; set FETCH_GDC_API=1 to
# re-query the API (this overwrites the included file).
# Runtimes are approximate (6 cores); the slow steps are 13, 21 (twice) and 24. Total: roughly 4 hours; steps 26 and 27 were not timed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
RSCRIPT=${RSCRIPT:-Rscript}
mkdir -p results cache figures
step() { printf '\n=== %s  (%s)\n' "$1" "$(date +%H:%M:%S)"; }

# ---- 0. raw data and annotation -------------------------------------------------------------------------------------
step "0  GDC download [network, ~5 GB]"
if [ ! -d ../data/raw ]; then
  gdc-client download -m ../gdc_manifest.2026-08-17.143921.txt -d ../data/raw
fi
GTF=external/gencode_v36/gencode.v36.annotation.gtf.gz
if [ ! -f "$GTF" ]; then
  echo "Download the GENCODE v36 annotation (gencode.v36.annotation.gtf.gz, https://www.gencodegenes.org) to analysis/$GTF" >&2
  exit 1
fi

# ---- cohort, matrices, annotation -----------------------------------------------------------------------------------
step "01 PAM50 and purity [~1 min]";                           $RSCRIPT 01_export_pam50_purity.R
step "02 expression matrices from data/raw [~2 min]";         $PY 02_build_matrices.py --rebuild
step "03 holdout random forest, data/r2_results.csv [~4 min]"; $PY 03_holdout_rf_r2.py
step "04 analysis set [<1 min]";                               $PY 04_select_analysis_set.py
step "05 plate barcodes [included file; FETCH_GDC_API=1 re-queries the GDC API]"
if [ "${FETCH_GDC_API:-0}" = 1 ]; then $PY 05_fetch_plate_barcodes.py; else echo "using external/gdc/file_aliquots.csv"; fi
step "06 host / locus annotation [~1 min]";                    $PY 06_annotate_hosts.py
step "07 low-expression miRNA hosts [<1 min]";                 $PY 07_annotate_low_expression_hosts.py
step "08 cross-mapped read flags [<1 min]";                   $PY 08_cross_mapped_reads.py

# ---- patient-grouped models and labels ------------------------------------------------------------------------------
step "09 baseline models [~10 min]";                           $PY 09_baseline_models.py
step "10 tumour-state models, programmes [~8 min]";            $PY 10_state_models.py
step "11 locus models [~4 min]";                               $PY 11_locus_models.py
step "12 robustness fits [~11 min]";                           $PY 12_robustness_fits.py
step "13 reference-model fits, all targets [~20 min]";         $PY 13_reference_model_fits.py
step "14 attribution labels, per-fold-maximum ceiling [<1 min]"; $PY 14_attribution_labels.py
step "15 technical variables vs components [a few min: reads 2,106 raw files once]";          $PY 15_technical_pcs.py
step "16 clinical baselines, pooled R2, site CV [~4 min]";     $PY 16_clinical_site_cv.py
step "17 plate vs components [<1 min]";                        $PY 17_plate_vs_components.py

# Snapshot of the outputs of 14 and 16. 18_ceiling_model.py and 19_site_balanced_cv.py rewrite audit_per_mirna_r2.csv,
# audit_site_pooled.csv and audit_clinical_r2.csv in place, but also read the versions written by 14 and 16 from
# results/prev_r2/. Take the copy here, after 14 and 16 and before 18; copying later would feed 18/19 their own output.
mkdir -p results/prev_r2
cp results/audit_per_mirna_r2.csv results/audit_site_pooled.csv results/audit_clinical_r2.csv results/prev_r2/

step "18 ceiling model by mean R2 (final labels) [<1 min]";    $PY 18_ceiling_model.py
step "19 balanced site folds [~3 min]";                        $PY 19_site_balanced_cv.py

# ---- plate-grouped CV (main design) ---------------------------------------------------------------------------------
step "20 plate-grouped CV, confounding, plate-adjusted components [~4 min]"; $PY 20_plate_cv_checks.py
step "21 plate-grouped fits [~22 min]";                        $PY 21_plate_cv_fits.py --design plate
step "21 plate-adjusted patient-grouped fits [~22 min]";       $PY 21_plate_cv_fits.py --design adj
step "22 labels under plate-grouped CV [<1 min]";              $PY 22_plate_labels.py
step "23 plate-CV metadata [<1 min]";                          $PY 23_plate_meta.py

# ---- random forest vs ridge, SHAP -----------------------------------------------------------------------------------
step "24 RF vs ridge, 5-fold x 3 repeats, 100 miRNAs [~40 min]"; $PY 24_repeated_cv_rf_ridge.py
step "25 summary of 24 [<1 min]";                              $PY 25_summarise_rf_ridge.py
step "26 holdout SHAP [not timed]";                              $PY 26_shap_holdout.py
step "27 SHAP stability across folds [not timed]";               $PY 27_shap_stability.py

# ---- public tables (last) -------------------------------------------------------------------------------------------
step "export_results.py: tables for results/";                 $PY export_results.py results ../results
echo; echo "Done. Figures: python report/make_figures.py (from the repository root)."
