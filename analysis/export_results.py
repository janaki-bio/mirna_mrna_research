"""Export the tables used by the report into results/ (repository root).

Reads the analysis outputs in analysis/results/ and writes the curated, documented tables in results/.
Run last, after analysis/run_all.sh:
  python analysis/export_results.py [analysis_results_dir] [out_dir]
Defaults: analysis/results  results
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "analysis/results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(SRC / "audit_per_mirna_plate.csv")
mp = json.loads((SRC / "audit_meta_plate.json").read_text())
m2 = json.loads((SRC / "audit_meta_v2.json").read_text())
cvp = pd.read_csv(SRC / "audit_plate_cv.csv").set_index("mirna")
site = pd.read_csv(SRC / "audit_site_pooled.csv").set_index("mirna")
adj = pd.read_csv(SRC / "audit_plate_adjusted_sym.csv").set_index("mirna")
unc = pd.read_csv(SRC / "audit_uncertainty_plate.csv").set_index("mirna")
clin = pd.read_csv(SRC / "audit_clinical_plate.csv").set_index("mirna")

# ---- the category rule must reproduce the stored labels
REFS = ["r2_host", "r2_state_20", "r2_host_state_20", "r2_locus_any", "r2_locus_state20", "r2_host_locus_state20"]
assert np.allclose(d[["r2_full"] + REFS].max(axis=1), d.r2_best)


def categorise(q, thr=0.8):
    out = []
    for _, r in q.iterrows():
        b = r.r2_best
        if b < 0.3:
            out.append("poorly predicted")
        elif pd.notna(r.r2_host) and r.r2_host >= thr * b:
            out.append("host")
        elif r.r2_state_20 >= thr * b:
            out.append("tumour state")
        elif r.r2_host_state_20 >= thr * b:
            out.append("host + state")
        elif r.r2_locus_state20 >= thr * b:
            out.append("paralog locus" if pd.notna(r.r2_locus_paralog) and r.r2_locus_paralog > r.r2_locus_own else "own locus")
        elif r.r2_full - r.r2_best_ref >= 0.10:
            out.append("residual")
        else:
            out.append("partly explained")
    return pd.Series(out, index=q.index)


assert (categorise(d) == d.label_r2).all()


def flags(r):
    f = []
    if r.frac_cross_mapped > 0.5:
        f.append("X")
    if pd.notna(r.paralog_group):
        f.append("P")
    if r.host_status == "not measurable":
        f.append("H")
    return " ".join(f)


m = d.mirna
detail = np.where(d.label_r2 == "tumour state", "tumour state: " + d.prog1 + np.where(d.prog1_rho < 0, " (−)", " (+)"),
                  d.label_r2)
att = pd.DataFrame({
    "mirna": m, "mir_gene": d.mir_gene, "host_genes": d.host_genes, "host_type": d.host_type,
    "host_status": d.host_status, "paralog_group": d.paralog_group, "paralog_members": d.paralog_members,
    "own_locus_genes": d.own_locus_genes,
    "frac_cross_mapped": d.frac_cross_mapped.round(3), "mean_log2rpm": d.mean_log2rpm.round(3),
    "r2_full": d.r2_full, "r2_full_sd": d.r2_full_sd, "r2_host": d.r2_host,
    "r2_state_5": d.r2_state_5, "r2_state_10": d.r2_state_10, "r2_state_20": d.r2_state_20, "r2_state_50": d.r2_state_50,
    "r2_host_state_5": d.r2_host_state_5, "r2_host_state_10": d.r2_host_state_10,
    "r2_host_state_20": d.r2_host_state_20, "r2_host_state_50": d.r2_host_state_50,
    "r2_locus_own": d.r2_locus_own, "r2_locus_paralog": d.r2_locus_paralog, "r2_locus_any": d.r2_locus_any,
    "r2_locus_state20": d.r2_locus_state20, "r2_host_locus_state20": d.r2_host_locus_state20,
    "r2_pam50_cpe": clin.loc[m, "r2_pam50_cpe"].values,
    "r2_pam50_cpe_lump": clin.loc[m, "r2_pam50_cpe_lump_indicator"].values,
    "ceiling_model": d.ceiling_model.str.replace("_500kb", "", regex=False), "r2_best": d.r2_best,
    "r2_best_reference": d.r2_best_ref, "residual_gap": d.r2_full - d.r2_best_ref,
    "category": d.label_r2, "category_detail": detail,
    "borderline": d.borderline, "robust": d.robust_label, "robust_incl_components": d.robust_label_also_k,
    "category_k10": d.label_r2_k10, "category_k50": d.label_r2_k50,
    "category_250kb": d.label_r2_win250kb, "category_1Mb": d.label_r2_win1Mb, "category_pc_excl": d.label_r2_pc_excl,
    "prog1": d.prog1, "prog1_rho": d.prog1_rho.round(3), "prog2": d.prog2, "prog2_rho": d.prog2_rho.round(3),
    "flags": d.apply(flags, axis=1),
    "r2_full_patient": d.r2_full_patient, "r2_best_patient": d.r2_best_patient, "category_patient": d.label_r2_patient,
    "r2_full_pooled_patient": cvp.loc[m, "r2_full_patient_pooled"].values,
    "r2_full_pooled_plate": cvp.loc[m, "r2_full_plate_pooled"].values,
    "r2_full_pooled_site": site.loc[m, "r2_full_site_balanced_pooled"].values,
    "r2_state20_pooled_patient": cvp.loc[m, "r2_state_20_patient_pooled"].values,
    "r2_state20_pooled_plate": cvp.loc[m, "r2_state_20_plate_pooled"].values,
    "r2_state20_pooled_site": site.loc[m, "r2_state_20_site_balanced_pooled"].values,
    "category_plate_adjusted": adj.loc[m, "label_patient_adjusted"].values,
})
for c in att.columns:
    if c.startswith("r2_") or c == "residual_gap":
        att[c] = att[c].round(4)
att.to_csv(OUT / "mirna_attribution.csv", index=False)

# ---- per-miRNA uncertainty
u = unc.reset_index()[["mirna", "label_plate", "ceiling_model", "r2_full", "r2_best", "test", "d_mean", "d_ci_lo", "d_ci_hi",
                       "share_folds_condition_holds", "borderline", "gate_borderline"]]
u = u.rename(columns={"label_plate": "category", "test": "deciding_condition", "d_mean": "difference_mean",
                      "d_ci_lo": "difference_ci_lo", "d_ci_hi": "difference_ci_hi",
                      "gate_borderline": "predictability_borderline"})
u["ceiling_model"] = u.ceiling_model.str.replace("_500kb", "", regex=False)
u.round(4).to_csv(OUT / "label_uncertainty.csv", index=False)

# ---- category counts under alternative settings
s = pd.read_csv(SRC / "audit_robustness_plate.csv")
s["design"] = s.design.replace({"plate-grouped CV": "plate-grouped (main)",
                                "patient CV, symmetric plate adjustment": "patient-grouped, plate-adjusted",
                                "patient CV (reference, r2 main)": "patient-grouped"})
s = s.rename(columns={"n_changed_vs_own_baseline": "n_changed", "n_r2_best_ge_0.3": "n_predictable"})
s.to_csv(OUT / "sensitivity_settings.csv", index=False)

# ---- cross-validation designs (means over the 337 miRNAs)
rows = []
for model, a, b, c in [("full", "r2_full_patient", "r2_full_site_balanced", "r2_full_plate"),
                       ("20 components", "r2_state_20_patient", "r2_state_20_site_balanced", "r2_state_20_plate")]:
    for design, src, pre in [("patient-grouped", cvp, a), ("site-grouped", site, b), ("plate-grouped", cvp, c)]:
        fm, po = src[pre + "_foldmean"], src[pre + "_pooled"]
        rows.append(dict(model=model, design=design, mean_r2_foldmean=fm.mean(), mean_r2_pooled=po.mean(),
                         n_gt_0_5_foldmean=int((fm > 0.5).sum()), n_gt_0_5_pooled=int((po > 0.5).sum())))
pd.DataFrame(rows).round(4).to_csv(OUT / "cv_design_comparison.csv", index=False)

# ---- principal components: programmes and technical variables
t = pd.read_csv(SRC / "audit_technical_pcs.csv")
p = pd.read_csv(SRC / "audit_pc_programmes.csv")
tp = p.merge(t[["pc", "eta2_tss", "eta2_tss_adj", "eta2_pam50_adj", "eta2_mrna_plate", "eta2_mrna_plate_adj",
                "eta2_mirna_plate", "eta2_mirna_plate_adj", "rho_rnaseq_library_size", "rho_mirna_mapped_reads"]], on="pc")
tp.round(4).to_csv(OUT / "pc_programmes_technical.csv", index=False)
pd.read_csv(SRC / "audit_plate_confounding.csv").round(4).to_csv(OUT / "plate_confounding.csv", index=False)

# ---- supplementary tables
low = pd.read_csv(SRC / "audit_low_expression.csv")
low.drop(columns=[c for c in low.columns if c.startswith(("label_", "delta_"))]).to_csv(OUT / "low_expression_mirnas.csv",
                                                                                       index=False)
pd.read_csv(SRC / "07_f4_per_mirna.csv").to_csv(OUT / "model_comparison_top100.csv", index=False)

# ---- summary
PAM50_COUNTS = {"LumA": 551, "LumB": 200, "Basal": 181, "Her2": 81, "Normal": 40}  # analysis set (audit_samples.csv from 04_select_analysis_set.py)
summ = {
    "analysis_set": {"tumours": 1053, "targets": int(len(d)), "genes": 27671, "pam50_counts": PAM50_COUNTS},
    "main_design": "5-fold cross-validation grouped by mRNA sequencing plate (40 plates; folds balanced by tumour count, "
                   "211/211/211/210/210), 3 repeats (seeds 42-44), 15 test folds",
    "comparison_designs": {"patient-grouped": "GroupKFold by patient, 5 folds x 3 repeats",
                           "site-grouped": "whole tissue source sites (40), folds balanced by tumour count, 3 repeats"},
    "category_counts": {k: mp["counts"][k2] for k, k2 in
                        [("plate-grouped (main)", "plate_cv"), ("patient-grouped", "patient_cv"),
                         ("patient-grouped, plate-adjusted", "patient_cv_symmetric_plate_adjusted")]},
    "median_share_of_r2_best": mp["median_shares_of_r2_best"],
    "robust": {k: mp["robust"][k] for k in ("plate_cv", "patient_cv")},
    "clinical": mp["clinical"],
    "programmes": {k: m2["programmes"][k] for k in ("definition", "sets")},
}
(OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
print("wrote", sorted(x.name for x in OUT.iterdir()))
