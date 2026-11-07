"""Figures for 'What do mRNA-based predictions of microRNA expression measure?'

Usage (from the repository root):
  python report/make_figures.py [results_dir] [out_dir]
Defaults: results  report/figures
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "report/figures")
OUT.mkdir(parents=True, exist_ok=True)

INK, GRID, MUTED = "#2b2f36", "#e7e9ee", "#8a8f99"
CAT = {  # attribution categories: order, label, colour
    "host": ("Host gene", "#2e8b57"),
    "tumour state": ("Tumour state", "#3b6fb6"),
    "host + state": ("Host gene + tumour state", "#7cb88f"),
    "own locus": ("Own locus", "#7b4fa8"),
    "paralog locus": ("Paralog locus", "#c39be0"),
    "residual": ("Residual signal", "#d9603b"),
    "partly explained": ("Partly explained", "#c9a227"),
    "poorly predicted": ("Poorly predicted", "#c4c7cd"),
}
ORDER = list(CAT)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})
short = lambda m: m.replace("hsa-", "")

d = pd.read_csv(RES / "mirna_attribution.csv")
summ = json.loads((RES / "summary.json").read_text())
pred = d[d.category != "poorly predicted"].copy()
D = d.set_index("mirna")

# ------------------------------------------------------------------ Figure 1: workflow
fig, ax = plt.subplots(figsize=(10.4, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(-2, 100)
ax.axis("off")


def box(x, y, w, h, title, body, fc="#eef3fb", ec="#3b6fb6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, fontweight="bold")
    ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=7.4, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#5a5f69", lw=0.9))


box(4, 83, 42, 15, "TCGA-BRCA (GDC)", "RNA-seq: STAR gene TPM\nmiRNA-seq: miRBase 21 precursor RPM\naliquot barcodes → sequencing plate", "#f3f4f6", "#80858f")
box(54, 83, 42, 15, "Annotations", "GENCODE v36: miRNA loci, host genes,\n±500 kb neighbourhoods, paralog groups\nTCGAbiolinks: PAM50, purity (CPE, LUMP)", "#f3f4f6", "#80858f")
box(18, 63, 64, 15, "Cohort and features", "remove FFPE, one tumour per patient, PAM50 + CPE available → 1,053 tumours\ntargets: 337 miRNA precursors with RPM > 1 in ≥ 50% of tumours\nfeatures: 27,671 genes with TPM > 0.5 in ≥ 10% of tumours; log2(x + 1)")
arrow(25, 83, 40, 78.6)
arrow(75, 83, 60, 78.6)
ax.text(50, 59.3, "Folds grouped by sequencing plate (whole plates held out), 5 folds × 3 repeats; the same folds for every model",
        ha="center", fontsize=7.8, style="italic", color="#444a55", zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
models = [
    (2, "Full model", "ridge on all\n27,671 genes"),
    (21.5, "Host gene", "linear model on the\nhost gene(s)"),
    (41, "Tumour state", "ridge on 20 principal\ncomponents of all genes"),
    (60.5, "Genomic locus", "ridge on genes within\n±500 kb (own, paralog)"),
    (80, "Clinical covariates", "PAM50 subtype,\npurity (CPE, LUMP)"),
]
for x, t, b in models:
    box(x, 39, 18, 14, t, b)
    arrow(50, 63, x + 9, 53.6)
box(14, 18, 72, 15, "Attribution of each miRNA",
    "ceiling = best R² of any model; poorly predicted if the ceiling < 0.3\nfirst reference model reaching 80% of the ceiling: host · tumour state · host + state · locus (own / paralog)\notherwise residual signal (full model ≥ 0.10 above every reference model) or partly explained",
    "#f4eefa", "#8a5bb5")
for x, *_ in models[:4]:
    arrow(x + 9, 39, 50 + (x + 9 - 50) * 0.55, 33.6)
box(4, 0, 28, 13, "Batch and generalisation", "same models with folds grouped\nby patient or by tissue source site;\nplate-adjusted comparison", "#f3f4f6", "#80858f")
box(36, 0, 28, 13, "Robustness", "components, locus window,\nthreshold, gene exclusion;\nfold-level uncertainty", "#f3f4f6", "#80858f")
box(68, 0, 28, 13, "Outputs", "per-miRNA table (337 rows),\nfigures, this report", "#fbecec", "#c0504d")
arrow(36, 18, 18, 13.6)
arrow(50, 18, 50, 13.6)
arrow(64, 18, 82, 13.6)
fig.savefig(OUT / "fig1_workflow.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: batch and cross-validation design
cv = pd.read_csv(RES / "cv_design_comparison.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [0.85, 1.15]})
designs = ["patient-grouped", "site-grouped", "plate-grouped"]
dcol = {"patient-grouped": "#9aa3b2", "site-grouped": "#6f8fc2", "plate-grouped": "#d9603b"}
w = 0.26
for i, des in enumerate(designs):
    vals = [cv[(cv.model == mdl) & (cv.design == des)].mean_r2_pooled.iloc[0] for mdl in ("full", "20 components")]
    xs = np.arange(2) + (i - 1) * w
    a1.bar(xs, vals, w * 0.92, color=dcol[des], label=des.replace("-grouped", ""))
    for xx, v in zip(xs, vals):
        a1.text(xx, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.4)
a1.set_xticks([0, 1], ["Full model\n(27,671 genes)", "Tumour state\n(20 components)"])
a1.set_ylabel("Mean pooled R² (337 miRNAs)")
a1.set_ylim(0, 0.5)
a1.legend(title="Folds grouped by", fontsize=7.8, title_fontsize=7.8, loc="upper right")
a1.grid(axis="x", visible=False)
a1.set_title("A  Held-out batches lower accuracy", loc="left")
x, y = d.r2_full_pooled_patient, d.r2_full_pooled_plate
a2.scatter(x, y, s=13, color="#3b6fb6", alpha=0.75, edgecolor="white", lw=0.2)
a2.plot([-0.2, 0.85], [-0.2, 0.85], color=INK, lw=0.8, ls="--")
a2.plot([-0.2, 0.85], [-0.3, 0.75], color=MUTED, lw=0.7, ls=":")
a2.text(0.80, 0.66, "−0.1", fontsize=7.5, color=MUTED)
for m, dx, dy in [("hsa-mir-143", 6, -2), ("hsa-mir-326", 6, -4), ("hsa-mir-330", 6, -6), ("hsa-mir-19a", -48, 16),
                  ("hsa-mir-17", -30, 5), ("hsa-mir-30c-1", 6, -4)]:
    a2.annotate(short(m), (D.loc[m, "r2_full_pooled_patient"], D.loc[m, "r2_full_pooled_plate"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.3)
a2.set_xlim(-0.05, 0.85)
a2.set_ylim(-0.2, 0.85)
a2.set_xlabel("Full-model R², folds grouped by patient")
a2.set_ylabel("Full-model R², folds grouped by plate")
a2.set_title("B  Each miRNA (pooled out-of-fold R²)", loc="left")
a2.text(0.02, 0.98, f"drop > 0.1: {int(((x - y) > 0.1).sum())} miRNAs\nR² > 0.5: {(x > 0.5).sum()} → {(y > 0.5).sum()}",
        transform=a2.transAxes, va="top", fontsize=8.2)
fig.tight_layout()
fig.savefig(OUT / "fig2_batch.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 3: predictability
fig, ax = plt.subplots(figsize=(7.4, 3.3))
bins = np.arange(-0.1, 0.925, 0.025)
vals = [d.loc[d.category == c, "r2_best"] for c in ORDER]
ax.hist(vals, bins=bins, stacked=True, color=[CAT[c][1] for c in ORDER], label=[CAT[c][0] for c in ORDER],
        edgecolor="white", linewidth=0.4)
ax.axvline(0.3, color=INK, lw=0.8, ls="--")
ax.axvline(0.5, color=INK, lw=0.8, ls=":")
ax.set_xlabel("Ceiling R² (best of the full and reference models; plate-grouped folds)")
ax.set_ylabel("Number of miRNAs")
ax.text(0.305, ax.get_ylim()[1] * 0.97, "0.3", fontsize=7.5, va="top")
ax.text(0.505, ax.get_ylim()[1] * 0.97, "0.5", fontsize=7.5, va="top")
ax.legend(fontsize=7.6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(OUT / "fig3_predictability.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 4: reference models
ks = [5, 10, 20, 50]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [0.8, 1.25]})
q = np.array([np.nanpercentile(pred[f"r2_state_{k}"] / pred.r2_best, [25, 50, 75]) for k in ks])
qh = np.array([np.nanpercentile(pred[f"r2_host_state_{k}"] / pred.r2_best, [25, 50, 75]) for k in ks])
xk = np.arange(len(ks))
a1.fill_between(xk, q[:, 0], q[:, 2], color="#3b6fb6", alpha=0.18, lw=0)
a1.plot(xk, q[:, 1], marker="o", color="#3b6fb6", label="Tumour state (components)")
a1.fill_between(xk, qh[:, 0], qh[:, 2], color="#2e8b57", alpha=0.15, lw=0)
a1.plot(xk, qh[:, 1], marker="s", color="#2e8b57", label="Host gene + tumour state")
a1.set_xticks(xk, [str(k) for k in ks])
a1.set_xlabel("Number of principal components")
a1.set_ylabel("Share of ceiling R² (median, IQR)")
a1.set_ylim(0, 1.1)
a1.axhline(1, color=INK, lw=0.7, ls="--")
a1.set_title("A  More components", loc="left")
a1.legend(fontsize=7.8, loc="lower right")
mods = [("r2_pam50_cpe", "PAM50 + purity (CPE)"), ("r2_host", "Host gene"), ("r2_state_20", "Tumour state (20 PCs)"),
        ("r2_host_state_20", "Host gene + tumour state"), ("r2_locus_state20", "Locus + tumour state"),
        ("r2_full", "Full model")]
data = [(pred[c] / pred.r2_best).dropna().clip(-0.2, 1.05) for c, _ in mods]
bp = a2.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                medianprops=dict(color=INK, lw=1.2))
cols = ["#b9c0cc", "#2e8b57", "#3b6fb6", "#7cb88f", "#7b4fa8", "#5a5f69"]
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c)
    patch.set_alpha(0.8)
    patch.set_edgecolor(INK)
for i, v in enumerate(data, 1):
    a2.text(1.12, i, f"{np.median(v):.2f}", va="center", fontsize=8)
a2.set_yticks(range(1, len(mods) + 1), [m[1] for m in mods])
a2.axvline(1, color=INK, lw=0.7, ls="--")
a2.axvline(0.8, color=MUTED, lw=0.7, ls=":")
a2.set_xlim(-0.2, 1.2)
a2.set_xlabel("Model R² / ceiling R²")
a2.set_title(f"B  Share of the ceiling, {len(pred)} predictable miRNAs (median at right)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig4_reference_models.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 5: attribution
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw={"width_ratios": [1.1, 0.9]})
for c in ORDER[:-1]:
    mk = pred.category == c
    a1.scatter(pred.r2_full[mk], pred.r2_best_reference[mk], s=16, color=CAT[c][1], label=CAT[c][0], edgecolor="white",
               linewidth=0.3)
xx = np.linspace(-0.05, 0.9, 10)
a1.plot(xx, xx, color=INK, lw=0.8, ls="--")
a1.plot(xx, xx - 0.10, color=MUTED, lw=0.8, ls=":")
for m, dx, dy in [("hsa-mir-150", -12, -13), ("hsa-mir-196a-1", 6, -3), ("hsa-mir-200c", 6, -4), ("hsa-mir-194-1", 6, -3),
                  ("hsa-mir-30a", -40, 4), ("hsa-mir-1270", 4, 4), ("hsa-mir-17", 4, -10)]:
    r = D.loc[m]
    a1.annotate(short(m), (r.r2_full, r.r2_best_reference), xytext=(dx, dy), textcoords="offset points", fontsize=7.3)
a1.set_xlim(-0.05, 0.85)
a1.set_ylim(0, 0.95)
a1.set_xlabel("Full-model R²")
a1.set_ylabel("Best reference-model R²")
a1.set_title("A  Each predictable miRNA", loc="left")
a1.legend(fontsize=7, loc="upper left", ncol=1)
cc = {"plate-grouped": d.category.value_counts(), "patient-grouped": d.category_patient.value_counts()}
yy = np.arange(len(ORDER))[::-1]
for j, (lab, ser) in enumerate(cc.items()):
    v = ser.reindex(ORDER).fillna(0).values
    a2.barh(yy + (0.2 if j == 0 else -0.2), v, height=0.38,
            color=[CAT[c][1] for c in ORDER] if j == 0 else "none",
            edgecolor=[CAT[c][1] for c in ORDER], hatch=None if j == 0 else "////", lw=0.8, label=lab)
    for yi, vi in zip(yy, v):
        a2.text(vi + 1.5, yi + (0.2 if j == 0 else -0.2), str(int(vi)), va="center", fontsize=7.2,
                color=INK if j == 0 else MUTED)
a2.set_yticks(yy, [CAT[c][0] for c in ORDER])
a2.set_xlabel("Number of miRNAs (of 337)")
a2.set_xlim(0, 125)
a2.grid(axis="y", visible=False)
a2.legend(fontsize=7.6, loc="upper right", title="Folds grouped by", title_fontsize=7.6)
a2.set_title("B  Categories by cross-validation design", loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig5_attribution.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 6: examples
ex = ["hsa-mir-150", "hsa-mir-18a", "hsa-mir-135b", "hsa-mir-1270", "hsa-mir-200b", "hsa-mir-758", "hsa-mir-196a-1",
      "hsa-mir-194-1"]
e = D.loc[ex]
series = [("r2_full", "Full model", INK, "o"), ("r2_host", "Host gene", "#2e8b57", "D"),
          ("r2_state_20", "Tumour state (20 PCs)", "#3b6fb6", "s"),
          ("r2_locus_own", "Own locus (±500 kb)", "#7b4fa8", "^"),
          ("r2_locus_paralog", "Paralog's locus", "#d9603b", "v"),
          ("r2_best", "Ceiling (best model)", "#c9a227", "*")]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
y = np.arange(len(ex))[::-1]
for col, lab, c, mk in series:
    v = e[col].clip(lower=-0.05)
    ax.scatter(v, y, color=c, marker=mk, s=80 if mk == "*" else 40, label=lab, zorder=3, edgecolor="white", linewidth=0.4)
for yi, (_, r) in zip(y, e.iterrows()):
    vv = [r[c] for c, *_ in series if pd.notna(r[c])]
    ax.plot([max(min(vv), -0.05), max(vv)], [yi, yi], color="#d6d9df", lw=1.2, zorder=1)
ax.set_yticks(y, [f"{short(m)}  ({CAT[e.loc[m, 'category']][0]})" for m in ex])
ax.set_xlim(-0.07, 1.0)
ax.set_xlabel("Cross-validated R² (plate-grouped folds)")
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.15), ncol=3, fontsize=7.6)
fig.tight_layout()
fig.savefig(OUT / "fig6_examples.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 7: paralogs
pl = d[d.paralog_members.notna() & (d.r2_full >= 0.3) & d.r2_locus_own.notna() & d.r2_locus_paralog.notna()]
fig, ax = plt.subplots(figsize=(5.0, 4.7))
win = pl.r2_locus_paralog > pl.r2_locus_own
ax.scatter(pl.r2_locus_own[~win], pl.r2_locus_paralog[~win], s=22, color="#9aa3b2", label="own locus higher",
           edgecolor="white", lw=0.3)
ax.scatter(pl.r2_locus_own[win], pl.r2_locus_paralog[win], s=22, color="#d9603b", label="paralog's locus higher",
           edgecolor="white", lw=0.3)
ax.plot([-0.1, 0.95], [-0.1, 0.95], color=INK, lw=0.8, ls="--")
for m, dx, dy in [("hsa-mir-196a-1", 7, -3), ("hsa-mir-196a-2", -30, 8), ("hsa-mir-9-2", 6, 0),
                  ("hsa-mir-194-1", 6, -2), ("hsa-mir-181b-1", -20, 14)]:
    if m in set(pl.mirna):
        r = D.loc[m]
        ax.annotate(short(m), (r.r2_locus_own, r.r2_locus_paralog), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.3)
ax.set_xlim(-0.1, 0.95)
ax.set_ylim(-0.1, 0.95)
ax.set_aspect("equal")
ax.set_xlabel("R², genes at the miRNA's own locus")
ax.set_ylabel("R², genes at its paralog's locus")
ax.legend(fontsize=7.8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig7_paralogs.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure S1: random forest vs ridge
per = pd.read_csv(RES / "model_comparison_top100.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1.05, 1]})
a1.errorbar(per["ridge_r2_mean"], per["rf_r2_mean"], xerr=per["ridge_r2_sd"], yerr=per["rf_r2_sd"], fmt="o", ms=3.2,
            color="#3b6fb6", ecolor="#a9bfe0", elinewidth=0.6)
a1.plot([-0.05, 0.85], [-0.05, 0.85], ls="--", lw=0.8, color=INK)
a1.axhline(0.5, color=MUTED, lw=0.7)
a1.axvline(0.5, color=MUTED, lw=0.7)
a1.set_xlim(-0.05, 0.85)
a1.set_ylim(-0.05, 0.85)
a1.set_aspect("equal")
a1.set_xlabel("Ridge regression R² (mean ± SD, 15 folds)")
a1.set_ylabel("Random forest R² (mean ± SD, 15 folds)")
a1.set_title("A  Same 100 miRNAs, same folds", loc="left")
a1.text(0.02, 0.98, f"R² > 0.5: ridge {(per.ridge_r2_mean > 0.5).sum()}, random forest {(per.rf_r2_mean > 0.5).sum()}",
        transform=a1.transAxes, va="top", fontsize=8.3)
stab = pd.DataFrame({"mirna": ["mir-934", "mir-150", "mir-577", "mir-18a", "mir-449a"],
                     "jacc": [0.26, 0.28, 0.39, 0.34, 0.41]})  # fold-to-fold Jaccard of the top-10 SHAP genes
a2.barh(stab.mirna[::-1], stab.jacc[::-1], color="#9aa3b2", height=0.6)
a2.axvline(1, color=INK, lw=0.8, ls="--")
for i, v in enumerate(stab.jacc[::-1]):
    a2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
a2.set_xlim(0, 1.05)
a2.set_xlabel("Mean overlap (Jaccard) of top-10 SHAP genes between folds")
a2.set_title("B  Explanations change between folds", loc="left")
a2.grid(axis="y", visible=False)
a2.text(0.98, 0.02, "1.0 = identical gene lists", transform=a2.transAxes, ha="right", fontsize=7.8, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "figS1_random_forest.png")
plt.close(fig)
print("figures written to", OUT)
