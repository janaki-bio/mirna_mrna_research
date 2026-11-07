"""Summaries of the RF vs ridge comparison.

Per-miRNA mean +- SD over the 15 folds with the holdout RF R2 from data/r2_results.csv (results/07_f4_per_mirna.csv, the
source of results/model_comparison_top100.csv), counts with R2 > 0.5 per repeat, a check for a dip in the R2
distributions near 0.45-0.5, agreement between paralog precursors, and two figures (analysis/figures/).
Reads results/06_f4_cv_folds.csv (24_repeated_cv_rf_ridge.py) and data/r2_results.csv (03_holdout_rf_r2.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde, kurtosis, skew, spearmanr

from common import FIGURES, RESULTS, ROOT, load_matrices, preprocess

PARALOGS = [["hsa-mir-9-1", "hsa-mir-9-2", "hsa-mir-9-3"], ["hsa-mir-196a-1", "hsa-mir-196a-2"],
            ["hsa-mir-105-1", "hsa-mir-105-2"], ["hsa-mir-135a-1", "hsa-mir-135a-2"],
            ["hsa-mir-486-1", "hsa-mir-486-2"]]

folds = pd.read_csv(RESULTS / "06_f4_cv_folds.csv")
assert len(folds) == 2 * 100 * 15, f"incomplete CV run: {len(folds)} rows"
holdout = pd.read_csv(ROOT / "data/r2_results.csv", index_col=0).iloc[:, 0]

# per miRNA: mean/SD over all 15 folds, and per-repeat means
per_rep = folds.groupby(["model", "mirna", "repeat"])["r2"].mean().unstack("repeat")
summ = folds.groupby(["model", "mirna"])["r2"].agg(r2_mean="mean", r2_sd="std").join(
    per_rep.rename(columns=lambda r: f"r2_repeat{r}"))
wide = summ["r2_mean"].unstack("model").rename(columns=lambda c: f"{c}_r2_mean").join(
    summ["r2_sd"].unstack("model").rename(columns=lambda c: f"{c}_r2_sd"))
wide["holdout_rf_r2"] = holdout
wide["rf_minus_ridge"] = wide["rf_r2_mean"] - wide["ridge_r2_mean"]
alpha = folds[folds.model == "ridge"].groupby("mirna")["alpha"].median()
wide["ridge_median_alpha"] = alpha
wide = wide.sort_values("rf_r2_mean", ascending=False)
wide.to_csv(RESULTS / "07_f4_per_mirna.csv")
summ.to_csv(RESULTS / "07_f4_per_mirna_long.csv")

# counts > 0.5
cnt = []
for model in ["rf", "ridge"]:
    pr = per_rep.loc[model]
    per_repeat_counts = [(pr[r] > 0.5).sum() for r in pr.columns]
    cnt.append({"model": model, "n_gt_0.5_overall_mean": (summ.loc[model, "r2_mean"] > 0.5).sum(),
                "n_gt_0.5_per_repeat": ";".join(map(str, per_repeat_counts)),
                "min": min(per_repeat_counts), "max": max(per_repeat_counts),
                "mean_r2_all_mirnas": summ.loc[model, "r2_mean"].mean(),
                "median_r2_all_mirnas": summ.loc[model, "r2_mean"].median()})
cnt = pd.DataFrame(cnt)
cnt["spearman_rf_vs_ridge_per_mirna"] = spearmanr(wide["rf_r2_mean"], wide["ridge_r2_mean"])[0]
cnt["n_mirnas_ridge_beats_rf"] = (wide["rf_minus_ridge"] < 0).sum()
cnt.to_csv(RESULTS / "07_f4_counts.csv", index=False)


# dip check: KDE local minima + Sarle's bimodality coefficient (BC > 0.555 suggests bimodality)
def bimodality(v):
    n = len(v)
    g, k = skew(v), kurtosis(v)  # excess kurtosis
    bc = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    grid = np.linspace(v.min() - 0.1, v.max() + 0.1, 500)
    for bw in ["scott", 0.15]:
        d = gaussian_kde(v, bw_method=bw)(grid)
        mins = grid[argrelextrema(d, np.less)[0]]
        yield {"bw": bw, "bimodality_coef": bc, "kde_local_minima": ";".join(f"{m:.3f}" for m in mins),
               "min_in_0.40_0.55": bool(((mins > 0.40) & (mins < 0.55)).any())}


dip = []
for name, v in [("holdout_rf", holdout.values), ("cv_rf", wide["rf_r2_mean"].values),
                ("cv_ridge", wide["ridge_r2_mean"].values)]:
    for d in bimodality(v):
        dip.append({"distribution": name, **d})
dip = pd.DataFrame(dip)
dip.to_csv(RESULTS / "07_f4_dip_check.csv", index=False)

# paralogs: R² agreement and correlation of the targets themselves
_, mirna_log, _, _, _ = preprocess(*load_matrices())
par = []
for grp in PARALOGS:
    for model in ["rf", "ridge"]:
        vals = wide.loc[grp, f"{model}_r2_mean"]
        par.append({"group": "/".join(g.replace("hsa-mir-", "") for g in grp), "model": model,
                    "r2_means": ";".join(f"{v:.3f}" for v in vals), "r2_range": vals.max() - vals.min(),
                    "min_target_spearman": mirna_log[grp].corr(method="spearman").values.min(),
                    "targets_identical": bool((mirna_log[grp].nunique(axis=1) == 1).all())})
par = pd.DataFrame(par)
par.to_csv(RESULTS / "07_f4_paralogs.csv", index=False)

# figure: holdout vs CV RF vs CV ridge histograms
bins = np.arange(-0.1, 0.85, 0.05)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (name, v) in zip(axes, [("Holdout RF (notebook)", holdout), ("CV RF, mean of 15 folds", wide["rf_r2_mean"]),
                                ("CV ridge, mean of 15 folds", wide["ridge_r2_mean"])]):
    ax.hist(v, bins=bins, color="steelblue", edgecolor="white")
    ax.axvspan(0.45, 0.5, color="orange", alpha=0.25, label="0.45-0.50")
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_title(name)
    ax.set_xlabel("R²")
axes[0].set_ylabel("Number of miRNAs")
axes[0].legend()
fig.tight_layout()
fig.savefig(FIGURES / "f4_r2_distributions.png", dpi=150)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.errorbar(wide["ridge_r2_mean"], wide["rf_r2_mean"], xerr=wide["ridge_r2_sd"], yerr=wide["rf_r2_sd"],
            fmt="o", ms=3, alpha=0.6, elinewidth=0.5)
lim = [min(wide[["rf_r2_mean", "ridge_r2_mean"]].min()) - 0.05, 0.85]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("Ridge CV R² (mean ± SD)")
ax.set_ylabel("RF CV R² (mean ± SD)")
ax.set_title("RF vs ridge, same folds")
fig.tight_layout()
fig.savefig(FIGURES / "f4_rf_vs_ridge.png", dpi=150)

print(cnt.round(3).to_string())
print(dip.to_string())
print(par.to_string())
print(wide.head(20).round(3).to_string())
