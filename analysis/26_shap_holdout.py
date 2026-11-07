"""Holdout random forest and TreeSHAP for the miRNAs with the highest holdout R2 and three other host-gene miRNAs.

Refits the random forest (all genes) on the 80/20 holdout split (random_state=42) and computes TreeSHAP on the test set.
Writes cache/shap_mean_abs.pkl, cache/shap_mean_signed.pkl, results/02_shap_top10.csv (top-10 genes per miRNA) and
results/02_shap_direction_goi.csv (Spearman correlation between gene value and SHAP value for genes of interest)."""
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from common import CACHE, RESULTS, RF_KW, gene_lookup, ids_for, load_matrices, preprocess

TOP5 = ["hsa-mir-934", "hsa-mir-150", "hsa-mir-577", "hsa-mir-18a", "hsa-mir-449a"]  # top 5 by holdout R²
EXTRA = ["hsa-mir-449b", "hsa-mir-20a", "hsa-mir-19a"]  # other miRNAs with a host gene
GENES_OF_INTEREST = ["VGLL1", "UGT8", "CDC20B", "MIR17HG", "MCIDAS", "CCNO",
                     "FOXA1", "SPDEF", "AR", "ESR1", "GATA3"]

X, mirna_log, top_mirnas, X_train, X_test = preprocess(*load_matrices())
lookup = gene_lookup()
goi = {g: ids_for(lookup, g)[0] for g in GENES_OF_INTEREST}

mean_abs, mean_signed, dir_rows = {}, {}, []
for mirna in tqdm(TOP5 + EXTRA):
    y = mirna_log[mirna]
    model = RandomForestRegressor(**RF_KW).fit(X_train, y.loc[X_train.index])
    sv = shap.TreeExplainer(model).shap_values(X_test)
    mean_abs[mirna] = np.abs(sv).mean(axis=0)
    mean_signed[mirna] = sv.mean(axis=0)
    for g, gid in goi.items():
        j = X_test.columns.get_loc(gid)
        rho = spearmanr(X_test[gid], sv[:, j])[0] if np.any(sv[:, j] != 0) else np.nan
        dir_rows.append({"mirna": mirna, "gene": g, "gene_id": gid, "rho_value_vs_shap": rho})

mean_abs = pd.DataFrame(mean_abs, index=X_test.columns)
mean_signed = pd.DataFrame(mean_signed, index=X_test.columns)
mean_abs.to_pickle(CACHE / "shap_mean_abs.pkl")
mean_signed.to_pickle(CACHE / "shap_mean_signed.pkl")
pd.DataFrame(dir_rows).to_csv(RESULTS / "02_shap_direction_goi.csv", index=False)

top10 = []
for mirna in mean_abs.columns:
    s = mean_abs[mirna].sort_values(ascending=False).head(10)
    for rank, (gid, v) in enumerate(s.items(), 1):
        top10.append({"mirna": mirna, "rank": rank, "gene_id": gid, "gene_name": lookup.get(gid, gid),
                      "mean_abs_shap": v, "mean_signed_shap": mean_signed.loc[gid, mirna]})
top10 = pd.DataFrame(top10)
top10.to_csv(RESULTS / "02_shap_top10.csv", index=False)
print(top10.pivot(index="rank", columns="mirna", values="gene_name").to_string())
