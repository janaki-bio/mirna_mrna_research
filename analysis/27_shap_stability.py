"""Stability of the SHAP explanations across folds.

For the five miRNAs with the highest holdout R2, the random forest (all 60,660 genes, as for the holdout SHAP) is refit
on each of the 5 patient-grouped folds of repeat 0 (GroupKFold, seed 42) and TreeSHAP is computed on each test fold.
Writes results/13_e8_shap_stability.csv (fold-to-fold Jaccard overlap of the top-10 genes, genes in the top 10 of all
5 folds, overlap with the holdout top 10, rank range of the host gene), results/13_e8_host_ranks.csv and
results/13_e8_fold_top10.csv. The mean Jaccard values are plotted in report/make_figures.py (figure S1).
Needs results/02_shap_top10.csv (26_shap_holdout.py)."""
import itertools

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from common import RESULTS, RF_KW, gene_lookup, ids_for, load_matrices, preprocess

TOP5 = {"hsa-mir-934": "VGLL1", "hsa-mir-150": None, "hsa-mir-577": "UGT8",
        "hsa-mir-18a": "MIR17HG", "hsa-mir-449a": "CDC20B"}

X, mirna_log, _, _, _ = preprocess(*load_matrices())
lookup = gene_lookup()
holdout_top10 = pd.read_csv(RESULTS / "02_shap_top10.csv").groupby("mirna")["gene_id"].apply(set)
folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, groups=X.index.str[:12]))

rank_rows, top_rows, summ = [], [], []
for m, host in TOP5.items():
    y = mirna_log[m].values
    tops = []
    for k, (tr, te) in enumerate(folds):
        rf = RandomForestRegressor(**RF_KW).fit(X.values[tr], y[tr])
        imp = pd.Series(np.abs(shap.TreeExplainer(rf).shap_values(X.values[te])).mean(axis=0), index=X.columns)
        ranks = imp.rank(ascending=False, method="min")
        top = set(imp.nlargest(10).index)
        tops.append(top)
        top_rows += [{"mirna": m, "fold": k, "rank": i + 1, "gene_name": lookup.get(g, g), "gene_id": g}
                     for i, g in enumerate(imp.nlargest(10).index)]
        if host:
            rank_rows.append({"mirna": m, "host": host, "fold": k, "host_rank": int(ranks[ids_for(lookup, host)[0]])})
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    in_all = set.intersection(*tops)
    summ.append({"mirna": m, "jaccard_mean": np.mean(jac), "jaccard_min": np.min(jac), "jaccard_max": np.max(jac),
                 "genes_in_all_5_top10": len(in_all),
                 "genes_in_all_5_names": ";".join(sorted(lookup.get(g, g) for g in in_all)),
                 "mean_overlap_with_holdout_top10": np.mean([len(t & holdout_top10[m]) for t in tops])})
summ = pd.DataFrame(summ)
ranks = pd.DataFrame(rank_rows)
host_range = ranks.groupby(["mirna", "host"])["host_rank"].agg(["min", "max", "median"]).reset_index()
summ = summ.merge(host_range, on="mirna", how="left")
summ.to_csv(RESULTS / "13_e8_shap_stability.csv", index=False)
ranks.to_csv(RESULTS / "13_e8_host_ranks.csv", index=False)
pd.DataFrame(top_rows).to_csv(RESULTS / "13_e8_fold_top10.csv", index=False)
print(summ.round(3).to_string())
