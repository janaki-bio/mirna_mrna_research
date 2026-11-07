"""Holdout random forest for the 100 most variable miRNA precursors.

For each of the 100 miRNAs with the highest variance of log2(RPM + 1), fits a RandomForestRegressor (common.RF_KW: 100 trees,
max_features="sqrt", random_state=42) on all genes of a random 80/20 split of the matched tumours (common.preprocess,
random_state=42) and records the R2 on the 20% test set. Writes data/r2_results.csv (miRNA, R2; highest first), read by
08_cross_mapped_reads.py and 25_summarise_rf_ridge.py.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

from common import RF_KW, ROOT, load_matrices, preprocess

X, mirna_log, top_mirnas, X_train, X_test = preprocess(*load_matrices())

r2 = {}
for mirna in tqdm(top_mirnas):
    y = mirna_log[mirna]
    model = RandomForestRegressor(**RF_KW).fit(X_train, y.loc[X_train.index])
    r2[mirna] = r2_score(y.loc[X_test.index], model.predict(X_test))

out = pd.Series(r2).sort_values(ascending=False)
(ROOT / "data").mkdir(exist_ok=True)
out.to_csv(ROOT / "data/r2_results.csv")
print(out.head(10).round(4))
print("miRNAs with R2 > 0.5:", int((out > 0.5).sum()))
