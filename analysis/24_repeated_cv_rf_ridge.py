"""Random forest vs ridge regression for the 100 most variable miRNAs: 5-fold CV x 3 repeats.

Patient-grouped folds (seeds 42/43/44), RF and RidgeCV on the same folds, all 1,087 matched tumours; genes filtered to
TPM > 1 in >= 20% of samples. Writes results/06_f4_cv_folds.csv (one row per model, miRNA, repeat and fold; checkpointed
after every fold and resumed from it) and results/06_f4_gene_filter.csv. Takes about 40 minutes (the random forests).
Summaries are produced by 25_summarise_rf_ridge.py."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import RESULTS, RF_KW, load_matrices, preprocess

SEEDS = [42, 43, 44]
ALPHAS = np.logspace(-2, 8, 41)
OUT = RESULTS / "06_f4_cv_folds.csv"

mrna, mirna = load_matrices()
X, mirna_log, top_mirnas, _, _ = preprocess(mrna, mirna)
keep = ((mrna > 1).mean() >= 0.2).values
X = X.loc[:, keep]
print(f"genes kept (TPM>1 in >=20% of samples): {X.shape[1]} of {len(keep)}")
pd.Series({"genes_total": len(keep), "genes_kept": X.shape[1]}).to_csv(RESULTS / "06_f4_gene_filter.csv")
Y = mirna_log[top_mirnas]
patients = X.index.str[:12]
Xv, Yv = X.values, Y.values

done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["model", "mirna", "repeat", "fold"])
done_keys = set(zip(done["model"], done["mirna"], done["repeat"], done["fold"]))
rows = done.to_dict("records")

t0 = time.time()
for rep, seed in enumerate(SEEDS):
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Xv, groups=patients))
    for k, (tr, te) in enumerate(folds):
        if not all(("ridge", m, rep, k) in done_keys for m in top_mirnas):
            sc = StandardScaler().fit(Xv[tr])
            ridge = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Xv[tr]), Yv[tr])
            P = ridge.predict(sc.transform(Xv[te]))
            for j, m in enumerate(top_mirnas):
                rows.append({"model": "ridge", "mirna": m, "repeat": rep, "fold": k,
                             "r2": r2_score(Yv[te, j], P[:, j]), "alpha": ridge.alpha_[j]})
        for j, m in enumerate(top_mirnas):
            if ("rf", m, rep, k) in done_keys:
                continue
            rf = RandomForestRegressor(**RF_KW).fit(Xv[tr], Yv[tr, j])
            rows.append({"model": "rf", "mirna": m, "repeat": rep, "fold": k,
                         "r2": r2_score(Yv[te, j], rf.predict(Xv[te])), "alpha": np.nan})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"repeat {rep} fold {k} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
