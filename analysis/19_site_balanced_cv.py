"""Site-grouped CV with balanced folds, and clinical shares against the final r2_best.

Site folds of 10_state_models.py: GroupKFold(5, shuffle=True, random_state=42) by tissue source site. With shuffle=True
sklearn splits the shuffled list of 40 sites into 5 groups of 8 sites each and ignores tumour counts, hence fold sizes
247/206/226/41/333.
Balanced folds: sites are kept whole; for each seed (42, 43, 44) 5,000 random site orders are tried (size x U(0.8, 1.2)
noise, largest first, each site to the currently lightest fold) and the partition with the smallest max-min fold size is
kept. Recomputes r2_full and r2_state_20 (same models as 16_clinical_site_cv.py) as pooled out-of-fold R2 and mean of
per-fold R2 on the balanced folds (3 repeats) and adds them to audit_site_pooled.csv (*_site_balanced_* columns).
Clinical: the shares in audit_clinical_r2.csv are recomputed from the stored R2 columns with r2_best / label_r2 of
18_ceiling_model.py.
Reads audit_per_mirna_r2.csv (18_) and results/prev_r2/audit_site_pooled.csv and audit_clinical_r2.csv (copies of the
16_ outputs, made by run_all.sh before 18_).
Rewrites audit_site_pooled.csv and audit_clinical_r2.csv; writes cache/meta_r2b_site_clinical.json."""
import json, time
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from audit_common import ALPHAS, SEEDS, audit_data
from common import CACHE, RESULTS

t0 = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
r2 = pd.read_csv(RESULTS / "audit_per_mirna_r2.csv").set_index("mirna")
site_prev = pd.read_csv(RESULTS / "prev_r2" / "audit_site_pooled.csv", index_col=0)
G, Y = mrna_log[genes].values, mirna_log[targets].values
n, T = Y.shape
site = pd.Series(s.index.str[5:7], index=s.index)
sizes = site.value_counts()
print("sites", len(sizes), "tumours per site: min", sizes.min(), "median", sizes.median(), "max", sizes.max(), "; ideal fold", n / 5)


def balanced(seed, trials=5000):
    rng = np.random.default_rng(seed)
    names, w = sizes.index.values, sizes.values.astype(float)
    best = None
    for _ in range(trials):
        order = np.argsort(-(w * rng.uniform(0.8, 1.2, len(w))))
        load, assign = np.zeros(5), {}
        for i in order:
            f = int(np.argmin(load)); load[f] += w[i]; assign[names[i]] = f
        spread = load.max() - load.min()
        if best is None or spread < best[0]:
            best = (spread, assign, load.copy())
    return best


folds_bal, fold_sizes = [], {}
for seed in SEEDS:
    spread, assign, load = balanced(seed)
    f = site.map(assign).values
    folds_bal.append([(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)])
    fold_sizes[seed] = [int(len(te)) for _, te in folds_bal[-1]]
    assert sum(fold_sizes[seed]) == n and all(len(set(site.values[tr]) & set(site.values[te])) == 0 for tr, te in folds_bal[-1])
    print("seed", seed, "balanced site folds:", fold_sizes[seed])


def pca_scores(Xtr, Xte, kmax):
    ev, U = np.linalg.eigh(Xtr @ Xtr.T); o = np.argsort(ev)[::-1][:kmax]; ev, U = ev[o], U[:, o]; sv = np.sqrt(ev)
    return U * sv, (Xte @ Xtr.T) @ U / sv


def ridge_pred(Ftr, Fte, ytr):
    sc = StandardScaler().fit(Ftr)
    return RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Ftr), ytr).predict(sc.transform(Fte))


res = {m: {"foldmean": [], "pooled": []} for m in ("full", "state20")}
for rep in folds_bal:
    pred = {m: np.zeros((n, T)) for m in res}; fr = {m: [] for m in res}
    for tr, te in rep:
        sc = StandardScaler().fit(G[tr]); Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
        Ptr, Pte = pca_scores(Xtr, Xte, 20); del Xtr, Xte
        pred["full"][te] = ridge_pred(G[tr], G[te], Y[tr]); pred["state20"][te] = ridge_pred(Ptr, Pte, Y[tr])
        for m in res: fr[m].append(r2_score(Y[te], pred[m][te], multioutput="raw_values"))
        print(f"  fold {(time.time() - t0) / 60:.1f} min", flush=True)
    for m in res:
        res[m]["foldmean"].append(np.vstack(fr[m]).mean(0))
        res[m]["pooled"].append(1 - ((Y - pred[m]) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0))
tab = site_prev.copy()
tab["r2_best"], tab["label_r2"] = r2.r2_best, r2.label_r2
for m, col in (("full", "r2_full"), ("state20", "r2_state_20")):
    tab[f"{col}_site_balanced_foldmean"] = np.mean(res[m]["foldmean"], 0)
    tab[f"{col}_site_balanced_pooled"] = np.mean(res[m]["pooled"], 0)
    tab[f"{col}_site_balanced_pooled_sd_over_repeats"] = np.std(res[m]["pooled"], 0, ddof=1)
    tab[f"{col}_site_balanced_pooled_minus_patient_pooled"] = tab[f"{col}_site_balanced_pooled"] - tab[f"{col}_patient_pooled"]
    tab[f"{col}_site_balanced_foldmean_minus_patient_foldmean"] = tab[f"{col}_site_balanced_foldmean"] - tab[f"{col}_patient_foldmean"]
tab.to_csv(RESULTS / "audit_site_pooled.csv", index_label="mirna")
summ = {}
for col in ("r2_full", "r2_state_20"):
    for k in ("patient_foldmean", "patient_pooled", "site_foldmean", "site_pooled", "site_balanced_foldmean", "site_balanced_pooled"):
        v = tab[f"{col}_{k}"]; summ[f"{col}_{k}"] = {"mean": float(v.mean()), "median": float(v.median()), "n_gt_0.5": int((v > 0.5).sum()), "n_ge_0.3": int((v >= 0.3).sum())}
    for kind, a, b in (("foldmean", "site_balanced_foldmean", "patient_foldmean"), ("pooled", "site_balanced_pooled", "patient_pooled")):
        dl = tab[f"{col}_{a}"] - tab[f"{col}_{b}"]
        summ[f"{col}_balanced_site_minus_patient_{kind}"] = {"mean": float(dl.mean()), "median": float(dl.median()), "n_drop_gt_0.1": int((dl < -0.1).sum())}
summ["original_fold_sizes"] = [247, 206, 226, 41, 333]; summ["balanced_fold_sizes"] = fold_sizes
summ["sites"] = {"n": int(len(sizes)), "min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max()), "ideal_fold": n / 5}
print(json.dumps(summ, indent=1))

# ---- clinical refresh ----
cl = pd.read_csv(RESULTS / "prev_r2" / "audit_clinical_r2.csv", index_col=0)
base_cols = ["r2_full", "r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]
new = cl[base_cols].copy(); new["r2_best"], new["label_r2"] = r2.r2_best, r2.label_r2
ok_best, ok_full = new.label_r2 != "poorly predicted", new.r2_full >= 0.3
clin = {}
for c in base_cols[1:]:
    nm = c[3:]
    new[f"share_{nm}_over_r2_best"] = new[c] / new.r2_best; new[f"share_{nm}_over_r2_full"] = new[c] / new.r2_full
    clin[c] = {"median_r2_all_337": float(new[c].median()), "median_share_over_r2_best_non_poor": float(new.loc[ok_best, f"share_{nm}_over_r2_best"].median()),
               "median_share_over_r2_full_r2full_ge_0.3": float(new.loc[ok_full, f"share_{nm}_over_r2_full"].median()),
               "max_share_over_r2_best_non_poor": float(new.loc[ok_best, f"share_{nm}_over_r2_best"].max()),
               "n_ge_0.8_r2_best_non_poor": int((new.loc[ok_best, f"share_{nm}_over_r2_best"] >= 0.8).sum())}
new.to_csv(RESULTS / "audit_clinical_r2.csv", index_label="mirna")
print(pd.DataFrame(clin).T.round(3).to_string())
json.dump({"site": summ, "clinical": clin}, open(CACHE / "meta_r2b_site_clinical.json", "w"), indent=2)
print(f"done {(time.time() - t0) / 60:.1f} min")
