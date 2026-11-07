"""Clinical baselines, and pooled out-of-fold R2 under patient-grouped and site-grouped CV.

Clinical baseline (linear regression, patient-grouped 15 folds, as r2_cov in 09_baseline_models.py):
  (a) PAM50 + CPE (no LUMP);  (b) PAM50 + CPE + LUMP + LUMP-missing indicator (LUMP mean-imputed from the training fold);
  (c) PAM50 + CPE + LUMP, mean-imputed, no indicator (r2_cov; recomputed as a check against audit_per_mirna_v3.csv).
  Shares = R2 / r2_best and R2 / r2_full, medians over the miRNAs that are not poorly predicted (r2_best >= 0.3) and
  over r2_full >= 0.3.
Pooled out-of-fold R2 for r2_full (ridge on all genes) and r2_state_20 (ridge on 20 fold-specific PCs):
  pooled R2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2) over all 1,053 tumours with one out-of-fold prediction per
  tumour per repeat; patient CV: mean of the 3 pooled R2 (seeds 42-44); site CV: one repeat (GroupKFold by tissue source
  site, seed 42, as in 10_state_models.py).  The mean of per-fold R2 is reported alongside (and checked against the
  columns of 10_state_models.py).
Reads audit_per_mirna_r2.csv (14_) and audit_per_mirna_v3.csv (11_).
Writes audit_clinical_r2.csv, audit_site_pooled.csv and cache/meta_clinical_site_r2.json."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from audit_common import ALPHAS, SEEDS, audit_data
from common import CACHE, RESULTS

t0 = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
r2 = pd.read_csv(RESULTS / "audit_per_mirna_r2.csv").set_index("mirna")
v3 = pd.read_csv(RESULTS / "audit_per_mirna_v3.csv").set_index("mirna")
assert list(r2.index) == targets
G = mrna_log[genes].values
Y = mirna_log[targets].values
n, T = Y.shape
PAM = pd.get_dummies(s.pam50, prefix="pam50", drop_first=True).astype(float).values
CPE = s[["cpe"]].values.astype(float)
LUMP = s[["lump"]].values.astype(float)
pfolds = []
for seed in SEEDS:
    pfolds.append(list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(G, groups=s.patient.values)))
site = s.index.str[5:7].values
sfolds = [list(GroupKFold(n_splits=5, shuffle=True, random_state=SEEDS[0]).split(G, groups=site))]


def lin(Ftr, Fte, ytr, yte):
    return r2_score(yte, LinearRegression().fit(Ftr, ytr).predict(Fte), multioutput="raw_values")


# ---------------- clinical baselines ----------------
rec = {"a": [], "b": [], "c": []}
for rep in pfolds:
    for tr, te in rep:
        mu = np.nanmean(LUMP[tr])
        l_tr, l_te = np.where(np.isnan(LUMP[tr]), mu, LUMP[tr]), np.where(np.isnan(LUMP[te]), mu, LUMP[te])
        rec["a"].append(lin(np.hstack([PAM[tr], CPE[tr]]), np.hstack([PAM[te], CPE[te]]), Y[tr], Y[te]))
        rec["b"].append(lin(np.hstack([PAM[tr], CPE[tr], l_tr, np.isnan(LUMP[tr])]), np.hstack([PAM[te], CPE[te], l_te, np.isnan(LUMP[te])]), Y[tr], Y[te]))
        rec["c"].append(lin(np.hstack([PAM[tr], CPE[tr], l_tr]), np.hstack([PAM[te], CPE[te], l_te]), Y[tr], Y[te]))
M = {k: np.vstack(v).mean(0) for k, v in rec.items()}
chk = float(np.abs(M["c"] - v3.r2_cov.values).max())
print("r2_cov recomputed vs v3 max abs diff:", chk)
assert chk < 1e-8
clin = pd.DataFrame({"r2_full": r2.r2_full, "r2_best": r2.r2_best, "label_r2": r2.label_r2,
                     "r2_pam50_cpe": M["a"], "r2_pam50_cpe_lump_indicator": M["b"], "r2_pam50_cpe_lump_meanimputed": M["c"]}, index=targets)
for c in ["r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]:
    clin[f"share_{c[3:]}_over_r2_best"] = clin[c] / clin.r2_best
    clin[f"share_{c[3:]}_over_r2_full"] = clin[c] / clin.r2_full
clin.reset_index().to_csv(RESULTS / "audit_clinical_r2.csv", index=False)
ok_best, ok_full = (clin.label_r2 != "poorly predicted"), (clin.r2_full >= 0.3)
clin_sum = {}
for c in ["r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]:
    nm = c[3:]
    clin_sum[c] = {"median_r2_all_337": float(clin[c].median()),
                   "median_share_over_r2_best_non_poor": float(clin.loc[ok_best, f"share_{nm}_over_r2_best"].median()),
                   "median_share_over_r2_full_r2full_ge_0.3": float(clin.loc[ok_full, f"share_{nm}_over_r2_full"].median()),
                   "max_share_over_r2_best_non_poor": float(clin.loc[ok_best, f"share_{nm}_over_r2_best"].max()),
                   "n_ge_0.8_r2_best_non_poor": int((clin.loc[ok_best, f"share_{nm}_over_r2_best"] >= 0.8).sum()),
                   "n_ge_0.8_r2_full_r2full_ge_0.3": int((clin.loc[ok_full, f"share_{nm}_over_r2_full"] >= 0.8).sum())}
print(pd.DataFrame(clin_sum).T.round(3).to_string())
print(f"task 5 done {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------- pooled out-of-fold R2 ----------------
def pca_scores(Xtr, Xte, kmax):
    ev, U = np.linalg.eigh(Xtr @ Xtr.T)
    o = np.argsort(ev)[::-1][:kmax]
    ev, U = ev[o], U[:, o]
    sv = np.sqrt(ev)
    return U * sv, (Xte @ Xtr.T) @ U / sv


def ridge_pred(Ftr, Fte, ytr):
    sc = StandardScaler().fit(Ftr)
    return RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Ftr), ytr).predict(sc.transform(Fte))


def run(reps):
    out = {m: {"foldmean": [], "pooled": []} for m in ("full", "state20")}
    for rep in reps:
        pred = {m: np.zeros((n, T)) for m in out}
        fold_r2 = {m: [] for m in out}
        for tr, te in rep:
            sc = StandardScaler().fit(G[tr])
            Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
            Ptr, Pte = pca_scores(Xtr, Xte, 20)
            del Xtr, Xte
            pred["full"][te] = ridge_pred(G[tr], G[te], Y[tr])
            pred["state20"][te] = ridge_pred(Ptr, Pte, Y[tr])
            for m in out:
                fold_r2[m].append(r2_score(Y[te], pred[m][te], multioutput="raw_values"))
            print(f"  fold done {(time.time() - t0) / 60:.1f} min", flush=True)
        for m in out:
            out[m]["foldmean"].append(np.vstack(fold_r2[m]).mean(0))
            out[m]["pooled"].append(1 - ((Y - pred[m]) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0))
    return {m: {k: np.mean(v, axis=0) for k, v in d.items()} | {"pooled_by_repeat": np.vstack(d["pooled"])} for m, d in out.items()}


pat = run(pfolds)
site_res = run(sfolds)
chk_p = float(np.abs(pat["full"]["foldmean"] - v3.r2_full.values).max())
chk_s = float(np.abs(site_res["full"]["foldmean"] - v3.r2_full_site.values).max())
chk_s20 = float(np.abs(site_res["state20"]["foldmean"] - v3.r2state20_site.values).max()) if "r2state20_site" in v3 else float(np.abs(site_res["state20"]["foldmean"] - v3.r2_state20_site.values).max())
print("checks (fold mean vs earlier): patient full", chk_p, " site full", chk_s, " site state20", chk_s20)
assert max(chk_p, chk_s, chk_s20) < 1e-8
tab = pd.DataFrame({"r2_best": r2.r2_best, "label_r2": r2.label_r2}, index=targets)
for m, col in (("full", "r2_full"), ("state20", "r2_state_20")):
    tab[f"{col}_patient_foldmean"] = pat[m]["foldmean"]
    tab[f"{col}_patient_pooled"] = pat[m]["pooled"]
    tab[f"{col}_patient_pooled_sd_over_repeats"] = pat[m]["pooled_by_repeat"].std(0, ddof=1)
    tab[f"{col}_site_foldmean"] = site_res[m]["foldmean"]
    tab[f"{col}_site_pooled"] = site_res[m]["pooled"]
    tab[f"{col}_site_pooled_minus_patient_pooled"] = tab[f"{col}_site_pooled"] - tab[f"{col}_patient_pooled"]
    tab[f"{col}_site_foldmean_minus_patient_foldmean"] = tab[f"{col}_site_foldmean"] - tab[f"{col}_patient_foldmean"]
tab.reset_index().to_csv(RESULTS / "audit_site_pooled.csv", index=False)
summ = {}
for col in ("r2_full", "r2_state_20"):
    for cv in ("patient", "site"):
        for kind in ("foldmean", "pooled"):
            v = tab[f"{col}_{cv}_{kind}"]
            summ[f"{col}_{cv}_{kind}"] = {"mean": float(v.mean()), "median": float(v.median()), "n_gt_0.5": int((v > 0.5).sum()),
                                           "n_ge_0.3": int((v >= 0.3).sum())}
    for kind in ("foldmean", "pooled"):
        dlt = tab[f"{col}_site_{kind}_minus_patient_{kind}" if kind == "pooled" else f"{col}_site_foldmean_minus_patient_foldmean"]
        summ[f"{col}_site_minus_patient_{kind}"] = {"mean": float(dlt.mean()), "median": float(dlt.median()),
                                                    "n_drop_gt_0.1": int((dlt < -0.1).sum())}
summ["fold_sizes_site"] = [int(len(te)) for _, te in sfolds[0]]
summ["n_site_levels"] = int(pd.Series(site).nunique())
summ["checks_fold_mean_vs_earlier"] = {"patient_full": chk_p, "site_full": chk_s, "site_state20": chk_s20}
print(json.dumps(summ, indent=1))
json.dump({"clinical": clin_sum, "r2_cov_check_max_abs_diff": chk, "site": summ,
           "clinical_definition": "linear regression, 15 patient-grouped folds; LUMP imputed with training-fold mean; indicator = LUMP missing (348 of 1,053)"},
          open(CACHE / "meta_clinical_site_r2.json", "w"), indent=2)
print(f"done {(time.time() - t0) / 60:.1f} min")
