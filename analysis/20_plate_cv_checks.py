"""Plate-grouped CV, plate confounding and plate-adjusted components.

mRNA plate = 5th field of the aliquot barcode of the STAR file (external/gdc/file_aliquots.csv).
1. Plate-grouped CV: whole mRNA plates per fold, balanced by tumour count exactly as the site folds of
   19_site_balanced_cv.py (5,000 random orders per seed, seeds 42-44, smallest max-min spread). r2_full (ridge, all genes)
   and r2_state_20 (ridge on 20 fold-specific PCs): pooled out-of-fold R2 (1 prediction per tumour per repeat, mean of 3
   repeats) and mean of per-fold R2; compared with patient CV (columns of audit_site_pooled.csv). Shares =
   r2_state_20 / r2_best (r2_best = main patient-CV ceiling of 18_ceiling_model.py) and r2_state_20 / r2_full within the
   same CV.
2. Cramer's V (plain and bias-corrected, Bergsma 2013): mRNA plate vs miRNA plate, mRNA plate vs tissue source site.
3. Plate-adjusted PCs: inside each training fold the per-gene plate means (one-hot regression, = plate mean minus training
   grand mean) are removed from the training expression matrix; the same training-fold shifts are applied to the test fold
   (a plate absent from the training fold gets no shift); then standardise, PCA (20), RidgeCV on the PCs
   (r2_state_20_pladj) and host + PCs for measurable hosts (r2_host_state_20_pladj). Patient-grouped 15 folds. Locus
   models keep the standard 20 PCs (as in the PC-exclusion check of 12_robustness_fits.py); the reference set uses the
   adjusted state / host + state columns. Labels: main rule (ceiling model by mean R2), same thresholds.
Reads audit_per_mirna_r2.csv (18_), audit_site_pooled.csv (19_), audit_hosts.csv.
Writes audit_plate_cv.csv, audit_plate_adjusted_r2.csv, audit_plate_confounding.csv and cache/meta_r2c.json (plate fold
sizes, checked by 21_plate_cv_fits.py)."""
import json, time
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from audit_common import ALPHAS, SEEDS, audit_data
from common import CACHE, HERE, RESULTS

t0 = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
al = pd.read_csv(HERE / "external/gdc/file_aliquots.csv").set_index("file_id")
plate_of = al.aliquot.str.split("-").str[5]
mp = s.mrna_file_id.map(plate_of); mip = s.mirna_file_id.map(plate_of)
site = pd.Series(s.index.str[5:7], index=s.index)
r2 = pd.read_csv(RESULTS / "audit_per_mirna_r2.csv").set_index("mirna")
sp = pd.read_csv(RESULTS / "audit_site_pooled.csv", index_col=0)
d = pd.read_pickle(CACHE / "audit_r2_fold_records.pkl")
F, measurable = d["F"], d["measurable"]
assert list(r2.index) == targets == list(sp.index)
G, Y = mrna_log[genes].values, mirna_log[targets].values
n, T = Y.shape
ok = (r2.label_r2 != "poorly predicted").values


# ---------------- 2. confounding ----------------
def cramers(a, b):
    ct = pd.crosstab(a, b); chi2 = chi2_contingency(ct, correction=False)[0]
    N = ct.values.sum(); r, k = ct.shape
    v = np.sqrt(chi2 / (N * (min(r, k) - 1)))
    phi2c = max(0, chi2 / N - (r - 1) * (k - 1) / (N - 1))
    rc, kc = r - (r - 1) ** 2 / (N - 1), k - (k - 1) ** 2 / (N - 1)
    return {"cramers_v": float(v), "cramers_v_bias_corrected": float(np.sqrt(phi2c / (min(rc, kc) - 1))), "chi2": float(chi2),
            "n_rows": int(r), "n_cols": int(k), "n_tumours": int(N), "n_nonzero_cells": int((ct.values > 0).sum()), "n_cells": int(ct.size)}


conf = {"mrna_plate_vs_mirna_plate": cramers(mp, mip), "mrna_plate_vs_tss": cramers(mp, site), "mirna_plate_vs_tss": cramers(mip, site)}
# nesting: is each plate within one site? (fraction of tumours in the plate's majority site)
conf["mrna_plate_majority_site_share"] = float(pd.crosstab(mp, site).max(axis=1).sum() / n)
conf["n_mrna_plates_in_single_site"] = int((pd.crosstab(mp, site) > 0).sum(axis=1).eq(1).sum())
conf["n_sites_in_single_mrna_plate"] = int((pd.crosstab(mp, site) > 0).sum(axis=0).eq(1).sum())
pd.DataFrame([{"pair": k, **v} for k, v in conf.items() if isinstance(v, dict)]).to_csv(RESULTS / "audit_plate_confounding.csv", index=False)
print(json.dumps(conf, indent=1))

# ---------------- 1. plate-grouped CV ----------------
sizes = mp.value_counts()
print("mRNA plates", len(sizes), "min", sizes.min(), "median", sizes.median(), "max", sizes.max(), "ideal fold", n / 5)


def balanced(seed, trials=5000):
    rng = np.random.default_rng(seed)
    names, w = sizes.index.values, sizes.values.astype(float)
    best = None
    for _ in range(trials):
        order = np.argsort(-(w * rng.uniform(0.8, 1.2, len(w))))
        load, assign = np.zeros(5), {}
        for i in order:
            f = int(np.argmin(load)); load[f] += w[i]; assign[names[i]] = f
        sp_ = load.max() - load.min()
        if best is None or sp_ < best[0]:
            best = (sp_, assign)
    return best[1]


plate_folds, fsizes = [], {}
for seed in SEEDS:
    f = mp.map(balanced(seed)).values
    plate_folds.append([(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)])
    fsizes[seed] = [int(len(te)) for _, te in plate_folds[-1]]
    assert all(not set(mp.values[tr]) & set(mp.values[te]) for tr, te in plate_folds[-1])
    print("seed", seed, "plate folds", fsizes[seed])


def pca_scores(Xtr, Xte, kmax):
    ev, U = np.linalg.eigh(Xtr @ Xtr.T); o = np.argsort(ev)[::-1][:kmax]; ev, U = ev[o], U[:, o]; sv = np.sqrt(ev)
    return U * sv, (Xte @ Xtr.T) @ U / sv


def ridge_pred(Ftr, Fte, ytr):
    sc = StandardScaler().fit(Ftr)
    return RidgeCV(alphas=ALPHAS, alpha_per_target=ytr.ndim > 1).fit(sc.transform(Ftr), ytr).predict(sc.transform(Fte))


res = {m: {"foldmean": [], "pooled": []} for m in ("full", "state20")}
for rep in plate_folds:
    pred = {m: np.zeros((n, T)) for m in res}; fr = {m: [] for m in res}
    for tr, te in rep:
        sc = StandardScaler().fit(G[tr]); Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
        Ptr, Pte = pca_scores(Xtr, Xte, 20); del Xtr, Xte
        pred["full"][te] = ridge_pred(G[tr], G[te], Y[tr]); pred["state20"][te] = ridge_pred(Ptr, Pte, Y[tr])
        for m in res: fr[m].append(r2_score(Y[te], pred[m][te], multioutput="raw_values"))
        print(f"  plate fold {(time.time() - t0) / 60:.1f} min", flush=True)
    for m in res:
        res[m]["foldmean"].append(np.vstack(fr[m]).mean(0))
        res[m]["pooled"].append(1 - ((Y - pred[m]) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0))
pc = pd.DataFrame({"r2_best": r2.r2_best, "label_r2": r2.label_r2}, index=targets)
for m, col in (("full", "r2_full"), ("state20", "r2_state_20")):
    pc[f"{col}_patient_foldmean"], pc[f"{col}_patient_pooled"] = sp[f"{col}_patient_foldmean"], sp[f"{col}_patient_pooled"]
    pc[f"{col}_plate_foldmean"] = np.mean(res[m]["foldmean"], 0); pc[f"{col}_plate_pooled"] = np.mean(res[m]["pooled"], 0)
    pc[f"{col}_plate_pooled_sd_over_repeats"] = np.std(res[m]["pooled"], 0, ddof=1)
    for k in ("foldmean", "pooled"):
        pc[f"{col}_plate_minus_patient_{k}"] = pc[f"{col}_plate_{k}"] - pc[f"{col}_patient_{k}"]
for k in ("foldmean", "pooled"):
    pc[f"share_state20_over_r2_best_patient_{k}"] = pc[f"r2_state_20_patient_{k}"] / pc.r2_best
    pc[f"share_state20_over_r2_best_plate_{k}"] = pc[f"r2_state_20_plate_{k}"] / pc.r2_best
    pc[f"share_state20_over_full_patient_{k}"] = pc[f"r2_state_20_patient_{k}"] / pc[f"r2_full_patient_{k}"]
    pc[f"share_state20_over_full_plate_{k}"] = pc[f"r2_state_20_plate_{k}"] / pc[f"r2_full_plate_{k}"]
pc.to_csv(RESULTS / "audit_plate_cv.csv", index_label="mirna")
cvsum = {}
for col in ("r2_full", "r2_state_20"):
    for k in ("patient_foldmean", "patient_pooled", "plate_foldmean", "plate_pooled"):
        v = pc[f"{col}_{k}"]; cvsum[f"{col}_{k}"] = {"mean": float(v.mean()), "median": float(v.median()), "n_gt_0.5": int((v > 0.5).sum()), "n_ge_0.3": int((v >= 0.3).sum())}
    for k in ("foldmean", "pooled"):
        dl = pc[f"{col}_plate_minus_patient_{k}"]
        cvsum[f"{col}_plate_minus_patient_{k}"] = {"mean": float(dl.mean()), "median": float(dl.median()), "n_drop_gt_0.1": int((dl < -0.1).sum())}
for k in ("foldmean", "pooled"):
    for nm in ("over_r2_best", "over_full"):
        cvsum[f"median_share_state20_{nm}_{k}"] = {cv: float(np.nanmedian(pc.loc[ok, f"share_state20_{nm}_{cv}_{k}"])) for cv in ("patient", "plate")}
cvsum["plate_fold_sizes"] = fsizes
cvsum["plates"] = {"n": int(len(sizes)), "min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max())}
print(json.dumps({k: v for k, v in cvsum.items() if k.startswith("median") or k == "plate_fold_sizes"}, indent=1))
print(f"plate CV done {(time.time() - t0) / 60:.1f} min", flush=True)

# ---------------- 3. plate-adjusted PCs ----------------
plates = mp.values; pid = pd.Series(plates).astype("category").cat.codes.values; K = pid.max() + 1
folds = []
for seed in SEEDS:
    folds += list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(G, groups=s.patient.values))
host_ids = {t: pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna").loc[t, "host_gene_ids"] for t in []}
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna")
HOSTF = {t: mrna_log[hosts.loc[t, "host_gene_ids"].split(";")].values for t in targets if measurable[t]}
rec = {"r2_state20_pladj": [], "r2_host_state20_pladj": []}
for tr, te in folds:
    D = np.zeros((len(tr), K)); D[np.arange(len(tr)), pid[tr]] = 1
    cnt = D.sum(0); mu = G[tr].mean(0)
    shift = np.zeros((K, G.shape[1])); seen = cnt > 0
    shift[seen] = (D.T @ G[tr])[seen] / cnt[seen, None] - mu
    Xa_tr, Xa_te = G[tr] - shift[pid[tr]], G[te] - shift[pid[te]]       # unseen plate: shift 0
    sc = StandardScaler().fit(Xa_tr)
    Ptr, Pte = pca_scores(sc.transform(Xa_tr), sc.transform(Xa_te), 20)
    del Xa_tr, Xa_te
    se = r2_score(Y[te], ridge_pred(Ptr, Pte, Y[tr]), multioutput="raw_values")
    hs = se.copy()
    for j, t in enumerate(targets):
        if measurable[t]:
            H = HOSTF[t]
            hs[j] = r2_score(Y[te, j], ridge_pred(np.hstack([H[tr], Ptr]), np.hstack([H[te], Pte]), Y[tr, j]))
    rec["r2_state20_pladj"].append(se); rec["r2_host_state20_pladj"].append(hs)
    print(f"  adj fold {(time.time() - t0) / 60:.1f} min  (test plates unseen in training: {len(set(pid[te]) - set(pid[tr]))})", flush=True)
for c, v in rec.items():
    F[c] = np.vstack(v)
MEAN = {c: np.nanmean(a, axis=0) for c, a in F.items()}
chk = float(np.abs(MEAN["r2_full"] - r2.r2_full.values).max()); assert chk < 1e-8

RULES = ["host", "tumour state", "host + state", "locus"]
CATS = ["host", "tumour state", "host + state", "own locus", "paralog locus", "residual", "partly explained", "poorly predicted"]


def attribute(st, hst, k=20, w="500kb", f=0.8):
    rules = {"host": "r2_host", "tumour state": st, "host + state": hst, "locus": f"r2_locus_state{k}_{w}"}
    refs = ["r2_host", st, hst, f"r2_locus_any_{w}", f"r2_locus_state{k}_{w}", f"r2_host_locus_state{k}_{w}"]
    names = ["r2_full"] + refs
    M = np.where(np.isnan(np.vstack([MEAN[c] for c in names])), -np.inf, np.vstack([MEAN[c] for c in names]))
    best = M.max(0); gap = MEAN["r2_full"] - M[1:].max(0)
    lab = np.array(["partly explained"] * T, dtype=object); done = best < 0.3; lab[done] = "poorly predicted"
    for cat in RULES:
        m = MEAN[rules[cat]]; hit = ~done & ~np.isnan(m) & (m >= f * best); lab[hit], done[hit] = cat, True
    own, par = MEAN[f"r2_locus_own_{w}"], MEAN[f"r2_locus_paralog_{w}"]
    is_par = ~np.isnan(par) & (np.isnan(own) | (par > np.nan_to_num(own, nan=-np.inf)))
    lab = np.where(lab == "locus", np.where(is_par, "paralog locus", "own locus"), lab)
    lab[~done & (gap >= 0.10)] = "residual"
    return pd.Series(lab, index=targets), pd.Series(best, index=targets)


base_lab, base_best = attribute("r2_state_20", "r2_host_state_20")
assert (base_lab == r2.label_r2).all() and np.allclose(base_best, r2.r2_best), "main labels not reproduced"
adj_lab, adj_best = attribute("r2_state20_pladj", "r2_host_state20_pladj")
cnt = lambda l: {c: int((l == c).sum()) for c in CATS}
print("main", cnt(base_lab)); print("plate-adjusted", cnt(adj_lab), "changed", int((adj_lab != base_lab).sum()))
print(pd.crosstab(base_lab, adj_lab).to_string())
ok_adj = (adj_lab != "poorly predicted").values
ad = pd.DataFrame({"r2_full": r2.r2_full, "r2_best": r2.r2_best, "r2_best_pladj": adj_best, "r2_state_20": r2.r2_state_20,
                   "r2_state20_pladj": MEAN["r2_state20_pladj"], "r2_host_state_20": r2.r2_host_state_20,
                   "r2_host_state20_pladj": MEAN["r2_host_state20_pladj"], "label_r2": base_lab, "label_r2_pladj": adj_lab,
                   "label_changed": base_lab != adj_lab})
ad["r2_state20_pladj_minus_state_20"] = ad.r2_state20_pladj - ad.r2_state_20
ad["share_state_20_over_r2_best"] = ad.r2_state_20 / ad.r2_best
ad["share_state20_pladj_over_r2_best"] = ad.r2_state20_pladj / ad.r2_best
ad["share_state20_pladj_over_r2_best_pladj"] = ad.r2_state20_pladj / ad.r2_best_pladj
ad["share_host_state_20_over_r2_best"] = ad.r2_host_state_20 / ad.r2_best
ad["share_host_state20_pladj_over_r2_best"] = ad.r2_host_state20_pladj / ad.r2_best
ad.to_csv(RESULTS / "audit_plate_adjusted_r2.csv", index_label="mirna")
med = lambda c, m: float(np.nanmedian(ad.loc[m, c]))
adjsum = {"n_labels_changed": int((adj_lab != base_lab).sum()), "counts_main": cnt(base_lab), "counts_plate_adjusted": cnt(adj_lab),
          "transitions": pd.crosstab(base_lab, adj_lab).to_dict(orient="index"),
          "median_share_state_20_over_r2_best": med("share_state_20_over_r2_best", ok),
          "median_share_state20_pladj_over_r2_best": med("share_state20_pladj_over_r2_best", ok),
          "median_share_state20_pladj_over_r2_best_pladj": med("share_state20_pladj_over_r2_best_pladj", ok_adj),
          "median_share_host_state_20_over_r2_best": med("share_host_state_20_over_r2_best", ok),
          "median_share_host_state20_pladj_over_r2_best": med("share_host_state20_pladj_over_r2_best", ok),
          "per_mirna_change_r2_state_20": {"min": float(ad.r2_state20_pladj_minus_state_20.min()), "median": float(ad.r2_state20_pladj_minus_state_20.median()),
                                           "max": float(ad.r2_state20_pladj_minus_state_20.max())},
          "n_not_poor_adjusted": int(ok_adj.sum()), "r2_full_check_max_abs": chk}
print(json.dumps({k: v for k, v in adjsum.items() if k != "transitions"}, indent=1))
json.dump({"confounding": conf, "plate_cv": cvsum, "plate_adjusted": adjsum,
           "method": "see 20_plate_cv_checks.py docstring", "runtime_min": round((time.time() - t0) / 60, 1)},
          open(CACHE / "meta_r2c.json", "w"), indent=2)
print(f"done {(time.time() - t0) / 60:.1f} min")
