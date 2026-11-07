"""Baseline models for the 337 target miRNAs: full-transcriptome ridge, host gene, PAM50 / purity covariates.

Patient-grouped 5-fold CV (GroupKFold, shuffled), seeds 42/43/44 (15 test folds), on the analysis set. Ridge =
RidgeCV(alphas=logspace(-2, 8, 41)) with features standardised within the training fold; linear models =
LinearRegression. Covariates: PAM50 one-hot (drop first) + CPE + LUMP (LUMP mean-imputed within the training fold).
Targets are precursors with RPM > 1 in >= 50% of the analysis set; genes have TPM > 0.5 in >= 10% (audit_common.py).
r2_full_nohost removes the host genes, the MIR gene and all genes within +-50 kb. It uses an exact kernel (dual) form of
RidgeCV's GCV so the Gram matrix is shared across targets; the form is checked against sklearn RidgeCV
(audit_kernel_ridge_check.csv) before use.
Reads audit_hosts.csv (06_annotate_hosts.py), audit_sample_steps.json and external/tcgabiolinks/versions.txt.
Writes audit_per_mirna.csv (R2 of each model, covariate-based label, top coefficients) and audit_meta.json (settings,
counts, package versions)."""
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from audit_common import ALPHAS, GENE_RULE, SEEDS, TARGET_RULE, audit_data
from common import HERE, RESULTS, ROOT, gene_lookup

NOHOST_BUDGET_MIN = 30
t_start = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna").loc[targets]
lookup = gene_lookup()

G = mrna_log[genes].values
Y = mirna_log[targets].values
n, T = Y.shape
pam = pd.get_dummies(s.pam50, prefix="pam50", drop_first=True).astype(float)  # reference: Basal
COV_PAM = pam.values
COV_PUR = s[["cpe", "lump"]].values.astype(float)
groups = s.patient.values
host_ids = {t: hosts.loc[t, "host_gene_ids"].split(";") if isinstance(hosts.loc[t, "host_gene_ids"], str) else []
            for t in targets}
gene_pos = {g: i for i, g in enumerate(genes)}


def impute(train, test):
    """Mean-impute NaNs (LUMP) using training-fold means."""
    mu = np.nanmean(train, axis=0)
    return np.where(np.isnan(train), mu, train), np.where(np.isnan(test), mu, test)


def lin_r2(Ftr, Fte, ytr, yte):
    P = LinearRegression().fit(Ftr, ytr).predict(Fte)
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)


def ridge_multi(Ftr, Fte, ytr, yte):
    sc = StandardScaler().fit(Ftr)
    P = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Ftr), ytr).predict(sc.transform(Fte))
    return r2_score(yte, P, multioutput="raw_values")


def kernel_ridgecv(K, Kte, y):
    """Dual form of sklearn RidgeCV (gcv_mode='eigen', fit_intercept=True) for centred X: K = X X^T + 1 1^T,
    intercept direction unpenalised, alpha chosen by mean squared LOO error. Returns test predictions."""
    ym = y.mean()
    yc = y - ym
    ev, Q = np.linalg.eigh(K)
    idim = np.argmax(np.abs(Q.T @ (np.ones(len(y)) / np.sqrt(len(y)))))
    QTy = Q.T @ yc
    best = (np.inf, None)
    for a in ALPHAS:
        w = 1.0 / (ev + a)
        w[idim] = 0.0
        c = Q @ (w * QTy)
        gdiag = (Q ** 2) @ w
        err = np.mean((c / gdiag) ** 2)
        if err < best[0]:
            best = (err, c)
    return Kte @ best[1] + ym


repeats = []
records = {k: [] for k in ["r2_full", "r2_all", "r2_pam50", "r2_purity", "r2_cov", "r2_host", "r2_host_cov"]}
fold_list = []
for seed in SEEDS:
    fold_list += [(seed, k, tr, te) for k, (tr, te) in
                  enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(G, groups=groups))]

for seed, k, tr, te in fold_list:
    pur_tr, pur_te = impute(COV_PUR[tr], COV_PUR[te])
    cov_tr, cov_te = np.hstack([COV_PAM[tr], pur_tr]), np.hstack([COV_PAM[te], pur_te])
    records["r2_full"].append(ridge_multi(G[tr], G[te], Y[tr], Y[te]))
    records["r2_all"].append(ridge_multi(np.hstack([G[tr], cov_tr]), np.hstack([G[te], cov_te]), Y[tr], Y[te]))
    records["r2_pam50"].append(lin_r2(COV_PAM[tr], COV_PAM[te], Y[tr], Y[te]))
    records["r2_purity"].append(lin_r2(pur_tr, pur_te, Y[tr], Y[te]))
    records["r2_cov"].append(lin_r2(cov_tr, cov_te, Y[tr], Y[te]))
    h, hc = np.full(T, np.nan), np.full(T, np.nan)
    for j, t in enumerate(targets):
        if host_ids[t]:
            H = mrna_log[host_ids[t]].values  # host genes used even if they fail the gene filter
            h[j] = lin_r2(H[tr], H[te], Y[tr, j], Y[te, j])
            hc[j] = lin_r2(np.hstack([H[tr], cov_tr]), np.hstack([H[te], cov_te]), Y[tr, j], Y[te, j])
    records["r2_host"].append(h)
    records["r2_host_cov"].append(hc)
    print(f"seed {seed} fold {k} done, {(time.time() - t_start) / 60:.1f} min", flush=True)

res = pd.DataFrame(index=pd.Index(targets, name="mirna"))
for key, vals in records.items():
    arr = np.vstack(vals)
    res[key] = arr.mean(axis=0)
    if key == "r2_full":
        res["r2_full_sd"] = arr.std(axis=0, ddof=1)
res["delta_beyond"] = res.r2_all - res.r2_host_cov.fillna(res.r2_cov)

# r2_full_nohost: hosted targets with r2_full >= 0.3; remove host genes, MIR gene and +-50 kb neighbours
elig = [t for t in targets if host_ids[t] and res.loc[t, "r2_full"] >= 0.3]
removal = {}
for t in elig:
    ids = set(host_ids[t]) | {hosts.loc[t, "mir_gene_id"]} | set(str(hosts.loc[t, "neighbour_ids_50kb"]).split(";"))
    removal[t] = [gene_pos[g] for g in ids if g in gene_pos]

# validation of the kernel form against sklearn on fold 0 (with and without removal)
seed, k, tr, te = fold_list[0]
sc = StandardScaler().fit(G[tr])
Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
K, Kte = Xtr @ Xtr.T + 1.0, Xte @ Xtr.T + 0.0
check = []
for t in elig[:3]:
    j = targets.index(t)
    mask = np.ones(G.shape[1], bool)
    mask[removal[t]] = False
    R, Rte = Xtr[:, ~mask], Xte[:, ~mask]
    p_kernel = kernel_ridgecv(K - R @ R.T, Kte - Rte @ R.T, Y[tr, j])
    p_sk = RidgeCV(alphas=ALPHAS).fit(Xtr[:, mask], Y[tr, j]).predict(Xte[:, mask])
    check.append({"mirna": t, "n_removed": int((~mask).sum()), "r2_kernel": r2_score(Y[te, j], p_kernel),
                  "r2_sklearn": r2_score(Y[te, j], p_sk), "max_abs_pred_diff": float(np.abs(p_kernel - p_sk).max())})
check = pd.DataFrame(check)
print(check.to_string())
assert (check.max_abs_pred_diff < 1e-6).all(), "kernel ridge does not reproduce sklearn RidgeCV"

t0 = time.time()
j0 = targets.index(elig[0])
kernel_ridgecv(K, Kte, Y[tr, j0])
per_fit = time.time() - t0
gram_time = 3.0  # per fold, measured below
t0 = time.time()
_ = Xtr @ Xtr.T
gram_time = time.time() - t0
est_min = (len(fold_list) * (gram_time + len(elig) * per_fit * 1.2)) / 60
print(f"r2_full_nohost: {len(elig)} targets; estimate {est_min:.1f} min for {len(fold_list)} folds")
nohost_ran = est_min <= NOHOST_BUDGET_MIN
if nohost_ran:
    vals = []
    for seed, k, tr, te in fold_list:
        sc = StandardScaler().fit(G[tr])
        Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
        K, Kte = Xtr @ Xtr.T + 1.0, Xte @ Xtr.T
        row = {}
        for t in elig:
            j = targets.index(t)
            R, Rte = Xtr[:, removal[t]], Xte[:, removal[t]]
            row[t] = r2_score(Y[te, j], kernel_ridgecv(K - R @ R.T, Kte - Rte @ R.T, Y[tr, j]))
        vals.append(row)
    res["r2_full_nohost"] = pd.DataFrame(vals).mean()
else:
    res["r2_full_nohost"] = np.nan
    print("r2_full_nohost skipped: estimate over budget")


# labels (rules fixed in advance)
def label(r):
    if r.r2_full < 0.3:
        return "poorly predicted"
    if pd.notna(r.r2_host) and r.r2_host >= 0.8 * r.r2_full:
        return "explained by host"
    if r.r2_cov >= 0.8 * r.r2_full:
        return "explained by subtype/composition (" + ("subtype" if r.r2_pam50 > r.r2_purity else "composition") + ")"
    if pd.notna(r.r2_host_cov) and r.r2_host_cov >= 0.8 * r.r2_full:
        return "explained by host + covariates"
    if r.delta_beyond >= 0.10:
        return "signal beyond known factors"
    return "partly explained"


res["label"] = res.apply(label, axis=1)

# top coefficients from a full-data fit (unstable; descriptive only)
sc = StandardScaler().fit(G)
full = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(G), Y)
names = np.array([lookup.get(g, g) for g in genes])
top_pos, top_neg = [], []
for j in range(T):
    c = full.coef_[j]
    o = np.argsort(c)
    top_pos.append(";".join(names[o[::-1][:5]]))
    top_neg.append(";".join(names[o[:5]]))
res["top_pos_genes"], res["top_neg_genes"] = top_pos, top_neg

rpm = mirna_rpm[targets]
out = pd.DataFrame({
    "mirna": targets, "mir_gene": hosts.mir_gene.values, "n_samples": n,
    "frac_zero": (rpm == 0).mean().values, "frac_rpm_gt1": (rpm > 1).mean().values,
    "mean_log2rpm": mirna_log[targets].mean().values,
    "paralog_group": hosts.paralog_group.values, "cluster_id": hosts.cluster_id.values,
    "host_genes": hosts.host_genes.fillna("").values, "host_type": hosts.host_type.values,
    "neighbours_50kb": hosts.neighbours_50kb.fillna("").values,
}).set_index("mirna").join(res[["r2_full", "r2_full_sd", "r2_host", "r2_pam50", "r2_purity", "r2_cov", "r2_host_cov",
                                 "r2_all", "delta_beyond", "r2_full_nohost", "label", "top_pos_genes", "top_neg_genes"]])
out.reset_index().to_csv(RESULTS / "audit_per_mirna.csv", index=False)
check.to_csv(RESULTS / "audit_kernel_ridge_check.csv", index=False)

# meta
steps = json.load(open(RESULTS / "audit_sample_steps.json"))
r_versions = (HERE / "external/tcgabiolinks/versions.txt").read_text().splitlines()
head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
headline = ["hsa-mir-934", "hsa-mir-577", "hsa-mir-449a", "hsa-mir-449b", "hsa-mir-150", "hsa-mir-18a"]
meta = {
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "commit": head, "working_tree": "review_checks/ untracked (audit code not committed)" if dirty else "clean",
    "sample_counts": steps,
    "pam50_counts": s.pam50.value_counts().to_dict(),
    "n_genes": len(genes), "n_targets": T,
    "filters": {"targets": f"miRNA precursors with RPM > {TARGET_RULE['rpm_gt']} in >= {TARGET_RULE['min_frac']:.0%} of analysis-set samples; log2(RPM+1)",
                "genes": f"TPM > {GENE_RULE['tpm_gt']} in >= {GENE_RULE['min_frac']:.0%} of analysis-set samples; log2(TPM+1)",
                "host_genes": "used for r2_host / r2_host_cov even if they fail the gene filter"},
    "headline_mirnas_in_target_set": {m: m in targets for m in headline},
    "cv": {"scheme": "GroupKFold(5, shuffle=True) by patient", "seeds": SEEDS, "n_test_folds": len(fold_list),
           "r2": "mean of per-fold test R2; r2_full_sd = SD over test folds"},
    "ridge": {"model": "RidgeCV, alpha_per_target for multi-target fits, GCV (eigen)", "alphas": "logspace(-2, 8, 41)",
              "alpha_min": float(ALPHAS[0]), "alpha_max": float(ALPHAS[-1]),
              "standardisation": "StandardScaler fitted on training folds (genes and, for r2_all, covariates)"},
    "covariates": {"pam50": "one-hot, drop_first (reference Basal)", "purity": "CPE and LUMP (TCGAbiolinks Tumor.purity)",
                   "lump_missing_in_analysis_set": steps["analysis_set_lump_missing"],
                   "lump_imputation": "training-fold mean"},
    "r2_full_nohost": {"eligible": len(elig), "ran": bool(nohost_ran), "estimate_min": round(est_min, 1),
                       "removed": "host genes + MIR gene + all genes within +-50 kb (GENCODE v36)",
                       "method": "exact dual/kernel form of RidgeCV GCV, validated against sklearn (audit_kernel_ridge_check.csv)"},
    "label_rules": [
        "poorly predicted: r2_full < 0.3",
        "explained by host: r2_host >= 0.8 * r2_full",
        "explained by subtype/composition: r2_cov >= 0.8 * r2_full (sub-label subtype if r2_pam50 > r2_purity, else composition)",
        "explained by host + covariates: r2_host_cov >= 0.8 * r2_full",
        "signal beyond known factors: delta_beyond >= 0.10",
        "partly explained: otherwise",
        "delta_beyond = r2_all - r2_host_cov (r2_cov when no host)"],
    "label_counts": out.label.value_counts().to_dict(),
    "notes": ["top_pos_genes/top_neg_genes: 5 largest positive/negative standardised ridge coefficients from one full-data fit; "
              "unstable across resamples (see E8) and descriptive only",
              f"r2_all vs r2_full: max |difference| {np.abs(out.r2_all - out.r2_full).max():.4f}; the 7 covariates add "
              "essentially nothing once all genes are in the ridge",
              f"no miRNA is 'explained by subtype/composition': max r2_cov / r2_full among r2_full >= 0.3 is "
              f"{(out.r2_cov / out.r2_full)[out.r2_full >= 0.3].max():.2f}",
              f"r2_host < -0.5 for {int((out.r2_host < -0.5).sum())} targets (hosts: "
              f"{', '.join(sorted(set(out.host_genes[out.r2_host < -0.5])))}): a nearly unexpressed host with one extreme "
              "sample dominates a test fold; r2_host_cov < r2_cov for "
              f"{int((out.r2_host_cov < out.r2_cov).sum())} targets. Labels changed if delta_beyond used "
              "max(r2_host_cov, r2_cov) instead: "
              f"{int(((out.label == 'signal beyond known factors') & (out.r2_all - np.fmax(out.r2_host_cov.fillna(-9), out.r2_cov) < 0.10)).sum())}",
              "'signal beyond known factors' compares a genome-wide ridge with a model of host gene(s) + 7 covariates; "
              "unmeasured composition (immune/stromal cell types), batch and other programmes all count as 'beyond'",
              "unmapped targets (no MIR gene in GENCODE v36): " + ", ".join(hosts.index[hosts.mir_gene.isna()])],
    "packages": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "R/TCGAbiolinks": r_versions[:2]},
    "runtime_min": round((time.time() - t_start) / 60, 1),
}
json.dump(meta, open(RESULTS / "audit_meta.json", "w"), indent=2)
print(json.dumps(meta["label_counts"], indent=2), f"runtime {meta['runtime_min']} min")
