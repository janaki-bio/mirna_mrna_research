"""Tumour-state reference model, programme scores, state labels, site-grouped CV and the low-expression miRNAs.

Same analysis set, folds (patient GroupKFold 5, seeds 42/43/44), ridge settings and covariates as 09_baseline_models.py.
State model: inside each training fold, genes are standardised, PCA is fitted on the training samples (via the n x n
Gram matrix) and test samples are projected; r2_state_k = RidgeCV on the top k PCs (k = 5, 10, 20, 50);
r2_host_state_k = RidgeCV on host gene(s) + top k PCs (= r2_state_k when there is no host); features are standardised
within folds. Also fits the baseline models for the four low-expression miRNAs and r2_full / r2_state_20 under
site-grouped CV (one repeat). The state labels are the label_v2 columns.
Programme scores (mean z-score of fixed marker-gene sets) are for interpretation only: Spearman correlation of each
miRNA, and of the top 10 PCs of the full analysis set, with each programme.
Reads audit_per_mirna.csv and audit_meta.json (09_), audit_hosts*.csv (06_, 07_), 10_e4_cross_mapped_all.csv (08_).
Writes audit_per_mirna_v2.csv, audit_low_expression.csv, audit_pc_programmes.csv, audit_state_curve.csv,
audit_programme_scores.csv, audit_mirna_programme_rho.csv, audit_meta_v2.json and cache/audit_v2_fold_records.pkl
(per-fold R2 arrays; delete to refit).

  python 10_state_models.py --probe   # time one fold of each loop and print a runtime estimate, write nothing"""
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from audit_common import ALPHAS, GENE_RULE, SEEDS, audit_data
from common import CACHE, RESULTS, gene_lookup, load_matrices

PROBE = "--probe" in sys.argv
KS = [5, 10, 20, 50]
LOW = ["hsa-mir-934", "hsa-mir-577", "hsa-mir-449a", "hsa-mir-449b"]
XMAP_FRAC = 0.5          # cross-mapped = Y in > 50% of samples
NEIGHBOUR_DELTA = 0.05
PROGRAMMES = {
    "proliferation": ["MKI67", "TOP2A", "CCNB1", "BUB1", "CDK1", "MCM2", "PCNA"],
    "T cell": ["CD3D", "CD3E", "CD2", "CD247"],
    "B/plasma": ["MS4A1", "CD19", "CD79A", "MZB1", "JCHAIN"],
    "myeloid": ["CD68", "CD163", "CSF1R", "CD14"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "endothelial": ["PECAM1", "VWF", "CDH5"],
    "ER/luminal": ["ESR1", "PGR", "FOXA1", "GATA3"],
    "basal": ["KRT5", "KRT14", "KRT17"],
    "HER2": ["ERBB2", "GRB7"],
    "hypoxia": ["CA9", "VEGFA", "ADM", "NDRG1"],
    "epithelial": ["EPCAM", "CDH1", "KRT8", "KRT18"],
    "ciliary": ["FOXJ1", "CDHR3", "DNAH5"],
}

t_start = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
v1 = pd.read_csv(RESULTS / "audit_per_mirna.csv").set_index("mirna")
hosts = pd.concat([pd.read_csv(RESULTS / "audit_hosts.csv"),
                   pd.read_csv(RESULTS / "audit_hosts_low_expression.csv")]).set_index("mirna")
assert list(v1.index) == targets and not set(LOW) & set(targets)
ALL = targets + LOW
lookup = gene_lookup().to_dict()

G = mrna_log[genes].values
Y = mirna_log[ALL].values
n, T = Y.shape
COV_PAM = pd.get_dummies(s.pam50, prefix="pam50", drop_first=True).astype(float).values  # reference: Basal
COV_PUR = s[["cpe", "lump"]].values.astype(float)
host_ids = {t: hosts.loc[t, "host_gene_ids"].split(";") if isinstance(hosts.loc[t, "host_gene_ids"], str) else []
            for t in ALL}
HOST = {t: mrna_log[host_ids[t]].values for t in ALL if host_ids[t]}  # used even if they fail the gene filter
gene_pos = {g: i for i, g in enumerate(genes)}
is_low = np.array([t in LOW for t in ALL])


def impute(train, test):
    mu = np.nanmean(train, axis=0)
    return np.where(np.isnan(train), mu, train), np.where(np.isnan(test), mu, test)


def lin_r2(Ftr, Fte, ytr, yte):
    P = LinearRegression().fit(Ftr, ytr).predict(Fte)
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)


def ridge_r2(Ftr, Fte, ytr, yte):
    sc = StandardScaler().fit(Ftr)
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=ytr.ndim > 1).fit(sc.transform(Ftr), ytr)
    P = m.predict(sc.transform(Fte))
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)


def kernel_ridgecv(K, Kte, y):
    """Dual form of sklearn RidgeCV (copied from 09_baseline_models.py, validated there)."""
    ym = y.mean()
    ev, Q = np.linalg.eigh(K)
    idim = np.argmax(np.abs(Q.T @ (np.ones(len(y)) / np.sqrt(len(y)))))
    QTy = Q.T @ (y - ym)
    best = (np.inf, None)
    for a in ALPHAS:
        w = 1.0 / (ev + a)
        w[idim] = 0.0
        c = Q @ (w * QTy)
        err = np.mean((c / ((Q ** 2) @ w)) ** 2)
        if err < best[0]:
            best = (err, c)
    return Kte @ best[1] + ym


def pca_scores(Xtr, Xte, kmax):
    """PCA of standardised (column-centred) training genes via the Gram matrix; returns train/test scores."""
    K = Xtr @ Xtr.T
    ev, U = np.linalg.eigh(K)
    o = np.argsort(ev)[::-1][:kmax]
    ev, U = ev[o], U[:, o]
    sv = np.sqrt(ev)
    return U * sv, (Xte @ Xtr.T) @ U / sv, K, ev


def state_fold(Xtr, Xte, tr, te, ks):
    """r2_state_k (all targets) and r2_host_state_k (hosted targets) for one fold."""
    Ptr, Pte, K, ev = pca_scores(Xtr, Xte, max(ks))
    out = {}
    for k in ks:
        out[f"r2_state_{k}"] = ridge_r2(Ptr[:, :k], Pte[:, :k], Y[tr], Y[te])
        hs = out[f"r2_state_{k}"].copy()
        for j, t in enumerate(ALL):
            if t in HOST:
                H = HOST[t]
                hs[j] = ridge_r2(np.hstack([H[tr], Ptr[:, :k]]), np.hstack([H[te], Pte[:, :k]]), Y[tr, j], Y[te, j])
        out[f"r2_host_state_{k}"] = hs
    return out, K


folds = []
for seed in SEEDS:
    folds += [(seed, k, tr, te) for k, (tr, te) in
              enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(G, groups=s.patient.values))]
site = s.index.str[5:7].values
site_folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEEDS[0]).split(G, groups=site))

# low-expression supplement: removal sets for r2_full_nohost
removal = {}
for t in LOW:
    ids = set(host_ids[t]) | {hosts.loc[t, "mir_gene_id"]} | set(str(hosts.loc[t, "neighbour_ids_50kb"]).split(";"))
    removal[t] = [gene_pos[g] for g in ids if g in gene_pos]

records = {}
def add(key, vals):
    records.setdefault(key, []).append(vals)


def patient_fold(seed, k, tr, te):
    sc = StandardScaler().fit(G[tr])
    Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
    add("r2_full", ridge_r2(G[tr], G[te], Y[tr], Y[te]))
    pur_tr, pur_te = impute(COV_PUR[tr], COV_PUR[te])
    cov_tr, cov_te = np.hstack([COV_PAM[tr], pur_tr]), np.hstack([COV_PAM[te], pur_te])
    # baseline-model columns for the 4 low-expression miRNAs only (the 337 targets have them in audit_per_mirna.csv)
    lo = np.where(is_low)[0]
    add("low_r2_all", ridge_r2(np.hstack([G[tr], cov_tr]), np.hstack([G[te], cov_te]), Y[tr][:, lo], Y[te][:, lo]))
    add("low_r2_pam50", lin_r2(COV_PAM[tr], COV_PAM[te], Y[tr][:, lo], Y[te][:, lo]))
    add("low_r2_purity", lin_r2(pur_tr, pur_te, Y[tr][:, lo], Y[te][:, lo]))
    add("low_r2_cov", lin_r2(cov_tr, cov_te, Y[tr][:, lo], Y[te][:, lo]))
    hh, hc, nh = [], [], []
    for j in lo:
        t, H = ALL[j], HOST[ALL[j]]
        hh.append(lin_r2(H[tr], H[te], Y[tr, j], Y[te, j]))
        hc.append(lin_r2(np.hstack([H[tr], cov_tr]), np.hstack([H[te], cov_te]), Y[tr, j], Y[te, j]))
    st, K = state_fold(Xtr, Xte, tr, te, KS)
    K1, Kte = K + 1.0, Xte @ Xtr.T
    for j in lo:
        R, Rte = Xtr[:, removal[ALL[j]]], Xte[:, removal[ALL[j]]]
        nh.append(r2_score(Y[te, j], kernel_ridgecv(K1 - R @ R.T, Kte - Rte @ R.T, Y[tr, j])))
    add("low_r2_host", np.array(hh)); add("low_r2_host_cov", np.array(hc)); add("low_r2_full_nohost", np.array(nh))
    for key, v in st.items():
        add(key, v)


def site_fold(tr, te):
    sc = StandardScaler().fit(G[tr])
    Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
    add("r2_full_site", ridge_r2(G[tr], G[te], Y[tr], Y[te]))
    Ptr, Pte, _, _ = pca_scores(Xtr, Xte, 20)
    add("r2_state20_site", ridge_r2(Ptr, Pte, Y[tr], Y[te]))


if PROBE:
    t0 = time.time(); patient_fold(*folds[0]); tp = time.time() - t0
    t0 = time.time(); site_fold(*site_folds[0]); ts = time.time() - t0
    est = (len(folds) * tp + len(site_folds) * ts) / 60
    print(f"load {(t0 - t_start) / 60 - tp / 60:.1f} min; patient fold {tp:.0f} s x {len(folds)}; "
          f"site fold {ts:.0f} s x {len(site_folds)}; estimate {est:.1f} min (+ load)")
    print("fold-0 r2_full vs v1 fold mean, first 5:", np.round(records["r2_full"][0][:5], 3), v1.r2_full.values[:5].round(3))
    sys.exit()

fold_cache = CACHE / "audit_v2_fold_records.pkl"   # per-fold R2 arrays; delete to refit
if fold_cache.exists():
    records = pd.read_pickle(fold_cache)
else:
    for f in folds:
        patient_fold(*f)
        print(f"seed {f[0]} fold {f[1]} done, {(time.time() - t_start) / 60:.1f} min", flush=True)
    for tr, te in site_folds:
        site_fold(tr, te)
    print(f"site folds done, {(time.time() - t_start) / 60:.1f} min", flush=True)
    pd.to_pickle(records, fold_cache)

res = pd.DataFrame(index=pd.Index(ALL, name="mirna"))
for key, vals in records.items():
    arr = np.vstack(vals).mean(axis=0)
    if key.startswith("low_"):
        res.loc[LOW, key[4:]] = arr
    else:
        res[key] = arr
res.loc[LOW, "r2_full_sd"] = np.vstack(records["r2_full"]).std(axis=0, ddof=1)[is_low]
check_full = float(np.abs(res.loc[targets, "r2_full"] - v1.r2_full).max())
print("max |r2_full - v1 r2_full| over 337 targets:", check_full)
assert check_full < 1e-8, "r2_full does not reproduce v1"

# ---- programme scores (analysis set; interpretation only) ----
missing = {p: [g for g in gs if g not in set(lookup.values())] for p, gs in PROGRAMMES.items()}
assert not any(missing.values()), missing
name_to_id = {}
for gid, name in lookup.items():
    if gid in mrna_log.columns and name not in name_to_id:
        name_to_id[name] = gid
dup = {g for p in PROGRAMMES.values() for g in p if sum(v == g for v in lookup.values()) > 1}
Z = (mrna_log - mrna_log.mean()) / mrna_log.std(ddof=0)
scores = pd.DataFrame({p: Z[[name_to_id[g] for g in gs]].mean(axis=1) for p, gs in PROGRAMMES.items()})
scores.to_csv(RESULTS / "audit_programme_scores.csv")
rho = pd.DataFrame({p: [spearmanr(mirna_log[t], scores[p])[0] for t in ALL] for p in PROGRAMMES}, index=ALL)
rho.to_csv(RESULTS / "audit_mirna_programme_rho.csv", index_label="mirna")


def top_progs(r, n_top):
    o = r.abs().sort_values(ascending=False).index[:n_top]
    return [(p, r[p]) for p in o]


for i in (1, 2):
    res[f"prog{i}"] = [top_progs(rho.loc[t], 2)[i - 1][0] for t in ALL]
    res[f"prog{i}_rho"] = [top_progs(rho.loc[t], 2)[i - 1][1] for t in ALL]

# top 10 PCs of the full analysis set (interpretation only; CV models use fold-specific PCs)
Xs = StandardScaler().fit_transform(G)
Ptr, _, _, ev = pca_scores(Xs, Xs[:1], 10)
pc_rows = []
for i in range(10):
    r = pd.Series({p: spearmanr(Ptr[:, i], scores[p])[0] for p in PROGRAMMES})
    row = {"pc": f"PC{i + 1}", "var_explained": ev[i] / (Xs ** 2).sum()}
    for j, (p, v) in enumerate(top_progs(r, 3), 1):
        row[f"prog{j}"], row[f"prog{j}_rho"] = p, v
    row["pam50_eta2"] = float(pd.Series(Ptr[:, i]).groupby(s.pam50.values).transform("mean").var(ddof=0) / Ptr[:, i].var())
    row["rho_cpe"] = spearmanr(Ptr[:, i], s.cpe)[0]
    pc_rows.append(row)
pc_tab = pd.DataFrame(pc_rows)
pc_tab.to_csv(RESULTS / "audit_pc_programmes.csv", index=False)

# ---- assemble per-miRNA table: baseline columns + state-model columns ----
rpm = mirna_rpm[LOW]
low_base = pd.DataFrame({
    "mir_gene": hosts.loc[LOW, "mir_gene"], "n_samples": n, "frac_zero": (rpm == 0).mean(),
    "frac_rpm_gt1": (rpm > 1).mean(), "mean_log2rpm": mirna_log[LOW].mean(),
    "paralog_group": hosts.loc[LOW, "paralog_group"], "cluster_id": hosts.loc[LOW, "cluster_members"],
    "host_genes": hosts.loc[LOW, "host_genes"], "host_type": hosts.loc[LOW, "host_type"],
    "neighbours_50kb": hosts.loc[LOW, "neighbours_50kb"]}, index=pd.Index(LOW, name="mirna"))
for c in ["r2_full", "r2_full_sd", "r2_host", "r2_pam50", "r2_purity", "r2_cov", "r2_host_cov", "r2_all", "r2_full_nohost"]:
    low_base[c] = res.loc[LOW, c]
low_base["delta_beyond"] = low_base.r2_all - low_base.r2_host_cov.fillna(low_base.r2_cov)
low_base[["top_pos_genes", "top_neg_genes"]] = ""


def label_v1(r):  # rules of 09_baseline_models.py
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


low_base["label"] = low_base.apply(label_v1, axis=1)
base = pd.concat([v1, low_base[v1.columns]]).rename(columns={"label": "label_v1"})
new_cols = [f"r2_state_{k}" for k in KS] + [f"r2_host_state_{k}" for k in KS]
out = base.join(res[new_cols + ["prog1", "prog1_rho", "prog2", "prog2_rho", "r2_full_site", "r2_state20_site"]])
out["delta_specific"] = out.r2_full - out.r2_host_state_20


def label_v2(r, k, detail=True):
    full = r.r2_full
    if full < 0.3:
        return "poorly predicted"
    if pd.notna(r.r2_host) and r.r2_host >= 0.8 * full:
        return "host"
    if r[f"r2_state_{k}"] >= 0.8 * full:
        return f"tumour state: {r.prog1} ({'+' if r.prog1_rho > 0 else '-'})" if detail else "tumour state"
    if r[f"r2_host_state_{k}"] >= 0.8 * full:
        return "host + state"
    if full - r[f"r2_host_state_{k}"] >= 0.10:
        return "specific beyond state"
    return "partly explained"


out["label_v2"] = out.apply(label_v2, axis=1, k=20)
for k in (10, 50):
    out[f"label_v2_k{k}"] = out.apply(label_v2, axis=1, k=k)
out["label_v2_class"] = out.apply(label_v2, axis=1, k=20, detail=False)

# ---- flags ----
xmap = pd.read_csv(RESULTS / "10_e4_cross_mapped_all.csv", index_col=0)["frac_samples_cross_mapped"]
out["frac_cross_mapped"] = xmap.reindex(out.index).values
raw_mrna, _ = load_matrices()
mean_tpm = raw_mrna.loc[s.index].mean()
gene_set = set(genes)
flags = []
for t, r in out.iterrows():
    f = []
    if t in LOW:
        f.append("low expression, zero-inflated")
    if r.frac_cross_mapped > XMAP_FRAC:
        f.append(f"cross-mapped reads ({r.frac_cross_mapped:.0%} of samples)")
    poor = [lookup.get(g, g) for g in host_ids[t] if g not in gene_set or mean_tpm[g] < 1]
    if poor:
        f.append(f"host poorly measured ({','.join(poor)})")
    if pd.notna(r.r2_host) and r.r2_host < 0:
        f.append("r2_host < 0")
    if pd.notna(r.r2_full_nohost) and r.r2_full - r.r2_full_nohost >= NEIGHBOUR_DELTA:
        hox = any(str(g).startswith("HOX") for g in str(r.neighbours_50kb).split(";") + str(r.host_genes).split(";"))
        f.append(f"{'HOX-locus ' if hox else ''}neighbour effect (r2_full - r2_full_nohost = {r.r2_full - r.r2_full_nohost:.2f})")
    if isinstance(r.paralog_group, str) and r.paralog_group:
        f.append(f"paralog group ({r.paralog_group})")
    flags.append("; ".join(f))
out["flags"] = flags

main = out.loc[targets]
low = out.loc[LOW]
main.reset_index().to_csv(RESULTS / "audit_per_mirna_v2.csv", index=False)
low.reset_index().to_csv(RESULTS / "audit_low_expression.csv", index=False)

# ---- state curve (r2_full >= 0.3) ----
sel = main[main.r2_full >= 0.3]
curve = pd.DataFrame([{
    "k": k, "n_mirnas": len(sel),
    "median_r2_state_over_r2_full": (sel[f"r2_state_{k}"] / sel.r2_full).median(),
    "median_r2_host_state_over_r2_full": (sel[f"r2_host_state_{k}"] / sel.r2_full).median(),
    "median_r2_state_all337": main[f"r2_state_{k}"].median(),
    "n_state_ge_0.8_full": int((sel[f"r2_state_{k}"] >= 0.8 * sel.r2_full).sum()),
    "n_host_state_ge_0.8_full": int((sel[f"r2_host_state_{k}"] >= 0.8 * sel.r2_full).sum()),
} for k in KS])
curve.to_csv(RESULTS / "audit_state_curve.csv", index=False)

counts = pd.DataFrame({
    f"k={k}": main.apply(label_v2, axis=1, k=k, detail=False).value_counts() for k in (10, 20, 50)}).fillna(0).astype(int)
counts.loc["total"] = counts.sum()

# ---- meta ----
v1_meta = json.load(open(RESULTS / "audit_meta.json"))
meta = {
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "commit": v1_meta["commit"], "working_tree": v1_meta["working_tree"],
    "version": "v2 (v1 = audit_per_mirna.csv / audit_meta.json, unchanged)",
    "rule_change": {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "what": "label rules v2 added alongside v1; 'explained by subtype/composition' (r2_cov, 7 covariates) replaced by "
                "'tumour state' (r2_state_20, top 20 PCs fitted within folds); 'host + covariates' -> 'host + state'; "
                "'signal beyond known factors' (delta_beyond) -> 'specific beyond state' (delta_specific)",
        "why": "7 covariates cannot stand in for overall tumour state, so v1 'signal beyond known factors' was a catch-all",
        "thresholds_unchanged": "0.3 (poorly predicted), 0.8 x r2_full (explained), 0.10 (beyond)"},
    "label_rules_v2": [
        "poorly predicted: r2_full < 0.3",
        "host: r2_host >= 0.8 * r2_full",
        "tumour state: r2_state_20 >= 0.8 * r2_full (suffix = programme with the largest |Spearman rho|, with sign)",
        "host + state: r2_host_state_20 >= 0.8 * r2_full",
        "specific beyond state: delta_specific >= 0.10",
        "partly explained: otherwise",
        "delta_specific = r2_full - r2_host_state_20; label_v2_k10 / label_v2_k50 use k = 10 / 50 instead of 20"],
    "label_counts_v2": {c: counts[c].drop("total").to_dict() for c in counts.columns},
    "label_counts_v2_detailed_k20": main.label_v2.value_counts().to_dict(),
    "label_counts_v1": main.label_v1.value_counts().to_dict(),
    "v1_to_v2_k20": pd.crosstab(main.label_v1, main.label_v2_class).to_dict(orient="index"),
    "state_model": {"ks": KS, "pca": "on StandardScaler-transformed training genes (27,671), fitted within each training fold; "
                    "test samples projected", "ridge": "RidgeCV(logspace(-2, 8, 41)) on PC scores (+ host genes), "
                    "standardised within folds; alpha_per_target for multi-target fits",
                    "check_r2_full_vs_v1_max_abs_diff": check_full},
    "state_curve_median_ratio_r2_full_ge_0.3": dict(zip(curve.k.astype(str), curve.median_r2_state_over_r2_full.round(3))),
    "site_cv": {"grouping": "tissue source site (sample ID characters 6-7)", "n_sites": int(pd.Series(site).nunique()),
                "folds": "GroupKFold(5, shuffle=True, random_state=42), one repeat",
                "fold_n_test": [int(len(te)) for _, te in site_folds],
                "mean_r2_full_site_minus_patient": float((main.r2_full_site - main.r2_full).mean()),
                "median_r2_full_site_minus_patient": float((main.r2_full_site - main.r2_full).median()),
                "mean_r2_state20_site_minus_patient": float((main.r2_state20_site - main.r2_state_20).mean()),
                "n_r2_full_gt_0.5": {"patient": int((main.r2_full > 0.5).sum()), "site": int((main.r2_full_site > 0.5).sum())},
                "note": "patient CV is 3 repeats, site CV is 1 repeat; site folds are unequal (one has 41 test samples), "
                        "so per-fold R2 is noisier; mean of per-fold R2 as in v1"},
    "programmes": {"definition": "mean of per-gene z-scores of log2(TPM+1) over the analysis set; all genes in the matrix "
                   "(not only filtered genes)", "sets": PROGRAMMES, "use": "interpretation only, not in any model",
                   "duplicate_symbols": sorted(dup)},
    "pc_programmes": "top 10 PCs of the full analysis set (not the fold-specific PCs); PC sign is arbitrary",
    "flags": {"cross-mapped": f"cross-mapped = Y in > {XMAP_FRAC:.0%} of the 1,087 raw files (E4, 10_e4_cross_mapped_all.csv)",
              "host poorly measured": "any host gene failing the gene filter (TPM > 0.5 in >= 10%) or with mean TPM < 1; gene names listed",
              "r2_host < 0": "host-only linear model worse than the mean",
              "neighbour effect": f"r2_full - r2_full_nohost >= {NEIGHBOUR_DELTA} ('HOX-locus' if a host or +-50 kb neighbour is a HOX gene)",
              "paralog group": "name differs only in numeric suffix (06_annotate_hosts.py rule)"},
    "flag_counts": {k: int(main["flags"].str.contains(k, regex=False).sum()) for k in
                    ["cross-mapped", "host poorly measured", "r2_host < 0", "neighbour effect", "paralog group"]},
    "notes": [
        f"host + PCs worse than PCs alone (r2_host_state_20 < r2_state_20 - 0.01) for "
        f"{int((main.r2_host_state_20 < main.r2_state_20 - 0.01).sum())} targets, hosts: "
        f"{', '.join(sorted(set(main.host_genes[main.r2_host_state_20 < main.r2_state_20 - 0.01])))} (the MEG8 outlier sample "
        "from v1). Specific labels lost if delta_specific used max(r2_host_state_20, r2_state_20): "
        + str(main.index[(main.label_v2_class == "specific beyond state")
                         & (main.r2_full - np.fmax(main.r2_host_state_20, main.r2_state_20) < 0.10)].tolist()),
        f"site CV: largest drops come from the 41-sample site fold (e.g. hsa-mir-628 r2_full_site "
        f"{main.loc['hsa-mir-628', 'r2_full_site']:.2f} vs {main.loc['hsa-mir-628', 'r2_full']:.2f}); "
        f"{int((main.r2_full_site - main.r2_full < -0.1).sum())} targets drop by > 0.1",
        "'host' labels are identical for k = 10, 20, 50 (the host rule is checked before the state rule)",
        "'specific beyond state' is still relative to 20 linear PCs; it does not establish regulation"],
    "low_expression_supplement": {"mirnas": LOW, "file": "audit_low_expression.csv",
                                  "hosts": "07_annotate_low_expression_hosts.py (06_annotate_hosts.py rules); cluster_id column holds cluster members"},
    "runtime_min": round((time.time() - t_start) / 60, 1),
    "runtime_note": "fold fits (15 patient + 5 site folds) took 6.8 min on the first full run and are cached in "
                    "cache/audit_v2_fold_records.pkl; runtime_min is this run (probe estimate was 5.8 min + load)",
}
json.dump(meta, open(RESULTS / "audit_meta_v2.json", "w"), indent=2, default=float)
print(counts.to_string())
print(curve.round(3).to_string())
print(pc_tab.round(2).to_string())
print(f"runtime {meta['runtime_min']} min")
