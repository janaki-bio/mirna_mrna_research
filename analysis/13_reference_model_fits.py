"""Per-fold fits of every reference model for all 337 targets (the ones with r2_full < 0.3 included).

Host, own-locus, paralog-locus, locus + components and host + locus + components models at +-250 kb / +-500 kb / +-1 Mb.
Same analysis set, 15 patient-grouped folds (seeds 42-44), RidgeCV settings, host / locus definitions and PCA as
12_robustness_fits.py (code copied from it), with two differences:
  * a host model is fitted only if at least one host gene passes the expression filter (TPM > 0.5 in >= 10% of tumours,
    i.e. the gene is in the 27,671-gene set); otherwise every host-based model is treated as 'no host';
  * own-locus and paralog-locus models (locus genes only) are fitted at each window for every target.
Per-fold r2_full, r2_state_k, r2_host_state_k come from cache/audit_v2_fold_records.pkl (first 337 columns).

Step 0 (always run first): re-derive the label_v3 labels (no host filter) from the cached fold records of the 270
miRNAs with r2_full >= 0.3 and assert the counts host 26 / tumour state 38 / host + state 78 / locus 57 / residual 47 /
partly explained 24 / poorly predicted 67. After the fits, every refitted column is checked against the cached values
(assert, 1e-8).

  python 13_reference_model_fits.py --probe   # step 0 + time one fold, print an estimate, write nothing
Writes cache/audit_r2_fold_records.pkl (per-fold R2 arrays, 15 x 337 per column)."""
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from audit_common import ALPHAS, SEEDS, audit_data
from common import CACHE, RESULTS

PROBE = "--probe" in sys.argv
BASE_K, BASE_WIN = 20, 500_000
KS = [10, 20, 50]
WINS = [250_000, 500_000, 1_000_000]
t_start = time.time()

s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
v3 = pd.read_csv(RESULTS / "audit_per_mirna_v3.csv").set_index("mirna")
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna")
gtf_genes, _ = pd.read_pickle(CACHE / "gencode_v36_genes_exons.pkl")
mir_genes = gtf_genes[gtf_genes.gene_type == "miRNA"].drop_duplicates("gene_name").set_index("gene_name")
mapping = pd.read_csv(RESULTS / "audit_mir_mapping.csv").set_index("mirna")["mir_gene"]
assert list(v3.index) == targets and len(targets) == 337
T = len(targets)
G = mrna_log[genes].values
gene_pos = {g: i for i, g in enumerate(genes)}
host_ids = {t: hosts.loc[t, "host_gene_ids"].split(";") if isinstance(hosts.loc[t, "host_gene_ids"], str) else []
            for t in targets}
measurable = {t: any(h in gene_pos for h in host_ids[t]) for t in targets}
HOSTF = {t: mrna_log[host_ids[t]].values for t in targets if measurable[t]}   # all host genes of each target
print(f"targets with host genes: {sum(bool(v) for v in host_ids.values())}; host measurable: {sum(measurable.values())}; "
      f"host present but not measurable: {sum(bool(host_ids[t]) and not measurable[t] for t in targets)}")

# ---------------- step 0: reproduce the label_v3 labels from the cached fold records ----------------
v2rec = pd.read_pickle(CACHE / "audit_v2_fold_records.pkl")
oldrec = pd.read_pickle(CACHE / "audit_robustness_fold_records.pkl")
SEL = [t for t in targets if v3.loc[t, "r2_full"] >= 0.3]
sel_idx = np.array([targets.index(t) for t in SEL])
M0 = pd.DataFrame({c: np.vstack(v).mean(axis=0) for c, v in oldrec.items()}, index=SEL)
for c in ["r2_full", "r2_state_20", "r2_host_state_20"]:
    M0[c] = np.vstack(v2rec[c])[:, sel_idx].mean(axis=0)
lab = pd.Series("partly explained", index=SEL)
done = pd.Series(False, index=SEL)
for name, col in [("host", "r2_host"), ("tumour state", "r2_state_20"), ("host + state", "r2_host_state_20"),
                  ("locus", "r2_locus_state20_500kb")]:
    hit = ~done & M0[col].notna() & (M0[col] >= 0.8 * M0.r2_full)
    lab[hit] = name
    done |= hit
refs = M0[["r2_host", "r2_state_20", "r2_host_state_20", "r2_locus_any_500kb", "r2_locus_state20_500kb",
           "r2_host_locus_state20_500kb"]]
lab[~done & (M0.r2_full - refs.max(axis=1) >= 0.10)] = "residual"
step0 = lab.value_counts().to_dict() | {"poorly predicted": T - len(SEL)}
print("step 0, reproduced current counts:", step0)
expect = {"host": 26, "tumour state": 38, "host + state": 78, "locus": 57, "residual": 47, "partly explained": 24,
          "poorly predicted": 67}
assert step0 == expect, (step0, expect)
assert (lab == v3.loc[SEL, "label_v3_class"].replace({"unexplained": "residual"})).all()
print("step 0 OK: counts and per-miRNA labels match label_v3_class")

# ---------------- locus definitions for all targets ----------------
coords = gtf_genes[gtf_genes.gene_id.isin(gene_pos)]


def window_genes(chrom, start, end, win):
    w = coords[(coords.chrom == chrom) & (coords.start <= end + win) & (coords.end >= start - win)]
    return list(w.gene_id)


def precursor_pos(m):
    g = mapping.get(m)
    if not isinstance(g, str):
        return None
    r = mir_genes.loc[g]
    return r.chrom, r.start, r.end


def paralog_base(name):  # 06_annotate_hosts.py rule
    mm = re.match(r"^(hsa-(?:mir|let)-[0-9a-z]+)-(\d+)$", name)
    return mm.group(1) if mm else None


pb = pd.Series({m: paralog_base(m) for m in mapping.index}).dropna()
pb = pb[pb.duplicated(keep=False)]
group_members = pb.groupby(pb).apply(lambda x: list(x.index))
loc = {w: {} for w in WINS}   # loc[w][t] = {"own", "par", "any"} gene-id lists
for w in WINS:
    for t in targets:
        pos = precursor_pos(t)
        own = window_genes(*pos, w) if pos else []
        grp = pb.get(t)
        others = [m for m in group_members[grp] if m != t] if isinstance(grp, str) else []
        par = sorted({g for m in others if precursor_pos(m) for g in window_genes(*precursor_pos(m), w)})
        loc[w][t] = {"own": own, "par": par, "any": sorted(set(own) | set(par))}

# genes removed before PCA (as in 12_robustness_fits.py): host genes of all 337 targets + genes within +-500 kb
host_all = sorted({h for t in targets for h in host_ids[t]})
win_all = sorted({g for t in targets if precursor_pos(t) for g in window_genes(*precursor_pos(t), BASE_WIN)})
excl = (set(host_all) & set(gene_pos)) | set(win_all)
keep_cols = np.array([g not in excl for g in genes])
assert int((~keep_cols).sum()) == 4080, int((~keep_cols).sum())


def ridge_r2(Ftr, Fte, ytr, yte):
    sc = StandardScaler().fit(Ftr)
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=ytr.ndim > 1).fit(sc.transform(Ftr), ytr)
    P = m.predict(sc.transform(Fte))
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)


def lin_r2(Ftr, Fte, ytr, yte):
    return r2_score(yte, LinearRegression().fit(Ftr, ytr).predict(Fte))


def pca_scores(Xtr, Xte, kmax):  # as in 10_state_models.py
    ev, U = np.linalg.eigh(Xtr @ Xtr.T)
    o = np.argsort(ev)[::-1][:kmax]
    ev, U = ev[o], U[:, o]
    sv = np.sqrt(ev)
    return U * sv, (Xte @ Xtr.T) @ U / sv


def wname(w):
    return f"{w // 1000}kb" if w < 1_000_000 else f"{w // 1_000_000}Mb"


def lf(t, w, kind="any"):
    return G[:, [gene_pos[g] for g in loc[w][t][kind]]]


def hlf(t, w):   # locus (any) + host genes not already in it; host only if measurable
    ids = list(loc[w][t]["any"])
    if measurable[t]:
        ids += [h for h in host_ids[t] if h not in set(ids)]
    return mrna_log[ids].values if ids else np.empty((len(mrna_log), 0))


folds = []
for seed in SEEDS:
    folds += list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(G, groups=s.patient.values))
Y = mirna_log[targets].values

COLS = (["r2_state_20_check", "r2_host", "r2_state20_excl", "r2_host_state20_excl"]
        + [f"r2_locus_{kind}_{wname(w)}" for kind in ("own", "paralog", "any") for w in WINS]
        + [f"r2_locus_state{k}_{wname(w)}" for k in KS for w in WINS if k == BASE_K or w == BASE_WIN]
        + [f"r2_host_locus_state{k}_{wname(w)}" for k in KS for w in WINS if k == BASE_K or w == BASE_WIN])
rec = {c: [] for c in COLS}


def run_fold(tr, te):
    sc = StandardScaler().fit(G[tr])
    Xtr, Xte = sc.transform(G[tr]), sc.transform(G[te])
    Ptr, Pte = pca_scores(Xtr, Xte, max(KS))
    Etr, Ete = pca_scores(Xtr[:, keep_cols], Xte[:, keep_cols], BASE_K)
    del Xtr, Xte
    rec["r2_state_20_check"].append(ridge_r2(Ptr[:, :20], Pte[:, :20], Y[tr], Y[te]))
    se = ridge_r2(Etr, Ete, Y[tr], Y[te])
    row = {c: np.full(T, np.nan) for c in COLS[1:]}
    row["r2_state20_excl"] = se
    hse = se.copy()
    for j, t in enumerate(targets):
        ytr, yte = Y[tr, j], Y[te, j]
        if measurable[t]:
            H = HOSTF[t]
            row["r2_host"][j] = lin_r2(H[tr], H[te], ytr, yte)
            hse[j] = ridge_r2(np.hstack([H[tr], Etr]), np.hstack([H[te], Ete]), ytr, yte)
        for w in WINS:
            ks = KS if w == BASE_WIN else [BASE_K]
            for kind, c in (("own", "own"), ("par", "paralog")):
                if loc[w][t][kind]:
                    F = lf(t, w, kind)
                    row[f"r2_locus_{c}_{wname(w)}"][j] = ridge_r2(F[tr], F[te], ytr, yte)
            if loc[w][t]["any"]:
                F = lf(t, w)
                row[f"r2_locus_any_{wname(w)}"][j] = ridge_r2(F[tr], F[te], ytr, yte)
                for k in ks:
                    row[f"r2_locus_state{k}_{wname(w)}"][j] = ridge_r2(
                        np.hstack([F[tr], Ptr[:, :k]]), np.hstack([F[te], Pte[:, :k]]), ytr, yte)
            F = hlf(t, w)
            if F.shape[1]:
                for k in ks:
                    row[f"r2_host_locus_state{k}_{wname(w)}"][j] = ridge_r2(
                        np.hstack([F[tr], Ptr[:, :k]]), np.hstack([F[te], Pte[:, :k]]), ytr, yte)
    row["r2_host_state20_excl"] = hse
    for c in COLS[1:]:
        rec[c].append(row[c])


load_min = (time.time() - t_start) / 60
if PROBE:
    t0 = time.time()
    run_fold(*folds[0])
    tf = time.time() - t0
    print(f"load {load_min:.1f} min; fold {tf:.0f} s x {len(folds)} = {tf * len(folds) / 60:.1f} min")
    sys.exit()

for i, (tr, te) in enumerate(folds):
    run_fold(tr, te)
    print(f"fold {i} done, {(time.time() - t_start) / 60:.1f} min", flush=True)
fit_min = round((time.time() - t_start) / 60 - load_min, 1)

# ---------------- assemble per-fold arrays (15 x 337) and validate against the cached R2 arrays ----------------
F = {c: np.vstack(v) for c, v in rec.items()}
for c in ["r2_full"] + [f"r2_state_{k}" for k in KS] + [f"r2_host_state_{k}" for k in KS]:
    F[c] = np.vstack(v2rec[c])[:, :T]
no_host_measure = np.array([not measurable[t] for t in targets])
for k in KS:   # host + state with no measurable host = state alone (same convention as 'no host' in 10_state_models.py)
    F[f"r2_host_state_{k}"][:, no_host_measure] = F[f"r2_state_{k}"][:, no_host_measure]
ok = ~no_host_measure[sel_idx]
checks = {}
for c, a in oldrec.items():
    a = np.vstack(a)
    new = F[c][:, sel_idx]
    comparable = np.ones(len(SEL), bool)
    if c in ("r2_host", "r2_host_state20_excl") or "host_locus" in c:
        comparable = ok       # host-based columns differ by design where no host gene passes the filter
    d = np.abs(new[:, comparable] - a[:, comparable])
    checks[c] = float(np.nanmax(d))
    assert (np.isnan(new[:, comparable]) == np.isnan(a[:, comparable])).all(), c
print(json.dumps(checks, indent=1))
assert max(checks.values()) < 1e-8, "refits do not reproduce the old caches"
Mn = {c: np.nanmean(a, axis=0) if not np.isnan(a).all() else a[0] for c, a in F.items()}
d_own = np.nanmax(np.abs(np.where(np.isnan(v3.loc[SEL, "r2_locus_own"].values), 0, v3.loc[SEL, "r2_locus_own"].values)
                         - np.nan_to_num(F["r2_locus_own_500kb"][:, sel_idx].mean(axis=0))))
d_par = np.nanmax(np.abs(np.nan_to_num(v3.loc[SEL, "r2_locus_paralog"].values)
                         - np.nan_to_num(F["r2_locus_paralog_500kb"][:, sel_idx].mean(axis=0))))
d_host = float(np.nanmax(np.abs(v3.loc[SEL, "r2_host"].values[ok] - F["r2_host"][:, sel_idx].mean(axis=0)[ok])))
print("own/paralog locus vs v3 (500 kb):", d_own, d_par, " r2_host vs v3 (host measurable):", d_host)
assert max(d_own, d_par, d_host) < 1e-8
checks.update({"r2_locus_own_500kb_vs_v3": float(d_own), "r2_locus_paralog_500kb_vs_v3": float(d_par), "r2_host_vs_v3": d_host})
pd.to_pickle({"F": F, "targets": targets, "measurable": measurable, "checks": checks, "step0": step0,
              "n_pc_genes_removed": int((~keep_cols).sum()), "fit_min": fit_min, "load_min": round(load_min, 1)},
             CACHE / "audit_r2_fold_records.pkl")
print(f"done: fits {fit_min} min")
