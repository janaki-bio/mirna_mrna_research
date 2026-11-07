"""Sensitivity of the attribution to settings, components without host / local genes, and fold-level uncertainty.

Same analysis set, 15 patient-grouped folds (seeds 42-44), RidgeCV settings, host / locus feature definitions and PCA as
09_-11_ (helper code copied from 11_locus_models.py). Settings: k = 10 / 20 / 50 components, locus window +-250 kb /
+-500 kb / +-1 Mb, components computed after removing host genes and genes within +-500 kb of any target.
Attribution rule (r2_full >= 0.3; first match wins), applied with the setting's k / window:
  host (r2_host >= 0.8 r2_full) -> tumour state (r2_state_k) -> host + state (r2_host_state_k)
  -> locus (r2_locus_state_k) -> residual (r2_full - max(refs) >= 0.10) -> partly explained;
  refs = r2_host, r2_state_k, r2_host_state_k, r2_locus_any, r2_locus_state_k, r2_host_locus_state_k.
Uncertainty: Nadeau-Bengio corrected resampled t on the per-fold differences that decide each label.
Per-fold r2_full / r2_state_k / r2_host_state_k come from cache/audit_v2_fold_records.pkl (r2_state_20 is recomputed as a
check of the PCs); r2_host (LinearRegression on host genes) and all locus models are refitted.
Reads audit_per_mirna_v3.csv (11_) and the files read by 11_.
Writes audit_sensitivity_settings.csv, audit_label_stability.csv, audit_pc_exclusion.csv, audit_uncertainty.csv,
audit_meta_robustness.json and cache/audit_robustness_fold_records.pkl (per-fold R2 arrays; delete to refit).

  python 12_robustness_fits.py --probe   # time one fold and print an estimate, write nothing"""
import json
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
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
EXCL_WIN = 500_000
TEST_TRAIN_RATIO = 0.25   # Nadeau-Bengio correction, 5-fold CV
t_start = time.time()

s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
v3 = pd.read_csv(RESULTS / "audit_per_mirna_v3.csv").set_index("mirna")
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna")
gtf_genes, _ = pd.read_pickle(CACHE / "gencode_v36_genes_exons.pkl")
mir_genes = gtf_genes[gtf_genes.gene_type == "miRNA"].drop_duplicates("gene_name").set_index("gene_name")
mapping = pd.read_csv(RESULTS / "audit_mir_mapping.csv").set_index("mirna")["mir_gene"]
assert list(v3.index) == targets
SEL = [t for t in targets if v3.loc[t, "r2_full"] >= 0.3]
sel_idx = np.array([targets.index(t) for t in SEL])

G = mrna_log[genes].values
gene_pos = {g: i for i, g in enumerate(genes)}
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
host_ids = {t: hosts.loc[t, "host_gene_ids"].split(";") if isinstance(hosts.loc[t, "host_gene_ids"], str) else []
            for t in targets}

loc = {w: {} for w in WINS}   # loc[win][mirna] = locus gene ids (own + paralog windows, filtered genes)
for w in WINS:
    for t in SEL:
        pos = precursor_pos(t)
        own = window_genes(*pos, w) if pos else []
        grp = pb.get(t)
        others = [m for m in group_members[grp] if m != t] if isinstance(grp, str) else []
        par = {g for m in others if precursor_pos(m) for g in window_genes(*precursor_pos(m), w)}
        loc[w][t] = sorted(set(own) | par)

# ---- genes removed before PCA: all host genes of the 337 targets + genes within +-500 kb of any target ----
host_all = sorted({h for t in targets for h in host_ids[t]})
win_all = sorted({g for t in targets if precursor_pos(t) for g in window_genes(*precursor_pos(t), EXCL_WIN)})
excl = (set(host_all) & set(gene_pos)) | set(win_all)
keep_cols = np.array([g not in excl for g in genes])
unmapped_targets = [t for t in targets if precursor_pos(t) is None]


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


folds = []
for seed in SEEDS:
    folds += list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(G, groups=s.patient.values))

Y = mirna_log[SEL].values
HOSTF = {t: mrna_log[host_ids[t]].values for t in SEL if host_ids[t]}  # host genes may fail the filter


def wname(w):
    return f"{w // 1000}kb" if w < 1_000_000 else f"{w // 1_000_000}Mb"


def lf(t, w):
    return G[:, [gene_pos[g] for g in loc[w][t]]]


def hlf(t, w):  # locus + host genes not already in the locus (11_locus_models.py 'host_any')
    ids = loc[w][t] + [h for h in host_ids[t] if h not in set(loc[w][t])]
    return mrna_log[ids].values


COLS = (["r2_state_20_check", "r2_host", "r2_state20_excl", "r2_host_state20_excl"]
        + [f"r2_locus_any_{wname(w)}" for w in WINS]
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
    row = {c: np.full(len(SEL), np.nan) for c in COLS[1:]}
    row["r2_state20_excl"] = se
    hse = se.copy()
    for j, t in enumerate(SEL):
        ytr, yte = Y[tr, j], Y[te, j]
        if t in HOSTF:
            H = HOSTF[t]
            row["r2_host"][j] = lin_r2(H[tr], H[te], ytr, yte)
            hse[j] = ridge_r2(np.hstack([H[tr], Etr]), np.hstack([H[te], Ete]), ytr, yte)
        for w in WINS:
            ks = KS if w == BASE_WIN else [BASE_K]
            if loc[w][t]:
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
    print(f"load {load_min:.1f} min; {len(SEL)} miRNAs; genes removed for PCA {int((~keep_cols).sum())} of {len(genes)}; "
          f"locus 'any' median size " + ", ".join(f"{wname(w)} {np.median([len(v) for v in loc[w].values()]):.0f}"
                                                for w in WINS)
          + f"; fold {tf:.0f} s x {len(folds)} = {tf * len(folds) / 60:.1f} min")
    sys.exit()

fold_cache = CACHE / "audit_robustness_fold_records.pkl"   # per-fold R2 arrays for SEL; delete to refit
if fold_cache.exists():
    rec = pd.read_pickle(fold_cache)
else:
    for i, (tr, te) in enumerate(folds):
        run_fold(tr, te)
        print(f"fold {i} done, {(time.time() - t_start) / 60:.1f} min", flush=True)
    pd.to_pickle(rec, fold_cache)
fit_min = round((time.time() - t_start) / 60 - load_min, 1)

# ---- per-fold matrices (15 x 270) ----
v2rec = pd.read_pickle(CACHE / "audit_v2_fold_records.pkl")  # columns = 337 targets + 4 low-expression miRNAs
F = {c: np.vstack(v) for c, v in rec.items()}
for c in ["r2_full"] + [f"r2_state_{k}" for k in KS] + [f"r2_host_state_{k}" for k in KS]:
    F[c] = np.vstack(v2rec[c])[:, sel_idx]
M = pd.DataFrame({c: a.mean(axis=0) for c, a in F.items()}, index=pd.Index(SEL, name="mirna"))
ref = v3.loc[SEL]
checks = {
    "r2_state_20_recomputed_vs_cache": float(np.abs(M.r2_state_20_check - M.r2_state_20).max()),
    "r2_state_20_cache_vs_v3": float(np.abs(M.r2_state_20 - ref.r2_state_20).max()),
    "r2_full_cache_vs_v3": float(np.abs(M.r2_full - ref.r2_full).max()),
    "r2_host_vs_v3": float(np.nanmax(np.abs(M.r2_host - ref.r2_host))),
    "r2_locus_any_500kb_vs_v3": float(np.nanmax(np.abs(M.r2_locus_any_500kb - ref.r2_locus_any))),
    "r2_locus_state20_500kb_vs_v3": float(np.nanmax(np.abs(M.r2_locus_state20_500kb - ref.r2_locus_state20))),
    "r2_host_locus_state20_500kb_vs_v3": float(np.nanmax(np.abs(M.r2_host_locus_state20_500kb - ref.r2_host_locus_state20))),
    "r2_host_state_10_vs_v3": float(np.abs(M.r2_host_state_10 - ref.r2_host_state_10).max()),
    "r2_host_state_50_vs_v3": float(np.abs(M.r2_host_state_50 - ref.r2_host_state_50).max()),
}
print(json.dumps(checks, indent=1))
assert max(checks.values()) < 1e-8, "refits do not reproduce v2/v3"
assert (M.r2_host.isna() == ref.r2_host.isna()).all()

CATS = ["host", "tumour state", "host + state", "locus", "residual", "partly explained"]
RULES = ["host", "tumour state", "host + state", "locus"]


def setting_cols(k=BASE_K, w=BASE_WIN, excl_pcs=False):
    """Model column names for each rule step (and the reference set) at a setting."""
    st, hst = (("r2_state20_excl", "r2_host_state20_excl") if excl_pcs else (f"r2_state_{k}", f"r2_host_state_{k}"))
    rules = {"host": "r2_host", "tumour state": st, "host + state": hst, "locus": f"r2_locus_state{k}_{wname(w)}"}
    refs = ["r2_host", st, hst, f"r2_locus_any_{wname(w)}", f"r2_locus_state{k}_{wname(w)}",
            f"r2_host_locus_state{k}_{wname(w)}"]
    return rules, refs


def attribute(D, rules, refs):
    """D: DataFrame of mean R2 (rows = miRNAs). Returns label, label-setting model and best reference model."""
    lab = pd.Series("partly explained", index=D.index)
    model = pd.Series("", index=D.index)
    done = pd.Series(False, index=D.index)
    for cat in RULES:
        hit = ~done & D[rules[cat]].notna() & (D[rules[cat]] >= 0.8 * D.r2_full)
        lab[hit], model[hit] = cat, rules[cat]
        done |= hit
    R = D[refs]
    best = R.idxmax(axis=1, skipna=True)
    gap = D.r2_full - R.max(axis=1, skipna=True)
    hit = ~done & (gap >= 0.10)
    lab[hit], model[hit] = "residual", "gap vs " + best[hit]
    return lab, model, best, gap


SETTINGS = {
    "baseline (k=20, +-500 kb)": dict(),
    "k=10": dict(k=10), "k=50": dict(k=50),
    "window +-250 kb": dict(w=250_000), "window +-1 Mb": dict(w=1_000_000),
}
labels = {}
for name, kw in SETTINGS.items():
    labels[name] = attribute(M, *setting_cols(**kw))[0]
assert (labels["baseline (k=20, +-500 kb)"] == ref.label_v3_class.replace({"unexplained": "residual"})).all(), \
    "baseline does not reproduce label_v3_class"
labels["PC exclusion (task 2)"] = attribute(M, *setting_cols(excl_pcs=True))[0]
current = labels["baseline (k=20, +-500 kb)"]
n_poor = int((v3.r2_full < 0.3).sum())

# ---- label stability across settings ----
sens = []
for name, lab in labels.items():
    kw = SETTINGS.get(name, {})
    row = {"setting": name, "k": kw.get("k", BASE_K), "window_kb": kw.get("w", BASE_WIN) // 1000,
           "pcs": "excluding host and +-500 kb target-locus genes" if "exclusion" in name else "all filtered genes"}
    row |= {c: int((lab == c).sum()) for c in CATS} | {"poorly predicted": n_poor}
    row["n_changed_vs_current"] = int((lab != current).sum())
    sens.append(row)
sens = pd.DataFrame(sens)
sens.to_csv(RESULTS / "audit_sensitivity_settings.csv", index=False)

T1 = [n for n in SETTINGS if not n.startswith("baseline")]
stab = pd.DataFrame({"r2_full": M.r2_full, "label_current": current})
for n in T1:
    stab[f"label_{n.replace('window ', 'win').replace(' ', '').replace('+-', '')}"] = labels[n]
stab["share_settings_agree"] = np.mean([labels[n] == current for n in T1], axis=0)
stab["n_distinct_labels"] = pd.DataFrame({n: labels[n] for n in T1 + ["baseline (k=20, +-500 kb)"]}).nunique(axis=1)
stab["label_pc_exclusion"] = labels["PC exclusion (task 2)"]
stab.reset_index().to_csv(RESULTS / "audit_label_stability.csv", index=False)

# ---- components without host / local genes ----
pe = pd.DataFrame({
    "r2_full": M.r2_full, "label_current": current,
    "r2_state_20": M.r2_state_20, "r2_state20_excl": M.r2_state20_excl,
    "r2_host_state_20": M.r2_host_state_20, "r2_host_state20_excl": M.r2_host_state20_excl})
for a in ["r2_state_20", "r2_state20_excl", "r2_host_state_20", "r2_host_state20_excl"]:
    pe[f"share_{a}"] = pe[a] / pe.r2_full
pe["label_excl"] = labels["PC exclusion (task 2)"]
pe["label_changed"] = pe.label_excl != pe.label_current
pe.reset_index().to_csv(RESULTS / "audit_pc_exclusion.csv", index=False)

# ---- fold-level uncertainty (Nadeau-Bengio corrected resampled t) ----
J = len(folds)
tcrit = stats.t.ppf(0.975, J - 1)


def nb(d):
    """mean, corrected 95% CI of per-fold differences d (J,)"""
    m, sd = d.mean(), d.std(ddof=1)
    half = tcrit * np.sqrt((1 / J + TEST_TRAIN_RATIO) * sd ** 2)
    return m, m - half, m + half


rules0, refs0 = setting_cols()
_, model0, best0, gap0 = attribute(M, rules0, refs0)
rows = []
for j, t in enumerate(SEL):
    lab = current[t]
    full = F["r2_full"][:, j]
    r = {"mirna": t, "label_current": lab, "r2_full": M.r2_full[t]}
    # d for every explain rule (NaN if the model is undefined)
    dd = {c: F[rules0[c]][:, j] - 0.8 * full for c in RULES}
    bm = best0[t]
    gap = full - F[bm][:, j]
    r["best_ref_model"], r["gap_mean"] = bm, gap.mean()
    r["gap_share_folds_ge_0.10"] = float((gap >= 0.10).mean())
    _, r["gap_ci_lo"], r["gap_ci_hi"] = nb(gap)
    _, r["gap_minus_0.10_ci_lo"], r["gap_minus_0.10_ci_hi"] = nb(gap - 0.10)
    if lab in RULES:
        d = dd[lab]
        r["label_model"], r["test"] = rules0[lab], "d = R2_ref - 0.8 R2_full >= 0"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(d)
        r["share_folds_condition_holds"] = float((d >= 0).mean())
        crosses = r["d_ci_lo"] < 0
        earlier = RULES[:RULES.index(lab)]
    elif lab == "residual":
        r["label_model"], r["test"] = bm, "gap - 0.10 >= 0 (gap = R2_full - R2_best_ref)"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(gap - 0.10)
        r["share_folds_condition_holds"] = r["gap_share_folds_ge_0.10"]
        crosses = r["d_ci_lo"] < 0
        earlier = RULES
    else:  # partly explained: closest explain rule (largest mean d) and the residual rule both fail
        cand = {c: v for c, v in dd.items() if not np.isnan(v).any()}
        c = max(cand, key=lambda c: cand[c].mean())
        r["label_model"], r["test"] = rules0[c], f"closest failed rule '{c}': d < 0; and gap - 0.10 < 0"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(cand[c])
        r["share_folds_condition_holds"] = float(((cand[c] < 0) & (gap < 0.10)).mean())
        crosses = (r["d_ci_hi"] > 0) or (r["gap_minus_0.10_ci_hi"] > 0)
        earlier = []
    r["borderline"] = bool(crosses)
    r["earlier_rules_ci_reaches_0"] = ";".join(c for c in earlier if not np.isnan(dd[c]).any() and nb(dd[c])[2] >= 0)
    rows.append(r)
unc = pd.DataFrame(rows)
unc.to_csv(RESULTS / "audit_uncertainty.csv", index=False)
bl = unc.groupby("label_current").agg(n=("borderline", "size"), borderline=("borderline", "sum"),
                                      earlier_rule_reaches_0=("earlier_rules_ci_reaches_0", lambda x: int((x != "").sum())))
bl = bl.reindex(CATS)

# ---- meta ----
v3_meta = json.load(open(RESULTS / "audit_meta_v3.json"))
meta = {
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "commit": v3_meta["commit"], "working_tree": v3_meta["working_tree"],
    "version": "robustness of v3 (post hoc); v1-v3 outputs unchanged",
    "rule": ["r2_full >= 0.3 (270 miRNAs); first match wins",
             "host: r2_host >= 0.8 r2_full", "tumour state: r2_state_k >= 0.8 r2_full",
             "host + state: r2_host_state_k >= 0.8 r2_full", "locus: r2_locus_state_k (window w) >= 0.8 r2_full",
             "residual: r2_full - max(r2_host, r2_state_k, r2_host_state_k, r2_locus_any(w), r2_locus_state_k(w), "
             "r2_host_locus_state_k(w)) >= 0.10 (= v3 'unexplained')", "partly explained: otherwise"],
    "reproduction": {"label_v3_class_counts_reproduced": True, "checks_max_abs_diff": checks},
    "settings": {
        "baseline": {"k": BASE_K, "window": "+-500 kb"},
        "task1_k": "k = 10, 50 PCs (state, host + state, locus + PCs, host + locus + PCs); locus_any at +-500 kb",
        "task1_window": "+-250 kb, +-1 Mb (k = 20) for locus_any, locus + 20 PCs, host + locus + 20 PCs; window applied "
                        "to own precursor and paralog members as in v3",
        "locus_any_median_genes": {wname(w): float(np.median([len(loc[w][t]) for t in SEL])) for w in WINS},
        "task2": {"removed": "host genes of all 337 targets (those in the filtered gene set) + filtered genes within "
                             "+-500 kb of any of the 337 precursors",
                  "n_genes_removed": int((~keep_cols).sum()), "n_genes_kept": int(keep_cols.sum()),
                  "n_host_genes_all_targets": len(host_all), "n_host_genes_in_filtered_set": len(set(host_all) & set(gene_pos)),
                  "n_window_genes": len(win_all), "targets_without_position": unmapped_targets,
                  "refitted": "r2_state20_excl, r2_host_state20_excl (PCA within folds on the kept genes; host features "
                              "unchanged); locus models keep the standard 20 PCs",
                  "label_rule": "baseline rule with r2_state_20 / r2_host_state_20 replaced by the _excl versions "
                                "(also in the residual reference set)"},
        "task3": {"folds": J, "test_train_ratio": TEST_TRAIN_RATIO, "t_df": J - 1, "t_crit": float(tcrit),
                  "ci": "mean +- t * sqrt((1/J + n_test/n_train) * var(d))",
                  "d": "per fold, d = R2_label_model - 0.8 R2_full (host / tumour state / host + state / locus)",
                  "residual": "per fold gap = R2_full - R2_best_ref, best_ref = reference model with the largest mean R2; "
                              "condition gap >= 0.10, interval reported for gap and for gap - 0.10; borderline if the "
                              "interval of gap - 0.10 crosses 0",
                  "partly_explained": "d of the closest explain rule (largest mean d) and gap - 0.10; borderline if either "
                                      "upper limit > 0",
                  "earlier_rules_ci_reaches_0": "additional: earlier rules in the chain whose d interval reaches 0 (the "
                                                "label could be pre-empted); not part of the borderline flag"}},
    "summary": {
        "category_counts": sens.set_index("setting")[CATS + ["poorly predicted", "n_changed_vs_current"]].to_dict(orient="index"),
        "stability": {"all_4_settings_agree": int((stab.share_settings_agree == 1).sum()),
                      "share_agree_by_current_label": stab.groupby("label_current").share_settings_agree.mean().round(3).to_dict(),
                      "n_all_agree_by_current_label": stab[stab.share_settings_agree == 1].label_current.value_counts().to_dict()},
        "pc_exclusion": {"n_genes_removed": int((~keep_cols).sum()),
                         "median_share_r2_state_20": float(pe.share_r2_state_20.median()),
                         "median_share_r2_state20_excl": float(pe.share_r2_state20_excl.median()),
                         "median_share_r2_host_state_20": float(pe.share_r2_host_state_20.median()),
                         "median_share_r2_host_state20_excl": float(pe.share_r2_host_state20_excl.median()),
                         "n_labels_changed": int(pe.label_changed.sum()),
                         "transitions": pd.crosstab(pe.label_current, pe.label_excl).to_dict(orient="index")},
        "uncertainty": {"borderline_by_category": bl.borderline.astype(int).to_dict(), "n_by_category": bl.n.astype(int).to_dict(),
                        "earlier_rule_reaches_0_by_category": bl.earlier_rule_reaches_0.astype(int).to_dict(),
                        "n_borderline": int(unc.borderline.sum())},
        "transitions_k10": pd.crosstab(current, labels["k=10"]).to_dict(orient="index"),
        "transitions_k50": pd.crosstab(current, labels["k=50"]).to_dict(orient="index"),
    },
    "runtime_min": {"load": round(load_min, 1), "fits": fit_min, "total": round((time.time() - t_start) / 60, 1),
                    "note": "fits cached in cache/audit_robustness_fold_records.pkl; 'fits' is ~0 when the cache is reused"},
}
json.dump(meta, open(RESULTS / "audit_meta_robustness.json", "w"), indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
pd.set_option("display.width", 250)
print(sens.to_string(index=False))
print(pd.Series(meta["summary"]["pc_exclusion"]).drop("transitions").to_string())
print(pd.crosstab(pe.label_current, pe.label_excl))
print(bl.to_string())
print("runtime", meta["runtime_min"])
