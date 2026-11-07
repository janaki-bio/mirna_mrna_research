"""Per-fold R2 of every model under plate-grouped CV or plate-adjusted patient CV, 15 folds x 337 targets.
Designs:
  plate : balanced mRNA-plate folds (20_plate_cv_checks.py: whole plates, 5,000 random orders per seed, seeds 42-44), raw data.
  adj   : patient-grouped folds (seeds 42-44) with SYMMETRIC plate adjustment inside each fold: per-plate means (computed on the training
          tumours, minus the training grand mean) are removed from the mRNA matrix (all 27,671 filtered genes + host genes) and from the
          miRNA targets, the same training-fold shifts are applied to the test tumours (a plate absent from training gets no shift);
          every model (full, host, components, locus, host + locus, components without host/local genes) uses the adjusted mRNA and is
          scored against the adjusted test targets. Clinical models are not fitted under 'adj'.
Models and settings are those of 13_reference_model_fits.py (+ components k = 5, 10, 20, 50 and host + components for every k, as in
10_state_models.py; + clinical under 'plate'): ridge = RidgeCV(logspace(-2, 8, 41)), standardisation and PCA inside each training fold.
Host models only if a host gene passes the expression filter (host not measurable -> NaN; host + components = components).
Clinical (plate design): LinearRegression on PAM50 (one-hot, drop Basal) + CPE; + LUMP (training-mean imputed) + missing indicator; and
PAM50 + CPE + LUMP mean-imputed (r2_cov).

  python 21_plate_cv_fits.py --probe          # fold 0 of the patient design unadjusted, checked against cache/audit_r2_fold_records.pkl,
                                                 # then one fold of plate and adj, timing; writes nothing
  python 21_plate_cv_fits.py --design plate|adj
Outputs: cache/audit_plate_fold_records.pkl (plate), cache/audit_plateadj_sym_fold_records.pkl (adj)."""
import json, re, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from audit_common import ALPHAS, SEEDS, audit_data
from common import CACHE, HERE, RESULTS

PROBE = "--probe" in sys.argv
DESIGN = sys.argv[sys.argv.index("--design") + 1] if "--design" in sys.argv else None
BASE_K, BASE_WIN = 20, 500_000
KS = [10, 20, 50]; ALLK = [5, 10, 20, 50]
WINS = [250_000, 500_000, 1_000_000]
t_start = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
v3 = pd.read_csv(RESULTS / "audit_per_mirna_v3.csv").set_index("mirna")
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna")
gtf_genes, _ = pd.read_pickle(CACHE / "gencode_v36_genes_exons.pkl")
mir_genes = gtf_genes[gtf_genes.gene_type == "miRNA"].drop_duplicates("gene_name").set_index("gene_name")
mapping = pd.read_csv(RESULTS / "audit_mir_mapping.csv").set_index("mirna")["mir_gene"]
assert list(v3.index) == targets
T, Gn = len(targets), len(genes)
gene_pos = {g: i for i, g in enumerate(genes)}
host_ids = {t: hosts.loc[t, "host_gene_ids"].split(";") if isinstance(hosts.loc[t, "host_gene_ids"], str) else [] for t in targets}
measurable = {t: any(h in gene_pos for h in host_ids[t]) for t in targets}
extra = sorted({h for t in targets if measurable[t] for h in host_ids[t] if h not in gene_pos})
cols_all = list(genes) + extra
pos_all = {g: i for i, g in enumerate(cols_all)}
Gall = mrna_log[cols_all].values
Y = mirna_log[targets].values
n = len(Y)
plate_of = pd.read_csv(HERE / "external/gdc/file_aliquots.csv").set_index("file_id").aliquot.str.split("-").str[5]
mp = s.mrna_file_id.map(plate_of)
pid = pd.Series(mp.values).astype("category").cat.codes.values; K = pid.max() + 1
PAM = pd.get_dummies(s.pam50, prefix="pam50", drop_first=True).astype(float).values
CPE = s[["cpe"]].values.astype(float); LUMP = s[["lump"]].values.astype(float)

# ---- locus definitions (13_reference_model_fits.py) ----
coords = gtf_genes[gtf_genes.gene_id.isin(gene_pos)]
def window_genes(chrom, start, end, win):
    w = coords[(coords.chrom == chrom) & (coords.start <= end + win) & (coords.end >= start - win)]
    return list(w.gene_id)
def precursor_pos(m):
    g = mapping.get(m)
    if not isinstance(g, str): return None
    r = mir_genes.loc[g]; return r.chrom, r.start, r.end
def paralog_base(name):
    mm = re.match(r"^(hsa-(?:mir|let)-[0-9a-z]+)-(\d+)$", name); return mm.group(1) if mm else None
pb = pd.Series({m: paralog_base(m) for m in mapping.index}).dropna(); pb = pb[pb.duplicated(keep=False)]
group_members = pb.groupby(pb).apply(lambda x: list(x.index))
loc = {w: {} for w in WINS}
for w in WINS:
    for t in targets:
        pos = precursor_pos(t); own = window_genes(*pos, w) if pos else []
        grp = pb.get(t); others = [m for m in group_members[grp] if m != t] if isinstance(grp, str) else []
        par = sorted({g for m in others if precursor_pos(m) for g in window_genes(*precursor_pos(m), w)})
        loc[w][t] = {"own": own, "par": par, "any": sorted(set(own) | set(par))}
host_all = sorted({h for t in targets for h in host_ids[t]})
win_all = sorted({g for t in targets if precursor_pos(t) for g in window_genes(*precursor_pos(t), BASE_WIN)})
excl = (set(host_all) & set(gene_pos)) | set(win_all)
keep_cols = np.array([g not in excl for g in genes]); assert int((~keep_cols).sum()) == 4080

def ridge_r2(Ftr, Fte, ytr, yte):
    sc = StandardScaler().fit(Ftr)
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=ytr.ndim > 1).fit(sc.transform(Ftr), ytr)
    P = m.predict(sc.transform(Fte))
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)
def lin_r2(Ftr, Fte, ytr, yte):
    P = LinearRegression().fit(Ftr, ytr).predict(Fte)
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)
def pca_scores(Xtr, Xte, kmax):
    ev, U = np.linalg.eigh(Xtr @ Xtr.T); o = np.argsort(ev)[::-1][:kmax]; ev, U = ev[o], U[:, o]; sv = np.sqrt(ev)
    return U * sv, (Xte @ Xtr.T) @ U / sv
wname = lambda w: f"{w // 1000}kb" if w < 1_000_000 else f"{w // 1_000_000}Mb"

COLS = (["r2_full", "r2_host"] + [f"r2_state_{k}" for k in ALLK] + [f"r2_host_state_{k}" for k in ALLK]
        + ["r2_state20_excl", "r2_host_state20_excl"]
        + [f"r2_locus_{kind}_{wname(w)}" for kind in ("own", "paralog", "any") for w in WINS]
        + [f"r2_locus_state{k}_{wname(w)}" for k in KS for w in WINS if k == BASE_K or w == BASE_WIN]
        + [f"r2_host_locus_state{k}_{wname(w)}" for k in KS for w in WINS if k == BASE_K or w == BASE_WIN])
CLIN = ["r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]

def plate_shift(A, tr):
    D = np.zeros((len(tr), K)); D[np.arange(len(tr)), pid[tr]] = 1
    cnt = D.sum(0); mu = A[tr].mean(0); shift = np.zeros((K, A.shape[1])); seen = cnt > 0
    shift[seen] = (D.T @ A[tr])[seen] / cnt[seen, None] - mu
    return shift

def run_fold(tr, te, adjust, clinical):
    if adjust:
        sg, sy = plate_shift(Gall, tr), plate_shift(Y, tr)
        Gtr, Gte = Gall[tr] - sg[pid[tr]], Gall[te] - sg[pid[te]]
        Ytr, Yte = Y[tr] - sy[pid[tr]], Y[te] - sy[pid[te]]
    else:
        Gtr, Gte, Ytr, Yte = Gall[tr], Gall[te], Y[tr], Y[te]
    row = {c: np.full(T, np.nan) for c in COLS}
    row["r2_full"] = ridge_r2(Gtr[:, :Gn], Gte[:, :Gn], Ytr, Yte)
    sc = StandardScaler().fit(Gtr[:, :Gn]); Xtr, Xte = sc.transform(Gtr[:, :Gn]), sc.transform(Gte[:, :Gn])
    Ptr, Pte = pca_scores(Xtr, Xte, 50)
    Etr, Ete = pca_scores(Xtr[:, keep_cols], Xte[:, keep_cols], BASE_K)
    del Xtr, Xte
    for k in ALLK:
        row[f"r2_state_{k}"] = ridge_r2(Ptr[:, :k], Pte[:, :k], Ytr, Yte)
    row["r2_state20_excl"] = ridge_r2(Etr, Ete, Ytr, Yte)
    for k in ALLK: row[f"r2_host_state_{k}"] = row[f"r2_state_{k}"].copy()
    row["r2_host_state20_excl"] = row["r2_state20_excl"].copy()
    for j, t in enumerate(targets):
        ytr, yte = Ytr[:, j], Yte[:, j]
        if measurable[t]:
            hi = [pos_all[h] for h in host_ids[t]]; Htr, Hte = Gtr[:, hi], Gte[:, hi]
            row["r2_host"][j] = lin_r2(Htr, Hte, ytr, yte)
            for k in ALLK:
                row[f"r2_host_state_{k}"][j] = ridge_r2(np.hstack([Htr, Ptr[:, :k]]), np.hstack([Hte, Pte[:, :k]]), ytr, yte)
            row["r2_host_state20_excl"][j] = ridge_r2(np.hstack([Htr, Etr]), np.hstack([Hte, Ete]), ytr, yte)
        for w in WINS:
            ks = KS if w == BASE_WIN else [BASE_K]
            for kind, c in (("own", "own"), ("par", "paralog"), ("any", "any")):
                if loc[w][t][kind]:
                    ix = [pos_all[g] for g in loc[w][t][kind]]
                    row[f"r2_locus_{c}_{wname(w)}"][j] = ridge_r2(Gtr[:, ix], Gte[:, ix], ytr, yte)
            ids = list(loc[w][t]["any"])
            if ids:
                ix = [pos_all[g] for g in ids]
                for k in ks:
                    row[f"r2_locus_state{k}_{wname(w)}"][j] = ridge_r2(np.hstack([Gtr[:, ix], Ptr[:, :k]]), np.hstack([Gte[:, ix], Pte[:, :k]]), ytr, yte)
            if measurable[t]: ids += [h for h in host_ids[t] if h not in set(ids)]
            if ids:
                ix = [pos_all[g] for g in ids]
                for k in ks:
                    row[f"r2_host_locus_state{k}_{wname(w)}"][j] = ridge_r2(np.hstack([Gtr[:, ix], Ptr[:, :k]]), np.hstack([Gte[:, ix], Pte[:, :k]]), ytr, yte)
    if clinical:
        mu = np.nanmean(LUMP[tr]); l_tr, l_te = np.where(np.isnan(LUMP[tr]), mu, LUMP[tr]), np.where(np.isnan(LUMP[te]), mu, LUMP[te])
        row["r2_pam50_cpe"] = lin_r2(np.hstack([PAM[tr], CPE[tr]]), np.hstack([PAM[te], CPE[te]]), Ytr, Yte)
        row["r2_pam50_cpe_lump_indicator"] = lin_r2(np.hstack([PAM[tr], CPE[tr], l_tr, np.isnan(LUMP[tr])]), np.hstack([PAM[te], CPE[te], l_te, np.isnan(LUMP[te])]), Ytr, Yte)
        row["r2_pam50_cpe_lump_meanimputed"] = lin_r2(np.hstack([PAM[tr], CPE[tr], l_tr]), np.hstack([PAM[te], CPE[te], l_te]), Ytr, Yte)
    return row

# ---- folds ----
patient_folds = []
for seed in SEEDS:
    patient_folds += list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Gall[:, :1], groups=s.patient.values))
sizes = mp.value_counts()
def balanced(seed, trials=5000):   # identical to 20_plate_cv_checks.py
    rng = np.random.default_rng(seed); names, w = sizes.index.values, sizes.values.astype(float); best = None
    for _ in range(trials):
        order = np.argsort(-(w * rng.uniform(0.8, 1.2, len(w)))); load, assign = np.zeros(5), {}
        for i in order:
            f = int(np.argmin(load)); load[f] += w[i]; assign[names[i]] = f
        sp_ = load.max() - load.min()
        if best is None or sp_ < best[0]: best = (sp_, assign)
    return best[1]
plate_folds, fsizes = [], []
for seed in SEEDS:
    f = mp.map(balanced(seed)).values
    for k in range(5): plate_folds.append((np.where(f != k)[0], np.where(f == k)[0]))
    fsizes.append([int((f == k).sum()) for k in range(5)])
exp = json.load(open(CACHE / "meta_r2c.json"))["plate_cv"]["plate_fold_sizes"]
assert fsizes == [exp[str(sd)] for sd in SEEDS], (fsizes, exp)
assert all(not set(mp.values[tr]) & set(mp.values[te]) for tr, te in plate_folds)
print("plate folds reproduce the 20_plate_cv_checks.py fold sizes:", fsizes)

if PROBE:
    old = pd.read_pickle(CACHE / "audit_r2_fold_records.pkl")["F"]
    t0 = time.time(); r = run_fold(*patient_folds[0], False, False); tf = time.time() - t0
    worst = {}
    for c in COLS:
        if c in old:
            a, b = r[c], old[c][0]
            comp = np.ones(T, bool)
            worst[c] = float(np.nanmax(np.abs(a - b))) if not (np.isnan(a) != np.isnan(b)).any() else float("nan")
    # r2_state_5 / r2_host_state_5 are not in cache/audit_r2_fold_records.pkl -> compare with cache/audit_v2_fold_records.pkl
    v2 = pd.read_pickle(CACHE / "audit_v2_fold_records.pkl")
    for k in (5,):
        worst[f"r2_state_{k}(v2 cache)"] = float(np.abs(r[f"r2_state_{k}"] - v2[f"r2_state_{k}"][0][:T]).max())
        mm = np.array([measurable[t] or not host_ids[t] for t in targets])
        worst[f"r2_host_state_{k}(v2 cache, hosted & measurable or no host)"] = float(np.abs(r[f"r2_host_state_{k}"] - v2[f"r2_host_state_{k}"][0][:T])[mm].max())
    bad = {c: v for c, v in worst.items() if not (v < 1e-8)}
    print("patient fold 0 unadjusted vs caches: max diff", max(v for v in worst.values() if v == v), "; columns not < 1e-8 (host columns differ by design where host not measurable):", bad)
    print(f"fold time {tf:.0f} s (incl. clinical off); full design estimate: {tf * 15 / 60:.1f} min each for plate and adj (+ ~2 min load)")
    t0 = time.time(); run_fold(*plate_folds[0], False, True); print(f"plate fold {time.time() - t0:.0f} s")
    t0 = time.time(); run_fold(*patient_folds[0], True, False); print(f"adj fold {time.time() - t0:.0f} s")
    sys.exit()

assert DESIGN in ("plate", "adj")
folds = plate_folds if DESIGN == "plate" else patient_folds
rec = {c: [] for c in COLS + (CLIN if DESIGN == "plate" else [])}
for i, (tr, te) in enumerate(folds):
    r = run_fold(tr, te, DESIGN == "adj", DESIGN == "plate")
    for c in rec: rec[c].append(r[c])
    print(f"fold {i} done, {(time.time() - t_start) / 60:.1f} min", flush=True)
F = {c: np.vstack(v) for c, v in rec.items()}
if DESIGN == "plate":   # reproduce 20_plate_cv_checks.py plate-CV results
    pc = pd.read_csv(RESULTS / "audit_plate_cv.csv", index_col=0)
    chk = {"r2_full": float(np.abs(F["r2_full"].mean(0) - pc.r2_full_plate_foldmean.values).max()),
           "r2_state_20": float(np.abs(F["r2_state_20"].mean(0) - pc.r2_state_20_plate_foldmean.values).max())}
    print("check vs 20_plate_cv_checks.py plate CV:", chk); assert max(chk.values()) < 1e-8
else:
    chk = {}
pd.to_pickle({"F": F, "design": DESIGN, "targets": targets, "measurable": measurable, "checks": chk,
              "fold_test_sizes": [len(te) for _, te in folds], "fit_min": round((time.time() - t_start) / 60, 1),
              "n_pc_genes_removed": int((~keep_cols).sum())},
             CACHE / ("audit_plate_fold_records.pkl" if DESIGN == "plate" else "audit_plateadj_sym_fold_records.pkl"))
print("done", round((time.time() - t_start) / 60, 1), "min")
