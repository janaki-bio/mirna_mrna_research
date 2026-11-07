"""Locus models: genes near the precursor (own locus) and near the other members of its paralog group (paralog locus).

For miRNAs with r2_full >= 0.3: genes within +-500 kb of the precursor (own locus) and of the other members of its
paralog group (paralog locus); RidgeCV on locus genes, locus + 20 PCs, and host + locus + 20 PCs. Same analysis set,
15 patient-grouped folds and ridge settings as 09_/10_. PCs are recomputed per fold exactly as in 10_state_models.py
(only R2 arrays were cached there); r2_state_20 is recomputed and must reproduce the 10_ result. Locus genes are the genes
of the filtered gene set (the r2_full features); host genes are added even if they fail the filter.
Reads audit_per_mirna_v2.csv and audit_meta_v2.json (10_), audit_hosts.csv, audit_mir_mapping.csv and
cache/gencode_v36_genes_exons.pkl (06_).
Writes audit_per_mirna_v3.csv (adds the locus columns and the label_v3 labels) and audit_meta_v3.json.

  python 11_locus_models.py --probe   # time one fold and print an estimate, write nothing"""
import json
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from audit_common import ALPHAS, SEEDS, audit_data
from common import CACHE, RESULTS

PROBE = "--probe" in sys.argv
WIN = 500_000
t_start = time.time()
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
v2 = pd.read_csv(RESULTS / "audit_per_mirna_v2.csv").set_index("mirna")
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna")
gtf_genes, _ = pd.read_pickle(CACHE / "gencode_v36_genes_exons.pkl")
mir_genes = gtf_genes[gtf_genes.gene_type == "miRNA"].drop_duplicates("gene_name").set_index("gene_name")
mapping = pd.read_csv(RESULTS / "audit_mir_mapping.csv").set_index("mirna")["mir_gene"]
SEL = [t for t in targets if v2.loc[t, "r2_full"] >= 0.3]

G = mrna_log[genes].values
gene_pos = {g: i for i, g in enumerate(genes)}
coords = gtf_genes[gtf_genes.gene_id.isin(gene_pos)]
name_of = dict(zip(gtf_genes.gene_id, gtf_genes.gene_name))


def window_genes(chrom, start, end):
    w = coords[(coords.chrom == chrom) & (coords.start <= end + WIN) & (coords.end >= start - WIN)]
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

loc = {}
for t in SEL:
    pos = precursor_pos(t)
    own = window_genes(*pos) if pos else []
    grp = pb.get(t)
    others = [m for m in group_members[grp] if m != t] if isinstance(grp, str) else []
    par = sorted({g for m in others if precursor_pos(m) for g in window_genes(*precursor_pos(m))})
    hid = hosts.loc[t, "host_gene_ids"].split(";") if isinstance(hosts.loc[t, "host_gene_ids"], str) else []
    loc[t] = {"own": own, "par": par, "any": sorted(set(own) | set(par)), "host": hid, "group": grp,
              "others": others}


def ridge_r2(Ftr, Fte, ytr, yte):
    sc = StandardScaler().fit(Ftr)
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=ytr.ndim > 1).fit(sc.transform(Ftr), ytr)
    P = m.predict(sc.transform(Fte))
    return r2_score(yte, P, multioutput="raw_values") if P.ndim > 1 else r2_score(yte, P)


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
cols = ["r2_state_20_check", "r2_locus_own", "r2_locus_paralog", "r2_locus_any", "r2_locus_state20",
        "r2_host_locus_state20"]
rec = {c: [] for c in cols}


def feats(t, kind):
    if kind == "host_any":
        ids = loc[t]["any"] + [h for h in loc[t]["host"] if h not in set(loc[t]["any"])]
        return mrna_log[ids].values  # host genes may fail the filter
    return G[:, [gene_pos[g] for g in loc[t][kind]]]


def run_fold(tr, te):
    sc = StandardScaler().fit(G[tr])
    Ptr, Pte = pca_scores(sc.transform(G[tr]), sc.transform(G[te]), 20)
    rec["r2_state_20_check"].append(ridge_r2(Ptr, Pte, Y[tr], Y[te]))
    row = {c: np.full(len(SEL), np.nan) for c in cols[1:]}
    for j, t in enumerate(SEL):
        y_tr, y_te = Y[tr, j], Y[te, j]
        for kind, c in [("own", "r2_locus_own"), ("par", "r2_locus_paralog"), ("any", "r2_locus_any")]:
            if loc[t][kind]:
                F = feats(t, kind)
                row[c][j] = ridge_r2(F[tr], F[te], y_tr, y_te)
        if loc[t]["any"]:
            F = feats(t, "any")
            row["r2_locus_state20"][j] = ridge_r2(np.hstack([F[tr], Ptr]), np.hstack([F[te], Pte]), y_tr, y_te)
        F = feats(t, "host_any")
        if F.shape[1]:
            row["r2_host_locus_state20"][j] = ridge_r2(np.hstack([F[tr], Ptr]), np.hstack([F[te], Pte]), y_tr, y_te)
    for c in cols[1:]:
        rec[c].append(row[c])


if PROBE:
    t0 = time.time()
    run_fold(*folds[0])
    tf = time.time() - t0
    print(f"{len(SEL)} miRNAs; locus sizes own median {np.median([len(v['own']) for v in loc.values()]):.0f}, "
          f"any max {max(len(v['any']) for v in loc.values())}; fold {tf:.0f} s x {len(folds)} = "
          f"{tf * len(folds) / 60:.1f} min")
    sys.exit()

for i, (tr, te) in enumerate(folds):
    run_fold(tr, te)
    print(f"fold {i} done, {(time.time() - t_start) / 60:.1f} min", flush=True)
fit_min = (time.time() - t_start) / 60

res = pd.DataFrame({c: np.vstack(v).mean(axis=0) for c, v in rec.items()}, index=pd.Index(SEL, name="mirna"))
check = float(np.abs(res.r2_state_20_check - v2.loc[SEL, "r2_state_20"]).max())
print("max |r2_state_20 - v2|:", check)
assert check < 1e-8, "PCs do not reproduce v2"
res = res.drop(columns="r2_state_20_check")
res["own_locus_genes"] = [";".join(name_of[g] for g in loc[t]["own"]) for t in SEL]
res["paralog_locus_genes"] = [";".join(name_of[g] for g in loc[t]["par"]) for t in SEL]
res["n_own_locus_genes"] = [len(loc[t]["own"]) for t in SEL]
res["n_paralog_locus_genes"] = [len(loc[t]["par"]) for t in SEL]
res["paralog_members"] = [";".join(loc[t]["others"]) for t in SEL]

out = v2.join(res)
out["delta_unexplained"] = out.r2_full - np.fmax(out.r2_host_state_20, out.r2_host_locus_state20)
out["paralog_locus_beats_own"] = np.where(out.r2_locus_paralog.notna(),
                                          (out.r2_locus_paralog > out.r2_locus_own.fillna(-np.inf)), None)


def label_v3(r):
    if r.r2_full < 0.3:
        return "poorly predicted"
    if r.label_v2_class in ("host", "tumour state", "host + state"):
        return r.label_v2
    if pd.notna(r.r2_locus_state20) and r.r2_locus_state20 >= 0.8 * r.r2_full:
        own, par = r.r2_locus_own, r.r2_locus_paralog
        return "locus: " + ("paralog locus" if pd.notna(par) and (pd.isna(own) or par > own) else "own locus")
    if r.delta_unexplained >= 0.10:
        return "unexplained"
    return "partly explained"


out["label_v3"] = out.apply(label_v3, axis=1)
out["label_v3_class"] = out.label_v3.str.split(":").str[0]
out.reset_index().to_csv(RESULTS / "audit_per_mirna_v3.csv", index=False)

sel = out.loc[SEL]
par_tab = sel[sel.r2_locus_paralog.notna()][["paralog_group", "paralog_members", "r2_full", "r2_locus_own",
                                              "r2_locus_paralog", "paralog_locus_beats_own", "label_v2", "label_v3"]]
unexpl = sel[sel.label_v3 == "unexplained"].sort_values("delta_unexplained", ascending=False)
v2_meta = json.load(open(RESULTS / "audit_meta_v2.json"))
meta = {
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "commit": v2_meta["commit"], "working_tree": v2_meta["working_tree"],
    "version": "v3 (v1 audit_per_mirna.csv / audit_meta.json and v2 audit_per_mirna_v2.csv / audit_meta_v2.json unchanged)",
    "rule_change": {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "post_hoc": "added after inspecting v2's 'specific beyond state' group; not fixed in advance",
        "what": "label_v3 keeps v2 'host', 'tumour state', 'host + state' and 'poorly predicted'; v2 'specific beyond "
                "state' / 'partly explained' are re-split into 'locus' (own / paralog), 'unexplained' and 'partly explained'",
        "thresholds_unchanged": "0.3, 0.8 x r2_full, 0.10"},
    "label_rules_v3": [
        "poorly predicted: r2_full < 0.3",
        "keep v2 label if v2 class is host, tumour state or host + state",
        "locus: r2_locus_state20 >= 0.8 * r2_full; sub-label 'paralog locus' if r2_locus_paralog > r2_locus_own, else 'own locus'",
        "unexplained: delta_unexplained >= 0.10, delta_unexplained = r2_full - max(r2_host_state_20, r2_host_locus_state20)",
        "partly explained: otherwise"],
    "locus": {"window": "+-500 kb of the precursor's MIR gene (GENCODE v36), any strand",
              "own_locus_genes": "genes in the filtered gene set (27,671) overlapping the window",
              "paralog_locus_genes": "union of the windows of all other members of the paralog group (06_annotate_hosts.py rule, all precursors in the matrix)",
              "host": "host gene(s) added for r2_host_locus_state20 even if they fail the gene filter",
              "models": "RidgeCV(logspace(-2, 8, 41)), standardised within folds; PCs as in v2",
              "n_mirnas": len(SEL),
              "n_own_locus_genes_median": float(np.median(sel.n_own_locus_genes)),
              "check_r2_state_20_vs_v2_max_abs_diff": check},
    "label_counts_v3": sel.label_v3_class.value_counts().to_dict() | {"poorly predicted": int((out.r2_full < 0.3).sum())},
    "label_counts_v3_detailed": out.label_v3.value_counts().to_dict(),
    "v2_to_v3": pd.crosstab(out.label_v2_class, out.label_v3_class).to_dict(orient="index"),
    "unexplained": unexpl[["r2_full", "r2_host_state_20", "r2_host_locus_state20", "delta_unexplained", "flags"]]
        .round(3).reset_index().fillna("").to_dict(orient="records"),
    "paralog_locus_table": par_tab.round(3).reset_index().fillna("").to_dict(orient="records"),
    "notes": [
        "locus labels use r2_locus_state20 (locus genes + 20 PCs); gain over r2_state_20 for 'locus' miRNAs: median "
        f"{(sel.r2_locus_state20 - sel.r2_state_20)[sel.label_v3_class == 'locus'].median():.3f}, "
        f"< 0.05 for {int(((sel.r2_locus_state20 - sel.r2_state_20)[sel.label_v3_class == 'locus'] < 0.05).sum())}",
        "no own-locus genes (r2_locus_own NaN): " + ", ".join(sel.index[sel.n_own_locus_genes == 0])
        + " (mir-3607 unmapped in GENCODE v36; mir-128-2 has no gene passing the gene filter within +-500 kb). "
        "Where r2_host_locus_state20 is NaN, delta_unexplained uses r2_host_state_20",
        "paralog locus beating own locus may reflect read assignment between near-identical precursors (e.g. "
        "mir-196a-1 own 0.12 vs paralog 0.88; F4d: paralog targets correlate up to rho 0.997) rather than biology",
        f"'unexplained' miRNAs with at least one flag: {int((unexpl['flags'].fillna('') != '').sum())} of {len(unexpl)}"],
    "runtime_min": round((time.time() - t_start) / 60, 1), "fit_min": round(fit_min, 1),
}
json.dump(meta, open(RESULTS / "audit_meta_v3.json", "w"), indent=2, default=lambda x: x if not isinstance(x, (np.bool_,)) else bool(x))
pd.set_option("display.width", 250)
print(pd.Series(meta["label_counts_v3"]).to_string())
print(par_tab.round(2).to_string())
print(unexpl[["r2_full", "r2_host_state_20", "r2_host_locus_state20", "delta_unexplained", "flags"]].round(2).to_string())
