"""Attribution labels with a ceiling model, robustness and fold uncertainty (with host-model validity checks).

Reads cache/audit_r2_fold_records.pkl (13_reference_model_fits.py): per-fold R2, 15 folds x 337 targets, for every model.
Nothing is refitted here except the 8 MEG8 host diagnostic fits.

Revised rule (r2_full, r2_best and the reference models are means over the 15 test folds):
  reference models (6, as in 12_robustness_fits.py): r2_host, r2_state_k, r2_host_state_k, r2_locus_any, r2_locus_state_k,
  r2_host_locus_state_k.  Host-based models exist only if >= 1 host gene passes the expression filter (13_).
  best_ref_f = max over reference models in fold f;  best_f = max(r2_full_f, best_ref_f)   [per-fold maximum, used
  everywhere: labels, shares and uncertainty];  r2_best = mean_f best_f;  gap = mean_f (r2_full_f - best_ref_f).
  poorly predicted: r2_best < 0.3 | host: r2_host >= 0.8 r2_best | tumour state: r2_state_k >= 0.8 r2_best |
  host + state: r2_host_state_k >= 0.8 r2_best | own locus / paralog locus: r2_locus_state_k >= 0.8 r2_best, 'paralog' if
  r2_locus_paralog > r2_locus_own (locus-only models at the same window) | residual: gap >= 0.10 |
  partly explained: otherwise. First match wins.
Outputs: audit_per_mirna_r2.csv, audit_robustness_r2.csv, audit_uncertainty_r2.csv, cache/meta_labels_r2.json.
18_ceiling_model.py rewrites the three CSVs with the ceiling model chosen by mean R2 instead of the per-fold maximum."""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from audit_common import SEEDS, audit_data
from common import CACHE, RESULTS

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice")
s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
v3 = pd.read_csv(RESULTS / "audit_per_mirna_v3.csv").set_index("mirna")
hosts = pd.read_csv(RESULTS / "audit_hosts.csv").set_index("mirna")
d = pd.read_pickle(CACHE / "audit_r2_fold_records.pkl")
F, measurable, fit_meta = d["F"], d["measurable"], d
assert d["targets"] == targets
T, J = len(targets), 15
BASE_K, BASE_W = 20, 500_000
RULES = ["host", "tumour state", "host + state", "locus"]
CATS = ["host", "tumour state", "host + state", "own locus", "paralog locus", "residual", "partly explained", "poorly predicted"]
tcrit, TT = stats.t.ppf(0.975, J - 1), 0.25   # Nadeau-Bengio: 5-fold CV, n_test/n_train = 1/4


def wn(w):
    return f"{w // 1000}kb" if w < 1_000_000 else f"{w // 1_000_000}Mb"


MEAN = {c: np.nanmean(a, axis=0) for c, a in F.items()}


def cols(k=BASE_K, w=BASE_W, excl=False):
    st, hst = ("r2_state20_excl", "r2_host_state20_excl") if excl else (f"r2_state_{k}", f"r2_host_state_{k}")
    rules = {"host": "r2_host", "tumour state": st, "host + state": hst, "locus": f"r2_locus_state{k}_{wn(w)}"}
    refs = ["r2_host", st, hst, f"r2_locus_any_{wn(w)}", f"r2_locus_state{k}_{wn(w)}", f"r2_host_locus_state{k}_{wn(w)}"]
    return rules, refs


def fold_best(refs):
    rb = np.nanmax(np.stack([F[c] for c in refs]), axis=0)          # 15 x T (state is always defined)
    return rb, np.maximum(F["r2_full"], rb)


def attribute(k=BASE_K, w=BASE_W, excl=False, f=0.8, ceiling="best", agg="foldmax"):
    """Labels at a setting.  ceiling 'full': thresholds relative to r2_full (0.8 r2_full, poorly = r2_full < 0.3), 'best': relative to r2_best.
    agg 'foldmax': per-fold maximum over models; 'meanmax': maximum of the 15-fold means."""
    rules, refs = cols(k, w, excl)
    rb_f, b_f = fold_best(refs)
    if agg == "foldmax":
        refbest, best, gap = rb_f.mean(0), b_f.mean(0), (F["r2_full"] - rb_f).mean(0)
    else:
        refbest = np.nanmax(np.stack([MEAN[c] for c in refs]), axis=0)
        best, gap = np.maximum(MEAN["r2_full"], refbest), MEAN["r2_full"] - refbest
    ceil = best if ceiling == "best" else MEAN["r2_full"]
    lab = np.array(["partly explained"] * T, dtype=object)
    done = np.zeros(T, bool)
    poor = ceil < 0.3
    lab[poor], done[poor] = "poorly predicted", True
    for cat in RULES:
        m = MEAN[rules[cat]]
        hit = ~done & ~np.isnan(m) & (m >= f * ceil)
        lab[hit], done[hit] = cat, True
    own, par = MEAN[f"r2_locus_own_{wn(w)}"], MEAN[f"r2_locus_paralog_{wn(w)}"]
    is_par = ~np.isnan(par) & (np.isnan(own) | (par > np.nan_to_num(own, nan=-np.inf)))
    lab[lab == "locus"] = np.where(is_par, "paralog locus", "own locus")[lab == "locus"]
    hit = ~done & (gap >= 0.10)
    lab[hit] = "residual"
    out = pd.DataFrame({"label": lab, "r2_best": best, "best_ref": refbest, "gap": gap}, index=targets)
    return out


def counts(lab):
    return {c: int((lab == c).sum()) for c in CATS}


# ---------------- host-model validity ----------------
host_status = pd.Series({t: ("no host gene" if not isinstance(hosts.loc[t, "host_gene_ids"], str) or not hosts.loc[t, "host_gene_ids"]
                              else ("measurable" if measurable[t] else "not measurable")) for t in targets})
print("host status:", host_status.value_counts().to_dict())
nm = [t for t in targets if host_status[t] == "not measurable"]
print("not measurable, by host gene:", pd.Series([hosts.loc[t, "host_genes"] for t in nm]).value_counts().head(8).to_dict())

# MEG8 diagnostic: outlier tumour, leverage and host R2 without it
OUT = "TCGA-AC-A2QH-01A"
meg8 = [t for t in targets if "MEG8" in str(hosts.loc[t, "host_genes"]).split(";")]
mg_ids = {t: hosts.loc[t, "host_gene_ids"].split(";") for t in meg8}
mg_id = sorted({g for t in meg8 for g, n in zip(mg_ids[t], hosts.loc[t, "host_genes"].split(";")) if n == "MEG8"})
assert len(mg_id) == 1, mg_id
x = mrna_log[mg_id[0]]
io = list(s.index).index(OUT)
folds = []
for seed in SEEDS:
    folds += list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros((len(s), 1)), groups=s.patient.values))
raw_tpm = pd.read_pickle(CACHE / "mrna_matrix.pkl").loc[s.index, mg_id[0]]
xv = x.values
lev_full = 1 / len(xv) + (xv[io] - xv.mean()) ** 2 / ((xv - xv.mean()) ** 2).sum()
in_test = [i for i, (tr, te) in enumerate(folds) if io in te]
lev_train, z_test = [], []
for i, (tr, te) in enumerate(folds):
    if io in tr:
        xt = xv[tr]
        lev_train.append(1 / len(xt) + (xv[io] - xt.mean()) ** 2 / ((xt - xt.mean()) ** 2).sum())
    else:
        xt = xv[tr]
        z_test.append((xv[io] - xt.mean()) / xt.std(ddof=1))
keep = np.ones(len(s), bool)
keep[io] = False
rows = []
for t in meg8:
    j = targets.index(t)
    y = mirna_log[t].values
    HX = mrna_log[mg_ids[t]].values   # full host set (mir-370 also has AL117190.1)
    r_in, r_ex = [], []
    for tr, te in folds:
        r_in.append(r2_score(y[te], LinearRegression().fit(HX[tr], y[tr]).predict(HX[te])))
        tr2, te2 = tr[keep[tr]], te[keep[te]]
        r_ex.append(r2_score(y[te2], LinearRegression().fit(HX[tr2], y[tr2]).predict(HX[te2])))
    r_in, r_ex = np.array(r_in), np.array(r_ex)
    rows.append({"mirna": t, "host": "MEG8", "r2_host_with_outlier": r_in.mean(), "r2_host_without_outlier": r_ex.mean(),
                 "r2_host_with_outlier_fold_containing_it_min": r_in[in_test].min(),
                 "r2_host_without_outlier_same_folds_min": r_ex[in_test].min(),
                 "r2_host_without_outlier_min_fold": r_ex.min(),
                 "outlier_mirna_z": (y[io] - y.mean()) / y.std(ddof=1), "r2_host_cache_check": float(np.nanmean(MEAN["r2_host"][j]))})
meg = pd.DataFrame(rows)
check_cache = float(np.abs(meg.r2_host_with_outlier.values - v3.loc[meg8, "r2_host"].values).max())
assert check_cache < 1e-8, check_cache
meg8_info = {
    "outlier": OUT, "meg8_gene": mg_id[0], "n_meg8_hosted_targets": len(meg8), "targets": meg8,
    "meg8_tpm_outlier": float(raw_tpm.iloc[io]), "meg8_tpm_max_other": float(raw_tpm.drop(OUT).max()),
    "meg8_frac_tumours_tpm_gt_0.5": float((raw_tpm > 0.5).mean()), "n_tumours_tpm_gt_0.5": int((raw_tpm > 0.5).sum()),
    "meg8_log2_value_outlier": float(xv[io]), "meg8_log2_median": float(np.median(xv)), "meg8_log2_sd_others": float(np.delete(xv, io).std(ddof=1)),
    "leverage_full_data_simple_regression": float(lev_full), "leverage_in_training_folds_min_max": [float(min(lev_train)), float(max(lev_train))],
    "n_training_folds_containing_outlier": len(lev_train), "test_folds_containing_outlier": in_test,
    "outlier_z_in_test_folds_vs_training": [float(z) for z in z_test],
    "r2_host_with_outlier_mean_over_targets": float(meg.r2_host_with_outlier.mean()),
    "r2_host_without_outlier_mean_over_targets": float(meg.r2_host_without_outlier.mean()),
    "r2_host_with_outlier_range": [float(meg.r2_host_with_outlier.min()), float(meg.r2_host_with_outlier.max())],
    "r2_host_without_outlier_range": [float(meg.r2_host_without_outlier.min()), float(meg.r2_host_without_outlier.max())],
    "cache_check_max_abs": check_cache, "per_target": meg.round(4).to_dict(orient="records"),
    "note": "MEG8 fails the expression filter (TPM > 0.5 in >= 10% of tumours), so the revised rule sets the host-based R2 of these targets to NaN; "
            "this diagnostic only confirms the cause of the earlier negative r2_host"}
print(json.dumps({k: v for k, v in meg8_info.items() if k != "per_target"}, indent=1))
print(meg.round(3).to_string(index=False))

# ---------------- ceiling and labels ----------------
cur = v3.label_v3_class.replace({"unexplained": "residual"})
base = attribute()                     # 0.8 x r2_best, per-fold maximum
stageA = attribute(ceiling="full", agg="meanmax")   # host filter only, thresholds relative to r2_full
stageB = attribute(agg="meanmax")      # ceiling, but best = max of means (sensitivity to the per-fold maximum)
cur_split = cur.copy()
# label_v3 locus labels split into own / paralog, so own/paralog changes are visible
cur_split[v3.label_v3.str.contains("paralog locus")] = "paralog locus"
cur_split[v3.label_v3.str.contains("own locus")] = "own locus"
print("v3 current:", counts(cur_split))
print("A host filter only:", counts(stageA.label), "changed", int((stageA.label != cur_split).sum()))
print("B ceiling (max of means):", counts(stageB.label), "changed vs v3", int((stageB.label != cur_split).sum()))
print("revised (per-fold max):", counts(base.label), "changed vs v3", int((base.label.replace({"own locus": "locus", "paralog locus": "locus"}) != cur).sum()))
changed = base.label != cur_split
trans = pd.crosstab(cur_split, base.label)
print(trans.to_string())
print("labels changed (incl. locus split):", int(changed.sum()), "; excluding the locus split:",
      int((base.label.replace({"own locus": "locus", "paralog locus": "locus"}) != cur).sum()))

# median shares (reference R2 / r2_best); non-poor miRNAs
rules0, refs0 = cols()
rb_f, b_f = fold_best(refs0)
ok = (base.label != "poorly predicted").values
ok_old = (v3.r2_full >= 0.3).values
share = {}
for name, c in [("host", "r2_host"), ("state_20", "r2_state_20"), ("host_state_20", "r2_host_state_20"),
                ("locus_state20", "r2_locus_state20_500kb"), ("host_locus_state20", "r2_host_locus_state20_500kb"),
                ("locus_any", "r2_locus_any_500kb")]:
    share[name] = {"median_over_r2_best": float(np.nanmedian(MEAN[c][ok] / base.r2_best.values[ok])), "n_defined": int((~np.isnan(MEAN[c][ok])).sum()),
                   "median_over_r2_full_r2full_ge_0.3": float(np.nanmedian(MEAN[c][ok_old] / MEAN["r2_full"][ok_old]))}
share["best_ref"] = {"median_over_r2_best": float(np.median(base.best_ref.values[ok] / base.r2_best.values[ok])), "n_defined": int(ok.sum()),
                     "median_over_r2_full_r2full_ge_0.3": float(np.median(base.best_ref.values[ok_old] / MEAN["r2_full"][ok_old]))}
share["r2_full"] = {"median_over_r2_best": float(np.median(MEAN["r2_full"][ok] / base.r2_best.values[ok])), "n_defined": int(ok.sum())}
print(pd.DataFrame(share).T.round(3).to_string())
r2best_gt_full = int(((base.r2_best - MEAN["r2_full"]) > 0.01).sum())
newly_not_poor = base.index[(v3.r2_full < 0.3).values & (base.label != "poorly predicted").values]
print("r2_best > r2_full + 0.01:", r2best_gt_full, "; r2_full < 0.3 but r2_best >= 0.3:", len(newly_not_poor), list(newly_not_poor)[:30])

# ---------------- robustness ----------------
SETTINGS = {
    "revised baseline (k=20, +-500 kb, 0.8)": dict(),
    "k=10": dict(k=10), "k=50": dict(k=50),
    "window +-250 kb": dict(w=250_000), "window +-1 Mb": dict(w=1_000_000),
    "PCs without host and +-500 kb target-locus genes": dict(excl=True),
    "threshold 0.7": dict(f=0.7), "threshold 0.9": dict(f=0.9),
}
labs = {n: attribute(**kw) for n, kw in SETTINGS.items()}
rows = []
for n, o in labs.items():
    kw = SETTINGS[n]
    r = {"setting": n, "k": kw.get("k", BASE_K), "window_kb": kw.get("w", BASE_W) // 1000, "fraction": kw.get("f", 0.8),
         "pcs": "excl. host + local genes" if kw.get("excl") else "all filtered genes", "rule": "revised (r2_best, per-fold max)"}
    r |= counts(o.label)
    r["n_changed_vs_revised_baseline"] = int((o.label != base.label).sum())
    r["n_r2_best_ge_0.3"] = int((o.label != "poorly predicted").sum())
    rows.append(r)
# comparison rows: host filter only (thresholds relative to r2_full), and the ceiling with max-of-means
for n, o, rl in [("previous rule, host filter only (0.8 x r2_full)", stageA, "0.8 x r2_full, host filter"),
                 ("ceiling, best = max of the 15-fold means", stageB, "r2_best = max of means")]:
    r = {"setting": n, "k": BASE_K, "window_kb": 500, "fraction": 0.8, "pcs": "all filtered genes", "rule": rl} | counts(o.label)
    r["n_changed_vs_revised_baseline"] = int((o.label != base.label).sum())
    r["n_r2_best_ge_0.3"] = int((o.label != "poorly predicted").sum())
    rows.append(r)
sens = pd.DataFrame(rows)
sens.to_csv(RESULTS / "audit_robustness_r2.csv", index=False)
pd.set_option("display.width", 280)
print(sens.drop(columns=["pcs", "rule"]).to_string(index=False))

T1 = ["k=10", "k=50", "window +-250 kb", "window +-1 Mb"]
share_agree = np.mean([labs[n].label.values == base.label.values for n in T1], axis=0)
agree_by_label = pd.Series(share_agree, index=targets).groupby(base.label).mean().round(3).to_dict()
n_all_agree = int((share_agree == 1).sum())
n_all_agree_nonpoor = int(((share_agree == 1) & ok).sum())
print("all 4 settings agree:", n_all_agree, "(non-poor", n_all_agree_nonpoor, "); mean agreement by label:", agree_by_label)
for n in ["PCs without host and +-500 kb target-locus genes", "k=10", "k=50"]:
    print(n, "transitions:\n", pd.crosstab(base.label, labs[n].label).to_string())
pcx = labs["PCs without host and +-500 kb target-locus genes"]
pc_share = {"median_share_state_20": float(np.median(MEAN["r2_state_20"][ok] / base.r2_best.values[ok])),
            "median_share_state_excl": float(np.median(MEAN["r2_state20_excl"][ok] / base.r2_best.values[ok])),
            "median_share_host_state_20": float(np.median(MEAN["r2_host_state_20"][ok] / base.r2_best.values[ok])),
            "median_share_host_state_excl": float(np.median(MEAN["r2_host_state20_excl"][ok] / base.r2_best.values[ok])),
            "per_mirna_change_r2_state_20": [float((MEAN["r2_state20_excl"] - MEAN["r2_state_20"]).min()), float((MEAN["r2_state20_excl"] - MEAN["r2_state_20"]).max())],
            "n_genes_removed": d["n_pc_genes_removed"], "n_labels_changed": int((pcx.label != base.label).sum())}
print(pc_share)

# ---------------- uncertainty (Nadeau-Bengio, per-fold maximum) ----------------
def nb(v):
    m, sd = v.mean(), v.std(ddof=1)
    h = tcrit * np.sqrt((1 / J + TT) * sd ** 2)
    return m, m - h, m + h


gap_f = F["r2_full"] - rb_f
rows = []
for j, t in enumerate(targets):
    lab = base.label[t]
    cat = "locus" if "locus" in lab else lab
    best_f = b_f[:, j]
    dd = {c: F[rules0[c]][:, j] - 0.8 * best_f for c in RULES}
    r = {"mirna": t, "label_revised": lab, "r2_full": MEAN["r2_full"][j], "r2_best": base.r2_best[t], "gap_mean": gap_f[:, j].mean()}
    r["best_ref_model"] = refs0[int(np.nanargmax([MEAN[c][j] for c in refs0]))]
    _, r["gate_lo"], r["gate_hi"] = nb(best_f - 0.3)                       # not-poorly-predicted gate: r2_best - 0.3 >= 0
    r["gate_share_folds_best_ge_0.3"] = float((best_f >= 0.3).mean())
    _, r["gap_minus_0.10_ci_lo"], r["gap_minus_0.10_ci_hi"] = nb(gap_f[:, j] - 0.10)
    if lab == "poorly predicted":
        r["test"] = "0.3 - r2_best > 0"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(0.3 - best_f)
        r["share_folds_condition_holds"] = float((best_f < 0.3).mean())
        crosses, earlier = r["d_ci_lo"] < 0, []
    elif cat in RULES:
        r["test"] = "d = R2_model - 0.8 r2_best >= 0"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(dd[cat])
        r["share_folds_condition_holds"] = float((dd[cat] >= 0).mean())
        crosses, earlier = r["d_ci_lo"] < 0, RULES[:RULES.index(cat)]
    elif lab == "residual":
        r["test"] = "gap - 0.10 >= 0 (gap = R2_full - best reference, per fold)"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(gap_f[:, j] - 0.10)
        r["share_folds_condition_holds"] = float((gap_f[:, j] >= 0.10).mean())
        crosses, earlier = r["d_ci_lo"] < 0, RULES
    else:
        cand = {c: v for c, v in dd.items() if not np.isnan(v).any()}
        c = max(cand, key=lambda c: cand[c].mean())
        r["test"] = f"closest failed rule '{c}': d < 0; and gap - 0.10 < 0"
        r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(cand[c])
        r["share_folds_condition_holds"] = float(((cand[c] < 0) & (gap_f[:, j] < 0.10)).mean())
        crosses, earlier = (r["d_ci_hi"] > 0) or (r["gap_minus_0.10_ci_hi"] > 0), []
    r["borderline"] = bool(crosses)
    r["gate_borderline"] = bool(r["gate_lo"] < 0 < r["gate_hi"]) if lab != "poorly predicted" else bool(r["d_ci_lo"] < 0)
    r["earlier_rules_ci_reaches_0"] = ";".join(c for c in earlier if not np.isnan(dd[c]).any() and nb(dd[c])[2] >= 0)
    rows.append(r)
unc = pd.DataFrame(rows)
unc.to_csv(RESULTS / "audit_uncertainty_r2.csv", index=False)
bl = unc.groupby("label_revised").agg(n=("borderline", "size"), borderline=("borderline", "sum"),
                                      gate_borderline=("gate_borderline", "sum"),
                                      earlier_rule_reaches_0=("earlier_rules_ci_reaches_0", lambda v: int((v != "").sum())),
                                      median_halfwidth=("d_ci_hi", "size")).reindex(CATS)
hw = unc.assign(hw=(unc.d_ci_hi - unc.d_ci_lo) / 2).groupby("label_revised").hw.median().reindex(CATS)
bl["median_ci_halfwidth"] = hw
bl["median_d"] = unc.groupby("label_revised").d_mean.median().reindex(CATS)
bl["median_share_folds_condition_holds"] = unc.groupby("label_revised").share_folds_condition_holds.median().reindex(CATS)
bl["all_15_folds"] = unc[unc.share_folds_condition_holds == 1].groupby("label_revised").size().reindex(CATS).fillna(0).astype(int)
bl = bl.drop(columns="median_halfwidth")
print(bl.round(3).to_string())
print("borderline total:", int(unc.borderline.sum()), "of", T, "; non-poor:", int(unc[unc.label_revised != "poorly predicted"].borderline.sum()),
      "of", int(ok.sum()), "; gate borderline among labelled:", int(unc[unc.label_revised != "poorly predicted"].gate_borderline.sum()))

# ---------------- per-miRNA output: label_v3 columns + ceiling-based columns ----------------
out = v3.copy()
unfiltered = ["r2_host", "r2_host_cov", "r2_host_state_20", "r2_host_locus_state20"]
for c in unfiltered:
    out[f"{c}_unfiltered"] = out[c]
nmask = np.array([host_status[t] == "not measurable" for t in targets])
out.loc[nmask, "r2_host"] = np.nan
out.loc[nmask, "r2_host_cov"] = np.nan
for k in (5, 10, 20, 50):
    out.loc[nmask, f"r2_host_state_{k}"] = out.loc[nmask, f"r2_state_{k}"]
for c, f_ in [("r2_locus_own", "r2_locus_own_500kb"), ("r2_locus_paralog", "r2_locus_paralog_500kb"), ("r2_locus_any", "r2_locus_any_500kb"),
              ("r2_locus_state20", "r2_locus_state20_500kb"), ("r2_host_locus_state20", "r2_host_locus_state20_500kb")]:
    out[c] = MEAN[f_]
out["host_status"] = host_status.values
out["host_measurable"] = (host_status == "measurable").values
out["flags"] = [(f + "; " if f else "") + "host not measurable" if n else f for f, n in zip(out["flags"].fillna(""), nmask)]
out["r2_best"] = base.r2_best.values
out["r2_best_ref"] = base.best_ref.values
out["gap_full_minus_best_ref"] = base.gap.values
out["r2_best_of_means"] = stageB.r2_best.values
out["share_best_ref_over_r2_best"] = base.best_ref.values / base.r2_best.values
for n, c in [("host", "r2_host"), ("state_20", "r2_state_20"), ("host_state_20", "r2_host_state_20"),
             ("locus_state20", "r2_locus_state20"), ("host_locus_state20", "r2_host_locus_state20")]:
    out[f"share_{n}_over_r2_best"] = out[c] / out.r2_best
out["label_r2"] = base.label.values
out["label_v3_split"] = cur_split.values
out["label_changed_vs_v3"] = changed.values
out["label_r2_hostfilter_only"] = stageA.label.values
out["label_r2_best_of_means"] = stageB.label.values
for n, tag in [("k=10", "k10"), ("k=50", "k50"), ("window +-250 kb", "win250kb"), ("window +-1 Mb", "win1Mb"),
               ("PCs without host and +-500 kb target-locus genes", "pc_excl"), ("threshold 0.7", "thr0.7"), ("threshold 0.9", "thr0.9")]:
    out[f"label_r2_{tag}"] = labs[n].label.values
out["share_4_settings_agree"] = share_agree
out["borderline"] = unc.set_index("mirna").borderline.reindex(out.index).values
out.reset_index().to_csv(RESULTS / "audit_per_mirna_r2.csv", index=False)

meta = {
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "step0_reproduced_current_counts": d["step0"], "host_status_counts": host_status.value_counts().to_dict(),
    "n_host_not_measurable": int(nmask.sum()),
    "host_not_measurable_hosts": pd.Series([hosts.loc[t, "host_genes"] for t in nm]).value_counts().to_dict(),
    "meg8": meg8_info,
    "fit_checks_max_abs_diff_vs_old_caches": d["checks"], "fit_min": d["fit_min"],
    "rule": __doc__.split("Revised rule")[1].split("Outputs:")[0].strip(),
    "counts": {"v3_current": counts(cur_split), "A_host_filter_only": counts(stageA.label),
               "B_ceiling_best_of_means": counts(stageB.label), "revised": counts(base.label)},
    "n_changed": {"v3_to_A": int((stageA.label != cur_split).sum()), "v3_to_B": int((stageB.label != cur_split).sum()),
                  "v3_to_revised_incl_split": int(changed.sum()),
                  "v3_to_revised_excl_split": int((base.label.replace({"own locus": "locus", "paralog locus": "locus"}) != cur).sum())},
    "transitions_v3_to_revised": trans.to_dict(orient="index"),
    "median_shares": share, "r2_best_gt_r2_full_by_0.01": r2best_gt_full,
    "newly_not_poorly_predicted": list(newly_not_poor),
    "settings": sens.set_index("setting")[CATS + ["n_changed_vs_revised_baseline"]].to_dict(orient="index"),
    "stability": {"n_all_4_settings_agree": n_all_agree, "n_all_4_agree_non_poor": n_all_agree_nonpoor,
                  "mean_agreement_by_label": agree_by_label,
                  "transitions_k10": pd.crosstab(base.label, labs["k=10"].label).to_dict(orient="index"),
                  "transitions_k50": pd.crosstab(base.label, labs["k=50"].label).to_dict(orient="index")},
    "pc_exclusion": pc_share,
    "uncertainty": {"per_category": bl.round(4).fillna("").to_dict(orient="index"), "n_borderline": int(unc.borderline.sum()),
                    "n_borderline_non_poor": int(unc[unc.label_revised != "poorly predicted"].borderline.sum()),
                    "n_non_poor": int(ok.sum()),
                    "method": "Nadeau-Bengio corrected resampled t over 15 folds (3 repeats x 5 folds): half-width = t(0.975, 14) * sqrt((1/15 + 0.25) var(d)); "
                              "d per fold = R2_label_model - 0.8 * best_f with best_f = max(r2_full_f, reference models_f) (per-fold maximum, "
                              "the same quantity that defines r2_best and the residual gap); residual d = gap_f - 0.10; poorly predicted d = 0.3 - best_f; "
                              "borderline if the interval crosses 0 (partly explained: upper limit of the closest explain rule's d or of gap - 0.10 > 0). "
                              "gate_borderline: interval of r2_best - 0.3 includes 0 for a labelled miRNA"},
}
json.dump(meta, open(CACHE / "meta_labels_r2.json", "w"), indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else (int(x) if isinstance(x, np.integer) else float(x)))
print("written")
