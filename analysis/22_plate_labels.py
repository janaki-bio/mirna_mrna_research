"""Attribution labels under plate-grouped CV and under plate-adjusted patient CV.

Applies the main rules of 18_ceiling_model.py unchanged:
  ceiling model m* = argmax mean R2 over {r2_full + six reference models}; r2_best = its mean R2; poorly predicted
  r2_best < 0.3; host / tumour state / host + state / locus (own or paralog, locus-only own vs paralog R2) at
  R2 >= 0.8 r2_best; residual gap (r2_full - best reference by mean R2) >= 0.10; else partly explained. Host models only
  where a host gene passes the expression filter.
  Nadeau-Bengio (t14, 1/15 + 0.25) on the ceiling model's per-fold values; robust = label unchanged at +-250 kb / +-1 Mb
  and with host/local genes removed from the components, and not borderline; 'also k' = also unchanged at
  k = 10 / 20 / 50.
Inputs: cache/audit_plate_fold_records.pkl (plate-grouped CV) and cache/audit_plateadj_sym_fold_records.pkl (patient CV,
symmetric plate adjustment) from 21_plate_cv_fits.py; results/audit_per_mirna_r2.csv (patient-CV labels, 18_);
cache/audit_r2_fold_records.pkl (patient per-fold R2); results/audit_clinical_r2.csv (19_).
Outputs: audit_per_mirna_plate.csv, audit_robustness_plate.csv, audit_uncertainty_plate.csv, audit_clinical_plate.csv,
audit_plate_adjusted_sym.csv, cache/meta_plate_labels.json."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from common import CACHE, RESULTS

warnings.filterwarnings("ignore")
pat = pd.read_csv(RESULTS / "audit_per_mirna_r2.csv").set_index("mirna")
targets = list(pat.index)
Fpat = pd.read_pickle(CACHE / "audit_r2_fold_records.pkl")["F"]
D_plate = pd.read_pickle(CACHE / "audit_plate_fold_records.pkl")
D_adj = pd.read_pickle(CACHE / "audit_plateadj_sym_fold_records.pkl")
assert D_plate["targets"] == targets == D_adj["targets"]
T, J = len(targets), 15
tcrit, TT = stats.t.ppf(0.975, J - 1), 0.25
RULES = ["host", "tumour state", "host + state", "locus"]
CATS = ["host", "tumour state", "host + state", "own locus", "paralog locus", "residual", "partly explained", "poorly predicted"]
wn = lambda w: f"{w // 1000}kb" if w < 1_000_000 else f"{w // 1_000_000}Mb"
counts = lambda lab: {c: int((lab == c).sum()) for c in CATS}


class Design:
    def __init__(self, F):
        self.F = F; self.M = {c: np.nanmean(a, axis=0) for c, a in F.items()}

    def cols(self, k=20, w=500_000, excl=False):
        st, hst = ("r2_state20_excl", "r2_host_state20_excl") if excl else (f"r2_state_{k}", f"r2_host_state_{k}")
        return ({"host": "r2_host", "tumour state": st, "host + state": hst, "locus": f"r2_locus_state{k}_{wn(w)}"},
                ["r2_host", st, hst, f"r2_locus_any_{wn(w)}", f"r2_locus_state{k}_{wn(w)}", f"r2_host_locus_state{k}_{wn(w)}"])

    def ceiling(self, refs):
        names = ["r2_full"] + refs
        M = np.vstack([self.M[c] for c in names]); M = np.where(np.isnan(M), -np.inf, M)
        star = M.argmax(0); A = np.stack([self.F[c] for c in names]); bref = M[1:].argmax(0) + 1
        take = lambda idx: np.take_along_axis(A, idx[None, None, :].repeat(J, 1), axis=0)[0]
        best_f, rb = take(star), take(bref)
        return best_f, A[0] - rb, np.array(names)[star], np.array(names)[bref]

    def attribute(self, k=20, w=500_000, excl=False, f=0.8):
        rules, refs = self.cols(k, w, excl)
        best_f, gap_f, mstar, mref = self.ceiling(refs)
        best, gap = best_f.mean(0), gap_f.mean(0)
        lab = np.array(["partly explained"] * T, dtype=object); done = best < 0.3; lab[done] = "poorly predicted"
        for cat in RULES:
            m = self.M[rules[cat]]; hit = ~done & ~np.isnan(m) & (m >= f * best); lab[hit], done[hit] = cat, True
        own, par = self.M[f"r2_locus_own_{wn(w)}"], self.M[f"r2_locus_paralog_{wn(w)}"]
        is_par = ~np.isnan(par) & (np.isnan(own) | (par > np.nan_to_num(own, nan=-np.inf)))
        lab = np.where(lab == "locus", np.where(is_par, "paralog locus", "own locus"), lab)
        lab[~done & (gap >= 0.10)] = "residual"
        return pd.DataFrame({"label": lab, "r2_best": best, "gap": gap, "ceiling_model": mstar,
                             "best_ref_model": mref}, index=targets)

    def uncertainty(self, lab):
        rules0, refs0 = self.cols()
        best_f, gap_f, mstar, _ = self.ceiling(refs0)
        nb = lambda v: (v.mean(), v.mean() - tcrit * np.sqrt((1 / J + TT) * v.var(ddof=1)), v.mean() + tcrit * np.sqrt((1 / J + TT) * v.var(ddof=1)))
        rows = []
        for j, t in enumerate(targets):
            l = lab[t]; cat = "locus" if "locus" in l else l; bf = best_f[:, j]
            dd = {c: self.F[rules0[c]][:, j] - 0.8 * bf for c in RULES}
            r = {"mirna": t, "label": l, "ceiling_model": mstar[j], "ceiling_model_fold_sd": bf.std(ddof=1), "r2_full": self.M["r2_full"][j], "r2_best": bf.mean(), "gap_mean": gap_f[:, j].mean()}
            _, r["gate_lo"], r["gate_hi"] = nb(bf - 0.3)
            _, r["gap_minus_0.10_ci_lo"], r["gap_minus_0.10_ci_hi"] = nb(gap_f[:, j] - 0.10)
            if l == "poorly predicted":
                r["test"] = "0.3 - r2_best > 0"; r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(0.3 - bf)
                r["share_folds_condition_holds"] = float((bf < 0.3).mean()); crosses, earlier = r["d_ci_lo"] < 0, []
            elif cat in RULES:
                r["test"] = "d = R2_model - 0.8 r2_best >= 0"; r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(dd[cat])
                r["share_folds_condition_holds"] = float((dd[cat] >= 0).mean()); crosses, earlier = r["d_ci_lo"] < 0, RULES[:RULES.index(cat)]
            elif l == "residual":
                r["test"] = "gap - 0.10 >= 0"; r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(gap_f[:, j] - 0.10)
                r["share_folds_condition_holds"] = float((gap_f[:, j] >= 0.10).mean()); crosses, earlier = r["d_ci_lo"] < 0, RULES
            else:
                cand = {c: v for c, v in dd.items() if not np.isnan(v).any()}; c = max(cand, key=lambda c: cand[c].mean())
                r["test"] = f"closest failed rule '{c}': d < 0; and gap - 0.10 < 0"; r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(cand[c])
                r["share_folds_condition_holds"] = float(((cand[c] < 0) & (gap_f[:, j] < 0.10)).mean())
                crosses, earlier = (r["d_ci_hi"] > 0) or (r["gap_minus_0.10_ci_hi"] > 0), []
            r["borderline"] = bool(crosses)
            r["gate_borderline"] = bool(r["gate_lo"] < 0 < r["gate_hi"]) if l != "poorly predicted" else bool(r["d_ci_lo"] < 0)
            r["earlier_rules_ci_reaches_0"] = ";".join(c for c in earlier if not np.isnan(dd[c]).any() and nb(dd[c])[2] >= 0)
            rows.append(r)
        return pd.DataFrame(rows).set_index("mirna")


SET = {"baseline (k=20, +-500 kb, 0.8)": {}, "k=10": dict(k=10), "k=50": dict(k=50), "window +-250 kb": dict(w=250_000), "window +-1 Mb": dict(w=1_000_000),
       "PCs without host and +-500 kb target-locus genes": dict(excl=True), "threshold 0.7": dict(f=0.7), "threshold 0.9": dict(f=0.9)}
TAGS = {"k10": "k=10", "k50": "k=50", "win250kb": "window +-250 kb", "win1Mb": "window +-1 Mb", "pc_excl": "PCs without host and +-500 kb target-locus genes",
        "thr0.7": "threshold 0.7", "thr0.9": "threshold 0.9"}


def analyse(D):
    a = {"D": D, "main": D.attribute(), "labs": {n: D.attribute(**kw) for n, kw in SET.items()}}
    a["lab"] = a["main"].label
    a["ok"] = (a["lab"] != "poorly predicted").values
    a["unc"] = D.uncertainty(a["lab"])
    same = lambda n: a["labs"][n].label == a["lab"]
    a["loc_ok"] = same("window +-250 kb") & same("window +-1 Mb") & same("PCs without host and +-500 kb target-locus genes")
    a["k_ok"] = same("k=10") & same("k=50")
    a["robust"] = a["loc_ok"] & ~a["unc"].borderline; a["robust_k"] = a["robust"] & a["k_ok"]
    a["share_agree"] = np.mean([same(n).values for n in ["k=10", "k=50", "window +-250 kb", "window +-1 Mb"]], axis=0)
    rules0, refs0 = D.cols(); r2b = a["main"].r2_best.values
    a["bref"] = np.nanmax(np.vstack([D.M[c] for c in refs0]), axis=0)
    sh = {}
    for nm, c in [("host", "r2_host"), ("state_20", "r2_state_20"), ("host_state_20", "r2_host_state_20"), ("locus_state20", "r2_locus_state20_500kb"), ("host_locus_state20", "r2_host_locus_state20_500kb")]:
        a[f"share_{nm}"] = D.M[c] / r2b
        sh[nm] = {"median": float(np.nanmedian(a[f"share_{nm}"][a["ok"]])), "n_defined": int(np.isfinite(a[f"share_{nm}"][a["ok"]]).sum())}
    sh["best_ref"] = {"median": float(np.median((a["bref"] / r2b)[a["ok"]])), "n_defined": int(a["ok"].sum())}
    sh["r2_full"] = {"median": float(np.median((D.M["r2_full"] / r2b)[a["ok"]])), "n_defined": int(a["ok"].sum())}
    a["shares"] = sh
    return a


def bl_table(u):
    g = u.groupby("label")
    return pd.DataFrame({"n": g.size(), "borderline": g.borderline.sum(), "gate_borderline": g.gate_borderline.sum(),
                         "earlier_rule_reaches_0": g.earlier_rules_ci_reaches_0.apply(lambda v: int((v != "").sum())),
                         "median_ci_halfwidth": ((u.d_ci_hi - u.d_ci_lo) / 2).groupby(u.label).median(),
                         "all_15_folds": u[u.share_folds_condition_holds == 1].groupby("label").size()}).reindex(CATS).fillna(0)


P, A = analyse(Design(D_plate["F"])), analyse(Design(D_adj["F"]))
# patient CV analysis recomputed from the same per-fold arrays (reference; must reproduce the stored main labels)
Pt = analyse(Design(Fpat))
assert (Pt["lab"] == pat.label_r2).all() and np.allclose(Pt["main"].r2_best, pat.r2_best), "patient main labels not reproduced"
print("patient main labels reproduced:", counts(Pt["lab"]))
print("PLATE CV counts:", counts(P["lab"]), "not poor", int(P["ok"].sum()))
print("ADJ (sym) counts:", counts(A["lab"]), "not poor", int(A["ok"].sum()))
cx = pd.crosstab(pat.label_r2, P["lab"], rownames=["patient CV"], colnames=["plate CV"])
cx_adj = pd.crosstab(pat.label_r2, A["lab"], rownames=["patient CV"], colnames=["adjusted patient CV"])
cx_adj_plate = pd.crosstab(P["lab"], A["lab"], rownames=["plate CV"], colnames=["adjusted patient CV"])
print(cx.to_string()); print("labels changed plate vs patient:", int((P["lab"] != pat.label_r2).sum()))
print(cx_adj.to_string()); print("labels changed adjusted vs patient:", int((A["lab"] != pat.label_r2).sum()))
print("shares plate:", {k: round(v["median"], 3) for k, v in P["shares"].items()}); print("shares patient:", {k: round(v["median"], 3) for k, v in Pt["shares"].items()})
print("shares adj:", {k: round(v["median"], 3) for k, v in A["shares"].items()})
blP, blA, blPt = bl_table(P["unc"]), bl_table(A["unc"]), bl_table(Pt["unc"])
print(blP.round(3).to_string()); print("borderline plate", int(P["unc"].borderline.sum()), "non-poor", int(P["unc"][P["ok"]].borderline.sum()), "| adj", int(A["unc"].borderline.sum()), "| patient", int(Pt["unc"].borderline.sum()))
rob = lambda a: {"unchanged_windows_and_pc_exclusion": int(a["loc_ok"].sum()), "k10_k50_unchanged": int(a["k_ok"].sum()), "not_borderline": int((~a["unc"].borderline).sum()),
                 "robust": int(a["robust"].sum()), "robust_also_k": int(a["robust_k"].sum()), "robust_non_poor": int((a["robust"] & a["ok"]).sum()),
                 "robust_also_k_non_poor": int((a["robust_k"] & a["ok"]).sum()),
                 "robust_by_label": a["robust"].groupby(a["lab"]).sum().reindex(CATS).astype(int).to_dict(),
                 "robust_also_k_by_label": a["robust_k"].groupby(a["lab"]).sum().reindex(CATS).astype(int).to_dict()}
robP, robA, robPt = rob(P), rob(A), rob(Pt)
print("robust plate", {k: v for k, v in robP.items() if "by_label" not in k}); print("robust patient", {k: v for k, v in robPt.items() if "by_label" not in k})

# ---- robustness csv ----
rows = []
for design, a in (("plate-grouped CV", P), ("patient CV, symmetric plate adjustment", A), ("patient CV (reference, r2 main)", Pt)):
    for n, kw in SET.items():
        o = a["labs"][n]; r = {"design": design, "setting": n, "k": kw.get("k", 20), "window_kb": kw.get("w", 500_000) // 1000, "fraction": kw.get("f", 0.8)} | counts(o.label)
        r["n_changed_vs_own_baseline"] = int((o.label != a["lab"]).sum()); r["n_r2_best_ge_0.3"] = int((o.label != "poorly predicted").sum()); rows.append(r)
sens = pd.DataFrame(rows); sens.to_csv(RESULTS / "audit_robustness_plate.csv", index=False)
print(sens[sens.design.str.startswith("plate")].drop(columns=["design", "fraction"]).to_string(index=False))

# ---- uncertainty csv ----
uo = P["unc"].rename(columns={"label": "label_plate"}); uo["label_patient"] = pat.label_r2; uo["borderline_patient"] = Pt["unc"].borderline
uo.reset_index().to_csv(RESULTS / "audit_uncertainty_plate.csv", index=False)

# ---- clinical csv ----
cl_prev = pd.read_csv(RESULTS / "audit_clinical_r2.csv", index_col=0)
Fp = D_plate["F"]; cl = pd.DataFrame({"r2_full": P["D"].M["r2_full"], "r2_best": P["main"].r2_best, "label_plate": P["lab"]}, index=targets)
for c in ["r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]:
    cl[c] = Fp[c].mean(0); cl[c + "_fold_sd"] = Fp[c].std(0, ddof=1); cl[f"share_{c[3:]}_over_r2_best"] = cl[c] / cl.r2_best; cl[f"share_{c[3:]}_over_r2_full"] = cl[c] / cl.r2_full
    cl[c + "_patient"] = cl_prev[c]; cl[f"share_{c[3:]}_over_r2_best_patient"] = cl_prev[f"share_{c[3:]}_over_r2_best"]
cl.to_csv(RESULTS / "audit_clinical_plate.csv", index_label="mirna")
okf = cl.r2_full >= 0.3
clinsum = {c: {"median_r2_all_337": float(cl[c].median()), "median_share_over_r2_best_non_poor": float(cl.loc[P["ok"], f"share_{c[3:]}_over_r2_best"].median()),
               "median_share_over_r2_full_r2full_ge_0.3": float(cl.loc[okf, f"share_{c[3:]}_over_r2_full"].median()),
               "max_share_over_r2_best_non_poor": float(cl.loc[P["ok"], f"share_{c[3:]}_over_r2_best"].max()),
               "n_ge_0.8_r2_best_non_poor": int((cl.loc[P["ok"], f"share_{c[3:]}_over_r2_best"] >= 0.8).sum()),
               "patient_median_share_over_r2_best_non_poor": float(cl_prev.loc[pat.label_r2 != "poorly predicted", f"share_{c[3:]}_over_r2_best"].median())}
           for c in ["r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]}
print(pd.DataFrame(clinsum).T.round(3).to_string())

# ---- adjusted symmetric csv ----
adf = pd.DataFrame({"label_patient_unadjusted": pat.label_r2, "label_plate_cv": P["lab"], "label_patient_adjusted": A["lab"],
                    "label_changed_adj_vs_patient": A["lab"] != pat.label_r2, "r2_best_patient": pat.r2_best, "r2_best_adjusted": A["main"].r2_best,
                    "ceiling_model_adjusted": A["main"].ceiling_model, "borderline_adjusted": A["unc"].borderline})
for nm, c in [("full", "r2_full"), ("state_20", "r2_state_20"), ("host_state_20", "r2_host_state_20"), ("locus_state20", "r2_locus_state20_500kb")]:
    adf[f"{c.replace('_500kb', '')}_patient"] = pat[c.replace("_500kb", "")]; adf[f"{c.replace('_500kb', '')}_adjusted"] = A["D"].M[c]
for nm in ("host", "state_20", "host_state_20", "locus_state20", "host_locus_state20"):
    adf[f"share_{nm}_over_r2_best_patient"] = pat[f"share_{nm}_over_r2_best"]; adf[f"share_{nm}_over_r2_best_adjusted"] = A[f"share_{nm}"]
adf["robust_label_adjusted"] = A["robust"]; adf["robust_label_also_k_adjusted"] = A["robust_k"]
adf.to_csv(RESULTS / "audit_plate_adjusted_sym.csv", index_label="mirna")

# ---- per-miRNA plate file ----
DP = P["D"]
rec_cols = {"r2_full": DP.M["r2_full"], "r2_full_sd": Fp["r2_full"].std(0, ddof=1), "r2_host": DP.M["r2_host"]}
for k in (5, 10, 20, 50):
    rec_cols[f"r2_state_{k}"] = DP.M[f"r2_state_{k}"]; rec_cols[f"r2_host_state_{k}"] = DP.M[f"r2_host_state_{k}"]
for c, f in [("r2_locus_own", "r2_locus_own_500kb"), ("r2_locus_paralog", "r2_locus_paralog_500kb"), ("r2_locus_any", "r2_locus_any_500kb"),
             ("r2_locus_state20", "r2_locus_state20_500kb"), ("r2_host_locus_state20", "r2_host_locus_state20_500kb")]:
    rec_cols[c] = DP.M[f]
rec_cols |= {"ceiling_model": P["main"].ceiling_model.values, "r2_best": P["main"].r2_best.values, "r2_best_model_fold_sd": P["unc"].ceiling_model_fold_sd.values,
             "gap_full_minus_best_ref": P["main"].gap.values, "r2_best_ref": P["bref"], "share_best_ref_over_r2_best": P["bref"] / P["main"].r2_best.values,
             "label_r2": P["lab"].values, "share_4_settings_agree": P["share_agree"], "borderline": P["unc"].borderline.values,
             "gate_borderline": P["unc"].gate_borderline.values, "robust_label": P["robust"].values, "robust_label_also_k": P["robust_k"].values}
for nm in ("host", "state_20", "host_state_20", "locus_state20", "host_locus_state20"):
    rec_cols[f"share_{nm}_over_r2_best"] = P[f"share_{nm}"]
for tag, n in TAGS.items():
    rec_cols[f"label_r2_{tag}"] = P["labs"][n].label.values
par_beats = np.where(np.isnan(rec_cols["r2_locus_paralog"]), np.nan, (rec_cols["r2_locus_paralog"] > np.nan_to_num(rec_cols["r2_locus_own"], nan=-np.inf)).astype(float))
rec_cols["paralog_locus_beats_own"] = par_beats
STATIC = ["mir_gene", "n_samples", "frac_zero", "frac_rpm_gt1", "mean_log2rpm", "paralog_group", "cluster_id", "host_genes", "host_type", "neighbours_50kb",
          "top_pos_genes", "top_neg_genes", "prog1", "prog1_rho", "prog2", "prog2_rho", "frac_cross_mapped", "own_locus_genes", "paralog_locus_genes",
          "n_own_locus_genes", "n_paralog_locus_genes", "paralog_members", "host_status", "host_measurable"]
out, not_recomputed = pd.DataFrame(index=pd.Index(targets, name="mirna")), []
for c in pat.columns:
    if c in STATIC: out[c] = pat[c].values
    elif c in rec_cols:
        out[c] = rec_cols[c]; out[f"{c}_patient"] = pat[c].values
    else:
        out[f"{c}_patient"] = pat[c].values; not_recomputed.append(c)
for c in rec_cols:
    if c not in out.columns: out[c] = rec_cols[c]
out["label_changed_vs_patient"] = (P["lab"] != pat.label_r2).values
for c in ["r2_pam50_cpe", "r2_pam50_cpe_lump_indicator", "r2_pam50_cpe_lump_meanimputed"]: out[c + "_plate"] = cl[c].values
out["label_patient_adjusted"] = A["lab"].values
out.reset_index().to_csv(RESULTS / "audit_per_mirna_plate.csv", index=False)

# ---- item 4: paralog comparison ----
def paral(Dd, fullcol="r2_full"):
    res = {}
    for w in ("250kb", "500kb", "1Mb"):
        own, par, full = Dd.M[f"r2_locus_own_{w}"], Dd.M[f"r2_locus_paralog_{w}"], Dd.M[fullcol]
        has = ~np.isnan(par)
        res[w] = {m: {"n": int((has & (full >= 0.3)).sum()),
                      "paralog_beats_own": int((has & (full >= 0.3) & (par > np.nan_to_num(own, nan=-np.inf))).sum()),
                      "paralog_beats_own_by_0.1": int((has & (full >= 0.3) & (par - np.nan_to_num(own, nan=-np.inf) > 0.1)).sum()),
                      "median_own": float(np.nanmedian(own[has & (full >= 0.3)])), "median_paralog": float(np.nanmedian(par[has & (full >= 0.3)]))} for m in ["x"]}
        res[w] = res[w]["x"]
    return res
Dpat = Pt["D"]
pc_same = {"plate": paral(DP), "patient": paral(Dpat)}
sel_pat = (pat.r2_full >= 0.3).values   # the miRNAs with r2_full >= 0.3 under patient CV, under both CVs
def paral_fixed(Dd, sel):
    own, par = Dd.M["r2_locus_own_500kb"], Dd.M["r2_locus_paralog_500kb"]; has = ~np.isnan(par) & sel
    return {"n": int(has.sum()), "paralog_beats_own": int((has & (par > np.nan_to_num(own, nan=-np.inf))).sum()),
            "paralog_beats_own_by_0.1": int((has & (par - np.nan_to_num(own, nan=-np.inf) > 0.1)).sum()),
            "median_own": float(np.nanmedian(own[has])), "median_paralog": float(np.nanmedian(par[has]))}
pc_fixed = {"plate": paral_fixed(DP, sel_pat), "patient": paral_fixed(Dpat, sel_pat)}
print("paralog (r2_full >= 0.3 within each CV):", json.dumps(pc_same)); print("paralog (patient r2_full >= 0.3 population):", pc_fixed)
sel_tab = pat.r2_full >= 0.3
ptab = pd.DataFrame({"paralog_group": pat.paralog_group, "paralog_members": pat.paralog_members, "r2_full_patient": pat.r2_full, "r2_full_plate": DP.M["r2_full"],
                     "own_patient": Dpat.M["r2_locus_own_500kb"], "paralog_patient": Dpat.M["r2_locus_paralog_500kb"],
                     "own_plate": DP.M["r2_locus_own_500kb"], "paralog_plate": DP.M["r2_locus_paralog_500kb"],
                     "label_patient": pat.label_r2, "label_plate": P["lab"]}, index=targets)
ptab = ptab[~ptab.paralog_plate.isna() & (sel_tab | (DP.M["r2_full"] >= 0.3))]
ptab["beats_patient"] = ptab.paralog_patient > ptab.own_patient.fillna(-np.inf); ptab["beats_plate"] = ptab.paralog_plate > ptab.own_plate.fillna(-np.inf)
ptab["flip"] = ptab.beats_patient != ptab.beats_plate
print(ptab.round(2).sort_values("paralog_plate", ascending=False).head(15).to_string())
print("flips:", int(ptab.flip.sum()), "of", len(ptab))
lab_counts_locus = {"patient": {c: int((pat.label_r2 == c).sum()) for c in ["own locus", "paralog locus"]}, "plate": {c: int((P["lab"] == c).sum()) for c in ["own locus", "paralog locus"]}}

tj = lambda x: bool(x) if isinstance(x, np.bool_) else (int(x) if isinstance(x, np.integer) else float(x))
json.dump({"counts": {"plate_cv": counts(P["lab"]), "patient_cv": counts(Pt["lab"]), "patient_cv_symmetric_plate_adjusted": counts(A["lab"])},
           "n_not_poor": {"plate_cv": int(P["ok"].sum()), "patient_cv": int(Pt["ok"].sum()), "adjusted": int(A["ok"].sum())},
           "n_changed": {"plate_vs_patient": int((P["lab"] != pat.label_r2).sum()), "adjusted_vs_patient": int((A["lab"] != pat.label_r2).sum()),
                         "adjusted_vs_plate": int((A["lab"] != P["lab"]).sum())},
           "crosstab_patient_vs_plate": cx.to_dict(orient="index"), "crosstab_patient_vs_adjusted": cx_adj.to_dict(orient="index"),
           "crosstab_plate_vs_adjusted": cx_adj_plate.to_dict(orient="index"),
           "median_shares_of_r2_best": {"plate_cv": P["shares"], "patient_cv": Pt["shares"], "adjusted": A["shares"]},
           "ceiling_model_counts": {"plate_cv": P["main"].ceiling_model.value_counts().to_dict(), "patient_cv": Pt["main"].ceiling_model.value_counts().to_dict(), "adjusted": A["main"].ceiling_model.value_counts().to_dict()},
           "uncertainty": {"plate_cv": blP.round(4).to_dict(orient="index"), "patient_cv": blPt.round(4).to_dict(orient="index"), "adjusted": blA.round(4).to_dict(orient="index"),
                           "borderline": {"plate_cv": int(P["unc"].borderline.sum()), "patient_cv": int(Pt["unc"].borderline.sum()), "adjusted": int(A["unc"].borderline.sum())},
                           "borderline_non_poor": {"plate_cv": int(P["unc"][P["ok"]].borderline.sum()), "patient_cv": int(Pt["unc"][Pt["ok"]].borderline.sum()), "adjusted": int(A["unc"][A["ok"]].borderline.sum())}},
           "robust": {"plate_cv": robP, "patient_cv": robPt, "adjusted": robA},
           "settings_plate_cv": sens[sens.design == "plate-grouped CV"].set_index("setting")[CATS + ["n_changed_vs_own_baseline"]].to_dict(orient="index"),
           "settings_adjusted": sens[sens.design.str.contains("symmetric")].set_index("setting")[CATS + ["n_changed_vs_own_baseline"]].to_dict(orient="index"),
           "clinical": clinsum, "paralog_comparison": {"within_each_cv_r2_full_ge_0.3": pc_same, "patient_r2_full_ge_0.3_population": pc_fixed, "n_flips_own_vs_paralog": int(ptab.flip.sum()),
                                                        "n_in_table": int(len(ptab)), "locus_label_counts": lab_counts_locus,
                                                        "table": ptab.round(4).reset_index().to_dict(orient="records")},
           "columns_not_recomputed_under_plate_cv_only_patient_suffix": not_recomputed,
           "fits": {"plate_fit_min": D_plate["fit_min"], "adj_fit_min": D_adj["fit_min"], "plate_checks_vs_33": D_plate["checks"], "plate_fold_test_sizes": D_plate["fold_test_sizes"]},
           }, open(CACHE / "meta_plate_labels.json", "w"), indent=2, default=tj)
print("written")
