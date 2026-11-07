"""Final attribution labels: the ceiling model is the model with the highest mean R2.

  ceiling model m* = argmax over {r2_full, r2_host, r2_state_k, r2_host_state_k, r2_locus_any, r2_locus_state_k,
  r2_host_locus_state_k} of the 15-fold mean R2;  r2_best = mean R2 of m*;  best reference = best of the six reference
  models by mean R2;  gap = r2_full - best reference.  Labels use 0.3 / 0.8 x r2_best / gap >= 0.10.
  Uncertainty (Nadeau-Bengio, t14, 1/15 + 0.25): per-fold d uses the per-fold R2 of m* (best_f = R2_f[m*]) and of the best
  reference model (gap_f); explain-rule d_f = R2_f[rule model] - 0.8 best_f; poorly predicted d_f = 0.3 - best_f;
  residual d_f = gap_f - 0.10.
  The per-fold-maximum ceiling (best_f = max over models of R2_f) of 14_attribution_labels.py is kept in the *_foldmax
  columns for comparison.
Reads cache/audit_r2_fold_records.pkl and results/prev_r2/audit_per_mirna_r2.csv (a copy of the 14_ output, made by
run_all.sh before this step).
Rewrites audit_per_mirna_r2.csv, audit_uncertainty_r2.csv and audit_robustness_r2.csv; writes audit_r2_best_moved.csv
(miRNAs whose r2_best moved) and cache/meta_r2b_labels.json."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from common import CACHE, RESULTS

warnings.filterwarnings("ignore")
d = pd.read_pickle(CACHE / "audit_r2_fold_records.pkl")
F, targets = d["F"], d["targets"]
prev = pd.read_csv(RESULTS / "prev_r2" / "audit_per_mirna_r2.csv").set_index("mirna")
assert list(prev.index) == targets
v3 = prev
T, J = len(targets), 15
tcrit, TT = stats.t.ppf(0.975, J - 1), 0.25
RULES = ["host", "tumour state", "host + state", "locus"]
CATS = ["host", "tumour state", "host + state", "own locus", "paralog locus", "residual", "partly explained", "poorly predicted"]
MEAN = {c: np.nanmean(a, axis=0) for c, a in F.items()}
wn = lambda w: f"{w // 1000}kb" if w < 1_000_000 else f"{w // 1_000_000}Mb"


def cols(k=20, w=500_000, excl=False):
    st, hst = ("r2_state20_excl", "r2_host_state20_excl") if excl else (f"r2_state_{k}", f"r2_host_state_{k}")
    return ({"host": "r2_host", "tumour state": st, "host + state": hst, "locus": f"r2_locus_state{k}_{wn(w)}"},
            ["r2_host", st, hst, f"r2_locus_any_{wn(w)}", f"r2_locus_state{k}_{wn(w)}", f"r2_host_locus_state{k}_{wn(w)}"])


def ceiling(refs, mode):
    """returns best_f (15xT), refbest_f, gap_f, ceiling model index into ['r2_full']+refs"""
    names = ["r2_full"] + refs
    M = np.vstack([MEAN[c] for c in names]); M = np.where(np.isnan(M), -np.inf, M)
    star = M.argmax(0)
    A = np.stack([F[c] for c in names])                      # models x 15 x T
    if mode == "foldmax":
        rb = np.nanmax(A[1:], axis=0)
        return np.maximum(A[0], rb), rb, A[0] - rb, star
    bref = M[1:].argmax(0) + 1
    best_f = np.take_along_axis(A, star[None, None, :].repeat(J, 1), axis=0)[0]
    rb = np.take_along_axis(A, bref[None, None, :].repeat(J, 1), axis=0)[0]
    return best_f, rb, A[0] - rb, star


def attribute(mode, k=20, w=500_000, excl=False, f=0.8):
    rules, refs = cols(k, w, excl)
    best_f, rb_f, gap_f, star = ceiling(refs, mode)
    best, gap = best_f.mean(0), gap_f.mean(0)
    lab = np.array(["partly explained"] * T, dtype=object)
    done = best < 0.3
    lab[done] = "poorly predicted"
    for cat in RULES:
        m = MEAN[rules[cat]]
        hit = ~done & ~np.isnan(m) & (m >= f * best)
        lab[hit], done[hit] = cat, True
    own, par = MEAN[f"r2_locus_own_{wn(w)}"], MEAN[f"r2_locus_paralog_{wn(w)}"]
    is_par = ~np.isnan(par) & (np.isnan(own) | (par > np.nan_to_num(own, nan=-np.inf)))
    lab = np.where(lab == "locus", np.where(is_par, "paralog locus", "own locus"), lab)
    lab[~done & (gap >= 0.10)] = "residual"
    return pd.DataFrame({"label": lab, "r2_best": best, "gap": gap, "ceiling_model": np.array(["r2_full"] + refs)[star]}, index=targets)


counts = lambda lab: {c: int((lab == c).sum()) for c in CATS}
main, fm = attribute("meanmax"), attribute("foldmax")
print("main", counts(main.label)); print("foldmax", counts(fm.label))
cur_split = prev.label_v3_split
changed_main = main.label != cur_split
print("changed vs v3: main", int(changed_main.sum()), " foldmax", int((fm.label != cur_split).sum()))
print(pd.crosstab(cur_split, main.label).to_string())
ok = (main.label != "poorly predicted").values

# ---- uncertainty ----
rules0, refs0 = cols()


def nb(v):
    m, sd = v.mean(), v.std(ddof=1)
    h = tcrit * np.sqrt((1 / J + TT) * sd ** 2)
    return m, m - h, m + h


def uncertainty(mode, lab):
    best_f, rb_f, gap_f, star = ceiling(refs0, mode)
    rows = []
    for j, t in enumerate(targets):
        l = lab[t]; cat = "locus" if "locus" in l else l
        bf = best_f[:, j]
        dd = {c: F[rules0[c]][:, j] - 0.8 * bf for c in RULES}
        r = {"mirna": t, "label": l, "r2_full": MEAN["r2_full"][j], "r2_best": bf.mean(), "gap_mean": gap_f[:, j].mean()}
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
            cand = {c: v for c, v in dd.items() if not np.isnan(v).any()}
            c = max(cand, key=lambda c: cand[c].mean())
            r["test"] = f"closest failed rule '{c}': d < 0; and gap - 0.10 < 0"; r["d_mean"], r["d_ci_lo"], r["d_ci_hi"] = nb(cand[c])
            r["share_folds_condition_holds"] = float(((cand[c] < 0) & (gap_f[:, j] < 0.10)).mean())
            crosses, earlier = (r["d_ci_hi"] > 0) or (r["gap_minus_0.10_ci_hi"] > 0), []
        r["borderline"] = bool(crosses)
        r["gate_borderline"] = bool(r["gate_lo"] < 0 < r["gate_hi"]) if l != "poorly predicted" else bool(r["d_ci_lo"] < 0)
        r["earlier_rules_ci_reaches_0"] = ";".join(c for c in earlier if not np.isnan(dd[c]).any() and nb(dd[c])[2] >= 0)
        rows.append(r)
    return pd.DataFrame(rows).set_index("mirna")


um, uf = uncertainty("meanmax", main.label), uncertainty("foldmax", fm.label)
star_names = np.array(["r2_full"] + refs0)
_, _, _, star = ceiling(refs0, "meanmax")
bf_main = ceiling(refs0, "meanmax")[0]
um.insert(1, "ceiling_model", main.ceiling_model.values)
um.insert(2, "ceiling_model_fold_sd", bf_main.std(0, ddof=1))
out_unc = um.copy()
for c in ["label", "test", "d_mean", "d_ci_lo", "d_ci_hi", "borderline", "gate_borderline", "r2_best", "share_folds_condition_holds"]:
    out_unc[f"{c}_foldmax"] = uf[c]
out_unc = out_unc.rename(columns={"label": "label_revised"})
out_unc.reset_index().to_csv(RESULTS / "audit_uncertainty_r2.csv", index=False)


def bl_table(u):
    g = u.groupby("label")
    b = pd.DataFrame({"n": g.size(), "borderline": g.borderline.sum(), "gate_borderline": g.gate_borderline.sum(),
                      "earlier_rule_reaches_0": g.earlier_rules_ci_reaches_0.apply(lambda v: int((v != "").sum())),
                      "median_ci_halfwidth": ((u.d_ci_hi - u.d_ci_lo) / 2).groupby(u.label).median(),
                      "all_15_folds": u[u.share_folds_condition_holds == 1].groupby("label").size()}).reindex(CATS)
    return b.fillna(0)


blm, blf = bl_table(um), bl_table(uf)
print("MAIN\n", blm.round(3).to_string(), "\nborderline", int(um.borderline.sum()), "non-poor", int(um[ok].borderline.sum()), "of", int(ok.sum()))
print("FOLDMAX\n", blf.round(3).to_string(), "\nborderline", int(uf.borderline.sum()))

# ---- settings (robustness) ----
SET = {"revised baseline (k=20, +-500 kb, 0.8)": {}, "k=10": dict(k=10), "k=50": dict(k=50), "window +-250 kb": dict(w=250_000),
       "window +-1 Mb": dict(w=1_000_000), "PCs without host and +-500 kb target-locus genes": dict(excl=True),
       "threshold 0.7": dict(f=0.7), "threshold 0.9": dict(f=0.9)}
labs = {m: {n: attribute(m, **kw) for n, kw in SET.items()} for m in ("meanmax", "foldmax")}
rows = []
for m, rule in (("meanmax", "MAIN: ceiling model by mean R2"), ("foldmax", "comparison: per-fold maximum")):
    for n, kw in SET.items():
        o = labs[m][n]; r = {"setting": n, "rule": rule, "k": kw.get("k", 20), "window_kb": kw.get("w", 500_000) // 1000, "fraction": kw.get("f", 0.8)}
        r |= counts(o.label); r["n_changed_vs_own_baseline"] = int((o.label != labs[m][list(SET)[0]].label).sum())
        r["n_r2_best_ge_0.3"] = int((o.label != "poorly predicted").sum()); rows.append(r)
sens = pd.DataFrame(rows); sens.to_csv(RESULTS / "audit_robustness_r2.csv", index=False)
pd.set_option("display.width", 250)
print(sens.drop(columns=["fraction"]).to_string(index=False))

# ---- robust-label columns (main rule) ----
B = main.label
same = lambda n: (labs["meanmax"][n].label == B)
loc_ok = same("window +-250 kb") & same("window +-1 Mb") & same("PCs without host and +-500 kb target-locus genes")
k_ok = same("k=10") & same("k=50")
nb_ok = ~um.borderline
robust, robust_k = loc_ok & nb_ok, loc_ok & k_ok & nb_ok
rob_summary = {"n_unchanged_windows_and_pc_exclusion": int(loc_ok.sum()), "n_not_borderline": int(nb_ok.sum()),
               "n_robust (windows + PC exclusion unchanged, not borderline)": int(robust.sum()),
               "n_robust_also_k10_k50": int(robust_k.sum()),
               "n_robust_non_poor": int((robust & ok).sum()), "n_robust_also_k_non_poor": int((robust_k & ok).sum()),
               "n_k10_k50_unchanged": int(k_ok.sum()),
               "robust_by_label": robust.groupby(B).sum().reindex(CATS).astype(int).to_dict(),
               "robust_also_k_by_label": robust_k.groupby(B).sum().reindex(CATS).astype(int).to_dict(),
               "label_counts": counts(B)}
print(json.dumps(rob_summary, indent=1))

# ---- per-miRNA output ----
o = prev.copy()
ren = ["r2_best", "r2_best_ref", "gap_full_minus_best_ref", "share_best_ref_over_r2_best", "label_r2", "label_changed_vs_v3", "share_4_settings_agree", "borderline",
       "share_host_over_r2_best", "share_state_20_over_r2_best", "share_host_state_20_over_r2_best", "share_locus_state20_over_r2_best",
       "share_host_locus_state20_over_r2_best"] + [c for c in o.columns if c.startswith("label_r2_") and c not in ("label_r2_hostfilter_only",)]
o = o.rename(columns={c: f"{c}_foldmax" for c in ren}).drop(columns=["r2_best_of_means_foldmax"] if "r2_best_of_means_foldmax" in o else [])
o = o.drop(columns=[c for c in ("r2_best_of_means", "label_r2_best_of_means_foldmax") if c in o.columns])
o["ceiling_model"] = main.ceiling_model.values
o["r2_best"] = main.r2_best.values
o["r2_best_model_fold_sd"] = bf_main.std(0, ddof=1)
o["gap_full_minus_best_ref"] = main.gap.values
o["label_r2"] = B.values
o["label_changed_vs_v3"] = changed_main.values
for tag, n in [("k10", "k=10"), ("k50", "k=50"), ("win250kb", "window +-250 kb"), ("win1Mb", "window +-1 Mb"),
               ("pc_excl", "PCs without host and +-500 kb target-locus genes"), ("thr0.7", "threshold 0.7"), ("thr0.9", "threshold 0.9")]:
    o[f"label_r2_{tag}"] = labs["meanmax"][n].label.values
o["share_4_settings_agree"] = np.mean([same(n).values for n in ["k=10", "k=50", "window +-250 kb", "window +-1 Mb"]], axis=0)
o["borderline"] = um.borderline.values
o["gate_borderline"] = um.gate_borderline.values
o["robust_label"] = robust.values
o["robust_label_also_k"] = robust_k.values
for n, c in [("host", "r2_host"), ("state_20", "r2_state_20"), ("host_state_20", "r2_host_state_20"), ("locus_state20", "r2_locus_state20"),
             ("host_locus_state20", "r2_host_locus_state20")]:
    o[f"share_{n}_over_r2_best"] = o[c] / o.r2_best
bref = np.nanmax(np.vstack([MEAN[c] for c in refs0]), axis=0)
o["r2_best_ref"] = bref; o["share_best_ref_over_r2_best"] = bref / o.r2_best
o.reset_index().to_csv(RESULTS / "audit_per_mirna_r2.csv", index=False)

shares = {n: {"median": float(np.nanmedian(o.loc[ok, f"share_{n}_over_r2_best"])), "n_defined": int(o.loc[ok, f"share_{n}_over_r2_best"].notna().sum())}
          for n in ["host", "state_20", "host_state_20", "locus_state20", "host_locus_state20", "best_ref"]}
shares["r2_full"] = {"median": float(np.median(o.r2_full[ok] / o.r2_best[ok])), "n_defined": int(ok.sum())}
print(shares)

# ---- item 6: miRNAs that left 'poorly predicted' ----
moved = (v3.r2_full < 0.3) & ((fm.label != "poorly predicted") | (main.label != "poorly predicted"))
mv = pd.DataFrame({"r2_full": o.r2_full, "r2_full_fold_sd": np.nanstd(F["r2_full"], axis=0, ddof=1), "winning_model_by_mean": o.ceiling_model,
                   "r2_best": o.r2_best, "winning_model_fold_sd": o.r2_best_model_fold_sd,
                   "winning_model_ci_lo_NB": um.gate_lo + 0.3, "winning_model_ci_hi_NB": um.gate_hi + 0.3,
                   "label_main": B, "label_foldmax": fm.label, "r2_best_foldmax": o.r2_best_foldmax,
                   "not_poor_under_main": main.label != "poorly predicted", "not_poor_under_foldmax": fm.label != "poorly predicted"})[moved]
mv = mv.sort_values("r2_best", ascending=False)
mv.reset_index().to_csv(RESULTS / "audit_r2_best_moved.csv", index=False)
print(mv.round(3).to_string())

json.dump({"counts_main": counts(B), "counts_foldmax": counts(fm.label), "n_changed_vs_v3_main": int(changed_main.sum()),
           "n_changed_vs_v3_foldmax": int((fm.label != cur_split).sum()), "transitions_v3_to_main": pd.crosstab(cur_split, B).to_dict(orient="index"),
           "n_not_poor_main": int(ok.sum()), "n_not_poor_foldmax": int((fm.label != "poorly predicted").sum()),
           "uncertainty_main": blm.round(4).to_dict(orient="index"), "uncertainty_foldmax": blf.round(4).to_dict(orient="index"),
           "borderline_main": int(um.borderline.sum()), "borderline_foldmax": int(uf.borderline.sum()),
           "borderline_main_non_poor": int(um[ok].borderline.sum()), "borderline_foldmax_non_poor": int(uf[(fm.label != "poorly predicted").values].borderline.sum()),
           "robust": rob_summary, "settings_main": sens[sens.rule.str.startswith("MAIN")].set_index("setting")[CATS + ["n_changed_vs_own_baseline"]].to_dict(orient="index"),
           "median_shares_of_r2_best": shares,
           "ceiling_model_counts": o.ceiling_model.value_counts().to_dict(),
           "moved_out_of_poorly_predicted": mv.round(4).reset_index().to_dict(orient="records")},
          open(CACHE / "meta_r2b_labels.json", "w"), indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else (int(x) if isinstance(x, np.integer) else float(x)))
