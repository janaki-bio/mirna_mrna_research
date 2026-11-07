"""Technical variables vs the top 10 principal components of the full analysis set.

PCA of the standardised 27,671 filtered genes over the 1,053 analysis-set tumours (the PCs of 10_state_models.py; PC sign
arbitrary). For each PC: eta^2 for tissue source site (TSS, sample barcode characters 6-7) and PAM50, with adjusted
eta^2 (epsilon^2 = (SS_b - (k-1) MS_w) / SS_t) because TSS has many small levels; Spearman rho with CPE, RNA-seq library
size (sum of the 'unstranded' counts over the ENSG gene rows of the STAR file; the N_* summary rows are excluded) and
total miRNA-mapped reads (sum of read_count over all rows of the miRNA quantification file).
Reads local files only (data/raw/<file_id>/...). Writes audit_technical_pcs.csv and cache/audit_r2_library_sizes.csv
(library-size cache); 17_plate_vs_components.py adds the plate columns."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from audit_common import audit_data
from common import CACHE, RESULTS, ROOT

s, mrna_log, mirna_log, mirna_rpm, targets, genes = audit_data()
RAW = ROOT / "data/raw"

lib_cache = CACHE / "audit_r2_library_sizes.csv"
if lib_cache.exists():
    lib = pd.read_csv(lib_cache, index_col=0)
else:
    rows = {}
    for sid, r in s.iterrows():
        f = next((RAW / r.mrna_file_id).glob("*.rna_seq.augmented_star_gene_counts.tsv"))
        g = pd.read_csv(f, sep="\t", skiprows=1, usecols=["gene_id", "unstranded"])
        g = g[g.gene_id.str.startswith("ENSG")]
        m = next((RAW / r.mirna_file_id).glob("*.mirbase21.mirnas.quantification.txt"))
        mi = pd.read_csv(m, sep="\t", usecols=["miRNA_ID", "read_count"])
        rows[sid] = {"rnaseq_library_size": int(g.unstranded.sum()), "mirna_mapped_reads": int(mi.read_count.sum()),
                     "n_gene_rows": len(g), "n_mirna_rows": len(mi)}
    lib = pd.DataFrame.from_dict(rows, orient="index")
    lib.to_csv(lib_cache)
lib = lib.loc[s.index]
print(lib.describe().round(0).to_string())

# top 10 PCs of the full analysis set, as in 10_state_models.py
Xs = StandardScaler().fit_transform(mrna_log[genes].values)
ev, U = np.linalg.eigh(Xs @ Xs.T)
o = np.argsort(ev)[::-1][:10]
P = U[:, o] * np.sqrt(ev[o])
var = ev[o] / (Xs ** 2).sum()

tss = pd.Series(s.index.str[5:7], index=s.index)
print("tissue source sites:", tss.nunique(), "smallest/largest site:", tss.value_counts().min(), tss.value_counts().max())


def eta2(x, g):
    g = pd.Series(g.values if hasattr(g, "values") else g)
    x = pd.Series(x)
    sst = ((x - x.mean()) ** 2).sum()
    ssb = (g.map(x.groupby(g).mean()) - x.mean()).pow(2).sum()
    k, n = g.nunique(), len(x)
    return ssb / sst, (ssb - (k - 1) * (sst - ssb) / (n - k)) / sst


rows = []
for i in range(10):
    x = P[:, i]
    e_t, a_t = eta2(x, tss)
    e_p, a_p = eta2(x, s.pam50)
    r = {"pc": f"PC{i + 1}", "var_explained": var[i], "eta2_tss": e_t, "eta2_tss_adj": a_t, "eta2_pam50": e_p,
         "eta2_pam50_adj": a_p}
    for name, v in (("cpe", s.cpe), ("rnaseq_library_size", lib.rnaseq_library_size),
                    ("mirna_mapped_reads", lib.mirna_mapped_reads)):
        rho, p = spearmanr(x, v)
        r[f"rho_{name}"], r[f"p_{name}"] = rho, p
    rows.append(r)
tab = pd.DataFrame(rows)
tab["n_tumours"] = len(s)
tab["n_tss_levels"] = tss.nunique()
tab["plate"] = "added by 17_plate_vs_components.py"
tab.to_csv(RESULTS / "audit_technical_pcs.csv", index=False)
pd.set_option("display.width", 250)
print(tab.drop(columns=["plate", "n_tumours", "n_tss_levels"]).round(3).to_string(index=False))
# check against the PAM50 eta2 / CPE rho in audit_pc_programmes.csv (10_state_models.py)
old = pd.read_csv(RESULTS / "audit_pc_programmes.csv")
print("max |eta2_pam50 - old|:", np.abs(old.pam50_eta2.values - tab.eta2_pam50.values).max(),
      " max ||rho_cpe| - |old||:", np.abs(np.abs(old.rho_cpe.values) - np.abs(tab.rho_cpe.values)).max())
