"""Sequencing plate vs the top 10 principal components.

Plate = 5th field of the aliquot barcode (TCGA-XX-XXXX-01A-11R-A12B-07 -> A12B) of the mRNA (STAR) file and of the miRNA
file used for each analysis-set tumour (external/gdc/file_aliquots.csv, from 05_fetch_plate_barcodes.py; the barcode
must belong to the same patient and sample). For each of the 10 PCs of 15_technical_pcs.py: eta^2 and adjusted eta^2 for
mRNA plate and miRNA plate, plus plate counts and plate sizes.
Adds these columns to audit_technical_pcs.csv in place; the PCs and the other columns are unchanged (asserted)."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from audit_common import audit_data
from common import HERE, RESULTS

s, mrna_log, *_ , genes = audit_data()
al = pd.read_csv(HERE / "external/gdc/file_aliquots.csv").set_index("file_id")
assert (al.n_entities == 1).all()
al["plate"] = al.aliquot.str.split("-").str[5]
mp, mip = s.mrna_file_id.map(al.plate), s.mirna_file_id.map(al.plate)
for f, p in ((s.mrna_file_id, "mrna"), (s.mirna_file_id, "mirna")):
    a = f.map(al.aliquot)
    assert (a.str[:12].values == s.patient.values).all(), p        # barcode belongs to the same patient
    assert (a.str[:16].values == s.index.str[:16].values).all(), p # same sample (e.g. -01A)
print("plates: mRNA", mp.nunique(), "miRNA", mip.nunique(), "; tumours with same plate id in both assays:", int((mp == mip).sum()))

Xs = StandardScaler().fit_transform(mrna_log[genes].values)
ev, U = np.linalg.eigh(Xs @ Xs.T)
o = np.argsort(ev)[::-1][:10]
P = U[:, o] * np.sqrt(ev[o])


def eta2(x, g):
    g = pd.Series(np.asarray(g)); x = pd.Series(x)
    sst = ((x - x.mean()) ** 2).sum(); ssb = (g.map(x.groupby(g).mean()) - x.mean()).pow(2).sum()
    k, n = g.nunique(), len(x)
    return ssb / sst, (ssb - (k - 1) * (sst - ssb) / (n - k)) / sst


tab = pd.read_csv(RESULTS / "audit_technical_pcs.csv")
assert np.allclose(tab.eta2_pam50, [eta2(P[:, i], s.pam50)[0] for i in range(10)])   # same PCs as in 15_technical_pcs.py
for name, pl in (("mrna_plate", mp), ("mirna_plate", mip)):
    e = [eta2(P[:, i], pl) for i in range(10)]
    tab[f"eta2_{name}"], tab[f"eta2_{name}_adj"] = [a for a, _ in e], [b for _, b in e]
    tab[f"n_{name}s"] = pl.nunique()
    tab[f"median_tumours_per_{name}"] = pl.value_counts().median()
    tab[f"n_{name}s_with_1_tumour"] = int((pl.value_counts() == 1).sum())
    tab[f"max_tumours_per_{name}"] = pl.value_counts().max()
tab["plate"] = "aliquot barcode field 5 (GDC API metadata, 1 query)"
tab.to_csv(RESULTS / "audit_technical_pcs.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(tab[["pc", "var_explained", "eta2_tss", "eta2_tss_adj", "eta2_pam50", "eta2_mrna_plate", "eta2_mrna_plate_adj", "eta2_mirna_plate", "eta2_mirna_plate_adj"]].round(3).to_string(index=False))
print(tab.loc[0].to_string())
