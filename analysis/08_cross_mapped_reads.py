"""Cross-mapped read flags in the raw miRNA quantification files.

Reads the miRNA file of every matched tumour (1,087) from data/raw. For six paralog groups (and mir-449a / mir-449b,
different miRNAs, as a control) writes how often the members have identical read counts and how often they are flagged
cross-mapped (results/10_e4_paralog_reads.csv). For every precursor writes the fraction of samples flagged cross-mapped
(results/10_e4_cross_mapped_all.csv; 10_state_models.py flags precursors above 50%). data/r2_results.csv
(03_holdout_rf_r2.py) is used only to mark the 100 most variable precursors."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import RESULTS, ROOT

GROUPS = [["hsa-mir-9-1", "hsa-mir-9-2", "hsa-mir-9-3"], ["hsa-mir-196a-1", "hsa-mir-196a-2"],
          ["hsa-mir-105-1", "hsa-mir-105-2"], ["hsa-mir-135a-1", "hsa-mir-135a-2"],
          ["hsa-mir-486-1", "hsa-mir-486-2"], ["hsa-mir-449a", "hsa-mir-449b"]]  # last: different miRNAs, control
ALL = [m for g in GROUPS for m in g]

sheet = pd.read_csv(ROOT / "gdc_sample_sheet.2026-08-17.tsv", sep="\t")
tp = sheet[(sheet["Tissue Type"] == "Tumor") & (sheet["Tumor Descriptor"] == "Primary")]
mrna_ids = set(tp.loc[tp["Data Type"] == "Gene Expression Quantification", "Sample ID"])
mi = tp[(tp["Data Type"] == "miRNA Expression Quantification") & tp["Sample ID"].isin(mrna_ids)].drop_duplicates("Sample ID")

counts, xmap, all_xmap = {}, {}, []
for _, r in tqdm(mi.iterrows(), total=len(mi)):
    df = pd.read_csv(ROOT / "data/raw" / r["File ID"] / r["File Name"], sep="\t").set_index("miRNA_ID")
    counts[r["Sample ID"]] = df.loc[ALL, "read_count"]
    xmap[r["Sample ID"]] = df.loc[ALL, "cross-mapped"].eq("Y")
    all_xmap.append(df["cross-mapped"].eq("Y"))
counts, xmap = pd.DataFrame(counts).T, pd.DataFrame(xmap).T
all_xmap = pd.concat(all_xmap, axis=1)

rows = []
for g in GROUPS:
    c = counts[g]
    expressed = (c > 0).any(axis=1)
    rows.append({"group": "/".join(m.replace("hsa-mir-", "") for m in g),
                 "samples": len(c), "samples_any_reads": int(expressed.sum()),
                 "frac_identical_counts_all": (c.nunique(axis=1) == 1).mean(),
                 "frac_identical_counts_expressed": (c[expressed].nunique(axis=1) == 1).mean(),
                 "median_read_count": ";".join(str(int(c[m].median())) for m in g),
                 "frac_cross_mapped_Y": ";".join(f"{xmap[m].mean():.3f}" for m in g)})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "10_e4_paralog_reads.csv", index=False)

# genome-wide: how often is any precursor flagged cross-mapped, and which of the 100 targets
top100 = pd.read_csv(ROOT / "data/r2_results.csv", index_col=0).index
frac = all_xmap.mean(axis=1)
pd.DataFrame({"frac_samples_cross_mapped": frac, "in_top100": frac.index.isin(top100)}).to_csv(
    RESULTS / "10_e4_cross_mapped_all.csv")
print(res.to_string())
print("top-100 targets with cross-mapped=Y in >50% of samples:",
      frac[frac.index.isin(top100) & (frac > 0.5)].round(3).to_dict())
