"""Check the GDC sample sheet and build the expression matrices from the raw downloads.

Primary-tumour samples with both an mRNA (STAR gene counts, TPM) and an miRNA (miRBase 21 quantification, RPM) file are
matched by Sample ID; the first file per Sample ID is used. Writes results/00_data_checks.csv (file counts, duplicated
Sample IDs, matrix shapes, row alignment).
With --rebuild, reads the matched files from data/raw and writes the sample x gene (TPM) and sample x miRNA (RPM)
matrices to cache/mrna_matrix.pkl and cache/mirna_matrix.pkl; the other scripts load them through
common.load_matrices(). Nothing in data/ is modified.

  python 02_build_matrices.py --rebuild"""
import sys

import pandas as pd

from common import CACHE, RESULTS, ROOT, load_matrices

sheet = pd.read_csv(ROOT / "gdc_sample_sheet.2026-08-17.tsv", sep="\t")
tp = sheet[(sheet["Tissue Type"] == "Tumor") & (sheet["Tumor Descriptor"] == "Primary")]
mrna_sheet = tp[tp["Data Type"] == "Gene Expression Quantification"]
mirna_sheet = tp[tp["Data Type"] == "miRNA Expression Quantification"]
matched = set(mrna_sheet["Sample ID"]) & set(mirna_sheet["Sample ID"])

rows = []
for name, s in [("mRNA", mrna_sheet), ("miRNA", mirna_sheet)]:
    sm = s[s["Sample ID"].isin(matched)]
    dup_ids = sm.loc[sm["Sample ID"].duplicated(keep=False), "Sample ID"].unique()
    rows.append({"data_type": name, "primary_files": len(s), "files_in_matched": len(sm),
                 "duplicated_sample_ids_in_matched": len(dup_ids),
                 "duplicated_sample_ids_all_primary": s["Sample ID"].duplicated(keep=False).pipe(
                     lambda m: s.loc[m, "Sample ID"].nunique()),
                 "after_drop_duplicates": len(sm.drop_duplicates("Sample ID")),
                 "dup_ids": ";".join(sorted(dup_ids))})
checks = pd.DataFrame(rows)
checks["matched_samples"] = len(matched)

if "--rebuild" in sys.argv:
    sys.path.insert(0, str(ROOT))
    from tqdm import tqdm

    def load_mrna_file(fid, fname, base=ROOT / "data/raw"):
        df = pd.read_csv(base / fid / fname, sep="\t", skiprows=1)
        return df[df["gene_id"].str.startswith("ENSG")].set_index("gene_id")["tpm_unstranded"]

    def load_mirna_file(fid, fname, base=ROOT / "data/raw"):
        return pd.read_csv(base / fid / fname, sep="\t").set_index("miRNA_ID")["reads_per_million_miRNA_mapped"]

    mm = mrna_sheet[mrna_sheet["Sample ID"].isin(matched)].drop_duplicates("Sample ID")
    mi = mirna_sheet[mirna_sheet["Sample ID"].isin(matched)].drop_duplicates("Sample ID")
    mrna = pd.DataFrame({r["Sample ID"]: load_mrna_file(r["File ID"], r["File Name"]) for _, r in tqdm(mm.iterrows(), total=len(mm))}).T.sort_index()
    mirna = pd.DataFrame({r["Sample ID"]: load_mirna_file(r["File ID"], r["File Name"]) for _, r in tqdm(mi.iterrows(), total=len(mi))}).T.sort_index()
    mrna.to_pickle(CACHE / "mrna_matrix.pkl")
    mirna.to_pickle(CACHE / "mirna_matrix.pkl")

mrna, mirna = load_matrices()
checks["mrna_matrix_shape"] = str(mrna.shape)
checks["mirna_matrix_shape"] = str(mirna.shape)
checks["rows_aligned"] = list(mrna.index) == list(mirna.index)
checks["unique_patients"] = mrna.index.str[:12].nunique()
checks.to_csv(RESULTS / "00_data_checks.csv", index=False)
print(checks.T.to_string())
