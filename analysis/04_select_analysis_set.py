"""Select the analysis set: one tumour per patient with PAM50 and purity (results/audit_samples.csv).

Starts from the 1,087 matched primary tumours, drops FFPE samples, keeps one sample per patient (preferring -01A), keeps
patients with a PAM50 call (matched by patient barcode) and samples with a CPE purity estimate (matched by sample
barcode); LUMP purity is recorded and may be missing.
Reads gdc_sample_sheet.2026-08-17.tsv, external/tcgabiolinks/*.csv (01_export_pam50_purity.R) and the matrices from
02_build_matrices.py.
Writes audit_samples.csv (sample_id, patient, mRNA and miRNA file ids, pam50, cpe, lump), audit_sample_steps.json (sample
counts after each step) and audit_duplicate_files.csv (Sample IDs with more than one file, and which file was used)."""
import json

import pandas as pd

from common import HERE, RESULTS, ROOT, load_matrices

sheet = pd.read_csv(ROOT / "gdc_sample_sheet.2026-08-17.tsv", sep="\t")
tp = sheet[(sheet["Tissue Type"] == "Tumor") & (sheet["Tumor Descriptor"] == "Primary")]
by_type = {"mrna": tp[tp["Data Type"] == "Gene Expression Quantification"],
           "mirna": tp[tp["Data Type"] == "miRNA Expression Quantification"]}
matched = set(by_type["mrna"]["Sample ID"]) & set(by_type["mirna"]["Sample ID"])
# files used = first file per Sample ID in sheet order in sample-sheet order
used = {k: s[s["Sample ID"].isin(matched)].drop_duplicates("Sample ID").set_index("Sample ID") for k, s in by_type.items()}

# record the duplicated Sample IDs and which file was used
dup_rows = []
for k, s in by_type.items():
    sm = s[s["Sample ID"].isin(matched)]
    for sid, g in sm[sm["Sample ID"].duplicated(keep=False)].groupby("Sample ID"):
        for i, (_, r) in enumerate(g.iterrows()):
            dup_rows.append({"sample_id": sid, "assay": k, "file_id": r["File ID"], "file_name": r["File Name"],
                             "used": i == 0})
pd.DataFrame(dup_rows).to_csv(RESULTS / "audit_duplicate_files.csv", index=False)

mrna, _ = load_matrices()
s = pd.DataFrame(index=mrna.index.rename("sample_id"))
s["patient"] = s.index.str[:12]
s["mrna_file_id"] = used["mrna"].loc[s.index, "File ID"]
s["mirna_file_id"] = used["mirna"].loc[s.index, "File ID"]
s["ffpe"] = (used["mrna"].loc[s.index, "Preservation Method"].eq("FFPE").values
             | used["mirna"].loc[s.index, "Preservation Method"].eq("FFPE").values)
steps = {"matched_primary_tumours": len(s)}

s = s[~s.ffpe]
steps["after_ffpe_removed"] = len(s)

s["pref"] = (s.index.str[13:16] != "01A").astype(int)
s = s.sort_values(["patient", "pref"]).groupby("patient").head(1).sort_index()
steps["after_one_per_patient"] = len(s)

sub = pd.read_csv(HERE / "external/tcgabiolinks/brca_subtype.csv").set_index("patient")["BRCA_Subtype_PAM50"]
s["pam50"] = s.patient.map(sub)
s = s[s.pam50.notna() & (s.pam50 != "NA")]
steps["after_pam50_available"] = len(s)

pur = pd.read_csv(HERE / "external/tcgabiolinks/tumor_purity_brca.csv", dtype=str).set_index("Sample.ID")
num = lambda c: pd.to_numeric(pur[c].str.replace(",", ".", regex=False), errors="coerce")  # comma decimals
s["cpe"] = s.index.map(num("CPE"))
s["lump"] = s.index.map(num("LUMP"))
s = s[s.cpe.notna()]
steps["analysis_set_after_cpe_available"] = len(s)
steps["analysis_set_lump_missing"] = int(s.lump.isna().sum())
steps["analysis_set_unique_patients"] = int(s.patient.nunique())
steps["analysis_set_used_duplicate_sample_ids"] = sorted(set(pd.DataFrame(dup_rows).sample_id) & set(s.index))

s.drop(columns=["pref", "ffpe"]).to_csv(RESULTS / "audit_samples.csv")
json.dump(steps, open(RESULTS / "audit_sample_steps.json", "w"), indent=2)
print(json.dumps(steps, indent=2))
print(s.pam50.value_counts().to_string())
print(s[["cpe", "lump"]].describe().round(3).to_string())
