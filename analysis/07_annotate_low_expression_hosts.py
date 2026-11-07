"""Host annotation for the four miRNAs below the expression filter (mir-934, mir-577, mir-449a, mir-449b).

Same rules as 06_annotate_hosts.py (uses its GENCODE cache and audit_mir_mapping.csv). Writes
results/audit_hosts_low_expression.csv with the same columns as audit_hosts.csv; the cluster_members column lists the
precursors within 10 kb on the same strand (direct neighbours, no chaining)."""
import re

import pandas as pd

from common import CACHE, RESULTS

LOW = ["hsa-mir-934", "hsa-mir-577", "hsa-mir-449a", "hsa-mir-449b"]
FLANK = 50_000

genes, exons = pd.read_pickle(CACHE / "gencode_v36_genes_exons.pkl")
mir_genes = genes[genes.gene_type == "miRNA"].drop_duplicates("gene_name").set_index("gene_name")
mapping = pd.read_csv(RESULTS / "audit_mir_mapping.csv").set_index("mirna")["mir_gene"]


def paralog_base(name):
    m = re.match(r"^(hsa-(?:mir|let)-[0-9a-z]+)-(\d+)$", name)
    return m.group(1) if m else None


pb = pd.Series({m: paralog_base(m) for m in mapping.index}).dropna()
para = pb[pb.duplicated(keep=False)]

rows = []
for t in LOW:
    m = mir_genes.loc[mapping[t]]
    r = {"mirna": t, "mir_gene": mapping[t], "paralog_group": para.get(t), "cluster_id": None, "cluster_members": None,
         "host_genes": "", "host_gene_ids": "", "host_type": "none", "host_gene_types": "",
         "neighbours_50kb": "", "neighbour_ids_50kb": "", "mir_gene_id": m.gene_id,
         "chrom": m.chrom, "start": m.start, "end": m.end, "strand": m.strand}
    same = genes[(genes.chrom == m.chrom) & (genes.gene_id != m.gene_id)]
    hosts = same[(same.start <= m.end) & (same.end >= m.start) & (same.strand == m.strand) & (same.gene_type != "miRNA")]
    if len(hosts):
        r["host_genes"] = ";".join(hosts.gene_name)
        r["host_gene_ids"] = ";".join(hosts.gene_id)
        r["host_gene_types"] = ";".join(hosts.gene_type)
        pc = hosts[hosts.gene_type == "protein_coding"]
        if len(pc):
            ex = exons[exons.gene_id.isin(pc.gene_id) & (exons.transcript_type == "protein_coding")
                       & (exons.start <= m.end) & (exons.end >= m.start)]
            r["host_type"] = "protein-coding exon" if len(ex) else "protein-coding intron"
        elif (hosts.gene_type == "lncRNA").any():
            r["host_type"] = "lncRNA host"
        else:
            r["host_type"] = "other host (" + ",".join(sorted(set(hosts.gene_type))) + ")"
    win = same[(same.start <= m.end + FLANK) & (same.end >= m.start - FLANK)]
    r["neighbours_50kb"] = ";".join(win.gene_name)
    r["neighbour_ids_50kb"] = ";".join(win.gene_id)
    rows.append(r)

out = pd.DataFrame(rows)
# cluster members: precursors whose MIR gene lies within 10 kb on the same strand (direct neighbours, no chaining)
mapped = mapping.dropna().to_frame().join(mir_genes[["chrom", "start", "end", "strand"]], on="mir_gene")
for i, r in out.iterrows():
    near = mapped[(mapped.chrom == r.chrom) & (mapped.strand == r.strand)
                  & (mapped.start <= r.end + 10_000) & (mapped.end >= r.start - 10_000)]
    if len(near) > 1:
        out.loc[i, "cluster_members"] = ";".join(near.index)
out.to_csv(RESULTS / "audit_hosts_low_expression.csv", index=False)
print(out[["mirna", "mir_gene", "host_genes", "host_type", "paralog_group", "neighbours_50kb"]].to_string())
