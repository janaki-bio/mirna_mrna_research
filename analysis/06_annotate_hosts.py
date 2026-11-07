"""Host-gene, locus and paralog annotation of every target miRNA from GENCODE v36 (the annotation of the GDC STAR pipeline).

Per precursor: MIR gene; host genes (non-miRNA genes overlapping the precursor on the same strand) and host type
(protein-coding exon / intron, lncRNA, other, none); genes within +-50 kb; paralog group (names that differ only in the
numeric suffix); cluster (single linkage, precursors within 10 kb on the same strand, over all mapped precursors).
Reads external/gencode_v36/gencode.v36.annotation.gtf.gz (download from gencodegenes.org) and caches the gene and exon
tables in cache/gencode_v36_genes_exons.pkl.
Writes audit_hosts.csv (one row per target) and audit_mir_mapping.csv (every precursor in the matrix)."""
import gzip
import re

import pandas as pd

from audit_common import audit_data
from common import CACHE, HERE, RESULTS

GTF = HERE / "external/gencode_v36/gencode.v36.annotation.gtf.gz"
FLANK, CLUSTER_GAP = 50_000, 10_000

cache = CACHE / "gencode_v36_genes_exons.pkl"
if cache.exists():
    genes, exons = pd.read_pickle(cache)
else:
    attr = re.compile(r'(gene_id|gene_name|gene_type|transcript_type) "([^"]+)"')
    g_rows, e_rows = [], []
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] not in ("gene", "exon"):
                continue
            a = dict(attr.findall(f[8]))
            rec = [f[0], int(f[3]), int(f[4]), f[6], a["gene_id"], a["gene_name"], a["gene_type"]]
            if f[2] == "gene":
                g_rows.append(rec)
            else:
                e_rows.append(rec + [a.get("transcript_type")])
    cols = ["chrom", "start", "end", "strand", "gene_id", "gene_name", "gene_type"]
    genes = pd.DataFrame(g_rows, columns=cols)
    exons = pd.DataFrame(e_rows, columns=cols + ["transcript_type"]).drop_duplicates(
        ["chrom", "start", "end", "gene_id", "transcript_type"])
    pd.to_pickle((genes, exons), cache)

s, mrna_log, mirna_log, mirna_rpm, targets, gene_cols = audit_data()
mir_genes = genes[genes.gene_type == "miRNA"].drop_duplicates("gene_name").set_index("gene_name")


def mir_symbol(name):
    m = re.match(r"hsa-(mir|let)-(.+)$", name)
    if not m:
        return None
    base = m.group(2).upper()
    prefix = "MIR" if m.group(1) == "mir" else "MIRLET"
    cands = [prefix + base, prefix + re.sub(r"([A-Z])-(\d+)$", r"\1\2", base)]
    return next((c for c in cands if c in mir_genes.index), None)


def paralog_base(name):
    m = re.match(r"^(hsa-(?:mir|let)-[0-9a-z]+)-(\d+)$", name)
    return m.group(1) if m else None


# map all precursors in the matrix, for clusters and the unmatched list
allmap = pd.DataFrame({"mirna": mirna_rpm.columns})
allmap["mir_gene"] = allmap.mirna.map(mir_symbol)
allmap.to_csv(RESULTS / "audit_mir_mapping.csv", index=False)
mapped = allmap.dropna(subset=["mir_gene"]).copy()
mapped = mapped.join(mir_genes[["chrom", "start", "end", "strand", "gene_id"]], on="mir_gene")

# clusters: single linkage within 10 kb, same chrom and strand
mapped = mapped.sort_values(["chrom", "strand", "start"])
cid, prev, cluster = 0, None, []
for _, r in mapped.iterrows():
    if prev is None or r.chrom != prev.chrom or r.strand != prev.strand or r.start - prev_end > CLUSTER_GAP:
        cid += 1
        prev_end = r.end
    prev_end = max(prev_end, r.end)
    prev = r
    cluster.append(cid)
mapped["cluster_raw"] = cluster
sizes = mapped.cluster_raw.value_counts()
multi = sizes[sizes > 1].index
names = {c: f"C{i + 1:03d}" for i, c in enumerate(sorted(multi))}
mapped["cluster_id"] = mapped.cluster_raw.map(names)
members = mapped.dropna(subset=["cluster_id"]).groupby("cluster_id")["mirna"].apply(lambda x: ";".join(x))
mapped = mapped.set_index("mirna")

# paralog groups (over all precursors in the matrix)
pb = pd.Series({m: paralog_base(m) for m in mirna_rpm.columns}).dropna()
groups = pb[pb.duplicated(keep=False)]

rows = []
for t in targets:
    r = {"mirna": t, "mir_gene": mir_symbol(t), "paralog_group": groups.get(t), "cluster_id": None,
         "cluster_members": None, "host_genes": "", "host_gene_ids": "", "host_type": "none",
         "host_gene_types": "", "neighbours_50kb": "", "neighbour_ids_50kb": ""}
    if r["mir_gene"] is None:
        r["host_type"] = "unmapped"
        rows.append(r)
        continue
    m = mapped.loc[t]
    r.update({"mir_gene_id": m.gene_id, "chrom": m.chrom, "start": m.start, "end": m.end, "strand": m.strand,
              "cluster_id": m.cluster_id if isinstance(m.cluster_id, str) else None})
    if r["cluster_id"]:
        r["cluster_members"] = members[r["cluster_id"]]
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

hosts = pd.DataFrame(rows)
hosts.to_csv(RESULTS / "audit_hosts.csv", index=False)
print("targets:", len(targets), " genes:", len(gene_cols))
print("precursors in matrix:", len(allmap), " unmapped:", allmap.mir_gene.isna().sum(), allmap.mirna[allmap.mir_gene.isna()].tolist())
print("unmapped targets:", hosts.mirna[hosts.mir_gene.isna()].tolist())
print(hosts.host_type.value_counts().to_string())
print("targets in paralog groups:", hosts.paralog_group.notna().sum(), " in clusters:", hosts.cluster_id.notna().sum())
print(hosts.loc[hosts.mirna.isin(["hsa-mir-934", "hsa-mir-577", "hsa-mir-449a", "hsa-mir-18a", "hsa-let-7a-1", "hsa-mir-196a-1", "hsa-mir-150"]),
                ["mirna", "mir_gene", "host_genes", "host_type", "paralog_group", "cluster_id"]].to_string())
