"""Shared paths, matrix loading and preprocessing for the analysis scripts.

ROOT is the repository root (data/ and the GDC sample sheet are there); results, cache and figures are written to
analysis/results, analysis/cache and analysis/figures."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for d in (RESULTS, FIGURES, CACHE):
    d.mkdir(exist_ok=True)

RF_KW = dict(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)


def load_matrices():
    """Raw TPM / RPM matrices (samples x features), read from the pickles in cache/ written by
    02_build_matrices.py --rebuild. If they are missing, data/mrna_matrix.csv and data/mirna_matrix.csv (the same
    matrices as CSV) are read when present and cached as pickles."""
    mrna_pq, mirna_pq = CACHE / "mrna_matrix.pkl", CACHE / "mirna_matrix.pkl"
    if mrna_pq.exists() and mirna_pq.exists():
        return pd.read_pickle(mrna_pq), pd.read_pickle(mirna_pq)
    mrna_csv, mirna_csv = ROOT / "data/mrna_matrix.csv", ROOT / "data/mirna_matrix.csv"
    if not (mrna_csv.exists() and mirna_csv.exists()):
        raise FileNotFoundError("data/*_matrix.csv missing: run 02_build_matrices.py --rebuild")
    mrna = pd.read_csv(mrna_csv, index_col=0)
    mirna = pd.read_csv(mirna_csv, index_col=0)
    mrna.to_pickle(mrna_pq)
    mirna.to_pickle(mirna_pq)
    return mrna, mirna


def preprocess(mrna, mirna):
    """log2(x+1), top 100 miRNAs by variance, 80/20 train/test split (random_state=42)."""
    from sklearn.model_selection import train_test_split

    mrna_log = np.log2(mrna + 1)
    mirna_log = np.log2(mirna + 1)
    top_mirnas = mirna_log.var().sort_values(ascending=False).head(100).index.tolist()
    X = mrna_log.loc[mirna_log.index]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X, mirna_log, top_mirnas, X_train, X_test


def gene_lookup():
    """Ensembl ID -> gene_name from one raw STAR file."""
    path = next((ROOT / "data/raw").glob("*/*.rna_seq.augmented_star_gene_counts.tsv"))
    g = pd.read_csv(path, sep="\t", skiprows=1)
    g = g[g["gene_id"].str.startswith("ENSG")]
    return g.set_index("gene_id")["gene_name"]


def ids_for(lookup, name):
    return lookup[lookup == name].index.tolist()
