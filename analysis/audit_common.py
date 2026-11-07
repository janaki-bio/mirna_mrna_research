"""Shared definitions for the analysis scripts: target and gene filters, ridge alpha grid, seeds, and loaders for the analysis set."""
import numpy as np
import pandas as pd

from common import RESULTS, load_matrices

TARGET_RULE = dict(rpm_gt=1.0, min_frac=0.5)   # RPM > 1 in >= 50% of analysis-set samples
GENE_RULE = dict(tpm_gt=0.5, min_frac=0.1)     # TPM > 0.5 in >= 10% of analysis-set samples
ALPHAS = np.logspace(-2, 8, 41)
SEEDS = [42, 43, 44]


def analysis_set():
    """Analysis-set samples (from 04_select_analysis_set.py) with PAM50, CPE, LUMP."""
    return pd.read_csv(RESULTS / "audit_samples.csv", index_col="sample_id")


def audit_data():
    """log2 matrices restricted to the analysis set, filtered targets and genes."""
    mrna, mirna = load_matrices()
    s = analysis_set()
    mrna, mirna = mrna.loc[s.index], mirna.loc[s.index]
    targets = mirna.columns[((mirna > TARGET_RULE["rpm_gt"]).mean() >= TARGET_RULE["min_frac"]).values]
    genes = mrna.columns[((mrna > GENE_RULE["tpm_gt"]).mean() >= GENE_RULE["min_frac"]).values]
    return s, np.log2(mrna + 1), np.log2(mirna + 1), mirna, list(targets), list(genes)
