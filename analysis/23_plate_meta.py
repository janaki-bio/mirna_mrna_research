"""Write results/audit_meta_plate.json from cache/meta_plate_labels.json (22_plate_labels.py) and a description of the design and rules."""
import json
from common import CACHE, RESULTS
m = json.load(open(CACHE / "meta_plate_labels.json"))
meta = {"design": "MAIN = plate-grouped CV (balanced mRNA-plate folds, 5 folds x 3 seeds; 211/211/211/210/210 tumours); patient-CV values are the *_patient columns / 'patient_cv' keys",
        "rules": "main rules (18_ceiling_model.py): ceiling model by mean R2, poorly predicted r2_best < 0.3, 0.8 x r2_best, residual gap >= 0.10, own/paralog locus split, host measurability, Nadeau-Bengio on the ceiling model's per-fold R2",
        "symmetric_plate_adjustment": "patient-grouped CV; inside each training fold per-plate means (minus the training grand mean) removed from the mRNA matrix (27,671 filtered genes + host genes) and from the 337 miRNA targets; training-fold shifts applied to test tumours (unseen plate: no shift); all models incl. host and locus use adjusted mRNA and are scored on adjusted test targets (R2 on the adjusted scale); clinical models not refitted",
        "scripts": ["21_plate_cv_fits.py", "22_plate_labels.py", "23_plate_meta.py"],
        "network": "none", "environment": "conda env omics; no installs",
        "files": ["audit_per_mirna_plate.csv", "audit_robustness_plate.csv", "audit_uncertainty_plate.csv", "audit_clinical_plate.csv", "audit_plate_adjusted_sym.csv"]} | m
json.dump(meta, open(RESULTS / "audit_meta_plate.json", "w"), indent=2)
