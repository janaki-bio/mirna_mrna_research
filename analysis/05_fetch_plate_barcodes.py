"""Fetch the aliquot barcode of every mRNA and miRNA file of the analysis set from the GDC API.

One POST request to https://api.gdc.cancer.gov/files with the 2,106 file ids (the mRNA and the miRNA file of each of the
1,053 tumours in results/audit_samples.csv). Only metadata is requested; no data files are downloaded. Needs network access.
Writes
  external/gdc/aliquots_response.json   the API response
  external/gdc/file_aliquots.csv        file_id, data_type, n_entities, aliquot, types (one row per file)
The plate of a file is the 5th field of its aliquot barcode (TCGA-XX-XXXX-01A-11R-A12B-07 -> A12B). Steps 17, 20 and 21 read
file_aliquots.csv. The repository includes a copy of file_aliquots.csv, so this step is only needed to regenerate it.

  python 05_fetch_plate_barcodes.py                  # query the API
  python 05_fetch_plate_barcodes.py --from-response  # rebuild the CSV from a saved aliquots_response.json (no network)
"""
import json
import sys
import urllib.request

import pandas as pd

from common import HERE, RESULTS

URL = "https://api.gdc.cancer.gov/files"
FIELDS = ",".join(["file_id", "data_type", "cases.submitter_id", "associated_entities.entity_submitter_id",
                   "associated_entities.entity_type"])
OUT = HERE / "external/gdc"
OUT.mkdir(parents=True, exist_ok=True)
RESPONSE = OUT / "aliquots_response.json"

if "--from-response" in sys.argv:
    response = json.loads(RESPONSE.read_text())
else:
    samples = pd.read_csv(RESULTS / "audit_samples.csv")
    ids = sorted(set(samples.mrna_file_id) | set(samples.mirna_file_id))
    body = {"filters": {"op": "in", "content": {"field": "file_id", "value": ids}}, "fields": FIELDS,
            "format": "JSON", "size": len(ids) + 10}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "mirna-attribution-analysis"})
    with urllib.request.urlopen(request, timeout=300) as r:
        response = json.load(r)
    RESPONSE.write_text(json.dumps(response))

hits = response["data"]["hits"]
assert len(hits) == response["data"]["pagination"]["total"], "response incomplete"
rows = [{"file_id": h["file_id"], "data_type": h["data_type"], "n_entities": len(h["associated_entities"]),
         "aliquot": ";".join(e["entity_submitter_id"] for e in h["associated_entities"]),
         "types": ";".join(e["entity_type"] for e in h["associated_entities"])} for h in hits]
pd.DataFrame(rows).to_csv(OUT / "file_aliquots.csv", index=False)
print(len(rows), "files;", sum(r["n_entities"] != 1 for r in rows), "with more than one associated entity")
