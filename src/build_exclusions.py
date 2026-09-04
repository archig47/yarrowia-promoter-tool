#!/usr/bin/env python3
"""Identify training rows whose sequence also appears in the primary test set.

The high-quality pTpA_3E5 library is a dilution of the same pool as pTpA_1E8
(GEO growth protocol), so the two overlap heavily. These rows must be excluded
from all training subsamples or every scaling curve is inflated.
Writes row indices into data/splits/ as a frozen artefact.
"""
import hashlib, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PROC, SPL = ROOT/"data"/"processed", ROOT/"data"/"splits"

def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)

test = pq.read_table(PROC/"test_highqual.parquet", columns=["seq"])["seq"].to_pylist()
tset = {h64(s) for s in test}
print(f"test sequences: {len(tset):,}", flush=True)

pf = pq.ParquetFile(PROC/"train_ptpa_glucose.parquet")
idx, off = [], 0
for b in pf.iter_batches(batch_size=500_000, columns=["seq"]):
    for i, s in enumerate(b["seq"].to_pylist()):
        if h64(s) in tset:
            idx.append(off + i)
    off += b.num_rows
    print(f"  scanned {off:,}  hits {len(idx):,}", flush=True)

idx = np.array(idx, dtype=np.int64)
np.save(SPL/"train_rows_excluded_test_overlap.npy", idx)
(SPL/"exclusions_report.json").write_text(json.dumps({
    "train_rows_total": off,
    "excluded_rows": int(idx.size),
    "excluded_frac": float(idx.size/off),
    "test_rows": len(test),
    "test_frac_found_in_train": float(idx.size/len(test)),
    "reason": "pTpA_3E5 high-quality library is a dilution of the same pool as pTpA_1E8 (GEO growth protocol); sequences overlap. Excluded from training to keep the primary test set genuinely held out.",
}, indent=2))
print("DONE", flush=True)
