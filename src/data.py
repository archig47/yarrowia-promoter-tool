#!/usr/bin/env python3
"""Loading and featurisation. Uses only the frozen split artefacts."""
from pathlib import Path
import json, gzip
import numpy as np, pyarrow.parquet as pq
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
PROC, SPL, RAW = ROOT/"data"/"processed", ROOT/"data"/"splits", ROOT/"data"/"raw"
P5, P3 = 17, 13
BASE = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate("ACGT"): BASE[ord(b)] = i

def _pool_cache():
    """Pool inserts + labels, cached. Row order matches pool_row_ids.npy."""
    f = PROC/"pool.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True); return list(d["seq"]), d["y"]
    ids = np.load(SPL/"pool_row_ids.npy"); want = set(ids.tolist())
    pf = pq.ParquetFile(PROC/"train_ptpa_glucose.parquet")
    got, off = {}, 0
    for b in pf.iter_batches(batch_size=500_000, columns=["seq", "label"]):
        s, l = b["seq"].to_pylist(), b["label"].to_pylist()
        for i in range(len(s)):
            g = off + i
            if g in want: got[g] = (s[i][P5:-P3], l[i])
        off += b.num_rows
    seq = [got[g][0] for g in ids]; y = np.array([got[g][1] for g in ids], dtype=np.float32)
    np.savez_compressed(f, seq=np.array(seq, dtype=object), y=y)
    return seq, y

def load_pool(): return _pool_cache()

def load_primary_test():
    t = pq.read_table(PROC/"test_highqual.parquet", columns=["seq", "label"])
    return [s[P5:-P3] for s in t["seq"].to_pylist()], np.array(t["label"].to_pylist(), np.float32)

def load_native80():
    rows = []
    with gzip.open(RAW/"GSE104878_20180808_processed_Native80_and_N80_spikein.txt.gz", "rt") as fh:
        fh.readline()
        for line in fh: rows.append(line.rstrip("\n").split("\t"))
    seq = [r[0][20:-20] for r in rows]
    nat = np.array([r[1] == "TRUE" for r in rows])
    y = np.array([float(r[5]) if r[5] not in ("", "NA") else np.nan for r in rows], np.float32)
    ok = np.isfinite(y)
    return ([s for s, o in zip(seq, ok) if o], y[ok], nat[ok])

def split(name):
    d = json.load(open(SPL/f"split_{name}.json"))
    return np.array(d["train"]), np.array(d["test"])

def subsamples(name): return np.load(SPL/f"subsamples_{name}.npz")

def kmer_counts(seqs, k=6):
    """Sparse k-mer count matrix, CSR (n, 4**k). ~74 nonzeros/row for 80bp, k=6,
    so sparse is ~50x smaller than dense and Ridge/LightGBM both accept it."""
    D = 4 ** k
    pw = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    indptr = [0]; idx = []; vals = []
    for s in seqs:
        a = BASE[np.frombuffer(s.encode(), dtype=np.uint8)].astype(np.int64)
        if a.size >= k:
            w = np.lib.stride_tricks.sliding_window_view(a, k)
            good = (w != 255).all(1)
            if good.any():
                u, c = np.unique((w[good] * pw).sum(1), return_counts=True)
                idx.append(u); vals.append(c)
        indptr.append(indptr[-1] + (len(idx[-1]) if idx else 0))
    if not idx: return sparse.csr_matrix((len(seqs), D), dtype=np.float32)
    return sparse.csr_matrix(
        (np.concatenate(vals).astype(np.float32), np.concatenate(idx), np.array(indptr)),
        shape=(len(seqs), D))

def onehot(seqs, L=95):
    X = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        a = BASE[np.frombuffer(s[:L].encode(), dtype=np.uint8)]
        m = a != 255
        X[i, a[m], np.arange(len(a))[m]] = 1.0
    return X
