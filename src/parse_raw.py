#!/usr/bin/env python3
"""Parse de Boer GEO text files to Parquet, with QC.

Streams the gzipped source so peak memory stays flat regardless of file size.
Sequence convention (frozen 2026-09-04, see STATE.md decision log):
  17bp 5' scaffold TGCATTTTTTTCACATC + insert + 13bp 3' scaffold GGTTACGGCTGTT
Full sequence is stored as-is; `insert` is seq[17:-13]. No length filtering.
"""
import gzip, sys, hashlib, json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

P5, P3 = "TGCATTTTTTTCACATC", "GGTTACGGCTGTT"
L5, L3 = len(P5), len(P3)
CHUNK = 500_000

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT/"data"/"raw", ROOT/"data"/"processed"
PROC.mkdir(parents=True, exist_ok=True)

def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)

def parse(src, dst, tag):
    schema = pa.schema([("seq", pa.string()), ("label", pa.float32()),
                        ("insert_len", pa.int16()), ("flank_ok", pa.bool_())])
    w = pq.ParquetWriter(dst, schema, compression="zstd")
    n = 0; bad = 0; hashes = []; lens = []; labels = []
    seqs, labs, ilens, oks = [], [], [], []
    with gzip.open(src, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                bad += 1; continue
            s, v = parts
            try: v = float(v)
            except ValueError:
                bad += 1; continue
            ok = s.startswith(P5) and s.endswith(P3)
            il = len(s) - L5 - L3
            seqs.append(s); labs.append(v); ilens.append(il); oks.append(ok)
            hashes.append(h64(s)); lens.append(il); labels.append(v)
            n += 1
            if len(seqs) >= CHUNK:
                w.write_table(pa.table({"seq": seqs, "label": np.array(labs, "float32"),
                                        "insert_len": np.array(ilens, "int16"),
                                        "flank_ok": oks}, schema=schema))
                seqs, labs, ilens, oks = [], [], [], []
                print(f"  [{tag}] {n:,} rows", flush=True)
    if seqs:
        w.write_table(pa.table({"seq": seqs, "label": np.array(labs, "float32"),
                                "insert_len": np.array(ilens, "int16"),
                                "flank_ok": oks}, schema=schema))
    w.close()
    return dict(rows=n, malformed=bad,
                hashes=np.array(hashes, dtype=np.int64),
                lens=np.array(lens, dtype=np.int32),
                labels=np.array(labels, dtype=np.float64))

def summarise(tag, r):
    L, V, H = r["lens"], r["labels"], r["hashes"]
    uniq = np.unique(H)
    return {
        "rows": r["rows"], "malformed": r["malformed"],
        "unique_seqs": int(uniq.size), "duplicate_rows": int(r["rows"] - uniq.size),
        "insert_len": {"min": int(L.min()), "p1": int(np.percentile(L,1)),
                       "median": int(np.median(L)), "p99": int(np.percentile(L,99)),
                       "max": int(L.max()), "frac_80": float((L==80).mean()),
                       "frac_within_2": float((np.abs(L-80)<=2).mean())},
        "label": {"min": float(V.min()), "median": float(np.median(V)),
                  "mean": float(V.mean()), "max": float(V.max()),
                  "sd": float(V.std()), "n_negative": int((V<0).sum()),
                  "all_integer": bool(np.all(V == np.round(V)))},
    }

if __name__ == "__main__":
    train_src = RAW/"GSE104878_20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled.txt.gz"
    test_src  = RAW/"GSE104878_20160503_average_promoter_ELs_per_seq_atLeast100Counts.txt.gz"

    print("parsing TEST (high-quality pTpA/glucose)...", flush=True)
    te = parse(test_src, PROC/"test_highqual.parquet", "test")
    print("parsing TRAIN (pTpA_1E8_YPD)...", flush=True)
    tr = parse(train_src, PROC/"train_ptpa_glucose.parquet", "train")

    rep = {"test": summarise("test", te), "train": summarise("train", tr)}

    # leakage check: exact-sequence overlap between train and the primary test set
    inter = np.intersect1d(np.unique(tr["hashes"]), np.unique(te["hashes"]))
    rep["leakage"] = {"train_test_shared_seqs": int(inter.size),
                      "frac_of_test": float(inter.size / te["rows"])}

    (PROC/"parse_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    print("DONE", flush=True)
