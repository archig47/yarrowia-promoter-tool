#!/usr/bin/env python3
"""Build and FREEZE the working pool, splits, and subsample indices.

Frozen artefacts (never regenerate - see CLAUDE.md invariants):
  data/splits/pool_row_ids.npy          row ids into train parquet (the working pool)
  data/splits/split_cluster.json        cluster-based held-out split
  data/splits/split_random.json         random held-out split
  data/splits/subsamples_{split}.npz    nested training subsamples, n x 3 seeds
  data/splits/cluster_report.json       cluster size distribution (evidence)

Design:
  - pool drawn from training rows EXCLUDING those overlapping the primary test set
  - 200k pool: 10k held out as the secondary test, 190k available for training
  - subsamples are NESTED within a seed, so scaling curves add data rather than
    resampling it
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PROC, SPL = ROOT/"data"/"processed", ROOT/"data"/"splits"
POOL_N, TEST_N = 200_000, 10_000
NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = [0, 1, 2]
POOL_SEED = 20260904

def main():
    excl = set(np.load(SPL/"train_rows_excluded_test_overlap.npy").tolist())
    pf = pq.ParquetFile(PROC/"train_ptpa_glucose.parquet")
    total = pf.metadata.num_rows
    print(f"train rows {total:,}, excluded {len(excl):,}", flush=True)

    rng = np.random.default_rng(POOL_SEED)
    cand = rng.choice(total, size=POOL_N + 2*len(excl) + 1000, replace=False)
    cand = np.array([i for i in cand if i not in excl][:POOL_N])
    cand.sort()
    print(f"pool rows selected: {cand.size:,}", flush=True)

    cache = PROC/"pool_inserts.fasta"
    if cache.exists():
        seqs, cur = {}, None
        for line in open(cache):
            if line.startswith(">"): cur = int(line[1:])
            else: seqs[cur] = line.strip()
        print(f"pool sequences from cache: {len(seqs):,}", flush=True)
    else:
        want, seqs, off = set(cand.tolist()), {}, 0
        for b in pf.iter_batches(batch_size=500_000, columns=["seq"]):
            col = b["seq"].to_pylist()
            for i, s in enumerate(col):
                g = off + i
                if g in want: seqs[g] = s[17:-13]
            off += b.num_rows
        with open(cache, "w") as fh:
            for g in cand: fh.write(f">{g}\n{seqs[g]}\n")
        print(f"pool sequences read: {len(seqs):,}", flush=True)

    ids = cand
    np.save(SPL/"pool_row_ids.npy", ids)

    # ---- cluster with mmseqs2 (nucleotide mode)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td/"pool.fasta"
        with open(fa, "w") as fh:
            for j, g in enumerate(ids):
                fh.write(f">{j}\n{seqs[g]}\n")
        out = td/"clu"
        cmd = ["mmseqs", "easy-cluster", str(fa), str(out), str(td/"tmp"),
               "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "1",
               "--dbtype", "2", "-v", "1"]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        members = {}
        with open(str(out)+"_cluster.tsv") as fh:
            for line in fh:
                rep, mem = line.split()
                members.setdefault(rep, []).append(int(mem))

    sizes = np.array([len(v) for v in members.values()])
    creport = {"n_sequences": int(ids.size), "n_clusters": int(sizes.size),
               "singletons": int((sizes == 1).sum()),
               "frac_singleton": float((sizes == 1).mean()),
               "largest_cluster": int(sizes.max()),
               "size_distribution": {str(k): int(v) for k, v in
                                     zip(*np.unique(sizes, return_counts=True))}}
    (SPL/"cluster_report.json").write_text(json.dumps(creport, indent=2))
    print(json.dumps(creport, indent=2), flush=True)

    # ---- cluster split: whole clusters to test until TEST_N reached
    r = np.random.default_rng(POOL_SEED)
    cl = list(members.values()); r.shuffle(cl)
    test_idx, n = [], 0
    for c in cl:
        if n >= TEST_N: break
        test_idx.extend(c); n += len(c)
    test_idx = np.array(sorted(test_idx))
    train_idx = np.setdiff1d(np.arange(ids.size), test_idx)
    json.dump({"test": test_idx.tolist(), "train": train_idx.tolist()},
              open(SPL/"split_cluster.json", "w"))
    print(f"cluster split: train {train_idx.size:,}  test {test_idx.size:,}", flush=True)

    # ---- random split
    r2 = np.random.default_rng(POOL_SEED + 1)
    perm = r2.permutation(ids.size)
    rtest, rtrain = np.sort(perm[:TEST_N]), np.sort(perm[TEST_N:])
    json.dump({"test": rtest.tolist(), "train": rtrain.tolist()},
              open(SPL/"split_random.json", "w"))

    # ---- nested subsamples per split
    for name, tr in (("cluster", train_idx), ("random", rtrain)):
        d = {}
        for s in SEEDS:
            rr = np.random.default_rng(1000 + s)
            order = rr.permutation(tr)
            for n_ in NS:
                if n_ <= order.size:
                    d[f"n{n_}_seed{s}"] = np.sort(order[:n_])
        np.savez_compressed(SPL/f"subsamples_{name}.npz", **d)
        print(f"subsamples_{name}.npz: {len(d)} index sets", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
