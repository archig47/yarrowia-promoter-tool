"""D3: ISM for a pretrained language model, run on the GPU box.

The CNN takes one-hot arrays, so its ISM mutates arrays. A language model takes
text, so this mutates the sequence strings instead and re-tokenises. Same idea,
same output format, so both feed the identical TF-MoDISco step.

Uses the same test sequences as the CNN run (same rng seed and draw), so the two
motif sets are directly comparable rather than merely similar.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_lora import build, fit

BASES = "ACGT"
PAD_L = 95
TRIM = None


def load():
    d = ROOT / "data"
    pool = pq.read_table(d / "pool.parquet")
    test = pq.read_table(d / "test_primary.parquet")
    return (pool["seq"].to_pylist(), np.array(pool["label"].to_pylist(), np.float32),
            test["seq"].to_pylist(), np.array(test["label"].to_pylist(), np.float32))


def onehot(seqs, L=PAD_L):
    X = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:L]):
            k = BASES.find(ch)
            if k >= 0: X[i, j, k] = 1.0
    return X


def ism_strings(predict, seqs, batch_seqs=20000):
    """Mutate every position to every base, predict, mean-centre per position."""
    n = len(seqs)
    lens = [min(len(s), PAD_L) for s in seqs]
    hyp = np.zeros((n, PAD_L, 4), dtype=np.float32)

    variants, index = [], []
    def flush():
        if not variants: return
        preds = predict(variants)
        for (i, p, b), v in zip(index, preds):
            hyp[i, p, b] = v
        variants.clear(); index.clear()

    for i, s in enumerate(seqs):
        for p in range(lens[i]):
            for b, ch in enumerate(BASES):
                variants.append(s[:p] + ch + s[p + 1:])
                index.append((i, p, b))
        if len(variants) >= batch_seqs:
            flush()
            print(f"    ISM {i+1}/{n} sequences", flush=True)
    flush()

    for i in range(n):
        hyp[i, :lens[i]] -= hyp[i, :lens[i]].mean(axis=1, keepdims=True)
    oh = onehot(seqs)
    return oh, hyp, (hyp * oh).astype(np.float32)


def main(kind, n_train, seed, n_seqs):
    t0 = time.time()
    pool_seq, pool_y, ts, ty = load()
    subs = np.load(ROOT / "data" / "subsamples_cluster.npz")
    idx = subs[f"n{n_train}_seed{seed}"]

    rng = np.random.default_rng(0)                      # SAME draw as the CNN run
    pick = rng.choice(len(ts), min(n_seqs, len(ts)), replace=False)
    test_seqs = [ts[i] for i in pick]

    tok, _, _ = build(kind)
    print(f"training {kind} on n={n_train:,} ...", flush=True)
    predict, extra = fit(kind, tok, [pool_seq[i] for i in idx], pool_y[idx], seed)
    print(f"  trained {extra} ({time.time()-t0:.0f}s)", flush=True)

    oh, hyp, contrib = ism_strings(predict, test_seqs)
    out = ROOT / "data" / f"ism_{kind}.npz"
    np.savez_compressed(out, onehot=oh, hyp=hyp, contrib=contrib,
                        seqs=np.array(test_seqs, dtype=object), y=ty[pick])
    print(f"  wrote {out.name} onehot{oh.shape} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  contribution range: {contrib.min():.3f} .. {contrib.max():.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seqs", type=int, default=4000)
    a = ap.parse_args()
    main(a.model, a.n, a.seed, a.n_seqs)
