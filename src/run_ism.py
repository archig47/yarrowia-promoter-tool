"""D3 step 1: in-silico mutagenesis (ISM).

For each test sequence, mutate every position to every other base and record how
the prediction changes. That gives a per-position, per-base importance map, which
TF-MoDISco then clusters into recurring patterns (candidate motifs).

PLAN.md specifies ISM rather than attention maps, because DNABERT-2's byte-pair
tokens are variable-length and do not align to single bases.

Output: data/processed/ism_<model>.npz with
  onehot  (N, L, 4)  the sequences, in TF-MoDISco's axis order
  hyp     (N, L, 4)  hypothetical contribution of each base at each position
  contrib (N, L, 4)  hyp masked to the base actually present
"""
import sys, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D

N_SEQS = 500          # overridable: PLAN.md says ~500, but TF-MoDISco
                      # needs far more seqlets to resolve weak motifs
PAD_L = 95
CHUNK = 4096


def ism(predict, seqs, pad_l=PAD_L):
    """Mean-centred ISM contributions. Returns (onehot, hyp, contrib), all (N, L, 4)."""
    X = D.onehot(seqs, L=pad_l)                    # (N, 4, L)
    n, _, L = X.shape
    lens = np.array([min(len(s), pad_l) for s in seqs])
    hyp = np.zeros((n, L, 4), dtype=np.float32)

    # build every single-base variant, predict in chunks
    variants, index = [], []
    for i in range(n):
        for pos in range(lens[i]):
            for b in range(4):
                v = X[i].copy()
                v[:, pos] = 0.0
                v[b, pos] = 1.0
                variants.append(v)
                index.append((i, pos, b))
            if len(variants) >= CHUNK:
                preds = predict(np.stack(variants))
                for (ii, pp, bb), pr in zip(index, preds):
                    hyp[ii, pp, bb] = pr
                variants, index = [], []
    if variants:
        preds = predict(np.stack(variants))
        for (ii, pp, bb), pr in zip(index, preds):
            hyp[ii, pp, bb] = pr

    # centre each position: contribution is relative to the average base there
    for i in range(n):
        hyp[i, :lens[i]] -= hyp[i, :lens[i]].mean(axis=1, keepdims=True)
    oh = X.transpose(0, 2, 1).copy()                # (N, L, 4)
    return oh, hyp, (hyp * oh).astype(np.float32)


def main(model_name, n_train=100_000, seed=0, n_seqs=N_SEQS):
    t0 = time.time()
    pool_seq, pool_y = D.load_pool()
    subs = D.subsamples("cluster")
    idx = subs[f"n{n_train}_seed{seed}"]
    ts, ty = D.load_primary_test()

    rng = np.random.default_rng(0)
    pick = rng.choice(len(ts), min(n_seqs, len(ts)), replace=False)
    test_seqs = [ts[i] for i in pick]
    print(f"training {model_name} on n={n_train:,}; ISM over {len(test_seqs)} test sequences", flush=True)

    if model_name == "cnn":
        from run_baselines import fit_cnn
        Xtr = D.onehot([pool_seq[i] for i in idx], L=PAD_L)
        predict, extra = fit_cnn(Xtr, pool_y[idx], seed)
        print(f"  trained: {extra} ({time.time()-t0:.0f}s)", flush=True)
    else:
        raise SystemExit(f"model '{model_name}' not supported locally - run on the GPU box")

    oh, hyp, contrib = ism(predict, test_seqs)
    out = ROOT / "data" / "processed" / f"ism_{model_name}.npz"
    np.savez_compressed(out, onehot=oh, hyp=hyp, contrib=contrib,
                        seqs=np.array(test_seqs, dtype=object), y=ty[pick])
    print(f"  wrote {out.name}  onehot{oh.shape} hyp{hyp.shape}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  contribution range: {contrib.min():.3f} .. {contrib.max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n-seqs", type=int, default=N_SEQS)
    a = ap.parse_args()
    main(a.model, a.n, n_seqs=a.n_seqs)
