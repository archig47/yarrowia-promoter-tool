#!/usr/bin/env python3
"""Day 3 baselines: k-mer ridge, LightGBM, CNN across all n x 3 seeds x 2 splits.

Every number goes through the frozen src/evaluate.py. Nothing is reported from here.
Evaluation sets, per run:
  primary   - de Boer high-quality pTpA/glucose (n=9,982). THE headline test set.
  secondary - held-out 10k from the working pool, per the split being used.
  native    - Native80 real yeast promoters (D4a ladder, random->natural transfer)
  spikein   - N80 random controls from the same Native80 experiment (matched control)
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
from evaluate import evaluate, log_run

ROOT = Path(__file__).resolve().parent.parent
NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = [0, 1, 2]
ALPHAS = np.logspace(-1, 5, 13)

def fit_ridge(Xtr, ytr, seed):
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); cut = max(int(0.8 * n), n - 2000)
    tr, va = perm[:cut], perm[cut:]
    best, best_a = -np.inf, ALPHAS[len(ALPHAS)//2]
    if va.size >= 5:
        for a in ALPHAS:
            m = Ridge(alpha=a, solver="sparse_cg", max_iter=2000)
            m.fit(Xtr[tr], ytr[tr])
            s = spearmanr(ytr[va], m.predict(Xtr[va]))[0]
            if np.isfinite(s) and s > best: best, best_a = s, a
    m = Ridge(alpha=best_a, solver="sparse_cg", max_iter=2000)
    m.fit(Xtr, ytr)
    return m, float(best_a)

def main(model):
    t0 = time.time()
    print("featurising...", flush=True)
    pool_seq, pool_y = D.load_pool()
    Xpool = D.kmer_counts(pool_seq)
    ts, ty = D.load_primary_test(); Xt = D.kmer_counts(ts)
    ns_, ny, nat = D.load_native80(); Xn = D.kmer_counts(ns_)
    evalsets = {"primary": (Xt, ty),
                "native": (Xn[nat], ny[nat]),
                "spikein": (Xn[~nat], ny[~nat])}
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)

    for split in ("cluster", "random"):
        tr_idx, te_idx = D.split(split)
        subs = D.subsamples(split)
        evalsets["secondary"] = (Xpool[te_idx], pool_y[te_idx])
        for n in NS:
            for seed in SEEDS:
                key = f"n{n}_seed{seed}"
                if key not in subs: continue
                idx = subs[key]
                t1 = time.time()
                m, alpha = fit_ridge(Xpool[idx], pool_y[idx], seed)
                for ename, (Xe, ye) in evalsets.items():
                    met = evaluate(ye, m.predict(Xe))
                    log_run({"model": model, "split": split, "n": n, "seed": seed,
                             "evalset": ename, "alpha": alpha, "k": 6}, met)
                    if ename == "primary":
                        print(f"  {split:7s} n={n:6d} s={seed} alpha={alpha:8.1f} "
                              f"rho={met['overall']['spearman']:+.4f} "
                              f"r2={met['overall']['r2']:+.4f} "
                              f"P@100={met['precision_at_k']['p_at_100']:.2f} "
                              f"({time.time()-t1:.0f}s)", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ridge")
    main(ap.parse_args().model)
