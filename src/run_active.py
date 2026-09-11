"""D2: selection curves. Does choosing which promoters to measure beat choosing at random?

Pool-based simulation, non-iterative (PLAN.md puts iterative AL in Checkpoint 2).
For each budget and strategy, pick that many sequences from the 190k training pool,
fit ridge on them, and score through the frozen src/evaluate.py.

Strategies
  random       uniform draw. The baseline everything is measured against.
  uncertainty  spend a seed set of 100 labels, fit a bootstrap ensemble, then take
               the pool sequences the ensemble disagrees most about.
  diversity    no labels needed: reduce k-mer space with SVD, k-means into as many
               clusters as the budget, take the sequence nearest each centre.
  hybrid       take the most uncertain 3x the remaining budget, then k-means within
               those so the chosen set is uncertain AND spread out.

The seed set counts against the budget for uncertainty and hybrid - otherwise they
would be scored on more labels than random gets, which would not be a fair comparison.

Ridge is the workhorse: it is stable at low n (no training loop to destabilise) so
differences between curves reflect the SELECTION, not training noise.

NOTE (deviation from PLAN.md): diversity uses k-means on k-mer features, not on
language-model embeddings, because the GPU box is unreachable. Recorded in STATE.md.
"""
import sys, time, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D
from evaluate import evaluate, log_run

BUDGETS = [100, 300, 1_000, 3_000]
SEEDS = [0, 1, 2]
SEED_SET = 100                      # labels spent before uncertainty can be computed
ALPHAS = np.logspace(-1, 5, 13)
SVD_DIM = 64
OUT = ROOT / "results" / "runs_d2.csv"


def fit_ridge(X, y, seed):
    """Same recipe as the Day 3 baseline: pick alpha on a held-out fifth, refit on all."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); cut = max(int(0.8 * n), n - 2000)
    tr, va = perm[:cut], perm[cut:]
    best, best_a = -np.inf, ALPHAS[len(ALPHAS) // 2]
    if va.size >= 5:
        for a in ALPHAS:
            m = Ridge(alpha=a, solver="sparse_cg", max_iter=2000).fit(X[tr], y[tr])
            s = spearmanr(y[va], m.predict(X[va]))[0]
            if np.isfinite(s) and s > best: best, best_a = s, a
    return Ridge(alpha=best_a, solver="sparse_cg", max_iter=2000).fit(X, y)


def pick(strategy, Xp, yp, Z, budget, seed):
    """Return `budget` pool indices chosen by `strategy`. Only labels inside the
    returned set may be used - that is what makes the comparison fair."""
    rng = np.random.default_rng(1000 + seed)
    n_pool = Xp.shape[0]

    if strategy == "random":
        return rng.choice(n_pool, budget, replace=False)

    if strategy == "diversity":
        km = MiniBatchKMeans(n_clusters=budget, random_state=seed, n_init=3,
                             batch_size=4096, max_iter=100).fit(Z)
        # nearest real sequence to each cluster centre
        chosen = []
        lab = km.labels_
        for k in range(budget):
            members = np.where(lab == k)[0]
            if members.size == 0: continue
            d = ((Z[members] - km.cluster_centers_[k]) ** 2).sum(1)
            chosen.append(members[d.argmin()])
        chosen = np.unique(chosen)
        if chosen.size < budget:      # top up if some clusters came out empty
            extra = rng.choice(np.setdiff1d(np.arange(n_pool), chosen),
                               budget - chosen.size, replace=False)
            chosen = np.concatenate([chosen, extra])
        return chosen[:budget]

    # uncertainty and hybrid both start by spending SEED_SET labels
    seed_idx = rng.choice(n_pool, min(SEED_SET, budget), replace=False)
    if budget <= SEED_SET:
        return seed_idx
    remaining = budget - seed_idx.size

    preds = []
    for b in range(5):                       # bootstrap ensemble over the seed set
        bi = np.random.default_rng(seed * 100 + b).integers(0, seed_idx.size, seed_idx.size)
        m = Ridge(alpha=100.0, solver="sparse_cg", max_iter=2000)
        m.fit(Xp[seed_idx[bi]], yp[seed_idx[bi]])
        preds.append(m.predict(Xp))
    var = np.var(np.vstack(preds), axis=0)
    var[seed_idx] = -np.inf                  # never re-pick what we already paid for

    if strategy == "uncertainty":
        return np.concatenate([seed_idx, np.argsort(-var)[:remaining]])

    # hybrid: cluster within the most uncertain candidates so the set is also spread out
    cand = np.argsort(-var)[:min(remaining * 3, n_pool - seed_idx.size)]
    km = MiniBatchKMeans(n_clusters=remaining, random_state=seed, n_init=3,
                         batch_size=4096, max_iter=100).fit(Z[cand])
    chosen, lab = [], km.labels_
    for k in range(remaining):
        members = np.where(lab == k)[0]
        if members.size == 0: continue
        d = ((Z[cand][members] - km.cluster_centers_[k]) ** 2).sum(1)
        chosen.append(cand[members[d.argmin()]])
    chosen = np.unique(chosen)
    if chosen.size < remaining:
        pool_left = np.setdiff1d(cand, chosen)
        chosen = np.concatenate([chosen, pool_left[:remaining - chosen.size]])
    return np.concatenate([seed_idx, chosen[:remaining]])


def main():
    t0 = time.time()
    pool_seq, pool_y = D.load_pool()
    Xpool = D.kmer_counts(pool_seq)
    ts, ty = D.load_primary_test(); Xt = D.kmer_counts(ts)
    tr_idx, _ = D.split("cluster")
    Xp, yp = Xpool[tr_idx], pool_y[tr_idx]
    print(f"pool for selection: {Xp.shape[0]:,} sequences", flush=True)

    print("reducing k-mer space for clustering...", flush=True)
    Z = TruncatedSVD(n_components=SVD_DIM, random_state=0).fit_transform(Xp).astype(np.float32)
    print(f"  SVD -> {Z.shape} ({time.time()-t0:.0f}s)", flush=True)

    for strategy in ("random", "uncertainty", "diversity", "hybrid"):
        for budget in BUDGETS:
            for seed in SEEDS:
                t1 = time.time()
                idx = pick(strategy, Xp, yp, Z, budget, seed)
                assert len(idx) == len(set(idx.tolist())), "duplicate selections"
                m = fit_ridge(Xp[idx], yp[idx], seed)
                met = evaluate(ty, m.predict(Xt))
                log_run({"model": "ridge", "deliverable": "D2", "strategy": strategy,
                         "budget": int(budget), "seed": seed, "n_selected": int(len(idx)),
                         "evalset": "primary", "split": "cluster"}, met, path=OUT)
                print(f"  {strategy:12s} budget={budget:5d} s={seed} "
                      f"rho={met['overall']['spearman']:+.4f} ({time.time()-t1:.0f}s)", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
