"""Is the 6,045-promoter RNA-seq benchmark learnable in-domain?

This is the gate the 81-promoter reporter benchmark failed. If ridge trained on
Yarrowia predicts held-out Yarrowia here, then cross-species transfer methods
finally become EVALUABLE - there is a reference point to compare against.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D

ALPHAS = np.logspace(0, 6, 13)

recs = json.load(open(ROOT / "data/yarrowia/yarrowia_rnaseq_benchmark.json"))
ups = [r["seq"] for r in recs]
y = np.log10(np.array([r["tpm"] for r in recs], float) + 1.0)
print(f"{len(recs):,} promoters | log10(TPM+1) {y.min():.2f}..{y.max():.2f}\n")

variants = {
    "proximal 80bp":  [u[-80:] for u in ups],
    "proximal 250bp": [u[-250:] for u in ups],
    "full 1000bp":    ups,
}
for name, seqs in variants.items():
    X = D.kmer_counts(seqs, k=6)
    scores = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(y)))):
        best, ba = -np.inf, ALPHAS[len(ALPHAS)//2]
        itr, iva = tr[:int(0.8*len(tr))], tr[int(0.8*len(tr)):]
        for a in ALPHAS:
            m = Ridge(alpha=a, solver="sparse_cg", max_iter=3000).fit(X[itr], y[itr])
            s = spearmanr(y[iva], m.predict(X[iva]))[0]
            if np.isfinite(s) and s > best: best, ba = s, a
        m = Ridge(alpha=ba, solver="sparse_cg", max_iter=3000).fit(X[tr], y[tr])
        scores.append(spearmanr(y[te], m.predict(X[te]))[0])
    scores = np.array(scores)
    # permutation null for the same procedure
    null = []
    for r in range(20):
        yp = np.random.default_rng(r).permutation(y)
        tr, te = next(iter(KFold(5, shuffle=True, random_state=0).split(np.arange(len(y)))))
        m = Ridge(alpha=1e3, solver="sparse_cg", max_iter=3000).fit(X[tr], yp[tr])
        null.append(spearmanr(yp[te], m.predict(X[te]))[0])
    print(f"  {name:16s} 5-fold CV rho = {scores.mean():+.3f} (sd {scores.std():.3f})   "
          f"null {np.mean(null):+.3f}+/-{np.std(null):.3f}")
