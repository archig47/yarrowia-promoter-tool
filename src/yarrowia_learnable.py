"""Before asking whether transfer helps: is there ANY learnable signal in these 81?

If a model trained on Yarrowia itself cannot predict held-out Yarrowia promoters,
then "transfer does not help" says nothing - there would be nothing to transfer TO.
This is the sanity check that makes the fine-tuning arm interpretable.

Two feature sets:
  proximal   the 80bp before the ATG (what the S. cerevisiae models consume)
  full       k-mer counts over all 1000bp upstream (no length constraint, since
             we are training from scratch here)
Repeated 5-fold cross-validation, 20 repeats, ridge with alpha chosen inside
each training fold only.
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

ALPHAS = np.logspace(-2, 6, 17)
REPEATS = 20


def cv_spearman(X, y, repeats=REPEATS, k=5):
    out = []
    for r in range(repeats):
        pred = np.zeros_like(y)
        for tr, te in KFold(k, shuffle=True, random_state=r).split(X):
            best, ba = -np.inf, ALPHAS[len(ALPHAS)//2]
            inner = KFold(4, shuffle=True, random_state=r).split(X[tr])
            for a in ALPHAS:
                sc = []
                for itr, iva in KFold(4, shuffle=True, random_state=r).split(np.arange(len(tr))):
                    m = Ridge(alpha=a).fit(X[tr][itr], y[tr][itr])
                    s = spearmanr(y[tr][iva], m.predict(X[tr][iva]))[0]
                    if np.isfinite(s): sc.append(s)
                if sc and np.mean(sc) > best: best, ba = np.mean(sc), a
            pred[te] = Ridge(alpha=ba).fit(X[tr], y[tr]).predict(X[te])
        out.append(spearmanr(y, pred)[0])
    return np.array(out)


recs = json.load(open(ROOT / "data/yarrowia/yarrowia_benchmark.json"))
ups = [r["seq"] for r in recs]
y = np.log10(np.array([r["strength"] for r in recs], float))
print(f"{len(recs)} Yarrowia promoters\n")

feats = {
    "proximal 80bp": D.kmer_counts([u[-80:] for u in ups], k=6).toarray(),
    "full 1000bp":   D.kmer_counts(ups, k=6).toarray(),
    "full 1000bp k=4": D.kmer_counts(ups, k=4).toarray(),
}
for name, X in feats.items():
    X = np.asarray(X, dtype=np.float64)
    s = cv_spearman(X, y)
    # permutation null for the same procedure
    null = []
    for r in range(50):
        yp = np.random.default_rng(r).permutation(y)
        null.append(cv_spearman(X, yp, repeats=1)[0])
    null = np.array(null)
    p = float((null >= s.mean()).mean())
    print(f"  {name:18s} CV rho = {s.mean():+.3f} (sd {s.std():.3f})   "
          f"null {np.mean(null):+.3f}+/-{np.std(null):.3f}   p={p:.3f}")
