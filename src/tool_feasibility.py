"""Is the Yarrowia model good enough to be a TOOL, not just a result?

rho = 0.39 is weak for ranking. But a lab rarely asks "rank these 6,000 promoters".
It asks "give me a strong promoter" or "is this one strong, medium or weak?".
Those are easier questions, and the right bar to test.

Three framings, hardest to easiest:
  ranking      spearman over everything                  (what we reported)
  tertile      can it sort into strong/medium/weak?      (what a lab asks)
  top-decile   of promoters it calls strong, how many are? (what you'd build with)
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

WINDOW, ALPHAS = 250, np.logspace(0, 6, 13)

recs = json.load(open(ROOT / "data/yarrowia/yarrowia_rnaseq_benchmark.json"))
seqs = [r["seq"][-WINDOW:] for r in recs]
y = np.log10(np.array([r["tpm"] for r in recs], float) + 1.0)
X = D.kmer_counts(seqs, k=6)
print(f"{len(y):,} Yarrowia promoters, {WINDOW}bp window\n")

# out-of-fold predictions so every promoter is scored by a model that never saw it
pred = np.zeros_like(y)
for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(y))):
    itr, iva = tr[:int(.8*len(tr))], tr[int(.8*len(tr)):]
    best, ba = -np.inf, ALPHAS[6]
    for a in ALPHAS:
        m = Ridge(alpha=a, solver="sparse_cg", max_iter=3000).fit(X[itr], y[itr])
        s = spearmanr(y[iva], m.predict(X[iva]))[0]
        if np.isfinite(s) and s > best: best, ba = s, a
    pred[te] = Ridge(alpha=ba, solver="sparse_cg", max_iter=3000).fit(X[tr], y[tr]).predict(X[te])

print(f"1. RANKING      spearman = {spearmanr(y, pred)[0]:+.3f}")

# 2. tertiles
def tert(v):
    q = np.quantile(v, [1/3, 2/3]); return np.digitize(v, q)
ta, tp = tert(y), tert(pred)
acc = float((ta == tp).mean())
adj = float((np.abs(ta - tp) <= 1).mean())
print(f"\n2. TERTILE      exact accuracy = {acc:.1%}  (chance 33.3%)")
print(f"                within one class = {adj:.1%}")
print(f"   confusion (rows = true weak/med/strong, cols = predicted):")
for i in range(3):
    row = [int(((ta == i) & (tp == j)).sum()) for j in range(3)]
    print(f"     {['weak  ','medium','strong'][i]} {row}")

# 3. the practical question: if you pick the model's top N, what do you get?
print(f"\n3. PICKING STRONG PROMOTERS")
print(f"   {'shortlist':>10} {'% truly in top tertile':>24} {'% truly in top decile':>23}")
top_t = ta == 2
top_d = y >= np.quantile(y, 0.9)
for N in (10, 25, 50, 100):
    idx = np.argsort(-pred)[:N]
    print(f"   {N:>10} {top_t[idx].mean()*100:>23.0f}% {top_d[idx].mean()*100:>22.0f}%")
print(f"   {'(baseline)':>10} {top_t.mean()*100:>23.0f}% {top_d.mean()*100:>22.0f}%")

# 4. what does that mean in lab terms?
hit = top_t[np.argsort(-pred)[:25]].mean()
base = top_t.mean()
print(f"\n4. IN LAB TERMS: to find one strong promoter you would screen")
print(f"     {1/base:.1f} promoters at random")
print(f"     {1/hit:.1f} promoters from the model's shortlist")
print(f"     -> {base and hit/base:.1f}x enrichment")
