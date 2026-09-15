"""Why does anchoring to S. cerevisiae hurt? Compare what each species' model learned.

If the two species' sequence-to-expression rules genuinely conflict, the learned
k-mer coefficient vectors should be uncorrelated or anti-correlated. That would
explain why shrinking toward the S. cerevisiae solution is worse than shrinking
toward zero: the prior points the wrong way.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D

BASES = "ACGT"

# --- S. cerevisiae model (80bp random inserts)
pool_seq, pool_y = D.load_pool()
Xp = D.kmer_counts(pool_seq)
idx = D.subsamples("cluster")[f"n100000_seed0"]
from run_baselines import fit_ridge
_, extra = fit_ridge(Xp[idx], pool_y[idx], 0)
sc = Ridge(alpha=extra["alpha"], solver="sparse_cg", max_iter=3000).fit(Xp[idx], pool_y[idx])

# --- Yarrowia model (80bp proximal, same feature space)
recs = json.load(open(ROOT / "data/yarrowia/yarrowia_rnaseq_benchmark.json"))
Xy = D.kmer_counts([r["seq"][-80:] for r in recs])
yy = np.log10(np.array([r["tpm"] for r in recs], float) + 1.0)
yl = Ridge(alpha=1e3, solver="sparse_cg", max_iter=3000).fit(Xy, yy)

a, b = sc.coef_.ravel(), yl.coef_.ravel()
print(f"6-mer coefficient vectors, both in the same 4,096-dim space\n")
print(f"  pearson  S.cerevisiae vs Yarrowia : {pearsonr(a, b)[0]:+.4f}")
print(f"  spearman                          : {spearmanr(a, b)[0]:+.4f}")

# do the two models agree on which k-mers matter most?
top_sc = set(np.argsort(-a)[:200]); top_yl = set(np.argsort(-b)[:200])
bot_sc = set(np.argsort(a)[:200]);  bot_yl = set(np.argsort(b)[:200])
print(f"\n  top-200 activating k-mers shared : {len(top_sc & top_yl)}/200")
print(f"  top-200 repressing k-mers shared : {len(bot_sc & bot_yl)}/200")
print(f"  S.cer activators that Yarrowia REPRESSES : {len(top_sc & bot_yl)}/200")

# GC content of each model's strongest k-mers
def gc_of(indices, k=6):
    out = []
    for i in indices:
        s, x = "", int(i)
        for _ in range(k):
            s = BASES[x % 4] + s; x //= 4
        out.append((s.count("G") + s.count("C")) / k)
    return float(np.mean(out))
print(f"\n  mean GC of top-200 activating k-mers:")
print(f"    S. cerevisiae : {gc_of(np.argsort(-a)[:200]):.3f}")
print(f"    Yarrowia      : {gc_of(np.argsort(-b)[:200]):.3f}")

# direct: GC content vs expression in each species
gc_y = np.array([(s.count("G")+s.count("C"))/len(s) for s in (r["seq"][-80:] for r in recs)])
ts, ty = D.load_primary_test()
gc_s = np.array([(s.count("G")+s.count("C"))/len(s) for s in ts])
print(f"\n  GC content vs expression, measured directly:")
print(f"    S. cerevisiae (n={len(ty):,}) : spearman {spearmanr(gc_s, ty)[0]:+.4f}")
print(f"    Yarrowia      (n={len(yy):,}) : spearman {spearmanr(gc_y, yy)[0]:+.4f}")
