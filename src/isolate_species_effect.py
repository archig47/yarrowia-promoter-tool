"""Decompose the transfer failure: how much is the ASSAY, how much is the SPECIES?

Two matched benchmarks, identical in label type (RNA-seq of native promoters in
native chromatin) and differing only in species:

    de Boer model -> S. cerevisiae RNA-seq   = assay + context change only
    de Boer model -> Yarrowia RNA-seq        = assay + context + species

The gap between those two is the species contribution. Also reports in-domain
learnability for each, so a low transfer number can be told apart from a
benchmark that simply has no learnable signal.
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
from run_baselines import fit_ridge

ALPHAS = np.logspace(0, 6, 13)
WINDOWS = [80, 250]


def in_domain(X, y, k=5):
    scores = []
    for tr, te in KFold(k, shuffle=True, random_state=0).split(np.arange(len(y))):
        itr, iva = tr[:int(0.8 * len(tr))], tr[int(0.8 * len(tr)):]
        best, ba = -np.inf, ALPHAS[6]
        for a in ALPHAS:
            m = Ridge(alpha=a, solver="sparse_cg", max_iter=3000).fit(X[itr], y[itr])
            s = spearmanr(y[iva], m.predict(X[iva]))[0]
            if np.isfinite(s) and s > best: best, ba = s, a
        m = Ridge(alpha=ba, solver="sparse_cg", max_iter=3000).fit(X[tr], y[tr])
        scores.append(spearmanr(y[te], m.predict(X[te]))[0])
    return float(np.mean(scores)), float(np.std(scores))


# --- the source model, trained on de Boer random 80mers
pool_seq, pool_y = D.load_pool()
Xp = D.kmer_counts(pool_seq)
idx = D.subsamples("cluster")["n100000_seed0"]
_, extra = fit_ridge(Xp[idx], pool_y[idx], 0)
src = Ridge(alpha=extra["alpha"], solver="sparse_cg", max_iter=3000).fit(Xp[idx], pool_y[idx])
w, b = src.coef_.astype(np.float64), float(src.intercept_)
print(f"source: de Boer ridge, n=100,000, alpha={extra['alpha']:.0f}\n")

# --- the two matched RNA-seq benchmarks
sc = json.load(open(ROOT / "data/processed/scer_rnaseq_benchmark.json"))
yl = json.load(open(ROOT / "data/yarrowia/yarrowia_rnaseq_benchmark.json"))
sets = {
    "S. cerevisiae RNA-seq": ([r["seq"] for r in sc],
                              np.log10(np.array([r["expr"] for r in sc], float) + 1.0)),
    "Yarrowia RNA-seq":      ([r["seq"] for r in yl],
                              np.log10(np.array([r["tpm"] for r in yl], float) + 1.0)),
}

# --- the de Boer in-assay references, for context
ts, ty = D.load_primary_test()
print(f"  {'reference (same assay AND species as training)':46s}")
print(f"    random 80mers, clean labels      rho = {spearmanr(D.kmer_counts(ts) @ w + b, ty)[0]:+.3f}")

print(f"\n  {'benchmark':24s} {'n':>6} {'win':>5} {'zero-shot':>10} {'in-domain':>12}")
res = {}
for name, (seqs, y) in sets.items():
    for win in WINDOWS:
        X = D.kmer_counts([s[-win:] for s in seqs])
        zs = float(spearmanr(X @ w + b, y)[0])
        idm, sd = in_domain(X, y)
        res[f"{name}|{win}"] = {"zeroshot": round(zs, 4), "in_domain": round(idm, 4),
                                "in_domain_sd": round(sd, 4), "n": len(y)}
        print(f"  {name:24s} {len(y):>6,} {win:>5} {zs:>+10.3f} {idm:>+9.3f}±{sd:.3f}")

print("\n=== decomposition ===")
for win in WINDOWS:
    a = res[f"S. cerevisiae RNA-seq|{win}"]["zeroshot"]
    c = res[f"Yarrowia RNA-seq|{win}"]["zeroshot"]
    print(f"  {win}bp window:")
    print(f"    transfer to SAME species, different assay/context : {a:+.3f}")
    print(f"    transfer to DIFFERENT species, same assay/context : {c:+.3f}")
    print(f"    -> species contribution                            : {c - a:+.3f}")
json.dump(res, open(ROOT / "results" / "species_vs_assay.json", "w"), indent=2)
