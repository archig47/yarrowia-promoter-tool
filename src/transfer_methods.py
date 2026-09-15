"""D4: cross-species transfer methods, compared head to head.

Now that the Yarrowia RNA-seq benchmark is learnable in-domain (CV rho 0.39),
transfer methods can be EVALUATED rather than just tried. Every method is scored
on the same fixed held-out Yarrowia promoters, so the comparison is clean.

Methods
  scratch      ridge on k Yarrowia promoters only. The baseline every transfer
               method must beat to have earned its keep.
  zeroshot     the S. cerevisiae model applied directly. No Yarrowia labels.
  prior        ridge shrunk toward the S. cerevisiae COEFFICIENTS instead of
               toward zero:  argmin ||y - Xw||^2 + a||w - w_sc||^2.
               Closed form via the substitution u = w - w_sc, so it costs the
               same as ordinary ridge. This is the natural way to say "start from
               what S. cerevisiae taught us, and move only as far as the Yarrowia
               data insists".
  frozen       the S. cerevisiae model's prediction used as a single FEATURE,
               refit on Yarrowia (a 1-D linear probe). Tests whether the source
               model's ranking carries information even if its scale is wrong.
  frozen+kmer  that feature concatenated with the k-mer counts.

The headline output is a learning curve: accuracy against the number of Yarrowia
measurements, for each method. Transfer value is then read off in the same units
as D1 - how many Yarrowia measurements does borrowed knowledge save?
"""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D

KS = [50, 100, 300, 1000, 3000]
SEEDS = [0, 1, 2]
TEST_FRAC = 0.25
WINDOW = 250                 # empirically best for Yarrowia (see STATE.md)
ALPHAS = np.logspace(0, 6, 13)
SC_N_TRAIN = 100_000


def pick_alpha(fit, Xtr, ytr, Xva, yva):
    best, ba = -np.inf, ALPHAS[len(ALPHAS) // 2]
    for a in ALPHAS:
        s = spearmanr(yva, fit(a, Xtr, ytr)(Xva))[0]
        if np.isfinite(s) and s > best: best, ba = s, a
    return ba


def ridge_scratch(a, X, y):
    m = Ridge(alpha=a, solver="sparse_cg", max_iter=3000).fit(X, y)
    return lambda Z: m.predict(Z)


def ridge_prior(w_src, b_src):
    """Ridge shrunk toward the source coefficients rather than toward zero.

    Minimising ||y - Xw||^2 + a||w - w_src||^2 is ordinary ridge on the residual
    of the source model: substitute u = w - w_src, solve for u, add w_src back.
    """
    def fit(a, X, y):
        resid = y - (X @ w_src + b_src)
        m = Ridge(alpha=a, solver="sparse_cg", max_iter=3000).fit(X, resid)
        return lambda Z: (Z @ w_src + b_src) + m.predict(Z)
    return fit


def main():
    t0 = time.time()
    recs = json.load(open(ROOT / "data/yarrowia/yarrowia_rnaseq_benchmark.json"))
    y = np.log10(np.array([r["tpm"] for r in recs], float) + 1.0)
    n = len(y)

    # Two feature spaces, and the distinction matters:
    #   X80  - 80bp proximal window. The ONLY space where the source model's
    #          coefficients are valid, since it was trained on 80bp inserts.
    #          A 250bp window has ~3x the k-mer counts, so applying w_src there
    #          puts the source predictions on a different scale entirely and the
    #          prior method degenerates to copying them.
    #   X250 - 250bp window, empirically best for Yarrowia in-domain.
    # Transfer methods run in X80 (a fair comparison against an 80bp scratch
    # baseline); the 250bp scratch result is reported as the in-domain ceiling.
    X80 = D.kmer_counts([r["seq"][-80:] for r in recs], k=6)
    X250 = D.kmer_counts([r["seq"][-WINDOW:] for r in recs], k=6)
    print(f"Yarrowia: {n:,} promoters | feature spaces: 80bp and {WINDOW}bp", flush=True)

    # --- the S. cerevisiae source model (trained on random 80mers)
    pool_seq, pool_y = D.load_pool()
    subs = D.subsamples("cluster")
    Xp = D.kmer_counts(pool_seq)
    idx = subs[f"n{SC_N_TRAIN}_seed0"]
    from run_baselines import fit_ridge
    _, extra = fit_ridge(Xp[idx], pool_y[idx], 0)
    src = Ridge(alpha=extra["alpha"], solver="sparse_cg", max_iter=3000).fit(Xp[idx], pool_y[idx])
    w_src, b_src = src.coef_.astype(np.float64), float(src.intercept_)
    print(f"source model: S. cerevisiae ridge, n={SC_N_TRAIN:,}, alpha={extra['alpha']:.1f}", flush=True)

    # source predictions on every Yarrowia promoter, used by zeroshot and frozen
    p_src80 = X80 @ w_src + b_src              # valid: same window size as training
    p_src250 = X250 @ w_src + b_src            # out of scale, kept only to show it
    print(f"  zero-shot, 80bp window : rho={spearmanr(y, p_src80)[0]:+.4f}")
    print(f"  zero-shot, {WINDOW}bp window: rho={spearmanr(y, p_src250)[0]:+.4f} "
          f"(out of the source model's trained scale)\n", flush=True)

    out = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_te = int(TEST_FRAC * n)
        te, avail = perm[:n_te], perm[n_te:]
        yte, pte = y[te], p_src80[te]

        for k in KS:
            if k + 200 > len(avail): continue
            tr = avail[:k]
            va = avail[k:k + min(500, len(avail) - k)]
            ytr, yva = y[tr], y[va]
            res = {}

            # --- all transfer methods run in the 80bp space, where w_src is valid
            a = pick_alpha(ridge_scratch, X80[tr], ytr, X80[va], yva)
            res["scratch80"] = spearmanr(yte, ridge_scratch(a, X80[tr], ytr)(X80[te]))[0]
            res["zeroshot"] = spearmanr(yte, pte)[0]

            fit_p = ridge_prior(w_src, b_src)
            a = pick_alpha(fit_p, X80[tr], ytr, X80[va], yva)
            res["prior"] = spearmanr(yte, fit_p(a, X80[tr], ytr)(X80[te]))[0]

            F = lambda I: np.asarray(p_src80[I]).reshape(-1, 1)
            m = Ridge(alpha=1.0).fit(F(tr), ytr)
            res["frozen"] = spearmanr(yte, m.predict(F(te)))[0]

            aug = lambda I: sparse.hstack([X80[I], sparse.csr_matrix(F(I))], format="csr")
            a = pick_alpha(ridge_scratch, aug(tr), ytr, aug(va), yva)
            res["frozen+kmer"] = spearmanr(yte, ridge_scratch(a, aug(tr), ytr)(aug(te)))[0]

            # --- in-domain ceiling: no transfer, but the better window
            a = pick_alpha(ridge_scratch, X250[tr], ytr, X250[va], yva)
            res["scratch250"] = spearmanr(yte, ridge_scratch(a, X250[tr], ytr)(X250[te]))[0]

            for m_, v in res.items():
                out.setdefault(m_, {}).setdefault(k, []).append(float(v))
            print(f"  seed={seed} k={k:5d}  " +
                  "  ".join(f"{m_}={v:+.3f}" for m_, v in res.items()), flush=True)

    summary = {m_: {str(k): {"mean": round(float(np.mean(v)), 4),
                             "sd": round(float(np.std(v)), 4)}
                    for k, v in ks.items()} for m_, ks in out.items()}
    json.dump(summary, open(ROOT / "results" / "transfer_methods.json", "w"), indent=2)

    print(f"\n=== mean over {len(SEEDS)} seeds, held-out Yarrowia (n={int(TEST_FRAC*n):,}) ===")
    methods = ["zeroshot", "scratch80", "prior", "frozen", "frozen+kmer", "scratch250"]
    print("  k      " + "".join(f"{m_:>14s}" for m_ in methods))
    for k in KS:
        if str(k) not in summary["scratch80"]: continue
        print(f"  {k:<6d} " + "".join(f"{summary[m_][str(k)]['mean']:>+14.3f}" for m_ in methods))
    print(f"\nDONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
