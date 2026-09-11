"""D4: zero-shot transfer to Yarrowia lipolytica.

81 endogenous promoters, 80bp upstream regions, measured strengths spanning
2,573-fold. Rank correlation only with bootstrap CIs, per PLAN.md - the assay is
completely different from de Boer's FACS bins, so absolute values are meaningless
and only the ordering can be compared.
"""
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D
from evaluate import evaluate, log_run

N_TRAIN, SEEDS = 100_000, [0, 1, 2]

def main():
    recs = json.load(open(ROOT / "data/yarrowia/yarrowia_benchmark.json"))
    yseq = [r["seq"] for r in recs]
    # strengths span orders of magnitude; rank correlation is unaffected by the
    # transform, but log makes the R2/RMSE columns interpretable rather than absurd
    yy = np.log10(np.array([r["strength"] for r in recs], dtype=np.float64)).astype(np.float32)
    print(f"Yarrowia benchmark: {len(yseq)} promoters, log10 strength "
          f"{yy.min():.2f}..{yy.max():.2f}", flush=True)

    pool_seq, pool_y = D.load_pool()
    subs = D.subsamples("cluster")
    Xk_pool, Xk_y = D.kmer_counts(pool_seq), D.kmer_counts(yseq)
    Xo_y = D.onehot(yseq)

    for model in ("ridge", "cnn"):
        rhos = []
        for seed in SEEDS:
            idx = subs[f"n{N_TRAIN}_seed{seed}"]
            t1 = time.time()
            if model == "ridge":
                from run_baselines import fit_ridge
                predict, _ = fit_ridge(Xk_pool[idx], pool_y[idx], seed)
                pred = predict(Xk_y)
            else:
                from run_baselines import fit_cnn
                predict, _ = fit_cnn(D.onehot([pool_seq[i] for i in idx]), pool_y[idx], seed)
                pred = predict(Xo_y)
            met = evaluate(yy, pred, n_boot=2000, seed=seed)
            log_run({"model": model, "split": "cluster", "n": N_TRAIN, "seed": seed,
                     "evalset": "yarrowia", "deliverable": "D4"}, met)
            ci = met["overall_ci95"]["spearman"]
            rhos.append(met["overall"]["spearman"])
            print(f"  {model:6s} seed={seed} rho={met['overall']['spearman']:+.4f} "
                  f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  ({time.time()-t1:.0f}s)", flush=True)
        print(f"  -> {model}: mean rho {np.mean(rhos):+.4f} (sd {np.std(rhos):.4f})\n", flush=True)

if __name__ == "__main__":
    main()
