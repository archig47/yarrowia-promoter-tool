"""D4, properly controlled: where in a Yarrowia promoter does the model find signal?

The first attempt used the 80bp immediately upstream of the ATG and got rho ~ 0.
But the literature says that window is the wrong place to look:
  - a minimal Y. lipolytica promoter spans 130-260bp upstream of ATG (Blazeck et al.)
  - the TATA box sits 40-120bp upstream of the TSS, often beyond an 80bp window
  - UAS elements, which drive strength, sit at the 5' END of the promoter
  - Y. lipolytica promoters are explicitly "enhancer limited"

So instead of one window, tile the 80bp window the models were TRAINED on across
1000bp of upstream sequence. Every model stays in its trained input regime (the CNN
included), and we get a profile of informativeness against distance from the ATG.

If the signal really lives in distal UAS elements, correlation should be higher for
distal windows than proximal ones. If nothing is informative anywhere, the failure is
not about window choice.
"""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D
from evaluate import evaluate, log_run

WIN, STRIDE, N_TRAIN = 80, 20, 100_000
SEEDS = [0, 1, 2]


def boot_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    s = np.array([v for v in s if np.isfinite(v)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    recs = json.load(open(ROOT / "data/yarrowia/yarrowia_benchmark.json"))
    ups = [r["seq"] for r in recs]
    y = np.log10(np.array([r["strength"] for r in recs], float))
    L = min(len(u) for u in ups)
    starts = list(range(0, L - WIN + 1, STRIDE))
    print(f"{len(recs)} promoters, {L}bp upstream, {len(starts)} tiled {WIN}bp windows "
          f"(stride {STRIDE})", flush=True)

    # windows[i][w] = the w-th window of promoter i
    tiles = [[u[s:s + WIN] for s in starts] for u in ups]
    flat = [t for row in tiles for t in row]

    pool_seq, pool_y = D.load_pool()
    subs = D.subsamples("cluster")
    Xk_pool = D.kmer_counts(pool_seq)
    Xk_flat = D.kmer_counts(flat)
    Xo_flat = D.onehot(flat)

    out = {}
    for model in ("ridge", "cnn"):
        prof_seeds, agg_seeds = [], {k: [] for k in ("max", "mean", "proximal", "distal")}
        for seed in SEEDS:
            idx = subs[f"n{N_TRAIN}_seed{seed}"]
            t1 = time.time()
            if model == "ridge":
                from run_baselines import fit_ridge
                predict, _ = fit_ridge(Xk_pool[idx], pool_y[idx], seed)
                p = predict(Xk_flat)
            else:
                from run_baselines import fit_cnn
                predict, _ = fit_cnn(D.onehot([pool_seq[i] for i in idx]), pool_y[idx], seed)
                p = predict(Xo_flat)
            P = np.array(p).reshape(len(recs), len(starts))     # (promoters, windows)

            # informativeness of each window position on its own
            prof = np.array([spearmanr(P[:, w], y)[0] for w in range(len(starts))])
            prof_seeds.append(prof)
            aggs = {"max": P.max(1), "mean": P.mean(1),
                    "proximal": P[:, -1], "distal": P[:, 0]}
            for k, v in aggs.items():
                agg_seeds[k].append(spearmanr(v, y)[0])
            met = evaluate(y, aggs["max"], n_boot=2000, seed=seed)
            log_run({"model": model, "split": "cluster", "n": N_TRAIN, "seed": seed,
                     "evalset": "yarrowia_tiled_max", "deliverable": "D4"}, met)
            print(f"  {model:6s} seed={seed}  best-window rho={np.nanmax(prof):+.3f} "
                  f"at -{L - starts[int(np.nanargmax(prof))] - WIN}bp   ({time.time()-t1:.0f}s)", flush=True)

        prof = np.nanmean(prof_seeds, axis=0)
        dist = [-(L - s - WIN) for s in starts]        # distance of window END from ATG
        out[model] = {"distance_from_atg": dist, "spearman_profile": [round(float(v), 4) for v in prof],
                      "aggregate": {k: round(float(np.mean(v)), 4) for k, v in agg_seeds.items()}}
        print(f"  -> {model} aggregates: " +
              "  ".join(f"{k}={np.mean(v):+.3f}" for k, v in agg_seeds.items()))
        b = int(np.nanargmax(np.abs(prof)))
        lo, hi = boot_ci(np.array([0.0]), np.array([0.0])) if False else (None, None)
        print(f"     strongest single window: rho={prof[b]:+.3f} ending {dist[b]}bp from ATG\n", flush=True)

    json.dump(out, open(ROOT / "results" / "yarrowia_window_profile.json", "w"), indent=2)
    print("wrote results/yarrowia_window_profile.json")


if __name__ == "__main__":
    main()
