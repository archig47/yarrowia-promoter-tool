"""Is the distal window signal real, or an artefact of picking the best of 47?

Selecting the maximum correlation over 47 tiled windows inflates it. The honest
test is a permutation: shuffle the measured strengths, repeat the ENTIRE window
sweep including the max, and see how often a value as large as the observed one
arises by chance.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D
from run_baselines import fit_ridge

WIN, STRIDE, N_TRAIN, N_PERM = 80, 20, 100_000, 2000

recs = json.load(open(ROOT / "data/yarrowia/yarrowia_benchmark.json"))
ups = [r["seq"] for r in recs]
y = np.log10(np.array([r["strength"] for r in recs], float))
L = min(len(u) for u in ups)
starts = list(range(0, L - WIN + 1, STRIDE))
tiles = [u[s:s + WIN] for u in ups for s in starts]

pool_seq, pool_y = D.load_pool()
subs = D.subsamples("cluster")
Xk_pool, Xk_tiles = D.kmer_counts(pool_seq), D.kmer_counts(tiles)

Ps = []
for seed in (0, 1, 2):
    predict, _ = fit_ridge(Xk_pool[subs[f"n{N_TRAIN}_seed{seed}"]], pool_y[subs[f"n{N_TRAIN}_seed{seed}"]], seed)
    Ps.append(np.array(predict(Xk_tiles)).reshape(len(recs), len(starts)))
P = np.mean(Ps, axis=0)

obs_prof = np.array([spearmanr(P[:, w], y)[0] for w in range(len(starts))])
obs_max = np.nanmax(np.abs(obs_prof))
best_w = int(np.nanargmax(np.abs(obs_prof)))
print(f"observed: best |rho| = {obs_max:.3f} at window ending -{L-starts[best_w]-WIN}bp from ATG")

rng = np.random.default_rng(0)
null = np.empty(N_PERM)
for i in range(N_PERM):
    yp = rng.permutation(y)
    prof = np.array([spearmanr(P[:, w], yp)[0] for w in range(len(starts))])
    null[i] = np.nanmax(np.abs(prof))
p = float((null >= obs_max).mean())
print(f"null max|rho| over {N_PERM} permutations: median {np.median(null):.3f}, "
      f"95th pct {np.percentile(null,95):.3f}, max {null.max():.3f}")
print(f"\npermutation p = {p:.4f}   ->  {'SIGNIFICANT' if p < 0.05 else 'NOT significant'} at 0.05")

# and the pre-specified proximal window, which needs no correction
prox = spearmanr(P[:, -1], y)
print(f"\npre-specified proximal 80bp window: rho={prox[0]:+.3f}, p={prox[1]:.3f} (no selection, no correction needed)")
json.dump({"observed_max_abs_rho": float(obs_max),
           "best_window_distance_from_atg": int(-(L-starts[best_w]-WIN)),
           "permutation_p": p, "n_permutations": N_PERM,
           "null_95th": float(np.percentile(null,95)),
           "proximal_rho": float(prox[0]), "proximal_p": float(prox[1])},
          open(ROOT/"results"/"yarrowia_permutation_test.json","w"), indent=2)
