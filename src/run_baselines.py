#!/usr/bin/env python3
"""Day 3 baselines: k-mer ridge, LightGBM, CNN across all n x 3 seeds x 2 splits.

Every number goes through the frozen src/evaluate.py. Nothing is reported from here.
Eval sets per run:
  primary   - de Boer high-quality pTpA/glucose (n=9,982). THE headline test set.
  secondary - held-out 10k from the working pool, per split.
  native    - Native80 real yeast promoters (D4a ladder: random -> natural)
  spikein   - N80 random controls from the same experiment (matched control)
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
from evaluate import evaluate, log_run

NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = [0, 1, 2]
ALPHAS = np.logspace(-1, 5, 13)

def _val_split(n, seed, frac=0.2, cap=2000):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); cut = max(int((1-frac)*n), n-cap)
    return perm[:cut], perm[cut:]

# ---------------------------------------------------------------- ridge
def fit_ridge(Xtr, ytr, seed):
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    tr, va = _val_split(Xtr.shape[0], seed)
    best, best_a = -np.inf, ALPHAS[len(ALPHAS)//2]
    if va.size >= 5:
        for a in ALPHAS:
            m = Ridge(alpha=a, solver="sparse_cg", max_iter=2000).fit(Xtr[tr], ytr[tr])
            s = spearmanr(ytr[va], m.predict(Xtr[va]))[0]
            if np.isfinite(s) and s > best: best, best_a = s, a
    m = Ridge(alpha=best_a, solver="sparse_cg", max_iter=2000).fit(Xtr, ytr)
    return m.predict, {"alpha": float(best_a)}

# ---------------------------------------------------------------- lightgbm
def fit_lgbm(Xtr, ytr, seed):
    import lightgbm as lgb
    tr, va = _val_split(Xtr.shape[0], seed)
    n = Xtr.shape[0]
    params = dict(objective="regression", metric="l2", verbosity=-1, seed=seed,
                  learning_rate=0.05, num_leaves=min(63, max(4, n//20)),
                  min_data_in_leaf=max(2, min(20, n//50)),
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1)
    dtr = lgb.Dataset(Xtr[tr], label=ytr[tr])
    dva = lgb.Dataset(Xtr[va], label=ytr[va], reference=dtr)
    bst = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    return bst.predict, {"num_trees": int(bst.num_trees()),
                         "num_leaves": params["num_leaves"]}

# ---------------------------------------------------------------- cnn
def fit_cnn(Xtr, ytr, seed):
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(4, 128, 8, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 96, 8, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(96, 64, 8, padding="same"), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Dropout(0.2), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x): return self.f(x).squeeze(-1)

    tr, va = _val_split(Xtr.shape[0], seed)
    mu, sd = float(ytr[tr].mean()), float(ytr[tr].std() + 1e-8)
    Xa = torch.tensor(Xtr[tr]); ya = torch.tensor((ytr[tr]-mu)/sd)
    Xv = torch.tensor(Xtr[va]).to(dev); yv = ytr[va]
    net = Net().to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = int(min(128, max(8, Xa.shape[0]//4)))
    best, best_state, patience = -np.inf, None, 0
    for epoch in range(200):
        net.train()
        perm = torch.randperm(Xa.shape[0])
        for i in range(0, Xa.shape[0], bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = ((net(Xa[b].to(dev)) - ya[b].to(dev))**2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv = net(Xv).cpu().numpy()
        s = spearmanr(yv, pv)[0]
        if np.isfinite(s) and s > best:
            best, patience = s, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 15: break
    if best_state: net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 2048):
                out.append(net(torch.tensor(X[i:i+2048]).to(dev)).cpu().numpy())
        return np.concatenate(out) * sd + mu
    return predict, {"params": npar, "epochs": epoch+1, "batch": bs}

FITTERS = {"ridge": fit_ridge, "lgbm": fit_lgbm, "cnn": fit_cnn}

def main(model):
    t0 = time.time()
    feat = D.onehot if model == "cnn" else D.kmer_counts
    print(f"featurising for {model}...", flush=True)
    pool_seq, pool_y = D.load_pool(); Xpool = feat(pool_seq)
    ts, ty = D.load_primary_test(); Xt = feat(ts)
    ns_, ny, nat = D.load_native80(); Xn = feat(ns_)
    evalsets = {"primary": (Xt, ty), "native": (Xn[nat], ny[nat]),
                "spikein": (Xn[~nat], ny[~nat])}
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)

    for split in ("cluster", "random"):
        tr_idx, te_idx = D.split(split)
        subs = D.subsamples(split)
        evalsets["secondary"] = (Xpool[te_idx], pool_y[te_idx])
        for n in NS:
            for seed in SEEDS:
                key = f"n{n}_seed{seed}"
                if key not in subs: continue
                idx = subs[key]; t1 = time.time()
                pred, extra = FITTERS[model](Xpool[idx], pool_y[idx], seed)
                for ename, (Xe, ye) in evalsets.items():
                    met = evaluate(ye, pred(Xe))
                    cfg = {"model": model, "split": split, "n": n, "seed": seed,
                           "evalset": ename}; cfg.update(extra)
                    log_run(cfg, met)
                    if ename == "primary":
                        print(f"  {split:7s} n={n:6d} s={seed} "
                              f"rho={met['overall']['spearman']:+.4f} "
                              f"r2={met['overall']['r2']:+.4f} "
                              f"P@100={met['precision_at_k']['p_at_100']:.2f} "
                              f"{extra} ({time.time()-t1:.0f}s)", flush=True)
    print(f"DONE {model} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ridge")
    main(ap.parse_args().model)
