#!/usr/bin/env python3
"""THE evaluation harness. Every reported number comes from here.

FROZEN after Day 2 (see CLAUDE.md). Changing it invalidates all prior results.

Metrics, per PLAN.md:
  pearson, spearman, r2, n           - over all test sequences
  ...and the same restricted to the top and bottom deciles of the TRUE label.

Decile restriction measures whether a model can still rank within the extremes,
where range is compressed. Correlations there are expected to be much lower than
overall; that is range restriction, not failure.

Bootstrap CIs are computed on request (required for Yarrowia, PLAN.md D4).
"""
from __future__ import annotations
import json, csv, argparse, datetime, hashlib
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT/"results"/"runs.csv"

def _core(y: np.ndarray, p: np.ndarray) -> dict:
    """Pearson, Spearman, R^2 and n. R^2 is 1 - SSE/SST against the true mean,
    i.e. explained variance of the predictions, NOT the square of Pearson."""
    if y.size < 3 or np.std(p) == 0 or np.std(y) == 0:
        return {"n": int(y.size), "pearson": float("nan"), "spearman": float("nan"),
                "r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    sse = float(np.sum((y - p) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(y.size),
            "pearson": float(stats.pearsonr(y, p)[0]),
            "spearman": float(stats.spearmanr(y, p)[0]),
            "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
            "rmse": float(np.sqrt(sse / y.size)),
            "mae": float(np.mean(np.abs(y - p)))}

def testset_fingerprint(y: np.ndarray) -> str:
    """Stable 16-hex digest of the true-label vector.

    Two runs scored on different test sets get different fingerprints, so results
    that are not comparable cannot be silently compared. Rounded to 6dp so that
    float formatting differences do not change the digest.
    """
    b = np.round(np.asarray(y, dtype=np.float64).ravel(), 6).tobytes()
    return hashlib.blake2b(b, digest_size=8).hexdigest()

def precision_at_k(y: np.ndarray, p: np.ndarray, ks=(10, 100)) -> dict:
    """Of the k sequences the model ranks highest, what fraction are truly in the
    top k? The design-relevant metric: Spearman can look fine while the extreme
    tail, which is the part you would actually build, is wrong."""
    out = {}
    n = y.size
    for k in ks:
        if k > n: out[f"p_at_{k}"] = float("nan"); continue
        true_top = set(np.argsort(-y, kind="stable")[:k].tolist())
        pred_top = np.argsort(-p, kind="stable")[:k]
        out[f"p_at_{k}"] = float(sum(int(i) in true_top for i in pred_top) / k)
    return out

def evaluate(y_true, y_pred, n_boot: int = 0, seed: int = 0, ks=(10, 100)) -> dict:
    """Return overall metrics plus top/bottom-decile metrics.

    Deciles are defined on y_true, so the same sequences are compared across
    every model. n_boot > 0 adds percentile bootstrap 95% CIs.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {p.shape}")
    ok = np.isfinite(y) & np.isfinite(p)
    dropped = int((~ok).sum())
    y, p = y[ok], p[ok]

    lo, hi = np.quantile(y, 0.10), np.quantile(y, 0.90)
    out = {"overall": _core(y, p),
           "bottom_decile": _core(y[y <= lo], p[y <= lo]),
           "top_decile": _core(y[y >= hi], p[y >= hi]),
           "precision_at_k": precision_at_k(y, p, ks),
           "testset_fingerprint": testset_fingerprint(y),
           "n_dropped_nonfinite": dropped}

    if n_boot:
        rng = np.random.default_rng(seed)
        keys = ("pearson", "spearman", "r2")
        acc = {k: [] for k in keys}
        for _ in range(n_boot):
            i = rng.integers(0, y.size, y.size)
            m = _core(y[i], p[i])
            for k in keys: acc[k].append(m[k])
        out["overall_ci95"] = {k: [float(np.nanpercentile(acc[k], 2.5)),
                                   float(np.nanpercentile(acc[k], 97.5))] for k in keys}
        out["n_boot"] = n_boot
    return out

def log_run(config: dict, metrics: dict, path: Path = RUNS) -> None:
    """Append one row per run to results/runs.csv (PLAN.md convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "testset_fingerprint": metrics.get("testset_fingerprint", "")}
    row.update({f"cfg_{k}": v for k, v in config.items()})
    row.update(metrics.get("precision_at_k", {}))
    for sect in ("overall", "top_decile", "bottom_decile"):
        for k, v in metrics.get(sect, {}).items():
            row[f"{sect}_{k}"] = v
    new = not path.exists()
    prev = []
    if not new:
        with open(path) as fh: prev = list(csv.DictReader(fh))
    cols = sorted({c for r in prev for c in r} | set(row))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in prev: w.writerow(r)
        w.writerow(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="npz with arrays y_true, y_pred")
    ap.add_argument("--split", required=True)
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.npz)
    m = evaluate(d["y_true"], d["y_pred"], n_boot=a.boot)
    m["split"] = a.split
    print(json.dumps(m, indent=2))
