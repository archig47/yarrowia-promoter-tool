#!/usr/bin/env python3
"""Predict Yarrowia lipolytica promoter strength from sequence.

    python predict.py --top 25                    strongest promoters genome-wide
    python predict.py --gene YALI0C09988g         look up one gene
    python predict.py --seq ACGT...               score any 250bp sequence
    python predict.py --top 25 --consistent       only promoters stable across carbon sources

Trained on 6,026 Yarrowia promoters with expression measured by RNA-seq across two
strains and three carbon sources (Lubuta et al. 2019). The model is k-mer ridge
regression on the 250bp upstream of the start codon - 250bp because that is where
this organism's promoters live (Blazeck et al.), and because it outperformed both
80bp and 1000bp in cross-validation.

WHAT IT PREDICTS, AND WHAT IT DOES NOT
  predicts   relative mRNA abundance driven by that promoter. Cross-validated
             Spearman 0.40. Of its top 25 genome-wide, ~14 are truly in the top
             decile, against ~2.5 by chance - a 5.6x enrichment.
  reports    measured consistency across carbon sources, for genes in the dataset.
             This CANNOT be predicted from sequence (CV Spearman 0.10), only
             looked up, so a novel sequence gets no consistency value.
  does NOT   predict protein level, promoter activity in a reporter construct, or
             cell-to-cell heterogeneity. That last one matters: Patel et al. 2026
             show a strong but macro-heterogeneous promoter can cost threefold in
             yield. Nothing here assesses it.

Use it to shortlist candidates for lab screening, not to choose a promoter blind.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data as D

BENCH = ROOT / "data" / "yarrowia" / "yarrowia_multicondition.json"
MODEL = ROOT / "results" / "promoter_model.pkl"
WINDOW = 250
ALPHAS = np.logspace(0, 6, 13)


def train():
    """Fit the model and cache it, with honest held-out performance recorded."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr

    recs = json.load(open(BENCH))
    X = D.kmer_counts([r["seq"][-WINDOW:] for r in recs], k=6)
    y = np.array([r["strength"] for r in recs], float)
    print(f"training on {len(y):,} Yarrowia promoters ({WINDOW}bp window)...", file=sys.stderr)

    # honest cross-validated score, and out-of-fold predictions for calibration
    oof = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(y))):
        itr, iva = tr[:int(.8 * len(tr))], tr[int(.8 * len(tr)):]
        best, ba = -np.inf, ALPHAS[6]
        for a in ALPHAS:
            m = Ridge(alpha=a, solver="sparse_cg", max_iter=3000).fit(X[itr], y[itr])
            s = spearmanr(y[iva], m.predict(X[iva]))[0]
            if np.isfinite(s) and s > best: best, ba = s, a
        oof[te] = Ridge(alpha=ba, solver="sparse_cg", max_iter=3000).fit(X[tr], y[tr]).predict(X[te])
    cv = float(spearmanr(y, oof)[0])

    final = Ridge(alpha=ALPHAS[6], solver="sparse_cg", max_iter=3000).fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    with open(MODEL, "wb") as fh:
        pickle.dump({"model": final, "cv_spearman": cv, "window": WINDOW,
                     "train_pred": oof, "train_y": y}, fh)
    print(f"  cross-validated Spearman {cv:.3f}  (cached to {MODEL.name})", file=sys.stderr)
    return pickle.load(open(MODEL, "rb"))


def load():
    if MODEL.exists():
        return pickle.load(open(MODEL, "rb"))
    return train()


def band(pct):
    return "STRONG" if pct >= 90 else "strong" if pct >= 67 else \
           "medium" if pct >= 33 else "weak"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--top", type=int, metavar="N", help="list the N strongest promoters")
    g.add_argument("--gene", metavar="YALI0...", help="look up one gene")
    g.add_argument("--seq", metavar="ACGT...", help="score an arbitrary sequence")
    ap.add_argument("--consistent", action="store_true",
                    help="with --top, keep only promoters stable across carbon sources")
    ap.add_argument("--tsv", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    blob = load()
    model, cv = blob["model"], blob["cv_spearman"]
    recs = json.load(open(BENCH))
    by_gene = {r["gene"]: r for r in recs}
    ytrue = blob["train_y"]

    # --- arbitrary sequence
    if a.seq:
        s = a.seq.strip().upper()
        if len(s) < 80:
            sys.exit(f"sequence is {len(s)}bp; give at least 80bp (250bp upstream of the ATG is ideal)")
        p = float(model.predict(D.kmer_counts([s[-WINDOW:]], k=6))[0])
        pct = float((ytrue < p).mean() * 100)
        print(f"predicted strength : {p:.2f} log10 TPM")
        print(f"percentile         : {pct:.0f} of Yarrowia promoters  [{band(pct)}]")
        print(f"consistency        : unknown - cannot be predicted from sequence")
        print(f"\nmodel cross-validated Spearman {cv:.2f}. Shortlisting tool, not a measurement.")
        return

    # --- one gene
    if a.gene:
        r = by_gene.get(a.gene) or by_gene.get(a.gene.replace("_", ""))
        if not r:
            sys.exit(f"{a.gene} not in the benchmark ({len(recs):,} genes). Use --seq to score its sequence directly.")
        p = float(model.predict(D.kmer_counts([r["seq"][-WINDOW:]], k=6))[0])
        pct = float((ytrue < r["strength"]).mean() * 100)
        cb = r["carbon"]
        print(f"gene                : {r['gene']}")
        print(f"measured strength   : {r['strength']:.2f} log10 TPM   [{band(pct)}, {pct:.0f}th percentile]")
        print(f"predicted strength  : {p:.2f}")
        print(f"consistency         : {r['consistency']:.2f}  (1.0 = identical across carbon sources)")
        print(f"strain robustness   : {r['robustness']:.2f}")
        print(f"glucose / glycerol / both : {cb['Glu']:.2f} / {cb['Gly']:.2f} / {cb['GlyGlu']:.2f}")
        return

    # --- genome-wide shortlist
    # These genes are in the training set, so the fitted model's predictions for
    # them would be optimistic. Use the out-of-fold predictions instead: each gene
    # scored by a model that never saw it. That makes the enrichment figure honest.
    pred = blob["train_pred"]
    order = np.argsort(-pred)
    if a.consistent:
        med = np.median([r["consistency"] for r in recs])
        order = [i for i in order if recs[i]["consistency"] >= med]
    order = list(order)[:a.top]

    if a.tsv:
        print("gene\tpredicted\tmeasured\tconsistency\trobustness\tGlu\tGly\tGlyGlu")
        for i in order:
            r = recs[i]; cb = r["carbon"]
            print(f"{r['gene']}\t{pred[i]:.3f}\t{r['strength']:.3f}\t{r['consistency']:.3f}\t"
                  f"{r['robustness']:.3f}\t{cb['Glu']:.3f}\t{cb['Gly']:.3f}\t{cb['GlyGlu']:.3f}")
        return

    print(f"Top {len(order)} predicted Yarrowia promoters"
          + (" (filtered to consistent ones)" if a.consistent else ""))
    print(f"{'gene':18s} {'pred':>6} {'measured':>9} {'consist':>8}  Glu/Gly/GlyGlu")
    for i in order:
        r = recs[i]; cb = r["carbon"]
        print(f"{r['gene']:18s} {pred[i]:>6.2f} {r['strength']:>9.2f} {r['consistency']:>8.2f}  "
              f"{cb['Glu']:.2f}/{cb['Gly']:.2f}/{cb['GlyGlu']:.2f}")

    thr = np.quantile(ytrue, 0.9)
    hits = sum(1 for i in order if recs[i]["strength"] >= thr)
    exp = 0.1 * len(order)
    print(f"\n{hits}/{len(order)} are genuinely in the top decile "
          f"({exp:.1f} expected by chance -> {hits/exp:.1f}x enrichment)")
    print("Ranked by out-of-fold prediction, so this enrichment is honest, not in-sample.")
    print("Shortlist for lab screening. Does not assess cell-to-cell heterogeneity.")


if __name__ == "__main__":
    main()
