"""D4 (substituted): does a model trained on RANDOM promoters predict DESIGNED ones?

de Boer also measured ~75k deliberately designed and in-silico-evolved 80bp
sequences in the same pTpA scaffold and the same assay (GSE104878,
pTpA_random_design_tiling_etc). Same length, same construct, no curation needed.

This is the engineering question the project is motivated by - can a model trained
on random sequence predict sequences somebody deliberately built? It is NOT the
species-transfer question Yarrowia was meant to answer; that remains open.

Sequences already in the primary test set are removed, so this set is independent.
Results break down by de Boer's own design category, encoded in the sequence IDs.
"""
import sys, gzip, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D
from evaluate import evaluate, log_run

P5, P3 = 17, 13
N_TRAIN, SEEDS = 100_000, [0, 1, 2]


def load_designed():
    rows = []
    with gzip.open(ROOT / "data/raw/GSE104878_pTpA_random_design_tiling_etc_YPD_expression.txt.gz", "rt") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                try: rows.append((p[0], float(p[1])))
                except ValueError: pass
    ids = {}
    with gzip.open(ROOT / "data/raw/GSE104878_pTpA_random_design_tiling_etc_sequence_IDs.txt.gz", "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2: ids[p[1]] = p[0]

    # drop anything already in the primary test set, so this stays independent
    ts, _ = D.load_primary_test()
    test_ins = set(ts)
    seqs, y, cat = [], [], []
    for s, v in rows:
        ins = s[P5:-P3]
        if ins in test_ins: continue
        seqs.append(ins); y.append(v)
        cat.append(ids.get(s, "unknown").split(".")[0])
    return seqs, np.array(y, np.float32), np.array(cat)


def main():
    t0 = time.time()
    dseq, dy, dcat = load_designed()
    print(f"designed set after removing test overlap: {len(dseq):,}", flush=True)

    pool_seq, pool_y = D.load_pool()
    subs = D.subsamples("cluster")
    Xk_pool = D.kmer_counts(pool_seq)
    Xk_des = D.kmer_counts(dseq)
    Xo_des = D.onehot(dseq)

    for model in ("ridge", "cnn"):
        for seed in SEEDS:
            idx = subs[f"n{N_TRAIN}_seed{seed}"]
            t1 = time.time()
            if model == "ridge":
                from run_baselines import fit_ridge
                predict, extra = fit_ridge(Xk_pool[idx], pool_y[idx], seed)
                pred = predict(Xk_des)
            else:
                from run_baselines import fit_cnn
                Xo_tr = D.onehot([pool_seq[i] for i in idx])
                predict, extra = fit_cnn(Xo_tr, pool_y[idx], seed)
                pred = predict(Xo_des)

            met = evaluate(dy, pred)
            log_run({"model": model, "split": "cluster", "n": N_TRAIN, "seed": seed,
                     "evalset": "designed", "deliverable": "D4"}, met)
            print(f"  {model:6s} seed={seed} rho={met['overall']['spearman']:+.4f} "
                  f"r2={met['overall']['r2']:+.4f} P@100={met['precision_at_k']['p_at_100']:.2f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

            if seed == 0:   # per-category breakdown, once per model
                from scipy.stats import spearmanr
                out = {}
                for c in sorted(set(dcat)):
                    m = dcat == c
                    if m.sum() >= 50:
                        out[c] = {"n": int(m.sum()),
                                  "spearman": round(float(spearmanr(dy[m], pred[m])[0]), 4)}
                (ROOT / "results" / f"designed_by_category_{model}.json").write_text(json.dumps(out, indent=2))
                top = sorted(out.items(), key=lambda kv: -kv[1]["n"])[:8]
                print(f"      by design category: " +
                      "  ".join(f"{k}={v['spearman']:+.3f}(n={v['n']:,})" for k, v in top), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
