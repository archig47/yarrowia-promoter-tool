"""Which sequence features mean the same thing in both species?

GC content does not: rho +0.604 in de Boer's S. cerevisiae library, -0.032 in
Yarrowia. That is why transfer fails. So the question is whether anything DOES
carry over - and TF motifs are the candidate, because they are mechanistic
(a protein binds a site) rather than statistical.

For each of de Boer's 245 motifs, score its occurrence in both species' sequences
and correlate with expression separately in each. A motif with the same sign and
comparable magnitude in both is a transferable feature; one that flips sign is a
species-specific rule; one near zero in Yarrowia is another GC-like dead end.
"""
import sys, json, pathlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D
from run_modisco import load_yetfasco

MIN_IC = 4.0          # skip near-uniform matrices; they score everything
TOP_FRAC = 0.9        # a "hit" is a window scoring above this quantile of the PWM's range


def pwm_max_score(seqs, ppm):
    """Best log-odds score of the PWM anywhere in each sequence, both strands."""
    lp = np.log2(np.clip(ppm, 1e-4, 1) / 0.25)
    rc = lp[::-1, ::-1]
    L = ppm.shape[0]
    idx = {b: i for i, b in enumerate("ACGT")}
    out = np.empty(len(seqs), dtype=np.float32)
    for n, s in enumerate(seqs):
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        code = np.full(a.size, -1, np.int8)
        for b, i in idx.items(): code[a == ord(b)] = i
        best = -1e9
        if code.size >= L:
            w = np.lib.stride_tricks.sliding_window_view(code, L)
            ok = (w >= 0).all(1)
            if ok.any():
                ww = w[ok]
                rows = np.arange(L)
                f = lp[rows, ww].sum(1)
                r = rc[rows, ww].sum(1)
                best = float(max(f.max(), r.max()))
        out[n] = best
    return out


def main():
    pwms, names = load_yetfasco()
    ic = {m: float((np.clip(p, 1e-6, 1) * np.log2(np.clip(p, 1e-6, 1) / 0.25)).sum())
          for m, p in pwms.items()}
    use = [m for m in pwms if ic[m] >= MIN_IC]
    print(f"{len(pwms)} motifs, {len(use)} with IC >= {MIN_IC}", flush=True)

    # S. cerevisiae: the high-quality test set (clean labels, 80bp inserts)
    sseq, sy = D.load_primary_test()
    # Yarrowia: the RNA-seq benchmark, 250bp window (empirically best there)
    yrec = json.load(open(ROOT / "data/yarrowia/yarrowia_rnaseq_benchmark.json"))
    yseq = [r["seq"][-250:] for r in yrec]
    yy = np.log10(np.array([r["tpm"] for r in yrec], float) + 1.0)
    print(f"S. cerevisiae n={len(sy):,} | Yarrowia n={len(yy):,}\n", flush=True)

    rows = []
    for i, m in enumerate(use):
        p = pwms[m]
        rs = spearmanr(pwm_max_score(sseq, p), sy)[0]
        ry = spearmanr(pwm_max_score(yseq, p), yy)[0]
        rows.append({"motif": m, "name": names.get(m, m), "ic": round(ic[m], 2),
                     "rho_scer": round(float(rs), 4), "rho_yarrowia": round(float(ry), 4)})
        if (i + 1) % 40 == 0: print(f"  scored {i+1}/{len(use)}", flush=True)

    json.dump(rows, open(ROOT / "results" / "motif_transfer.json", "w"), indent=2)
    a = np.array([r["rho_scer"] for r in rows])
    b = np.array([r["rho_yarrowia"] for r in rows])

    print(f"\n=== do motif effects agree between species? ===")
    print(f"  correlation of the two effect vectors: spearman {spearmanr(a, b)[0]:+.4f}")
    print(f"  motifs with the same sign: {int((np.sign(a) == np.sign(b)).sum())}/{len(a)}")
    strong_s = np.abs(a) > 0.15
    print(f"  of the {int(strong_s.sum())} with |rho|>0.15 in S. cerevisiae, "
          f"{int((np.sign(a[strong_s])==np.sign(b[strong_s])).sum())} keep their sign in Yarrowia")

    order = np.argsort(-np.minimum(np.abs(a), np.abs(b)) * (np.sign(a) == np.sign(b)))
    print("\n  TRANSFERABLE - consistent sign, largest in both:")
    print(f"    {'motif':16s} {'S.cer':>8} {'Yarrowia':>9}")
    shown = 0
    for i in order:
        if np.sign(a[i]) == np.sign(b[i]) and abs(b[i]) > 0.05:
            print(f"    {rows[i]['name'][:16]:16s} {a[i]:>+8.3f} {b[i]:>+9.3f}")
            shown += 1
            if shown >= 10: break
    if not shown: print("    (none)")

    flip = np.argsort(-(np.abs(a) * (np.sign(a) != np.sign(b))))
    print("\n  SIGN-FLIPPED - strong in S. cerevisiae, opposite in Yarrowia:")
    for i in flip[:6]:
        if np.sign(a[i]) != np.sign(b[i]):
            print(f"    {rows[i]['name'][:16]:16s} {a[i]:>+8.3f} {b[i]:>+9.3f}")


if __name__ == "__main__":
    main()
