"""D3 steps 2-3: cluster ISM maps into motifs, then match them to known yeast TFs.

TF-MoDISco groups recurring high-importance windows ("seqlets") into patterns.
Each pattern is then compared against the 245 motifs de Boer used (Supp. Table 2),
whose matrices come from YeTFaSCo.

Two details that silently break this if missed:
  - YeTFaSCo PFM rows are ordered A, T, G, C. Our one-hot is A, C, G, T.
  - Sequences are padded to 95 but inserts are ~80; padding is trimmed off first,
    otherwise the all-zero tail invents patterns.

PLAN.md asks for non-matches to be reported too - a recovered pattern matching
nothing known is either novel signal or an artefact, and both are worth stating.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
PFM_DIR = ROOT / "data" / "raw" / "yetfasco" / "1.02" / "ALIGNED_ENOLOGO_FORMAT_PFMS"
TRIM_L = 80
MIN_OVERLAP = 6


def load_yetfasco():
    """Return {motif_id: PPM (L,4) in ACGT order}, restricted to de Boer's list."""
    import pandas as pd
    d = pd.read_excel(ROOT / "data/raw/deBoer2020_SuppTable2_motifs.xlsx",
                      sheet_name="Motifs used", header=2)
    d = d.rename(columns={d.columns[1]: "MotifID", d.columns[2]: "MotifName"}).dropna(subset=["MotifID"])
    names = dict(zip(d.MotifID.astype(str), d.MotifName.astype(str)))

    out = {}
    for mid in names:
        f = PFM_DIR / f"{mid}.pfm"
        if not f.exists():
            continue
        rows = {}
        for line in open(f):
            p = line.split()
            if len(p) > 1:
                rows[p[0].upper()] = np.array([float(x) for x in p[1:]], dtype=np.float32)
        if not {"A", "C", "G", "T"} <= set(rows):
            continue
        ppm = np.stack([rows["A"], rows["C"], rows["G"], rows["T"]], axis=1)  # -> ACGT
        s = ppm.sum(axis=1, keepdims=True); s[s == 0] = 1
        out[mid] = ppm / s

    # CLAUDE.md: ground truth is Supp. Table 2 PLUS a poly-A motif. It is not in
    # YeTFaSCo, so it is constructed here or it can never be matched.
    polyA = np.tile(np.array([[0.85, 0.05, 0.05, 0.05]], dtype=np.float32), (6, 1))
    out["polyA_AAAAAA"] = polyA
    names["polyA_AAAAAA"] = "poly-A"
    out["polyT_TTTTTT"] = polyA[:, ::-1].copy()
    names["polyT_TTTTTT"] = "poly-T"
    return out, names


def best_match(query, target, min_overlap=MIN_OVERLAP):
    """Max correlation over offsets and both strands. query/target are (L,4) PPMs."""
    best = -1.0
    for t in (target, target[::-1, ::-1]):                 # forward and reverse complement
        lq, lt = len(query), len(t)
        for off in range(-lt + min_overlap, lq - min_overlap + 1):
            qs, ts = max(0, off), max(0, -off)
            n = min(lq - qs, lt - ts)
            if n < min_overlap:
                continue
            a = query[qs:qs + n].ravel(); b = t[ts:ts + n].ravel()
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best:
                best = c
    return best


def consensus(ppm):
    """IUPAC-ish consensus: uppercase where the position is informative."""
    bases = "ACGT"
    out = []
    for row in ppm:
        i = int(np.argmax(row))
        ic = float((np.clip(row, 1e-6, 1) * np.log2(np.clip(row, 1e-6, 1) / 0.25)).sum())
        out.append(bases[i] if ic > 0.5 else bases[i].lower())
    return "".join(out)


def info_content(ppm):
    p = np.clip(ppm, 1e-6, 1)
    return float((p * np.log2(p / 0.25)).sum())


def main(model, threshold, reuse=False):
    import modiscolite
    h5 = ROOT / "results" / f"modisco_{model}.h5"
    d = np.load(ROOT / "data/processed" / f"ism_{model}.npz", allow_pickle=True)
    seqs = list(d["seqs"])
    keep = [i for i, s in enumerate(seqs) if len(s) >= TRIM_L]
    oh = d["onehot"][keep][:, :TRIM_L, :].astype(np.float32)
    hyp = d["hyp"][keep][:, :TRIM_L, :].astype(np.float32)
    print(f"{model}: {len(keep)}/{len(seqs)} sequences >= {TRIM_L}bp -> {oh.shape}", flush=True)

    if h5.exists() and reuse:
        print(f"  reusing existing {h5.name}", flush=True)
    else:
        pos, neg = modiscolite.tfmodisco.TFMoDISco(
            one_hot=oh, hypothetical_contribs=hyp,
            sliding_window_size=10, flank_size=3, trim_to_window_size=15,
            initial_flank_to_add=3, min_metacluster_size=50, final_min_cluster_size=15,
            target_seqlet_fdr=0.2, n_leiden_runs=5, max_seqlets_per_metacluster=20000,
            verbose=False)
        modiscolite.io.save_hdf5(str(h5), pos, neg, window_size=10)
        print(f"  saved {h5.name}", flush=True)

    pwms, names = load_yetfasco()
    print(f"  matching against {len(pwms)} de Boer / YeTFaSCo motifs", flush=True)

    report = []
    with h5py.File(h5, "r") as f:
        for group in ("pos_patterns", "neg_patterns"):
            if group not in f:
                continue
            for pname in f[group]:
                ppm = np.array(f[group][pname]["sequence"])
                sl = f[group][pname]["seqlets"]
                key = "start" if "start" in sl else list(sl.keys())[0]
                n_seqlets = int(np.array(sl[key]).shape[0])
                scored = sorted(((best_match(ppm, p), mid) for mid, p in pwms.items()), reverse=True)
                top = [(names[m], m, round(c, 3)) for c, m in scored[:3]]
                report.append({"group": group, "pattern": pname, "n_seqlets": n_seqlets,
                               "ic": round(info_content(ppm), 2), "width": int(ppm.shape[0]),
                               "consensus": consensus(ppm),
                               "best_corr": round(scored[0][0], 3),
                               "matched": bool(scored[0][0] >= threshold), "top3": top})

    report.sort(key=lambda r: -r["n_seqlets"])
    (ROOT / "results" / f"modisco_matches_{model}.json").write_text(json.dumps(report, indent=2))

    matched = sum(r["matched"] for r in report)
    print(f"\n{len(report)} patterns; {matched} match a known motif at r >= {threshold}\n")
    print(f"  {'pattern':20s} {'seqlets':>7} {'IC':>6}  {'consensus':<24s} {'r':>6}  top 3 known matches")
    for r in report:
        flag = "" if r["matched"] else "  [NO MATCH]"
        top = ", ".join(f"{n} {c:.2f}" for n, _, c in r["top3"])
        print(f"  {r['group'][:3]}/{r['pattern']:<15s} {r['n_seqlets']:>7} {r['ic']:>6.2f}  "
              f"{r['consensus']:<24s} {r['best_corr']:>6.3f}  {top}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--reuse", action="store_true", help="reuse an existing modisco h5")
    a = ap.parse_args()
    main(a.model, a.threshold, a.reuse)
