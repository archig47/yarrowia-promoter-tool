"""A larger Yarrowia benchmark: genome-wide RNA-seq expression as a promoter proxy.

The 81-promoter reporter benchmark is not learnable in-domain (CV rho ~ 0), which
makes transfer unevaluable. Lubuta et al. 2019 (G3; FigShare 10.25387/g3.8335217)
give TPM for 8,605 genes in glucose chemostats, two strains, three replicates.

IMPORTANT CAVEAT, stated up front: mRNA abundance is NOT promoter strength. It is
confounded by transcript stability, and these are promoters in their native
chromatin context rather than a fixed reporter scaffold. It is a proxy - but it is
the standard proxy, it is how the Zhang et al. paper chose its 82 candidates, and
it is ~100x more data. Both benchmarks are kept and reported.
"""
import csv, json, gzip, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UPSTREAM = 1000
GLU_COLS_PREFIX = ("IBT_Glu", "W29_Glu")     # glucose only, to match de Boer


def main():
    rows = list(csv.DictReader(open(RAW / "yarrowia_TPM_Lubuta2019.csv")))
    cols = [c for c in rows[0] if c.startswith(GLU_COLS_PREFIX)]
    print(f"{len(rows):,} genes | glucose columns: {cols}")

    expr = {}
    for r in rows:
        gid = (r.get("YALI0_ID") or "").strip()
        if not gid or gid == "NA":
            continue
        vals = []
        for c in cols:
            try: vals.append(float(r[c]))
            except (TypeError, ValueError): pass
        if len(vals) >= 3:
            expr[gid.replace("_", "")] = float(np.mean(vals))
    print(f"genes with a YALI0 id and >=3 glucose replicates: {len(expr):,}")

    # reuse the genome readers from the reporter-benchmark builder
    import importlib.util
    spec = importlib.util.spec_from_file_location("by", ROOT / "src" / "build_yarrowia.py")
    by = importlib.util.module_from_spec(spec); spec.loader.exec_module(by)
    genome, genes = by.read_genome(), by.read_genes()

    out = []
    for gid, tpm in expr.items():
        g = genes.get(gid)
        if not g: continue
        contig, start, end, strand = g
        chrom = genome[contig]
        up = (chrom[max(0, start - 1 - UPSTREAM):start - 1] if strand == "+"
              else by.revcomp(chrom[end:min(len(chrom), end + UPSTREAM)])).upper()
        if len(up) < 200 or set(up) - set("ACGT"):
            continue
        out.append({"gene": gid, "tpm": tpm, "strand": strand, "seq": up})

    tpm = np.array([o["tpm"] for o in out])
    print(f"\nbenchmark: {len(out):,} promoters with {UPSTREAM}bp upstream")
    print(f"  TPM: min {tpm.min():.2f} median {np.median(tpm):.1f} max {tpm.max():.0f}")
    keep = [o for o, t in zip(out, tpm) if t >= 1.0]      # drop unexpressed genes
    print(f"  with TPM >= 1: {len(keep):,}")

    json.dump(keep, open(ROOT / "data" / "yarrowia" / "yarrowia_rnaseq_benchmark.json", "w"))
    print("wrote data/yarrowia/yarrowia_rnaseq_benchmark.json")

    # how many of the 81 reporter-measured promoters are also here, and do the two agree?
    rep = json.load(open(ROOT / "data/yarrowia/yarrowia_benchmark.json"))
    m = {o["gene"]: o["tpm"] for o in keep}
    pairs = [(r["strength"], m[r["gene"]]) for r in rep if r["gene"] in m]
    if pairs:
        from scipy.stats import spearmanr
        a, b = zip(*pairs)
        print(f"\ncross-check: {len(pairs)} promoters have BOTH a reporter measurement and TPM")
        print(f"  reporter strength vs TPM: spearman {spearmanr(a, b)[0]:+.3f}")
        print("  (a strong positive would validate TPM as a proxy; near zero means they measure different things)")


if __name__ == "__main__":
    main()
