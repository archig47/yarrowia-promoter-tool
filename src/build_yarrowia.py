"""D4: build the Yarrowia lipolytica transfer benchmark.

82 endogenous promoters with measured strengths (Zhang et al., preprint
rs-1993869). The paper identifies promoters by gene, not by sequence, so the
sequences come from the CLIB122 reference genome (GCF_000002525.2).

ASSUMPTION, and it is a real one: we take the 80 bp immediately upstream of each
gene's annotated start. De Boer's random 80mers sit in a fixed scaffold upstream
of a TSS, so this is the closest positional analogue, but Yarrowia TSSs are not
annotated here and the true promoter may extend much further upstream. Stated in
the write-up, not buried.
"""
import gzip, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UPSTREAM = 1000        # extract long; downstream code tiles 80bp windows across it


def read_genome():
    seqs, name, buf = {}, None, []
    with gzip.open(RAW / "GCF_000002525.2_ASM252v1_genomic.fna.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name: seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name: seqs[name] = "".join(buf)
    return seqs


def read_genes():
    genes = {}
    with gzip.open(RAW / "GCF_000002525.2_ASM252v1_genomic.gff.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene": continue
            m = re.search(r"locus_tag=([^;]+)", f[8]) or re.search(r"Name=([^;]+)", f[8])
            if not m: continue
            key = m.group(1).replace("_", "")          # YALI0_B22308g -> YALI0B22308g
            genes[key] = (f[0], int(f[3]), int(f[4]), f[6])
    return genes


def revcomp(s):
    return s.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def main():
    recs = json.load(open(RAW / "yarrowia_promoters_raw.json"))
    genome, genes = read_genome(), read_genes()
    print(f"genome: {len(genome)} contigs | annotated genes: {len(genes):,}")

    out, missing = [], []
    for r in recs:
        g = genes.get(r["gene"])
        if not g:
            missing.append(r["gene"]); continue
        contig, start, end, strand = g
        chrom = genome[contig]
        if strand == "+":
            s, e = max(0, start - 1 - UPSTREAM), start - 1
            up = chrom[s:e]
        else:
            up = revcomp(chrom[end:min(len(chrom), end + UPSTREAM)])
        up = up.upper()
        # keep anything with at least a minimal Yarrowia promoter (130bp, Blazeck et al.)
        if len(up) < 130 or set(up) - set("ACGT"):
            missing.append(r["gene"]); continue
        out.append({**r, "contig": contig, "strand": strand, "seq": up.upper()})

    print(f"extracted {len(out)}/{len(recs)} promoters ({UPSTREAM}bp upstream)")
    if missing: print(f"  not resolved: {missing}")

    v = np.array([o["strength"] for o in out])
    lens = np.array([len(o["seq"]) for o in out])
    print(f"  upstream lengths: min {lens.min()} median {int(np.median(lens))} max {lens.max()}")
    prox = [o["seq"][-80:] for o in out]          # the 80bp closest to the ATG
    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in prox])
    print(f"  strength {v.min():.2e} .. {v.max():.2e} | GC {gc.mean():.3f} +/- {gc.std():.3f}")
    from scipy.stats import spearmanr
    print(f"  sanity: GC of proximal 80bp vs strength spearman {spearmanr(gc, v)[0]:+.3f}")
    print("\n  example:", out[0]["promoter"], out[0]["gene"], out[0]["seq"][-50:], "(proximal end)")

    json.dump(out, open(ROOT / "data" / "yarrowia" / "yarrowia_benchmark.json", "w"), indent=2)
    print(f"\nwrote data/yarrowia/yarrowia_benchmark.json ({len(out)} promoters)")


if __name__ == "__main__":
    main()
