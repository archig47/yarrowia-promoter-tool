"""The missing control: S. cerevisiae native promoters with an RNA-seq label.

The Yarrowia test changes FOUR things at once relative to de Boer training data:
species, sequence type, assay, and chromatin context. The Native80 rung of the
ladder controls sequence type (cost: 0.09). This benchmark controls the other two.

Same species as training, but native promoters in native chromatin with an
RNA-seq label - exactly matching the Yarrowia benchmark's assay and context.
So:
    de Boer model -> this            = assay + context change only
    de Boer model -> Yarrowia        = assay + context + SPECIES
and the difference isolates the species contribution.

Expression: GSE316459, BY4741 wild-type, DESeq2-normalised counts, 3 replicates.
Sequences: 1000bp upstream of each gene from R64 (GCF_000146045.2).
"""
import csv, gzip, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UPSTREAM = 1000
WT_COLS = ["BY4741_1_DESeq2_normc", "BY4741_2_DESeq2_normc", "BY4741_3_DESeq2_normc"]


def read_fasta_gz(p):
    seqs, name, buf = {}, None, []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name: seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else: buf.append(line.strip())
    if name: seqs[name] = "".join(buf)
    return seqs


def read_genes_gz(p):
    genes = {}
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line.startswith("#"): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene": continue
            m = re.search(r"locus_tag=([^;]+)", f[8])
            if m: genes[m.group(1)] = (f[0], int(f[3]), int(f[4]), f[6])
    return genes


def revcomp(s):
    return s.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def main():
    rows = []
    with gzip.open(RAW / "GSE316459_counts_and_normalized_counts_per_gene.txt.gz", "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
    missing = [c for c in WT_COLS if c not in rows[0]]
    if missing: raise SystemExit(f"columns not found: {missing}")
    print(f"{len(rows):,} genes | wild-type columns: {WT_COLS}")

    expr = {}
    for r in rows:
        gid = (r.get("Gene_stable_ID") or "").strip()
        if not gid.startswith("Y"): continue
        try: vals = [float(r[c]) for c in WT_COLS]
        except (TypeError, ValueError): continue
        expr[gid] = float(np.mean(vals))
    print(f"genes with a systematic id and wild-type values: {len(expr):,}")

    genome = read_fasta_gz(RAW / "GCF_000146045.2_R64_genomic.fna.gz")
    genes = read_genes_gz(RAW / "GCF_000146045.2_R64_genomic.gff.gz")
    print(f"genome: {len(genome)} contigs | annotated genes: {len(genes):,}")

    out = []
    for gid, e in expr.items():
        g = genes.get(gid)
        if not g: continue
        contig, start, end, strand = g
        chrom = genome[contig]
        up = (chrom[max(0, start - 1 - UPSTREAM):start - 1] if strand == "+"
              else revcomp(chrom[end:min(len(chrom), end + UPSTREAM)])).upper()
        if len(up) < 250 or set(up) - set("ACGT"): continue
        out.append({"gene": gid, "expr": e, "strand": strand, "seq": up})

    v = np.array([o["expr"] for o in out])
    keep = [o for o, x in zip(out, v) if x >= 1.0]
    print(f"\nbenchmark: {len(out):,} promoters, {len(keep):,} with normalised count >= 1")
    vv = np.array([o["expr"] for o in keep])
    print(f"  expression: min {vv.min():.1f} median {np.median(vv):.0f} max {vv.max():.0f}")

    json.dump(keep, open(ROOT / "data" / "processed" / "scer_rnaseq_benchmark.json", "w"))
    print("wrote data/processed/scer_rnaseq_benchmark.json")


if __name__ == "__main__":
    main()
