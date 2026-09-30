"""Rebuild the Yarrowia benchmark with condition information, not just glucose.

The glucose-only label cannot distinguish two very different promoters:
  - constitutively strong  : high in every carbon source. What you usually want.
  - conditionally high     : high in glucose because glucose induces it.
Recommending them interchangeably would send someone into a glycerol fermentation
with a glucose-specific promoter.

Lubuta et al. measured 2 strains (IBT, W29) x 3 carbon sources (glucose, glycerol,
glucose+glycerol). That gives three axes per gene:
  strength     mean log expression across all conditions
  consistency  how little it varies across carbon sources (1 = constitutive)
  robustness   agreement between the two strains
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UPSTREAM = 1000

GROUPS = {
    "IBT_Glu":    ["IBT_Glu_1", "IBT_Glu_2", "IBT_Glu_3"],
    "IBT_Gly":    ["IBT_Gly_1", "IBT_Gly_2", "IBT_Gly_3"],
    "IBT_GlyGlu": ["IBT_GlyGlu_2", "IBT_GlyGlu_3"],
    "W29_Glu":    ["W29_Glu_1", "W29_Glu_2"],
    "W29_Gly":    ["W29_Gly_1", "W29_Gly_2", "W29_Gly_3"],
    "W29_GlyGlu": ["W29_GlyGlu_1", "W29_GlyGlu_2", "W29_GlyGlu_3"],
}
CARBON = {"Glu": ["IBT_Glu", "W29_Glu"], "Gly": ["IBT_Gly", "W29_Gly"],
          "GlyGlu": ["IBT_GlyGlu", "W29_GlyGlu"]}


def main():
    rows = list(csv.DictReader(open(RAW / "yarrowia_TPM_Lubuta2019.csv")))
    print(f"{len(rows):,} genes in the TPM table")

    per_gene = {}
    for r in rows:
        gid = (r.get("YALI0_ID") or "").strip()
        if not gid or gid == "NA":
            continue
        cond = {}
        for g, cols in GROUPS.items():
            vals = []
            for c in cols:
                try: vals.append(float(r[c]))
                except (TypeError, ValueError, KeyError): pass
            if vals: cond[g] = float(np.mean(vals))
        if len(cond) == len(GROUPS):
            per_gene[gid.replace("_", "")] = cond
    print(f"genes with all {len(GROUPS)} condition groups: {len(per_gene):,}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("by", ROOT / "src" / "build_yarrowia.py")
    by = importlib.util.module_from_spec(spec); spec.loader.exec_module(by)
    genome, genes = by.read_genome(), by.read_genes()

    out = []
    for gid, cond in per_gene.items():
        g = genes.get(gid)
        if not g: continue
        contig, start, end, strand = g
        chrom = genome[contig]
        up = (chrom[max(0, start - 1 - UPSTREAM):start - 1] if strand == "+"
              else by.revcomp(chrom[end:min(len(chrom), end + UPSTREAM)])).upper()
        if len(up) < 250 or set(up) - set("ACGT"): continue

        L = {k: np.log10(v + 1.0) for k, v in cond.items()}
        carbon_means = {c: float(np.mean([L[g_] for g_ in gs])) for c, gs in CARBON.items()}
        strength = float(np.mean(list(L.values())))
        # consistency: 1 when identical across carbon sources, falls toward 0 as they diverge.
        # scaled by 1 log unit (10-fold), which is the difference a user would care about.
        spread = float(np.std(list(carbon_means.values())))
        consistency = float(np.exp(-spread / 0.5))
        ibt = np.array([L["IBT_Glu"], L["IBT_Gly"], L["IBT_GlyGlu"]])
        w29 = np.array([L["W29_Glu"], L["W29_Gly"], L["W29_GlyGlu"]])
        robustness = float(np.exp(-np.abs(ibt - w29).mean() / 0.5))

        out.append({"gene": gid, "seq": up, "strand": strand,
                    "strength": round(strength, 4),
                    "consistency": round(consistency, 4),
                    "robustness": round(robustness, 4),
                    "carbon": {c: round(v, 4) for c, v in carbon_means.items()},
                    "conditions": {k: round(v, 4) for k, v in L.items()}})

    keep = [o for o in out if o["strength"] >= np.log10(2.0)]
    print(f"\n{len(out):,} promoters extracted, {len(keep):,} with mean expression >= 1 TPM")

    s = np.array([o["strength"] for o in keep])
    c = np.array([o["consistency"] for o in keep])
    r_ = np.array([o["robustness"] for o in keep])
    print(f"  strength    {s.min():.2f} .. {s.max():.2f}  (log10 TPM)")
    print(f"  consistency {c.min():.2f} .. {c.max():.2f}  (1 = same in all carbon sources)")
    print(f"  robustness  {r_.min():.2f} .. {r_.max():.2f}  (1 = same in both strains)")

    from scipy.stats import spearmanr
    print(f"\n  are strong promoters also consistent? spearman {spearmanr(s, c)[0]:+.3f}")
    strong = s >= np.quantile(s, 0.9)
    print(f"  of the top 10% by strength, {100*(c[strong] >= np.median(c)).mean():.0f}% are above-median consistency")

    # the practical question: how many strong promoters are glucose-specific?
    glu = np.array([o["carbon"]["Glu"] for o in keep])
    gly = np.array([o["carbon"]["Gly"] for o in keep])
    gap = glu - gly
    n_specific = int(((gap > 0.5) & strong).sum())
    print(f"\n  WARNING CASE: {n_specific} of the top-10%-by-strength promoters are >3x")
    print(f"  higher in glucose than glycerol - they would underperform on glycerol.")

    json.dump(keep, open(ROOT / "data" / "yarrowia" / "yarrowia_multicondition.json", "w"))
    print(f"\nwrote data/yarrowia/yarrowia_multicondition.json ({len(keep):,} promoters)")


if __name__ == "__main__":
    main()
