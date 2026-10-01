---
license: mit
language:
  - en
tags:
  - biology
  - genomics
  - promoter
  - yeast
  - synthetic-biology
  - regulatory-genomics
pretty_name: Yarrowia lipolytica Promoter Strength
size_categories:
  - 1K<n<10K
task_categories:
  - tabular-regression
---

# Yarrowia lipolytica promoter strength

Sequence-and-expression pairs for promoters of *Yarrowia lipolytica*, an oleaginous yeast
used industrially for lipids, organic acids and heterologous protein production.

**Why this exists.** *Y. lipolytica* is widely used but barely characterised: perhaps a
hundred of its promoters have ever had their strength measured, against ~10⁸ for
*Saccharomyces cerevisiae*. That gap blocks predictive expression engineering. Neither of
these two tables existed in machine-readable form before; both were assembled from
published sources and the reference genome.

## Files

### `yarrowia_promoters.csv` — 6,026 promoters

Expression measured by RNA-seq across two strains and three carbon sources
(Lubuta et al. 2019, chemostat, TPM).

| column | |
|---|---|
| `gene` | YALI0 systematic identifier |
| `strand` | gene orientation |
| `upstream_1000bp` | 1000 bp immediately 5′ of the start codon, from CLIB122 (GCF_000002525.2) |
| `strength_log10tpm` | log10(TPM + 1), averaged over all six condition groups |
| `consistency` | 1.0 = identical across carbon sources; lower = condition-dependent |
| `strain_robustness` | 1.0 = identical in strains IBT and W29 |
| `glucose`, `glycerol`, `glucose_glycerol` | per-carbon-source means, log10(TPM + 1) |

### `yarrowia_reporter_measured.csv` — 81 promoters

Strengths measured directly with a fluorescent reporter (Zhang et al., Research Square
rs-1993869). Smaller but a more direct measurement of promoter activity. Extraction was
validated against the source paper's own Strong/Medium/Weak labels (Spearman +0.91).

## Known results on this data

From the [accompanying study](https://github.com/archig47/yeast-promoter-lm):

- **k-mer ridge regression on the 250 bp proximal window reaches Spearman 0.40**
  (5-fold cross-validated) on `strength_log10tpm`. Ranking its top 25 candidates, 16 are
  genuinely in the top decile against 2.5 expected by chance — a 6.4× enrichment.
- **250 bp outperforms both 80 bp (0.33) and 1000 bp (0.35)**, consistent with
  Blazeck et al. placing a minimal *Y. lipolytica* promoter at 130–260 bp upstream of the ATG.
- **`consistency` is not predictable from sequence** (Spearman 0.10). Sequence encodes how
  strong a promoter is, not how condition-responsive it is.
- **The 81-promoter reporter set is too small to learn from** — cross-validated Spearman ≈ 0,
  not significant against permutation nulls. It is included for evaluation and for anyone
  assembling a larger reporter-measured set, not as a training set.

## Important limitations

- **`strength_log10tpm` is mRNA abundance, not promoter strength.** It is confounded by
  transcript stability and reflects promoters in native chromatin rather than a fixed
  reporter construct. On the 80 promoters present in both tables, RNA-seq and
  reporter-measured values correlate at Spearman **0.43** — related, not interchangeable.
- **The 1000 bp window is genomic upstream sequence**, not necessarily the region any
  published study cloned.
- **Cell-to-cell heterogeneity is not captured.** Patel et al. (2026) show that a strong but
  macro-heterogeneous promoter can cost threefold in product yield; nothing here assesses it.
- Conditions are limited to three carbon sources in steady-state chemostats.

## Sources

- Lubuta P, Workman M, Kerkhoven EJ, Workman CT (2019). Investigating the influence of
  glycerol on the utilization of glucose in *Yarrowia lipolytica* using RNA-Seq-based
  transcriptomics. *G3* 9(12):4059–4071.
- Zhang et al. Characterization of endogenous promoters in *Yarrowia lipolytica*.
  Research Square rs-1993869.
- Genome: *Y. lipolytica* CLIB122, GCF_000002525.2.
- Blazeck J, Liu L, Redden H, Alper H (2011). Tuning gene expression in *Yarrowia lipolytica*
  by a hybrid promoter approach. *Appl Environ Microbiol* 77(22):7905–7914.

## Citation

Assembled by Archita Gupta, 2026. MIT licensed. If you use it, please also cite the
underlying studies above.
