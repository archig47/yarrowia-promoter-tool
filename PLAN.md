# Data efficiency of DNA language models for promoter strength prediction in yeast

**Question:** How few experimental measurements do you need — and which ones should you
choose — before sequence-based prediction becomes useful in a yeast with little
characterised data?

**Motivation:** Large promoter–expression datasets exist for *S. cerevisiae* (~10^7
labelled sequences) but not for industrially relevant non-conventional yeasts such as
*Yarrowia lipolytica*, where perhaps a few hundred promoters have ever been
characterised. This project uses the large dataset as a *simulator of data scarcity*
to ask how much labelled data is actually required, whether genomic pretraining
substitutes for missing labels, and whether choosing which sequences to measure beats
measuring at random.

---

## Checkpoint 1 — Days 1-14

### D1. Scaling curves
Accuracy vs. training set size (n = 100 -> 100,000, log x-axis) for:
k-mer ridge, LightGBM on k-mer counts, CNN from scratch (~200k params),
LoRA-adapted Nucleotide Transformer, LoRA-adapted DNABERT-2.
Three seeds per point. Error bars. Noise ceiling drawn as a horizontal line.

### D2. Selection curves
Uncertainty and diversity sampling vs. random selection at low n.
Headline framing: how many randomly chosen measurements does a well-chosen 300 replace?

### D3. Motif recovery
ISM -> TF-MoDISco -> match against yeast TF PWMs. Ground truth: de Boer Supplementary
Table 2 (motifs from YeTFaSCo, plus a poly-A motif AAAAA).

### D4. Yarrowia transfer benchmark
Curated set of characterised *Y. lipolytica* promoters from the hybrid-promoter
literature. Zero-shot transfer, then LoRA on ~50 examples.
Rank correlation only. Bootstrap CIs. Caveats loud.

---

## Definition of done — Checkpoint 1

- [ ] Every number produced by one frozen `src/evaluate.py`, unedited after Day 2
- [ ] Primary test set = high-quality pTpA/glucose set; cluster split reported alongside
- [ ] Three seeds everywhere below n = 10,000
- [ ] `PREDICTIONS.md` written before any modelling, never edited
- [ ] Baselines that could have beaten the LM, outcome stated in README first paragraph
- [ ] Limitations section: single-dataset scope, limited tuning budget, Yarrowia heterogeneity
- [ ] `git clone` -> one command -> figures regenerate
- [ ] Tagged release `v1.0`

---

## Schedule

### Days 1-2 — Data and harness. No modelling.
- de Boer et al. 2020, GEO **GSE104878**. pTpA scaffold, glucose condition.
  80 bp random inserts, label = expression level (EL), a weighted average over the 18
  FACS sorting bins, range ~1.5-16.7. (The paper's "log2(YFP/RFP)" phrasing does not
  describe the published values — corrected 2026-09-04 from the data.)
- Download three things: pTpA/glucose training data; the **high-quality pTpA/glucose
  test set (n ~ 9,982)**; Supplementary Table 2 (TF motifs). Grab galactose and
  glycerol conditions too — free external validation later at no extra effort.
- ~~**Filter by read count.**~~ **NOT POSSIBLE — de Boer published no per-sequence read
  counts.** Both distributed files carry sequence + expression level only. The
  `atLeast100Counts` file IS the read-count-filtered subset (>=100 reads) and serves as
  the test set; the training data cannot be filtered. The ~24% noise estimate below
  quantifies exactly this. Consequence: absolute data-efficiency numbers are pessimistic;
  method comparisons are unaffected. Goes in Limitations. Superseded 2026-09-04.
- **Noise ceiling is known:** authors estimate ~24% noise in training data
  (their model explains 68.3% of held-out training data vs 92.6% of high-quality data).
  Draw this on every scaling curve.
- Cluster sequences (MMseqs2 or CD-HIT); build `cluster` and `random` splits;
  freeze as fixed index files in `data/splits/`.
- Build subsample index sets for every n x 3 seeds, once, now.
- Write `src/evaluate.py`: predictions + split name -> Pearson, Spearman, R^2, plus the
  same restricted to top and bottom deciles. **Freeze it.**

> **GO/NO-GO, end of Day 2.** If the data is not parsed, filtered, split and frozen by
> end of Day 2, cut D4 (Yarrowia) immediately rather than compressing Days 3-8.
> Second go/no-go: if the high-quality test set is not available in usable form on GEO,
> fall back to the cluster split as primary and say so in STATE.md.

### Days 3-4 — Baselines and LoRA across all n
- Day 3: k-mer ridge, LightGBM, CNN. Every n, 3 seeds. Fast at low n.
- Day 4: LoRA fine-tunes. Start at lr 1e-4, alpha 16, dropout 0.05, r=8, batch 32,
  50 warmup steps, weight decay 0.01. Regression head, num_labels=1, MSE on log2 ratio.
- **End of Day 4: headline figure exists.**

### Days 5-6 — Active learning
Pool-based simulation: random vs uncertainty (ensemble variance / MC-dropout) vs
diversity (k-means on embeddings) vs hybrid. 3 seeds.
Report as: a chosen 300 performs equivalently to a random N.

### Day 7 — Interpretability
ISM over ~500 test sequences x 80 positions x 3 alternate bases. TF-MoDISco on the
attribution matrices. Match to yeast PWMs. Report non-matches too.
Use ISM, not attention maps — DNABERT-2 BPE tokens are variable-length.

### Day 8 — Consolidate. Everything to here is shippable.

### Days 9-11 — Yarrowia transfer benchmark
Curate sequence/strength pairs from the hybrid-promoter literature (Blazeck/Alper
hybrid promoters; UAS + core promoter architectures). Expect 80-300 sequences, not more.
Several papers report sequences only in supplementary tables or reference plasmids
without sequence — budget more time than seems necessary. Cite each sequence to source.
Zero-shot, then LoRA on ~50, vs a Yarrowia-only baseline. Rank correlation, bootstrap CIs.

### Days 12-13 — Write-up
4-6 pages: question, data, splits and why they matter, baselines, scaling curves,
selection curves, motif recovery, Yarrowia transfer, predictions vs outcomes, limitations.

### Day 14 — Slack. You will need it.

---

## Checkpoint 2 — Days 15-28, optional

Priority order:
**A. External validation** — de Boer galactose/glycerol conditions, native yeast
promoter fragments, DREAM 2022 promoter challenge test set (public, published scores).
**B. Proper tuning budget** — closes the obvious criticism of Checkpoint 1.
**C. Iterative active learning** — pick 100, train, pick next 100 informed, repeat.
**D. Generative arm** — regLM-style conditional HyenaDNA (Lal et al. 2024, Genome
Research), already published on this exact dataset: 80 bp prefixed with activity-label
tokens, trained from scratch, 100 epochs, 1 A100, AdamW, lr 3e-4, batch 2048, context 84.
Two-oracle protocol: oracle A filters, oracle B (disjoint data) evaluates.
**E. Design output** — ~50 designed Yarrowia candidates at target strengths, orderable.
**F. Calibration** — do confidence estimates mean anything?

Done means: all of Checkpoint 1 still passing, plus A, B, E minimum.

---

## Rules

1. **Checkpoint 1 does not leak.** If Day 14 arrives unfinished, cut D4. Never cut
   seeds, baselines, or the write-up.
2. **Tag `v1.0` before extending.**
3. **Send at the boundary regardless** — email Twig, send Rodrigo the v1.0 link.
4. **If Checkpoint 1 finishes early,** start with Checkpoint 2 item A, not everything.

---

## Key references

- de Boer et al. (2020). Deciphering eukaryotic gene-regulatory logic with 100 million
  random promoters. *Nat Biotechnol* 38(1):56-65. GEO: GSE104878.
- Vaishnav et al. (2022). The evolution, evolvability and engineering of gene
  regulatory DNA. *Nature* 603:455-463.
- Lal, Garfield, Biancalani & Eraslan (2024). regLM. *Genome Research* 34:1411.
- Zhou et al. (2024). DNABERT-2. ICLR 2024.
- Dalla-Torre et al. (2023). The Nucleotide Transformer.
- Hu et al. (2022). LoRA. ICLR 2022.
- Blazeck et al. (2011, 2013). Hybrid promoter approaches in *Yarrowia lipolytica*.
