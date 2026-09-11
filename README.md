# yeast-promoter-lm

**Headline result: genomic pretraining substitutes for experimental measurements, but only
in the data-scarce regime, and only for one of the two language models tested.**

A LoRA-adapted DNABERT-2 reaches Spearman 0.70 from 300 measured promoters — the same
accuracy a from-scratch CNN needs 700 for, and a k-mer ridge regression needs 1,390 for.
That is roughly a 2–4× saving in wet-lab work, which is the quantity this project set out
to measure.

**The simple baselines were competitive and sometimes won, and that is reported here first
because it could easily have gone the other way.** A 156k-parameter CNN trained from
scratch beats DNABERT-2 at n = 10,000 and 30,000, and beats the *second* language model
(Nucleotide Transformer, 500M) at almost every size. NT is four times larger than
DNABERT-2 and loses to it everywhere, falling below plain k-mer ridge at n = 300. So the
defensible claim is narrower than "pretraining helps": **pretraining can substitute for
measurements, and which pretrained model you use matters more than how big it is.**

A second finding is negative and equally practical: **choosing which promoters to measure
does not beat choosing at random.** Uncertainty sampling was consistently *worse* — a
carefully chosen 300 was worth about 219 randomly chosen ones.

---

## The question

Large promoter–expression datasets exist for *S. cerevisiae* (~10⁸ measured sequences) but
not for industrially useful non-conventional yeasts such as *Yarrowia lipolytica*, where
perhaps a few hundred promoters have ever been characterised. This project uses the large
dataset as a **simulator of data scarcity**: how few labelled measurements are needed
before sequence-based prediction becomes useful, and does pretraining on genomes
substitute for the measurements you cannot afford?

Data: de Boer et al. 2020, [GSE104878](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104878)
— 31.3M random 80 bp promoters measured in living yeast by FACS sorting into 18 bins.

## Results

### Accuracy against training set size (D1)

Spearman ρ on a held-out 9,982-sequence high-quality test set, mean of 3 seeds.
Measured noise ceiling for this assay is **0.98** (agreement between independent replicates).

| measurements | k-mer ridge | LightGBM | CNN | DNABERT-2 | NT-500M |
|---|---|---|---|---|---|
| 100 | 0.422 | 0.066 | 0.521 | **0.563** | 0.471 |
| 300 | 0.563 | 0.327 | 0.639 | **0.702** | 0.540 |
| 1,000 | 0.679 | 0.406 | 0.728 | **0.771** | 0.729 |
| 3,000 | 0.756 | 0.692 | 0.794 | **0.801** | 0.788 |
| 10,000 | 0.806 | 0.784 | **0.838** | 0.831 | 0.828 |
| 30,000 | 0.832 | 0.824 | **0.868** | 0.868 | 0.859 |
| 100,000 | 0.865 | 0.839 | 0.897 | **0.902** | 0.888 |

![scaling curves](figures/fig1_scaling_curves.png)

**In measurement units:** DNABERT-2 trained on 300 matches a CNN on 700, NT on 842, or
ridge on 1,390.

### Reliability, and the transfer ladder (D1 / D4a)

At n = 100 the run-to-run spread is large enough to matter — the CNN varies by 0.22
Spearman across three random seeds with identical data. Single-seed results at small n are
not results. Reliability stabilises by n = 1,000.

Evaluated on **real evolved yeast promoters** (Native80) against **random controls measured
in the same experiment**, every model loses 0.089–0.105 Spearman. That cost is a property
of the task, not of any one architecture, and it sets expectations for transfer to real
promoters in another species.

![reliability and transfer](figures/fig2_reliability.png)

### Choosing what to measure (D2)

| budget | random | uncertainty | diversity | hybrid |
|---|---|---|---|---|
| 300 | **0.557** | 0.517 | 0.551 | 0.538 |
| 1,000 | **0.676** | 0.639 | 0.667 | 0.656 |
| 3,000 | 0.748 | 0.735 | **0.751** | 0.732 |

No strategy beats random. Uncertainty sampling selects sequences with mean label 10.30
against a pool mean of 9.51 — a skewed training sample that generalises worse. Diversity
sampling has nothing to exploit: MMseqs2 finds 99.84% singletons, so a random library has
no under-sampled regions to target.

![selection curves](figures/fig3_selection_curves.png)

## Reproduce

```
git clone <repo> && cd yeast-promoter-lm
make all          # creates the venv, installs deps, regenerates every figure
```

`make all` works from a clean clone with no GPU and no data download: the 840 logged runs
are committed in `results/`, and every figure is rebuilt from them.

Longer targets rebuild the results themselves:

```
make data         # download GSE104878, parse, freeze splits  (~1 GB, ~15 min)
make baselines    # ridge, LightGBM, CNN                       (~1 h, CPU)
make lora         # DNABERT-2 and NT LoRA sweeps               (~9 h, CUDA GPU)
make active       # D2 selection curves                        (~2 min, CPU)
```

## What is in here

| path | |
|---|---|
| `PLAN.md` | the fixed specification. Scope is not changed against it silently |
| `STATE.md` | living status board and append-only decision log |
| `PREDICTIONS.md` | written blind, before any data was downloaded. Never edited |
| `src/evaluate.py` | **frozen** after Day 2. Every reported number comes from here |
| `data/splits/` | **frozen** index files. Never regenerated |
| `results/runs.csv` | one row per run, 840 runs |

## Limitations

**Measurement noise cannot be filtered.** `PLAN.md` called for filtering training sequences
by read count, but de Boer published no per-sequence read counts. Training labels carry the
authors' estimated ~24% noise. Method comparisons are unaffected — every model sees the same
data — but the absolute measurement counts reported here are **pessimistic**: a lab making
cleaner measurements would likely need fewer than these numbers suggest.

**Test-set overlap had to be removed.** The high-quality test library is a dilution of the
same pool as the training library, not an independent synthesis, so 6,651 of 9,982 test
sequences (66.6%) also appeared in training. Those rows were excluded from all training
subsamples. Uncaught, this would have inflated every curve silently.

**Limited tuning budget.** Hyperparameters were fixed a priori, not searched. LightGBM's
collapse at n = 100 (ρ = 0.066) is substantially an artefact of early stopping judged on a
20-sequence validation set, and part of NT's underperformance may be the same. The honest
claim is "with this tuning budget", not "this model cannot do better".

**One deviation from the specified training protocol.** `PLAN.md` specifies 50 warmup steps.
At n = 100 that is ~17 epochs, and two of three seeds stopped mid-warmup having never
trained at the intended learning rate. Warmup was changed to `min(50, 10% of planned steps)`
with a guard against stopping before it completes. This affects n ≤ 300 only; every run at
n ≥ 1,000 uses the specified 50 steps unchanged. n = 100 moved from 0.332 to 0.584.

**Nucleotide Transformer is the 500M checkpoint, not 2.5B**, because of disk constraints on
the available GPU machine. Its architecture also required different LoRA target modules and
it uses a different tokenisation, so it is not a like-for-like comparison with DNABERT-2.

**Single dataset, single context.** All results come from one library, one promoter scaffold
(pTpA), one carbon source (glucose), one organism. Galactose and glycerol conditions and a
second scaffold exist in the same GEO record and were not used.

**These are random sequences, not evolved ones.** The transfer ladder quantifies that cost
within *S. cerevisiae* (0.089–0.105 Spearman), but the jump to another species is untested
here.

**The negative active-learning result is specific to a random library.** With 99.84%
singletons there is no cluster structure for diversity sampling to exploit. A natural,
evolved promoter collection may behave differently, and this result should not be read as a
general argument against active learning. Selection was also one-shot rather than iterative,
and diversity clustered k-mer features rather than language-model embeddings.

**Sequence length weakly confounds expression** (ρ = 0.09 test, 0.06 train; mean label 9.12
at exactly 80 bp against 8.03 for shorter inserts). k-mer counts are unnormalised, so models
can see length indirectly. Too small to explain performance of ~0.9, but present.

**The cluster-based split is degenerate here.** It is reported alongside the random split as
specified, but with 99.84% singletons the two coincide (agreeing to <0.005 at n ≥ 1,000).

**Ranking is good, picking the best is not.** precision@100 never exceeds 0.17 at any
training size: of the 100 sequences a model ranks highest, ~17 are truly in the top 100.
Useful for triage, not for designing a maximally strong promoter.

## References

- de Boer et al. (2020). Deciphering eukaryotic gene-regulatory logic with 100 million
  random promoters. *Nat Biotechnol* 38(1):56–65. GEO: GSE104878.
- Zhou et al. (2024). DNABERT-2. *ICLR 2024*.
- Dalla-Torre et al. (2023). The Nucleotide Transformer.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Steinegger & Söding (2017). MMseqs2. *Nat Biotechnol* 35(11):1026–1028.
