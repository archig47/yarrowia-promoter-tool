# Data efficiency of DNA language models for promoter strength prediction in yeast

**Checkpoint 1 write-up.** All numbers come from the frozen evaluation harness
(`src/evaluate.py`, sha256 `a5ed94c66327340c…`, unchanged since Day 2) and are reproducible
from `results/runs.csv` via `make all`.

---

## 1. Question

How few experimental measurements are needed before sequence-based promoter strength
prediction becomes useful, and does pretraining on genomes substitute for measurements you
cannot afford?

The motivation is an asymmetry. *Saccharomyces cerevisiae* has on the order of 10⁸ measured
promoters. *Yarrowia lipolytica* — an oleaginous yeast used industrially for lipids, organic
acids and heterologous protein production — has perhaps a hundred that have ever been
characterised. That gap is what blocks predictive expression engineering in non-model
organisms: you cannot tune expression without predicting promoter behaviour, and you cannot
train a predictor without measurements.

The design treats a large dataset as a **simulator of scarcity**. Rather than asking "how well
can a model do", it asks "how well can a model do given *n* measurements", for *n* from 100 to
100,000, and reports the answer in the units a wet lab budgets in: promoters that must be
built and assayed.

## 2. Data

**Training and primary evaluation.** de Boer et al. (2020), GEO
[GSE104878](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104878). A library of ~10⁸
random 80 bp sequences was synthesised, inserted into a fixed pTpA promoter scaffold in living
*S. cerevisiae*, and expression measured by FACS sorting into 18 bins by YFP/RFP ratio. The
distributed `pTpA_1E8_YPD` file gives **31,349,363** sequence–expression pairs in glucose. A
separately assayed lower-complexity library, `pTpA_3E5_YPD`, gives **9,982** sequences with
higher-precision labels; this is the primary test set.

Two assumptions in the project plan did not survive contact with the data:

- **Read-count filtering is impossible, not merely skipped.** The plan called for filtering
  low-read sequences, but de Boer published no per-sequence read counts — both distributed
  files carry sequence and expression level only. The `atLeast100Counts` file *is* the
  read-count-filtered subset and serves as the test set; the training data cannot be filtered.
  The authors' estimated ~24% training-label noise quantifies exactly this.
- **Labels are not log2(YFP/RFP).** The published values are an expression level on a 0–17
  scale (a weighted average over the 18 sorting bins), not a log ratio. Spearman is unaffected;
  Pearson, R² and MSE scaling are.

**Sequence structure.** Each 110 bp construct is a 17 bp constant 5′ scaffold
(`TGCATTTTTTTCACATC`), a variable insert, and a 13 bp constant 3′ scaffold (`GGTTACGGCTGTT`),
confirmed both by per-position base conservation (99.2% / 99.99%) and independently by the GEO
series design. Inserts are exactly 80 bp in 80.5% of cases and 65–95 bp overall. The scaffold
is trimmed; **no length filtering is applied**, because indels concentrate in homopolymer runs
and filtering would systematically deplete the poly-A motif that the interpretability analysis
sets out to recover.

**Additional evaluation sets**, all from the same GEO record and the same assay:

| set | n | what it is |
|---|---|---|
| Native80 | 62,897 | real evolved *S. cerevisiae* promoter fragments, same scaffold |
| N80 spike-ins | 8,027 | random controls measured in the same experiment |
| designed | 69,697 | deliberately designed and in-silico-evolved sequences |

The Native80 file also carries two independent replicates, which give a **measured noise
ceiling**: rep1 vs rep2 Spearman is **0.980** for random sequences and **0.934** for native
ones. No model can exceed the assay's agreement with itself.

## 3. Splits, and why they matter

**A 66.6% data leak was present and had to be removed.** 6,651 of the 9,982 primary test
sequences appear verbatim in the training data. The cause is in the GEO growth protocol: the
high-quality `pTpA_3E5` library is a *dilution of the same pool* as `pTpA_1E8`, not an
independent synthesis. Testing on sequences the model trained on measures memorisation, and
critically **nothing errors when this happens** — the scores simply come out quietly too good,
and the inflation grows with *n*. Those 6,651 rows (0.021% of training data) are excluded from
every training subsample; the test set is untouched.

This was caught because the overlap check was written into the parsing script rather than left
as something to verify later. It is the single most consequential decision in the project.

**Frozen artefacts.** A 200,000-sequence working pool was drawn (excluding the overlap rows),
clustered with MMseqs2, and split two ways: a cluster-based held-out split and a random one,
10,000 held out in each. Nested subsample indices were generated once for every *n* × 3 seeds,
so increasing *n* *adds* data rather than resampling it. All index files are frozen.

**The cluster split is degenerate here, and that is itself a result.** MMseqs2 finds 199,680
clusters among 200,000 sequences: **99.84% singletons, largest cluster size 2**. For a random
library there is no homology for a cluster-aware split to protect against. Empirically the two
splits agree to within 0.017 at *n* ≥ 1,000. Both are reported as specified; the coincidence
is a property of random-sequence libraries worth stating rather than a corner cut.

**Evaluation.** `src/evaluate.py` takes true and predicted labels only — it cannot see which
model produced them. It returns Pearson, Spearman, R², RMSE and MAE, each computed overall and
restricted to the true top and bottom deciles, plus precision@k and a hash of the true-label
vector so that results scored on different test sets cannot be silently compared. It was frozen
on Day 2 and its checksum recorded.

**Negative control.** Retraining on deliberately shuffled labels gives Spearman between −0.06
and +0.10 at every *n*, against +0.68 to +0.90 with real labels. Nothing leaks through the
features, the indices or the evaluation code.

## 4. Baselines

Five model families, deliberately spanning from trivial to large:

| model | parameters / features | what it can represent |
|---|---|---|
| GC content | 1 feature | base composition only |
| dinucleotide frequencies | 16 features | local composition |
| k-mer ridge | 4,096 features | weighted sum of 6-mer counts; position-blind |
| LightGBM | ~hundreds of trees | thresholds and combinations of 6-mer counts |
| CNN from scratch | 156,065 parameters | learned motifs, their positions and interactions |
| DNABERT-2 + LoRA | 117M pretrained | as above, plus multi-species genomic pretraining |
| Nucleotide Transformer 500M + LoRA | 493M pretrained | as above, larger |

**The composition baselines are reported first because they materially change how the rest
should be read.** On the primary test set at n=100,000:

| | features | Spearman |
|---|---|---|
| GC content alone | 1 | **0.604** |
| base composition | 4 | 0.620 |
| dinucleotide frequencies | 16 | **0.763** |
| k-mer ridge | 4,096 | 0.865 |
| CNN | 156k params | 0.897 |
| DNABERT-2 | 117M params | **0.902** |

Sixteen numbers come within 0.14 of a 117-million-parameter language model. The deep models add
real and reproducible value, but most of this task is base composition. This is consistent with
a growing benchmarking literature finding that genomic language models offer limited advantage
over simple supervised baselines on regulatory tasks (DART-Eval 2025; Tang & Koo 2024;
Specialized Foundation Models Struggle to Beat Supervised Baselines 2024).

**Two LoRA implementation notes**, both of which silently produce wrong results if missed.
DNABERT-2 fuses attention into a single `Wqkv` module and has no `query` or `value` — the
repository's default `lora_target_modules="query,value"` matches *nothing*, attaches LoRA to no
layers, and produces a flat loss curve indistinguishable from a model that cannot learn. And a
flat 50-step warmup, as specified in the plan, is ~1% of a long run but **17 epochs** of a run
at n=100, where 80 training sequences at batch 32 give only 3 optimiser steps per epoch; two of
three seeds stopped mid-warmup having never trained at the intended learning rate. Warmup was
changed to `min(50, 10% of planned steps)` with a guard against early stopping before it
completes. This affects n ≤ 300 only, and moved n=100 from 0.332 to 0.584.

## 5. Scaling curves (D1)

Spearman on the primary test set, mean of 3 seeds, 840 logged runs.

| measurements | ridge | LightGBM | CNN | DNABERT-2 | NT-500M |
|---|---|---|---|---|---|
| 100 | 0.422 | 0.066 | 0.521 | **0.563** | 0.471 |
| 300 | 0.563 | 0.327 | 0.639 | **0.702** | 0.540 |
| 1,000 | 0.679 | 0.406 | 0.728 | **0.771** | 0.729 |
| 3,000 | 0.756 | 0.692 | 0.794 | **0.801** | 0.788 |
| 10,000 | 0.806 | 0.784 | **0.838** | 0.831 | 0.828 |
| 30,000 | 0.832 | 0.824 | **0.868** | 0.868 | 0.859 |
| 100,000 | 0.865 | 0.839 | 0.897 | **0.902** | 0.888 |

**In measurement units — the project's headline.** DNABERT-2 trained on 300 promoters matches
a from-scratch CNN trained on **700** and k-mer ridge trained on **1,390**. At n=1,000 the
equivalences are 2,067 (CNN) and 4,352 (ridge). **Genomic pretraining is worth roughly a 2–4×
reduction in measurements in the data-scarce regime**, and the advantage decays as data grows:
+0.063 at n=300, +0.043 at n=1,000, +0.008 at n=3,000, before the from-scratch CNN overtakes at
n=10,000–30,000.

**Model identity matters more than model size.** NT-500M is four times larger than DNABERT-2
and loses to it at every training size, falling *below* plain k-mer ridge at n=300 (0.540 vs
0.563). A single-language-model study would have concluded "pretraining works"; two turn that
into a narrower claim.

**Reliability is a separate axis from accuracy.** Standard deviation across three seeds at
n=100: ridge 0.034, NT 0.058, DNABERT-2 0.052, LightGBM 0.066, **CNN 0.083** — the CNN's three
runs on identical data gave 0.37, 0.53 and 0.59. Single-seed results at small *n* are not
results. Spread collapses below 0.02 by n=1,000. The practically useful statement is not
"simple models win when data is scarce" but **"simple models are *reliable* when data is
scarce"**: the CNN has the better mean at n=100 and the worse worst case.

## 6. Selection curves (D2)

Does choosing *which* promoters to measure beat choosing at random? Pool-based, one-shot,
ridge as the workhorse so that differences reflect selection rather than training noise. For
uncertainty and hybrid, the 100 seed labels needed to fit the ensemble **count against the
budget**, so every strategy is scored on the same number of measurements.

| budget | random | uncertainty | diversity | hybrid |
|---|---|---|---|---|
| 100 | 0.417 | 0.417* | 0.407 | 0.417* |
| 300 | **0.557** | 0.517 | 0.551 | 0.538 |
| 1,000 | **0.676** | 0.639 | 0.667 | 0.656 |
| 3,000 | 0.748 | 0.735 | **0.751** | 0.732 |

\* at budget 100 the entire budget is the seed set, so these are random draws by construction.

**No strategy beats random, and uncertainty sampling is consistently worse.** Inverting the
plan's framing: a carefully chosen 300 is worth about **219** randomly chosen measurements.

Two mechanisms, both evidenced. Uncertainty sampling introduces **sampling bias**: at budget
1,000 it selects sequences with mean label 10.30 against a pool mean of 9.51, and 23.1% from
the extreme deciles against a 20% baseline. A training set that does not match the test
distribution generalises worse, and that cost exceeds the information gained. Diversity
sampling has **nothing to exploit**: the 99.84%-singleton clustering result means a random
library has no under-sampled regions to target, which is exactly why it ties with random
(0.91–0.95×).

**Practical consequence:** for a random library, do not spend effort on experimental design.
Measure any 300 and spend the effort on the model instead. This should not be read as a general
argument against active learning — an evolved, structured promoter collection may behave
differently, and selection here was one-shot rather than iterative.

## 7. Motif recovery (D3)

In-silico mutagenesis (ISM) over 4,000 held-out test sequences — every position mutated to
every base, predictions mean-centred per position — then TF-MoDISco clustering of the
attribution maps, then matching against the 245 motifs de Boer used (YeTFaSCo position
frequency matrices; 244 of 245 IDs matched exactly) plus a constructed poly-A motif. ISM was
used rather than attention maps because DNABERT-2's byte-pair tokens are variable-length and do
not align to single bases.

| | CNN | DNABERT-2 |
|---|---|---|
| patterns recovered | 6 | **11** |
| matching a known motif at r ≥ 0.75 | 6 | 10 |
| poly-A / poly-T | ✅ 3 patterns | ✅ 3 patterns |
| RSC30 / RSC3 | ✅ r = 0.95 | ✅ r = 0.98 |
| **REB1** | ✗ | ✅ **r = 0.95** |
| repressive (negative) patterns | 1 | **5** |

**This is a mechanism for the D1 result rather than a restatement of it.** DNABERT-2 recovers
REB1 — a canonical yeast activator — which the from-scratch CNN misses entirely, plus five
distinct repressive motifs against the CNN's one. Pretraining did not only improve the numbers;
it found more of the real regulatory grammar. Reported honestly as partial: Rap1 and Abf1,
which de Boer highlight, were recovered by neither, and one DNABERT-2 pattern matches nothing
known.

The poly-A recovery is a direct consequence of the Day 1 decision not to length-filter.
Indels concentrate in homopolymer runs, so filtering the 19.5% of non-80bp sequences would have
depleted exactly the motif class this analysis was designed to find.

## 8. Transfer (D4, D4a)

### 8.1 A ladder, so failures can be attributed

Each rung changes one variable relative to the training data. DNABERT-2, n = 100,000:

| rung | what changes | Spearman | precision@100 |
|---|---|---|---|
| held-out random 80-mers | nothing | 0.902 | 0.16 |
| N80 spike-ins | different experiment, same assay | 0.902 | 0.21 |
| Native80 | random → **evolved sequence** | 0.813 | 0.06 |
| designed | random → **engineered sequence** | 0.764 | 0.01 |

Changing sequence type costs ~0.1 Spearman, and the cost replicates across all five model
families (+0.089 to +0.105), so it is a property of the task rather than of an architecture.
The spike-in control reproducing the primary test set (0.902 vs 0.902) from a different file and
experiment validates the pipeline.

### 8.2 Building a *Yarrowia* benchmark

None existed in machine-readable form. Two were assembled:

| benchmark | n | construction | learnable in-domain? |
|---|---|---|---|
| reporter-measured | 81 | strengths from Zhang et al. (preprint rs-1993869) supplementary table; sequences from the CLIB122 reference genome, since the paper identifies promoters by gene | **no** — CV ρ ≈ 0, p > 0.1 |
| RNA-seq derived | **6,045** | TPM from Lubuta et al. (2019) glucose chemostats; same genome | **yes** — CV ρ = 0.390 |

Extraction of the 81 was validated against the source paper's own Strong/Medium/Weak labels
(Spearman +0.912; class counts 15/42/25 against a published 15/41/25).

**The 81-promoter set — the largest assemblable from published reporter assays — is too small
to learn from at all.** Ridge trained on Yarrowia and cross-validated on Yarrowia gives ρ from
−0.037 to +0.096 depending on features, none significant against permutation nulls. That makes
transfer *unevaluable* rather than refuted, and it is the project's premise demonstrated: the
available data sits below the threshold at which sequence-to-expression learning becomes
possible, and §5 shows where that threshold is.

### 8.3 Window length, and a confirmation of promoter architecture

On the RNA-seq benchmark, in-domain accuracy varies with how much upstream sequence is used:
80 bp gives 0.325, **250 bp gives 0.390**, 1000 bp gives 0.345. Blazeck et al. place a minimal
functional *Y. lipolytica* promoter at 130–260 bp upstream of the ATG, with the TATA box
40–120 bp upstream of the TSS and UAS elements at the 5′ end. The empirical optimum falls
inside that range. **The 80 bp window imposed by de Boer's training data is therefore a genuine
handicap for this species** — measured, not argued.

### 8.4 Seven transfer methods, none beating from-scratch

All scored on the same held-out 1,511 *Yarrowia* promoters, 3 seeds:

| *k* (Yarrowia promoters) | ridge 80bp | ridge 250bp | DNABERT-2 | DNABERT-2 via *S. cer* |
|---|---|---|---|---|
| 50 | 0.133 | **0.166** | 0.081 | 0.089 |
| 100 | 0.196 | **0.216** | 0.026 | 0.128 |
| 300 | 0.248 | **0.282** | 0.211 | 0.151 |
| 1,000 | 0.276 | **0.339** | 0.312 | 0.297 |
| 3,000 | 0.307 | 0.380 | **0.399** | 0.377 |

Also tested: zero-shot (ρ = −0.010), ridge shrunk toward the *S. cerevisiae* coefficients
rather than toward zero (0.076 at k=3,000 — **worse** than shrinking toward zero), the source
prediction as a frozen feature (≈0), and that feature concatenated with k-mers (≈ from-scratch).
Tiling the 80 bp window across 1,000 bp of upstream sequence produced a best single window of
ρ = 0.256 at −760 bp, but a permutation test repeating the full window sweep gives a null median
of 0.251 and **p = 0.457** — the apparent distal signal is exactly what selecting the maximum
over 47 windows produces from noise.

**The within-species ordering reverses.** DNABERT-2 is significantly *worse* than k-mer ridge
at k = 50, 100 and 300 (gap exceeding twice the pooled standard deviation) and indistinguishable
at k = 1,000 and 3,000 — despite being the only model that can exploit the better 250 bp window,
since a tokeniser accepts any length while ridge trained on 80 bp counts breaks on 250 bp inputs.

### 8.5 The barrier is the assay and chromatin context, not the species

A matched control isolates the variable. *S. cerevisiae* native promoters with an RNA-seq label
(GSE316459, BY4741 wild-type, 5,484 promoters) differ from the Yarrowia RNA-seq benchmark
*only in species* — same label type, same native chromatin context. Zero-shot from the de Boer
ridge model, 250 bp window:

| target | what differs from training | Spearman |
|---|---|---|
| random 80-mers, same assay | nothing | **+0.864** |
| *S. cerevisiae* RNA-seq | assay + chromatin context | **+0.121** |
| Yarrowia RNA-seq | assay + context + **species** | **+0.075** |
| | → species contribution | **−0.046** |

**The model fails just as badly on its own species' native promoters as on Yarrowia's.** The
corrected decomposition:

```
random → native sequence, same scaffold and assay    0.902 → 0.813   (−0.09)
       → native chromatin, RNA-seq readout                 → 0.121   (−0.69)
       → different species                                 → 0.075   (−0.05)
```

The dominant term, by an order of magnitude, is the assay and chromatin context.

### 8.6 Why: learnable and transferable are different feature sets

Scoring all 243 usable motifs in both species and correlating with expression separately:
the two effect vectors **agree at ρ = 0.468**, and 109 of the 145 motifs strong in
*S. cerevisiae* (75%) keep their sign in Yarrowia. **Conserved regulatory signal exists.** But
magnitudes collapse — the largest consistent Yarrowia effect is ~0.11 (ECM23, GAT4, ADR1,
MIG3), while *S. cerevisiae* effects reach 0.52. Meanwhile the strongest *S. cerevisiae* motifs
are inert in Yarrowia: RSC3 **+0.518 → −0.021**, PDR3 +0.396 → −0.007, WAR1 +0.351 → −0.002.

**Those are precisely the motifs §7 found the models learning.** RSC30/RSC3 was the CNN's
dominant ISM pattern (354 seqlets) and DNABERT-2 recovered WAR1 at r=0.96 and RSC3 at r=0.98.
RSC is a chromatin-remodelling complex, and de Boer's assay places random inserts in a fixed
scaffold where nucleosome positioning dominates — a constraint that does not apply to native
promoters in native chromatin. The learned 6-mer coefficient vectors are consequently
orthogonal between species (Pearson −0.0003), sharing 2 of each model's top-200 activating
6-mers, fewer than the ~10 expected by chance.

The composition-level cause is the same: **GC content predicts expression at ρ = +0.604 in
de Boer's random library and −0.032 in Yarrowia's native promoters.** The rule the models learn
most strongly is the one that does not carry over.

**Transfer fails not because nothing is transferable, but because the learnable signal and the
transferable signal are different sets.**

### 8.7 Is this a useful predictive model?

| evaluation set | Spearman | R² | precision@100 |
|---|---|---|---|
| random 80-mers, clean labels | 0.902 | 0.77 | 0.16 |
| random spike-in controls | 0.902 | 0.66 | 0.21 |
| random 80-mers, noisy labels | 0.776 | 0.58 | 0.12 |
| real evolved *S. cerevisiae* promoters | 0.813 | 0.61 | 0.06 |
| designed sequences | 0.764 | 0.45 | 0.01 |
| *Yarrowia* promoters | −0.015 | −22.0 | — |

As a hierarchy of claims: ranking sequences from its training distribution is **strong** (0.902
against a 0.98 measured ceiling); predicting absolute values is **moderate** (RMSE ≈ 2.1 on an
0–17 scale); ranking within the true top decile is **weak** (ρ = 0.223); identifying the best
few **fails** at every training size, with precision@100 never exceeding 0.26 across all models,
sizes and evaluation sets, and reaching 0.01 on designed sequences; and working on native
promoter expression **does not happen**, in any species.

## 9. Predictions versus outcomes

`PREDICTIONS.md` was written and committed before any data was downloaded, and has not been
edited. Q1 was answered in full; Q2–Q7 were deliberately recorded as "no prediction" rather than
filled with uninformed guesses, a decision logged at the time.

| | predicted | actual |
|---|---|---|
| ridge at n=100 | 0.20 | **0.422** |
| ridge at n=1,000 | 0.30 | **0.679** |
| winner at n=100 | ridge | **DNABERT-2** (ridge second) |
| winner at n=10,000 | DNABERT-2 | **CNN** |
| plateau | ~0.70 | passed by n≈3,000; 0.902 at n=100,000, still rising |

**The prediction was wrong in the optimistic direction and wrong about the ordering.** The task
is substantially more learnable from small samples than expected: 100 measurements give ρ≈0.5,
not 0.2. The stated belief that "this task needs a large amount of data before it works at all"
is falsified. What survives is a weaker and more precise version of the intuition behind it —
that constrained models are *reliable* where flexible ones are erratic (§5).

The prediction also served its operational purpose. Its value was never accuracy but the
tripwire: a recorded expectation of ~0.2 at n=100 meant that a result near 0.85 would have been
investigated rather than celebrated. That is the class of error the 66.6% leak would have
produced.

## 10. Limitations

**Most of the task is composition.** GC content alone gives 0.604 and 16 dinucleotide
frequencies give 0.763, against 0.902 for a 117M-parameter model. Claims about what these
models "learn" must be read against that baseline.

**Measurement noise cannot be filtered**, because de Boer published no per-sequence read counts.
Training labels carry the authors' estimated ~24% noise. Method comparisons are unaffected, but
the absolute measurement counts reported here are **pessimistic** — a lab making cleaner
measurements would need fewer than these numbers suggest.

**Fixed hyperparameters, no search.** LightGBM's collapse at n=100 (ρ=0.066) is substantially an
artefact of early stopping judged on a 20-sequence validation set, and part of NT-500M's
underperformance may be the same. Every claim is "with this tuning budget", not "this model
cannot do better".

**One documented protocol deviation.** The warmup change described in §4, affecting n ≤ 300
only.

**Nucleotide Transformer is the 500M checkpoint, not 2.5B**, because of disk constraints on the
available GPU. Its architecture required different LoRA target modules and it uses a different
tokenisation, so it is not a like-for-like comparison with DNABERT-2.

**The large *Yarrowia* label is a proxy.** mRNA abundance is confounded by transcript stability
and reflects promoters in native chromatin rather than a fixed reporter scaffold. It correlates
with reporter-measured strength at only ρ=0.428 on the 80 promoters present in both benchmarks.

**The two RNA-seq benchmarks are not perfectly matched.** The Yarrowia labels are
length-normalised TPM; the *S. cerevisiae* labels are DESeq2-normalised counts, which are not.
The Yarrowia data is chemostat steady-state and the *S. cerevisiae* data batch culture. The
in-domain accuracies (0.390 vs 0.169) are therefore **not directly comparable**, and the
apparent greater predictability of Yarrowia promoters is more likely a label-processing
difference than biology. The zero-shot comparison in §8.5, which uses the same source model
against both, is unaffected.

**We do not know what promoter region the reporter study cloned.** Sequences for the
81-promoter benchmark were taken as genomic upstream sequence. If Zhang et al. cloned a
different span, those sequences do not correspond to what was measured — which alone could
explain that benchmark's failure.

**Cross-species conclusions rest on one species pair**, and a very distant one. Nothing here
bounds how much evolutionary distance transfer survives.

**The negative active-learning result is specific to a random library**, where 99.84%
singletons leave no cluster structure for diversity sampling to exploit. Selection was one-shot
rather than iterative, and diversity clustered k-mer features rather than language-model
embeddings.

**Single dataset, single context.** One library, one scaffold (pTpA), one carbon source
(glucose), one organism for training. Galactose, glycerol and a second scaffold exist in the
same GEO record and were not used.

**Sequence length weakly confounds expression** (ρ=0.09 test, 0.06 train; mean label 9.12 at
exactly 80 bp against 8.03 for shorter inserts). k-mer counts are unnormalised, so models can
see length indirectly.

**Findings 2 and 3 confirm rather than establish.** That genomic language models offer limited
advantage over simple supervised baselines on regulatory tasks is an active and independently
supported result in the benchmarking literature. The contribution here is the measurement-budget
framing, the *Yarrowia* benchmarks, and the assay-versus-species decomposition.

## 11. What the next experiments should be

Two, both directly motivated by §8.6 rather than speculative.

**Bound the domain shift.** Orthologous core promoters from four *Saccharomyces* species, all
assayed in one host (Lubliner et al. 2015 report 238), would isolate evolutionary distance with
the assay held constant — turning a single failed pair into a curve.

**Suppress the dominant compositional features during training** — regress GC and the
RSC-family motif scores out of the *S. cerevisiae* labels — and test whether the weakly
conserved motif signal transfers once it is not being drowned out. A falsifiable prediction
follows: such a model should perform *worse* in-domain and *better* out-of-domain.

---

## References

- de Boer CG, Vaishnav ED, Sadeh R, Abeyta EL, Friedman N, Regev A (2020). Deciphering
  eukaryotic gene-regulatory logic with 100 million random promoters. *Nat Biotechnol*
  38(1):56–65.
- Vaishnav ED et al. (2022). The evolution, evolvability and engineering of gene regulatory DNA.
  *Nature* 603:455–463.
- Zhou Z et al. (2024). DNABERT-2: efficient foundation model and benchmark for multi-species
  genome. *ICLR 2024*.
- Dalla-Torre H et al. (2023). The Nucleotide Transformer: building and evaluating robust
  foundation models for human genomics.
- Hu EJ et al. (2022). LoRA: low-rank adaptation of large language models. *ICLR 2022*.
- Shrikumar A, Tian K et al. TF-MoDISco: transcription factor motif discovery from importance
  scores.
- Steinegger M, Söding J (2017). MMseqs2 enables sensitive protein sequence searching for the
  analysis of massive data sets. *Nat Biotechnol* 35(11):1026–1028.
- de Boer et al. Supplementary Table 2 motifs, via YeTFaSCo (Spivak AT, Stormo GD 2012).
- Blazeck J, Liu L, Redden H, Alper H (2011). Tuning gene expression in *Yarrowia lipolytica*
  by a hybrid promoter approach. *Appl Environ Microbiol* 77(22):7905–7914.
- Lubuta P, Workman M, Kerkhoven EJ, Workman CT (2019). Investigating the influence of glycerol
  on the utilization of glucose in *Yarrowia lipolytica* using RNA-Seq-based transcriptomics.
  *G3* 9(12):4059–4071.
- Lubliner S et al. (2015). Core promoter sequence in yeast is a major determinant of expression
  level. *Genome Res* 25(7):1008–1017.
- Tang Z, Koo PK (2024). Evaluating the representational power of pre-trained DNA language
  models for regulatory genomics.
- Patel A et al. (2025). DART-Eval: a comprehensive DNA language model evaluation benchmark on
  regulatory DNA.
