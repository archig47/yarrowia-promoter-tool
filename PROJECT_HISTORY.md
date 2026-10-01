# Project history

This project was started twice. The first attempt is preserved in
`notebooks/june2026_prior_attempt.bundle` (recoverable with
`git clone notebooks/june2026_prior_attempt.bundle`). This note explains what changed
and why, because the reason is itself one of the project's findings.

## June 2026 — the original plan

Three commits, all on 24 June, then nothing. The motivation was the same as now:
*Yarrowia lipolytica* and other non-conventional yeasts have almost no characterised
promoters, so a model trained on *S. cerevisiae* would be valuable if it transferred.

The method was different. The stated aims were:

1. Fine-tune a pretrained DNA language model on de Boer et al.'s yeast promoter data
   using LoRA
2. **Generate synthetic promoter sequences with the pretrained model and annotate them
   with predicted expression values**
3. **Evaluate whether augmenting real training data with these synthetic sequences
   improves performance**
4. Compare model attribution scores against known TF binding sites

Aims 2 and 3 are the synthetic data augmentation idea: if real measurements are scarce,
manufacture more.

## Why it was abandoned

Two reasons, and only one of them is interesting.

**The honest, boring one:** the plan was underspecified. It had no evaluation protocol,
no splits, no stated success criterion, and no budget. It stopped after a day's setup
because there was no defined next step, not because anyone had disproved it.

**The one worth recording:** the plan has a flaw that this project went on to measure.

Synthetic data augmentation of this kind labels generated sequences with the model's own
predictions and trains on them. That only helps if the model's predictions are reliable
for the sequences it generates — and a generator will preferentially produce sequences it
believes are strong, which is exactly the regime where these models are weakest.

The current work quantifies that weakness. From the write-up (§8.1, §8.7):

| evaluation set | Spearman | precision@100 |
|---|---|---|
| random held-out sequences | 0.902 | 0.16 |
| **deliberately designed sequences** | 0.764 | **0.01** |

**Of the 100 designed sequences the model ranks highest, one is genuinely in the true top
100.** A model that cannot identify which of its own candidates are good is not a usable
oracle for filtering generated sequences. The augmentation loop would have amplified its
errors rather than correcting them.

That number did not exist in June. It exists now because the project was redirected toward
measuring what these models can and cannot do, rather than assuming they could do enough.

## What replaced it

The question became: **how many real measurements do you actually need, and does pretraining
substitute for them?** Instead of manufacturing data, measure the value of data — by
withholding most of a large dataset and training at 100, 300, 1,000 … 100,000 examples.

That reframing produced the measurement-budget curves (`README.md`), the two *Yarrowia*
benchmarks, and the shortlisting tool (`predict.py`).

## What carried over

- The motivating problem: promoter engineering in a yeast with almost no data
- The dataset: de Boer et al. 2020, GSE104878
- The method: LoRA fine-tuning of DNABERT-2 and Nucleotide Transformer
- Aim 4: motif recovery by attribution, which became D3 and is one of the stronger results —
  DNABERT-2 recovers REB1, a canonical activator the from-scratch CNN misses entirely

## What this is not

It is not evidence of sustained work across the summer. It is one day in June and a
restart in September. The value of recording it is that the first plan's flaw is something
the second project went on to measure, which is a better reason to abandon an approach than
losing interest in it.
