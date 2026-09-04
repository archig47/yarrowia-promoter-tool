# PREDICTIONS

Written before any modelling. **Append-only. Never edit an earlier entry.**
The point is to be able to report where I was wrong.

**Date written:** 2026-09-04 (Day 0, before any data was downloaded or inspected)

**Scope note.** Q1 was answered in full. Q2-Q7 were deliberately left unanswered
rather than filled in with guesses carrying no information. This was a decision taken
on Day 0, not an oversight, and not something to be filled in later — see the decision
log in `STATE.md`. Anything written after results exist is not a prediction.

**Metric convention.** All numbers below are Spearman on the primary test set
(de Boer high-quality pTpA/glucose, n ~ 9,982), per `PLAN.md`.

---

## Q1. Which method wins at each training set size?

- **n = 100:** k-mer ridge wins, at ~0.20 Spearman, ~0.05 ahead of the runner-up.
  No prediction made as to which method comes second.
- **n = 1,000:** k-mer ridge still wins, at ~0.30 Spearman.
- **n = 10,000:** ridge loses the lead. **DNABERT-2 (LoRA) takes over.**
- **n = 100,000:** DNABERT-2 still leads. The curve plateaus around here, at
  **~0.70 Spearman**.

**Shape of the curve and reasoning.** Accuracy climbs steadily with more data and
flattens near n = 100,000. Underlying belief: this task needs a large amount of data
before it works at all. The simple, heavily constrained model wins early precisely
because it cannot overfit; the pretrained model needs more examples before its
capacity pays off, and then overtakes.

**What would falsify this.** Any of: ridge not leading at n = 100 or n = 1,000; ridge
still leading at n = 10,000; DNABERT-2 not the leader at n >= 10,000; a plateau
materially above or below ~0.70; or performance at n = 100 already near the plateau
(which would mean the task is far less data-hungry than assumed).

## Q2. Does genomic pretraining help, and where?

**No prediction recorded.** (Partially implied by Q1: pretraining is expected to win
from n = 10,000 upward. Nothing was predicted about whether it helps at all at
n = 100-1,000, which is the more interesting regime for this project.)

## Q3. Where does the LM/CNN crossover happen, if at all?

**No prediction recorded.**

## Q4. Active learning — how many random samples does a chosen 300 replace?

**No prediction recorded.**

## Q5. Will ISM recover known yeast TF motifs?

**No prediction recorded.**

## Q6. Will zero-shot transfer to Yarrowia work at all?

**No prediction recorded.**

## Q7. What is most likely to go wrong in the next two weeks?

**No prediction recorded.**

---

## Outcomes

(filled in at write-up. State plainly where the above was wrong and why.)
