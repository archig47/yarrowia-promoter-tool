# STATE

Living handoff document. Update at the end of every working session.
Keep it under one page — status board, not a log.

**Last updated:** 2026-09-04 (Day 1)
**Day:** 1 of 14
**Checkpoint:** 1

---

## Where things stand

Day 1 in progress. PREDICTIONS.md committed blind before any data. Data secured and
verified: pTpA/glucose training set (pTpA_1E8_YPD) and high-quality test set
(pTpA_3E5_YPD, n=9,982) both downloaded, checksummed, gzip-intact, and confirmed as the
correct scaffold+condition from GEO metadata. Sequence structure characterised.
Not yet done: parse to Parquet, cluster, freeze splits, build subsample indices,
write and freeze evaluate.py.

---

## Deliverable status

| ID | Deliverable | Status | Notes |
|----|-------------|--------|-------|
| D1 | Scaling curves | not started | |
| D2 | Selection curves | not started | |
| D3 | Motif recovery | not started | |
| D4 | Yarrowia transfer | not started | |

Status values: not started / in progress / blocked / done

---

## Definition-of-done checklist

- [ ] evaluate.py frozen after Day 2
- [x] High-quality test set secured — exactly 9,982 rows, no fallback needed
- [ ] Three seeds below n = 10,000
- [x] PREDICTIONS.md written before any modelling
- [ ] Baselines reported honestly in README first paragraph
- [ ] Limitations section written
- [ ] One-command reproduction works from clean clone
- [ ] v1.0 tagged

---

## Open decisions

- ~~Read-count filter threshold~~ — CLOSED: no read counts published, no threshold exists
- Compute: local machine is an M2 / 16GB / no CUDA. Unsuitable for Day 4 LoRA runs
  (DNABERT-2's triton/flash-attn stack is CUDA-oriented). Rented GPU effectively
  required. Not decided which; blocks Day 4, not Day 1-3.

---

## Blockers

None yet.

---

## Decision log

(append only, one line each: date — decision — reason)

- 2026-09-04 — Read-count filter dropped as impossible, not skipped — de Boer published no per-sequence read counts in either distributed file. Training data used unfiltered; the provided atLeast100Counts file (>=100 reads) is the test set. The authors' ~24% noise estimate quantifies precisely this. Method comparisons unaffected; absolute data-efficiency numbers are pessimistic. Must appear in Limitations.
- 2026-09-04 — Label description corrected in PLAN.md and CLAUDE.md: values are expression level (18-bin weighted average, ~1.5-16.7), not log2(YFP/RFP). Spearman unaffected; affects Pearson/R^2 interpretation and MSE scaling. Training labels integer, test labels continuous.
- 2026-09-04 — Sequence convention frozen: trim 17bp 5' + 13bp 3' constant scaffold (confirmed empirically by base conservation AND independently by the GEO series design). Keep all sequences, no length filtering, pad to 95 for CNN and TF-MoDISco. Rationale: indels concentrate in homopolymer runs, so length-filtering would systematically deplete the poly-A motif that D3 is meant to recover.
- 2026-09-04 — Train and test come from separately constructed libraries (pTpA_1E8 vs pTpA_3E5), so the primary evaluation is structurally leakage-proof. To be verified empirically by set intersection when the training file is parsed.
- 2026-09-04 — Prior June 2026 attempt at this project (different scope: synthetic data augmentation) deleted after confirming its data files were byte-identical. Git history preserved as notebooks/june2026_prior_attempt.bundle.

- 2026-09-04 — PREDICTIONS.md answers Q1 only; Q2-Q7 recorded as "no prediction" — no prior modelling experience to base them on, and guesses carrying no information would dilute the one prediction that does carry a signal. Q1 is the headline scaling curve, so the leakage tripwire (a result far above ~0.20 Spearman at n=100 indicates a broken split, not success) is preserved where it matters. Not to be back-filled.
- 2026-09-04 — .gitignore corrected so data/raw/.gitkeep and data/processed/.gitkeep are tracked — bootstrap ignored the directories wholesale, so a fresh clone would not recreate them, breaking one-command reproduction.

---

## Schedule slip

**Days used vs planned:** 1 / 1
**Cut so far:** none

Cut order if behind: D4 (Yarrowia) first. Never cut seeds, baselines, or write-up.
