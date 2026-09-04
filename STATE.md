# STATE

Living handoff document. Update at the end of every working session.
Keep it under one page — status board, not a log.

**Last updated:** 2026-09-04
**Day:** 0 of 14
**Checkpoint:** 1

---

## Where things stand

Repo bootstrapped. PREDICTIONS.md written and committed before any data was
downloaded. Day 1 (data + harness) not started.

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
- [ ] High-quality test set secured (or documented fallback to cluster split)
- [ ] Three seeds below n = 10,000
- [x] PREDICTIONS.md written before any modelling
- [ ] Baselines reported honestly in README first paragraph
- [ ] Limitations section written
- [ ] One-command reproduction works from clean clone
- [ ] v1.0 tagged

---

## Open decisions

- Read-count filter threshold: not yet chosen
- Compute: local vs rented GPU — not yet decided

---

## Blockers

None yet.

---

## Decision log

(append only, one line each: date — decision — reason)

- 2026-09-04 — PREDICTIONS.md answers Q1 only; Q2-Q7 recorded as "no prediction" — no prior modelling experience to base them on, and guesses carrying no information would dilute the one prediction that does carry a signal. Q1 is the headline scaling curve, so the leakage tripwire (a result far above ~0.20 Spearman at n=100 indicates a broken split, not success) is preserved where it matters. Not to be back-filled.
- 2026-09-04 — .gitignore corrected so data/raw/.gitkeep and data/processed/.gitkeep are tracked — bootstrap ignored the directories wholesale, so a fresh clone would not recreate them, breaking one-command reproduction.

---

## Schedule slip

**Days used vs planned:** 0 / 0
**Cut so far:** none

Cut order if behind: D4 (Yarrowia) first. Never cut seeds, baselines, or write-up.
