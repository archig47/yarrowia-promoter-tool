# STATE

Living handoff document. Update at the end of every working session.
Keep it under one page — status board, not a log.

**Last updated:** 2026-09-04 (Day 1)
**Day:** 4 of 14 in progress (Days 1-3 done; Days 1-2 were finished early on Day 1)
**Checkpoint:** 1

---

## Where things stand

Day 1 in progress. PREDICTIONS.md committed blind before any data. Data secured and
verified: pTpA/glucose training set (pTpA_1E8_YPD) and high-quality test set
(pTpA_3E5_YPD, n=9,982) both downloaded, checksummed, gzip-intact, and confirmed as the
correct scaffold+condition from GEO metadata. Sequence structure characterised.
Days 1-2 data+harness work COMPLETE on Day 1. Parsed 31.3M train rows to Parquet;
found and excluded 6,651 train/test overlaps; clustered the 200k working pool
(99.84% singletons - cluster split is degenerate for a random library, as expected);
froze both splits and 21 subsample index sets per split; wrote and froze evaluate.py.
Day 3 baselines complete. Next: Day 4 LoRA fine-tunes (DNABERT-2, Nucleotide Transformer)
on the RTX 5090 over SSH. Bar to beat: CNN 0.897 at n=100,000 and 0.521 at n=100.

---

## Deliverable status

| ID | Deliverable | Status | Notes |
|----|-------------|--------|-------|
| D1 | Scaling curves | 4 of 5 models | 3 baselines + DNABERT-2 LoRA = 672 runs. NT-500M running. Figure not yet generated. |
| D2 | Selection curves | not started | |
| D3 | Motif recovery | not started | |
| D4 | Yarrowia transfer | not started | |
| D4a | Transfer ladder | replicated | random->natural costs +0.094/+0.097/+0.103 Spearman for ridge/lgbm/cnn at n=100k. Consistent across architectures, so a property of the task. |

Status values: not started / in progress / blocked / done

---

## Definition-of-done checklist

- [x] evaluate.py FROZEN 2026-09-04 (Day 1) — sha256 a5ed94c66327340c...
      Any change to this file invalidates every number produced before it.
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
- Compute: WORKING. RTX 5090 (32GB) reachable as `ssh gpu`, key auth, torch 2.11.0+cu128,
  sm_120 confirmed, GPU matmul verified. Machine is WSL2 on Windows - the Windows host
  sleeping will kill runs, so tmux protects against client disconnect only.
  DISK IS TIGHT: 26GB free of 1TB (98% full) after the torch install.
- (superseded) RTX 5090 (32GB) available via SSH on a family machine. Ample for
  DNABERT-2 and NT-2.5B under LoRA. Setup notes: Blackwell needs CUDA 12.8+ and a
  matching recent PyTorch; DNABERT-2's pinned triton/flash-attn will likely need
  disabling on a card this new. Duration of access not yet confirmed.

---

## Blockers

None yet.

---

## Decision log

(append only, one line each: date — decision — reason)

- 2026-09-04 — Read-count filter dropped as impossible, not skipped — de Boer published no per-sequence read counts in either distributed file. Training data used unfiltered; the provided atLeast100Counts file (>=100 reads) is the test set. The authors' ~24% noise estimate quantifies precisely this. Method comparisons unaffected; absolute data-efficiency numbers are pessimistic. Must appear in Limitations.
- 2026-09-04 — Label description corrected in PLAN.md and CLAUDE.md: values are expression level (18-bin weighted average, ~1.5-16.7), not log2(YFP/RFP). Spearman unaffected; affects Pearson/R^2 interpretation and MSE scaling. Training labels integer, test labels continuous.
- 2026-09-04 — Sequence convention frozen: trim 17bp 5' + 13bp 3' constant scaffold (confirmed empirically by base conservation AND independently by the GEO series design). Keep all sequences, no length filtering, pad to 95 for CNN and TF-MoDISco. Rationale: indels concentrate in homopolymer runs, so length-filtering would systematically deplete the poly-A motif that D3 is meant to recover.
- 2026-09-04 — WRONG, corrected same day: train and test are NOT independent. 6,651 of 9,982 test sequences (66.6%) appear verbatim in training. Cause: the high-quality pTpA_3E5 library is a dilution of the same pool as pTpA_1E8, not a separate synthesis (GEO growth protocol). Those 6,651 training rows (0.021%) are excluded from all subsamples. Test set untouched at 9,982.
- 2026-09-04 — Clustering measured, not assumed: 200k pool gives 199,680 clusters, 99.84% singletons, largest cluster 2 (320 near-duplicate pairs, consistent with PCR/sequencing error). Cluster split is therefore near-identical to a random split. Both are kept and reported per PLAN.md; the coincidence is itself a finding about random-sequence libraries. MMseqs2 run on the 200k working pool rather than all 31.3M - confirming 31M singletons would cost hours and change nothing.
- 2026-09-07 — HEADLINE RESULT (DNABERT-2 vs baselines, primary test, both splits, 3 seeds). Pretraining wins exactly where measurements are scarce and the advantage decays as data grows: n=100 0.563 vs cnn 0.521 vs ridge 0.422; n=300 0.702/0.639/0.563; n=1,000 0.771/0.728/0.679; n=3,000 0.801/0.794/0.756. The from-scratch CNN then overtakes at n=10,000 (0.838 vs 0.831) and ties at 30,000, before DNABERT-2 edges ahead again at 100,000 (0.902 vs 0.897). CLAUDE.md's expectation that a CNN beats the LM at high n is confirmed, in a window rather than permanently.
- 2026-09-07 — In the units PLAN.md's question actually asks: pretraining is worth ~2x fewer measurements than a from-scratch CNN and ~4x fewer than k-mer ridge in the scarce regime. DNABERT-2 on 300 measurements matches a CNN on 700 and ridge on 1,390; on 1,000 it matches a CNN on 2,067 and ridge on 4,352.
- 2026-09-07 — WARMUP DEFECT FOUND AND FIXED, and it changed the headline. PLAN.md specifies 50 warmup steps. At n=100 there are only 3 optimiser steps per epoch, so the learning rate was still ramping at epoch 17; two of three seeds early-stopped at 6 and 8 epochs and never trained at the intended rate (rho 0.106 and 0.276 against 0.614 for the seed that got past warmup). Fix: warmup = min(50, 10%% of planned steps), plus a guard forbidding early stopping until warmup has completed. n=100 mean went 0.332 -> 0.584 and seed spread 0.508 -> 0.078. Only n<=300 is affected; every run at n>=1,000 keeps PLAN.md's exact 50 steps and is unchanged, so no baseline re-run was needed. The 48 superseded flat-warmup rows were dropped from runs.csv. This is a deviation from PLAN.md and must be stated in the write-up.
- 2026-09-07 — D4a ladder with DNABERT-2: native 0.813 / spikein 0.902, gap +0.089 - the best native score and the smallest random->natural drop of the three model families (ridge +0.094, cnn +0.103).
- 2026-09-07 — DNABERT-2 loads and runs, after three separate incompatibilities. (a) Its remote code is from the transformers 4.28 era and fails outright on transformers 5.x (meta-device errors); pinned transformers==4.44.2 / peft==0.13.2. (b) Auto* classes refuse it even on 4.44 - the remote BertModel declares transformers' built-in BertConfig as config_class while the loaded config is the remote one, so Auto.register() rejects the mismatch; fixed by fetching the class via get_class_from_dynamic_module and attaching our own regression head. (c) The bundled triton flash-attention calls tl.dot(trans_b=True), removed from modern triton; bert_layers.py has a standard-attention fallback guarded on `flash_attn_qkvpacked_func is None`, so that module global is set to None at load time. Costs nothing - sequences are ~19 BPE tokens. All three documented in src/dnabert.py.
- 2026-09-07 — CLAUDE.md's LoRA trap CONFIRMED empirically, not assumed. DNABERT-2's Linear leaf names are exactly ['Wqkv', 'dense', 'gated_layers', 'wo']. There is no 'query' and no 'value', so the repo's default lora_target_modules would match zero modules and produce a flat loss that looks like a model failure. LoRA targets Wqkv (attention is fused).
- 2026-09-07 — Nucleotide Transformer will use the 500M checkpoint, not 2.5B. Reason: 26GB free disk on the GPU box; a ~10GB checkpoint leaves no margin. Deviation from the implicit reading of PLAN.md D1; record in Limitations alongside the tuning budget.
- 2026-09-04 — DAY 3 COMPLETE. All three baselines across 7 sizes x 3 seeds x 2 splits x 4 eval sets (504 runs). Primary test, mean Spearman: CNN 0.521/0.728/0.838/0.897 at n=100/1k/10k/100k; ridge 0.422/0.679/0.806/0.865; LightGBM 0.066/0.406/0.784/0.839. CNN wins at EVERY n; ridge second everywhere; LightGBM last everywhere.
- 2026-09-04 — Q1 falsified more completely than first recorded: the prediction that ridge wins at n=100 and n=1,000 is wrong - the CNN beats it at every size. What survives is a weaker, more precise claim: ridge is more RELIABLE at low n (seed sd 0.034 vs CNN 0.083 at n=100). CNN has the better mean and the worse worst case. Record both in Outcomes.
- 2026-09-04 — LightGBM's n=100 result (0.066) is partly a tuning-budget artefact: early stopping is judged on a 20-sequence validation set. Honest claim is 'gradient boosting with this tuning budget fails at n=100', not 'trees cannot do this'. Checkpoint 2 item B addresses it.
- 2026-09-04 — Split equivalence quantified: cluster and random splits agree to <0.005 at n>=1,000; largest discrepancy 0.049 at n=100 where seed variance dominates. (An earlier note claiming agreement to 3dp overall was too strong.)
- 2026-09-04 — Insert length weakly correlates with expression: spearman +0.092 (primary test), +0.056 (train). Mean label 9.12 at exactly 80bp vs 8.03 for shorter inserts. k-mer counts are raw (not length-normalised) so models can see length indirectly. Too small to explain performance of ~0.87, but must be stated in the write-up as a minor confound. Plausibly real biology (fewer bases, fewer motifs; indels disrupt motifs) rather than artefact.
- 2026-09-04 — Ridge baseline complete and VERIFIED. Negative control (shuffled training labels) gives rho -0.06..+0.10 vs +0.68..+0.87 with real labels: no leakage through features, indices or evaluation. Four internal checks pass: cluster and random splits agree to 3dp (0.8647 vs 0.8644 at n=100k); primary beats secondary by exactly the margin label noise predicts (0.865 vs 0.738); the Native80 spike-in control matches the primary test set (0.874 vs 0.865) despite different files/experiments; native sequences sit 0.09 below matched random controls.
- 2026-09-04 — Q1 PREDICTION FALSIFIED (do not edit PREDICTIONS.md - record in Outcomes at write-up). Predicted ridge 0.20 at n=100 and a ~0.70 plateau. Actual: 0.426 at n=100, 0.683 at n=1,000, 0.865 at n=100,000 and still climbing. The task is far more learnable from 6-mer features at low n than expected. Sets a high bar for the LMs on Day 4.
- 2026-09-04 — precision@100 <= 0.16 at every n despite Spearman up to 0.865. Bulk ranking is good, extreme-tail ranking is poor. Relevant to any design application; invisible if only correlation is reported.
- 2026-09-04 — Transfer ladder (D4a) added as approved scope change: evaluate on Native80 (62,897 real yeast 80-mers, same scaffold/assay/organism) to isolate the random->natural shift before the two-variable Yarrowia jump. Zero cost - 4MB file already on GEO.
- 2026-09-04 — Noise ceiling MEASURED, not estimated: rep1 vs rep2 Spearman 0.980 (random spike-ins), 0.934 (native). Distinct from de Boer's ~24% estimate, which describes the noisy 1E8 training library. Both belong on scaling curves.
- 2026-09-04 — Native80 overlap checked at INSERT level after an initial full-sequence check gave a false 'zero overlap' (constructs are 120bp with 20bp flanks, vs 110bp/17+13bp for pTpA). Correct result: all 8,027 random spike-ins are already in the primary test set (matched control, not new data); 62,897 native sequences are new; zero overlap with the training pool.
- 2026-09-04 — evaluate.py frozen with three additions beyond PLAN.md's spec: test-set fingerprint (makes non-comparable results detectable - this is the guard against the leakage class of bug), RMSE/MAE (absolute error, not just correlation), precision@k (tail accuracy; a model at rho=0.82 had only 3/10 of its top-10 correct).
- 2026-09-04 — Prior June 2026 attempt at this project (different scope: synthetic data augmentation) deleted after confirming its data files were byte-identical. Git history preserved as notebooks/june2026_prior_attempt.bundle.

- 2026-09-04 — PREDICTIONS.md answers Q1 only; Q2-Q7 recorded as "no prediction" — no prior modelling experience to base them on, and guesses carrying no information would dilute the one prediction that does carry a signal. Q1 is the headline scaling curve, so the leakage tripwire (a result far above ~0.20 Spearman at n=100 indicates a broken split, not success) is preserved where it matters. Not to be back-filled.
- 2026-09-04 — .gitignore corrected so data/raw/.gitkeep and data/processed/.gitkeep are tracked — bootstrap ignored the directories wholesale, so a fresh clone would not recreate them, breaking one-command reproduction.

---

## Schedule slip

**Days used vs planned:** 1 / 1
**Cut so far:** none

Cut order if behind: D4 (Yarrowia) first. Never cut seeds, baselines, or write-up.
