# STATE

Living handoff document. Update at the end of every working session.
Keep it under one page — status board, not a log.

**Last updated:** 2026-09-11
**Day:** 4 of 14 in progress. Days 1-3 complete (Days 1-2 were finished early, both on Day 1).
**Calendar:** started 2026-09-04; today 2026-09-11.
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
Day 4 mostly done. Four of five models complete and committed (672 runs): ridge,
LightGBM, CNN, DNABERT-2. Headline result established - genomic pretraining
substitutes for measurements in the scarce regime, worth ~2x fewer measurements
than a from-scratch CNN and ~4x fewer than k-mer ridge. Figures regenerate from
results/runs.csv.

BLOCKED: the NT-500M sweep was launched on the GPU box on 2026-09-07 and the
machine is no longer reachable - ssh gpu points at 192.168.1.166, which resolves
only on that local network, and Tailscale is not installed on the laptop. Its
results (if the run survived) are in ~/yeast-promoter-lm/results/runs.csv on that
box. Recover by rejoining that network, or install Tailscale for permanent access.

Next after NT: Day 5-6 active learning (D2).

---

## Deliverable status

| ID | Deliverable | Status | Notes |
|----|-------------|--------|-------|
| D1 | Scaling curves | COMPLETE | All 5 models x 7 n x 3 seeds x 2 splits x 4 eval sets = 840 runs. Figures regenerate from results/. |
| D2 | Selection curves | done (negative) | 48 runs. NO strategy beats random; uncertainty is consistently worse. A chosen 300 is worth ~219 random. fig3 generated. Re-run diversity with LM embeddings when the GPU returns. |
| D3 | Motif recovery | COMPLETE | Both models. CNN: 6 patterns, all matched. DNABERT-2: 11 patterns, 10 matched, incl. REB1 (r=0.95) which the CNN missed entirely, plus 5 repressive motifs vs the CNN's 1. |
| D4 | Yarrowia transfer | REOPENED - benchmark fixed | 81 promoters curated. Zero-shot fails AND the benchmark is not learnable in-domain: Yarrowia-trained, Yarrowia-tested CV rho = -0.04 (proximal 80bp) to +0.10 (full 1000bp), all p>0.1. SOLVED 09-15 by building a second, 6,045-promoter benchmark from genome-wide RNA-seq (Lubuta et al. 2019): in-domain CV rho = +0.390 at 250bp. Transfer is now evaluable; methods being tested. |
| D4a | Transfer ladder | all 5 models | random->natural costs +0.089 to +0.105 Spearman across all model families at n=100k. Consistent, so a property of the task. DNABERT-2 has the best native score (0.813) and smallest drop. |

Status values: not started / in progress / blocked / done

---

## Definition-of-done checklist

- [x] evaluate.py FROZEN 2026-09-04 (Day 1) — sha256 a5ed94c66327340c...
      Any change to this file invalidates every number produced before it.
- [x] High-quality test set secured — exactly 9,982 rows, no fallback needed
- [x] Three seeds below n = 10,000 — in fact 3 seeds at every n, both splits
- [x] PREDICTIONS.md written before any modelling
- [x] Baselines reported honestly in README first paragraph — CNN beats DNABERT-2 at n=10k/30k and beats NT almost everywhere; NT falls below ridge at n=300
- [x] Limitations section written — 12 items in README.md
- [x] One-command reproduction works from clean clone — `make all` builds venv and regenerates all figures from the committed 840 runs
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

None. (GPU access resolved 2026-09-11 via ZeroTier - see decision log.)

---

## Decision log

(append only, one line each: date — decision — reason)

- 2026-09-15 — D4 REOPENED AND UNBLOCKED. The 81-promoter reporter benchmark was too small to learn from, so a second benchmark was built from genome-wide RNA-seq: Lubuta et al. 2019 (G3, FigShare 10.25387/g3.8335217) give TPM for 8,605 Yarrowia genes in glucose chemostats. After mapping YALI0 ids to the CLIB122 annotation and keeping TPM>=1, that is 6,045 promoters with 1000bp upstream - ~75x the reporter set. IT IS LEARNABLE: 5-fold CV ridge rho = +0.325 (proximal 80bp), +0.390 (proximal 250bp), +0.345 (full 1000bp), against permutation nulls of 0.00 +/- 0.03. Cross-species transfer is therefore EVALUABLE and the fine-tuning arm is back on.
- 2026-09-15 — INDEPENDENT CONFIRMATION OF THE WINDOW-LENGTH ARGUMENT: 250bp (rho 0.390) beats both 80bp (0.325) and 1000bp (0.345). Blazeck et al. state a minimal Y. lipolytica promoter spans 130-260bp upstream of the ATG; the empirical optimum lands inside that range. So the 80bp window imposed by de Boer's training data IS a real handicap for this species, now measured rather than argued, and 1000bp adds noise beyond the functional region.
- 2026-09-15 — CAVEAT on the RNA-seq benchmark, to be stated prominently: mRNA abundance is NOT promoter strength. It is confounded by transcript stability and reflects promoters in native chromatin rather than a fixed reporter scaffold. Cross-check on the 80 promoters present in both benchmarks: reporter strength vs TPM spearman +0.428 - related but far from identical. BOTH benchmarks are kept and reported; the reporter set is the cleaner measurement, the RNA-seq set is the only one large enough to learn from.
- 2026-09-11 — D4 DECISIVE NEGATIVE CONTROL: the benchmark is not learnable in-domain. Ridge trained on Yarrowia itself and cross-validated on Yarrowia (20 repeats of 5-fold, alpha chosen inside training folds only) gives rho -0.037 on the proximal 80bp, +0.096 on full 1000bp k-mer counts, +0.035 on k=4 - none significant against per-feature-set permutation nulls (p=0.30, 0.12, 0.32). So the failure of cross-species transfer CANNOT be attributed to the species barrier, the window, or the models: there is no learnable in-domain signal to transfer to. The fine-tuning arm PLAN.md specifies is therefore deliberately not run - LoRA on 50 of these would produce an uninterpretable number. This is the project's own premise demonstrated: 81 promoters is below the threshold at which sequence-to-expression learning becomes possible, and fig1 shows where that threshold sits even with clean data.
- 2026-09-11 — D4 CAVEATS that must appear in the write-up: (a) we do not know what promoter region Zhang et al. actually cloned into their reporter - we took 1000bp of genomic upstream sequence, and if they cloned a different span our sequences do not correspond to what was measured, which alone could explain the null; (b) endogenous promoter strength depends on chromatin and trans-factors, not only proximal sequence, more so than de Boer's synthetic inserts in a fixed scaffold; (c) 81 examples against 4,096 k-mer features is a hard regime regardless of species.
- 2026-09-11 — D4 BENCHMARK BUILT. 82 Yarrowia lipolytica promoters with measured strengths extracted from the Zhang et al. preprint (Research Square rs-1993869) supplementary Word table; sequences taken from the CLIB122 reference genome (GCF_000002525.2) since the paper identifies promoters by gene, not sequence. 81 of 82 resolved (YALI0C24750g deprecated). Strengths span 2,573-fold. Extraction validated against the paper's own Strong/Medium/Weak labels: spearman +0.912, class counts 15/42/25 matching the published 15/41/25.
- 2026-09-11 — D4 ZERO-SHOT RESULT: NULL, and properly controlled. Pre-specified 80bp proximal window gives rho=-0.008 (p=0.94). Because the literature says that window is wrong (Blazeck et al.: minimal Y. lipolytica promoter spans 130-260bp upstream of ATG; TATA 40-120bp upstream of TSS; UAS at the 5' end; promoters 'enhancer limited'), the 80bp window the models were trained on was then TILED across 1000bp - 47 windows, keeping every model in its trained input regime. Best single window rho=+0.256 at -760bp, consistent across 3 seeds. BUT a permutation test (2000 shuffles, repeating the full window sweep including the max) gives null median max|rho| = 0.251 against observed 0.256, p=0.457. The distal signal is exactly what selection over 47 windows produces from noise. Ridge and CNN also disagreed on its location (-760bp vs -140bp), which was the tell.
- 2026-09-11 — TWO ASSISTANT ERRORS CORRECTED BY TESTING, both worth remembering. (a) Claimed the 80bp proximal window was the right positional analogue of de Boer's construct; the Yarrowia promoter-architecture literature says it excludes the TATA box and all UAS elements. (b) Then claimed the -760bp window carried real signal; the permutation test showed it is chance. Without the permutation test this would have entered the write-up as a finding.
- 2026-09-11 — D3 FIRST RESULT (CNN, n=100,000, cluster split, seed 0). ISM over 4,000 held-out test sequences -> TF-MoDISco -> 6 patterns, all 6 matching a known motif at r>=0.75. Dominant pattern (354 seqlets, r=0.948) is RSC30/RSC3, the RSC chromatin-remodelling complex. poly-A/poly-T recovered in 3 of 6 patterns (r=0.94, 0.88, and within pattern_2) - the motif CLAUDE.md names as ground truth, and the one the Day 1 decision not to length-filter was taken to protect. One NEGATIVE pattern (STP3/STP4, 28 seqlets), i.e. repressive signal.
- 2026-09-11 — D3 SECOND RESULT (DNABERT-2, same 4,000 test sequences, same pipeline) and it gives a biological mechanism for D1's headline. 11 patterns vs the CNN's 6; 10 of 11 matched. Recovers REB1 at r=0.95 - a canonical yeast activator the from-scratch CNN missed completely - plus WAR1 0.96, RSC3 0.98, poly-A in 3 patterns, and FIVE distinct repressive patterns against the CNN's one. So pretraining did not merely make the model more accurate: it found more of the real regulatory grammar, including motifs the from-scratch model never learned. That is a concrete mechanism for the ~2x measurement saving rather than an efficiency claim alone. One DNABERT-2 pattern (neg/pattern_2, 22 seqlets, r=0.715) matches nothing known.
- 2026-09-11 — D3 honest non-findings: the canonical strong yeast activators Rap1, Reb1 and Abf1, which de Boer highlight, were NOT recovered. Six patterns is a partial recovery. Also note poly-A and poly-T are the same motif reverse-complemented and the matcher is strand-agnostic, so they are one finding not two.
- 2026-09-11 — D3 deviations from PLAN.md: (a) ISM over 4,000 test sequences, not ~500 - at 500 only 2 patterns cleared TF-MoDISco's threshold (RSC30, PUT3), because 397 usable sequences yield too few seqlets. ISM costs ~4s per 500 sequences once the model is trained, so this is free and only raises statistical power; the method is unchanged. (b) The poly-A motif is constructed synthetically for matching, since CLAUDE.md names it as ground truth but it is not in YeTFaSCo. (c) Matching uses offset+strand correlation against YeTFaSCo PPMs rather than Tomtom, to avoid a MEME-suite dependency.
- 2026-09-11 — D3 ground truth fully assembled and verified. YeTFaSCo 1.02 All_PFMs (1,887 position frequency matrices) downloaded from yetfasco.ccbr.utoronto.ca; 244 of de Boer's 245 motif IDs match PFM filenames exactly, so no name reconciliation is needed. The tarball is gitignored (extracted files) but the archive is kept in data/raw/.
- 2026-09-11 — D1 COMPLETE, all five models, 840 runs. Primary test, mean Spearman at n=100/300/1k/3k/10k/30k/100k: ridge .422/.563/.679/.756/.806/.832/.865; lgbm .066/.327/.406/.692/.784/.824/.839; cnn .521/.639/.728/.794/.838/.868/.897; dnabert .563/.702/.771/.801/.831/.868/.902; nt .471/.540/.729/.788/.828/.859/.888.
- 2026-09-11 — THE SECOND LM DOES NOT REPLICATE THE FIRST, and this sharpens the headline. NT-500M loses to DNABERT-2 at every n and to the from-scratch CNN at most; at n=300 it falls below k-mer ridge (0.540 vs 0.563). So the defensible claim is not 'pretraining helps' but 'pretraining CAN substitute for measurements, and which pretrained model matters more than its size' - NT is 4x larger and loses. Caveats belonging with it: NT is the 500M checkpoint not 2.5B (disk), its LoRA targets and tokenisation differ because the architecture differs, and part of the gap may be tuning budget rather than the model, exactly as with LightGBM at n=100.
- 2026-09-11 — Measurement equivalence with all five models: DNABERT-2 on 300 measurements matches a CNN on 700, NT on 842, ridge on 1,390. On 1,000 it matches a CNN on 2,067, NT on 2,213, ridge on 4,352.
- 2026-09-11 — GPU access RESOLVED via ZeroTier (network WOLNetwork). The old ssh config pointed at the LAN address 192.168.1.166, reachable only on his home WiFi; the ZeroTier address 192.168.194.73 works from anywhere. Tailscale was not used - ZeroTier was his choice. The NT sweep launched on 09-07 had in fact completed successfully (42 runs, 21,798s) and survived; nothing was lost.
- 2026-09-11 — D2 RESULT IS NEGATIVE and that is the finding: no selection strategy beats random at any budget. Mean Spearman (3 seeds, primary test): random 0.417/0.557/0.676/0.748 at budgets 100/300/1k/3k; uncertainty 0.417/0.517/0.639/0.735; diversity 0.407/0.551/0.667/0.751; hybrid 0.417/0.538/0.656/0.732. Inverting PLAN.md's framing: a chosen 300 is worth ~219 random measurements under uncertainty sampling - actively worse than not choosing. At budget 100 the whole budget is the seed set, so uncertainty and hybrid ARE random draws by construction, hence identical.
- 2026-09-11 — Mechanism for D2, evidenced not assumed: uncertainty sampling introduces SAMPLING BIAS. At budget 1,000 it selects sequences with mean label 10.30 against a pool mean of 9.51, and 23.1% from the extreme deciles against 20% baseline. Training on a distribution that does not match the test distribution costs more than the information gained. (An earlier prediction that label-noise chasing would be the mechanism is only weakly supported - the extreme-decile skew is modest; the mean shift is the clearer signal.) Diversity's near-tie with random is separately explained by the MMseqs2 result: 99.84% singletons means a random library has no under-sampled regions for diversity sampling to find.
- 2026-09-11 — D2 CAVEATS for the write-up: (a) diversity clustered k-mer features, not LM embeddings as PLAN.md specifies, because the GPU box is unreachable - re-run when recovered; (b) selection is one-shot, not iterative (iterative is Checkpoint 2 item C); (c) ridge is the workhorse so that differences reflect selection rather than training noise; (d) seed spreads overlap, so only uncertainty's deficit at budgets 300 and 1,000 is clearly separated; (e) this is a RANDOM library - a structured natural promoter set such as Yarrowia may behave differently, and that limits how far the negative result generalises.
- 2026-09-11 — de Boer Supplementary Table 2 obtained (247 TF motifs, YeTFaSCo IDs) via Europe PMC -> Springer static content; Nature's own page requires auth. Saved as data/raw/deBoer2020_SuppTable2_motifs.xlsx. D3 unblocked. NOTE: it lists motif IDs and names only, NOT position weight matrices - the YeTFaSCo PWMs are still needed before Day 7 matching can run.
- 2026-09-11 — Schedule: days are treated as a work timeline, not calendar days, and NOTHING is cut. Plan-day 4 of 14 reached on calendar day 8, but Days 1-2 were both finished on calendar day 1 and the intent is to keep working above one plan-day per day. PLAN.md's cut rules stay on the books but are deliberately not invoked. Flagged by the assistant, decided by Archita.

- 2026-09-04 — Read-count filter dropped as impossible, not skipped — de Boer published no per-sequence read counts in either distributed file. Training data used unfiltered; the provided atLeast100Counts file (>=100 reads) is the test set. The authors' ~24% noise estimate quantifies precisely this. Method comparisons unaffected; absolute data-efficiency numbers are pessimistic. Must appear in Limitations.
- 2026-09-04 — Label description corrected in PLAN.md and CLAUDE.md: values are expression level (18-bin weighted average, ~1.5-16.7), not log2(YFP/RFP). Spearman unaffected; affects Pearson/R^2 interpretation and MSE scaling. Training labels integer, test labels continuous.
- 2026-09-04 — Sequence convention frozen: trim 17bp 5' + 13bp 3' constant scaffold (confirmed empirically by base conservation AND independently by the GEO series design). Keep all sequences, no length filtering, pad to 95 for CNN and TF-MoDISco. Rationale: indels concentrate in homopolymer runs, so length-filtering would systematically deplete the poly-A motif that D3 is meant to recover.
- 2026-09-04 — WRONG, corrected same day: train and test are NOT independent. 6,651 of 9,982 test sequences (66.6%) appear verbatim in training. Cause: the high-quality pTpA_3E5 library is a dilution of the same pool as pTpA_1E8, not a separate synthesis (GEO growth protocol). Those 6,651 training rows (0.021%) are excluded from all subsamples. Test set untouched at 9,982.
- 2026-09-04 — Clustering measured, not assumed: 200k pool gives 199,680 clusters, 99.84% singletons, largest cluster 2 (320 near-duplicate pairs, consistent with PCR/sequencing error). Cluster split is therefore near-identical to a random split. Both are kept and reported per PLAN.md; the coincidence is itself a finding about random-sequence libraries. MMseqs2 run on the 200k working pool rather than all 31.3M - confirming 31M singletons would cost hours and change nothing.
- 2026-09-07 — NT-500M needed its own loading workaround: it ships a custom EsmConfig and its auto_map exposes AutoModelForMaskedLM but not AutoModel, so AutoModel.from_pretrained fails outright. Load the masked-LM wrapper and take its .esm encoder as the backbone. LoRA targets are now declared per model and validated against the loaded modules, refusing to run if none match - DNABERT-2 ['Wqkv','gated_layers','wo'], NT ['query','key','value','dense']. Both are attention+MLP; the names differ because DNABERT-2 fuses QKV and NT is ESM-style.
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

**Plan days completed:** 4 of 14 (Day 4 partially - NT outstanding)
**Calendar days elapsed:** 8 (2026-09-04 to 2026-09-11)
**Cut so far:** none, and none planned.

Work is deliberately front-loaded rather than paced one plan-day per calendar day:
Days 1-2 were both completed on calendar day 1, and Day 3 the same day. The cut
rules below are therefore NOT being invoked - decision taken 2026-09-11, see log.

Cut order if that changes: D4 (Yarrowia) first. Never cut seeds, baselines, or write-up.
