# yeast-promoter-lm

Two-week solo research project. Read `STATE.md` first — it says where the work stands.
`PLAN.md` is the fixed spec; do not silently change scope against it.

## What this is

Testing how few experimental measurements are needed before sequence-based promoter
strength prediction becomes useful, using de Boer et al. 2020 yeast data (GSE104878)
as a simulator of data scarcity. Four deliverables: scaling curves, active-learning
selection curves, motif recovery via ISM, and transfer to a curated *Yarrowia
lipolytica* benchmark.

## Invariants — do not break these

- `src/evaluate.py` is **frozen** after Day 2. Every reported number comes from it.
  If it needs changing, stop and flag it; changing it invalidates prior results.
- Split index files in `data/splits/` are **frozen**. Never regenerate them.
- `PREDICTIONS.md` is **append-only** and was written before results existed.
  Never edit earlier entries.
- Primary test set is the de Boer **high-quality pTpA/glucose set (n ~ 9,982)**.
  Secondary is a cluster-based held-out split of training data. Report both.
- Three seeds minimum for any n below 10,000. Single-seed results are not results.
- Labels are de Boer *expression level* (EL), a weighted average over the 18 FACS bins,
  range ~1.5-16.7. NOT log2(YFP/RFP) despite the paper's phrasing — corrected 2026-09-04
  from the data. Monotone in the underlying ratio, so Spearman is unaffected; Pearson/R^2
  interpretation and MSE loss scaling are. Standardise the target before training.
  Report Spearman as the headline metric.
- Training labels are integers (bin indices); high-quality test labels are continuous.
- Constant scaffold: 17bp 5' TGCATTTTTTTCACATC, 13bp 3' GGTTACGGCTGTT, confirmed both
  empirically and from the GEO series design. Trim it; it carries zero information.
  Inserts are 80bp in 80.5% of cases, 65-95bp overall. Pad, never filter by length —
  indels concentrate in homopolymers and filtering would deplete the poly-A motif.
- Yarrowia results: rank correlation only, with bootstrap CIs. Assays differ across
  source papers, so absolute values are not comparable.

## Conventions

- Python, PyTorch, HuggingFace Transformers + PEFT
- All experiment configs are explicit dicts logged to `results/runs.csv`, one row per run
- Figures regenerate from `results/` — never hand-edit a figure
- Long runs must be resumable; assume the machine will be interrupted
- Commit `STATE.md` at the end of every session, message = that day's summary

## Known traps

- DNABERT-2 needs `trust_remote_code=True` and `einops`; it complains about
  triton/flash-attention. Its own repo defaults `lora_target_modules` to `"query,value"`,
  but the architecture uses a fused `Wqkv`. Print `model.named_modules()` and target
  what is actually there. Wrong targets produce a flat loss curve.
- de Boer labels come from FACS sorting into 18 bins by YFP/RFP ratio. Low-read
  sequences have very noisy labels. Filtering by read count is not optional.
- The authors estimate the training data carries ~24% noise. Do not expect r^2 much
  above 0.75 on ordinary training data — that is the ceiling, not a model failure.
- These are random 80mers, so genomic pretraining may not transfer. A CNN beating the
  LM at high n is an expected result, not a bug.

## Working style

- Say when a result looks wrong rather than explaining it away.
- Prefer the smallest experiment that answers the question.
- If something takes longer than its budget in `PLAN.md`, say so rather than absorbing it.
- Do not re-plan the project. If you think `PLAN.md` is wrong, say so explicitly.
