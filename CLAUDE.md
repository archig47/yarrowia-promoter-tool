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
- Labels are log2(YFP/RFP). Report Spearman as the headline metric.
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
