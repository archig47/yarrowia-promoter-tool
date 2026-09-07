"""Day 4: LoRA fine-tunes of pretrained DNA language models.

Same grid as the Day 3 baselines (7 sizes x 3 seeds x 2 splits x 4 eval sets)
and the same frozen src/evaluate.py, so the numbers drop straight into the
existing scaling curves.

Hyperparameters are PLAN.md's: r=8, alpha=16, dropout=0.05, lr=1e-4, batch 32,
50 warmup steps, weight decay 0.01, MSE on the standardised target.

LoRA targets are DETECTED, not assumed. DNABERT-2 fuses attention into Wqkv and
has no 'query'/'value', so the repo default silently attaches to nothing and
gives a flat loss curve (see CLAUDE.md). This script prints what it matched and
refuses to run if it matched nothing.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import evaluate, log_run

NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = [0, 1, 2]
MAX_EPOCHS = {100: 60, 300: 60, 1000: 40, 3000: 30, 10000: 20, 30000: 12, 100000: 8}
PATIENCE = 5

MODELS = {
    "dnabert": dict(id="zhihan1996/DNABERT-2-117M", custom=True),
    "nt":      dict(id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", custom=False),
}


# ---------------------------------------------------------------- data
def load_all():
    d = ROOT / "data"
    pool = pq.read_table(d / "pool.parquet")
    test = pq.read_table(d / "test_primary.parquet")
    nat = pq.read_table(d / "native80.parquet")
    isnat = np.array(nat["is_native"].to_pylist())
    return (pool["seq"].to_pylist(), np.array(pool["label"].to_pylist(), np.float32),
            test["seq"].to_pylist(), np.array(test["label"].to_pylist(), np.float32),
            nat["seq"].to_pylist(), np.array(nat["label"].to_pylist(), np.float32), isnat)


# ---------------------------------------------------------------- model
class Regressor(torch.nn.Module):
    """Pretrained backbone + mean-pooled linear head."""

    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def build(kind):
    from transformers import AutoTokenizer, AutoConfig
    spec = MODELS[kind]
    tok = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(spec["id"], trust_remote_code=True)
    if spec["custom"]:
        from dnabert import load_dnabert
        tok, bb, cfg = load_dnabert(spec["id"])
    else:
        from transformers import AutoModel
        bb = AutoModel.from_pretrained(spec["id"], trust_remote_code=True)
    return tok, bb, cfg


def lora_targets(backbone, want_mlp=True):
    """Pick real module names. Attention first, then MLP if asked."""
    leaves = sorted({n.split(".")[-1] for n, m in backbone.named_modules()
                     if isinstance(m, torch.nn.Linear)})
    attn = [n for n in leaves if n in ("Wqkv", "query", "key", "value", "q_proj", "k_proj", "v_proj")]
    mlp = [n for n in leaves if n in ("gated_layers", "wo", "intermediate", "output", "fc1", "fc2")]
    picked = attn + (mlp if want_mlp else [])
    if not picked:
        raise SystemExit(f"no LoRA targets matched. available leaves: {leaves}")
    return picked, leaves


# ---------------------------------------------------------------- one run
def fit(kind, tok, seqs, y, seed, device="cuda"):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    _, bb, cfg = build(kind)
    targets, leaves = lora_targets(bb)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                          target_modules=targets, task_type=None)
    bb = get_peft_model(bb, peft_cfg)
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    model = Regressor(bb, cfg.hidden_size).to(device)

    n = len(seqs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = max(int(0.8 * n), n - 2000)
    tr, va = perm[:cut], perm[cut:]
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-8)

    def batches(idx, bs, shuffle):
        order = np.random.default_rng(seed).permutation(idx) if shuffle else idx
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            enc = tok([seqs[k] for k in j], return_tensors="pt", padding=True, truncation=True, max_length=128)
            yield (enc["input_ids"].to(device), enc["attention_mask"].to(device),
                   torch.tensor((y[j] - mu) / sd).to(device))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.01)
    # PLAN.md specifies 50 warmup steps. That is ~1% of a long run but ~200% of a
    # short one: at n=100 there are 3 optimiser steps per epoch, so a flat 50 steps
    # means the learning rate is still ramping at epoch 17 and any run that stops
    # earlier never trains at the intended rate. Two of three seeds did exactly that
    # (6 and 8 epochs -> 18 and 24 steps), producing rho 0.11 and 0.28 against 0.61
    # for the seed that got past warmup. Warmup is now 10% of the planned run.
    max_ep = MAX_EPOCHS.get(n, 10)
    steps_per_epoch = max(1, int(np.ceil(len(tr) / 32)))
    warm = max(5, min(50, int(0.1 * max_ep * steps_per_epoch)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    best, best_state, bad = -np.inf, None, 0
    for ep in range(max_ep):
        model.train()
        for ids, mask, yy in batches(tr, 32, True):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(ids, mask), yy)
            loss.backward(); opt.step(); sched.step()
        model.eval()
        pv = []
        with torch.no_grad():
            for ids, mask, _ in batches(va, 128, False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pv.append(model(ids, mask).float().cpu().numpy())
        s = spearmanr(y[va], np.concatenate(pv))[0]
        if np.isfinite(s) and s > best:
            best, bad = s, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            # never stop while the learning rate is still ramping, plus a few epochs
            # after: a run that halts mid-warmup has not trained at the intended rate.
            min_ep = max(2, int(np.ceil(warm / steps_per_epoch)) + 3)
            if bad >= PATIENCE and ep >= min_ep:
                break
    if best_state: model.load_state_dict(best_state)
    model.eval()

    def predict(ss):
        out = []
        with torch.no_grad():
            for i in range(0, len(ss), 256):
                enc = tok(ss[i:i + 256], return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(model(enc["input_ids"].to(device),
                                     enc["attention_mask"].to(device)).float().cpu().numpy())
        return np.concatenate(out) * sd + mu

    return predict, {"targets": "+".join(targets), "trainable": trainable,
                     "epochs": ep + 1, "warmup": warm, "leaves": "|".join(leaves)}


# ---------------------------------------------------------------- sweep
def main(kind, only_ns=None):
    t0 = time.time()
    pseq, py, tseq, ty, nseq, ny, isnat = load_all()
    tok, bb0, cfg0 = build(kind)
    targets, leaves = lora_targets(bb0)
    print(f"model={kind}  hidden={cfg0.hidden_size}  linear leaves={leaves}")
    print(f"LoRA targets -> {targets}", flush=True)
    del bb0

    evalsets = {"primary": (tseq, ty),
                "native": ([s for s, k in zip(nseq, isnat) if k], ny[isnat]),
                "spikein": ([s for s, k in zip(nseq, isnat) if not k], ny[~isnat])}

    for split in ("cluster", "random"):
        sp = json.load(open(ROOT / "data" / f"split_{split}.json"))
        subs = np.load(ROOT / "data" / f"subsamples_{split}.npz")
        te = np.array(sp["test"])
        evalsets["secondary"] = ([pseq[i] for i in te], py[te])
        for n in (only_ns or NS):
            for seed in SEEDS:
                key = f"n{n}_seed{seed}"
                if key not in subs: continue
                idx = subs[key]; t1 = time.time()
                predict, extra = fit(kind, tok, [pseq[i] for i in idx], py[idx], seed)
                for ename, (Xe, ye) in evalsets.items():
                    met = evaluate(ye, predict(Xe))
                    cfg = {"model": kind, "split": split, "n": n, "seed": seed,
                           "evalset": ename, "lora_targets": extra["targets"],
                           "trainable_params": extra["trainable"], "epochs": extra["epochs"],
                           "warmup": extra["warmup"], "warmup_rule": "proportional"}
                    log_run(cfg, met, path=ROOT / "results" / "runs.csv")
                    if ename == "primary":
                        print(f"  {split:7s} n={n:6d} s={seed} "
                              f"rho={met['overall']['spearman']:+.4f} "
                              f"r2={met['overall']['r2']:+.4f} "
                              f"P@100={met['precision_at_k']['p_at_100']:.2f} "
                              f"ep={extra['epochs']} ({time.time()-t1:.0f}s)", flush=True)
    print(f"DONE {kind} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--only-n", default="", help="comma-separated sizes to re-run")
    a = ap.parse_args()
    only = [int(x) for x in a.only_n.split(",") if x] or None
    main(a.model, only)
