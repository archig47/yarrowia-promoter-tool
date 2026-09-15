"""D4: can multi-species genomic pretraining cross the species barrier?

The ridge and CNN transfer attempts all failed, with a clear mechanism: what
S. cerevisiae teaches (GC-rich means strong, rho +0.604 there) is worthless in
Yarrowia (rho -0.032). But those models only ever saw S. cerevisiae.

DNABERT-2 is different in two ways that matter here:
  1. It was pretrained on genomes from MANY species, so it may carry structure
     relevant to Yarrowia that an S. cerevisiae-only model cannot.
  2. It is length-flexible. Ridge trained on 80bp cannot use a 250bp window
     without a scale artefact; DNABERT-2 tokenises whatever it is given, so it
     can exploit the 250bp window that is empirically best for Yarrowia.

Three arms, all scored on the same held-out Yarrowia promoters:
  direct   DNABERT-2 + LoRA on k Yarrowia examples. Genomic pretraining only,
           no S. cerevisiae stage. Tests (1) above.
  two_stage DNABERT-2 + LoRA on 100k S. cerevisiae, THEN continue on k Yarrowia.
           Tests whether the S. cerevisiae stage helps or, as the GC mechanism
           predicts, actively hurts.
  ridge    from-scratch k-mer ridge on the same k, as the reference to beat.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_lora import build, lora_targets, MODELS
from evaluate import evaluate, log_run

KS = [50, 100, 300, 1000, 3000]
SEEDS = [0, 1, 2]
TEST_FRAC = 0.25
WINDOW = 250
SC_N = 100_000
DEV = "cuda"


class Reg(torch.nn.Module):
    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone, self.head = backbone, torch.nn.Sequential(
            torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 1))

    def forward(self, ids, mask):
        o = self.backbone(input_ids=ids, attention_mask=mask)
        h = o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def make_model(seed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    tok, bb, cfg = build("dnabert")
    targets, _ = lora_targets(bb, "dnabert")
    bb = get_peft_model(bb, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                       bias="none", target_modules=targets, task_type=None))
    return tok, Reg(bb, cfg.hidden_size).to(DEV)


def train(model, tok, seqs, y, seed, max_epochs, bs=32, lr=1e-4, patience=5):
    n = len(seqs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); cut = max(int(0.8 * n), n - 500)
    tr, va = perm[:cut], perm[cut:]
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-8)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    spe = max(1, int(np.ceil(len(tr) / bs)))
    warm = max(5, min(50, int(0.1 * max_epochs * spe)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    def batches(idx, b, shuffle):
        order = np.random.default_rng(seed).permutation(idx) if shuffle else idx
        for i in range(0, len(order), b):
            j = order[i:i + b]
            e = tok([seqs[k] for k in j], return_tensors="pt", padding=True,
                    truncation=True, max_length=192)
            yield e["input_ids"].to(DEV), e["attention_mask"].to(DEV), \
                  torch.tensor(((y[j] - mu) / sd).astype(np.float32)).to(DEV)

    best, state, bad = -np.inf, None, 0
    min_ep = max(2, int(np.ceil(warm / spe)) + 3)
    for ep in range(max_epochs):
        model.train()
        for ids, m, yy in batches(tr, bs, True):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(ids, m), yy)
            loss.backward(); opt.step(); sch.step()
        model.eval(); pv = []
        with torch.no_grad():
            for ids, m, _ in batches(va, 128, False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pv.append(model(ids, m).float().cpu().numpy())
        s = spearmanr(y[va], np.concatenate(pv))[0] if va.size else -np.inf
        if np.isfinite(s) and s > best:
            best, bad = s, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and ep >= min_ep: break
    if state: model.load_state_dict(state)
    model.eval()
    return mu, sd, ep + 1


def predict(model, tok, seqs, mu, sd, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            e = tok(seqs[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=192)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(e["input_ids"].to(DEV),
                                 e["attention_mask"].to(DEV)).float().cpu().numpy())
    return np.concatenate(out) * sd + mu


def main(arms):
    import pyarrow.parquet as pq
    t0 = time.time()
    recs = json.load(open(ROOT / "data/yarrowia_rnaseq_benchmark.json"))
    yseq = [r["seq"][-WINDOW:] for r in recs]
    yy = np.log10(np.array([r["tpm"] for r in recs], float) + 1.0)
    n = len(yy)
    print(f"Yarrowia: {n:,} promoters, {WINDOW}bp window", flush=True)

    pool = pq.read_table(ROOT / "data/pool.parquet")
    pseq = pool["seq"].to_pylist()
    py = np.array(pool["label"].to_pylist(), np.float32)
    sc_idx = np.load(ROOT / "data/subsamples_cluster.npz")[f"n{SC_N}_seed0"]
    sc_seq = [pseq[i] for i in sc_idx]; sc_y = py[sc_idx]

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n); n_te = int(TEST_FRAC * n)
        te, avail = perm[:n_te], perm[n_te:]
        te_seq = [yseq[i] for i in te]; te_y = yy[te]

        # the S. cerevisiae stage is trained ONCE per seed and reused for every k
        sc_state = None
        if "two_stage" in arms:
            cache = ROOT / "results" / f"stage1_dnabert_seed{seed}.pt"
            tok, model = make_model(seed)
            if cache.exists():
                blob = torch.load(cache, map_location=DEV, weights_only=False)
                model.load_state_dict(blob["state"], strict=False)
                print(f"  [seed {seed}] stage 1 loaded from cache "
                      f"(zero-shot rho was {blob['zeroshot']:+.4f})", flush=True)
                results.setdefault("two_stage_zeroshot", {}).setdefault(0, []).append(blob["zeroshot"])
            else:
                print(f"  [seed {seed}] stage 1: LoRA on {SC_N:,} S. cerevisiae ...", flush=True)
                mu, sd, ep = train(model, tok, sc_seq, sc_y, seed, max_epochs=8)
                p0 = predict(model, tok, te_seq, mu, sd)
                zs = float(spearmanr(te_y, p0)[0])
                print(f"    stage-1 zero-shot on Yarrowia: rho={zs:+.4f} "
                      f"({ep} epochs, {time.time()-t0:.0f}s)", flush=True)
                # save only trainable tensors - the frozen backbone is reloaded
                trainable = {k for k, v in model.named_parameters() if v.requires_grad}
                keep = {k: v.detach().cpu() for k, v in model.state_dict().items()
                        if k in trainable or k.startswith("head.")}
                torch.save({"state": keep, "zeroshot": zs}, cache)
                print(f"    cached stage 1 ({sum(v.numel() for v in keep.values())/1e6:.2f}M tensors)", flush=True)
                results.setdefault("two_stage_zeroshot", {}).setdefault(0, []).append(zs)
            sc_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            del model; torch.cuda.empty_cache()

        for k in KS:
            tr = avail[:k]
            tr_seq = [yseq[i] for i in tr]; tr_y = yy[tr]
            me = {50: 60, 100: 60, 300: 40, 1000: 25, 3000: 15}[k]

            if "direct" in arms:
                tok, model = make_model(seed)
                mu, sd, ep = train(model, tok, tr_seq, tr_y, seed, max_epochs=me)
                r = spearmanr(te_y, predict(model, tok, te_seq, mu, sd))[0]
                results.setdefault("direct", {}).setdefault(k, []).append(float(r))
                print(f"  seed={seed} k={k:5d} direct    rho={r:+.4f} ep={ep} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                del model; torch.cuda.empty_cache()

            if "two_stage" in arms:
                tok, model = make_model(seed)
                model.load_state_dict(sc_state, strict=False)
                mu, sd, ep = train(model, tok, tr_seq, tr_y, seed, max_epochs=me)
                r = spearmanr(te_y, predict(model, tok, te_seq, mu, sd))[0]
                results.setdefault("two_stage", {}).setdefault(k, []).append(float(r))
                print(f"  seed={seed} k={k:5d} two_stage rho={r:+.4f} ep={ep} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                del model; torch.cuda.empty_cache()

    summ = {a: {str(k): {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4)}
                for k, v in ks.items()} for a, ks in results.items()}
    json.dump(summ, open(ROOT / "results" / "transfer_dnabert.json", "w"), indent=2)
    print("\n=== mean over seeds ===")
    for a, ks in summ.items():
        print(f"  {a}: " + "  ".join(f"k={k}:{v['mean']:+.3f}" for k, v in sorted(ks.items(), key=lambda x: int(x[0]))))
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="direct,two_stage")
    main(ap.parse_args().arms.split(","))
