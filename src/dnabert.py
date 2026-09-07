"""Load DNABERT-2 on a modern stack. Two workarounds, both necessary.

1. Auto* classes refuse this model on transformers 4.4x: its remote BertModel
   declares transformers' built-in BertConfig as config_class while the loaded
   config is the remote one, and Auto.register() rejects the mismatch. So we
   fetch the class directly with get_class_from_dynamic_module.

2. The bundled triton flash-attention calls tl.dot(trans_b=True), an argument
   removed from modern triton, so the forward pass dies. bert_layers.py already
   has a standard-attention fallback guarded by `flash_attn_qkvpacked_func is
   None`, so we set that module global to None. Costs nothing here - sequences
   are ~19 BPE tokens, far too short for flash attention to matter.

Requires transformers 4.4x. Does not work on transformers 5.x at all.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_ID = "zhihan1996/DNABERT-2-117M"


def load_dnabert(model_id=MODEL_ID, disable_flash=True):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    Cls = get_class_from_dynamic_module("bert_layers.BertModel", model_id)
    if disable_flash:
        mod = sys.modules[Cls.__module__]
        if getattr(mod, "flash_attn_qkvpacked_func", None) is not None:
            mod.flash_attn_qkvpacked_func = None
    backbone = Cls.from_pretrained(model_id, config=cfg)
    return tok, backbone, cfg


class DNABertRegressor(torch.nn.Module):
    """Backbone + mean-pooled regression head (PLAN.md: regression, one output)."""

    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask, **kw):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


if __name__ == "__main__":
    tok, bb, cfg = load_dnabert()
    print("backbone params:", round(sum(p.numel() for p in bb.parameters()) / 1e6, 1), "M")
    names = [n for n, mod in bb.named_modules() if isinstance(mod, torch.nn.Linear)]
    print("distinct Linear leaf names:", sorted({n.split(".")[-1] for n in names}))
    for n in names[:12]:
        print("   ", n)
    print("   ... total", len(names), "Linear modules")

    model = DNABertRegressor(bb, cfg.hidden_size).cuda().eval()
    s = ["TGTACATCCGTGGTACATCCGAGCGAGGAGCGCGAGGAGTGAGCTTATGTCGGACGCTTTTTTTTCGCTATCAGTTCTCA",
         "TAATCAGAAGAAATACACGATTTATAAGTTATGCAACAGGCTAGTCAATTGTATTTCTTCTACATTCGAAATTGATGGGA"]
    enc = tok(s, return_tensors="pt", padding=True)
    with torch.no_grad():
        y = model(enc["input_ids"].cuda(), enc["attention_mask"].cuda())
    print("input:", tuple(enc["input_ids"].shape), "-> predictions:", [round(v, 4) for v in y.tolist()])
    print(">>> DNABERT-2 OK")
