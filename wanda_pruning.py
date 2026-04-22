import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def _prunable_linears(model):
    pairs = []
    for name, module in model.model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "embed" in name:
            continue
        pairs.append((name, module))
    return pairs


def _calib_batches(tokenizer, device, nsamples, seqlen):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    parts = [t for t in dataset["text"][:2000] if t.strip()]
    text = "\n\n".join(parts)
    need = nsamples * seqlen
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    ids = None
    while ids is None or ids.shape[1] < need:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < need:
            text = text + "\n\n" + text
        else:
            ids = ids[:, :need]
    for i in range(nsamples):
        yield ids[:, i * seqlen : (i + 1) * seqlen]


def apply_wanda_pruning(model, tokenizer, sparsity, nsamples=32, seqlen=512):
    if sparsity <= 0 or sparsity >= 1:
        raise ValueError("sparsity must be in (0, 1)")
    device = next(model.parameters()).device
    model.eval()
    linears = _prunable_linears(model)
    accum = {name: None for name, _ in linears}
    handles = []

    def make_hook(n):
        def hook(mod, inp, _):
            x = inp[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach()
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            s = x.float().pow(2).sum(dim=0)
            if accum[n] is None:
                accum[n] = s.clone()
            else:
                accum[n] = accum[n] + s

        return hook

    for name, mod in linears:
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in tqdm(
            list(_calib_batches(tokenizer, device, nsamples, seqlen)),
            desc="wanda calib",
        ):
            model(batch)

    for h in handles:
        h.remove()

    for name, mod in linears:
        w = mod.weight.data
        in_features = w.shape[1]
        sq = accum[name]
        if sq is None:
            col_norm = torch.ones(in_features, device=w.device, dtype=torch.float32)
        else:
            col_norm = torch.sqrt(sq.to(w.device).float() + 1e-8)
        scores = w.abs().float() * col_norm.unsqueeze(0)
        flat = scores.flatten()
        k_prune = int(sparsity * flat.numel())
        if k_prune <= 0:
            continue
        _, prune_idx = torch.topk(flat, k_prune, largest=False)
        mask = torch.ones_like(flat)
        mask[prune_idx] = 0
        mask = mask.view_as(w).to(w.dtype)
        w.mul_(mask)


def evaluate_wanda_sweep(
    model_cls,
    model_name,
    tokenizer,
    sparsities,
    eval_gsm8k,
    eval_arc,
    eval_ppl,
    gsm8k_samples=200,
    arc_samples=200,
    ppl_samples=50,
    torch_dtype=torch.float16,
    device_map="auto",
    wanda_nsamples=32,
    wanda_seqlen=512,
):
    out = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for sp in sparsities:
        m = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        m.eval()
        apply_wanda_pruning(
            m, tokenizer, sp, nsamples=wanda_nsamples, seqlen=wanda_seqlen
        )
        out[sp] = {
            "gsm8k": eval_gsm8k(m, tokenizer, num_samples=gsm8k_samples),
            "arc_challenge": eval_arc(m, tokenizer, num_samples=arc_samples),
            "perplexity": eval_ppl(m, tokenizer, num_samples=ppl_samples),
        }
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
