import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from wanda_pruning import _prunable_linears, _calib_batches


def _normalize(tensor):
    """Min-max normalize a tensor to [0, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


def _get_layer_index(name):
    """
    Extract layer index from a weight name like
    'layers.3.self_attn.q_proj' → 3
    Returns None if no layer index found.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None

def apply_composite_pruning(model, composite_scores, sparsity):
    """
    Prune model weights using precomputed composite scores.
    Weights with the lowest composite scores get zeroed out.

    Args:
        model            : loaded LLM (will be modified in place)
        composite_scores : composite scores calculated in evaulate_composite_sweep()
        sparsity         : fraction of weights to prune e.g. 0.4
    """
    if sparsity <= 0 or sparsity >= 1:
        raise ValueError("sparsity must be in (0, 1)")

    linears = _prunable_linears(model)

    for name, mod in linears:
        if name not in composite_scores:
            continue

        w = mod.weight.data
        scores = composite_scores[name]

        # Find bottom-k weights (lowest composite score = least important)
        flat = scores.flatten()
        k_prune = int(sparsity * flat.numel())
        if k_prune <= 0:
            continue

        _, prune_idx = torch.topk(flat, k_prune, largest=False)
        mask = torch.ones_like(flat)
        mask[prune_idx] = 0
        mask = mask.view_as(w).to(w.dtype)
        w.mul_(mask)

    print(f"Composite pruning applied at sparsity={sparsity}")

def compute_wanda_scores_only(model, tokenizer, nsamples=32, seqlen=512):
    """
    Compute and return raw Wanda scores for all linear layers.
    """
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
            accum[n] = s.clone() if accum[n] is None else accum[n] + s
        return hook

    for name, mod in linears:
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in tqdm(
            list(_calib_batches(tokenizer, device, nsamples, seqlen)),
            desc="collecting activations"
        ):
            model(batch)

    for h in handles:
        h.remove()

    # Compute raw wanda scores
    wanda_scores = {}
    for name, mod in linears:
        w = mod.weight.data
        sq = accum[name]
        if sq is None:
            col_norm = torch.ones(
                w.shape[1], device=w.device, dtype=torch.float32
            )
        else:
            col_norm = torch.sqrt(sq.to(w.device).float() + 1e-8)
        wanda_scores[name] = w.abs().float() * col_norm.unsqueeze(0)

    return wanda_scores

def evaluate_composite_sweep(
    model_cls,
    model_name,
    tokenizer,
    sensitivity_results,
    sparsities,
    alphas,
    eval_gsm8k,
    eval_arc,
    eval_ppl,
    gsm8k_samples=200,
    arc_samples=200,
    ppl_samples=50,
    torch_dtype=torch.float16,
    nsamples=32,
    seqlen=512,
):
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    # Precompute normalized sensitivity scores
    per_layer = sensitivity_results["per_layer"]
    raw_drops = torch.tensor(
        [max(item["drop"], 0.0) for item in per_layer],
        dtype=torch.float32
    )
    sensitivity_normalized = _normalize(raw_drops)

    for sp in sparsities:
        print(f"\n{'='*50}")
        print(f"Sparsity = {sp}")
        print(f"{'='*50}")

        # Load model ONCE to compute Wanda scores
        m = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        m.eval()

        wanda_scores_raw = compute_wanda_scores_only(
            m, tokenizer, nsamples=nsamples, seqlen=seqlen
        )

        # Move ALL wanda scores to CPU
        for k in wanda_scores_raw:
            wanda_scores_raw[k] = wanda_scores_raw[k].cpu()

        # Delete model BEFORE alpha loop
        del m
        torch.cuda.empty_cache()
        gc.collect()

        # Loop over alphas
        for alpha in alphas:
            print(f"\n-- Alpha = {alpha} --")

            # Load fresh model
            m_alpha = model_cls.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            m_alpha.eval()

            linears = _prunable_linears(m_alpha)
            composite_scores = {}

            for name, mod in linears:
                layer_idx = _get_layer_index(name)

                # Move wanda score to GPU ONLY when needed
                wanda_norm = _normalize(
                    wanda_scores_raw[name].to(mod.weight.device)
                )

                # Sensitivity scalar
                if (layer_idx is not None and
                        layer_idx < len(sensitivity_normalized)):
                    layer_sensitivity = sensitivity_normalized[layer_idx].item()
                else:
                    layer_sensitivity = 0.0

                sensitivity_matrix = torch.full_like(
                    wanda_norm, fill_value=layer_sensitivity
                )

                composite_scores[name] = (
                    alpha * wanda_norm +
                    (1 - alpha) * sensitivity_matrix
                )

            # Prune
            apply_composite_pruning(m_alpha, composite_scores, sparsity=sp)

            if sp not in results:
                results[sp] = {}

            # Evaluate
            with torch.no_grad():
                results[sp][alpha] = {
                    "gsm8k": eval_gsm8k(
                        m_alpha, tokenizer, num_samples=gsm8k_samples
                    ),
                    "arc_challenge": eval_arc(
                        m_alpha, tokenizer, num_samples=arc_samples
                    ),
                    "perplexity": eval_ppl(
                        m_alpha, tokenizer, num_samples=ppl_samples
                    ),
                }

            # FULL CLEANUP
            del m_alpha, composite_scores, wanda_norm, sensitivity_matrix
            torch.cuda.empty_cache()
            gc.collect()

    return results