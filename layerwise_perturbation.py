import torch
from tqdm import tqdm


def _layer_output_hook(scale=None, noise_std=None):
    if scale is None and noise_std is None:
        raise ValueError("set scale or noise_std")

    def perturb(t):
        if scale is not None:
            return t * scale
        return t + torch.randn_like(t) * noise_std

    def hook(_module, _inp, output):
        if not isinstance(output, tuple):
            return perturb(output)
        h = output[0]
        return (perturb(h),) + tuple(output[1:])

    return hook


def layerwise_reasoning_sensitivity(
    model,
    tokenizer,
    evaluate_gsm8k_fn,
    num_samples=200,
    scale=0.5,
    noise_std=None,
):
    if noise_std is None and scale is None:
        raise ValueError("set scale or noise_std")
    model.eval()
    layers = model.model.layers
    baseline = evaluate_gsm8k_fn(model, tokenizer, num_samples=num_samples)
    scores = []
    for i in tqdm(range(len(layers)), desc="layer perturb"):
        if noise_std is not None:
            h = layers[i].register_forward_hook(
                _layer_output_hook(scale=None, noise_std=noise_std)
            )
        else:
            h = layers[i].register_forward_hook(_layer_output_hook(scale=scale))
        try:
            acc = evaluate_gsm8k_fn(model, tokenizer, num_samples=num_samples)
        finally:
            h.remove()
        scores.append(
            {
                "layer": i,
                "accuracy": acc,
                "drop": baseline - acc,
                "sensitivity": baseline - acc,
            }
        )
    return {"baseline": baseline, "per_layer": scores}
