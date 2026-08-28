"""Topiary's salience-shaped width truncation, ported to dense HF models.

Topiary (https://github.com/jesusluque/topiary) targets MoE experts on MLX.
The technique itself is architecture-agnostic: a dense FFN is a single
always-routed expert, so the "routed salience" statistic reduces to plain
activation statistics over the calibration traffic:

    salience^2_i = E[h_i^2] * ||W_down[:, i]||^2,   h = act(gate(x)) * up(x)

Neurons are ordered by salience (descending) and the FFN width is truncated
to the top-k prefix, exactly as topiary's bake_permutation + truncate_in_place
do for each expert — here implemented as row/column slicing of the bf16
gate/up/down projections of every decoder layer (Qwen3-style MLP).

No quantization-group constraint applies in bf16, but we keep k a multiple of
GROUP=64 to stay faithful to topiary's dial granularity.
"""

from __future__ import annotations

import torch

GROUP = 64


def decoder_layers(model):
    """Text decoder layers across architectures (plain causal LM or
    multimodal wrappers like Gemma3ForConditionalGeneration)."""
    m = model.model
    if hasattr(m, "layers"):
        return m.layers
    if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        return m.language_model.layers
    raise AttributeError("cannot locate decoder layers on this model")


def _mlp_of(layer):
    mlp = layer.mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(mlp, name), f"MLP lacks {name}; not a gate/up/down FFN"
    return mlp


@torch.no_grad()
def calibrate_salience(model, input_ids_batches, device="cuda"):
    """Accumulate sum(h^2) per FFN neuron over calibration traffic.

    h (the input of down_proj) is captured with a forward-pre-hook, mirroring
    topiary's online profiler which sums h^2 inside the expert forward.
    Returns {layer_idx: salience float64 tensor [inter]}.
    """
    layers = decoder_layers(model)
    acc = {i: torch.zeros(_mlp_of(l).down_proj.in_features, dtype=torch.float64, device=device)
           for i, l in enumerate(layers)}
    count = 0
    hooks = []

    def make_hook(idx):
        def hook(module, args):
            h = args[0]
            acc[idx] += (h.float() ** 2).sum(dim=(0, 1)).double()
        return hook

    for i, l in enumerate(layers):
        hooks.append(_mlp_of(l).down_proj.register_forward_pre_hook(make_hook(i)))

    try:
        for ids in input_ids_batches:
            ids = ids.to(device)
            model(input_ids=ids, use_cache=False)
            count += ids.numel()
    finally:
        for h in hooks:
            h.remove()

    salience = {}
    for i, l in enumerate(layers):
        wd = _mlp_of(l).down_proj.weight  # [out, inter]
        col2 = (wd.float() ** 2).sum(dim=0).double()  # ||W_down[:,i]||^2
        mean_h2 = acc[i] / max(count, 1)
        salience[i] = (mean_h2 * col2).cpu()
    return salience, count


def orders_from_salience(salience):
    """Neuron order per layer, most important first (topiary's orders.npz)."""
    return {i: torch.argsort(s, descending=True) for i, s in salience.items()}


def snapshot_mlp_weights(model):
    """CPU copies of the original gate/up/down weights, for restore()."""
    snap = {}
    for i, l in enumerate(decoder_layers(model)):
        mlp = _mlp_of(l)
        snap[i] = {n: getattr(mlp, n).weight.detach().cpu().clone()
                   for n in ("gate_proj", "up_proj", "down_proj")}
    return snap


@torch.no_grad()
def restore_mlp_weights(model, snap, device="cuda"):
    for i, l in enumerate(decoder_layers(model)):
        mlp = _mlp_of(l)
        for n in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, n)
            w = snap[i][n].to(device)
            proj.weight = torch.nn.Parameter(w, requires_grad=False)
            if n == "down_proj":
                proj.in_features = w.shape[1]
            else:
                proj.out_features = w.shape[0]


@torch.no_grad()
def truncate_in_place(model, orders, k_per_layer, device="cuda"):
    """Bake the salience permutation and keep the top-k prefix (per layer).

    gate/up: permute+slice output rows.  down: permute+slice input columns.
    The model function is unchanged by the permutation itself; only the
    truncation removes (least-salient) capacity — same contract as topiary.
    """
    for i, l in enumerate(decoder_layers(model)):
        mlp = _mlp_of(l)
        k = k_per_layer[i] if not isinstance(k_per_layer, int) else k_per_layer
        keep = orders[i][:k].to(device)
        for n in ("gate_proj", "up_proj"):
            proj = getattr(mlp, n)
            w = proj.weight.data.index_select(0, keep).contiguous()
            proj.weight = torch.nn.Parameter(w, requires_grad=False)
            proj.out_features = k
        proj = mlp.down_proj
        w = proj.weight.data.index_select(1, keep).contiguous()
        proj.weight = torch.nn.Parameter(w, requires_grad=False)
        proj.in_features = k
    torch.cuda.empty_cache()
