import torch.nn as nn
from torch.optim import AdamW

def build_differential_optimizer(
    model: nn.Module,
    lr_lora: float = 1e-4,
    lr_head: float = 5e-5,
    lr_embedding: float = 2e-5,
    lr_norm: float = 2e-5,
    weight_decay: float = 1e-4,
) -> AdamW:

    lora_params = []
    head_params = []
    embedding_params = []
    norm_bias_params = []
    other_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # LoRA matrices
        if "lora_" in name_lower:
            lora_params.append(param)

        # Output vocabulary projection
        elif "dense.weight" in name_lower:
            head_params.append(param)

        elif "dense.bias" in name_lower:
            head_params.append(param)

        # Embedding
        elif "embedding" in name_lower:
            embedding_params.append(param)

        # LayerNorm
        elif (
            "layernorm" in name_lower
            or "norm" in name_lower
            or param.ndim <= 1
        ):
            norm_bias_params.append(param)

        # Anything unexpected
        else:
            other_params.append(param)

    param_groups = []

    if lora_params:
        param_groups.append({
            "params": lora_params,
            "lr": lr_lora,
            "weight_decay": weight_decay,
            "name": "lora",
        })

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": 0.0,
            "name": "output_head",
        })

    if embedding_params:
        param_groups.append({
            "params": embedding_params,
            "lr": lr_embedding,
            "weight_decay": weight_decay,
            "name": "embedding",
        })

    if norm_bias_params:
        param_groups.append({
            "params": norm_bias_params,
            "lr": lr_norm,
            "weight_decay": 0.0,
            "name": "norm_bias",
        })

    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": lr_lora,
            "weight_decay": weight_decay,
            "name": "other",
        })

    print("\n" + "=" * 70)
    print("OPTIMIZER PARAMETER GROUPS")
    print("=" * 70)

    total_trainable = 0

    for group in param_groups:
        count = sum(
            p.numel()
            for p in group["params"]
        )

        total_trainable += count

        print(
            f"{group['name']:15s} "
            f"| Params: {count:>12,d} "
            f"| LR: {group['lr']:.2e} "
            f"| WD: {group['weight_decay']:.2e}"
        )

    print("-" * 70)
    print(f"Total trainable parameters: {total_trainable:,}")
    print("=" * 70)

    return AdamW(param_groups)