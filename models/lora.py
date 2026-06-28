import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.05):
        super().__init__()

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)

        lora_out = F.linear(
            F.linear(self.dropout(x), self.lora_A),
            self.lora_B
        )

        return base_out + self.scaling * lora_out


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def apply_lora_to_decoder(model, r=8, alpha=16, dropout=0.05):
    freeze_model(model)

    for layer in model.decoder.dec_layers:
        layer.self_mha.wq = LoRALinear(layer.self_mha.wq, r, alpha, dropout)
        layer.self_mha.wk = LoRALinear(layer.self_mha.wk, r, alpha, dropout)
        layer.self_mha.wv = LoRALinear(layer.self_mha.wv, r, alpha, dropout)
        layer.self_mha.dense = LoRALinear(layer.self_mha.dense, r, alpha, dropout)

        layer.src_mha.wq = LoRALinear(layer.src_mha.wq, r, alpha, dropout)
        layer.src_mha.wk = LoRALinear(layer.src_mha.wk, r, alpha, dropout)
        layer.src_mha.wv = LoRALinear(layer.src_mha.wv, r, alpha, dropout)
        layer.src_mha.dense = LoRALinear(layer.src_mha.dense, r, alpha, dropout)

        layer.ffn.w_1 = LoRALinear(layer.ffn.w_1, r, alpha, dropout)
        layer.ffn.w_2 = LoRALinear(layer.ffn.w_2, r, alpha, dropout)

    return model


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def save_lora(
    epoch: int,
    model: torch.nn.Module,
    save_dir: str,
    adapter_name: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    filename = f"student_lora_{adapter_name}_{epoch + 1:02d}.pth"
    save_path = os.path.join(save_dir, filename)

    lora_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }

    if not lora_state:
        raise RuntimeError(
            "No LoRA parameters were found in the model."
        )

    torch.save(lora_state, save_path)

    print(
        f"Saved {len(lora_state)} LoRA tensors to: {save_path}"
    )

    return save_path


def load_lora(model, path, device):
    state_dict = torch.load(
        path,
        map_location=device,
    )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The LoRA checkpoint must be a state dictionary."
        )

    lora_keys = [
        key for key in state_dict
        if "lora_" in key
    ]

    if not lora_keys:
        raise RuntimeError(
            f"No LoRA parameters found in checkpoint: {path}"
        )

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    unexpected_lora_keys = [
        key for key in result.unexpected_keys
        if "lora_" in key
    ]

    if unexpected_lora_keys:
        raise RuntimeError(
            "These LoRA parameters did not match the model:\n"
            + "\n".join(unexpected_lora_keys)
        )

    print(f"Loaded {len(lora_keys)} LoRA tensors.")
    return model