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

    # def forward(self, x):
    #     base_out = self.base(x)

    #     lora_out = F.linear(
    #         F.linear(self.dropout(x), self.lora_A),
    #         self.lora_B
    #     )

    #     return base_out + self.scaling * lora_out

    def forward(self, x):
        # 1. Get base model output
        base_out = self.base(x)

        # 2. Compute LoRA output using matrix multiplication (@)
        # x shape: (batch, ..., in_features)
        # lora_A.t() shape: (in_features, r)
        # lora_B.t() shape: (r, out_features)
        after_A = self.dropout(x) @ self.lora_A.t()
        lora_out = after_A @ self.lora_B.t()

        # 3. Combine with scaling
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

    # 3. Explicitly unfreeze ONLY the LoRA parameters just to be safe
    # (Since your LoRALinear code sets base params to False, but lora_A/B default to True)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    return model

# --- Verification Utility ---
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} | "
        f"All params: {all_param} | "
        f"Trainable %: {100 * trainable_params / all_param:.4f}%"
    )

def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def save_language_adapter(
    epoch,
    model,
    save_dir,
    adapter_name,
    optimizer=None,
):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"student_adapter_{adapter_name}_{epoch + 1:02d}.pth",
    )

    adaptation_state = {}

    for name, tensor in model.state_dict().items():
        should_save = (
            "lora_" in name
            or name.startswith("decoder.embedding.")
            or name.startswith("dense.")
            or "layernorm" in name
        )

        if should_save:
            adaptation_state[name] = tensor.detach().cpu()

    checkpoint = {
        "epoch": epoch + 1,
        "adapter_name": adapter_name,
        "model_state_dict": adaptation_state,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)

    print(
        f"Saved {len(adaptation_state)} adaptation tensors to "
        f"{save_path}"
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


def enable_language_adaptation(
    model,
    train_embedding=True,
    train_output_head=True,
    train_layer_norm=True,
):
    """
    Keep the base student frozen while enabling the small set of
    parameters needed for language adaptation.
    """

    if train_embedding:
        for parameter in model.decoder.embedding.parameters():
            parameter.requires_grad = True

    if train_output_head:
        for parameter in model.dense.parameters():
            parameter.requires_grad = True

    if train_layer_norm:
        for layer in model.decoder.dec_layers:
            for parameter in layer.layernorm1.parameters():
                parameter.requires_grad = True

            for parameter in layer.layernorm2.parameters():
                parameter.requires_grad = True

            for parameter in layer.layernorm3.parameters():
                parameter.requires_grad = True

    return model


def adaptation_parameters(model):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

def load_language_adapter(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    adapter_keys = [
        key
        for key in state_dict.keys()
        if (
            "lora_" in key
            or key.startswith("decoder.embedding.")
            or key.startswith("dense.")
            or "layernorm" in key
        )
    ]

    unexpected_adapter_keys = [
        key
        for key in result.unexpected_keys
        if (
            "lora_" in key
            or key.startswith("decoder.embedding.")
            or key.startswith("dense.")
            or "layernorm" in key
        )
    ]

    if not adapter_keys:
        raise RuntimeError(
            f"No adaptation parameters found in {path}"
        )

    if unexpected_adapter_keys:
        raise RuntimeError(
            "Some adapter parameters did not match:\n"
            + "\n".join(unexpected_adapter_keys)
        )

    print(f"Loaded {len(adapter_keys)} adaptation tensors.")
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

    return model