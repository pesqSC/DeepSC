import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def save_lora(epoch, model, path, len):
    path = os.path.join(
                args.save_dir,
                f"student_lora{len}_{epoch+1:02d}.pth"
            )

    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
        path
    )


def load_lora(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model