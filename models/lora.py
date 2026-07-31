import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.05):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "LoRALinear expects an nn.Linear layer."
            )

        if r <= 0:
            raise ValueError(
                "LoRA rank r must be > 0."
            )
        
        self.base = base_layer
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.dropout = nn.Dropout(dropout)

        # Freeze params of base layer
        for p in self.base.parameters():
            p.requires_grad = False

        # Low-rank matrices
        self.lora_A = nn.Parameter(
                torch.empty(
                    r,
                    base_layer.in_features
                )
            )
        
        self.lora_B = nn.Parameter(
                torch.empty(
                    base_layer.out_features,
                    r
                )
            )

        # A start random initialization
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5)
        )
        
        # B = 0 means LoRA initially has no effect
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original pretrained transformer
        base_out = self.base(x)

         # Low-rank adaptation
        #
        # x:
        # [B, L, in_features]
        #
        # A.T:
        # [in_features, r]
        #
        # after_A:
        # [B, L, r]
        #
        after_A =F.linear(
                self.dropout(x),
                self.lora_A
            )

        # B:
        # [r, out_features]
        #
        # lora_out:
        # [B, L, out_features]
        #
        lora_out = F.linear(
                after_A,
                self.lora_B
            )

        # 3. Combine with scaling
        return base_out + self.scaling * lora_out


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def apply_lora_to_decoder(
    model,
    r=8,
    alpha=16,
    dropout=0.05,
    target_modules=("self_q", "self_v", "src_q", "src_v"),
    layers=None,
    verbose=True,
):
    """
    Apply LoRA adapters to selected Linear layers of the DeepSC decoder.

    Parameters
    ----------
    model : nn.Module
        DeepSC/student model containing:
            model.decoder.dec_layers

    r : int
        LoRA rank.

    alpha : float
        LoRA scaling parameter.
        Effective scale = alpha / r.

    dropout : float
        Dropout applied only to the LoRA branch.

    target_modules : tuple/list/set of str
        Decoder projections where LoRA will be inserted.

        Available targets:
            self_q      -> self-attention query
            self_k      -> self-attention key
            self_v      -> self-attention value
            self_out    -> self-attention output projection

            src_q       -> cross-attention query
            src_k       -> cross-attention key
            src_v       -> cross-attention value
            src_out     -> cross-attention output projection

            ffn_1       -> first FFN linear layer
            ffn_2       -> second FFN linear layer

        Useful presets can be passed explicitly, for example:

            ("self_q", "self_v")
            ("src_q", "src_v")
            ("self_q", "self_v", "src_q", "src_v")

    layers : None or iterable of int
        Decoder layer indexes where LoRA should be applied.

        None:
            apply to every decoder layer

        Example:
            layers=[0, 1]

    verbose : bool
        Print information about inserted LoRA modules.

    Returns
    -------
    model
        Model with LoRA adapters inserted.
    """

    if r <= 0:
        raise ValueError(
            f"LoRA rank must be > 0, received r={r}"
        )

    if alpha <= 0:
        raise ValueError(
            f"LoRA alpha must be > 0, received alpha={alpha}"
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            f"LoRA dropout must be in [0, 1), received {dropout}"
        )

    valid_targets = {
        "self_q",
        "self_k",
        "self_v",
        "self_out",
        "src_q",
        "src_k",
        "src_v",
        "src_out",
        "ffn_1",
        "ffn_2",
    }

    target_modules = set(target_modules)

    unknown_targets = target_modules - valid_targets

    if unknown_targets:
        raise ValueError(
            f"Unknown LoRA target modules: {sorted(unknown_targets)}. "
            f"Available targets: {sorted(valid_targets)}"
        )

    if not hasattr(model, "decoder"):
        raise AttributeError(
            "Model does not contain 'decoder'."
        )

    if not hasattr(model.decoder, "dec_layers"):
        raise AttributeError(
            "model.decoder does not contain 'dec_layers'."
        )

    decoder_layers = model.decoder.dec_layers
    num_layers = len(decoder_layers)

    if layers is None:
        selected_layers = list(range(num_layers))
    else:
        selected_layers = sorted(set(layers))

        for idx in selected_layers:
            if idx < 0 or idx >= num_layers:
                raise IndexError(
                    f"Decoder layer index {idx} is invalid. "
                    f"Model contains {num_layers} decoder layers."
                )

    freeze_model(model)

    # Keep information about what we changed
    injected_modules = []

    def inject_lora(parent, attribute_name, full_name):
        """
        Replace parent.attribute_name with LoRALinear.

        Prevents accidentally wrapping the same layer twice.
        """

        layer = getattr(parent, attribute_name)

        # Avoid:
        #
        # LoRALinear(
        #     LoRALinear(
        #         nn.Linear(...)
        #     )
        # )
        #
        if isinstance(layer, LoRALinear):
            if verbose:
                print(
                    f"[LoRA] Skipping already wrapped layer: "
                    f"{full_name}"
                )
            return

        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"{full_name} must be nn.Linear, "
                f"but received {type(layer).__name__}"
            )

        setattr(
            parent,
            attribute_name,
            LoRALinear(
                base_layer=layer,
                r=r,
                alpha=alpha,
                dropout=dropout,
            ),
        )

        injected_modules.append(full_name)

    for layer_idx in selected_layers:

        layer = decoder_layers[layer_idx]

        if "self_q" in target_modules:
            inject_lora(
                layer.self_mha,
                "wq",
                f"decoder.dec_layers.{layer_idx}.self_mha.wq",
            )

        if "self_k" in target_modules:
            inject_lora(
                layer.self_mha,
                "wk",
                f"decoder.dec_layers.{layer_idx}.self_mha.wk",
            )

        if "self_v" in target_modules:
            inject_lora(
                layer.self_mha,
                "wv",
                f"decoder.dec_layers.{layer_idx}.self_mha.wv",
            )

        if "self_out" in target_modules:
            inject_lora(
                layer.self_mha,
                "dense",
                f"decoder.dec_layers.{layer_idx}.self_mha.dense",
            )


        if "src_q" in target_modules:
            inject_lora(
                layer.src_mha,
                "wq",
                f"decoder.dec_layers.{layer_idx}.src_mha.wq",
            )

        if "src_k" in target_modules:
            inject_lora(
                layer.src_mha,
                "wk",
                f"decoder.dec_layers.{layer_idx}.src_mha.wk",
            )

        if "src_v" in target_modules:
            inject_lora(
                layer.src_mha,
                "wv",
                f"decoder.dec_layers.{layer_idx}.src_mha.wv",
            )

        if "src_out" in target_modules:
            inject_lora(
                layer.src_mha,
                "dense",
                f"decoder.dec_layers.{layer_idx}.src_mha.dense",
            )

        if "ffn_1" in target_modules:
            inject_lora(
                layer.ffn,
                "w_1",
                f"decoder.dec_layers.{layer_idx}.ffn.w_1",
            )

        if "ffn_2" in target_modules:
            inject_lora(
                layer.ffn,
                "w_2",
                f"decoder.dec_layers.{layer_idx}.ffn.w_2",
            )

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    if verbose:
        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in model.parameters()
        )

        percentage = (
            100.0 * trainable / total
            if total > 0
            else 0.0
        )

        print("\n" + "=" * 70)
        print("LoRA applied to DeepSC decoder")
        print("=" * 70)

        print(f"Rank (r)          : {r}")
        print(f"Alpha             : {alpha}")
        print(f"Scaling           : {alpha / r:.4f}")
        print(f"Dropout           : {dropout}")

        print(
            f"Decoder layers    : "
            f"{selected_layers}"
        )

        print(
            f"Target modules    : "
            f"{sorted(target_modules)}"
        )

        print(
            f"Injected modules  : "
            f"{len(injected_modules)}"
        )

        print(
            f"Trainable params  : "
            f"{trainable:,}"
        )

        print(
            f"Total params      : "
            f"{total:,}"
        )

        print(
            f"Trainable percent : "
            f"{percentage:.4f}%"
        )

        print("=" * 70)

        for module_name in injected_modules:
            print(f"  [LoRA] {module_name}")

        print("=" * 70 + "\n")

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
    extra_info=None,
):
    """
    Save only the parameters used for language adaptation.

    This includes all parameters with requires_grad=True:
      - LoRA A/B matrices
      - decoder embedding, if enabled
      - output head, if enabled
      - LayerNorm parameters, if enabled
      - any future adaptation module that is explicitly unfrozen
    """

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"student_adapter_{adapter_name}_{epoch + 1:02d}.pth",
    )

    # Get names of parameters that are actually trainable
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    # Save only trainable/adaptation parameters
    adaptation_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }

    if not adaptation_state:
        raise RuntimeError(
            "No trainable adaptation parameters found. "
            "Check LoRA injection and enable_language_adaptation()."
        )

    checkpoint = {
        "epoch": epoch + 1,
        "adapter_name": adapter_name,
        "model_state_dict": adaptation_state,
        "num_adapter_tensors": len(adaptation_state),
        "num_adapter_parameters": sum(
            tensor.numel()
            for tensor in adaptation_state.values()
        ),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    torch.save(checkpoint, save_path)

    print("\n" + "=" * 70)
    print("Language adapter saved")
    print("=" * 70)
    print(f"Adapter          : {adapter_name}")
    print(f"Epoch            : {epoch + 1}")
    print(f"Tensors saved    : {len(adaptation_state)}")
    print(
        f"Parameters saved : "
        f"{sum(t.numel() for t in adaptation_state.values()):,}"
    )
    print(f"Path             : {save_path}")
    print("=" * 70)

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

def load_language_adapter(
    model,
    path,
    device,
    optimizer=None,
    strict_adapter=True,
    verbose=True,
):
    """
    Load a language-adaptation checkpoint into a model.

    The checkpoint may contain:
        - LoRA parameters
        - decoder embedding parameters
        - output-head parameters
        - LayerNorm parameters
        - any other parameters that were trainable when saved

    Parameters
    ----------
    model : nn.Module
        Model whose LoRA modules must already be inserted.

    path : str
        Path to the saved adapter checkpoint.

    device : torch.device or str
        Device used when loading the checkpoint.

    optimizer : torch.optim.Optimizer, optional
        If provided and optimizer state exists in the checkpoint,
        restore it.

    strict_adapter : bool
        If True, raise an error when checkpoint parameters do not
        exist in the current model.

    verbose : bool
        Print loading information.

    Returns
    -------
    model
        Model with the adapter loaded.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Adapter checkpoint must be a dictionary, "
            f"received {type(checkpoint).__name__}"
        )

    # new checkpoint:
    # {
    #     "epoch": ...,
    #     "adapter_name": ...,
    #     "model_state_dict": {...}
    # }
    #
    # and old raw state_dict checkpoints.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "'model_state_dict' must be a dictionary."
        )

    if len(state_dict) == 0:
        raise RuntimeError(
            f"Adapter checkpoint contains no parameters: {path}"
        )

    model_state = model.state_dict()

    unknown_keys = [
        key
        for key in state_dict.keys()
        if key not in model_state
    ]

    if unknown_keys and strict_adapter:
        raise RuntimeError(
            "The following adapter parameters do not exist "
            "in the current model:\n"
            + "\n".join(unknown_keys)
        )

    shape_mismatches = []

    for key, tensor in state_dict.items():

        if key not in model_state:
            continue

        if tensor.shape != model_state[key].shape:
            shape_mismatches.append(
                (
                    key,
                    tuple(tensor.shape),
                    tuple(model_state[key].shape),
                )
            )

    if shape_mismatches:
        message = [
            "Adapter tensor shape mismatch:"
        ]

        for key, saved_shape, model_shape in shape_mismatches:
            message.append(
                f"  {key}: "
                f"checkpoint={saved_shape}, "
                f"model={model_shape}"
            )

        raise RuntimeError("\n".join(message))

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    # Only unexpected keys matter here.
    #
    # Missing keys are expected because an adapter checkpoint
    # intentionally stores only a small subset of the model.
    if result.unexpected_keys and strict_adapter:
        raise RuntimeError(
            "Unexpected adapter keys:\n"
            + "\n".join(result.unexpected_keys)
        )

    optimizer_loaded = False

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        optimizer_loaded = True

    epoch = checkpoint.get("epoch")
    adapter_name = checkpoint.get("adapter_name")
    extra_info = checkpoint.get("extra_info")

    if verbose:
        num_tensors = len(state_dict)

        num_parameters = sum(
            tensor.numel()
            for tensor in state_dict.values()
        )

        print("\n" + "=" * 70)
        print("Language adapter loaded")
        print("=" * 70)

        if adapter_name is not None:
            print(f"Adapter          : {adapter_name}")

        if epoch is not None:
            print(f"Epoch            : {epoch}")

        print(f"Tensors loaded   : {num_tensors}")
        print(f"Parameters loaded: {num_parameters:,}")
        print(f"Path             : {path}")

        if optimizer is not None:
            print(
                f"Optimizer state  : "
                f"{'loaded' if optimizer_loaded else 'not found'}"
            )

        if extra_info is not None:
            print(f"Extra info       : {extra_info}")

        print("=" * 70 + "\n")

    return model