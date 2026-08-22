import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    LoRA wrapper around an nn.Linear layer.
    
    Implements: y = x @ W_base.T + scaling * (x @ dropout(A.T) @ B.T)
    """

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base layer.")

        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Freeze base layer parameters
        for p in self.base.parameters():
            p.requires_grad = False

        # Low-rank parameters:
        # lora_A: [r, in_features]
        # lora_B: [out_features, r]
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.empty(base_layer.out_features, r))

        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, zeros for B (so LoRA starts as identity/zero-delta)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge LoRA weights into the base layer for zero-latency inference."""
        if not self.merged:
            # delta_w = scaling * (lora_B @ lora_A) -> shape: [out_features, in_features]
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.base.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from base layer to resume training."""
        if self.merged:
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.base.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)

        # Base forward pass
        base_out = self.base(x)

        # LoRA forward pass: x -> dropout -> A -> B -> scale
        lora_in = self.dropout(x)
        after_A = F.linear(lora_in, self.lora_A)  # [B, ..., r]
        lora_out = F.linear(after_A, self.lora_B)  # [B, ..., out_features]

        return base_out + self.scaling * lora_out


def freeze_model(model: nn.Module):
    """Freeze all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def merge_lora_weights(model: nn.Module):
    """Recursively merge LoRA weights in all LoRALinear submodules."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Recursively unmerge LoRA weights in all LoRALinear submodules."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def apply_lora_to_decoder(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules=("self_q", "self_v", "src_q", "src_v"),
    layers=None,
    verbose: bool = True,
):
    """Apply LoRA adapters to selected Linear layers of the DeepSC decoder."""
    valid_targets = {
        "self_q", "self_k", "self_v", "self_out",
        "src_q", "src_k", "src_v", "src_out",
        "ffn_1", "ffn_2",
    }

    target_modules = set(target_modules)
    unknown_targets = target_modules - valid_targets
    if unknown_targets:
        raise ValueError(
            f"Unknown LoRA target modules: {sorted(unknown_targets)}. "
            f"Available targets: {sorted(valid_targets)}"
        )

    if not hasattr(model, "decoder") or not hasattr(model.decoder, "dec_layers"):
        raise AttributeError("Model structure missing 'decoder.dec_layers'.")

    decoder_layers = model.decoder.dec_layers
    num_layers = len(decoder_layers)
    selected_layers = list(range(num_layers)) if layers is None else sorted(set(layers))

    # Freeze base model first
    freeze_model(model)

    injected_modules = []

    def inject_lora(parent: nn.Module, attribute_name: str, full_name: str):
        layer = getattr(parent, attribute_name)
        if isinstance(layer, LoRALinear):
            if verbose:
                print(f"[LoRA] Skipping already wrapped layer: {full_name}")
            return

        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"{full_name} must be nn.Linear, but received {type(layer).__name__}"
            )

        setattr(
            parent,
            attribute_name,
            LoRALinear(base_layer=layer, r=r, alpha=alpha, dropout=dropout),
        )
        injected_modules.append(full_name)

    for layer_idx in selected_layers:
        layer = decoder_layers[layer_idx]

        mapping = {
            "self_q": (layer.self_mha, "wq", f"decoder.dec_layers.{layer_idx}.self_mha.wq"),
            "self_k": (layer.self_mha, "wk", f"decoder.dec_layers.{layer_idx}.self_mha.wk"),
            "self_v": (layer.self_mha, "wv", f"decoder.dec_layers.{layer_idx}.self_mha.wv"),
            "self_out": (layer.self_mha, "dense", f"decoder.dec_layers.{layer_idx}.self_mha.dense"),
            "src_q": (layer.src_mha, "wq", f"decoder.dec_layers.{layer_idx}.src_mha.wq"),
            "src_k": (layer.src_mha, "wk", f"decoder.dec_layers.{layer_idx}.src_mha.wk"),
            "src_v": (layer.src_mha, "wv", f"decoder.dec_layers.{layer_idx}.src_mha.wv"),
            "src_out": (layer.src_mha, "dense", f"decoder.dec_layers.{layer_idx}.src_mha.dense"),
            "ffn_1": (layer.ffn, "w_1", f"decoder.dec_layers.{layer_idx}.ffn.w_1"),
            "ffn_2": (layer.ffn, "w_2", f"decoder.dec_layers.{layer_idx}.ffn.w_2"),
        }

        for target_key in target_modules:
            parent_mod, attr, name = mapping[target_key]
            inject_lora(parent_mod, attr, name)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = (100.0 * trainable / total) if total > 0 else 0.0

        print("\n" + "=" * 70)
        print("LoRA applied to DeepSC decoder")
        print("=" * 70)
        print(f"Rank (r)          : {r}")
        print(f"Alpha             : {alpha}")
        print(f"Scaling           : {alpha / r:.4f}")
        print(f"Dropout           : {dropout}")
        print(f"Injected modules  : {len(injected_modules)}")
        print(f"Trainable params  : {trainable:,} / {total:,} ({pct:.4f}%)")
        print("=" * 70 + "\n")

    return model


def enable_language_adaptation(
    model: nn.Module,
    train_embedding: bool = True,
    train_output_head: bool = True,
    train_layer_norm: bool = True,
):
    """Enable parameters for language adaptation alongside LoRA."""
    if train_embedding and hasattr(model.decoder, "embedding"):
        for param in model.decoder.embedding.parameters():
            param.requires_grad = True

    if train_output_head and hasattr(model, "dense"):
        for param in model.dense.parameters():
            param.requires_grad = True

    if train_layer_norm and hasattr(model.decoder, "dec_layers"):
        for layer in model.decoder.dec_layers:
            for norm in [getattr(layer, f"layernorm{i}", None) for i in (1, 2, 3)]:
                if norm is not None:
                    for param in norm.parameters():
                        param.requires_grad = True

    return model


# def save_language_adapter(
#     epoch: int,
#     model: nn.Module,
#     save_dir: str,
#     adapter_name: str,
#     optimizer=None,
#     extra_info=None,
# ):
#     """Save only the adaptation parameters (LoRA + unfrozen headers/norms)."""
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"student_adapter_{adapter_name}_{epoch + 1:02d}.pth")

#     trainable_names = {
#         name for name, param in model.named_parameters() if param.requires_grad
#     }

#     adaptation_state = {
#         name: tensor.detach().cpu()
#         for name, tensor in model.state_dict().items()
#         if name in trainable_names
#     }

#     if not adaptation_state:
#         raise RuntimeError("No trainable parameters found to save!")

#     checkpoint = {
#         "epoch": epoch + 1,
#         "adapter_name": adapter_name,
#         "model_state_dict": adaptation_state,
#         "num_adapter_tensors": len(adaptation_state),
#         "num_adapter_parameters": sum(t.numel() for t in adaptation_state.values()),
#     }

#     if optimizer is not None:
#         checkpoint["optimizer_state_dict"] = optimizer.state_dict()
#     if extra_info is not None:
#         checkpoint["extra_info"] = extra_info

#     torch.save(checkpoint, save_path)
#     print(f"[Saver] Adapter saved successfully to: {save_path}")
#     return save_path

def save_language_adapter(
    epoch: int,
    model: nn.Module,
    save_dir: str,
    adapter_name: str,
    optimizer=None,
    scheduler=None,
    extra_info=None,
    lora_config=None,
):
    """
    Save all trainable adaptation parameters:
      - LoRA matrices
      - optional decoder embedding
      - optional output head
      - optional LayerNorms
    """

    os.makedirs(save_dir, exist_ok=True)

    if adapter_name == "latest_resume":
        filename = f"student_adapter_{adapter_name}.pth"
    else:
        filename = (
            f"student_adapter_{adapter_name}_{epoch + 1:02d}.pth"
        )

    save_path = os.path.join(save_dir, filename)

    trainable_names = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    adaptation_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }

    if not adaptation_state:
        raise RuntimeError(
            "No trainable adaptation parameters found to save."
        )

    lora_keys = [
        name
        for name in adaptation_state
        if "lora_" in name
    ]

    if not lora_keys:
        raise RuntimeError(
            "No LoRA parameters found in adaptation checkpoint."
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

    if lora_config is not None:
        checkpoint["lora_config"] = lora_config

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    torch.save(checkpoint, save_path)

    print(
        f"[Saver] Adapter saved: {save_path}"
    )
    print(
        f"[Saver] Tensors: {len(adaptation_state)}"
    )
    print(
        f"[Saver] LoRA tensors: {len(lora_keys)}"
    )
    return save_path


def load_language_adapter_auto(
    model: nn.Module,
    path: str,
    device,
    apply_lora_fn,
    optimizer=None,
    scheduler=None,
    strict_adapter: bool = True,
    verbose: bool = True,
):
    """
    Automatically load a saved language adapter.

    This function:

      1. Loads the adapter checkpoint.
      2. Reads the saved LoRA configuration.
      3. Automatically injects LoRA into the base model.
      4. Validates adapter parameter names and tensor shapes.
      5. Loads:
           - LoRA A/B matrices
           - optional embedding
           - optional output head
           - optional LayerNorm parameters
      6. Optionally restores optimizer and scheduler state.

    IMPORTANT
    ---------
    `model` must be the SAME base student architecture/checkpoint
    that was used before LoRA fine-tuning.

    Example:

        lora_pt = copy.deepcopy(student1)

        lora_pt = load_language_adapter_auto(
            model=lora_pt,
            path=adapter_path_pt,
            device=device,
            apply_lora_fn=apply_lora_to_decoder,
        )

        lora_pt.eval()
    """

    # 1. Check checkpoint path
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Adapter checkpoint not found:\n{path}"
        )

    # 2. Load checkpoint
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Invalid checkpoint format in:\n{path}"
        )

    # 3. Read state dictionary
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"Invalid or empty adapter state dictionary in:\n{path}"
        )

    # 4. Read saved metadata
    saved_epoch = checkpoint.get(
        "epoch",
        None,
    )

    adapter_name = checkpoint.get(
        "adapter_name",
        None,
    )

    lora_config = checkpoint.get(
        "lora_config",
        None,
    )

    extra_info = checkpoint.get(
        "extra_info",
        {},
    )

    # 5. LoRA config is required for automatic loading
    if lora_config is None:
        raise RuntimeError(
            "\nThis checkpoint does not contain 'lora_config'.\n\n"
            "Automatic LoRA injection is therefore impossible.\n"
            "For old checkpoints, manually call "
            "apply_lora_to_decoder(...) before loading."
        )

    # 6. Extract LoRA configuration
    required_config = [
        "r",
        "alpha",
        "dropout",
        "targets",
    ]

    missing_config = [
        key
        for key in required_config
        if key not in lora_config
    ]

    if missing_config:
        raise RuntimeError(
            "Incomplete LoRA configuration in checkpoint.\n"
            f"Missing fields: {missing_config}"
        )

    lora_r = int(
        lora_config["r"]
    )

    lora_alpha = float(
        lora_config["alpha"]
    )

    lora_dropout = float(
        lora_config["dropout"]
    )

    lora_targets = list(
        lora_config["targets"]
    )

    # 7. Sanity checks
    if lora_r <= 0:
        raise RuntimeError(
            f"Invalid LoRA rank: {lora_r}"
        )

    if lora_alpha <= 0:
        raise RuntimeError(
            f"Invalid LoRA alpha: {lora_alpha}"
        )

    if not 0.0 <= lora_dropout < 1.0:
        raise RuntimeError(
            f"Invalid LoRA dropout: {lora_dropout}"
        )

    if not lora_targets:
        raise RuntimeError(
            "LoRA target module list is empty."
        )

    # 8. Make sure model does not already contain LoRA
    existing_lora = [
        name
        for name, _ in model.named_parameters()
        if "lora_" in name
    ]

    if existing_lora:
        raise RuntimeError(
            "\nThe supplied model already contains LoRA parameters.\n"
            "load_language_adapter_auto() expects the ORIGINAL "
            "base student before LoRA injection.\n\n"
            "Example:\n"
            "    lora_pt = copy.deepcopy(student1)\n"
            "    lora_pt = load_language_adapter_auto(...)\n"
        )

    # 9. Automatically inject LoRA
    model = apply_lora_fn(
        model,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_targets,
    )

    model = model.to(device)

    # 10. Get current model state
    model_state = model.state_dict()

    # 11. Check adapter keys
    unknown_keys = [
        key
        for key in state_dict
        if key not in model_state
    ]

    if unknown_keys:

        message = (
            "Adapter contains parameters that do not exist "
            "in the current model:\n"
        )

        message += "\n".join(
            f"  - {key}"
            for key in unknown_keys
        )

        if strict_adapter:
            raise RuntimeError(message)

        if verbose:
            print("\n[Loader] WARNING")
            print(message)

    # 12. Check tensor shapes
    shape_errors = []

    for key, tensor in state_dict.items():

        if key not in model_state:
            continue

        expected_shape = tuple(
            model_state[key].shape
        )

        checkpoint_shape = tuple(
            tensor.shape
        )

        if checkpoint_shape != expected_shape:

            shape_errors.append(
                (
                    key,
                    checkpoint_shape,
                    expected_shape,
                )
            )

    if shape_errors:

        message = [
            "Adapter parameter shape mismatch:"
        ]

        for (
            key,
            checkpoint_shape,
            model_shape,
        ) in shape_errors:

            message.append(
                f"  - {key}\n"
                f"      checkpoint: {checkpoint_shape}\n"
                f"      model     : {model_shape}"
            )

        raise RuntimeError(
            "\n".join(message)
        )

    lora_keys = [
        key
        for key in state_dict
        if "lora_" in key
    ]

    if not lora_keys:
        raise RuntimeError(
            "Checkpoint contains no LoRA tensors."
        )

    lora_a_keys = [
        key
        for key in lora_keys
        if key.endswith("lora_A")
    ]

    lora_b_keys = [
        key
        for key in lora_keys
        if key.endswith("lora_B")
    ]

    if len(lora_a_keys) != len(lora_b_keys):
        raise RuntimeError(
            "Invalid LoRA checkpoint.\n"
            f"lora_A tensors: {len(lora_a_keys)}\n"
            f"lora_B tensors: {len(lora_b_keys)}"
        )

    for a_key in lora_a_keys:

        prefix = a_key[:-len("lora_A")]

        b_key = (
            prefix
            + "lora_B"
        )

        if b_key not in state_dict:
            raise RuntimeError(
                "Incomplete LoRA pair:\n"
                f"  A: {a_key}\n"
                f"  Missing B: {b_key}"
            )

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if result.unexpected_keys:

        message = (
            "Unexpected keys while loading adapter:\n"
        )

        message += "\n".join(
            f"  - {key}"
            for key in result.unexpected_keys
        )

        if strict_adapter:
            raise RuntimeError(message)

        if verbose:
            print(
                "\n[Loader] WARNING"
            )
            print(message)

    # Missing base-model weights are EXPECTED because
    # adapter checkpoints only contain adapted parameters.
    missing_adapter_keys = []

    for key in lora_keys:

        if key in result.missing_keys:
            missing_adapter_keys.append(
                key
            )

    if missing_adapter_keys:

        raise RuntimeError(
            "Some LoRA parameters were not loaded:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing_adapter_keys
            )
        )

    optimizer_loaded = False

    if optimizer is not None:

        optimizer_state = checkpoint.get(
            "optimizer_state_dict",
            None,
        )

        if optimizer_state is not None:

            optimizer.load_state_dict(
                optimizer_state
            )

            optimizer_loaded = True

        elif verbose:

            print(
                "[Loader] WARNING: optimizer was provided "
                "but optimizer_state_dict was not saved."
            )

    scheduler_loaded = False

    if scheduler is not None:

        scheduler_state = checkpoint.get(
            "scheduler_state_dict",
            None,
        )

        if scheduler_state is not None:

            scheduler.load_state_dict(
                scheduler_state
            )

            scheduler_loaded = True

        elif verbose:

            print(
                "[Loader] WARNING: scheduler was provided "
                "but scheduler_state_dict was not saved."
            )

    total_tensors = len(
        state_dict
    )

    total_parameters = sum(
        tensor.numel()
        for tensor in state_dict.values()
    )

    embedding_keys = [
        key
        for key in state_dict
        if "embedding" in key.lower()
    ]

    dense_keys = [
        key
        for key in state_dict
        if key.startswith("dense.")
    ]

    norm_keys = [
        key
        for key in state_dict
        if "layernorm" in key.lower()
    ]

    expected_num_tensors = checkpoint.get(
        "num_adapter_tensors"
    )

    expected_num_parameters = checkpoint.get(
        "num_adapter_parameters"
    )

    if (
        expected_num_tensors is not None
        and expected_num_tensors != total_tensors
    ):
        raise RuntimeError(
            "Adapter tensor-count verification failed.\n"
            f"Saved value : {expected_num_tensors}\n"
            f"Actual value: {total_tensors}"
        )

    if (
        expected_num_parameters is not None
        and expected_num_parameters != total_parameters
    ):
        raise RuntimeError(
            "Adapter parameter-count verification failed.\n"
            f"Saved value : {expected_num_parameters:,}\n"
            f"Actual value: {total_parameters:,}"
        )

    if verbose:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Language Adapter Loaded"
        )

        print(
            "=" * 70
        )

        print(
            f"Checkpoint       : {path}"
        )

        print(
            f"Adapter name     : {adapter_name}"
        )

        print(
            f"Training epoch   : {saved_epoch}"
        )

        # LoRA configuration
        print(
            "\nLoRA configuration"
        )

        print(
            f"  Rank           : {lora_r}"
        )

        print(
            f"  Alpha          : {lora_alpha}"
        )

        print(
            f"  Scaling        : "
            f"{lora_alpha / lora_r:.4f}"
        )

        print(
            f"  Dropout        : {lora_dropout}"
        )

        print(
            f"  Targets        : "
            f"{lora_targets}"
        )

        # Adapter tensors
        print(
            "\nAdapter parameters"
        )

        print(
            f"  Total tensors  : "
            f"{total_tensors}"
        )

        print(
            f"  Parameters     : "
            f"{total_parameters:,}"
        )

        print(
            f"  LoRA tensors   : "
            f"{len(lora_keys)}"
        )

        print(
            f"    lora_A       : "
            f"{len(lora_a_keys)}"
        )

        print(
            f"    lora_B       : "
            f"{len(lora_b_keys)}"
        )

        print(
            f"  Embedding      : "
            f"{len(embedding_keys)}"
        )

        print(
            f"  Dense          : "
            f"{len(dense_keys)}"
        )

        print(
            f"  LayerNorm      : "
            f"{len(norm_keys)}"
        )

        # Extra metadata
        if extra_info:

            print(
                "\nExtra information"
            )

            for key, value in extra_info.items():

                print(
                    f"  {key:<18}: {value}"
                )

        # Resume status
        if optimizer is not None:

            print(
                f"\nOptimizer loaded : "
                f"{optimizer_loaded}"
            )

        if scheduler is not None:

            print(
                f"Scheduler loaded : "
                f"{scheduler_loaded}"
            )

        print(
            "=" * 70
            + "\n"
        )

    return model


def load_language_adapter(
    model: nn.Module,
    path: str,
    device,
    optimizer=None,
    strict_adapter: bool = True,
    verbose: bool = True,
):
    """Load adapter checkpoint into a model with pre-injected LoRA layers."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # for name, tensor in state_dict.items():
    #     print(name, tensor.shape)

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Invalid or empty checkpoint state dictionary in {path}")

    # Verify keys and shapes
    model_state = model.state_dict()
    for key, tensor in state_dict.items():
        if key in model_state and tensor.shape != model_state[key].shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint={tuple(tensor.shape)}, model={tuple(model_state[key].shape)}"
            )

    result = model.load_state_dict(state_dict, strict=False)

    if strict_adapter and result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys encountered: {result.unexpected_keys}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if verbose:
        num_parameters = sum(t.numel() for t in state_dict.values())
        print(f"[Loader] Loaded adapter from {path} ({num_parameters:,} params)")

    return model