import torch
import torch.nn.functional as F


def feature_distillation_loss(student_feat, teacher_feat, targets, pad_idx):
    """
    MSE loss for feature-level distillation
    Args:
        student_feat: [B, T, D] student features
        teacher_feat: [B, T, D] teacher features  
        targets: [B, T] target tokens (for masking)
        pad_idx: padding index
    """
    # Create mask for valid positions
    mask = (targets != pad_idx).unsqueeze(-1).float()  # [B, T, 1]
    
    # Compute MSE loss
    mse_loss = F.mse_loss(student_feat, teacher_feat, reduction='none')  # [B, T, D]
    
    # Apply mask and average
    masked_loss = (mse_loss * mask).sum() / mask.sum().clamp_min(1.0)
    
    return masked_loss

# Feature Distillation Loss - Cosine Similarity
def feature_distillation_loss_cosine(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    temperature: float = 1.0,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    """
    Cosine similarity loss for feature-level distillation.
    
    Args:
        student_feat: [B, T, D] student features
        teacher_feat: [B, T, D] teacher features  
        targets: [B, T] target tokens (for masking)
        pad_idx: padding index
        temperature: temperature scaling for cosine similarity
        loss_scale: scaling factor for the loss
    """
    # Input validation
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            f"Feature shape mismatch: student={student_feat.shape}, "
            f"teacher={teacher_feat.shape}"
        )
    
    # Create mask [B, T]
    mask = (targets != pad_idx)
    num_valid = mask.sum()
    
    # Handle edge case: no valid tokens
    if num_valid == 0:
        return torch.tensor(0.0, device=student_feat.device, requires_grad=True)
    
    # Normalize features (L2 norm)
    student_norm = F.normalize(student_feat, p=2, dim=-1)  # [B, T, D]
    teacher_norm = F.normalize(teacher_feat, p=2, dim=-1)  # [B, T, D]
    
    # Compute cosine similarity: -1 to 1
    # Loss = 1 - cosine_similarity (0 when aligned, 2 when opposite)
    cosine_sim = (student_norm * teacher_norm).sum(dim=-1)  # [B, T]
    
    # Scale by temperature
    cosine_loss = (1 - cosine_sim) / temperature  # [B, T]
    
    # Apply mask and average
    masked_loss = (cosine_loss * mask).sum() / num_valid.float()
    
    return masked_loss * loss_scale

# Feature Distillation Loss - Cosine Similarity - Normalized
def feature_distillation_loss_cosine_normalized(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cosine similarity loss with additional feature stabilization.
    """
    # Normalize with stability
    student_norm = student_feat / (student_feat.norm(dim=-1, keepdim=True) + eps)
    teacher_norm = teacher_feat / (teacher_feat.norm(dim=-1, keepdim=True) + eps)
    
    # Cosine similarity
    cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
    
    # MSE on cosine similarity (alternative formulation)
    loss = F.mse_loss(cosine_sim, torch.ones_like(cosine_sim), reduction='none')
    
    mask = (targets != pad_idx)
    return (loss * mask).sum() / mask.sum().float()


# Logit Distillation Loss - Cosine Similarity
def logit_distillation_loss_cosine(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Cosine similarity loss on softmax probabilities.
    """
    # Apply softmax with temperature
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Normalize probabilities (L2 norm)
    student_norm = F.normalize(student_probs, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_probs, p=2, dim=-1)
    
    # Cosine similarity
    cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
    
    # Mask and average
    mask = (targets != pad_idx)
    loss = ((1 - cosine_sim) * mask).sum() / mask.sum().float()
    
    return loss


# def kd_kl_loss(
#     student_logits: torch.Tensor,
#     teacher_logits: torch.Tensor,
#     targets: torch.Tensor,
#     pad_idx: int,
#     temperature: float,
# ) -> torch.Tensor:

#     min_len = min(student_logits.size(1), teacher_logits.size(1), targets.size(1))

#     if student_logits.size(1) != min_len or teacher_logits.size(1) != min_len:
#         student_logits = student_logits[:, :min_len, :]
#         teacher_logits = teacher_logits[:, :min_len, :]
#         targets = targets[:, :min_len]

#     # apply mask so PAD tokens don't dominate KD
#     s_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
#     t_prob = F.softmax(teacher_logits / temperature, dim=-1)

#     kl_per_token = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)  # [B,T]

#     # Mask out padding positions and average over valid tokens
#     valid_mask = (targets != pad_idx).to(kl_per_token.dtype)
#     n_valid = valid_mask.sum().clamp_min(1.0)

#     masked_kl = (kl_per_token * valid_mask).sum() / n_valid

#     # Return average KL over valid tokens, scaled by temperature^2
#     return masked_kl * (temperature**2)


def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    temperature: float,
) -> torch.Tensor:
    """
    Knowledge-distillation KL divergence over non-PAD tokens.

    student_logits: [B, T, V]
    teacher_logits: [B, T, V]
    targets:        [B, T]
    """

    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}"
        )

    if student_logits.ndim != 3:
        raise ValueError(
            f"Expected student logits [B,T,V], "
            f"got {student_logits.shape}"
        )

    if teacher_logits.ndim != 3:
        raise ValueError(
            f"Expected teacher logits [B,T,V], "
            f"got {teacher_logits.shape}"
        )

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher shape mismatch: "
            f"student={student_logits.shape}, "
            f"teacher={teacher_logits.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets [B,T], got {targets.shape}"
        )

    if student_logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Logits/targets shape mismatch: "
            f"logits={student_logits.shape}, "
            f"targets={targets.shape}"
        )

    T = float(temperature)

    student_log_probs = F.log_softmax(
        student_logits / T,
        dim=-1,
    )

    with torch.no_grad():
        teacher_probs = F.softmax(
            teacher_logits / T,
            dim=-1,
        )

    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    valid_mask = (targets != pad_idx).to(
        kl_per_token.dtype
    )

    n_valid = valid_mask.sum().clamp_min(1.0)

    kd_loss = (
        (kl_per_token * valid_mask).sum()
        / n_valid
    )

    return kd_loss * (T ** 2)