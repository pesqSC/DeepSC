import torch.nn.functional as F


def kd_student_rx_loss(
    student_logits,
    target,
    student_rx_feat,
    teacher_rx_feat,
    pad_idx,
    lambda_feat=0.5,
):
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_idx
    )

    valid = (target != pad_idx).unsqueeze(-1).float()

    min_len = min(student_rx_feat.size(1), teacher_rx_feat.size(1))
    student_rx_feat = student_rx_feat[:, :min_len, :]
    teacher_rx_feat = teacher_rx_feat[:, :min_len, :]
    valid = valid[:, :min_len, :]

    feat_loss = ((student_rx_feat - teacher_rx_feat.detach()) ** 2 * valid).sum()
    feat_loss = feat_loss / valid.sum().clamp(min=1.0)

    return ce_loss + lambda_feat * feat_loss