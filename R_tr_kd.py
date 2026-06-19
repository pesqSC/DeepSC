import os
import json
import math
import time
import random
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC


# -----------------------------
# Reproducibility
# -----------------------------
def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Masks
# -----------------------------
def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    # seq: [B, L]
    return (seq == pad_idx).unsqueeze(1).float()


def create_look_ahead_mask(size: int, device: torch.device) -> torch.Tensor:
    # upper triangular without diagonal
    mask = torch.triu(torch.ones((size, size), device=device), diagonal=1)
    return mask


def create_masks(src: torch.Tensor, trg_inp: torch.Tensor, pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_padding_mask = create_padding_mask(src, pad_idx)                # [B,1,S]
    trg_padding_mask = create_padding_mask(src, pad_idx)                # memory mask for cross-attn
    look_ahead_mask = create_look_ahead_mask(trg_inp.size(1), src.device)  # [T,T]
    dec_target_padding_mask = create_padding_mask(trg_inp, pad_idx).squeeze(1)  # [B,T]
    look_ahead_mask = torch.maximum(
        look_ahead_mask.unsqueeze(0),  # [1,T,T]
        dec_target_padding_mask.unsqueeze(1).float()  # [B,1,T]
    )
    return src_padding_mask, look_ahead_mask, trg_padding_mask


# -----------------------------
# Channel models
# -----------------------------
def snr_db_to_noise_std(snr_db: float) -> float:
    # assuming unit power
    return 10 ** (-snr_db / 20.0)


def power_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # normalize last dimension average power
    power = torch.mean(x.pow(2), dim=-1, keepdim=True)
    return x / torch.sqrt(power + eps)


def awgn_channel(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_std
    return x + noise


def rayleigh_channel(x: torch.Tensor, noise_std: float, eps: float = 1e-8) -> torch.Tensor:
    # simple real-valued fading approximation
    h = torch.sqrt(
        torch.clamp(torch.randn_like(x).pow(2) + torch.randn_like(x).pow(2), min=eps) / 2.0
    )
    y = h * x + torch.randn_like(x) * noise_std
    return y / (h + eps)


def rician_channel(x: torch.Tensor, noise_std: float, K: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    los = math.sqrt(K / (K + 1.0))
    nlos_scale = math.sqrt(1.0 / (K + 1.0))
    h = los + nlos_scale * torch.randn_like(x)
    y = h * x + torch.randn_like(x) * noise_std
    return y / (h + eps)


def apply_channel(x: torch.Tensor, channel_type: str, noise_std: float) -> torch.Tensor:
    x = power_normalize(x)

    channel_type = channel_type.lower()
    if channel_type == "awgn":
        return awgn_channel(x, noise_std)
    if channel_type == "rayleigh":
        return rayleigh_channel(x, noise_std)
    if channel_type == "rician":
        return rician_channel(x, noise_std)
    raise ValueError(f"Unsupported channel: {channel_type}")


# -----------------------------
# Receiver-only student
# -----------------------------
class ReceiverOnly(nn.Module):
    """
    Student contains only TR/RX:
      - channel_decoder
      - semantic decoder
      - output head
    """

    def __init__(self, channel_decoder: nn.Module, decoder: nn.Module, dense: nn.Module):
        super().__init__()
        self.channel_decoder = channel_decoder
        self.decoder = decoder
        self.dense = dense

    def forward(
        self,
        z_noisy: torch.Tensor,
        trg_inp: torch.Tensor,
        look_ahead_mask: torch.Tensor,
        trg_padding_mask: torch.Tensor,
    ):
        ch_dec_out = self.channel_decoder(z_noisy)
        dec_out = self.decoder(trg_inp, ch_dec_out, look_ahead_mask, trg_padding_mask)
        logits = self.dense(dec_out)
        return logits, ch_dec_out, dec_out


# -----------------------------
# Loss
# -----------------------------
def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    # logits: [B,T,V], targets: [B,T]
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
        ignore_index=pad_idx,
    )
    valid = (targets.reshape(-1) != pad_idx).float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    temperature: float,
) -> torch.Tensor:
    # apply mask so PAD tokens don't dominate KD
    s_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)  # [B,T]
    valid = (targets != pad_idx).float()
    kl = (kl * valid).sum() / valid.sum().clamp_min(1.0)
    return kl * (temperature ** 2)


def masked_mse_loss(student_feat, teacher_feat, feat_mask=None):
    """
    student_feat: [B, L, D]
    teacher_feat: [B, L, D]
    feat_mask:    [B, L] with 1 for valid positions, 0 for padding
    """
    if feat_mask is None:
        return F.mse_loss(student_feat, teacher_feat)

    valid = feat_mask.float().unsqueeze(-1)   # [B, L, 1]
    diff = (student_feat - teacher_feat).pow(2) * valid
    return diff.sum() / valid.sum().clamp_min(1.0) / student_feat.size(-1)

# -----------------------------
# Build models
# -----------------------------
def build_teacher(
    num_vocab: int,
    max_len: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    dff: int,
    dropout: float,
    device: torch.device,
) -> DeepSC:
    model = DeepSC(
        num_layers=num_layers,
        src_vocab_size=num_vocab,
        trg_vocab_size=num_vocab,
        src_max_len=max_len,
        trg_max_len=max_len,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout=dropout,
    ).to(device)
    return model


def build_student_receiver_from_teacher(
    teacher: DeepSC,
    device: torch.device,
) -> ReceiverOnly:
    student = ReceiverOnly(
        channel_decoder=teacher.channel_decoder.__class__(
            teacher.channel_decoder.linear1.in_features,
            teacher.channel_decoder.linear1.out_features,
            teacher.channel_decoder.linear2.out_features,
        ),
        decoder=teacher.decoder.__class__(
            len(teacher.decoder.dec_layers),
            teacher.decoder.embedding.num_embeddings,
            teacher.decoder.pos_encoding.pe.size(1),
            teacher.decoder.d_model,
            teacher.decoder.dec_layers[0].self_mha.num_heads,
            teacher.decoder.dec_layers[0].ffn.w_1.out_features,
            teacher.decoder.pos_encoding.dropout.p,
        ),
        dense=nn.Linear(
            teacher.dense.in_features,
            teacher.dense.out_features,
        ),
    ).to(device)
    return student


def load_teacher_checkpoint(teacher: DeepSC, checkpoint_path: str, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(state, strict=True)


def maybe_init_student_from_teacher(student: ReceiverOnly, teacher: DeepSC, init_from_teacher: bool = True) -> None:
    if not init_from_teacher:
        return
    student.channel_decoder.load_state_dict(teacher.channel_decoder.state_dict(), strict=True)
    student.decoder.load_state_dict(teacher.decoder.state_dict(), strict=True)
    student.dense.load_state_dict(teacher.dense.state_dict(), strict=True)


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_student_receiver(student: ReceiverOnly, path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "channel_decoder": student.channel_decoder.state_dict(),
        "decoder": student.decoder.state_dict(),
        "dense": student.dense.state_dict(),
    }
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def save_shared_tx(teacher: DeepSC, path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "encoder": teacher.encoder.state_dict(),
        "channel_encoder": teacher.channel_encoder.state_dict(),
    }
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


# -----------------------------
# Train / eval
# -----------------------------
def teacher_tx_forward(
    teacher: DeepSC,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    channel_type: str,
    noise_std: float,
):
    with torch.no_grad():
        sem_feat = teacher.encoder(src, src_mask)
        tx_feat = teacher.channel_encoder(sem_feat)
        z_noisy = apply_channel(tx_feat, channel_type=channel_type, noise_std=noise_std)
    return z_noisy


def teacher_tr_forward(
    teacher: DeepSC,
    z_noisy: torch.Tensor,
    trg_inp: torch.Tensor,
    look_ahead_mask: torch.Tensor,
    trg_padding_mask: torch.Tensor,
):
    with torch.no_grad():
        t_rx = teacher.channel_decoder(z_noisy)
        t_dec = teacher.decoder(trg_inp, t_rx, look_ahead_mask, trg_padding_mask)
        t_logits = teacher.dense(t_dec)
    return t_logits, t_rx, t_dec


def train_one_epoch(
    teacher: DeepSC,
    student: ReceiverOnly,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    args,
):
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    for batch in loader:
        src = batch.to(device)
        trg = batch.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask, trg_padding_mask = create_masks(src, trg_inp, pad_idx)

        if args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = np.random.uniform(args.snr_db_low, args.snr_db_high)

        noise_std = snr_db_to_noise_std(float(snr_db))

        optimizer.zero_grad()

        z_noisy = teacher_tx_forward(
            teacher=teacher,
            src=src,
            src_mask=src_mask,
            channel_type=args.channel,
            noise_std=noise_std,
        )

        t_logits, t_rx, t_dec = teacher_tr_forward(
            teacher=teacher,
            z_noisy=z_noisy,
            trg_inp=trg_inp,
            look_ahead_mask=look_ahead_mask,
            trg_padding_mask=trg_padding_mask,
        )

        s_logits, s_rx, s_dec = student(
            z_noisy=z_noisy.detach(),
            trg_inp=trg_inp,
            look_ahead_mask=look_ahead_mask,
            trg_padding_mask=trg_padding_mask,
        )

        ce = masked_ce_loss(s_logits, trg_real, pad_idx)
        kd = kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
        # feat = masked_mse_loss(s_rx, t_rx.detach(), trg_real, pad_idx)
        rx_mask = (src != pad_idx).float()
        feat = masked_mse_loss(s_rx, t_rx.detach(), rx_mask)

        loss = args.alpha * ce + args.beta * kd + args.gamma * feat
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

        optimizer.step()

        total_loss += float(loss.item())
        total_ce += float(ce.item())
        total_kd += float(kd.item())
        total_feat += float(feat.item())

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kd": total_kd / n,
        "feat": total_feat / n,
    }


@torch.no_grad()
def validate_one_epoch(
    teacher: DeepSC,
    student: ReceiverOnly,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
    args,
):
    teacher.eval()
    student.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    for batch in loader:
        src = batch.to(device)
        trg = batch.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask, trg_padding_mask = create_masks(src, trg_inp, pad_idx)

        if args.val_snr_db is not None:
            snr_db = args.val_snr_db
        elif args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = (args.snr_db_low + args.snr_db_high) / 2.0

        noise_std = snr_db_to_noise_std(float(snr_db))

        z_noisy = teacher_tx_forward(
            teacher=teacher,
            src=src,
            src_mask=src_mask,
            channel_type=args.channel,
            noise_std=noise_std,
        )

        t_logits, t_rx, t_dec = teacher_tr_forward(
            teacher=teacher,
            z_noisy=z_noisy,
            trg_inp=trg_inp,
            look_ahead_mask=look_ahead_mask,
            trg_padding_mask=trg_padding_mask,
        )

        s_logits, s_rx, s_dec = student(
            z_noisy=z_noisy,
            trg_inp=trg_inp,
            look_ahead_mask=look_ahead_mask,
            trg_padding_mask=trg_padding_mask,
        )

        ce = masked_ce_loss(s_logits, trg_real, pad_idx)
        kd = kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
        feat = masked_mse_loss(s_rx, t_rx.detach(), trg_real, pad_idx)

        loss = args.alpha * ce + args.beta * kd + args.gamma * feat

        total_loss += float(loss.item())
        total_ce += float(ce.item())
        total_kd += float(kd.item())
        total_feat += float(feat.item())

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kd": total_kd / n,
        "feat": total_feat / n,
    }


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Receiver-only KD for DeepSC")

    # files
    parser.add_argument("--vocab-file", type=str, default="./data/train/europarl/vocab.json")
    parser.add_argument("--teacher-checkpoint", type=str, default="./checkpoints/deepsc-Rayleigh/checkpoint_100.pth")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/tr_kd")

    # model config (must match teacher checkpoint)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    # channel
    parser.add_argument("--channel", type=str, default="Rayleigh", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr-mode", type=str, default="range", choices=["fixed", "range"])
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-db-low", type=float, default=5.0)
    parser.add_argument("--snr-db-high", type=float, default=10.0)
    parser.add_argument("--val-snr-db", type=float, default=8.0)

    # KD
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="CE weight")
    parser.add_argument("--beta", type=float, default=0.5, help="KD weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Feature MSE weight")
    parser.add_argument("--init-student-from-teacher", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    token_to_idx = vocab["token_to_idx"]
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"] if "<PAD>" in token_to_idx else token_to_idx[""]

    train_set = EurDataset("train")
    test_set = EurDataset("test")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_data,
    )
    val_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_data,
    )

    teacher = build_teacher(
        num_vocab=num_vocab,
        max_len=num_vocab,   # keep same style as original repo
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        dropout=args.dropout,
        device=device,
    )

    load_teacher_checkpoint(teacher, args.teacher_checkpoint, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_student_receiver_from_teacher(teacher, device=device)
    maybe_init_student_from_teacher(
        student=student,
        teacher=teacher,
        init_from_teacher=args.init_student_from_teacher,
    )

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    best_val = float("inf")

    # save shared TX once
    save_shared_tx(
        teacher,
        os.path.join(args.save_dir, "shared_tx.pth"),
        meta={
            "channel": args.channel,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dff": args.dff,
        },
    )

    for epoch in range(args.epochs):
        start = time.time()

        train_stats = train_one_epoch(
            teacher=teacher,
            student=student,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_idx=pad_idx,
            args=args,
        )

        val_stats = validate_one_epoch(
            teacher=teacher,
            student=student,
            loader=val_loader,
            device=device,
            pad_idx=pad_idx,
            args=args,
        )

        elapsed = time.time() - start

        print(
            f"[Epoch {epoch+1:03d}] "
            f"train_loss={train_stats['loss']:.5f} "
            f"(ce={train_stats['ce']:.5f}, kd={train_stats['kd']:.5f}, feat={train_stats['feat']:.5f}) | "
            f"val_loss={val_stats['loss']:.5f} "
            f"(ce={val_stats['ce']:.5f}, kd={val_stats['kd']:.5f}, feat={val_stats['feat']:.5f}) | "
            f"time={elapsed:.1f}s"
        )

        latest_path = os.path.join(args.save_dir, "student_tr_latest.pth")
        save_student_receiver(
            student,
            latest_path,
            meta={
                "epoch": epoch + 1,
                "val_loss": val_stats["loss"],
                "temperature": args.temperature,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "channel": args.channel,
            },
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_path = os.path.join(args.save_dir, "student_tr_best.pth")
            save_student_receiver(
                student,
                best_path,
                meta={
                    "epoch": epoch + 1,
                    "val_loss": val_stats["loss"],
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "gamma": args.gamma,
                    "channel": args.channel,
                },
            )
            print(f"  -> saved best student TR to: {best_path}")


if __name__ == "__main__":
    main()