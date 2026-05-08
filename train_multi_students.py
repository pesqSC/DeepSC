import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import EurDataset, collate_data

import argparse
import json
import os
import random
import time
import numpy as np

from student import Student
from teacher import build_teacher
from models.rx_model import Receiver
from models.tx_model import Transmitter
from utils import create_masks, loss_function, validate_one_epoch, save_student_receiver
from utils import kd_kl_loss, masked_ce_loss, feature_distillation_loss, SNR_to_noise



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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="CE weight")
    parser.add_argument("--beta", type=float, default=0.5, help="KD weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Feature MSE weight")
    parser.add_argument("--init-student-from-teacher", action="store_true")

    return parser.parse_args()



def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def snr_db_to_noise_std(snr_db: float) -> float:
    # assuming unit power
    return 10 ** (-snr_db / 20.0)

# def train_step():

def train(
    transmitter: Transmitter, 
    teacher: Receiver, 
    students: [Student], 
    train_loader: DataLoader, 
    optimizer: [optim.Adam],
    pad_idx,
    channel,
    noise_std,
    device: torch.device,
    criterion,
    epoch,
    args
    ):
    
    for p in teacher.parameters():
        p.requires_grad = False
    
    for p in transmitter.parameters():
        p.requires_grad = False
    
    transmitter.eval()
    teacher.eval()

    student_1 = students[0]
    student_2 = students[1]
    
    student_1.train()
    student_2.train()

    total_loss_s1 = 0.0
    total_ce_s1 = 0.0
    total_kd_s1 = 0.0

    total_loss_s2 = 0.0
    total_ce_s2 = 0.0
    total_kd_s2 = 0.0

    pbar = tqdm(train_loader)

    for batch in pbar:
        # print(batch.shape)
        sents = batch.to(device)
        targets = batch.to(device)

        trg_inp = targets[:, :-1]
        trg_real = targets[:, 1:]

        opt_s_1 = optimizer[0]
        opt_s_2 = optimizer[1]

        opt_s_1.zero_grad()
        opt_s_2.zero_grad()

        src_mask, look_ahead_mask = create_masks(sents, trg_inp, pad_idx)

        with torch.no_grad():
            tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
                sents, 
                src_mask, 
                channel, 
                noise_std
            )
            
            t_logits, rx_ch_dec_out, rx_dec_out = teacher(
                z_noisy=z_noisy, 
                trg_inp=trg_inp, 
                look_ahead_mask=look_ahead_mask,
                src_mask=src_mask
            )

        s1_logits, s1_ch_dec_out, s1_dec_out = student_1(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )


        s2_logits, s2_ch_dec_out, s2_dec_out = student_2(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )

        s1_ce = loss_function(
            s1_logits.contiguous().view(-1, s1_logits.size(-1)),
            trg_real.contiguous().view(-1),
            pad_idx,
            criterion
        )

        s2_ce = loss_function(
            s2_logits.contiguous().view(-1, s2_logits.size(-1)),
            trg_real.contiguous().view(-1),
            pad_idx,
            criterion
        )

        kd_s1 = kd_kl_loss(s1_logits, t_logits, trg_real, pad_idx, args.temperature)
        kd_s2 = kd_kl_loss(s2_logits, t_logits, trg_real, pad_idx, args.temperature)

        # feat = masked_ce_loss(s_ch_dec_out, rx_ch_dec_out.detach(), pad_idx)
        # feat =  loss_function(
        #     s_ch_dec_out.contiguous().view(-1, s_ch_dec_out.size(-1)), 
        #     rx_ch_dec_out.detach().contiguous().view(-1), 
        #     pad_idx, 
        #     criterion
        # )

        # feat = feature_distillation_loss(s_ch_dec_out, rx_ch_dec_out.detach(), trg_real, pad_idx)

        # loss = args.alpha * ce + args.beta * kd + args.gamma * feat
        loss_s1 = args.alpha * s1_ce + args.beta * kd_s1
        loss_s2 = args.alpha * s2_ce + args.beta * kd_s2

        loss_s1.backward()
        loss_s2.backward()

        pbar.set_description(
                'Epoch: {};  Type: Train; Loss_s1: {:.4f}\nLoss_s2: {:.4f}'
                .format(epoch + 1, loss_s1.item(), loss_s2.item())
                .format(args.current_epoch + 1, loss_s1.item(), loss_s2.item())
            )
        
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student_1.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(student_2.parameters(), args.grad_clip)
        
        opt_s_1.step()
        opt_s_2.step()

        total_loss_s1 += float(loss_s1.item())
        total_ce_s1 += float(s1_ce.item())
        total_kd_s1 += float(kd_s1.item())
        # total_feat += float(feat.item())
        
        total_loss_s2 += float(loss_s2.item())
        total_ce_s2 += float(s2_ce.item())
        total_kd_s2 += float(kd_s2.item())
        # pbar.set_description(f"Loss: {loss.item():.4f}")
    
    n = len(train_loader)
    return [
        {
            "loss": total_loss_s1 / n,
            "ce": total_ce_s1 / n,
            "kd": total_kd_s1 / n,
            # "feat": total_feat / n
        },
        {
            "loss": total_loss_s2 / n,
            "ce": total_ce_s2 / n,
            "kd": total_kd_s2 / n,
            # "feat": total_feat / n
        }
    ]

def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    token_to_idx = vocab["token_to_idx"]
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"] if "<PAD>" in token_to_idx else token_to_idx[""]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

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

    deep_sc = build_teacher(
        num_vocab, 
        args.max_len, 
        args.num_layers, 
        args.d_model, 
        args.num_heads, 
        args.dff, 
        args.dropout, 
        device
    )

    
    deep_sc.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    # deep_sc.load_state_dict(torch.load('deepsc_12n.pth', map_location=device))
    deep_sc.eval()

    transmitter = Transmitter(deep_sc.encoder, deep_sc.channel_encoder)
    
    receiver = Receiver(
        channel_decoder=deep_sc.channel_decoder, 
        decoder=deep_sc.decoder, 
        dense=deep_sc.dense
    )

    student_1 = Student(
        2, 
        num_vocab, 
        num_vocab, 
        num_vocab, 
        num_vocab, 
        args.d_model, 
        args.num_heads, 
        args.dff, 
        args.dropout
    ).to(device)
    
    student_2 = Student(
        2, 
        num_vocab, 
        num_vocab, 
        num_vocab, 
        num_vocab, 
        args.d_model, 
        args.num_heads, 
        args.dff, 
        args.dropout
    ).to(device)

    students = [student_1, student_2]

    # noise_std = snr_db_to_noise_std(float(args.snr_db))
    noise_std = np.random.uniform(
            SNR_to_noise(args.snr_db_low), 
            SNR_to_noise(args.snr_db_high), 
            # size=(1)
        )

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer_1 = torch.optim.Adam(
            student_1.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay = 5e-4
        )
    
    optimizer_2 = torch.optim.Adam(
            student_2.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay = 5e-4
        )

    optimizer = [optimizer_1, optimizer_2]

    best_val = [float("inf"), float("inf")]
    
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        start_time = time.time()

        train_stats = train(
            transmitter=transmitter, 
            teacher=receiver, 
            students=students, 
            train_loader=train_loader, 
            optimizer=optimizer,
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            noise_std=noise_std,
            criterion=criterion,
            epoch=epoch,
            args=args
        )

        s1_val_stats = validate_one_epoch(
            transmitter=transmitter, 
            teacher=receiver, 
            student=student[0], 
            val_loader=val_loader, 
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            noise_std=noise_std,
            criterion=criterion,
            args=args
        )

        s2_val_stats = validate_one_epoch(
            transmitter=transmitter, 
            teacher=receiver, 
            student=students[1], 
            val_loader=val_loader, 
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            noise_std=noise_std,
            criterion=criterion,
            args=args
        )

        elapsed = time.time() - start_time
        val_stats = [s1_val_stats, s2_val_stats]

        for i in range(len(students)):
            print(
                f"[Epoch {epoch+1:03d}] Student {i+1} "
                f"train_loss={train_stats[i]['loss']:.5f} "
                f"(ce={train_stats[i]['ce']:.5f}, kd={train_stats[i]['kd']:.5f}) | "
                f"val_loss={val_stats[i]['loss']:.5f} "
                f"(ce={val_stats[i]['ce']:.5f}, kd={val_stats[i]['kd']:.5f}) | "
                f"time={elapsed:.1f}s"
            )

        # latest_path = os.path.join(args.save_dir, "student_tr_{}.pth".format(epoch + 1).zfill(2))
        # save_student_receiver(
        #     student,
        #     latest_path,
        #     meta={
        #         "epoch": epoch + 1,
        #         "val_loss": val_stats["loss"],
        #         "temperature": args.temperature,
        #         "alpha": args.alpha,
        #         "beta": args.beta,
        #         "gamma": args.gamma,
        #         "channel": args.channel,
        #     },
        # )

        for i, student in enumerate(students):
            latest_path = os.path.join(
                args.save_dir,
                f"student{i+1}/student_{i+1}_tr_{epoch+1:02d}.pth"
            )

            save_student_receiver(
                student,
                latest_path,
                meta={
                    "student_id": i + 1,
                    "epoch": epoch + 1,
                    "val_loss": val_stats[i]["loss"],
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "gamma": args.gamma,
                    "channel": args.channel,
                },
            )

            if val_stats[i]["loss"] < best_val[i]:
                best_val[i] = val_stats[i]["loss"]

                best_path = os.path.join(
                    args.save_dir,
                    f"student_{i+1}_tr_best.pth"
                )

                save_student_receiver(
                    student,
                    best_path,
                    meta={
                        "student_id": i + 1,
                        "epoch": epoch + 1,
                        "val_loss": val_stats[i]["loss"],
                        "temperature": args.temperature,
                        "alpha": args.alpha,
                        "beta": args.beta,
                        "gamma": args.gamma,
                        "channel": args.channel,
                    },
                )

                print(f"  -> saved best student {i+1} TR to: {best_path}")

        # if val_stats["loss"] < best_val:
        #     best_val = val_stats["loss"]
        #     best_path = os.path.join(args.save_dir, "student_tr_best.pth")
        #     save_student_receiver(
        #         student,
        #         best_path,
        #         meta={
        #             "epoch": epoch + 1,
        #             "val_loss": val_stats["loss"],
        #             "temperature": args.temperature,
        #             "alpha": args.alpha,
        #             "beta": args.beta,
        #             "gamma": args.gamma,
        #             "channel": args.channel,
        #         },
        #     )
        #     print(f"  -> saved best student TR to: {best_path}")


if __name__ == "__main__":
    main()