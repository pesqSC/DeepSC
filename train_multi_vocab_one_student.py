import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_multilingual import EurParallelDataset, collate_parallel
from dataset import EurDataset, collate_data

from datetime import date
import argparse
import json
import os
import random
import time
import numpy as np

from models.transceiver import DeepSC
from student import Student
from teacher import build_teacher
from models.rx_model import Receiver
from models.tx_model import Transmitter
from utils import create_masks, loss_function, validate_multi_epoch, save_student_receiver
from utils import (
    kd_kl_loss, 
    masked_ce_loss, 
    feature_distillation_loss, 
    SNR_to_noise, 
    masked_ce_loss2,
    save_epoch_results
)


def parse_args():
    parser = argparse.ArgumentParser(description="Receiver-only KD for DeepSC")

    # files
    parser.add_argument("--vocab-file", type=str, default="./data/train/europarl/vocab_multilingual.json")
    parser.add_argument("--teacher-checkpoint", type=str, default="./checkpoints/deepsc-Rayleigh/multilingual/2026-08-09")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/deepsc-Rayleigh/multi_vocab_kd")

    # model config (must match teacher checkpoint)
    parser.add_argument("--max-len", type=int, default=33)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--batch-size", type=int, default=60)
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
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="CE weight")
    parser.add_argument("--beta", type=float, default=0.5, help="KD weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Feature MSE weight")
    parser.add_argument("--init-student-from-teacher", action="store_true")
    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)

    parser.add_argument('--pt', default='pt_pt', type=str)
    parser.add_argument('--pt-pt', default='pt_en', type=str)
    parser.add_argument('--pt-es', default='pt_es', type=str)
    parser.add_argument('--pt-fr', default='pt_fr', type=str)

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

def validate(epoch, args, pad_idx, criterion, net, device):
    test_eur = EurParallelDataset(args.en, 'test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_parallel)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)
            loss = val_step(net, src, trg, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)

@torch.no_grad()
def validate_epoch(
    epoch,
    transmitter,
    teacher: Receiver,
    student: Student,
    val_loader,
    pad_idx,
    channel,
    noise_std,
    device: torch.device,
    criterion,
    args,
):
    transmitter.eval()
    teacher.eval()
    student.eval()


    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    pbar = tqdm(val_loader)

    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)

            trg_inp = trg[:, :-1]
            trg_real = trg[:, 1:]

            src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

            tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
                src, 
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

            s_logits, s_ch_dec_out, s_dec_out = student(
                z_noisy, 
                trg_inp, 
                look_ahead_mask, 
                src_mask
            )

            ce = masked_ce_loss(
                s_logits,
                trg_real,
                pad_idx
            )

            # ce = masked_ce_loss(s_logits, trg_real, pad_idx)
            # ce_s1 = loss_function(
            #     s1_logits.contiguous().view(-1, s1_logits.size(-1)),
            #     trg_real.contiguous().view(-1), 
            #     pad_idx, 
            #     criterion
            # )

            kd = kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)

            src_valid = (src != pad_idx).float()

            feat = feature_distillation_loss(
                student_feat=s_ch_dec_out, 
                teacher_feat=rx_ch_dec_out.detach(), 
                targets=src,
                pad_idx=pad_idx,
            )

            loss_s = (args.alpha * ce) + (args.beta * kd) + args.gamma * feat

            total_loss += float(loss_s.item())
            total_ce += float(args.alpha * ce.item())
            total_kd += float(args.beta * kd.item())
            total_feat += float(feat.item())

            pbar.set_description(f"Epoch {epoch + 1} Valid")

            pbar.set_postfix(
                Loss=f"{loss_s.item():.3f}",
                CE=f"{s_ce.item():.3f}",
                KD=f"{kd_s.item():.3f}",
                FEAT=f"{feat.item():.3f}"
            )

    n = max(len(val_loader), 1)

    return{
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kd": total_kd / n,
        "feat": total_feat / n,
    }


def train(
    transmitter: Transmitter, 
    teacher: Receiver, 
    student: Student, 
    train_loader: DataLoader, 
    optimizer: optim.Adam,
    pad_idx,
    channel,
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

    student.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    num_batches=0.0

    pbar = tqdm(train_loader)

    for src, trg in pbar:
        # print(batch.shape)
        src = src.to(device)
        trg = trg.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        opt = optimizer

        opt.zero_grad()

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

        if args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = np.random.uniform(
                args.snr_db_low,
                args.snr_db_high
            )

        noise_std = SNR_to_noise(snr_db)

        with torch.no_grad():
            _, _, _, z_noisy = transmitter(
                src, 
                src_mask, 
                channel, 
                noise_std
            )
            
            t_logits, t_rx_ch, _ = teacher(
                z_noisy=z_noisy, 
                trg_inp=trg_inp, 
                look_ahead_mask=look_ahead_mask,
                src_mask=src_mask
            )

        s_logits, s_rx_ch, s_dec_out = student(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )

        s_ce = masked_ce_loss(
            s_logits,
            trg_real,
            pad_idx
        )

        # s1_ce = loss_function(
        #     s1_logits.contiguous().view(-1, s1_logits.size(-1)),
        #     trg_real.contiguous().view(-1),
        #     pad_idx,
        #     criterion
        # )

        kd = kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)

        src_valid = (src != pad_idx).float()

        feat = feature_distillation_loss(
            student_feat=s1_rx_ch,
            teacher_feat=t_rx_ch.detach(),
            targets=src,
            pad_idx=pad_idx,
        )

        # feat = masked_ce_loss(s_ch_dec_out, rx_ch_dec_out.detach(), pad_idx)
        # feat =  loss_function(
        #     s_ch_dec_out.contiguous().view(-1, s_ch_dec_out.size(-1)), 
        #     rx_ch_dec_out.detach().contiguous().view(-1), 
        #     pad_idx, 
        #     criterion
        # )

        # feat = feature_distillation_loss(s_ch_dec_out, rx_ch_dec_out.detach(), trg_real, pad_idx)

        # loss = args.alpha * ce + args.beta * kd + args.gamma * feat
        loss = (args.alpha * ce) + (args.beta * kd) + args.gamma * feat

        loss.backward()
        
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student_1.parameters(), args.grad_clip)
            # torch.nn.utils.clip_grad_norm_(student_2.parameters(), args.grad_clip)
        
        opt.step()
        # opt_s_2.step()

        total_loss += float(loss.item())
        total_ce += float(ce.item())
        total_kd += float(kd.item())
        total_feat += float(feat.item())
        
        num_batches += 1

        pbar.set_description(f"Epoch {epoch + 1} Train")

        pbar.set_postfix(
            Loss=f"{loss.item():.3f}",
            CE=f"{ce.item():.3f}",
            KD=f"{kd.item():.3f}",
            TF=f"{feat.item():.3f}"
        )

    n = len(train_loader)

    return {
            "loss": total_loss / n,
            "ce": total_ce / n,
            "kd": total_kd / n,
            "tf": total_feat / n
        }

def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    token_to_idx = vocab["token_to_idx"]
    num_vocab = len(token_to_idx)
    
    pad_idx = token_to_idx["<PAD>"] if "<PAD>" in token_to_idx else token_to_idx[""]
    print(pad_idx)
    
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    train_set = EurParallelDataset(args.en, 'train')
    test_set = EurParallelDataset(args.en, "test")

    # load dataset
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_parallel,
    )
    val_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_parallel,
    )

    deepsc = DeepSC(
        args.num_layers, 
        num_vocab,
        num_vocab,
        args.max_len,
        args.max_len,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1
    ).to(device)

    # deepsc = build_teacher(
    #     num_vocab, 
    #     args.max_len, 
    #     args.num_layers, 
    #     args.d_model, 
    #     args.num_heads, 
    #     args.dff, 
    #     args.dropout, 
    #     device
    # )
    enc_model_path = os.path.join(args.teacher_checkpoint, 'encoder_35.pth')
    dec_model_path = os.path.join(args.teacher_checkpoint, 'decoder_35.pth')

    enc_checkpoint = torch.load(enc_model_path, map_location=device)
    dec_checkpoint = torch.load(dec_model_path, map_location=device)

    # Load the models
    deepsc.encoder.load_state_dict(enc_checkpoint['encoder'])
    deepsc.channel_encoder.load_state_dict(enc_checkpoint['channel_encoder'])
    deepsc.channel_decoder.load_state_dict(dec_checkpoint['channel_decoder'])
    deepsc.decoder.load_state_dict(dec_checkpoint['decoder'])
    deepsc.dense.load_state_dict(dec_checkpoint['dense'])

    deepsc.eval()
    # encoder = deepsc.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    # deep_sc.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    # deep_sc.load_state_dict(torch.load('deepsc_12n.pth', map_location=device))
    # deep_sc.eval()

    transmitter = Transmitter(deepsc.encoder, deepsc.channel_encoder)
    
    receiver = Receiver(
        channel_decoder=deepsc.channel_decoder, 
        decoder=deepsc.decoder, 
        dense=deepsc.dense
    )

    student= Student(
        2, 
        num_vocab, 
        num_vocab, 
        args.max_len, 
        args.max_len, 
        args.d_model, 
        4, 
        args.dff, 
        args.dropout
    ).to(device)

    # noise_std = snr_db_to_noise_std(float(args.snr_db))
    noise_std = np.random.uniform(
            SNR_to_noise(args.snr_db_low), 
            SNR_to_noise(args.snr_db_high), 
            # size=(1)
        )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer= torch.optim.Adam(
            student.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay = 5e-4
        )
    
    best_val = float("inf")

    os.makedirs(args.save_dir, exist_ok=True)

    today = date.today()
    root_dir = args.save_dir + f'/one_student/{today.strftime("%Y-%m-%d")}'
    
    os.makedirs(root_dir, exist_ok=True)

    csv_path = f"{root_dir}/results_{today.strftime('%Y-%m-%d')}.csv"

    for epoch in range(args.epochs):
        start_time = time.time()

        train_stats = train(
            transmitter=transmitter, 
            teacher=receiver, 
            student=student, 
            train_loader=train_loader, 
            optimizer=optimizer,
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            criterion=criterion,
            epoch=epoch,
            args=args
        )

        val_stats = validate_epoch(
            epoch=epoch,
            transmitter=transmitter, 
            teacher=receiver, 
            students=student, 
            val_loader=val_loader, 
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            noise_std=noise_std,
            criterion=criterion,
            args=args
        )

        elapsed = time.time() - start_time

        epoch_metrics = {
            "loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "ce": train_stats["ce"],
            "kd": train_stats["kd"],
            "tf": train_stats["feature"],
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "temperature": args.temperature,
            "time": elapsed
        }

        save_epoch_results(
            csv_path=csv_path,
            epoch=epoch + 1,
            metrics=epoch_metrics,
        )

        latest_path = os.path.join(
            root_dir, f"student_{epoch+1:02d}.pth"
        )
        # save the student model by epoch
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

        # save the best student model        
        if val_stats["loss"] < best_val:
            best_val[i] = val_stats[i]["loss"]

            best_path = os.path.join(
                root_dir,
                "student_best.pth"
            )

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