import argparse
import json
import torch
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import date
from functools import partial

from dataset_multilingual import EurParallelDataset, collate_parallel
from models.transceiver import DeepSC
from student import Student
from models.tx_model import Transmitter

from models.lora2 import (
    apply_lora_to_decoder,
    enable_language_adaptation,
    save_language_adapter,
)

from utils import (
    SNR_to_noise, 
    create_masks, 
    setup_seed, 
    loss_function,
    Channels,
    save_epoch_results,
    build_differential_optimizer
)


def val_step(transmitter, LoRA, src, trg, pad, criterion, channel, noise_std):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    with torch.no_grad():
        _, _, _, z_noisy = transmitter(
            src, 
            src_mask, 
            channel, 
            noise_std
        )

        logits, _, _ = LoRA(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )

    ntokens = logits.size(-1)
    loss = loss_function(
        logits.contiguous().view(-1, ntokens), 
        trg_real.contiguous().view(-1), 
        pad, 
        criterion
    )
    
    return loss.item()


def validate(
    epoch, 
    args, 
    transmitter, 
    LoRA, 
    criterion, 
    device, 
    pad_idx,
    noise_std,
    train_lag
):
    test_eur = EurParallelDataset(train_lag, 'test')

    test_iterator = DataLoader(
        test_eur, 
        batch_size=args.batch_size, 
        num_workers=2,
        pin_memory=True, 
        collate_fn=partial(collate_parallel, pad_idx=pad_idx)
    )

    transmitter.eval()
    LoRA.eval()

    total_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_iterator, desc=f"Epoch {epoch + 1} VAL")
        for src, trg in pbar:
            src = src.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            loss = val_step(
                transmitter,
                LoRA, 
                src, 
                trg, 
                pad_idx,
                criterion, 
                args.channel,
                noise_std
            )

            total_loss += loss
            pbar.set_postfix(loss=f"{loss:.5f}")

    return total_loss / len(test_iterator)


def train_lora_epoch(
    epoch, 
    transmitter, 
    LoraModel, 
    loader, 
    optimizer, 
    device, 
    pad_idx,
    criterion,
    args,
):
    LoraModel.train()
    transmitter.eval()
    
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} Train")

    for src, trg in pbar:
        src = src.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

        snr_db = np.random.uniform(args.snr_db_low, args.snr_db_high)
        noise_std = SNR_to_noise(snr_db)

        optimizer.zero_grad()

        # Frozen Transmitter Forward Pass
        with torch.no_grad():
            _, _, _, z_noisy = transmitter(
                src, 
                src_mask, 
                args.channel, 
                noise_std
            )

        # Trainable Receiver/Decoder Pass
        logits, _, _ = LoraModel(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )

        ntokens = logits.size(-1)
        loss = loss_function(
            logits.contiguous().view(-1, ntokens), 
            trg_real.contiguous().view(-1), 
            pad_idx, 
            criterion
        )

        loss.backward()
        
        # Optional Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(LoraModel.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab-file", default="data/train/europarl/vocab_multilingual.json")
    parser.add_argument("--student-checkpoint", default="./checkpoints/deepsc-Rayleigh/multi_vocab_kd/one_student/2026-08-12")
    parser.add_argument("--transmitter-checkpoint", type=str, default="./checkpoints/deepsc-Rayleigh/multilingual/2026-08-09")
    parser.add_argument("--save-lora", default="./checkpoints/deepsc-Rayleigh/lora")

    parser.add_argument("--channel", default="Rayleigh", type=str, choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr-db-low", type=float, default=5.0)
    parser.add_argument("--snr-db-high", type=float, default=10.0)
    parser.add_argument("--val-snr-db", type=float, default=8.0)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    # Adapter hyperparameters
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=[
                                                        "self_q", 
                                                        "self_v",
                                                        "src_q",
                                                        "src_v", 
                                                    ])

    parser.add_argument("--adapt-heads-and-norms", action="store_true", default=True, help="Unfreeze Embeddings, Output Head, and LayerNorms alongside LoRA")

    parser.add_argument("--embedding-lr", type=float, default=2e-5)
    parser.add_argument("--output-lr", type=float, default=5e-5)
    parser.add_argument("--norm-lr", type=float, default=2e-5)

    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)

    parser.add_argument('--MAX_LEN', default=33, type=int)

    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)

    parser.add_argument('--pt', default='pt_pt', type=str)
    parser.add_argument('--pt-pt', default='pt_en', type=str)
    parser.add_argument('--pt-es', default='pt_es', type=str)
    parser.add_argument('--pt-fr', default='pt_fr', type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(42)

    vocab = json.load(open(args.vocab_file, "r", encoding="utf-8"))
    token_to_idx = vocab["token_to_idx"]
    vocab_size = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]

    train_lag = args.en_pt
    dataset = EurParallelDataset(train_lag, 'train')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(collate_parallel, pad_idx=pad_idx)
    )

    deepsc = DeepSC(
        args.num_layers, vocab_size, vocab_size, 
        args.MAX_LEN, args.MAX_LEN, args.d_model, 
        args.num_heads, args.dff, 0.1
    )
    
    transmitter = Transmitter(deepsc.encoder, deepsc.channel_encoder).to(device)

    student = Student(
        2, 
        vocab_size,
        vocab_size, 
        args.MAX_LEN,
        args.MAX_LEN, 
        args.d_model, 
        4, 
        args.dff, 
        0.1
    ).to(device)

    # Load Weights
    enc_model_path = os.path.join(args.transmitter_checkpoint, 'encoder_35.pth')
    student_1_path = os.path.join(args.student_checkpoint, 'student_03.pth')

    enc_checkpoint = torch.load(enc_model_path, map_location=device)
    transmitter.encoder.load_state_dict(enc_checkpoint['encoder'])
    transmitter.channel_encoder.load_state_dict(enc_checkpoint['channel_encoder'])

    student_1_checkpoint = torch.load(student_1_path, map_location=device)
    student.channel_decoder.load_state_dict(student_1_checkpoint['channel_decoder'])
    student.decoder.load_state_dict(student_1_checkpoint['decoder'])
    student.dense.load_state_dict(student_1_checkpoint['dense'])

    # Freeze Transmitter
    transmitter.eval()
    for param in transmitter.parameters():
        param.requires_grad = False

    # Inject LoRA
    LoRA = apply_lora_to_decoder(
        student,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_targets,
    ).to(device)

    # Optionally Enable Extra Language Adaptation Modules
    if args.adapt_heads_and_norms:
        LoRA = enable_language_adaptation(
            LoRA,
            train_embedding=True,
            train_output_head=True,
            train_layer_norm=True,
        )

    # Build Optimizer Parameter Groups
    # lora_params, embedding_params, output_params, norm_params = [], [], [], []

    # for name, parameter in LoRA.named_parameters():
    #     if not parameter.requires_grad:
    #         continue

    #     if "lora_" in name:
    #         lora_params.append(parameter)
    #     elif name.startswith("decoder.embedding"):
    #         embedding_params.append(parameter)
    #     elif name.startswith("dense"):
    #         output_params.append(parameter)
    #     elif "layernorm" in name:
    #         norm_params.append(parameter)

    # parameter_groups = []
    # if lora_params:
    #     parameter_groups.append({"params": lora_params, "lr": args.lr, "name": "lora"})
    # if embedding_params:
    #     parameter_groups.append({"params": embedding_params, "lr": args.embedding_lr, "name": "embedding"})
    # if output_params:
    #     parameter_groups.append({"params": output_params, "lr": args.output_lr, "name": "output"})
    # if norm_params:
    #     parameter_groups.append({"params": norm_params, "lr": args.norm_lr, "name": "layernorm"})

    # optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    optimizer = build_differential_optimizer(
        model=LoRA,
        lr_lora=args.lr,
        lr_head_embed=args.embedding_lr,
        lr_norm=args.norm_lr,
        weight_decay=1e-4
    )

    # Setup directories
    today = date.today()
    root_dir = os.path.join(args.save_lora, train_lag, today.strftime("%Y-%m-%d"))
    os.makedirs(root_dir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    best_val_loss = float('inf')
    val_noise_std = SNR_to_noise(args.val_snr_db)

    for epoch in range(args.epochs):
        loss = train_lora_epoch(
            epoch, transmitter, LoRA, loader, optimizer, device, pad_idx, criterion, args
        )

        val_loss = validate(
            epoch=epoch, args=args, transmitter=transmitter, LoRA=LoRA,
            criterion=criterion, device=device, pad_idx=pad_idx,
            noise_std=val_noise_std, train_lag=train_lag
        )

        save_epoch_results(
            os.path.join(root_dir, 'results.csv'),
            epoch,
            {'epoch': epoch, 'loss': loss, 'val_loss': val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_language_adapter(
                epoch=epoch,
                model=LoRA,
                save_dir=root_dir,
                adapter_name=f'{train_lag}_best',
                optimizer=optimizer,
            )
            print(f"[*] New best validation loss: {val_loss:.5f}. Adapter saved to {root_dir}")


if __name__ == "__main__":
    main()