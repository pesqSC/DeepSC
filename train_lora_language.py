import argparse
import json
import torch
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import date

from dataset_multilingual import EurParallelDataset, collate_parallel
from models.transceiver import DeepSC
from student import Student
from models.tx_model import Transmitter
from models.lora import (
    apply_lora_to_decoder,
    enable_language_adaptation,
    adaptation_parameters,
    save_language_adapter,
)
from utils import (
    SNR_to_noise, 
    create_masks, 
    setup_seed, 
    validate_multi_epoch,
    loss_function,
    Channels,
    save_epoch_results
)

def val_step(transmitter, LoRA, src, trg, n_var, pad, criterion, channel, noise_std):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
        src, 
        src_mask, 
        channel, 
        noise_std
    )

    logits, rx_ch, dec_out = LoRA(
        z_noisy, 
        trg_inp, 
        look_ahead_mask, 
        src_mask
    )

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = logits.size(-1)
    loss = loss_function(logits.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)
    
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
                        num_workers=0,
                        pin_memory=True, 
                        collate_fn=lambda batch: collate_parallel(
                            batch,
                            pad_idx,
                        )
                    )

    transmitter.eval()

    for parameter in transmitter.parameters():
        parameter.requires_grad = False

    LoRA.eval()

    pbar = tqdm(test_iterator)
    
    total = 0.0
    
    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)
            loss = val_step(transmitter,LoRA, src, trg, 0.1, pad_idx,
                             criterion, args.channel,noise_std)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)

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
    total_loss = 0.0
    num_batches=0.0

    # noise_std = SNR_to_noise(snr)
    
    pbar = tqdm(loader)

    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

        snr_db = np.random.uniform(
            args.snr_db_low,
            args.snr_db_high,
        )

        noise_std = SNR_to_noise(snr_db)

        optimizer.zero_grad()

        with torch.no_grad():
            tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
                src, 
                src_mask, 
                args.channel, 
                noise_std
            )

        logits, l1_ch_dec_out, l1_dec_out = LoraModel(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )

        # logits = transmitter(
        #     src,
        #     trg_input,
        #     src_mask,
        #     look_ahead_mask,
        #     noise_std,
        #     channel
        # )
        ntokens = logits.size(-1)

        loss = loss_function(logits.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad_idx, criterion)
        # loss = torch.nn.functional.cross_entropy(
        #     l1_logits.reshape(-1, l1_logits.size(-1)),
        #     trg_real.reshape(-1),
        #     ignore_index=pad_idx
        # )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_description(f"Epoch {epoch + 1} Train; Loss: {loss.item():.5f}")

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab-file", default="data/train/europarl/vocab_multilingual.json")
    # parser.add_argument("--train-data", required=True)
    parser.add_argument("--student-checkpoint", default="./checkpoints/deepsc-Rayleigh/multi_vocab_kd")
    parser.add_argument("--transmitter-checkpoint", type=str, default="./checkpoints/deepsc-Rayleigh/multilingual")
    parser.add_argument("--save-lora", default="./checkpoints/deepsc-Rayleigh/lora")

    parser.add_argument("--channel", default="Rayleigh", type=str, choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr", default=12, type=float)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-db-low", type=float, default=5.0)
    parser.add_argument("--snr-db-high", type=float, default=10.0)
    parser.add_argument("--val-snr-db", type=float, default=8.0)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--embedding-lr", type=float, default=2e-5)
    parser.add_argument("--output-lr", type=float, default=5e-5)
    parser.add_argument("--norm-lr", type=float, default=2e-5)

    parser.add_argument("--num-layers", default=12, type=int)
    parser.add_argument("--num-heads", default=16, type=int)
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

    train_lag = args.pt_es

    dataset = EurParallelDataset(train_lag, 'train')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_parallel(b, pad_idx)
    )

    deepsc = DeepSC(
        args.num_layers,
        vocab_size,
        vocab_size,
        args.MAX_LEN,
        args.MAX_LEN,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1
    )
    
    transmitter = Transmitter(deepsc.encoder, deepsc.channel_encoder).to(device)

    student =  Student(
        2,
        vocab_size,
        vocab_size,
        args.MAX_LEN,
        args.MAX_LEN,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1
    ).to(device)

    enc_model_path = os.path.join(f'{args.transmitter_checkpoint}/2026-07-30', 'encoder_50.pth')
    # dec_model_path = 'decoder_20.pth'
    # student_path = 'checkpoints/tr_kd/student_tr_best.pth'
    student_1_path = os.path.join(args.student_checkpoint, 'student_1_tr_best.pth')
    student_2_path = 'student_2_mult_best.pth'

    # enc_checkpoint = torch.load(os.path.join(my_vars.checkpoint_path, enc_model_path), map_location=device)
    # dec_checkpoint = torch.load(os.path.join(my_vars.checkpoint_path, dec_model_path), map_location=device)

    enc_checkpoint = torch.load(enc_model_path, map_location=device)
    # dec_checkpoint = torch.load("decoder_200.pth", map_location=device)
    
    # Ecoder
    transmitter.encoder.load_state_dict(enc_checkpoint['encoder'])
    transmitter.channel_encoder.load_state_dict(enc_checkpoint['channel_encoder'])

    student_1_checkpoint = torch.load(student_1_path, map_location=device)

    # Student
    student.channel_decoder.load_state_dict(student_1_checkpoint['channel_decoder'])
    student.decoder.load_state_dict(student_1_checkpoint['decoder'])
    student.dense.load_state_dict(student_1_checkpoint['dense'])

    # Decoder
    # deepsc.channel_decoder.load_state_dict(dec_checkpoint['channel_decoder'])
    # deepsc.decoder.load_state_dict(dec_checkpoint['decoder'])
    # deepsc.dense.load_state_dict(dec_checkpoint['dense'])
    r=8
    alpha=16

    for parameter in student.parameters():
        parameter.requires_grad = False

    LoRA = apply_lora_to_decoder(
        student,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    LoRA = enable_language_adaptation(
        LoRA,
        train_embedding=True,
        train_output_head=True,
        train_layer_norm=True,
    )

    for name, parameter in LoRA.named_parameters():
        if parameter.requires_grad:
            print("TRAINABLE:", name, tuple(parameter.shape))

    LoRA = LoRA.to(device)

    # optimizer = torch.optim.Adam(lora_parameters(student), lr=args.lr)

    lora_params = []
    embedding_params = []
    output_params = []
    norm_params = []

    for name, parameter in LoRA.named_parameters():
        if not parameter.requires_grad:
            continue

        if "lora_" in name:
            lora_params.append(parameter)
        elif name.startswith("decoder.embedding"):
            embedding_params.append(parameter)
        elif name.startswith("dense"):
            output_params.append(parameter)
        elif "layernorm" in name:
            norm_params.append(parameter)

    parameter_groups = []

    if lora_params:
        parameter_groups.append({
            "params": lora_params,
            "lr": args.lr,
            "name": "lora",
        })

    if embedding_params:
        parameter_groups.append({
            "params": embedding_params,
            "lr": args.embedding_lr,
            "name": "embedding",
        })

    if output_params:
        parameter_groups.append({
            "params": output_params,
            "lr": args.output_lr,
            "name": "output",
        })

    if norm_params:
        parameter_groups.append({
            "params": norm_params,
            "lr": args.norm_lr,
            "name": "layernorm",
        })

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
    )
    
    os.makedirs(args.save_lora, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # pbar = tqdm(range(args.epochs))

    best_val_loss = float('inf')


    transmitter.eval()

    for parameter in transmitter.parameters():
        parameter.requires_grad = False


    today = date.today()
    root_dir = f'{args.save_lora}/{train_lag}/{today.strftime("%Y-%m-%d")}'

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for epoch in range(args.epochs):

        loss = train_lora_epoch(
            epoch,
            transmitter,
            LoRA,
            loader,
            optimizer,
            device,
            pad_idx,
            criterion,
            args
        )

        val_noise_std = SNR_to_noise(8.0)   
        
        val_loss = validate(
            epoch=epoch,
            args=args,
            transmitter=transmitter,
            LoRA=LoRA,
            criterion=criterion,
            device=device,
            pad_idx=pad_idx,
            noise_std=val_noise_std,
            train_lag=train_lag,
        )

        save_epoch_results(
            os.path.join(root_dir, 'results.csv'),
            epoch,
            {   
                'epoch': epoch,
                'loss': loss,
                'val_loss': val_loss
            }
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
            print(f"Saved LoRA adapter to {root_dir}")


if __name__ == "__main__":
    main()