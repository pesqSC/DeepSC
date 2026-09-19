"""

@author: Prinako
"""
import os
import math
import argparse
import time
import json
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import date

from dataset_multilingual import EurParallelDatasetBPE, collate_parallelBPE
from models.transceiver import DeepSC
from models.mutual_info import Mine
from utils.model_utils import (
    snr_to_noise, 
    initNetParams, 
    save_epoch_results
)
from utils.train_utils import (
    train_step, 
    val_step, 
    train_mi,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, pad_idx, criterion, net):
    val_eur = EurParallelDatasetBPE(args.en, 'val')
    
    val_iterator = DataLoader(
        val_eur,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_parallelBPE,
    )

    net.eval()
    pbar = tqdm(val_iterator)
    
    total_ce_sum = 0.0
    total_correct = 0
    total_tokens = 0

    # Fixed validation SNR
    noise_std = snr_to_noise(
        args.val_snr_db
    )

    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)

            stats = val_step(
                net,
                src,
                trg,
                noise_std,
                pad_idx,
                criterion,
                args.channel,
            )

            total_ce_sum += (
                stats["ce"]
                * stats["num_tokens"]
            )

            total_correct += stats["num_correct"]
            total_tokens += stats["num_tokens"]

            pbar.set_description(
                f"Epoch: {epoch + 1}; Type: VAL"
            )

            pbar.set_postfix(
                CE=f"{stats['ce']:.4f}",
                PPL=f"{stats['perplexity']:.2f}",
                ACC=f"{stats['token_accuracy']:.4f}",
                SNR=f"{args.val_snr_db:.1f}",
            )
    epoch_ce = (
        total_ce_sum
        / max(total_tokens, 1)
    )

    epoch_ppl = math.exp(
        min(epoch_ce, 700.0)
    )

    epoch_token_accuracy = (
        total_correct
        / max(total_tokens, 1)
    )

    return {
        "ce": epoch_ce,
        "perplexity": epoch_ppl,
        "token_accuracy": epoch_token_accuracy,
        "num_tokens": total_tokens,
        "snr_db": float(args.val_snr_db),
    }


def train(epoch, args, pad_idx, optimizer, criterion, net)->float:
    train_eur= EurParallelDatasetBPE(args.en, 'train')
    
    train_iterator = DataLoader(
        train_eur,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_parallelBPE,
    )

    pbar = tqdm(train_iterator)

    # total_loss: float = 0.0
    # num_batches: int = 0
    total_ce_sum: float = 0.0
    total_correct: int = 0
    total_tokens: int = 0

    snr_values = []

    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)

        if args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = np.random.uniform(
                args.snr_db_low,
                args.snr_db_high,
            )

        noise_std = snr_to_noise(snr_db)
        snr_values.append(float(snr_db))

        # loss = train_step(
        #             net, src, trg, noise_std, pad_idx,
        #             optimizer, criterion, args.channel
        #         )

        stats = train_step(
            net,
            src,
            trg,
            noise_std,
            pad_idx,
            optimizer,
            criterion,
            args.channel,
        )

        total_ce_sum += (
            stats["ce"] * stats["num_tokens"]
        )

        total_correct += stats["num_correct"]
        total_tokens += stats["num_tokens"]
        
        pbar.set_description(
            f"Epoch: {epoch + 1}; Type: Train"
        )

        pbar.set_postfix(
            CE=f"{stats['ce']:.4f}",
            PPL=f"{stats['perplexity']:.2f}",
            ACC=f"{stats['token_accuracy']:.4f}",
            SNR=f"{snr_db:.2f}",
        )

    epoch_ce = total_ce_sum / max(total_tokens, 1)

    epoch_ppl = math.exp(
        min(epoch_ce, 700.0)
    )

    epoch_token_accuracy = (
        total_correct / max(total_tokens, 1)
    )

    return {
        "ce": epoch_ce,
        "perplexity": epoch_ppl,
        "token_accuracy": epoch_token_accuracy,
        "num_tokens": total_tokens,
        "snr_mean": float(np.mean(snr_values)),
        "snr_min": float(np.min(snr_values)),
        "snr_max": float(np.max(snr_values)),
    }

def save_model(net: DeepSC, root_dir: str, epoch: int):
    encoder_state_dict = {
        "encoder": net.encoder.state_dict(),
        "channel_encoder": net.channel_encoder.state_dict(),
    }
    decoder_state_dict = {
        "channel_decoder": net.channel_decoder.state_dict(),
        "decoder": net.decoder.state_dict(),
        "dense": net.dense.state_dict(),
    }
    encode_path = root_dir + '/encoder_{}.pth'.format(str(epoch + 1).zfill(2))
    decode_path = root_dir + '/decoder_{}.pth'.format(str(epoch + 1).zfill(2))
    with open(encode_path, 'wb') as f:
        torch.save(encoder_state_dict, f)
    with open(decode_path, 'wb') as f:
        torch.save(decoder_state_dict, f)

def main():
    setup_seed(42)
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
    parser.add_argument('--vocab-file', default='europarl_bpe/vocab_bpe.json', type=str)
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh/multilingual_bpe', type=str)
    # channel
    parser.add_argument("--channel", type=str, default="Rayleigh", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr-mode", type=str, default="range", choices=["fixed", "range"])
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-db-low", type=float, default=2)
    parser.add_argument("--snr-db-high", type=float, default=18)
    parser.add_argument("--val-snr-db", type=float, default=8.0)

    parser.add_argument('--MAX-LENGTH', default=68, type=int)
    parser.add_argument('--MIN-LENGTH', default=4, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=8, type=int)
    parser.add_argument('--num-heads', default=16, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int) 

    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)

    parser.add_argument('--pt', default='pt_pt', type=str)
    parser.add_argument('--pt-pt', default='pt_en', type=str)
    parser.add_argument('--pt-es', default='pt_es', type=str)
    parser.add_argument('--pt-fr', default='pt_fr', type=str)

    args = parser.parse_args()

    args.vocab_file = './data/train/' + args.vocab_file

    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = vocab['vocab_size']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    """ define optimizer and loss function """
    deepsc = DeepSC(
                args.num_layers, 
                num_vocab, 
                num_vocab,
                args.MAX_LENGTH, 
                args.MAX_LENGTH, 
                args.d_model, 
                args.num_heads,
                args.dff, 
                0.1
            ).to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')

    optimizer = torch.optim.Adam(
                    deepsc.parameters(),
                    lr=1e-4, 
                    betas=(0.9, 0.98), 
                    eps=1e-8, 
                    weight_decay = 5e-4
                )

    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    today = date.today()
    
    root_dir = args.checkpoint_path + '/' + today.strftime("%Y-%m-%d")

    initNetParams(deepsc) # init net parameters

    best_val_loss = float("inf")

    if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    
    
    for epoch in range(args.epochs):
        start = time.time()
        

        train_stats = train(epoch, args, pad_idx, optimizer, criterion, deepsc)
        val_stats = validate(epoch, args, pad_idx, criterion, deepsc)

        end = time.time()
        
        save_epoch_results(
            os.path.join(
                root_dir, f'results_{args.channel}_{today.strftime("%Y-%m-%d")}.csv'
            ), 
            epoch, 
            {
                "train_ce": train_stats["ce"],
                "train_ppl": train_stats["perplexity"],
                "train_token_acc": train_stats["token_accuracy"],

                "val_ce": val_stats["ce"],
                "val_ppl": val_stats["perplexity"],
                "val_token_acc": val_stats["token_accuracy"],

                "train_snr_mean": train_stats["snr_mean"],
                "train_snr_min": train_stats["snr_min"],
                "train_snr_max": train_stats["snr_max"],
                "val_snr_db": val_stats["snr_db"],

                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": end - start,
            }
        )
        
        if val_stats["ce"] < best_val_loss:
            save_model(
                deepsc,
                root_dir,
                epoch
            )

            best_val_loss = val_stats["ce"]
        
    # record_loss = []

if __name__ == '__main__':
    import torch
    import gc

    if torch.cuda.is_available():

        # Remove Python references that are no longer used
        gc.collect()

        # Wait for pending CUDA operations
        torch.cuda.synchronize()

        # Release unused cached GPU memory
        torch.cuda.empty_cache()

        # Get free and total VRAM
        free_mem, total_mem = torch.cuda.mem_get_info(0)

        print(f"Cleared.")
        print(f"Free VRAM : {free_mem / 1024**3:.2f} GB")
        print(f"Total VRAM: {total_mem / 1024**3:.2f} GB")

    else:
        print("CUDA not available")
    main()