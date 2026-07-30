"""

@author: Prinako
"""
import os
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

from dataset_multilingual import EurParallelDataset, collate_parallel
from models.transceiver import DeepSC
from models.mutual_info import Mine
from utils import (
    SNR_to_noise, 
    initNetParams, 
    train_step, 
    val_step, 
    train_mi,
    save_epoch_results
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, pad_idx, criterion, net):
    test_eur = EurParallelDataset(args.pt, 'test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_parallel)
    net.eval()
    pbar = tqdm(test_iterator)
    
    total: float = 0.0
    num_batches: int = 0
    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)
            loss = val_step(
                        net, src, trg, 0.1, pad_idx,
                        criterion, args.channel
                    )

            total += loss
            num_batches += 1

            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total / max(num_batches, 1)


def train(epoch, args, pad_idx, optimizer, criterion, net):
    train_eur= EurParallelDataset(args.pt, 'train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_parallel)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    total_loss: float = 0.0
    num_batches: int = 0

    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)

        loss = train_step(
                    net, src, trg, noise_std[0], pad_idx,
                    optimizer, criterion, args.channel
                )
        
        pbar.set_description(
            'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                epoch + 1, loss
            )
        )

        total_loss += loss
        num_batches += 1

        # if mi_net is not None:
        #     mi = train_mi(net, mi_net, src, trg, 0.1, pad_idx, mi_opt, args.channel)
        #     loss = train_step(net, src, trg, 0.1, pad_idx,
        #                       optimizer, criterion, args.channel, mi_net)
        #     pbar.set_description(
        #         'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
        #             epoch + 1, loss, mi
        #         )
        #     )
        # else:
    
    return total_loss / max(num_batches, 1)

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
    parser.add_argument('--vocab-file', default='europarl/vocab_multilingual.json', type=str)
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh/multilingual', type=str)
    parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
    parser.add_argument('--MAX-LENGTH', default=33, type=int)
    parser.add_argument('--MIN-LENGTH', default=4, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=8, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
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
    num_vocab = len(token_to_idx)
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

    for epoch in range(args.epochs):
        start = time.time()
        
        bast_acc: float = 0.0

        train(epoch, args, pad_idx, optimizer, criterion, deepsc)
        val_loss = validate(epoch, args, pad_idx, criterion, deepsc)

        end = time.time()
        
        save_epoch_results(
            os.path.join(
                args.checkpoint_path, f'results_{args.channel}_{date.today().strftime("%Y-%m-%d")}.csv'
            ), 
            epoch, 
            {   
                'epoch': epoch + 1,
                'loss': loss,
                'val_loss': val_loss,
                'time': end - start
            }
        )

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if bast_acc == 0.0:
            save_model(deepsc, root_dir, epoch)
            bast_acc = val_loss

        if val_loss < bast_acc:
            save_model(deepsc, root_dir, epoch)
            bast_acc = val_loss
        
    record_loss = []



if __name__ == '__main__':
    # setup_seed(10)
    main()