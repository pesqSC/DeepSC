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
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset_multilingual import EurParallelDataset, collate_parallel
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, pad_idx, criterion, net):
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


def train(epoch, args, pad_idx, optimizer, criterion, mi_opt, net, mi_net=None):
    train_eur= EurParallelDataset(args.en, 'train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_parallel)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, src, trg, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, src, trg, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = train_step(net, src, trg, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
    parser.add_argument('--vocab-file', default='europarl/vocab_multilingual.json', type=str)
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh/muiltilingual', type=str)
    parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--MIN-LENGTH', default=4, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--batch-size', default=74, type=int)
    parser.add_argument('--epochs', default=20, type=int) 
    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)

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
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, pad_idx, optimizer, criterion, mi_opt, deepsc)
        avg_acc = validate(epoch, args, pad_idx, criterion, deepsc)

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            print(deepsc.state_dict())
            encoder_state_dict = {
                "encoder": deepsc.encoder.state_dict(),
                "channel_encoder": deepsc.channel_encoder.state_dict(),
            }
            decoder_state_dict = {
                "channel_decoder": deepsc.channel_decoder.state_dict(),
                "decoder": deepsc.decoder.state_dict(),
                "dense": deepsc.dense.state_dict(),
            }
            encode_path = args.checkpoint_path + '/encoder_{}.pth'.format(str(epoch + 1).zfill(2))
            decode_path = args.checkpoint_path + '/decoder_{}.pth'.format(str(epoch + 1).zfill(2))
            with open(encode_path, 'wb') as f:
                torch.save(encoder_state_dict, f)
            with open(decode_path, 'wb') as f:
                torch.save(decoder_state_dict, f)
            record_acc = avg_acc
    record_loss = []



if __name__ == '__main__':
    # setup_seed(10)
    main()