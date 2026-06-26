import argparse
import json
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multilingual import EurParallelDataset, collate_parallel
from models.transceiver import DeepSC
from models.lora import apply_lora_to_decoder, lora_parameters, save_lora
from utils import SNR_to_noise, create_masks, setup_seed
from student import Student
from models.tx_model import Transmitter


def train_lora_epoch(epoch, transmitter, LoraModel, loader, optimizer, device, pad_idx, channel="AWGN", snr=12):
    LoraModel.train()
    total_loss = 0

    noise_std = SNR_to_noise(snr)
    
    pbar = tqdm(loader)

    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

        optimizer.zero_grad()

        with torch.no_grad():
            tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
                src, 
                src_mask, 
                channel, 
                noise_std
            )

        l1_logits, l1_ch_dec_out, l1_dec_out = LoraModel(
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

        loss = torch.nn.functional.cross_entropy(
            l1_logits.reshape(-1, l1_logits.size(-1)),
            trg_real.reshape(-1),
            ignore_index=pad_idx
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch + 1} Train; Loss: {loss.item():.5f}")

    return LoraModel, (total_loss / len(loader))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab-file", default="data/train/europarl/vocab_multilingual.json")
    # parser.add_argument("--train-data", required=True)
    parser.add_argument("--student-checkpoint", default="./checkpoints/deepsc-Rayleigh/multi_vocab_kd")
    parser.add_argument("--transmitter-checkpoint", type=str, default="./checkpoints/deepsc-Rayleigh/muiltilingual")
    parser.add_argument("--save-lora", default="./checkpoints/deepsc-Rayleigh/lora")

    parser.add_argument("--channel", default="Rayleigh", type=str, choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr", default=12, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)

    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)


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
        collate_fn=lambda b: collate_parallel(b, pad_idx)
    )

    deepsc = DeepSC(
        args.num_layers,
        vocab_size,
        vocab_size,
        vocab_size,
        vocab_size,
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
        vocab_size,
        vocab_size,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1
    ).to(device)

    enc_model_path = os.path.join(args.transmitter_checkpoint, 'encoder_200.pth')
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


    LoRA = apply_lora_to_decoder(student, r=8, alpha=16, dropout=0.05).to(device)

    optimizer = torch.optim.Adam(lora_parameters(student), lr=args.lr)
    
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        model, loss = train_lora_epoch(
            epoch,
            transmitter,
            LoRA,
            loader,
            optimizer,
            device,
            pad_idx,
            args.channel,
            args.snr
        )

        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | LoRA loss: {loss:.4f}")

        save_lora(epoch, model, args.save_lora, train_lag)
        print(f"Saved LoRA adapter to {args.save_lora}")


if __name__ == "__main__":
    main()