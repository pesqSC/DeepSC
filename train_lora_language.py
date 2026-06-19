import argparse
import json
import torch
from torch.utils.data import DataLoader

from dataset_multilingual import EurParallelDataset, collate_parallel
from models.transceiver import DeepSC
from models.lora import apply_lora_to_decoder, lora_parameters, save_lora
from utils import SNR_to_noise, create_masks


def train_lora_epoch(model, loader, optimizer, device, pad_idx, channel="AWGN", snr=12):
    model.train()
    total_loss = 0

    noise_std = SNR_to_noise(snr)

    for src, trg in loader:
        src = src.to(device)
        trg = trg.to(device)

        trg_input = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask = create_masks(src, trg_input, pad_idx)

        optimizer.zero_grad()

        logits = model(
            src,
            trg_input,
            src_mask,
            look_ahead_mask,
            noise_std,
            channel
        )

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            trg_real.reshape(-1),
            ignore_index=pad_idx
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab-file", default="data/train/europarl/vocab_multilingual.json")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--save-lora", required=True)

    parser.add_argument("--channel", default="AWGN")
    parser.add_argument("--snr", default=12, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-heads", default=8, type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.load(open(args.vocab_file, "r", encoding="utf-8"))
    token_to_idx = vocab["token_to_idx"]
    vocab_size = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]

    dataset = EurParallelDataset(args.train_data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_parallel(b, pad_idx)
    )

    model = DeepSC(
        args.num_layers,
        vocab_size,
        vocab_size,
        vocab_size,
        vocab_size,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1
    ).to(device)

    model.load_state_dict(torch.load(args.student_checkpoint, map_location=device))

    model = apply_lora_to_decoder(model, r=8, alpha=16, dropout=0.05)

    optimizer = torch.optim.Adam(lora_parameters(model), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_lora_epoch(
            model,
            loader,
            optimizer,
            device,
            pad_idx,
            args.channel,
            args.snr
        )

        print(f"Epoch {epoch + 1}/{args.epochs} | LoRA loss: {loss:.4f}")

    save_lora(model, args.save_lora)
    print(f"Saved LoRA adapter to {args.save_lora}")


if __name__ == "__main__":
    main()