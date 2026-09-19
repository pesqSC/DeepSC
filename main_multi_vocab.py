"""
Multilingual DeepSC training for the synchronized SentencePiece-BPE dataset.

The transmitter always receives the shared English TX sequence. Each training
batch selects exactly one target language (EN/PT/ES/FR), implementing
batch-interleaved cross-lingual training without duplicating the source data.
"""

import argparse
import json
import math
import os
import random
import time
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multilingual import (
    SUPPORTED_LANGUAGES,
    EurMultilingualDatasetBPE,
    collate_multilingual_bpe,
)
from models.transceiver import DeepSC
from utils.model_utils import initNetParams, save_epoch_results, snr_to_noise
from utils.train_utils import train_step, val_step


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_languages(value: str):
    languages = tuple(
        part.strip().lower()
        for part in value.split(",")
        if part.strip()
    )

    if not languages:
        raise argparse.ArgumentTypeError("At least one language is required.")

    unsupported = set(languages).difference(SUPPORTED_LANGUAGES)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"Unsupported languages: {sorted(unsupported)}. "
            f"Supported: {list(SUPPORTED_LANGUAGES)}"
        )

    if len(languages) != len(set(languages)):
        raise argparse.ArgumentTypeError("Duplicate languages are not allowed.")

    return languages


def resolve_data_dir(path: str) -> str:
    """
    Resolve a preprocessing output directory.

    If `path` directly contains train_multilingual.pkl, use it. Otherwise,
    search date-named child directories and use the latest complete output.
    The resolved directory is printed and saved in run_config.json.
    """
    root = Path(path)

    if (root / "train_multilingual.pkl").exists():
        return str(root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue

        required = (
            child / "train_multilingual.pkl",
            child / "val_multilingual.pkl",
            child / "test_multilingual.pkl",
            child / "vocab_bpe.json",
            child / "tokenizer_bpe.model",
        )
        if all(item.exists() for item in required):
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            "Could not find a synchronized multilingual BPE preprocessing "
            f"output under: {root}"
        )

    resolved = sorted(candidates, key=lambda item: item.name)[-1]
    print(f"Using latest preprocessing directory: {resolved}")
    return str(resolved)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def new_meter():
    return {
        "ce_sum": 0.0,
        "num_correct": 0,
        "num_tokens": 0,
        "num_batches": 0,
    }


def update_meter(meter, stats) -> None:
    num_tokens = int(stats["num_tokens"])
    meter["ce_sum"] += float(stats["ce"]) * num_tokens
    meter["num_correct"] += int(stats["num_correct"])
    meter["num_tokens"] += num_tokens
    meter["num_batches"] += 1


def finalize_meter(meter):
    num_tokens = max(meter["num_tokens"], 1)
    ce = meter["ce_sum"] / num_tokens
    return {
        "ce": ce,
        "perplexity": math.exp(min(ce, 700.0)),
        "token_accuracy": meter["num_correct"] / num_tokens,
        "num_tokens": meter["num_tokens"],
        "num_batches": meter["num_batches"],
    }


def choose_language(
    languages,
    schedule: str,
    epoch: int,
    batch_index: int,
    batches_per_epoch: int,
    rng,
):
    if schedule == "round_robin":
        global_batch_index = epoch * batches_per_epoch + batch_index
        return languages[global_batch_index % len(languages)]

    if schedule == "random":
        return rng.choice(languages)

    raise ValueError(f"Unsupported language schedule: {schedule}")


def make_loader(dataset, args, pad_idx: int, shuffle: bool, epoch: int = 0):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_multilingual_bpe, pad_idx=pad_idx),
        generator=generator,
    )


def train(epoch, args, pad_idx, optimizer, criterion, net, dataset):
    train_iterator = make_loader(
        dataset,
        args,
        pad_idx,
        shuffle=True,
        epoch=epoch,
    )

    net.train()
    pbar = tqdm(train_iterator)

    overall_meter = new_meter()
    language_meters = {lang: new_meter() for lang in args.languages}
    language_rng = random.Random(args.seed + 10_000 + epoch)
    snr_values = []

    for batch_index, batch in enumerate(pbar):
        lang = choose_language(
            languages=args.languages,
            schedule=args.language_schedule,
            epoch=epoch,
            batch_index=batch_index,
            batches_per_epoch=len(train_iterator),
            rng=language_rng,
        )

        src = batch["src"].to(device, non_blocking=True)
        trg = batch["targets"][lang].to(device, non_blocking=True)

        if args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = np.random.uniform(args.snr_db_low, args.snr_db_high)

        noise_std = snr_to_noise(snr_db)
        snr_values.append(float(snr_db))

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

        update_meter(overall_meter, stats)
        update_meter(language_meters[lang], stats)

        pbar.set_description(
            f"Epoch {epoch + 1}; Train; lang={lang.upper()}"
        )
        pbar.set_postfix(
            CE=f"{stats['ce']:.4f}",
            PPL=f"{stats['perplexity']:.2f}",
            ACC=f"{stats['token_accuracy']:.4f}",
            SNR=f"{snr_db:.2f}",
        )

    result = finalize_meter(overall_meter)
    result["languages"] = {
        lang: finalize_meter(meter)
        for lang, meter in language_meters.items()
    }
    result["snr_mean"] = float(np.mean(snr_values)) if snr_values else 0.0
    result["snr_min"] = float(np.min(snr_values)) if snr_values else 0.0
    result["snr_max"] = float(np.max(snr_values)) if snr_values else 0.0
    return result


@torch.no_grad()
def validate(epoch, args, pad_idx, criterion, net, dataset):
    val_iterator = make_loader(
        dataset,
        args,
        pad_idx,
        shuffle=False,
    )

    net.eval()
    pbar = tqdm(val_iterator)
    noise_std = snr_to_noise(args.val_snr_db)

    overall_meter = new_meter()
    language_meters = {lang: new_meter() for lang in args.languages}

    for batch in pbar:
        src = batch["src"].to(device, non_blocking=True)

        # Validation reports every enabled receiver language separately.
        for lang in args.languages:
            trg = batch["targets"][lang].to(device, non_blocking=True)

            stats = val_step(
                net,
                src,
                trg,
                noise_std,
                pad_idx,
                criterion,
                args.channel,
            )

            update_meter(overall_meter, stats)
            update_meter(language_meters[lang], stats)

        lang_acc = {
            lang: finalize_meter(language_meters[lang])["token_accuracy"]
            for lang in args.languages
        }
        pbar.set_description(f"Epoch {epoch + 1}; Validation")
        pbar.set_postfix(
            **{f"ACC_{lang.upper()}": f"{acc:.3f}" for lang, acc in lang_acc.items()}
        )

    result = finalize_meter(overall_meter)
    result["languages"] = {
        lang: finalize_meter(meter)
        for lang, meter in language_meters.items()
    }
    result["snr_db"] = float(args.val_snr_db)
    return result


def save_model(net: DeepSC, root_dir: str, epoch: int) -> None:
    encoder_state_dict = {
        "encoder": net.encoder.state_dict(),
        "channel_encoder": net.channel_encoder.state_dict(),
    }
    decoder_state_dict = {
        "channel_decoder": net.channel_decoder.state_dict(),
        "decoder": net.decoder.state_dict(),
        "dense": net.dense.state_dict(),
    }

    encode_path = os.path.join(root_dir, f"encoder_{epoch + 1:02d}.pth")
    decode_path = os.path.join(root_dir, f"decoder_{epoch + 1:02d}.pth")

    torch.save(encoder_state_dict, encode_path)
    torch.save(decoder_state_dict, decode_path)


def flatten_epoch_metrics(train_stats, val_stats, args, epoch_time):
    payload = {
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
        "lr": args.current_lr,
        "epoch_time_sec": epoch_time,
    }

    for lang in args.languages:
        train_lang = train_stats["languages"][lang]
        val_lang = val_stats["languages"][lang]

        payload[f"train_{lang}_ce"] = train_lang["ce"]
        payload[f"train_{lang}_acc"] = train_lang["token_accuracy"]
        payload[f"train_{lang}_batches"] = train_lang["num_batches"]
        payload[f"val_{lang}_ce"] = val_lang["ce"]
        payload[f"val_{lang}_acc"] = val_lang["token_accuracy"]

    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-interleaved multilingual DeepSC BPE training."
    )

    parser.add_argument(
        "--data-dir",
        default="./data/train/europarl_bpe",
        type=str,
        help=(
            "Directory containing train_multilingual.pkl, etc. If this is "
            "the parent directory, the latest complete dated output is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/deepsc-Rayleigh/multilingual_bpe_interleaved",
        type=str,
    )
    parser.add_argument("--seed", default=48, type=int)
    parser.add_argument("--num-workers", default=0, type=int)

    parser.add_argument(
        "--languages",
        default=parse_languages("en,pt,es,fr"),
        type=parse_languages,
        help="Comma-separated targets, e.g. en,pt,es,fr",
    )
    parser.add_argument(
        "--language-schedule",
        choices=["round_robin", "random"],
        default="round_robin",
        help="Choose one target language per training batch.",
    )

    parser.add_argument(
        "--channel",
        default="Rayleigh",
        choices=["AWGN", "Rayleigh", "Rician"],
    )
    parser.add_argument(
        "--snr-mode",
        default="range",
        choices=["fixed", "range"],
    )
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-db-low", type=float, default=2.0)
    parser.add_argument("--snr-db-high", type=float, default=18.0)
    parser.add_argument("--val-snr-db", type=float, default=8.0)

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Defaults to max_seq_len from preprocessing_config.json.",
    )
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-layers", default=8, type=int)
    parser.add_argument("--num-heads", default=16, type=int)
    parser.add_argument("--batch-size", default=80, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_seed(args.seed)

    args.data_dir = resolve_data_dir(args.data_dir)

    vocab_path = os.path.join(args.data_dir, "vocab_bpe.json")
    tokenizer_path = os.path.join(args.data_dir, "tokenizer_bpe.model")
    preprocessing_config_path = os.path.join(
        args.data_dir, "preprocessing_config.json"
    )

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Missing BPE vocabulary: {vocab_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing SentencePiece model: {tokenizer_path}")

    vocab = load_json(vocab_path)
    token_to_idx = vocab["token_to_idx"]
    num_vocab = int(vocab["vocab_size"])

    pad_idx = int(token_to_idx["<PAD>"])
    start_idx = int(token_to_idx["<START>"])
    end_idx = int(token_to_idx["<END>"])

    preprocessing_config = {}
    if os.path.exists(preprocessing_config_path):
        preprocessing_config = load_json(preprocessing_config_path)

    preprocessing_max_len = preprocessing_config.get("max_seq_len")
    if args.max_length is None:
        args.max_length = int(preprocessing_max_len or 64)
    elif preprocessing_max_len and args.max_length < int(preprocessing_max_len):
        raise ValueError(
            f"--max-length={args.max_length} is smaller than the preprocessing "
            f"max_seq_len={preprocessing_max_len}."
        )

    print("=" * 70)
    print("MULTILINGUAL BPE TRAINING")
    print("=" * 70)
    print(f"Device             : {device}")
    print(f"Dataset            : {args.data_dir}")
    print(f"Vocabulary size    : {num_vocab:,}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Target languages   : {', '.join(args.languages)}")
    print(f"Language schedule  : {args.language_schedule}")
    print(f"Special IDs        : PAD={pad_idx} START={start_idx} END={end_idx}")

    train_dataset = EurMultilingualDatasetBPE(
        split="train",
        data_dir=args.data_dir,
        languages=args.languages,
    )
    val_dataset = EurMultilingualDatasetBPE(
        split="val",
        data_dir=args.data_dir,
        languages=args.languages,
    )

    print(f"Train samples      : {len(train_dataset):,}")
    print(f"Validation samples : {len(val_dataset):,}")

    deepsc = DeepSC(
        args.num_layers,
        num_vocab,
        num_vocab,
        args.max_length,
        args.max_length,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(
        deepsc.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=5e-4,
    )

    initNetParams(deepsc)

    today = date.today()
    root_dir = os.path.join(
        args.checkpoint_path,
        today.strftime("%Y-%m-%d"),
    )
    os.makedirs(root_dir, exist_ok=True)

    run_config = vars(args).copy()
    run_config["languages"] = list(args.languages)
    run_config["device"] = str(device)
    run_config["vocab_path"] = vocab_path
    run_config["tokenizer_path"] = tokenizer_path
    run_config["vocab_size"] = num_vocab
    run_config["preprocessing_config"] = preprocessing_config

    with open(
        os.path.join(root_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    best_val_ce = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_stats = train(
            epoch,
            args,
            pad_idx,
            optimizer,
            criterion,
            deepsc,
            train_dataset,
        )
        val_stats = validate(
            epoch,
            args,
            pad_idx,
            criterion,
            deepsc,
            val_dataset,
        )

        epoch_time = time.time() - start_time
        args.current_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = flatten_epoch_metrics(
            train_stats,
            val_stats,
            args,
            epoch_time,
        )

        save_epoch_results(
            os.path.join(
                root_dir,
                f"results_{args.channel}_{today.strftime('%Y-%m-%d')}.csv",
            ),
            epoch,
            epoch_metrics,
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train CE={train_stats['ce']:.4f} "
            f"ACC={train_stats['token_accuracy']:.4f} | "
            f"val CE={val_stats['ce']:.4f} "
            f"ACC={val_stats['token_accuracy']:.4f}"
        )

        for lang in args.languages:
            tr = train_stats["languages"][lang]
            va = val_stats["languages"][lang]
            print(
                f"  {lang.upper()}: "
                f"train CE={tr['ce']:.4f} ACC={tr['token_accuracy']:.4f} "
                f"batches={tr['num_batches']} | "
                f"val CE={va['ce']:.4f} ACC={va['token_accuracy']:.4f}"
            )

        if val_stats["ce"] < best_val_ce:
            save_model(deepsc, root_dir, epoch)
            best_val_ce = val_stats["ce"]
            print(f"  Saved new best checkpoint (val CE={best_val_ce:.4f}).")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
