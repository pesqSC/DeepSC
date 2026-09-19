#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multilingual DeepSC preprocessing with a shared SentencePiece BPE tokenizer.
Designed for Batch-Interleaved / Dynamic Cross-Lingual Training (EN -> EN, PT, ES, FR).

Output Structure per Record
---------------------------
{
    "id": "ep-0001",
    "tx": [START_ID, token_1, ..., END_ID],
    "targets": {
        "en": [START_ID, EN_ID, token_1, ..., END_ID],
        "pt": [START_ID, PT_ID, token_1, ..., END_ID],
        "es": [START_ID, ES_ID, token_1, ..., END_ID],
        "fr": [START_ID, FR_ID, token_1, ..., END_ID],
    }
}

Key guarantees
--------------
1. Deterministic train/validation/test split with a fixed seed.
2. Split IDs are saved to JSON for exact reproducibility.
3. BPE tokenizer is trained only from the training partition.
4. Records must exist in all language-pair datasets.
5. English source text must match across en_en/en_pt/en_es/en_fr.
6. BPE corpus preserves sentence frequency across records while avoiding
   duplicated source/target text inside the same aligned record.
7. Sequence filtering accounts for special tokens:
      TX target length = content + <START> + <END>
      RX target length = content + <START> + <LANG> + <END>
"""

import argparse
import json
import os
import pickle
import random
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Special tokens
# -----------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

LANGUAGE_TOKENS = {
    "en": "<EN>",
    "pt": "<PT>",
    "es": "<ES>",
    "fr": "<FR>",
}

ACTIVE_LANGS = tuple(LANGUAGE_TOKENS.keys())

SPECIAL_TOKENS = [
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
    LANGUAGE_TOKENS["en"],
    LANGUAGE_TOKENS["pt"],
    LANGUAGE_TOKENS["es"],
    LANGUAGE_TOKENS["fr"],
]

PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


# -----------------------------------------------------------------------------
# Text / JSON utilities
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Standard multilingual Unicode NFKC normalization."""
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}.")

    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} must contain a JSON list.")

    required_fields = {"id", "source", "target"}
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {idx} in {file_path} is not a JSON object.")

        missing = required_fields.difference(record)
        if missing:
            raise ValueError(
                f"Record {idx} in {file_path} is missing required fields: "
                f"{sorted(missing)}"
            )

    return data


def index_by_id(records: Sequence[Dict[str, Any]], dataset_name: str) -> Dict[str, Dict[str, Any]]:
    """Create a deterministic ID -> record mapping and reject duplicate IDs."""
    indexed: Dict[str, Dict[str, Any]] = {}

    for record in records:
        rec_id = str(record["id"])
        if rec_id in indexed:
            raise ValueError(f"Duplicate ID '{rec_id}' detected in {dataset_name}.")
        indexed[rec_id] = record

    return indexed


# -----------------------------------------------------------------------------
# Split handling
# -----------------------------------------------------------------------------
def validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")


def make_split_ids(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Return deterministic ordered split ID lists."""
    validate_split_ratios(train_ratio, val_ratio)

    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected in the base dataset.")

    rng = random.Random(seed)
    rng.shuffle(ids)

    train_end = int(len(ids) * train_ratio)
    val_end = train_end + int(len(ids) * val_ratio)

    return (
        ids[:train_end],
        ids[train_end:val_end],
        ids[val_end:],
    )


def save_split_ids(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    path: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Alignment checks
# -----------------------------------------------------------------------------
def check_aligned_id(
    rec_id: str,
    datasets_by_id: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[bool, str]:
    """
    Validate one multilingual sample before tokenizer training / encoding.

    Returns
    -------
    (True, "ok") when the sample exists everywhere and the English source
    is identical across all language-pair files after normalization.
    """
    for lang in ACTIVE_LANGS:
        key = f"en_{lang}"
        if key not in datasets_by_id or rec_id not in datasets_by_id[key]:
            return False, "missing_alignment"

    base_source = normalize_text(datasets_by_id["en_en"][rec_id]["source"])
    if not base_source:
        return False, "empty_source"

    for lang in ACTIVE_LANGS:
        key = f"en_{lang}"
        candidate_source = normalize_text(datasets_by_id[key][rec_id]["source"])
        if candidate_source != base_source:
            return False, "source_mismatch"

    for lang in ACTIVE_LANGS:
        key = f"en_{lang}"
        target = normalize_text(datasets_by_id[key][rec_id]["target"])
        if not target:
            return False, "empty_target"

    return True, "ok"


def filter_aligned_ids(
    ids: Sequence[str],
    datasets_by_id: Dict[str, Dict[str, Dict[str, Any]]],
    desc: str,
) -> Tuple[List[str], Dict[str, int]]:
    """Filter IDs using alignment/source-equality checks only."""
    accepted: List[str] = []
    stats = {
        "total_checked": 0,
        "accepted": 0,
        "missing_alignment": 0,
        "source_mismatch": 0,
        "empty_source": 0,
        "empty_target": 0,
    }

    for rec_id in tqdm(ids, desc=desc, leave=False):
        stats["total_checked"] += 1
        ok, reason = check_aligned_id(rec_id, datasets_by_id)

        if not ok:
            stats[reason] += 1
            continue

        accepted.append(rec_id)
        stats["accepted"] += 1

    return accepted, stats


# -----------------------------------------------------------------------------
# SentencePiece corpus / tokenizer
# -----------------------------------------------------------------------------
def iter_training_sentences(
    datasets: Dict[str, Dict[str, Dict[str, Any]]],
    train_ids: Sequence[str],
) -> Iterable[str]:
    """
    Yield tokenizer-training sentences from aligned training IDs.

    Frequency is preserved across records. Duplicate sentences are removed
    only inside a single aligned record, mainly to avoid counting the shared
    English source several times because it appears in each language-pair file.
    """
    for rec_id in tqdm(train_ids, desc="BPE corpus", leave=False):
        sentences = [
            normalize_text(datasets["en_en"][rec_id]["source"]),
        ]

        for lang in ACTIVE_LANGS:
            key = f"en_{lang}"
            sentences.append(normalize_text(datasets[key][rec_id]["target"]))

        local_seen = set()
        for sentence in sentences:
            if sentence and sentence not in local_seen:
                local_seen.add(sentence)
                yield sentence


def write_bpe_training_corpus(
    datasets: Dict[str, Dict[str, Dict[str, Any]]],
    train_ids: Sequence[str],
    corpus_path: str,
) -> int:
    count = 0
    with open(corpus_path, "w", encoding="utf-8") as file:
        for sentence in iter_training_sentences(datasets, train_ids):
            file.write(sentence + "\n")
            count += 1
    return count


def train_bpe_tokenizer(
    corpus_path: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    byte_fallback: bool,
) -> None:
    user_symbols = list(LANGUAGE_TOKENS.values())

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        bos_id=START_ID,
        eos_id=END_ID,
        unk_id=UNK_ID,
        pad_piece=PAD_TOKEN,
        bos_piece=START_TOKEN,
        eos_piece=END_TOKEN,
        unk_piece=UNK_TOKEN,
        user_defined_symbols=user_symbols,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        normalization_rule_name="identity",
        hard_vocab_limit=False,
        input_sentence_size=0,
        shuffle_input_sentence=False,
    )


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor(model_file=model_path)

    expected = {
        PAD_TOKEN: PAD_ID,
        START_TOKEN: START_ID,
        END_TOKEN: END_ID,
        UNK_TOKEN: UNK_ID,
    }

    for token, expected_id in expected.items():
        actual_id = tokenizer.piece_to_id(token)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Special token {token} ID mismatch: expected {expected_id}, got {actual_id}."
            )

    for token in LANGUAGE_TOKENS.values():
        token_id = tokenizer.piece_to_id(token)
        if token_id == tokenizer.unk_id():
            raise RuntimeError(f"Language token {token} is missing from the tokenizer.")

    return tokenizer


def save_vocab_json(tokenizer: spm.SentencePieceProcessor, path: str) -> None:
    payload = {
        "tokenizer_type": "sentencepiece_bpe",
        "vocab_size": tokenizer.get_piece_size(),
        "token_to_idx": {
            tokenizer.id_to_piece(i): i
            for i in range(tokenizer.get_piece_size())
        },
        "idx_to_token": {
            str(i): tokenizer.id_to_piece(i)
            for i in range(tokenizer.get_piece_size())
        },
        "special_tokens": {
            token: tokenizer.piece_to_id(token)
            for token in SPECIAL_TOKENS
        },
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------
def encode_content(
    sentence: str,
    tokenizer: spm.SentencePieceProcessor,
) -> List[int]:
    return tokenizer.encode(
        normalize_text(sentence),
        out_type=int,
        add_bos=False,
        add_eos=False,
    )


def encode_tx(
    sentence: str,
    tokenizer: spm.SentencePieceProcessor,
) -> List[int]:
    """TX transmits English content: <START> content <END>."""
    return [
        tokenizer.bos_id(),
        *encode_content(sentence, tokenizer),
        tokenizer.eos_id(),
    ]


def encode_rx_target(
    sentence: str,
    tokenizer: spm.SentencePieceProcessor,
    lang: str,
) -> List[int]:
    """RX target: <START> <LANG> content <END>."""
    lang = lang.lower()
    if lang not in LANGUAGE_TOKENS:
        raise ValueError(f"Unsupported language: {lang}")

    lang_id = tokenizer.piece_to_id(LANGUAGE_TOKENS[lang])
    return [
        tokenizer.bos_id(),
        lang_id,
        *encode_content(sentence, tokenizer),
        tokenizer.eos_id(),
    ]


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------
def build_interleaved_dataset(
    datasets_by_id: Dict[str, Dict[str, Dict[str, Any]]],
    allowed_ids: Sequence[str],
    tokenizer: spm.SentencePieceProcessor,
    min_content_len: int,
    max_seq_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build aligned multi-target samples.

    `max_seq_len` means the complete encoded sequence length, including
    special tokens. Therefore:
      TX content max = max_seq_len - 2
      RX content max = max_seq_len - 3
    """
    if min_content_len < 1:
        raise ValueError("min_content_len must be >= 1.")
    if max_seq_len < 4:
        raise ValueError("max_seq_len must be >= 4.")

    max_tx_content_len = max_seq_len - 2
    max_rx_content_len = max_seq_len - 3

    if min_content_len > max_rx_content_len:
        raise ValueError(
            "min_content_len is larger than the maximum RX content length "
            f"allowed by max_seq_len={max_seq_len}."
        )

    dataset: List[Dict[str, Any]] = []
    stats = {
        "total_checked": 0,
        "accepted": 0,
        "missing_alignment": 0,
        "source_mismatch": 0,
        "empty_source": 0,
        "empty_target": 0,
        "tx_length_rejected": 0,
        "target_length_rejected": 0,
    }

    for rec_id in tqdm(allowed_ids, desc="Encoding", leave=False):
        stats["total_checked"] += 1

        # Defensive re-check. This keeps each output split independently safe.
        ok, reason = check_aligned_id(rec_id, datasets_by_id)
        if not ok:
            stats[reason] += 1
            continue

        source_en = datasets_by_id["en_en"][rec_id]["source"]
        tx_content = encode_content(source_en, tokenizer)

        if not (min_content_len <= len(tx_content) <= max_tx_content_len):
            stats["tx_length_rejected"] += 1
            continue

        encoded_targets: Dict[str, List[int]] = {}
        invalid_target_length = False

        for lang in ACTIVE_LANGS:
            key = f"en_{lang}"
            target_text = datasets_by_id[key][rec_id]["target"]
            target_content = encode_content(target_text, tokenizer)

            if not (
                min_content_len
                <= len(target_content)
                <= max_rx_content_len
            ):
                invalid_target_length = True
                break

            encoded_targets[lang] = target_content

        if invalid_target_length:
            stats["target_length_rejected"] += 1
            continue

        sample = {
            "id": rec_id,
            "tx": [
                tokenizer.bos_id(),
                *tx_content,
                tokenizer.eos_id(),
            ],
            "targets": {
                lang: [
                    tokenizer.bos_id(),
                    tokenizer.piece_to_id(LANGUAGE_TOKENS[lang]),
                    *encoded_targets[lang],
                    tokenizer.eos_id(),
                ]
                for lang in ACTIVE_LANGS
            },
        }

        # Final defensive assertions for model compatibility.
        assert len(sample["tx"]) <= max_seq_len
        assert all(
            len(target) <= max_seq_len
            for target in sample["targets"].values()
        )

        dataset.append(sample)
        stats["accepted"] += 1

    return dataset, stats


# -----------------------------------------------------------------------------
# Persistence / reporting
# -----------------------------------------------------------------------------
def save_pickle(data: Any, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def print_stats(title: str, stats: Dict[str, int]) -> None:
    print(f"\n{title}")
    for key, value in stats.items():
        print(f"  {key:24s}: {value:,}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual DeepSC preprocessing for batch-interleaved training."
    )

    parser.add_argument("--data-dir", default="data/europarl/json")
    parser.add_argument("--output-dir", default="data/train/europarl_bpe")

    parser.add_argument("--en-en-file", default="en_en.json")
    parser.add_argument("--en-pt-file", default="en_pt.json")
    parser.add_argument("--en-es-file", default="en_es.json")
    parser.add_argument("--en-fr-file", default="en_fr.json")

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Shared SentencePiece vocabulary size.",
    )
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument(
        "--byte-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-retrain-tokenizer", action="store_true")

    parser.add_argument(
        "--min-content-len",
        type=int,
        default=4,
        help="Minimum number of BPE content tokens, excluding special tokens.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
        help="Maximum complete encoded sequence length, including special tokens.",
    )

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=48)

    parser.add_argument(
        "--no-date-subdir",
        action="store_true",
        help="Write directly to --output-dir instead of adding YYYY-MM-DD.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    if not args.no_date_subdir:
        args.output_dir = os.path.join(
            args.output_dir,
            date.today().strftime("%Y-%m-%d"),
        )

    os.makedirs(args.output_dir, exist_ok=True)

    files = {
        "en_en": args.en_en_file,
        "en_pt": args.en_pt_file,
        "en_es": args.en_es_file,
        "en_fr": args.en_fr_file,
    }

    # ------------------------------------------------------------------
    # Load / index datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING AND INDEXING JSON DATASETS")
    print("=" * 70)

    datasets_by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for name, filename in files.items():
        path = os.path.join(args.data_dir, filename)
        records = load_json_dataset(path)
        datasets_by_id[name] = index_by_id(records, name)
        print(f"Loaded {name}: {len(records):,} records")

    base_records = list(datasets_by_id["en_en"].values())

    # ------------------------------------------------------------------
    # Deterministic shared split
    # ------------------------------------------------------------------
    train_ids, val_ids, test_ids = make_split_ids(
        base_records,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    split_path = os.path.join(args.output_dir, "split_ids.json")
    save_split_ids(
        train_ids,
        val_ids,
        test_ids,
        split_path,
        args.seed,
        args.train_ratio,
        args.val_ratio,
    )

    print("\nShared split:")
    print(f"  Train: {len(train_ids):,}")
    print(f"  Val  : {len(val_ids):,}")
    print(f"  Test : {len(test_ids):,}")
    print(f"  Saved split IDs -> {split_path}")

    # ------------------------------------------------------------------
    # Alignment pre-filter
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATING MULTILINGUAL ALIGNMENT")
    print("=" * 70)

    aligned_train_ids, train_alignment_stats = filter_aligned_ids(
        train_ids,
        datasets_by_id,
        "Train alignment",
    )
    aligned_val_ids, val_alignment_stats = filter_aligned_ids(
        val_ids,
        datasets_by_id,
        "Val alignment",
    )
    aligned_test_ids, test_alignment_stats = filter_aligned_ids(
        test_ids,
        datasets_by_id,
        "Test alignment",
    )

    print_stats("TRAIN alignment statistics:", train_alignment_stats)
    print_stats("VAL alignment statistics:", val_alignment_stats)
    print_stats("TEST alignment statistics:", test_alignment_stats)

    alignment_stats = {
        "train": train_alignment_stats,
        "val": val_alignment_stats,
        "test": test_alignment_stats,
    }
    save_json(
        alignment_stats,
        os.path.join(args.output_dir, "alignment_stats.json"),
    )

    # ------------------------------------------------------------------
    # Train / load tokenizer from TRAIN ONLY
    # ------------------------------------------------------------------
    model_prefix = os.path.join(args.output_dir, "tokenizer_bpe")
    model_path = model_prefix + ".model"

    if args.force_retrain_tokenizer or not os.path.exists(model_path):
        print("\n" + "=" * 70)
        print("TRAINING SHARED SENTENCEPIECE BPE TOKENIZER")
        print("=" * 70)

        corpus_path = os.path.join(
            args.output_dir,
            "_bpe_training_corpus.txt",
        )

        corpus_size = write_bpe_training_corpus(
            datasets_by_id,
            aligned_train_ids,
            corpus_path,
        )

        print(f"BPE training sentences: {corpus_size:,}")

        train_bpe_tokenizer(
            corpus_path,
            model_prefix,
            args.vocab_size,
            args.character_coverage,
            args.byte_fallback,
        )

        if os.path.exists(corpus_path):
            os.remove(corpus_path)
    else:
        print(f"\nUsing existing tokenizer: {model_path}")

    tokenizer = load_tokenizer(model_path)
    vocab_path = os.path.join(args.output_dir, "vocab_bpe.json")
    save_vocab_json(tokenizer, vocab_path)

    print(f"Tokenizer vocabulary size: {tokenizer.get_piece_size():,}")
    print(f"Vocabulary metadata -> {vocab_path}")

    print("\nLanguage token IDs:")
    for lang in ACTIVE_LANGS:
        token = LANGUAGE_TOKENS[lang]
        print(f"  {token:6s} -> {tokenizer.piece_to_id(token)}")

    # ------------------------------------------------------------------
    # Build encoded splits
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING SYNCHRONIZED MULTILINGUAL DATASETS")
    print("=" * 70)

    split_inputs = [
        ("train", aligned_train_ids),
        ("val", aligned_val_ids),
        ("test", aligned_test_ids),
    ]

    processing_stats: Dict[str, Dict[str, int]] = {}

    for split_name, allowed_ids in split_inputs:
        data, stats = build_interleaved_dataset(
            datasets_by_id=datasets_by_id,
            allowed_ids=allowed_ids,
            tokenizer=tokenizer,
            min_content_len=args.min_content_len,
            max_seq_len=args.max_seq_len,
        )

        out_path = os.path.join(
            args.output_dir,
            f"{split_name}_multilingual.pkl",
        )
        save_pickle(data, out_path)

        processing_stats[split_name] = stats

        print_stats(f"{split_name.upper()} encoding statistics:", stats)
        print(f"  saved                   : {out_path}")

        if data:
            max_tx = max(len(sample["tx"]) for sample in data)
            max_target = max(
                len(target)
                for sample in data
                for target in sample["targets"].values()
            )
            print(f"  max encoded TX length   : {max_tx}")
            print(f"  max encoded target len  : {max_target}")

    save_json(
        processing_stats,
        os.path.join(args.output_dir, "processing_stats.json"),
    )

    # ------------------------------------------------------------------
    # Experiment metadata
    # ------------------------------------------------------------------
    metadata = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": tokenizer.get_piece_size(),
        "character_coverage": args.character_coverage,
        "byte_fallback": args.byte_fallback,
        "min_content_len": args.min_content_len,
        "max_seq_len": args.max_seq_len,
        "max_tx_content_len": args.max_seq_len - 2,
        "max_rx_content_len": args.max_seq_len - 3,
        "languages": list(ACTIVE_LANGS),
        "language_tokens": {
            lang: {
                "piece": LANGUAGE_TOKENS[lang],
                "id": tokenizer.piece_to_id(LANGUAGE_TOKENS[lang]),
            }
            for lang in ACTIVE_LANGS
        },
        "special_token_ids": {
            "pad": tokenizer.pad_id(),
            "bos": tokenizer.bos_id(),
            "eos": tokenizer.eos_id(),
            "unk": tokenizer.unk_id(),
        },
    }

    save_json(
        metadata,
        os.path.join(args.output_dir, "preprocessing_config.json"),
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
