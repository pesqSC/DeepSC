#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multilingual DeepSC preprocessing with a shared SentencePiece BPE tokenizer.

Key properties
--------------
1. Trains ONE shared BPE tokenizer for all enabled languages.
2. Keeps DeepSC special tokens:
      <PAD>   = 0
      <START> = 1
      <END>   = 2
      <UNK>   = 3
      <EN>, <PT>, <ES>, <FR>
3. Uses BPE token counts for sequence-length filtering.
4. Produces the same basic DeepSC dataset structure:
      [
          ([source token ids], [target token ids]),
          ...
      ]
5. Saves:
      tokenizer_bpe.model
      tokenizer_bpe.vocab
      vocab_bpe.json
      train_en_en.pkl / test_en_en.pkl
      train_en_pt.pkl / test_en_pt.pkl
      ...
6. Trains the tokenizer ONLY on the training partition to avoid vocabulary
   leakage from the test set.
"""

import os
import re
import json
import pickle
import random
import argparse
import tempfile
import unicodedata

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
from tqdm import tqdm


# Special tokens
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


# Text normalization
def normalize_text(text: str) -> str:
    """
    Conservative multilingual normalization.

    Unlike the old word-level preprocessing, we do NOT remove accents
    and we do not restrict text to a-z. BPE can learn directly from
    Unicode text.

    For EN/PT/ES/FR this preserves forms such as:
        comunicação, votação, não, français, español, 1999
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    # Put spaces around common punctuation. SentencePiece would work
    # without this, but keeping punctuation explicit makes behavior
    # closer to the previous DeepSC preprocessing.
    text = re.sub(r"([.!?,;:])", r" \1 ", text)

    # Collapse whitespace only. Do not aggressively delete Unicode.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# JSON loading / validation
def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{file_path} must contain a JSON list."
        )

    return data


def validate_record(
    record: Dict[str, Any],
    dataset_name: str,
    index: int,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            f"{dataset_name}, record {index}: record must be a JSON object."
        )

    required_fields = (
        "id",
        "source_language",
        "target_language",
        "source",
        "target",
    )

    for field in required_fields:
        if field not in record:
            raise ValueError(
                f"{dataset_name}, record {index}: missing field '{field}'."
            )

    if not isinstance(record["source"], str):
        raise ValueError(
            f"{dataset_name}, record {index}: 'source' must be a string."
        )

    if not isinstance(record["target"], str):
        raise ValueError(
            f"{dataset_name}, record {index}: 'target' must be a string."
        )


def validate_alignment(
    base_records: List[Dict[str, Any]],
    translated_records: List[Dict[str, Any]],
    dataset_name: str,
) -> None:
    """
    Verify translated datasets against EN->EN using record ID and source text.
    """
    base_by_id = {}

    for index, record in enumerate(base_records):
        validate_record(record, "en_en", index)

        record_id = record["id"]

        if record_id in base_by_id:
            raise ValueError(
                f"Duplicate ID {record_id} found in en_en."
            )

        base_by_id[record_id] = record

    seen_ids = set()

    for index, record in enumerate(translated_records):
        validate_record(record, dataset_name, index)

        record_id = record["id"]

        if record_id in seen_ids:
            raise ValueError(
                f"Duplicate ID {record_id} found in {dataset_name}."
            )

        seen_ids.add(record_id)

        if record_id not in base_by_id:
            raise ValueError(
                f"{dataset_name}: ID {record_id} does not exist in en_en."
            )

        base_record = base_by_id[record_id]

        if record["source"] != base_record["source"]:
            raise ValueError(
                f"{dataset_name}: source mismatch for ID {record_id}.\n"
                f"Base:       {base_record['source']}\n"
                f"Translated: {record['source']}"
            )


# Shared train/test split by ID
def make_split_ids(
    base_records: List[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> Tuple[set, set]:
    """
    Split using EN->EN record IDs.

    All language datasets then use the same train/test IDs, which prevents
    alignment from being broken by independent shuffles.
    """
    ids = [record["id"] for record in base_records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected in en_en.")

    rng = random.Random(seed)
    rng.shuffle(ids)

    split_index = int(len(ids) * train_ratio)

    train_ids = set(ids[:split_index])
    test_ids = set(ids[split_index:])

    return train_ids, test_ids


# BPE training corpus
def iter_unique_training_sentences(
    datasets: Dict[str, List[Dict[str, Any]]],
    train_ids: set,
) -> Iterable[str]:
    """
    Yield unique normalized source/target sentences from TRAINING IDs only.

    Duplicate English source sentences across en_en/en_pt/etc. are emitted
    once, avoiding artificial over-weighting of English.
    """
    seen = set()

    for dataset_name, records in datasets.items():
        print(f"\nCollecting BPE training text from {dataset_name}...")

        for index, record in enumerate(
            tqdm(records, desc=dataset_name, leave=False)
        ):
            validate_record(record, dataset_name, index)

            if record["id"] not in train_ids:
                continue

            for field in ("source", "target"):
                sentence = normalize_text(record[field])

                if not sentence:
                    continue

                if sentence in seen:
                    continue

                seen.add(sentence)
                yield sentence


def write_bpe_training_corpus(
    datasets: Dict[str, List[Dict[str, Any]]],
    train_ids: set,
    corpus_path: str,
) -> int:
    count = 0

    with open(corpus_path, "w", encoding="utf-8") as file:
        for sentence in iter_unique_training_sentences(
            datasets=datasets,
            train_ids=train_ids,
        ):
            file.write(sentence + "\n")
            count += 1

    return count


# SentencePiece BPE
def train_bpe_tokenizer(
    corpus_path: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    byte_fallback: bool,
) -> None:
    """
    Train one shared multilingual SentencePiece BPE tokenizer.

    IDs 0..3 are deliberately kept compatible with the original DeepSC
    special-token convention.
    """
    user_symbols = [
        LANGUAGE_TOKENS["en"],
        LANGUAGE_TOKENS["pt"],
        LANGUAGE_TOKENS["es"],
        LANGUAGE_TOKENS["fr"],
    ]

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,

        # DeepSC-compatible special IDs.
        pad_id=PAD_ID,
        bos_id=START_ID,
        eos_id=END_ID,
        unk_id=UNK_ID,

        pad_piece=PAD_TOKEN,
        bos_piece=START_TOKEN,
        eos_piece=END_TOKEN,
        unk_piece=UNK_TOKEN,

        # Language control tokens.
        user_defined_symbols=user_symbols,

        # Multilingual / Unicode behavior.
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,

        # We already normalize text ourselves.
        normalization_rule_name="identity",

        # Prevent trainer failure if the requested vocab is slightly larger
        # than the number of extractable pieces.
        hard_vocab_limit=False,

        # Keep enough sentences for a large Europarl corpus.
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
                f"Special token {token} has ID {actual_id}; "
                f"expected {expected_id}."
            )

    for language, token in LANGUAGE_TOKENS.items():
        token_id = tokenizer.piece_to_id(token)

        if token_id == tokenizer.unk_id():
            raise RuntimeError(
                f"Language token {token} was not registered correctly."
            )

    return tokenizer


def save_vocab_json(
    tokenizer: spm.SentencePieceProcessor,
    path: str,
) -> None:
    """
    Save a JSON lookup table for legacy DeepSC code that still expects
    token_to_idx / idx_to_token.

    SentencePiece .model remains the authoritative tokenizer file.
    """
    token_to_idx = {
        tokenizer.id_to_piece(i): i
        for i in range(tokenizer.get_piece_size())
    }

    # JSON object keys are strings, so store idx_to_token with string keys.
    idx_to_token = {
        str(i): tokenizer.id_to_piece(i)
        for i in range(tokenizer.get_piece_size())
    }

    payload = {
        "tokenizer_type": "sentencepiece_bpe",
        "vocab_size": tokenizer.get_piece_size(),
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "special_tokens": {
            token: tokenizer.piece_to_id(token)
            for token in SPECIAL_TOKENS
        },
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


# Encoding / decoding
def language_token_id(
    tokenizer: spm.SentencePieceProcessor,
    language: str,
) -> int:
    language = language.lower()

    if language not in LANGUAGE_TOKENS:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {sorted(LANGUAGE_TOKENS)}"
        )

    token = LANGUAGE_TOKENS[language]
    token_id = tokenizer.piece_to_id(token)

    if token_id == tokenizer.unk_id():
        raise RuntimeError(
            f"Language token {token} is missing from the tokenizer."
        )

    return token_id


def encode_content(
    sentence: str,
    tokenizer: spm.SentencePieceProcessor,
) -> List[int]:
    """
    Encode only content, without DeepSC control tokens.
    """
    return tokenizer.encode(
        normalize_text(sentence),
        out_type=int,
        add_bos=False,
        add_eos=False,
    )


def encode_sentence(
    sentence: str,
    tokenizer: spm.SentencePieceProcessor,
    language: str,
) -> List[int]:
    """
    DeepSC sequence:

        <START> <LANG> BPE-piece-1 ... BPE-piece-N <END>
    """
    content_ids = encode_content(
        sentence=sentence,
        tokenizer=tokenizer,
    )

    lang_id = language_token_id(
        tokenizer=tokenizer,
        language=language,
    )

    return [
        tokenizer.bos_id(),
        lang_id,
        *content_ids,
        tokenizer.eos_id(),
    ]


def decode_sentence(
    ids: Sequence[int],
    tokenizer: spm.SentencePieceProcessor,
    remove_special_tokens: bool = True,
) -> str:
    """
    Decode a predicted DeepSC BPE sequence back to normal text.

    Use this instead of:
        " ".join(idx_to_token[id] ...)
    """
    cleaned = []

    special_ids = {
        tokenizer.pad_id(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
    }

    for token in LANGUAGE_TOKENS.values():
        token_id = tokenizer.piece_to_id(token)
        if token_id >= 0:
            special_ids.add(token_id)

    for token_id in ids:
        token_id = int(token_id)

        if token_id == tokenizer.eos_id():
            break

        if remove_special_tokens and token_id in special_ids:
            continue

        cleaned.append(token_id)

    return tokenizer.decode(cleaned).strip()


# BPE length validation
def valid_bpe_length(
    sentence: str,
    tokenizer: spm.SentencePieceProcessor,
    min_len: int,
    max_len: int,
) -> Tuple[bool, int]:
    """
    min_len/max_len refer to CONTENT BPE tokens only.

    <START>, language token and <END> are not counted.
    """
    content_length = len(
        encode_content(
            sentence=sentence,
            tokenizer=tokenizer,
        )
    )

    return (
        min_len <= content_length <= max_len,
        content_length,
    )


# Dataset creation
def make_dataset(
    records: List[Dict[str, Any]],
    tokenizer: spm.SentencePieceProcessor,
    expected_target_language: str,
    allowed_ids: set,
    min_len: int,
    max_len: int,
    dataset_name: str,
) -> Tuple[List[Tuple[List[int], List[int]]], Dict[str, int]]:
    """
    Convert JSON records into DeepSC source-target BPE-ID pairs.
    """
    dataset = []

    stats = {
        "accepted": 0,
        "not_in_split": 0,
        "empty": 0,
        "source_length": 0,
        "target_length": 0,
        "source_language": 0,
        "target_language": 0,
    }

    seen_ids = set()

    for index, record in enumerate(
        tqdm(
            records,
            desc=f"{dataset_name}: EN->{expected_target_language.upper()}",
            leave=False,
        )
    ):
        validate_record(record, dataset_name, index)

        record_id = record["id"]

        if record_id in seen_ids:
            raise ValueError(
                f"Duplicate ID detected in {dataset_name}: {record_id}"
            )

        seen_ids.add(record_id)

        if record_id not in allowed_ids:
            stats["not_in_split"] += 1
            continue

        src_language = record["source_language"].strip().lower()
        trg_language = record["target_language"].strip().lower()

        if src_language != "en":
            stats["source_language"] += 1
            continue

        if trg_language != expected_target_language:
            stats["target_language"] += 1
            continue

        source = normalize_text(record["source"])
        target = normalize_text(record["target"])

        if not source or not target:
            stats["empty"] += 1
            continue

        source_valid, _ = valid_bpe_length(
            source,
            tokenizer,
            min_len,
            max_len,
        )

        if not source_valid:
            stats["source_length"] += 1
            continue

        target_valid, _ = valid_bpe_length(
            target,
            tokenizer,
            min_len,
            max_len,
        )

        if not target_valid:
            stats["target_length"] += 1
            continue

        source_ids = encode_sentence(
            sentence=source,
            tokenizer=tokenizer,
            language="en",
        )

        target_ids = encode_sentence(
            sentence=target,
            tokenizer=tokenizer,
            language=expected_target_language,
        )

        dataset.append((source_ids, target_ids))
        stats["accepted"] += 1

    return dataset, stats


# Saving
def save_pickle(data: List, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(
            data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def print_stats(name: str, stats: Dict[str, int]) -> None:
    print(f"\n{name}")
    print(f"  Accepted             : {stats['accepted']:,}")
    print(f"  Outside split        : {stats['not_in_split']:,}")
    print(f"  Empty rejected       : {stats['empty']:,}")
    print(f"  Source length reject : {stats['source_length']:,}")
    print(f"  Target length reject : {stats['target_length']:,}")
    print(f"  Wrong source language: {stats['source_language']:,}")
    print(f"  Wrong target language: {stats['target_language']:,}")


# Diagnostics
def show_tokenizer_example(
    tokenizer: spm.SentencePieceProcessor,
) -> None:
    examples = [
        ("en", "the parliament approved the telecommunications proposal ."),
        ("pt", "o parlamento aprovou a proposta de telecomunicações ."),
    ]

    print("\n" + "=" * 70)
    print("BPE TOKENIZER EXAMPLES")
    print("=" * 70)

    for language, sentence in examples:
        normalized = normalize_text(sentence)
        content_ids = encode_content(normalized, tokenizer)
        pieces = [
            tokenizer.id_to_piece(i)
            for i in content_ids
        ]
        full_ids = encode_sentence(
            normalized,
            tokenizer,
            language,
        )
        decoded = decode_sentence(
            full_ids,
            tokenizer,
        )

        print(f"\nLanguage : {language.upper()}")
        print(f"Text     : {normalized}")
        print(f"Pieces   : {pieces}")
        print(f"IDs      : {full_ids}")
        print(f"Decoded  : {decoded}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create multilingual DeepSC datasets with a shared "
            "SentencePiece BPE tokenizer."
        )
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        default="data/europarl/json",
    )

    parser.add_argument(
        "--output-dir",
        default="data/train/europarl_bpe",
    )

    parser.add_argument(
        "--en-en-file",
        default="en_en.json",
    )

    parser.add_argument(
        "--en-pt-file",
        default="en_pt.json",
    )

    parser.add_argument(
        "--en-es-file",
        default="en_es.json",
    )

    parser.add_argument(
        "--en-fr-file",
        default="en_fr.json",
    )

    # BPE
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target SentencePiece BPE vocabulary size.",
    )

    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="1.0 is appropriate for EN/PT/ES/FR.",
    )

    parser.add_argument(
        "--byte-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use byte pieces as fallback for unseen Unicode. "
            "Recommended to minimize <UNK>."
        ),
    )

    parser.add_argument(
        "--force-retrain-tokenizer",
        action="store_true",
        help="Retrain tokenizer even if tokenizer_bpe.model already exists.",
    )

    # Sequence filtering
    parser.add_argument(
        "--min-len",
        type=int,
        default=4,
        help="Minimum number of CONTENT BPE tokens.",
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=48,
        help=(
            "Maximum number of CONTENT BPE tokens. "
            "BPE sequences are usually longer than word sequences."
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            "--vocab-size is too small."
        )

    if args.min_len < 1:
        raise ValueError(
            "--min-len must be at least 1."
        )

    if args.max_len < args.min_len:
        raise ValueError(
            "--max-len must be >= --min-len."
        )

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError(
            "--train-ratio must be between 0 and 1."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Enabled datasets
    dataset_paths = {
        "en_en": os.path.join(
            args.data_dir,
            args.en_en_file,
        ),
        "en_pt": os.path.join(
            args.data_dir,
            args.en_pt_file,
        ),

        # Enable later when ready:
        # "en_es": os.path.join(args.data_dir, args.en_es_file),
        # "en_fr": os.path.join(args.data_dir, args.en_fr_file),
    }

    language_map = {
        "en_en": "en",
        "en_pt": "pt",

        # "en_es": "es",
        # "en_fr": "fr",
    }

    # Load
    print("\n" + "=" * 70)
    print("LOADING JSON DATASETS")
    print("=" * 70)

    raw_datasets = {}

    for name, path in dataset_paths.items():
        print(f"\n{name}: {path}")

        records = load_json_dataset(path)
        raw_datasets[name] = records

        print(f"Records: {len(records):,}")

    # Validate alignment
    print("\n" + "=" * 70)
    print("VALIDATING DATASET ALIGNMENT")
    print("=" * 70)

    base_records = raw_datasets["en_en"]

    for name in raw_datasets:
        if name == "en_en":
            continue

        print(f"\nChecking EN_EN against {name}...")

        validate_alignment(
            base_records=base_records,
            translated_records=raw_datasets[name],
            dataset_name=name,
        )

        print("Alignment OK.")

    # Shared split IDs
    train_ids, test_ids = make_split_ids(
        base_records=base_records,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("SHARED TRAIN / TEST SPLIT")
    print("=" * 70)
    print(f"Train IDs: {len(train_ids):,}")
    print(f"Test IDs : {len(test_ids):,}")

    # Train/load BPE
    model_prefix = os.path.join(
        args.output_dir,
        "tokenizer_bpe",
    )

    model_path = model_prefix + ".model"
    vocab_sp_path = model_prefix + ".vocab"

    should_train = (
        args.force_retrain_tokenizer
        or not os.path.exists(model_path)
    )

    if should_train:
        print("\n" + "=" * 70)
        print("BUILDING BPE TRAINING CORPUS")
        print("=" * 70)

        corpus_path = os.path.join(
            args.output_dir,
            "_bpe_training_corpus.txt",
        )

        sentence_count = write_bpe_training_corpus(
            datasets=raw_datasets,
            train_ids=train_ids,
            corpus_path=corpus_path,
        )

        print(
            f"\nUnique training sentences for BPE: "
            f"{sentence_count:,}"
        )

        if sentence_count == 0:
            raise RuntimeError(
                "No training sentences were collected for BPE."
            )

        print("\n" + "=" * 70)
        print("TRAINING SHARED MULTILINGUAL BPE TOKENIZER")
        print("=" * 70)

        train_bpe_tokenizer(
            corpus_path=corpus_path,
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            byte_fallback=args.byte_fallback,
        )

        # The corpus can be very large, so remove it after training.
        try:
            os.remove(corpus_path)
        except OSError:
            pass

    else:
        print("\n" + "=" * 70)
        print("LOADING EXISTING BPE TOKENIZER")
        print("=" * 70)
        print(model_path)

    tokenizer = load_tokenizer(model_path)

    # Export JSON vocabulary for legacy DeepSC utilities
    vocab_json_path = os.path.join(
        args.output_dir,
        "vocab_bpe.json",
    )

    save_vocab_json(
        tokenizer=tokenizer,
        path=vocab_json_path,
    )

    print(f"\nActual BPE vocabulary size: {tokenizer.get_piece_size():,}")
    print(f"Tokenizer model           : {model_path}")
    print(f"Tokenizer vocab           : {vocab_sp_path}")
    print(f"DeepSC vocab JSON         : {vocab_json_path}")

    print("\nSpecial tokens:")
    for token in SPECIAL_TOKENS:
        print(
            f"  {token:8s} -> "
            f"{tokenizer.piece_to_id(token)}"
        )

    show_tokenizer_example(tokenizer)

    # Encode TRAIN and TEST separately using shared split IDs
    print("\n" + "=" * 70)
    print("ENCODING DEEPSC BPE DATASETS")
    print("=" * 70)

    for dataset_name, target_language in language_map.items():
        records = raw_datasets[dataset_name]

        print(f"\nCreating {dataset_name.upper()} TRAIN")

        train_data, train_stats = make_dataset(
            records=records,
            tokenizer=tokenizer,
            expected_target_language=target_language,
            allowed_ids=train_ids,
            min_len=args.min_len,
            max_len=args.max_len,
            dataset_name=dataset_name,
        )

        print_stats(
            f"{dataset_name.upper()} TRAIN",
            train_stats,
        )

        print(f"\nCreating {dataset_name.upper()} TEST")

        test_data, test_stats = make_dataset(
            records=records,
            tokenizer=tokenizer,
            expected_target_language=target_language,
            allowed_ids=test_ids,
            min_len=args.min_len,
            max_len=args.max_len,
            dataset_name=dataset_name,
        )

        print_stats(
            f"{dataset_name.upper()} TEST",
            test_stats,
        )

        train_path = os.path.join(
            args.output_dir,
            f"train_{dataset_name}.pkl",
        )

        test_path = os.path.join(
            args.output_dir,
            f"test_{dataset_name}.pkl",
        )

        save_pickle(train_data, train_path)
        save_pickle(test_data, test_path)

        print(f"\nSaved:")
        print(f"  {train_path}")
        print(f"  {test_path}")

    # Finished
    print("\n" + "=" * 70)
    print("BPE PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"\nTokenizer model : {model_path}")
    print(f"Vocabulary JSON : {vocab_json_path}")
    print(f"Vocabulary size : {tokenizer.get_piece_size():,}")
    print(f"PAD ID          : {tokenizer.pad_id()}")
    print(f"START ID        : {tokenizer.bos_id()}")
    print(f"END ID          : {tokenizer.eos_id()}")
    print(f"UNK ID          : {tokenizer.unk_id()}")


if __name__ == "__main__":
    main()
