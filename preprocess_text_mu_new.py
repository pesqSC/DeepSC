"""
preprocess_text_mu.py

Create aligned multilingual DeepSC datasets from a directory structure like:

txt/
├── en/
│   ├── ep-00-01-17.txt
│   ├── ep-00-01-18.txt
│   └── ...
├── pt/
├── es/
└── fr/

The script matches files by relative filename:

txt/en/ep-00-01-17.txt
    ↕
txt/pt/ep-00-01-17.txt

It reads each matched file pair line by line and always filters both sides
together, preventing source-target misalignment.

Generated files:
    vocab_multilingual.json

    train_en_en.pkl
    test_en_en.pkl

    train_en_pt.pkl
    test_en_pt.pkl

    train_en_es.pkl
    test_en_es.pkl

    train_en_fr.pkl
    test_en_fr.pkl

    inspection_en_en.json
    inspection_en_pt.json
    inspection_en_es.json
    inspection_en_fr.json

    preprocessing_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import unicodedata

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Iterator, Sequence


# ---------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<PAD>",
    "<START>",
    "<END>",
    "<UNK>",
    "<EN>",
    "<PT>",
    "<ES>",
    "<FR>",
]

LANGUAGE_TOKENS = {
    "en": "<EN>",
    "pt": "<PT>",
    "es": "<ES>",
    "fr": "<FR>",
}


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

@dataclass
class DatasetStatistics:
    dataset: str

    matched_files: int = 0
    skipped_line_count_files: int = 0
    missing_target_files: int = 0
    missing_source_files: int = 0

    raw_pairs: int = 0
    kept_pairs: int = 0

    rejected_empty: int = 0
    rejected_source_length: int = 0
    rejected_target_length: int = 0
    rejected_length_ratio: int = 0

    source_content_tokens: int = 0
    source_unk_tokens: int = 0

    target_content_tokens: int = 0
    target_unk_tokens: int = 0

    @property
    def source_unk_rate(self) -> float:
        return self.source_unk_tokens / max(self.source_content_tokens, 1)

    @property
    def target_unk_rate(self) -> float:
        return self.target_unk_tokens / max(self.target_content_tokens, 1)


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize multilingual text while preserving accented characters.

    Examples preserved:
        português
        comunicação
        sesión
        français
    """

    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    # Remove XML/HTML-like tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Separate common punctuation into tokens.
    text = re.sub(r"([.!?,;:()])", r" \1 ", text)

    # Remove unsupported characters but preserve:
    # - Unicode letters/numbers
    # - whitespace
    # - common punctuation
    # - apostrophes
    # - hyphens
    text = text.replace("_", " ")
    text = re.sub(
        r"[^\w\s.!?,;:()'\-]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def token_count(text: str) -> int:
    return len(text.split())


def valid_length(
    text: str,
    min_content_length: int,
    max_sequence_length: int,
) -> bool:
    """
    The full sequence is:

        <START> <LANG> content... <END>

    Therefore the content can use at most max_sequence_length - 3 tokens.
    """

    maximum_content_length = max_sequence_length - 3
    length = token_count(text)

    return min_content_length <= length <= maximum_content_length


# ---------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------

def collect_text_files(directory: str | Path) -> dict[Path, Path]:
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Language directory does not exist: {directory}"
        )

    return {
        path.relative_to(directory): path
        for path in directory.rglob("*.txt")
    }


def find_matching_files(
    source_dir: str | Path,
    target_dir: str | Path,
) -> tuple[
    list[tuple[Path, Path, Path]],
    list[Path],
    list[Path],
]:
    """
    Match files using their relative filename.

    Returns:
        matched_files:
            [(source_path, target_path, relative_name), ...]

        source_only:
            files present only in source_dir

        target_only:
            files present only in target_dir
    """

    source_files = collect_text_files(source_dir)
    target_files = collect_text_files(target_dir)

    source_names = set(source_files)
    target_names = set(target_files)

    common_names = sorted(source_names & target_names)
    source_only = sorted(source_names - target_names)
    target_only = sorted(target_names - source_names)

    matched_files = [
        (
            source_files[name],
            target_files[name],
            name,
        )
        for name in common_names
    ]

    return matched_files, source_only, target_only


def count_lines(path: str | Path) -> int:
    with open(
        path,
        "r",
        encoding="utf-8",
        errors="strict",
    ) as file:
        return sum(1 for _ in file)


def iter_aligned_lines(
    source_file: str | Path,
    target_file: str | Path,
) -> Iterator[tuple[int, str, str]]:
    """
    Read two aligned files without silently truncating either side.
    """

    with (
        open(
            source_file,
            "r",
            encoding="utf-8",
            errors="strict",
        ) as source_stream,
        open(
            target_file,
            "r",
            encoding="utf-8",
            errors="strict",
        ) as target_stream,
    ):
        for line_number, pair in enumerate(
            zip_longest(source_stream, target_stream),
            start=1,
        ):
            source_line, target_line = pair

            if source_line is None or target_line is None:
                raise RuntimeError(
                    "Unexpected line mismatch while reading:\n"
                    f"  source: {source_file}\n"
                    f"  target: {target_file}\n"
                    f"  line: {line_number}"
                )

            yield line_number, source_line, target_line


# ---------------------------------------------------------------------
# Parallel pair loading
# ---------------------------------------------------------------------

def load_parallel_directory(
    dataset_name: str,
    source_dir: str | Path,
    target_dir: str | Path,
    min_content_length: int,
    max_sequence_length: int,
    max_length_ratio: float | None,
    skip_mismatched_files: bool,
) -> tuple[
    list[tuple[str, str, str, int]],
    DatasetStatistics,
    list[dict],
]:
    """
    Load aligned sentence pairs from matching source/target files.

    Each returned item contains:
        source_text
        target_text
        relative_filename
        original_line_number
    """

    matched_files, source_only, target_only = find_matching_files(
        source_dir,
        target_dir,
    )

    statistics = DatasetStatistics(
        dataset=dataset_name,
        matched_files=len(matched_files),
        missing_target_files=len(source_only),
        missing_source_files=len(target_only),
    )

    mismatched_files_report: list[dict] = []
    pairs: list[tuple[str, str, str, int]] = []

    print(f"\n{dataset_name}")
    print(f"  Matching files: {len(matched_files):,}")
    print(f"  Source-only files: {len(source_only):,}")
    print(f"  Target-only files: {len(target_only):,}")

    for source_path, target_path, relative_name in matched_files:
        source_lines = count_lines(source_path)
        target_lines = count_lines(target_path)

        if source_lines != target_lines:
            statistics.skipped_line_count_files += 1

            mismatch = {
                "file": str(relative_name),
                "source_file": str(source_path),
                "target_file": str(target_path),
                "source_lines": source_lines,
                "target_lines": target_lines,
            }

            mismatched_files_report.append(mismatch)

            message = (
                f"Line-count mismatch in {relative_name}: "
                f"source={source_lines}, target={target_lines}"
            )

            if skip_mismatched_files:
                print(f"  Skipping: {message}")
                continue

            raise ValueError(message)

        for line_number, raw_source, raw_target in iter_aligned_lines(
            source_path,
            target_path,
        ):
            statistics.raw_pairs += 1

            source_text = normalize_text(raw_source)
            target_text = normalize_text(raw_target)

            # Always reject the whole pair.
            if not source_text or not target_text:
                statistics.rejected_empty += 1
                continue

            if not valid_length(
                source_text,
                min_content_length,
                max_sequence_length,
            ):
                statistics.rejected_source_length += 1
                continue

            if not valid_length(
                target_text,
                min_content_length,
                max_sequence_length,
            ):
                statistics.rejected_target_length += 1
                continue

            if max_length_ratio is not None:
                source_length = token_count(source_text)
                target_length = token_count(target_text)

                ratio = max(
                    source_length,
                    target_length,
                ) / max(
                    min(source_length, target_length),
                    1,
                )

                if ratio > max_length_ratio:
                    statistics.rejected_length_ratio += 1
                    continue

            pairs.append(
                (
                    source_text,
                    target_text,
                    str(relative_name),
                    line_number,
                )
            )

            statistics.kept_pairs += 1

    print(
        f"  Kept pairs: "
        f"{statistics.kept_pairs:,}/{statistics.raw_pairs:,}"
    )

    return pairs, statistics, mismatched_files_report


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------

def build_vocabulary(
    texts: Iterable[str],
    min_count: int,
) -> dict:
    counter: Counter[str] = Counter()

    for text in texts:
        counter.update(text.split())

    token_to_idx: dict[str, int] = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    # Deterministic ordering:
    # - higher frequency first
    # - alphabetical order for ties
    candidates = sorted(
        (
            (token, count)
            for token, count in counter.items()
            if count >= min_count and token not in token_to_idx
        ),
        key=lambda item: (-item[1], item[0]),
    )

    for token, _ in candidates:
        token_to_idx[token] = len(token_to_idx)

    idx_to_token = [
        token
        for token, _ in sorted(
            token_to_idx.items(),
            key=lambda item: item[1],
        )
    ]

    return {
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "special_tokens": SPECIAL_TOKENS,
        "min_count": min_count,
    }


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------

def encode_sentence(
    text: str,
    language: str,
    token_to_idx: dict[str, int],
    max_sequence_length: int,
) -> tuple[list[int], int, int]:
    language = language.lower()

    if language not in LANGUAGE_TOKENS:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    unk_idx = token_to_idx["<UNK>"]

    content_tokens = text.split()

    content_ids = [
        token_to_idx.get(token, unk_idx)
        for token in content_tokens
    ]

    sequence = [
        token_to_idx["<START>"],
        token_to_idx[LANGUAGE_TOKENS[language]],
        *content_ids,
        token_to_idx["<END>"],
    ]

    if len(sequence) > max_sequence_length:
        raise ValueError(
            f"Encoded sequence length {len(sequence)} exceeds "
            f"max_sequence_length={max_sequence_length}"
        )

    unknown_count = sum(
        token_id == unk_idx
        for token_id in content_ids
    )

    return sequence, unknown_count, len(content_ids)


def encode_parallel_pairs(
    pairs: Sequence[tuple[str, str, str, int]],
    source_language: str,
    target_language: str,
    token_to_idx: dict[str, int],
    max_sequence_length: int,
    statistics: DatasetStatistics,
) -> tuple[
    list[tuple[list[int], list[int]]],
    list[dict],
]:
    encoded_data: list[tuple[list[int], list[int]]] = []
    metadata: list[dict] = []

    for (
        source_text,
        target_text,
        relative_file,
        line_number,
    ) in pairs:
        source_ids, source_unknowns, source_tokens = encode_sentence(
            source_text,
            source_language,
            token_to_idx,
            max_sequence_length,
        )

        target_ids, target_unknowns, target_tokens = encode_sentence(
            target_text,
            target_language,
            token_to_idx,
            max_sequence_length,
        )

        statistics.source_content_tokens += source_tokens
        statistics.source_unk_tokens += source_unknowns

        statistics.target_content_tokens += target_tokens
        statistics.target_unk_tokens += target_unknowns

        encoded_data.append(
            (
                source_ids,
                target_ids,
            )
        )

        metadata.append(
            {
                "file": relative_file,
                "line": line_number,
                "source_text": source_text,
                "target_text": target_text,
            }
        )

    return encoded_data, metadata


def build_en_en_dataset(
    english_texts: Iterable[str],
    token_to_idx: dict[str, int],
    max_sequence_length: int,
) -> tuple[
    list[tuple[list[int], list[int]]],
    list[dict],
]:
    """
    Build EN->EN reconstruction data from the union of all valid English
    source sentences.

    Duplicate sentences are removed.
    """

    unique_texts = list(
        dict.fromkeys(english_texts)
    )

    data: list[tuple[list[int], list[int]]] = []
    metadata: list[dict] = []

    for index, text in enumerate(unique_texts):
        ids, _, _ = encode_sentence(
            text,
            "en",
            token_to_idx,
            max_sequence_length,
        )

        data.append(
            (
                ids,
                list(ids),
            )
        )

        metadata.append(
            {
                "index": index,
                "source_text": text,
                "target_text": text,
            }
        )

    return data, metadata


# ---------------------------------------------------------------------
# Splitting and saving
# ---------------------------------------------------------------------

def split_data_and_metadata(
    data: Sequence,
    metadata: Sequence,
    train_ratio: float,
    seed: int,
) -> tuple[list, list, list, list]:
    if len(data) != len(metadata):
        raise ValueError(
            "Data and metadata have different lengths."
        )

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)

    split_position = int(
        len(indices) * train_ratio
    )

    train_indices = indices[:split_position]
    test_indices = indices[split_position:]

    train_data = [
        data[index]
        for index in train_indices
    ]

    test_data = [
        data[index]
        for index in test_indices
    ]

    train_metadata = [
        metadata[index]
        for index in train_indices
    ]

    test_metadata = [
        metadata[index]
        for index in test_indices
    ]

    return (
        train_data,
        test_data,
        train_metadata,
        test_metadata,
    )


def save_pickle(
    data: object,
    path: str | Path,
) -> None:
    with open(path, "wb") as file:
        pickle.dump(
            data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def save_json(
    data: object,
    path: str | Path,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def choose_inspection_samples(
    metadata: Sequence[dict],
    sample_count: int,
    seed: int,
) -> list[dict]:
    if not metadata:
        return []

    sample_count = min(
        sample_count,
        len(metadata),
    )

    return random.Random(seed).sample(
        list(metadata),
        sample_count,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned multilingual DeepSC datasets from "
            "language directories."
        )
    )

    parser.add_argument(
        "--en-dir",
        default="data/txt/en",
        help="Directory containing English .txt files.",
    )

    parser.add_argument(
        "--pt-dir",
        default="data/txt/pt",
        help="Directory containing Portuguese .txt files.",
    )

    parser.add_argument(
        "--es-dir",
        default="data/txt/es",
        help="Directory containing Spanish .txt files.",
    )

    parser.add_argument(
        "--fr-dir",
        default="data/txt/fr",
        help="Directory containing French .txt files.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/train/europarl",
    )

    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Minimum content-token count per sentence.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help=(
            "Maximum full sequence length, including "
            "<START>, language token and <END>."
        ),
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum token frequency for vocabulary inclusion.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help=(
            "Reject a pair when longer_length/shorter_length exceeds "
            "this value. Set 0 to disable."
        ),
    )

    parser.add_argument(
        "--inspection-samples",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--strict-line-count",
        action="store_true",
        help=(
            "Stop immediately when a matching file pair has different "
            "line counts. By default, such files are skipped and reported."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_length_ratio = (
        None
        if args.max_length_ratio <= 0
        else args.max_length_ratio
    )

    directory_pairs = {
        "en_pt": (
            args.en_dir,
            args.pt_dir,
            "pt",
        ),
        "en_es": (
            args.en_dir,
            args.es_dir,
            "es",
        ),
        "en_fr": (
            args.en_dir,
            args.fr_dir,
            "fr",
        ),
    }

    cleaned_datasets: dict[
        str,
        list[tuple[str, str, str, int]],
    ] = {}

    statistics: dict[
        str,
        DatasetStatistics,
    ] = {}

    mismatched_files: dict[
        str,
        list[dict],
    ] = {}

    # --------------------------------------------------------------
    # Load aligned bilingual corpora
    # --------------------------------------------------------------

    for dataset_name, (
        source_dir,
        target_dir,
        target_language,
    ) in directory_pairs.items():
        (
            pairs,
            dataset_statistics,
            mismatch_report,
        ) = load_parallel_directory(
            dataset_name=dataset_name,
            source_dir=source_dir,
            target_dir=target_dir,
            min_content_length=args.min_len,
            max_sequence_length=args.max_length,
            max_length_ratio=max_length_ratio,
            skip_mismatched_files=not args.strict_line_count,
        )

        if not pairs:
            raise RuntimeError(
                f"No valid aligned pairs remain for {dataset_name}."
            )

        cleaned_datasets[dataset_name] = pairs
        statistics[dataset_name] = dataset_statistics
        mismatched_files[dataset_name] = mismatch_report

    # --------------------------------------------------------------
    # Build shared multilingual vocabulary
    # --------------------------------------------------------------

    all_texts: list[str] = []

    for pairs in cleaned_datasets.values():
        for (
            source_text,
            target_text,
            _,
            _,
        ) in pairs:
            all_texts.append(source_text)
            all_texts.append(target_text)

    vocabulary = build_vocabulary(
        all_texts,
        min_count=args.min_count,
    )

    token_to_idx = vocabulary["token_to_idx"]

    vocabulary_path = (
        output_dir / "vocab_multilingual.json"
    )

    save_json(
        vocabulary,
        vocabulary_path,
    )

    print(
        f"\nVocabulary size: {len(token_to_idx):,}"
    )
    print(
        f"Saved vocabulary: {vocabulary_path}"
    )

    # --------------------------------------------------------------
    # Encode bilingual datasets
    # --------------------------------------------------------------

    encoded_datasets: dict[str, list] = {}
    metadata_datasets: dict[str, list] = {}

    for dataset_name, pairs in cleaned_datasets.items():
        target_language = directory_pairs[
            dataset_name
        ][2]

        encoded_data, metadata = encode_parallel_pairs(
            pairs=pairs,
            source_language="en",
            target_language=target_language,
            token_to_idx=token_to_idx,
            max_sequence_length=args.max_length,
            statistics=statistics[dataset_name],
        )

        encoded_datasets[dataset_name] = encoded_data
        metadata_datasets[dataset_name] = metadata

    # --------------------------------------------------------------
    # Build EN->EN dataset from all valid English source sentences
    # --------------------------------------------------------------

    english_texts = [
        source_text
        for pairs in cleaned_datasets.values()
        for (
            source_text,
            _,
            _,
            _,
        ) in pairs
    ]

    en_en_data, en_en_metadata = build_en_en_dataset(
        english_texts=english_texts,
        token_to_idx=token_to_idx,
        max_sequence_length=args.max_length,
    )

    encoded_datasets["en_en"] = en_en_data
    metadata_datasets["en_en"] = en_en_metadata

    # --------------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------------

    report = {
        "configuration": {
            "en_dir": args.en_dir,
            "pt_dir": args.pt_dir,
            "es_dir": args.es_dir,
            "fr_dir": args.fr_dir,
            "output_dir": args.output_dir,
            "min_len": args.min_len,
            "max_length": args.max_length,
            "min_count": args.min_count,
            "train_ratio": args.train_ratio,
            "max_length_ratio": max_length_ratio,
            "seed": args.seed,
            "strict_line_count": args.strict_line_count,
        },
        "vocabulary_size": len(token_to_idx),
        "datasets": {},
        "mismatched_files": mismatched_files,
    }

    ordered_datasets = [
        "en_en",
        "en_pt",
        "en_es",
        "en_fr",
    ]

    for dataset_offset, dataset_name in enumerate(
        ordered_datasets
    ):
        data = encoded_datasets[dataset_name]
        metadata = metadata_datasets[dataset_name]

        (
            train_data,
            test_data,
            train_metadata,
            test_metadata,
        ) = split_data_and_metadata(
            data=data,
            metadata=metadata,
            train_ratio=args.train_ratio,
            seed=args.seed + dataset_offset,
        )

        train_path = (
            output_dir /
            f"train_{dataset_name}.pkl"
        )

        test_path = (
            output_dir /
            f"test_{dataset_name}.pkl"
        )

        inspection_path = (
            output_dir /
            f"inspection_{dataset_name}.json"
        )

        save_pickle(
            train_data,
            train_path,
        )

        save_pickle(
            test_data,
            test_path,
        )

        # Use both train and test metadata for random inspection.
        combined_metadata = (
            list(train_metadata) +
            list(test_metadata)
        )

        inspection_samples = choose_inspection_samples(
            metadata=combined_metadata,
            sample_count=args.inspection_samples,
            seed=args.seed + dataset_offset,
        )

        save_json(
            inspection_samples,
            inspection_path,
        )

        dataset_report = {
            "total": len(data),
            "train": len(train_data),
            "test": len(test_data),
        }

        if dataset_name in statistics:
            stats = statistics[dataset_name]

            dataset_report.update(
                asdict(stats)
            )

            dataset_report[
                "source_unk_rate"
            ] = stats.source_unk_rate

            dataset_report[
                "target_unk_rate"
            ] = stats.target_unk_rate

        report["datasets"][
            dataset_name
        ] = dataset_report

        print(
            f"\n{dataset_name}"
            f"\n  Train: {len(train_data):,}"
            f"\n  Test:  {len(test_data):,}"
            f"\n  Saved: {train_path}"
            f"\n  Saved: {test_path}"
            f"\n  Inspect: {inspection_path}"
        )

    report_path = (
        output_dir /
        "preprocessing_report.json"
    )

    save_json(
        report,
        report_path,
    )

    print(
        f"\nSaved report: {report_path}"
    )

    print(
        "\nPreprocessing finished."
        "\nBefore training, inspect:"
        "\n  inspection_en_pt.json"
        "\n  inspection_en_es.json"
        "\n  inspection_en_fr.json"
        "\n"
        "\nEvery source sentence must correspond to the target "
        "sentence directly below it."
    )


if __name__ == "__main__":
    main()
