#!/usr/bin/env python3
"""Build a clean English-to-English JSON corpus from Europarl SGML files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

DEFAULT_OUTPUT = Path("data/europarl/json/en_en.json")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
TAG_RE = re.compile(r"^<[^>]+>$", re.DOTALL)
SPEAKER_RE = re.compile(r'^\s*[<"]?\s*SPEAKER\s+ID\s*=', re.I)
TRUNCATED_SPEAKER_RE = re.compile(r"^\s*<\s*SPEAKER\b", re.I)
STRUCTURE_RE = re.compile(
    r'^\s*[<"]?/?\s*(?:P|CHAPTER|DOC|DOCUMENT|TEXT|BODY|HEAD|TITLE|SESSION)\b', re.I
)
ATTRIBUTE_RE = re.compile(
    r'^\s*(?:ID|NAME|LANGUAGE|TYPE|LEVEL|ALIGN)\s*=\s*["\']?.*?["\']?\s*>?$', re.I
)
BOUNDARY_RE = re.compile(
    r'(?:(?<=[.!?])|(?<=[.!?]["\'])|(?<=[.!?][)\]]))\s+(?=["\']?[A-Za-z0-9])'
)
ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Hon|Rev|Gen|Sen|Rep|Pres|Gov|No|Nos|"
    r"Fig|Figs|Eq|Eqs|Dept|Inc|Ltd|Co|vs|etc|e\.g|i\.e)\.", re.I
)
PERIOD_PLACEHOLDER = "\ue000"
QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u2035": "'", "\uff07": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u201f": '"', "\u2033": '"', "\u2036": '"', "\u00ab": '"', "\u00bb": '"',
    "\u301d": '"', "\u301e": '"', "\u301f": '"', "\uff02": '"',
})


class Record(TypedDict):
    """Output record schema."""

    id: int
    source_language: str
    target_language: str
    source: str
    target: str


@dataclass
class Statistics:
    """Processing counters printed at the end of a run."""

    physical_lines: int = 0
    speaker_metadata: int = 0
    structural_tags: int = 0
    paragraphs: int = 0
    candidates: int = 0
    too_short: int = 0
    too_long: int = 0
    invalid: int = 0


def normalize_text(text: str) -> str:
    """Normalize whitespace, quotes, and separated apostrophe suffixes."""
    text = SPACE_RE.sub(" ", text.translate(QUOTE_MAP)).strip()
    return re.sub(r"\b([A-Za-z]+)'\s+([sSmMdDtT])\b", r"\1'\2", text)


def _is_speaker_line(text: str) -> bool:
    return bool(SPEAKER_RE.match(text) or TRUNCATED_SPEAKER_RE.match(text))


def is_metadata_line(text: str) -> bool:
    """Detect complete or clearly malformed Europarl SGML/XML metadata."""
    text = normalize_text(text)
    return bool(text) and bool(
        _is_speaker_line(text)
        or re.match(r"^<!--", text)
        or re.match(r"^<\?", text)
        or TAG_RE.fullmatch(text)
        or STRUCTURE_RE.match(text)
        or ATTRIBUTE_RE.fullmatch(text)
    )


def is_paragraph_boundary(text: str) -> bool:
    """Return whether a line terminates the current paragraph."""
    text = normalize_text(text)
    return not text or is_metadata_line(text)


def find_english_files(input_folder: Path) -> list[Path]:
    """Find supported English filenames recursively and deterministically."""
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    def matches(path: Path) -> bool:
        name = path.name.casefold()
        return name.endswith(".en") or name.endswith(".en.txt") or name == "english.txt"

    try:
        return sorted(
            (path for path in input_folder.rglob("*") if path.is_file() and matches(path)),
            key=lambda path: str(path).casefold(),
        )
    except OSError as exc:
        raise OSError(f"Could not search '{input_folder}': {exc}") from exc


def _paragraphs_from_lines(lines: Iterable[str], stats: Statistics) -> list[str]:
    """Reconstruct paragraphs from an iterable of physical lines."""
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraph = normalize_text(" ".join(current))
            if paragraph:
                paragraphs.append(paragraph)
            current.clear()

    for raw_line in lines:
        stats.physical_lines += 1
        line = normalize_text(raw_line)
        if not line:
            flush()
        elif _is_speaker_line(line):
            flush()
            stats.speaker_metadata += 1
        elif is_metadata_line(line):
            flush()
            stats.structural_tags += 1
        else:
            current.append(line)
    flush()
    stats.paragraphs += len(paragraphs)
    return paragraphs


def extract_paragraphs(file_path: Path, stats: Statistics | None = None) -> list[str]:
    """Read a UTF-8 Europarl file and reconstruct speech paragraphs."""
    stats = stats if stats is not None else Statistics()
    try:
        with file_path.open("r", encoding="utf-8") as source:
            return _paragraphs_from_lines(source, stats)
    except UnicodeDecodeError as exc:
        raise ValueError(f"Source file is not valid UTF-8: {file_path} ({exc})") from exc
    except OSError as exc:
        raise OSError(f"Could not read '{file_path}': {exc}") from exc


def _fallback_split(paragraph: str) -> list[str]:
    """Apply a dependency-free, abbreviation-aware sentence splitter."""
    protected = re.sub(r"(?<=\d)\.(?=\d)", PERIOD_PLACEHOLDER, paragraph)

    def protect(match: re.Match[str]) -> str:
        return match.group(0).replace(".", PERIOD_PLACEHOLDER)

    protected = ABBREVIATION_RE.sub(protect, protected)
    protected = re.sub(r"\b(?:[A-Z]\.){2,}", protect, protected)
    return [
        normalize_text(part.replace(PERIOD_PLACEHOLDER, "."))
        for part in BOUNDARY_RE.split(protected)
        if part.strip()
    ]


def split_into_sentences(paragraph: str) -> list[str]:
    """Use NLTK Punkt when usable, otherwise use the built-in splitter.

    Install optional Punkt data with ``python -m nltk.downloader punkt punkt_tab``.
    """
    paragraph = normalize_text(paragraph)
    if not paragraph:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        try:
            return [normalize_text(item) for item in sent_tokenize(paragraph) if item.strip()]
        except LookupError:
            pass
    except ImportError:
        pass
    return _fallback_split(paragraph)


def count_words(sentence: str) -> int:
    """Count words while handling Unicode, contractions, and hyphenation."""
    return len(WORD_RE.findall(sentence))


def is_valid_sentence(sentence: str, min_words: int, max_words: int) -> bool:
    """Validate natural-language content and inclusive word limits."""
    sentence = normalize_text(sentence)
    if not sentence or is_metadata_line(sentence):
        return False
    nonspace = [character for character in sentence if not character.isspace()]
    letters = sum(character.isalpha() for character in nonspace)
    if not nonspace or letters == 0 or letters / len(nonspace) < 0.25:
        return False
    return min_words <= count_words(sentence) <= max_words


def load_sentence_candidates(
    files: Sequence[Path], min_words: int, max_words: int
) -> tuple[list[str], Statistics]:
    """Reconstruct, segment, validate, and collect sentences from all files."""
    stats = Statistics()
    accepted: list[str] = []
    for file_path in files:
        print(f"Processing: {file_path}")
        for paragraph in extract_paragraphs(file_path, stats):
            for sentence in split_into_sentences(paragraph):
                sentence = normalize_text(sentence)
                stats.candidates += 1
                words = count_words(sentence)
                if is_metadata_line(sentence) or not any(char.isalpha() for char in sentence):
                    stats.invalid += 1
                elif words < min_words:
                    stats.too_short += 1
                elif words > max_words:
                    stats.too_long += 1
                elif not is_valid_sentence(sentence, min_words, max_words):
                    stats.invalid += 1
                else:
                    accepted.append(sentence)
    return accepted, stats


def remove_duplicates(sentences: Sequence[str]) -> list[str]:
    """Remove normalized, case-insensitive duplicates, keeping the first text."""
    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        key = normalize_text(sentence).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(sentence)
    return unique


def create_base_dataset(sentences: Sequence[str]) -> list[Record]:
    """Create sequential English-to-English records after final filtering."""
    return [
        {"id": index, "source_language": "en", "target_language": "en",
         "source": sentence, "target": sentence}
        for index, sentence in enumerate(sentences)
    ]


def save_json(records: Sequence[Record], output_file: Path) -> None:
    """Create the parent directory and write formatted UTF-8 JSON."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as destination:
            json.dump(records, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
    except OSError as exc:
        raise OSError(f"Could not write '{output_file}': {exc}") from exc


def _run_internal_example() -> None:
    """Verify the specification's paragraph and segmentation example."""
    sample = [
        '<SPEAKER ID=2 NAME="Evans, Robert J">',
        "Madam President, on a point of order. You will be aware of the recent events.",
        "<P>",
        "The House will discuss the matter tomorrow.",
    ]
    paragraphs = _paragraphs_from_lines(sample, Statistics())
    actual = [sentence for paragraph in paragraphs for sentence in split_into_sentences(paragraph)]
    expected = [
        "Madam President, on a point of order.",
        "You will be aware of the recent events.",
        "The House will discuss the matter tomorrow.",
    ]
    if actual != expected:
        raise AssertionError(f"Internal example failed: expected {expected!r}, got {actual!r}")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, default="txt/en")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args(argv)
    if args.min_words < 1:
        parser.error("--min-words must be at least 1")
    if args.max_words < args.min_words:
        parser.error("--max-words must be greater than or equal to --min-words")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the corpus-building command-line workflow."""
    args = parse_arguments(argv)
    try:
        _run_internal_example()
        files = find_english_files(args.input_folder)
        print(f"Matching input files: {len(files)}")
        if not files:
            raise FileNotFoundError(
                f"No matching files found in '{args.input_folder}'; expected *.en, "
                "*.en.txt, or english.txt."
            )
        candidates, stats = load_sentence_candidates(files, args.min_words, args.max_words)
        sentences = list(candidates) if args.keep_duplicates else remove_duplicates(candidates)
        records = create_base_dataset(sentences)
        save_json(records, args.output_file)

        print(f"Total physical lines read: {stats.physical_lines}")
        print(f"Speaker metadata lines ignored: {stats.speaker_metadata}")
        print(f"Paragraph/structural tags ignored: {stats.structural_tags}")
        print(f"Paragraphs reconstructed: {stats.paragraphs}")
        print(f"Sentence candidates generated: {stats.candidates}")
        print(f"Rejected for being too short: {stats.too_short}")
        print(f"Rejected for being too long: {stats.too_long}")
        print(f"Rejected as invalid or metadata: {stats.invalid}")
        print(f"Duplicates removed: {len(candidates) - len(sentences)}")
        print(f"Final JSON records: {len(records)}")
        print(f"Output path: {args.output_file}")
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError, AssertionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
