# preprocess_text.py

import os
import re
import json
import pickle
import argparse
from collections import Counter
from tqdm import tqdm

try:
    from w3lib.html import remove_tags
except ImportError:
    def remove_tags(text):
        return re.sub(r"<[^>]+>", " ", text)


SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>", "<UNK>", "<EN>", "<PT>", "<ES>", "<FR>"]


def normalize_text(text):
    text = remove_tags(text)
    text = text.lower().strip()
    text = re.sub(r"([.!?,])", r" \1 ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.!?,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_text_files(path):
    if os.path.isfile(path):
        yield path
        return

    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            file_path = os.path.join(path, name)
            if os.path.isfile(file_path) and name.endswith(".txt"):
                yield file_path
        return

    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_file(path):
    lines = []

    for file_path in iter_text_files(path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines.extend(f.readlines())

    return lines


def paired_text_files(src_path, trg_path):
    if os.path.isfile(src_path) and os.path.isfile(trg_path):
        return [(src_path, trg_path)]

    if os.path.isdir(src_path) and os.path.isdir(trg_path):
        src_files = {
            os.path.basename(path): path
            for path in iter_text_files(src_path)
        }
        trg_files = {
            os.path.basename(path): path
            for path in iter_text_files(trg_path)
        }
        names = sorted(src_files.keys() & trg_files.keys())

        if not names:
            raise ValueError(f"No matching .txt files found in {src_path} and {trg_path}")

        return [(src_files[name], trg_files[name]) for name in names]

    raise ValueError(
        "Parallel inputs must both be files or both be directories: "
        f"{src_path}, {trg_path}"
    )


def build_vocab(sentences, min_count=2):
    counter = Counter()

    for sent in sentences:
        counter.update(sent.split())

    token_to_idx = {}

    for tok in SPECIAL_TOKENS:
        token_to_idx[tok] = len(token_to_idx)

    for tok, count in counter.most_common():
        if count >= min_count and tok not in token_to_idx:
            token_to_idx[tok] = len(token_to_idx)

    idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}

    return {
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
    }


def encode_sentence(sentence, token_to_idx, lang_token):
    tokens = ["<START>", lang_token] + sentence.split() + ["<END>"]
    return [token_to_idx.get(tok, token_to_idx["<UNK>"]) for tok in tokens]


def valid_sentence(sentence, min_len, max_len):
    n = len(sentence.split())
    return min_len <= n <= max_len


def make_parallel_dataset(src_path, trg_path, token_to_idx, trg_lang, min_len, max_len):
    data = []

    for src_file, trg_file in paired_text_files(src_path, trg_path):
        src_lines = read_file(src_file)
        trg_lines = read_file(trg_file)

        for src, trg in zip(src_lines, trg_lines):
            src = normalize_text(src)
            trg = normalize_text(trg)

            if not src or not trg:
                continue

            if not valid_sentence(src, min_len, max_len):
                continue

            if not valid_sentence(trg, min_len, max_len):
                continue

            src_ids = encode_sentence(src, token_to_idx, "<EN>")
            trg_ids = encode_sentence(trg, token_to_idx, f"<{trg_lang.upper()}>")

            data.append((src_ids, trg_ids))

    return data


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default="data/txt")
    parser.add_argument("--output-dir", default="data/train/europarl")

    parser.add_argument("--en-file", default="en")
    parser.add_argument("--pt-file", default="pt")
    parser.add_argument("--es-file", default="es")
    parser.add_argument("--fr-file", default="fr")

    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.9)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = {
        "en": os.path.join(args.data_dir, args.en_file),
        "pt": os.path.join(args.data_dir, args.pt_file),
        "es": os.path.join(args.data_dir, args.es_file),
        "fr": os.path.join(args.data_dir, args.fr_file),
    }

    # 1. Build multilingual vocab
    all_sentences = []

    for path in tqdm(files.values(), desc="Processing files"):
        for line in tqdm(read_file(path), desc=f"Reading {path.name}", leave=False):
            line = normalize_text(line)
            if valid_sentence(line, args.min_len, args.max_len):
                all_sentences.append(line)
                
    vocab = build_vocab(all_sentences, args.min_count)

    vocab_path = os.path.join(args.output_dir, "vocab_multilingual.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    token_to_idx = vocab["token_to_idx"]

    print("Saved vocab:", vocab_path)
    print("Vocab size:", len(token_to_idx))

    # 2. Create EN→EN for teacher/KD
    en_en_data = make_parallel_dataset(
        files["en"],
        files["en"],
        token_to_idx,
        "en",
        args.min_len,
        args.max_len,
    )

    # 3. Create EN→PT / EN→ES / EN→FR for LoRA
    datasets = {
        "en_en": en_en_data,
        "en_pt": make_parallel_dataset(files["en"], files["pt"], token_to_idx, "pt", args.min_len, args.max_len),
        "en_es": make_parallel_dataset(files["en"], files["es"], token_to_idx, "es", args.min_len, args.max_len),
        "en_fr": make_parallel_dataset(files["en"], files["fr"], token_to_idx, "fr", args.min_len, args.max_len),
    }

    for name, data in tqdm(datasets.items()):
        split = int(len(data) * args.train_ratio)

        train_data = data[:split]
        test_data = data[split:]

        save_pickle(train_data, os.path.join(args.output_dir, f"train_{name}.pkl"))
        save_pickle(test_data, os.path.join(args.output_dir, f"test_{name}.pkl"))

        print(name, "train:", len(train_data), "test:", len(test_data))


if __name__ == "__main__":
    main()
