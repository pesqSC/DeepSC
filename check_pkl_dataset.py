# check_pkl_dataset.py
import pickle
import json
from pathlib import Path

PKL_PATH = "data/train/europarl/train_en_en.pkl"
VOCAB_PATH = "data/train/europarl/vocab_multilingual.json"

def datasetValidator(plk_path, vocab_path, index):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    token_to_idx = vocab["token_to_idx"]
    idx_to_token = {int(v): k for k, v in token_to_idx.items()}

    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    en_idx = token_to_idx["<EN>"]

    with open(plk_path, "rb") as f:
        data = pickle.load(f)

    print("Total samples:", len(data))
    print("Type:", type(data))
    print("First item type:", type(data[index]))
    print("First item length:", len(data[index]))

    src, trg = data[index]

    print("\nSRC ids:", src[:34])
    print("TRG ids:", trg[:34])

    def decode(ids):
        return " ".join(idx_to_token.get(int(i), "<MISSING>") for i in ids)

    print("\nSRC text:")
    print(decode(src))

    print("\nTRG text:")
    print(decode(trg))

    # Basic checks
    errors = 0

    for i, item in enumerate(data[:1000]):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            print("Bad item format at:", i)
            errors += 1
            continue

        src, trg = item

        if src[0] != start_idx or trg[0] != start_idx:
            print("Missing <START> at:", i)
            errors += 1

        if src[-1] != end_idx or trg[-1] != end_idx:
            print("Missing <END> at:", i)
            errors += 1

        if src[1] != en_idx or trg[1] != en_idx:
            print("Missing <EN> at:", i)
            errors += 1

        if max(src) >= len(token_to_idx) or max(trg) >= len(token_to_idx):
            print("Token id outside vocab at:", i)
            errors += 1

    print("\nErrors found:", errors)

    if errors == 0:
        print("✅ Dataset looks correct.")
    else:
        print("❌ Dataset has problems.")