import os
import pickle
from typing import Dict, Iterable, Sequence

import torch
from torch.utils.data import Dataset

SUPPORTED_LANGUAGES = ("en", "pt", "es", "fr")


class EurMultilingualDatasetBPE(Dataset):
    """
    Multilingual DeepSC dataset produced by preprocess_text_mu_bpe.py.

    Expected pickle record:
        {
            "id": "ep-0001",
            "tx": [<START>, ..., <END>],
            "targets": {
                "en": [<START>, <EN>, ..., <END>],
                "pt": [<START>, <PT>, ..., <END>],
                "es": [<START>, <ES>, ..., <END>],
                "fr": [<START>, <FR>, ..., <END>],
            },
        }

    The source sequence is shared across languages. The training loop chooses
    which target language to optimize for each batch.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: str = "./data/train/europarl_bpe",
        languages: Sequence[str] = SUPPORTED_LANGUAGES,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got {split!r}"
            )

        self.languages = tuple(lang.lower() for lang in languages)
        unsupported = set(self.languages).difference(SUPPORTED_LANGUAGES)
        if unsupported:
            raise ValueError(
                f"Unsupported languages: {sorted(unsupported)}. "
                f"Supported: {list(SUPPORTED_LANGUAGES)}"
            )

        if not self.languages:
            raise ValueError("At least one target language must be enabled.")

        self.pkl_path = os.path.join(data_dir, f"{split}_multilingual.pkl")
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(
                f"Multilingual BPE dataset not found: {self.pkl_path}"
            )

        with open(self.pkl_path, "rb") as file:
            self.data = pickle.load(file)

        if not isinstance(self.data, list):
            raise ValueError(
                f"Unexpected dataset format in {self.pkl_path}: expected a list."
            )

        if self.data:
            self._validate_sample(self.data[0], sample_index=0)

    def _validate_sample(self, sample, sample_index: int) -> None:
        if not isinstance(sample, dict):
            raise ValueError(
                f"Sample {sample_index} in {self.pkl_path} must be a dict."
            )

        required = {"id", "tx", "targets"}
        missing = required.difference(sample)
        if missing:
            raise ValueError(
                f"Sample {sample_index} in {self.pkl_path} is missing: "
                f"{sorted(missing)}"
            )

        if not isinstance(sample["tx"], (list, tuple)):
            raise ValueError(
                f"Sample {sample_index}: 'tx' must be a token-ID sequence."
            )

        if not isinstance(sample["targets"], dict):
            raise ValueError(
                f"Sample {sample_index}: 'targets' must be a dict."
            )

        missing_langs = set(self.languages).difference(sample["targets"])
        if missing_langs:
            raise ValueError(
                f"Sample {sample_index} is missing targets for: "
                f"{sorted(missing_langs)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            "id": str(sample["id"]),
            "src": torch.tensor(sample["tx"], dtype=torch.long),
            "targets": {
                lang: torch.tensor(sample["targets"][lang], dtype=torch.long)
                for lang in self.languages
            },
        }


def collate_multilingual_bpe(batch, pad_idx: int = 0):
    """
    Pad the shared TX sequence and every enabled target language.

    Returns
    -------
    {
        "ids": [str, ...],
        "src": LongTensor[B, T_src],
        "targets": {
            "en": LongTensor[B, T_en],
            ...
        }
    }
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    ids = [sample["id"] for sample in batch]
    src_batch = torch.nn.utils.rnn.pad_sequence(
        [sample["src"] for sample in batch],
        batch_first=True,
        padding_value=pad_idx,
    )

    languages = tuple(batch[0]["targets"].keys())
    target_batches: Dict[str, torch.Tensor] = {}

    for lang in languages:
        if any(lang not in sample["targets"] for sample in batch):
            raise ValueError(f"Inconsistent target language '{lang}' in batch.")

        target_batches[lang] = torch.nn.utils.rnn.pad_sequence(
            [sample["targets"][lang] for sample in batch],
            batch_first=True,
            padding_value=pad_idx,
        )

    return {
        "ids": ids,
        "src": src_batch,
        "targets": target_batches,
    }


# -----------------------------------------------------------------------------
# Legacy pair-wise BPE dataset retained for older experiments/scripts.
# -----------------------------------------------------------------------------
class EurParallelDatasetBPE(Dataset):
    """Legacy pair-wise BPE dataset: [(source_ids, target_ids), ...]."""

    def __init__(
        self,
        language,
        split="train",
        data_dir="./data/train/europarl_bpe",
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got {split!r}"
            )

        pkl_path = os.path.join(data_dir, f"{split}_{language}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"BPE dataset not found: {pkl_path}")

        with open(pkl_path, "rb") as file:
            self.data = pickle.load(file)

        if self.data:
            sample = self.data[0]
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise ValueError(
                    f"Unexpected dataset format in {pkl_path}. "
                    "Expected (source_ids, target_ids) pairs."
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        return torch.LongTensor(src), torch.LongTensor(trg)


def collate_parallelBPE(batch, pad_idx=0):
    """Pad variable-length legacy BPE source/target sequences."""
    src_batch, trg_batch = zip(*batch)

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_idx,
    )
    trg_batch = torch.nn.utils.rnn.pad_sequence(
        trg_batch,
        batch_first=True,
        padding_value=pad_idx,
    )

    return src_batch, trg_batch


# -----------------------------------------------------------------------------
# Original word-level dataset retained for backward compatibility.
# -----------------------------------------------------------------------------
class EurParallelDataset(Dataset):
    def __init__(self, language, split="train"):
        pkl_path = "./data/train/europarl/{}_{}.pkl".format(split, language)
        with open(pkl_path, "rb") as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        return torch.LongTensor(src), torch.LongTensor(trg)


def collate_parallel(batch, pad_idx=0):
    src_batch, trg_batch = zip(*batch)

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    trg_batch = torch.nn.utils.rnn.pad_sequence(
        trg_batch, batch_first=True, padding_value=pad_idx
    )

    return src_batch, trg_batch
