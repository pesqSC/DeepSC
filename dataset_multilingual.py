import os
import pickle

import torch
from torch.utils.data import Dataset

class EurParallelDatasetBPE(Dataset):
    """
    Parallel source/target dataset for multilingual DeepSC BPE data.

    Expected pickle format:
        [
            ([source_ids...], [target_ids...]),
            ...
        ]
    """

    def __init__(
        self,
        language,
        split="train",
        data_dir="./data/train/europarl_bpe",
    ):
        if split not in {"train", "test"}:
            raise ValueError(
                f"split must be 'train' or 'test', got {split!r}"
            )

        pkl_path = os.path.join(
            data_dir,
            f"{split}_{language}.pkl",
        )

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"BPE dataset not found: {pkl_path}"
            )

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

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
    """Pad variable-length BPE source/target sequences."""
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


class EurParallelDataset(Dataset):
    def __init__(self, language,  split="train"):
        pkl_path = "./data/train/europarl/{}_{}.pkl".format(split, language)
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)  # [(src_ids, trg_ids), ...]

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