import pickle
import torch
from torch.utils.data import Dataset


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