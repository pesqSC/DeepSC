# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:00:00 2026

@author: Prinako
model_utils.py
"""
import os 
import re
import csv
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from typing import Optional

import random

from student import Student

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay




class SeqtoText:
    """
    Convert token IDs back into readable text.

    Features:
      - stops at <END>
      - skips special/control tokens
      - supports multilingual language tokens
      - handles unknown token IDs safely
      - fixes punctuation spacing
    """

    DEFAULT_SKIP_TOKENS = {
        "<PAD>",
        "<START>",
        "<EN>",
        "<PT>",
        "<ES>",
        "<FR>",
    }

    def __init__(
        self,
        vocb_dictionary,
        end_idx,
        skip_tokens=None,
        unk_token="<UNK>",
    ):
        """
        Parameters
        ----------
        vocb_dictionary : dict
            token -> index vocabulary.

        end_idx : int
            Index of <END>.

        skip_tokens : iterable, optional
            Tokens that should not appear in final text.

        unk_token : str
            Representation for unknown token IDs.
        """

        if not isinstance(vocb_dictionary, dict):
            raise TypeError(
                "vocb_dictionary must be a dictionary."
            )

        self.reverse_word_map = {
            idx: token
            for token, idx in vocb_dictionary.items()
        }

        self.end_idx = int(end_idx)

        if skip_tokens is None:
            self.skip_tokens = set(
                self.DEFAULT_SKIP_TOKENS
            )
        else:
            self.skip_tokens = set(skip_tokens)

        self.unk_token = unk_token

    def sequence_to_tokens(
        self,
        list_of_indices,
        keep_unknown=True,
    ):
        """
        Convert IDs to tokens before text formatting.
        """

        tokens = []

        for idx in list_of_indices:

            # Torch scalar -> Python int
            if hasattr(idx, "item"):
                idx = idx.item()

            idx = int(idx)

            # Stop decoding at <END>
            if idx == self.end_idx:
                break

            token = self.reverse_word_map.get(idx)

            # Unknown ID
            if token is None:
                if keep_unknown:
                    tokens.append(self.unk_token)
                continue

            # Ignore control tokens
            if token in self.skip_tokens:
                continue

            tokens.append(token)

        return tokens

    def sequence_to_text(self,list_of_indices,keep_unknown=True):
        """
        Convert token IDs into clean readable text.
        """

        tokens = self.sequence_to_tokens(
            list_of_indices,
            keep_unknown=keep_unknown,
        )

        text = " ".join(tokens)

        # No space before:
        # . , ! ? ; : %
        text = re.sub(
            r"\s+([.,!?;:%])",
            r"\1",
            text,
        )

        # Opening brackets should not have trailing space
        text = re.sub(
            r"([\(\[\{])\s+",
            r"\1",
            text,
        )

        # Closing brackets should not have leading space
        text = re.sub(
            r"\s+([\)\]\}])",
            r"\1",
            text,
        )

        # Normalize multiple spaces
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def text_to_indices(text, token_to_idx, start_idx, end_idx, pad_idx, max_length, lang_token="<EN>"):
    if max_length < 3:
        raise ValueError("max_length must fit START, language token, and END.")

    unk_idx = token_to_idx["<UNK>"]
    # print(unk_idx)
    lang_idx = token_to_idx[lang_token]

    tokens = text.lower().strip().split()
    token_ids = [token_to_idx.get(tok, unk_idx) for tok in tokens]

    token_ids = token_ids[: max_length - 3]

    seq = [start_idx, lang_idx] + token_ids + [end_idx]
    seq += [pad_idx] * (max_length - len(seq))

    return seq


def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?,;:])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)



def power_normalize(x: torch.Tensor)-> torch.Tensor:
    """
    Normalize the power of a signal to 1.

    Args:
        x: The signal to normalize with shape [batch_size, seq_len, num_channels].

    Returns:
        The normalized signal with shape [batch_size, seq_len, num_channels].
    """
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


def snr_to_noise(snr: float)-> float:
    """
    Convert a signal-to-noise ratio in dB to a noise standard deviation.

    Args:
        snr: The signal-to-noise ratio in dB.

    Returns:
        The noise standard deviation.
    """
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def send_through_channel(tx: torch.Tensor, channel: str, snr: float)-> torch.Tensor:
    """
    Send a signal through a channel and return the received signal.

    Args:
        tx: The transmitted signal with shape [batch_size, seq_len, num_channels].
        channel: The channel to send the signal through.
        snr: The signal-to-noise ratio in dB.

    Returns:
        The received signal with shape [batch_size, seq_len, num_channels] 
    """
    noise_std = snr_to_noise(snr)
    channels = Channels()

    if channel == "AWGN":
        return channels.AWGN(tx, noise_std)
    elif channel == "Rayleigh":
        return channels.Rayleigh(tx, noise_std)
    elif channel == "Rician":
        return channels.Rician(tx, noise_std)
    else:
        raise ValueError("Channel must be AWGN, Rayleigh or Rician")

    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    Greedy decode the received signal.

    Args:
        model: The model to decode the signal.
        src: The source signal with shape [batch_size, seq_len, num_channels].
        n_var: The noise variance.
        max_len: The maximum length of the decoded signal.
        padding_idx: The padding index.
        start_symbol: The start symbol.
        channel: The channel to send the signal through.

    Returns:
        The decoded signal with shape [batch_size, seq_len, num_channels].
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = power_normalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            
    #channel_enc_output = model.blind_csi(channel_enc_output)
          
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


# Load the teacher model
def load_teacher(model, encoder_path, decoder_path, device):
    """
    Load the teacher model.

    Args:
        model: The model to load the teacher model.
        encoder_path: The path to the encoder model.
        decoder_path: The path to the decoder model.
        device: The device to load the model on.
    """
    enc = torch.load(encoder_path, map_location=device, weights_only=False)
    dec = torch.load(decoder_path, map_location=device, weights_only=False)

    model.encoder.load_state_dict(enc["encoder"])
    model.channel_encoder.load_state_dict(enc["channel_encoder"])
    model.channel_decoder.load_state_dict(dec["channel_decoder"])
    model.decoder.load_state_dict(dec["decoder"])
    model.dense.load_state_dict(dec["dense"])

    print("Teacher loaded")
    return model

# Load the student model
@torch.no_grad()
def load_student(student, checkpoint_path, device):
    """
    Load the student model.

    Args:
        student: The student model to load.
        checkpoint_path: The path to the checkpoint.
        device: The device to load the model on.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if all(k in ckpt for k in ("channel_decoder", "decoder", "dense")):
        student.channel_decoder.load_state_dict(ckpt["channel_decoder"])
        student.decoder.load_state_dict(ckpt["decoder"])
        student.dense.load_state_dict(ckpt["dense"])
    else:
        state = ckpt.get("model_state_dict", ckpt)
        student.load_state_dict(state)

    print("Student loaded:", checkpoint_path)
    return student


# Checkpoint helpers
def save_student_receiver(student: Student, path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "channel_decoder": student.channel_decoder.state_dict(),
        "decoder": student.decoder.state_dict(),
        "dense": student.dense.state_dict(),
    }
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Token-level cross-entropy averaged over non-PAD positions.

    logits:  [B, T, V]
    targets: [B, T]
    """

    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits with shape [B,T,V], got {logits.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets with shape [B,T], got {targets.shape}"
        )

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, targets={targets.shape}"
        )

    vocab_size = logits.size(-1)
    flat_logits = logits.flatten(0, 1)
    flat_targets = targets.flatten()

    ce_loss_raw = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )

    valid = (flat_targets != pad_idx).to(ce_loss_raw.dtype)
    n_valid = valid.sum().clamp_min(1.0)


    avg_loss = (ce_loss_raw * valid ).sum() / n_valid

    # Convert scalar loss to float
    loss_val = avg_loss.detach().item()

    if math.isnan(loss_val) or math.isinf(loss_val):
        perplexity = float("inf")
    else:
        # Clamp exponent input to avoid math.exp overflow
        clamped_loss = min(loss_val, 700.0)
        perplexity = math.exp(clamped_loss)
    
    return avg_loss, perplexity

def masked_ce_loss2(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    """
    Cross-entropy loss that ignores PAD tokens.

    Args:
        student_logits:
            Model predictions with shape [batch_size, sequence_length, vocab_size].

        targets:
            Correct token IDs with shape [batch_size, sequence_length].

        pad_idx:
            Vocabulary index of the <PAD> token.

    Returns:
        Scalar cross-entropy loss averaged over non-PAD tokens.
    """

    # student_logits: [B, T, V]
    # targets:        [B, T]

    if student_logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits have sequence shape "
            f"{student_logits.shape[:2]}, but targets have shape {targets.shape}"
        )

    batch_size, sequence_length, vocab_size = student_logits.shape

    # CrossEntropyLoss expects:
    # predictions: [N, V]
    # targets:     [N]
    logits_flat = student_logits.reshape(
        batch_size * sequence_length,
        vocab_size
    )

    targets_flat = targets.reshape(
        batch_size * sequence_length
    )

    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_idx,
        reduction="mean",
    )

    return loss


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_epoch_results(csv_path, epoch, metrics):
    """
    Save one epoch of training results to a CSV file.

    Args:
        csv_path: Output CSV path.
        epoch: Current epoch number.
        metrics: Dictionary containing metric names and values.
    """

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    row = {
        "epoch": epoch+1,
        **metrics,
    }

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

