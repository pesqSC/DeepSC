# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information
from typing import Optional, Tuple
from tqdm import tqdm


from models.rx_model import Receiver
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
    def __init__(self, vocb_dictionary, end_idx, skip_tokens=None):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        self.skip_tokens = set(skip_tokens or ("<PAD>", "<START>", "<EN>", "<PT>", "<ES>", "<FR>"))
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            word = self.reverse_word_map.get(idx)
            if word is not None and word not in self.skip_tokens:
                words.append(word)
        words = ' '.join(words)
        return(words) 


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

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
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

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    
    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    
    #y_est = x +  torch.matmul(n, torch.inverse(H))
    #loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine
    # loss = loss_function(pred, trg_real, pad)

    loss.backward()
    opt.step()

    return loss.item()


def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)
    
    return loss.item()
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

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



@torch.no_grad()
def validate_multi_epoch(
    transmitter,
    teacher: Receiver,
    students: [Student],
    val_loader,
    pad_idx,
    channel,
    noise_std,
    device: torch.device,
    criterion,
    args,
):
    transmitter.eval()
    teacher.eval()
    for student in students:
        student.eval()

    student_1, student_2 = students

    total_loss_s1 = 0.0
    total_ce_s1 = 0.0
    total_kd_s1 = 0.0
    total_feat_s1 = 0.0

    total_loss_s2 = 0.0
    total_ce_s2 = 0.0
    total_kd_s2 = 0.0
    total_feat_s2 = 0.0

    pbar = tqdm(val_loader)

    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)

            trg_inp = trg[:, :-1]
            trg_real = trg[:, 1:]

            src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

            tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
                src, 
                src_mask, 
                channel, 
                noise_std
            )

            t_logits, rx_ch_dec_out, rx_dec_out = teacher(
                z_noisy=z_noisy, 
                trg_inp=trg_inp, 
                look_ahead_mask=look_ahead_mask,
                src_mask=src_mask
            )

            s1_logits, s1_ch_dec_out, s1_dec_out = student_1(
                z_noisy, 
                trg_inp, 
                look_ahead_mask, 
                src_mask
            )

            s2_logits, s2_ch_dec_out, s2_dec_out = student_2(
                z_noisy, 
                trg_inp, 
                look_ahead_mask, 
                src_mask
            )
            

            # ce = masked_ce_loss(s_logits, trg_real, pad_idx)
            ce_s1 = loss_function(
                s1_logits.contiguous().view(-1, s1_logits.size(-1)),
                trg_real.contiguous().view(-1), 
                pad_idx, 
                criterion
            )

            ce_s2 = loss_function(
                s2_logits.contiguous().view(-1, s2_logits.size(-1)),
                trg_real.contiguous().view(-1), 
                pad_idx, 
                criterion
            )


            kd_s1 = kd_kl_loss(s1_logits, t_logits, trg_real, pad_idx, args.temperature)
            kd_s2 = kd_kl_loss(s2_logits, t_logits, trg_real, pad_idx, args.temperature)
            
            # feat = masked_mse_loss(s_ch_dec_out, rx_ch_dec_out.detach(), trg_real, pad_idx)

            loss_s1 = (args.alpha * ce_s1) + (args.beta * kd_s1) #args.gamma # * feat
            loss_s2 = (args.alpha * ce_s2) + (args.beta * kd_s2) #args.gamma # * feat

            total_loss_s1 += float(loss_s1.item())
            total_ce_s1 += float(ce_s1.item())
            total_kd_s1 += float(kd_s1.item())
            # total_feat += float(feat.item())

            total_loss_s2 += float(loss_s2.item())
            total_ce_s2 += float(ce_s2.item())
            total_kd_s2 += float(kd_s2.item())

            pbar.set_description(f"Epoch {epoch + 1} Valid")

            pbar.set_postfix(
                L1=f"{loss_s1.item():.3f}",
                CE1=f"{s1_ce.item():.3f}",
                KD1=f"{kd_s1.item():.3f}",
                L2=f"{loss_s2.item():.3f}",
                CE2=f"{s2_ce.item():.3f}",
                KD2=f"{kd_s2.item():.3f}",
            )


    n = max(len(val_loader), 1)
    return[{
        "loss": total_loss_s1 / n,
        "ce": total_ce_s1 / n,
        "kd": total_kd_s1 / n,
        # "feat": total_feat / n,
    }, {
        "loss": total_loss_s2 / n,
        "ce": total_ce_s2 / n,
        "kd": total_kd_s2 / n,
        # "feat": total_feat / n,
    }]


# -----------------------------
# Checkpoint helpers
# -----------------------------
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


def feature_distillation_loss(student_feat, teacher_feat, targets, pad_idx):
    """
    MSE loss for feature-level distillation
    Args:
        student_feat: [B, T, D] student features
        teacher_feat: [B, T, D] teacher features  
        targets: [B, T] target tokens (for masking)
        pad_idx: padding index
    """
    # Create mask for valid positions
    mask = (targets != pad_idx).unsqueeze(-1).float()  # [B, T, 1]
    
    # Compute MSE loss
    mse_loss = F.mse_loss(student_feat, teacher_feat, reduction='none')  # [B, T, D]
    
    # Apply mask and average
    masked_loss = (mse_loss * mask).sum() / mask.sum().clamp_min(1.0)
    
    return masked_loss

def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
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

    flat_targets = targets.reshape(-1)

    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        flat_targets,
        reduction="none",
        ignore_index=pad_idx,
    )

    valid = (flat_targets != pad_idx).to(token_loss.dtype)

    return (token_loss * valid).sum() / valid.sum().clamp_min(1.0)

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

def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    temperature: float,
) -> torch.Tensor:

    min_len = min(student_logits.size(1), teacher_logits.size(1), targets.size(1))

    student_logits = student_logits[:, :min_len, :]
    teacher_logits = teacher_logits[:, :min_len, :]
    targets = targets[:, :min_len]

    # apply mask so PAD tokens don't dominate KD
    s_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)  # [B,T]
    valid = (targets != pad_idx).float()
    kl = (kl * valid).sum() / valid.sum().clamp_min(1.0)
    return kl * (temperature ** 2)
