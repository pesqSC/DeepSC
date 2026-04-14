import numpy as np
import torch
import torch.nn as nn

from models.transceiver import ChannelDecoder, Decoder


class Student(nn.Module):
    
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, 
                 trg_max_len, d_model, num_heads, dff, dropout = 0.1):
        super(Student, self).__init__()
        
        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        self.dense = nn.Linear(d_model, trg_vocab_size)

    def forward(self, z_noisy: torch.Tensor,rg_inp: torch.Tensor, look_ahead_mask: torch.Tensor, sru_mask: torch.Tensor):
        ch_dec_out = self.channel_decoder(z_noisy)
        dec_out = self.decoder( rg_inp, ch_dec_out, look_ahead_mask, sru_mask)
        logits = self.dense(dec_out)
        return logits, ch_dec_out, dec_out