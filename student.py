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