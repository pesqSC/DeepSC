
class Receiver(nn.Module):
    def __init__(self, channel_decoder: nn.Module, decoder: nn.Module, dense: nn.Module):
        super(Receiver, self).__init__()
        self.channel_decoder = channel_decoder    
        self.decoder = decoder
        self.dense = dense
    
    def forward(self, z_noisy: torch.Tensor, trg_inp: torch.Tensor, look_ahead_mask: torch.Tensor, trg_padding_mask: torch.Tensor):
        ch_dec_out = self.channel_decoder(z_noisy)
        dec_out = self.decoder(trg_inp, ch_dec_out, look_ahead_mask, trg_padding_mask)
        logits = self.dense(dec_out)
        return logits, ch_dec_out, dec_out