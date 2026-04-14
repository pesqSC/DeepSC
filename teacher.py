import torch
from models.transceiver import DeepSC

def build_teacher(num_vocab: int, max_len: int, num_layers: int, d_model: int, num_heads: int, dff: int, dropout: float, device: torch.device) -> DeepSC:
    model = DeepSC(
        num_layers, 
        num_vocab, 
        num_vocab, 
        num_vocab, 
        num_vocab, 
        d_model, 
        num_heads, 
        dff, 
        dropout=dropout
    ).to(device)
    return model


def teacher_tx_forward(teacher: DeepSC, src: torch.Tensor, src_mask: torch.Tensor, channel_type: str, noise_std: float) -> torch.Tensor:
    with torch.no_grad():
        sem_feat = teacher.encoder(src, src_mask)
        tx_feat = teacher.channel_encoder(sem_feat)
        z_noisy = apply_channel(tx_feat, channel_type=channel_type, noise_std=noise_std)
    return z_noisy


def teacher_rx_forward(teacher, z_noisy: torch.Tensor, trg_inp: torch.Tensor, look_ahead_mask: torch.Tensor, trg_padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        t_ch_dec_out = teacher.channel_decoder(z_noisy)
        t_dec_out = teacher.decoder(trg_inp, t_ch_dec_out, look_ahead_mask, trg_padding_mask)
        logits = teacher.dense(t_dec_out)
    return logits, t_ch_dec_out, t_dec_out
