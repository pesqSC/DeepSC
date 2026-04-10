from models.transceiver import DeepSC

def build_teacher(
    num_vocab: int,
    max_len: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    dff: int,
    dropout: float,
    device: torch.device,
):
    model = DeepSC(
        num_layers=num_layers,
        src_vocab_size=num_vocab,
        trg_vocab_size=num_vocab,
        src_max_len=max_len,
        trg_max_len=max_len,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout=dropout,
    ).to(device)
    return model


def teacher_tx_forward(
    teacher: DeepSC,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    channel_type: str,
    noise_std: float,
):
    with torch.no_grad():
        sem_feat = teacher.encoder(src, src_mask)
        tx_feat = teacher.channel_encoder(sem_feat)
        z_noisy = apply_channel(tx_feat, channel_type=channel_type, noise_std=noise_std)
    return z_noisy


def teacher_tr_forward(
    teacher: DeepSC,
    z_noisy: torch.Tensor,
    trg_inp: torch.Tensor,
    look_ahead_mask: torch.Tensor,
    trg_padding_mask: torch.Tensor,
):
    with torch.no_grad():
        t_rx = teacher.channel_decoder(z_noisy)
        t_dec = teacher.decoder(trg_inp, t_rx, look_ahead_mask, trg_padding_mask)
        t_logits = teacher.dense(t_dec)
    return t_logits, t_rx, t_dec
