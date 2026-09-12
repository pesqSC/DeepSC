import torch

from tqdm import tqdm
from models.rx_model import Receiver
from student import Student
from models.mutual_info import sample_batch, mutual_information

from utils.model_utils import (Channels, PowerNormalize, create_masks, masked_ce_loss)
from utils.kd_utils import (kd_kl_loss, feature_distillation_loss)



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


def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel, device):
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


@torch.no_grad()
def validate_one_epoch(
    transmitter,
    teacher: Receiver,
    student: Student,
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
    student.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    pbar = tqdm(val_loader)

    for batch in pbar:
        sents = batch.to(device)
        targets = batch.to(device)

        trg_inp = targets[:, :-1]
        trg_real = targets[:, 1:]

        src_mask, look_ahead_mask = create_masks(sents, trg_inp, pad_idx)

        tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
            sents, 
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

        s_logits, s_ch_dec_out, s_dec_out = student(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )
        

        # ce = masked_ce_loss(s_logits, trg_real, pad_idx)
        ce = loss_function(
            s_logits.contiguous().view(-1, s_logits.size(-1)),
            trg_real.contiguous().view(-1), 
            pad_idx, 
            criterion
        )

        kd = kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
        
        # feat = masked_mse_loss(s_ch_dec_out, rx_ch_dec_out.detach(), trg_real, pad_idx)

        loss = args.alpha * ce + args.beta * kd + args.gamma # * feat

        total_loss += float(loss.item())
        total_ce += float(ce.item())
        total_kd += float(kd.item())
        # total_feat += float(feat.item())

    n = max(len(val_loader), 1)
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kd": total_kd / n,
        # "feat": total_feat / n,
    }


@torch.no_grad()
def validate_multi_epoch(
    epoch,
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

            # s2_logits, s2_ch_dec_out, s2_dec_out = student_2(
            #     z_noisy, 
            #     trg_inp, 
            #     look_ahead_mask, 
            #     src_mask
            # )
            s1_ce = masked_ce_loss(
                s1_logits,
                trg_real,
                pad_idx
            )
            

            # ce = masked_ce_loss(s_logits, trg_real, pad_idx)
            # ce_s1 = loss_function(
            #     s1_logits.contiguous().view(-1, s1_logits.size(-1)),
            #     trg_real.contiguous().view(-1), 
            #     pad_idx, 
            #     criterion
            # )

            # ce_s2 = loss_function(
            #     s2_logits.contiguous().view(-1, s2_logits.size(-1)),
            #     trg_real.contiguous().view(-1), 
            #     pad_idx, 
            #     criterion
            # )


            kd_s1 = kd_kl_loss(s1_logits, t_logits, trg_real, pad_idx, args.temperature)
            # kd_s2 = kd_kl_loss(s2_logits, t_logits, trg_real, pad_idx, args.temperature)

            src_valid = (src != pad_idx).float()

            feat = feature_distillation_loss(
                student_feat=s1_ch_dec_out, 
                teacher_feat=rx_ch_dec_out.detach(), 
                targets=src,
                pad_idx=pad_idx,
            )

            loss_s1 = (args.alpha * s1_ce) + (args.beta * kd_s1) + args.gamma * feat
            # loss_s2 = (args.alpha * ce_s2) + (args.beta * kd_s2) #args.gamma # * feat

            total_loss_s1 += float(loss_s1.item())
            total_ce_s1 += float(args.alpha * s1_ce.item())
            total_kd_s1 += float(args.beta * kd_s1.item())
            # total_feat += float(feat.item())

            # total_loss_s2 += float(loss_s2.item())
            # total_ce_s2 += float(ce_s2.item())
            # total_kd_s2 += float(kd_s2.item())

            pbar.set_description(f"Epoch {epoch + 1} Valid")

            pbar.set_postfix(
                L1=f"{loss_s1.item():.3f}",
                CE1=f"{s1_ce.item():.3f}",
                KD1=f"{kd_s1.item():.3f}",
                # L2=f"{loss_s2.item():.3f}",
                # CE2=f"{ce_s2.item():.3f}",
                # KD2=f"{kd_s2.item():.3f}",
            )


    n = max(len(val_loader), 1)
    return[{
        "loss": total_loss_s1 / n,
        "ce": total_ce_s1 / n,
        "kd": total_kd_s1 / n,
        # "feat": total_feat / n,
    }
    #, {
    #     "loss": total_loss_s2 / n,
    #     "ce": total_ce_s2 / n,
    #     "kd": total_kd_s2 / n,
    #     # "feat": total_feat / n,
    # }
    ]



def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()