import torch
import torch.nn as nn
from utils import PowerNormalize, Channels

class Transmitter(nn.Module):
    """
    Transmitter class for the DeepSC model.
    """
    def __init__(self, encoder: nn.Module, channel_encoder: nn.Module):
        """
        Initialize the Transmitter model with an encoder and a channel encoder.

        Args:
            encoder (nn.Module): encoder model
            channel_encoder (nn.Module): channel encoder model
        """
        super(Transmitter, self).__init__()
        self.encoder = encoder # encoder model
        self.channel_encoder = channel_encoder # channel encoder model

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, channel_type: str, noise_std: float) -> torch.Tensor:
        """
        Forward pass through the transmitter model.
        
        Args:
            src (torch.Tensor): input tensor of shape (batch_size, seq_len)
            src_mask (torch.Tensor): mask tensor of shape (batch_size, seq_len)
        
        Returns:
            en_out (torch.Tensor): output tensor from encoder of shape (batch_size, seq_len, d_model)
            ch_en_out (torch.Tensor): output tensor from channel encoder of shape (batch_size, seq_len, d_model)
            Tx_sig (torch.Tensor): normalized transmitted signal of shape (batch_size, seq_len, d_model)
        """
        en_out = self.encoder(src, src_mask)
        ch_en_out = self.channel_encoder(en_out)
        Tx_sig = PowerNormalize(ch_en_out)
        z_noisy = self.apply_channel(Tx_sig, channel_type=channel_type, noise_std=noise_std)
        return en_out, ch_en_out, Tx_sig, z_noisy
    
    def apply_channel(self, Tx_sig: torch.Tensor, channel_type: str, noise_std: float) -> torch.Tensor:
        """
        Apply a channel to the transmitted signal.
        
        Args:
            Tx_sig (torch.Tensor): transmitted signal of shape (batch_size, seq_len, d_model)
            channel_type (str): type of channel to apply
            noise_std (float): standard deviation of noise to add
        
        Returns:
            z_noisy (torch.Tensor): noisy signal of shape (batch_size, seq_len, d_model)
        """
        channels = Channels()
        if channel_type == 'AWGN':
            z_noisy = channels.AWGN(Tx_sig, noise_std)
        elif channel_type == 'Rayleigh':
            z_noisy = channels.Rayleigh(Tx_sig, noise_std)
        elif channel_type == 'Rician':
            z_noisy = channels.Rician(Tx_sig, noise_std)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
        return z_noisy
