from utils import PowerNormalize

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

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
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
        return en_out, ch_en_out, Tx_sig
