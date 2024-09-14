import torch.nn as nn 
from typing import List

class Base(nn.Module):
    def __init__(
        self,
        hifi_gan: nn.Module,  ##
        discrete_ssl: nn.Module,
        ssl_layers: List[int],
        attention_mlp: nn.Module,
        lm_model: nn.Module,
        embedding: nn.Module,
        head: nn.Module,
        base_loss,
        vocab_size: int,
        num_spks=2,
    ):
        super().__init__()
        self.hifi_gan = load_freeze_model(hifi_gan)
        self.discrete_ssl = load_freeze_model(discrete_ssl)
        self.ssl_layers = ssl_layers
        self.attention_mlp = attention_mlp
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.head = head
        self.lm = lm_model
        self.num_spks = num_spks
        self.pit_loss = PitWrapper(base_loss)
        print(
            f"model parameters {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    @torch.no_grad()
    def sig_to_toks(self, audio):
        toks, _, _ = self.discrete_ssl(audio, SSL_layers=self.ssl_layers)
        return toks  # [B, N, K]

    @torch.no_grad()
    def toks_to_sig(self, toks):
        # toks: [B, N, K]
        self.hifi_gan.device = toks.device
        self.hifi_gan.to(toks.device)
        all_layer_ids = [1, 3, 7, 12, 18, 23]
        offsets = torch.arange(
            0,
            len(all_layer_ids) * self.vocab_size,
            self.vocab_size,
            device=toks.device,
        )
        offset_idxes = [all_layer_ids.index(x) for x in self.ssl_layers]
        offsets = offsets[offset_idxes]
        toks = toks + offsets + 1

        # Handle missing codebooks
        if len(self.ssl_layers) < len(all_layer_ids):
            full_toks = torch.zeros(
                *toks.shape[:2],
                len(all_layer_ids),
                dtype=toks.dtype,
                device=toks.device,
            )
            for i, idx in enumerate(offset_idxes):
                full_toks[..., idx] = toks[..., i]
            toks = full_toks
        self.hifi_gan.tokenize = False
        sig = self.hifi_gan(toks)[:, 0]  # [B,T]
        return sig

    def _error(self, out_toks, true_toks):
        """
        Calculate the error in percentage (0-100)
        """
        error = (1 - (out_toks == true_toks).sum() / out_toks.numel()) * 100
        return error

    @torch.no_grad()
    def recon(self, toks: torch.Tensor):
        """
        Reconstruct the audio using the token and vocoder.

        Args:
            toks: the tokens of shape [B,N,S,K] if num_spk is not 1 else [B, N, K]
        Returns:
            audio: the audio of shape [B * S, T]
        """
        if self.num_spks == 1:
            toks = toks.unsqueeze(2)
        toks = toks.movedim(-2, -3).contiguous()  # [B,S,N,K]
        rec_sig = self.toks_to_sig(toks.flatten(end_dim=1))  # [BS,T]
        return rec_sig
