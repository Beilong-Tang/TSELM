"""
Wrapper class for WavLM Large of HuggingFace
https://huggingface.co/microsoft/wavlm-large
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel


class WavLM(nn.Module):
    def __init__(
        self,
        model_path: str,
        output_norm=False,
        output_all_hiddens=True,
        normalize_wav=True,
    ):
        """Initialize WavLM Large

        Arguments
        ---------
        source: str
            The WavLM url for Hugging face or local directory
        output_norm: bool
            Whether to normalize the output.
        output_all_hiddens: bool
            Whether to output all the hidden layers
        normalize_wav:
            Whether to normalize the input wav before processing

        Example
        -------
        >>> wavlm = WavLM('microsoft/wavlm-large')
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_norm = output_norm
        self.output_all_hiddens = output_all_hiddens
        self.normalize_wav = normalize_wav

    @torch.no_grad()
    def extract_features(self, wav):
        """
        Arguments
        ---------
        wav: [B,T]
            The input wav.

        Returns
        -------
        out: torch.Tensor
            WavLM output.

        """
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        with torch.no_grad():
            out = self.model(
                wav,
                attention_mask=None,
                output_hidden_states=self.output_all_hiddens,
            )
        if self.output_all_hiddens:
            out = torch.stack(list(out.hidden_states), dim=0)
            norm_shape = out.shape[-3:]
        else:
            out = out.last_hidden_state
            norm_shape = out.shape
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, norm_shape[1:])
        return out
