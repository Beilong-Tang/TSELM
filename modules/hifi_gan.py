"""
Wrapper class for WavLM Scalable HiFi-GAN from https://huggingface.co/speechbrain/hifigan-wavlm-l1-3-7-12-18-23-k1000-LibriTTS 
"""
import torch 
import torch.nn as nn 
import os.path as op
from hyperpyyaml import load_hyperpyyaml

class HiFiGAN(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        ckpt_path = torch.load(op.join(model_path, "generator.ckpt"), map_location="cpu")
        with open(op.join(model_path,"hyperparams.yaml"), "r") as f:
            config = load_hyperpyyaml(f)
        model = config.get("generator")
        model.load_state_dict(ckpt_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
    
    @torch.no_grad()
    def forward(self, toks):
        """Reconstruct audio from given tokens
        
        Arguments
        ---------
        toks: [B, T, N]
            N stands for the number of layers
        
        Returns
        -------
        wav: [B,T]
            The reconstructed wav
        """
        return self.model(toks)[0].squeeze(1)
