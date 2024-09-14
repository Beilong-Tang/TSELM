from collections import OrderedDict
from hyperpyyaml import load_hyperpyyaml
import torch
import torch.nn as nn
from typing import Optional


def strip_ddp_state_dict(state_dict):
    # Create a new state dict without DDP keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            # Remove "module." prefix
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_model(config_path: str, ckpt_path: str, device="cuda", strict=False):
    """
    Load the model from config_path and load the checkpoint of the model
    """
    with open(config_path, "r") as f:
        config: dict = load_hyperpyyaml(f)
    model: nn.Module = config.get("model")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = strip_ddp_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.cuda(device)
    model.eval()
    return model


def load_ckpt(
    model: nn.Module, ckpt_path: Optional[str], device="cuda", strict=True, freeze=True
):
    model.to(device)
    if device == "cuda":
        model.cuda(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = strip_ddp_state_dict(ckpt["model_state_dict"])
        ### output missing part
        # missing, _, _, msg = _find_mismatched_keys(model.state_dict(), state_dict)
        # if missing:
        #     print(msg)
        model.load_state_dict(state_dict, strict=strict)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval()
    return model


def _find_mismatched_keys(model_state_dict, checkpoint_state_dict):
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(checkpoint_state_dict.keys())

    # Keys that are in the model but not in the checkpoint
    missing_in_checkpoint = model_keys - checkpoint_keys
    # Keys that are in the checkpoint but not in the model
    missing_in_model = checkpoint_keys - model_keys
    # check if there is missing
    missing = (len(missing_in_checkpoint) != 0) or (len(missing_in_model) != 0)
    # missing message
    msg = f"Keys that are in the model but not in the checkpoint:\n{missing_in_checkpoint}\nKeys that are in the checkpoint but not in the model:\n{missing_in_model}"

    return missing, missing_in_checkpoint, missing_in_model, msg


def load_freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def count_prams(model: nn.Module, ignore_freeze=True):
    if ignore_freeze:
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        return sum([p.numel() for p in model.parameters()])
