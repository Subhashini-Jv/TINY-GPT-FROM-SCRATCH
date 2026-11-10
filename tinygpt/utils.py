from __future__ import annotations

from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Return (model, tokenizer, checkpoint dict) from a file written by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = CharTokenizer(ckpt["chars"])
    return model, tok, ckpt
