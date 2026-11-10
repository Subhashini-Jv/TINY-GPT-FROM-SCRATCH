"""A small GPT, written out piece by piece so every line can be read and understood.

The model is a decoder-only transformer:

    token ids -> token embedding + position embedding
              -> N x [LayerNorm -> causal self-attention -> residual add
                      LayerNorm -> MLP                    -> residual add]
              -> final LayerNorm -> linear head -> logits over the vocabulary

Attention is implemented by hand (no fused kernels) so that the attention weights
can be captured and drawn as a heatmap. A key/value cache makes generation fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # longest context the model can see, in characters
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False  # bias terms in Linear and LayerNorm layers

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position may only look backwards."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # One matrix produces queries, keys and values for every head at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: row i has ones in columns 0..i.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)
        self.last_attn: torch.Tensor | None = None  # filled when capture is on
        self.capture = False

    def forward(self, x: torch.Tensor, kv_cache: tuple | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                past = past_k.size(2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        S = k.size(2)
        # Query i (absolute position past+i) may see keys 0..past+i.
        causal = self.mask[:, :, past:past + T, :S]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.capture:
            self.last_attn = att.detach()
        att = self.attn_drop(att)
        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue heads back together
        return self.resid_drop(self.proj(y)), new_cache


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer layer. Pre-norm: normalise first, then add the result back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + a  # "communicate": tokens exchange information
        x = x + self.mlp(self.ln2(x))  # "compute": each token thinks on its own
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the same matrix reads tokens in and writes them out.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the residual projections so the residual stream doesn't
        # grow with depth (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # Position embeddings are usually left out of the headline count.
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_caches: list | None = None, start_pos: int = 0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, (
            f"sequence of length {start_pos + T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, c = block(x, cache)
            new_caches.append(c)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        # At inference time only the last position's prediction is needed.
        logits = self.head(x[:, -1:, :])
        return logits, new_caches

    def set_attention_capture(self, on: bool):
        for b in self.blocks:
            b.attn.capture = on
            if not on:
                b.attn.last_attn = None

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """AdamW, with weight decay only on matrices (not on norms or biases)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 use_cache: bool = True, stream=None) -> torch.Tensor:
        """Sample one token at a time: x_{t+1} ~ softmax(z_t / T).

        temperature < 1 makes the model more confident, > 1 more random.
        top_k keeps only the k most likely characters; top_p keeps the smallest set
        whose probabilities add up to p. `stream` is called with each new token id.
        """
        was_training = self.training
        self.eval()
        bs = self.cfg.block_size
        caches = None
        pos = 0
        for _ in range(max_new_tokens):
            if use_cache and caches is not None and pos < bs:
                # Feed only the newest token; earlier keys/values are cached.
                logits, caches = self(idx[:, -1:], kv_caches=caches, start_pos=pos)
                pos += 1
            else:
                # First step, or the context window is full: run on the last
                # block_size tokens from scratch.
                ctx = idx[:, -bs:] if idx.size(1) > bs else idx
                if use_cache and caches is not None:
                    ctx = idx[:, -(bs // 2):]  # slide the window by half to amortise recompute
                logits, caches = self(ctx)
                pos = ctx.size(1)
                if not use_cache:
                    caches = None
            next_id = sample_from_logits(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_id], dim=1)
            if stream is not None:
                stream(next_id[0, 0].item())
        if was_training:
            self.train()
        return idx


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)  # greedy
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop a token if the tokens before it already cover top_p.
        drop = cum - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
