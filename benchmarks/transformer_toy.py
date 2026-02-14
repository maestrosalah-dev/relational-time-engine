# benchmarks/transformer_toy.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.flops_counter import FlopsCounter


@dataclass
class ToyConfig:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 4
    seq_len: int = 128


class SelfAttention(nn.Module):
    def __init__(self, cfg: ToyConfig, counter: FlopsCounter):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.counter = counter
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, D]
        """
        B, S, D = x.shape
        self.counter.add_tokens(B * S)

        # QKV linear
        self.counter.add_linear(B, S, D, 3 * D)
        qkv = self.qkv(x)  # [B,S,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to heads
        H = self.cfg.n_heads
        q = q.view(B, S, H, self.head_dim).transpose(1, 2)  # [B,H,S,hd]
        k = k.view(B, S, H, self.head_dim).transpose(1, 2)
        v = v.view(B, S, H, self.head_dim).transpose(1, 2)

        # Attention scores: Q @ K^T -> [B,H,S,S]
        kt = k.transpose(-2, -1)  # [B,H,hd,S]
        self.counter.add_matmul(q, kt)
        scores = torch.matmul(q, kt) / (self.head_dim ** 0.5)

        attn = F.softmax(scores, dim=-1)  # softmax cost ignored
        # Attn @ V -> [B,H,S,hd]
        self.counter.add_matmul(attn, v)
        out = torch.matmul(attn, v)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # output projection
        self.counter.add_linear(B, S, D, D)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: ToyConfig, counter: FlopsCounter):
        super().__init__()
        self.cfg = cfg
        self.counter = counter
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # fc1
        self.counter.add_linear(B, S, D, self.cfg.d_ff)
        x = self.fc1(x)
        x = F.gelu(x)  # activation cost ignored
        # fc2
        self.counter.add_linear(B, S, self.cfg.d_ff, D)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ToyConfig, counter: FlopsCounter):
        super().__init__()
        self.attn = SelfAttention(cfg, counter)
        self.ff = FeedForward(cfg, counter)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, gate_on: bool, counter: FlopsCounter) -> torch.Tensor:
        # If gate off: identity (still stable), we skip compute-heavy parts
        counter.mark_gate(gate_on)
        if gate_on:
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, cfg: ToyConfig, counter: FlopsCounter):
        super().__init__()
        self.cfg = cfg
        self.counter = counter
        self.blocks = nn.ModuleList([Block(cfg, counter) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, gate_mask: torch.Tensor) -> torch.Tensor:
        """
        gate_mask: [n_layers] boolean tensor.
        """
        for i, blk in enumerate(self.blocks):
            x = blk(x, bool(gate_mask[i].item()), self.counter)
        return self.ln_f(x)