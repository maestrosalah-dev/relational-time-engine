# benchmarks/flops_counter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn


@dataclass
class FlopsReport:
    flops_total: int
    flops_by_op: Dict[str, int]
    calls: Dict[str, int]
    tokens: int
    gated_on: int
    gated_off: int


class FlopsCounter:
    """
    Runtime FLOPs counter for:
    - Linear layers (counts multiply-add as 2 FLOPs)
    - Attention matmuls (QK^T and AV)
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.flops_by_op: Dict[str, int] = {"linear": 0, "matmul": 0}
        self.calls: Dict[str, int] = {"linear": 0, "matmul": 0}
        self.tokens: int = 0
        self.gated_on: int = 0
        self.gated_off: int = 0

    @staticmethod
    def _linear_flops(batch: int, seq: int, in_f: int, out_f: int) -> int:
        # y = xW + b  -> batch*seq*(in_f*out_f mul + in_f*out_f add) ~= 2*B*S*in*out
        return int(2 * batch * seq * in_f * out_f)

    @staticmethod
    def _matmul_flops(a: torch.Tensor, b: torch.Tensor) -> int:
        # For (..., m, k) x (..., k, n) => (..., m, n)
        # FLOPs ~ 2*m*n*k per matmul (mul+add)
        if a.ndim < 2 or b.ndim < 2:
            return 0
        m = a.shape[-2]
        k = a.shape[-1]
        n = b.shape[-1]
        # include batch-like dims product
        batch_dims = 1
        for d in a.shape[:-2]:
            batch_dims *= int(d)
        return int(2 * batch_dims * m * n * k)

    def add_linear(self, batch: int, seq: int, in_f: int, out_f: int) -> None:
        self.flops_by_op["linear"] += self._linear_flops(batch, seq, in_f, out_f)
        self.calls["linear"] += 1

    def add_matmul(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self.flops_by_op["matmul"] += self._matmul_flops(a, b)
        self.calls["matmul"] += 1

    def add_tokens(self, n: int) -> None:
        self.tokens += int(n)

    def mark_gate(self, on: bool) -> None:
        if on:
            self.gated_on += 1
        else:
            self.gated_off += 1

    def report(self) -> FlopsReport:
        total = int(self.flops_by_op["linear"] + self.flops_by_op["matmul"])
        return FlopsReport(
            flops_total=total,
            flops_by_op=dict(self.flops_by_op),
            calls=dict(self.calls),
            tokens=self.tokens,
            gated_on=self.gated_on,
            gated_off=self.gated_off,
        )