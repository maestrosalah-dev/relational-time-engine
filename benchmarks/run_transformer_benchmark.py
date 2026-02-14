# benchmarks/run_transformer_benchmark.py
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
import random
import torch

from benchmarks.flops_counter import FlopsCounter
from benchmarks.transformer_toy import ToyConfig, ToyTransformer


def make_gate_mask(n_layers: int, mode: str, noise: float) -> torch.Tensor:
    """
    Simple gating policy for benchmark ONLY:
    - baseline: all True
    - strict: lower activation (more compute saving)
    - flex: higher activation (safer)
    noise increases activation probability slightly (harder environment)
    """
    if mode == "baseline":
        return torch.ones(n_layers, dtype=torch.bool)

    # base activation probabilities
    if mode == "strict":
        p0 = 0.12
    elif mode == "flex":
        p0 = 0.18
    else:
        raise ValueError("mode must be baseline|strict|flex")

    # noise makes system slightly more active
    p = min(0.95, p0 + 0.10 * noise)

    mask = [random.random() < p for _ in range(n_layers)]
    return torch.tensor(mask, dtype=torch.bool)


def run_once(cfg: ToyConfig, mode: str, noise: float, batches: int, batch_size: int, seed: int) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    counter = FlopsCounter()
    model = ToyTransformer(cfg, counter).eval()

    # random input (no training)
    for _ in range(batches):
        x = torch.randn(batch_size, cfg.seq_len, cfg.d_model)
        gate = make_gate_mask(cfg.n_layers, mode, noise)
        _ = model(x, gate)

    rep = counter.report()
    # saving relative to always-on baseline is computed outside
    return {
        "mode": mode,
        "noise": noise,
        "batches": batches,
        "batch_size": batch_size,
        "seq_len": cfg.seq_len,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "flops_total": rep.flops_total,
        "flops_linear": rep.flops_by_op["linear"],
        "flops_matmul": rep.flops_by_op["matmul"],
        "calls_linear": rep.calls["linear"],
        "calls_matmul": rep.calls["matmul"],
        "tokens": rep.tokens,
        "gated_on": rep.gated_on,
        "gated_off": rep.gated_off,
        "rho_layers": (rep.gated_on / max(rep.gated_on + rep.gated_off, 1)),
    }


def main() -> None:
    cfg = ToyConfig(
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=4,
        seq_len=128,
    )

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # choose a few noise regimes
    noises = [0.2, 0.5, 0.8]
    modes = ["baseline", "strict", "flex"]

    for noise in noises:
        for mode in modes:
            rows.append(run_once(cfg, mode, noise, batches=50, batch_size=8, seed=123))

    # compute savings vs baseline per noise
    by_noise = {}
    for r in rows:
        by_noise.setdefault(r["noise"], {})[r["mode"]] = r["flops_total"]

    for r in rows:
        base = by_noise[r["noise"]]["baseline"]
        r["saving_vs_baseline"] = 1.0 - (r["flops_total"] / max(base, 1))

    out_csv = out_dir / "transformer_toy_flops.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Transformer Toy FLOPs Benchmark ===")
    print(f"Saved -> {out_csv}")
    print("noise | mode     | rho_layers | saving_vs_baseline | flops_total")
    print("---------------------------------------------------------------")
    for r in rows:
        print(f"{r['noise']:.2f} | {r['mode']:<8} | {r['rho_layers']:.3f}     | {r['saving_vs_baseline']:.3f}             | {r['flops_total']}")


if __name__ == "__main__":
    main()