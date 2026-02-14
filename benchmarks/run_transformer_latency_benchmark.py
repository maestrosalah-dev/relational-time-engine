import csv
import os
import time
import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Config
# -----------------------------
@dataclass
class BenchConfig:
    seed: int = 42
    device: str = "cpu"

    batch_size: int = 8
    seq_len: int = 128
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 4
    ff_mult: int = 4
    n_layers: int = 8

    # timing
    warmup_iters: int = 10
    timed_iters: int = 50

    # repeat whole measurement to get stable mean/std
    runs: int = 10

    # Early-exit policy mapping noise -> rho (engineering baseline)
    strict_base_rho: float = 0.18
    strict_noise_gain: float = 0.22

    flex_base_rho: float = 0.25
    flex_noise_gain: float = 0.18

    rho_min: float = 0.10
    rho_max: float = 1.00

    # Threads (optional stabilization on CPU)
    set_threads: bool = True
    threads: int | None = None          # if None -> half cores
    interop_threads: int = 1


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def choose_rho(noise: float, mode: str, cfg: BenchConfig) -> float:
    if mode == "baseline":
        return 1.0
    if mode == "strict":
        rho = cfg.strict_base_rho + cfg.strict_noise_gain * noise
        return clip(rho, cfg.rho_min, cfg.rho_max)
    if mode == "flex":
        rho = cfg.flex_base_rho + cfg.flex_noise_gain * noise
        return clip(rho, cfg.rho_min, cfg.rho_max)
    raise ValueError(f"Unknown mode: {mode}")


def choose_exit_layers(noise: float, mode: str, cfg: BenchConfig) -> int:
    rho = choose_rho(noise, mode, cfg)
    layers = max(1, int(round(cfg.n_layers * rho)))
    return min(cfg.n_layers, layers)


# -----------------------------
# Toy Transformer with Early Exit
# -----------------------------
class ToyTransformer(nn.Module):
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Important: create distinct layers (no shared references)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * cfg.ff_mult,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    @torch.no_grad()
    def forward_baseline(self, x_tokens: torch.Tensor) -> torch.Tensor:
        h = self.emb(x_tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def forward_early_exit(self, x_tokens: torch.Tensor, exec_layers: int) -> torch.Tensor:
        h = self.emb(x_tokens)
        for i in range(exec_layers):
            h = self.layers[i](h)
        h = self.norm(h)
        return self.head(h)


# -----------------------------
# Measurement helpers
# -----------------------------
def time_forward(fn, warmup: int, iters: int) -> float:
    # Warmup
    for _ in range(warmup):
        _ = fn()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


@torch.no_grad()
def output_drift_mae(baseline_logits: torch.Tensor, gated_logits: torch.Tensor) -> float:
    # MAE over logits as a simple quality proxy
    # (You can replace later with KL divergence or task accuracy.)
    return (baseline_logits - gated_logits).abs().mean().item()


def main():
    cfg = BenchConfig()
    torch.manual_seed(cfg.seed)

    # Stabilize CPU threads (optional but recommended for reproducibility)
    if cfg.set_threads:
        try:
            threads = cfg.threads or max(1, (os.cpu_count() or 4) // 2)
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(cfg.interop_threads)
        except Exception:
            pass

    device = torch.device(cfg.device)
    model = ToyTransformer(cfg).to(device)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)

    noise_levels = [0.2, 0.5, 0.8]
    modes = ["baseline", "strict", "flex"]

    out_dir = os.path.join("benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "transformer_latency_cpu_raw.csv")
    summary_path = os.path.join(out_dir, "transformer_latency_cpu_summary.csv")

    print("\n=== Transformer CPU Latency Benchmark (Early-Exit, multi-run) ===")
    print(f"Python: {os.sys.version.split()[0]} | Torch: {torch.__version__} | Device: {device}")
    try:
        print(f"CPU cores: {os.cpu_count()} | torch threads: {torch.get_num_threads()} | interop: {torch.get_num_interop_threads()}")
    except Exception:
        pass
    print(f"Config: layers={cfg.n_layers}, batch={cfg.batch_size}, seq={cfg.seq_len}, d={cfg.d_model}, heads={cfg.n_heads}")
    print(f"Runs={cfg.runs}, warmup={cfg.warmup_iters}, iters={cfg.timed_iters}")
    print(f"Saved raw -> {raw_path}")
    print(f"Saved summary -> {summary_path}")

    raw_rows: List[Dict] = []

    # Run baseline logits once per noise per run to compute drift consistently
    for noise in noise_levels:
        for mode in modes:
            exec_layers = choose_exit_layers(noise, mode, cfg)
            rho_layers = exec_layers / cfg.n_layers
            saving = 1.0 - rho_layers

            latencies_ms: List[float] = []
            throughputs: List[float] = []
            drifts: List[float] = []

            for run_idx in range(cfg.runs):
                # For drift: compute baseline output for this run (same input) to compare against gated.
                # This keeps drift measurement aligned to current model state (which is constant in eval).
                base_logits = model.forward_baseline(x)

                if mode == "baseline":
                    fn = lambda: model.forward_baseline(x)
                else:
                    fn = lambda: model.forward_early_exit(x, exec_layers)

                mean_s = time_forward(fn, cfg.warmup_iters, cfg.timed_iters)
                latency_ms = mean_s * 1000.0
                throughput_sps = cfg.batch_size / mean_s

                # Drift: compare baseline vs gated outputs for one forward
                if mode == "baseline":
                    drift = 0.0
                else:
                    gated_logits = model.forward_early_exit(x, exec_layers)
                    drift = output_drift_mae(base_logits, gated_logits)

                latencies_ms.append(latency_ms)
                throughputs.append(throughput_sps)
                drifts.append(drift)

                raw_rows.append(dict(
                    noise_level=noise,
                    mode=mode,
                    run=run_idx,
                    exec_layers=exec_layers,
                    n_layers=cfg.n_layers,
                    rho_layers=rho_layers,
                    saving_vs_baseline=saving,
                    latency_ms=latency_ms,
                    throughput_samples_per_s=throughput_sps,
                    drift_mae=drift,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    timed_iters=cfg.timed_iters,
                    warmup_iters=cfg.warmup_iters,
                    seed=cfg.seed,
                    torch_version=torch.__version__,
                    device=str(device),
                    threads=getattr(torch, "get_num_threads", lambda: None)(),
                ))

            # Summary per (noise, mode)
            lat_mean = mean(latencies_ms)
            lat_std = pstdev(latencies_ms)  # population std
            thr_mean = mean(throughputs)
            thr_std = pstdev(throughputs)
            drift_mean = mean(drifts)
            drift_std = pstdev(drifts)

            print(
                f"noise={noise:.2f} mode={mode:<8} layers={exec_layers}/{cfg.n_layers} "
                f"rho={rho_layers:.3f} saving={saving:.3f} "
                f"lat_ms={lat_mean:.2f}±{lat_std:.2f} thr={thr_mean:.2f}±{thr_std:.2f} "
                f"drift_mae={drift_mean:.6f}±{drift_std:.6f}"
            )

    # Write raw CSV
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)

    # Build summary CSV
    # group rows by (noise, mode)
    groups: Dict[Tuple[float, str], List[Dict]] = {}
    for r in raw_rows:
        key = (float(r["noise_level"]), r["mode"])
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict] = []
    for (noise, mode), rs in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        lat = [float(x["latency_ms"]) for x in rs]
        thr = [float(x["throughput_samples_per_s"]) for x in rs]
        drift = [float(x["drift_mae"]) for x in rs]

        summary_rows.append(dict(
            noise_level=noise,
            mode=mode,
            exec_layers=int(rs[0]["exec_layers"]),
            n_layers=int(rs[0]["n_layers"]),
            rho_layers=float(rs[0]["rho_layers"]),
            saving_vs_baseline=float(rs[0]["saving_vs_baseline"]),
            latency_ms_mean=mean(lat),
            latency_ms_std=pstdev(lat),
            throughput_samples_per_s_mean=mean(thr),
            throughput_samples_per_s_std=pstdev(thr),
            drift_mae_mean=mean(drift),
            drift_mae_std=pstdev(drift),
            runs=cfg.runs,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            torch_version=torch.__version__,
            device=str(device),
            threads=getattr(torch, "get_num_threads", lambda: None)(),
        ))

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\nDone ✅")


if __name__ == "__main__":
    main()