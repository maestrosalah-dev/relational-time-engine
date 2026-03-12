
# Relational Time Engine (RTE)
### Runtime Density Regulation for Efficient Transformer Inference
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18644899.svg)](https://doi.org/10.5281/zenodo.18644899)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research-green)


Relational Time Engine (RTE) is a lightweight execution gating layer that reduces unnecessary transformer layer computation during inference.

Instead of executing all layers uniformly, RTE dynamically evaluates representational change and exits early when further computation becomes redundant.

The system operates as a runtime mechanism and **does not require model retraining or architecture modification.**

---

## Key Idea

Traditional transformer inference executes every layer regardless of whether meaningful representational change is still occurring.

RTE introduces a **relational activation density control**:

ρ = activated_layers / total_layers  

Saving = 1 − ρ

When representational drift falls below a structural threshold, the engine halts further layer execution.

---
## Research Context

RTE is part of a broader research direction exploring relational execution dynamics and runtime density regulation in adaptive AI systems.

The goal is to reduce unnecessary compute while preserving structural information flow in deep neural architectures.
---

## Benchmark Results (CPU)

Configuration:

- 8-layer transformer
- sequence length = 128
- batch size = 8
- multi-run averaged measurements

| Noise | Mode | Layers Used | Saving | Latency | Throughput |
|------|------|-------------|--------|--------|-----------|
|0.20|baseline|8/8|0%|199 ms|41 sps|
|0.20|strict|2/8|75%|146 ms|54 sps|
|0.50|strict|2/8|75%|153 ms|52 sps|
|0.80|strict|3/8|62%|162 ms|49 sps|

Observed effects:

• up to **75% layer reduction**  
• **~40% latency improvement**  
• increased throughput  
• bounded output drift

On distilbert-base-uncased, the RTE early-exit wrapper reduced mean CPU latency from 29.89 ms to 17.89 ms, achieving a 1.67x speedup while preserving output dimensionality and maintaining bounded structural drift.
---

## Architecture

Input
↓
Transformer Layer Li
↓
Representation Drift Measurement
↓
Threshold Gate
↓
Continue / Early Exit


The mechanism dynamically regulates execution depth according to signal stability.
Input
  ↓
Layer L1
  ↓
Representation Drift
  ↓
Threshold Gate
  ↓
Continue → L2
Exit → Output

---

## Repository Structure

rte/
benchmarks/
examples/
industrial/
spec/
tests/


Important files:

- `benchmarks/run_transformer_latency_benchmark.py`
- `industrial/RTE_Industrial_Whitepaper_v1.pdf`

---

## Quick Start

Clone repository:

git clone https://github.com/maestrosalah-dev/relational-time-engine.git
cd relational-time-engine


Install requirements:

pip install -r requirements.txt


Run benchmark:

python benchmarks/run_transformer_latency_benchmark.py

## DistilBERT Early-Exit Benchmark

Using `distilbert-base-uncased` on CPU, RTE achieved the following best result:

- Threshold: `0.25`
- Baseline latency: `27.64 ms`
- RTE latency: `12.94 ms`
- Speedup: `2.14x`
- Executed layers: `3 / 6`
- Saving: `50%`
- Drift (MAE proxy): `0.208`

This demonstrates that RTE can operate as a real runtime early-exit controller on an actual HuggingFace model.
## DistilBERT Early-Exit Benchmark

RTE was integrated as a runtime early-exit controller into `distilbert-base-uncased` and evaluated on CPU.

### Best observed result

- Threshold: `0.25`
- Baseline latency: `27.64 ms`
- RTE latency: `12.94 ms`
- Speedup: `2.14x`
- Executed layers: `3 / 6`
- Layer saving: `50%`
- Drift (MAE proxy): `0.208`

This result demonstrates that RTE can function as a real runtime gating mechanism on an actual HuggingFace model, not only on synthetic transformer benchmarks.
## GPT-2 Runtime Early-Exit Benchmark Suite

RTE was integrated into `gpt2` as a causal decoder early-exit controller and evaluated across multiple prompts and sequence lengths.

### Threshold summary

| Threshold | Baseline Latency | RTE Latency | Speedup | Saving | Drift |
|-----------|------------------|-------------|---------|--------|-------|
| 0.75 | 61.30 ms | 10.64 ms | 5.84x | 83.3% | 0.752 |
| 1.00 | 61.30 ms | 9.11 ms | 7.69x | 85.6% | 0.819 |
| 1.25 | 61.30 ms | 5.89 ms | 10.68x | 91.7% | 1.024 |

A practical operating point is `threshold = 1.00`, which provides strong acceleration while keeping drift lower than the more aggressive configuration.
## GPT-2 Generation-Step Benchmark

RTE was also evaluated in a multi-step autoregressive decoding setting using `gpt2`.

### Threshold summary

| Threshold | Baseline Latency/Step | RTE Latency/Step | Speedup | Saving | Drift |
|-----------|------------------------|------------------|---------|--------|-------|
| 0.75 | 67.74 ms | 17.90 ms | 3.84x | 83.3% | 0.803 |
| 1.00 | 64.26 ms | 17.92 ms | 3.61x | 83.3% | 0.803 |
| 1.25 | 49.68 ms | 10.63 ms | 4.69x | 91.7% | 1.081 |

A practical operating point is `threshold = 0.75`, which provides strong generation-time acceleration while remaining more conservative than the most aggressive configuration.
---

## Whitepaper

Industrial whitepaper:

industrial/RTE_Industrial_Whitepaper_v1.pdf

Zenodo archive:

https://doi.org/10.5281/zenodo.18644899
On distilbert-base-uncased, RTE achieved a best observed CPU speedup of 2.14x at threshold 0.25, reducing executed layers from 6 to 3 while maintaining bounded drift (MAE ≈ 0.208).
## DistilBERT Runtime Integration

To validate RTE beyond synthetic transformer benchmarks, we integrated the gating mechanism into `distilbert-base-uncased` and applied runtime early-exit based on representational drift thresholds.

In the best observed CPU configuration:

- baseline latency = `27.64 ms`
- gated latency = `12.94 ms`
- speedup = `2.14x`
- executed layers reduced from `6` to `3`
- effective layer saving = `50%`
- bounded drift proxy (MAE) = `0.208`

This experiment demonstrates that RTE can operate as a practical runtime controller inside a real HuggingFace transformer architecture, achieving measurable latency reduction without retraining or architectural redesign.

## GPT-2 Runtime Integration

To evaluate RTE on a causal decoder architecture, we integrated the gating mechanism into `gpt2` and ran a multi-prompt, multi-length benchmark suite.

Across the benchmark suite, RTE achieved the following threshold-level averages:

- threshold `0.75`: `5.84x` speedup with `83.3%` layer saving
- threshold `1.00`: `7.69x` speedup with `85.6%` layer saving
- threshold `1.25`: `10.68x` speedup with `91.7%` layer saving

A balanced operating point was observed at `threshold = 1.00`, where RTE maintained strong acceleration while avoiding the higher drift of the most aggressive setting.

These results suggest that runtime layer gating can extend beyond encoder models and operate effectively on decoder-style transformer architectures relevant to LLM serving.
## GPT-2 Autoregressive Generation Benchmark

To move closer to realistic LLM serving conditions, RTE was evaluated in a multi-step autoregressive decoding scenario using `gpt2`.

Across generation steps, RTE maintained substantial acceleration:

- threshold `0.75`: `3.84x` speedup with `83.3%` saving
- threshold `1.00`: `3.61x` speedup with `83.3%` saving
- threshold `1.25`: `4.69x` speedup with `91.7%` saving

These results suggest that runtime drift-based early exit can remain effective beyond single-pass inference and may extend to token-by-token decoding workloads relevant to LLM serving systems.
---

## Citation

If you use or reference this work:

Salah, A. (2026).  
Relational Time Engine (RTE): Runtime Density Regulation for Efficient AI Inference.

---

## Author

Athmani Salah  
Independent Researcher
ORCID: 0009-0004-9350-9216  
maestro.salah@gmail.com
GitHub  
https://github.com/maestrosalah-dev


For collaboration or hardware integration inquiries, open an issue or contact via GitHub.






