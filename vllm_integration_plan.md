# RTE vLLM Integration Plan

## Objective

Integrate Relational Time Engine (RTE) into a decoder-style LLM serving runtime as a block-level early-exit policy.

The goal is to reduce executed decoder blocks during token generation while maintaining bounded structural drift.

---

## Current Evidence

RTE has already been validated in three stages:

### 1. Synthetic / Toy Transformer Benchmarks
- runtime density regulation
- FLOPs proxy reduction
- CPU latency savings

### 2. Real HuggingFace Model Integration
- DistilBERT early-exit wrapper
- GPT-2 single-pass early-exit wrapper

### 3. GPT-2 Generation-Step Benchmark
- multi-step autoregressive decoding
- substantial latency reduction per step
- layer-saving behavior under threshold control

This provides proof-of-feasibility across both encoder and decoder transformer families.

---

## Why vLLM Is the Right Target

vLLM is relevant because it provides:

- practical decoder serving runtime
- token-by-token generation execution
- cache-aware inference
- serving-oriented batching and scheduling

If RTE can be attached to a vLLM-like execution path, it moves from research prototype to serving infrastructure candidate.

---

## Core Integration Concept

RTE does not need to replace the model.

It acts as a runtime decision layer:

1. observe hidden-state evolution across decoder blocks
2. compute drift after each block
3. decide whether remaining blocks are necessary
4. exit early when structural change falls below threshold

---

## Key Technical Challenges

### 1. KV Cache Compatibility
A decoder runtime such as vLLM depends heavily on KV cache reuse.

RTE must not break:
- cache indexing
- block consistency
- token-step semantics

### 2. Layer Exit Semantics
Skipping later blocks must preserve:
- hidden-state dimensional validity
- output compatibility
- stable generation behavior

### 3. Runtime Overhead
The policy evaluation itself must remain negligible relative to the saved block execution.

### 4. Drift Calibration
Thresholds may need to vary by:
- model family
- prompt length
- generation step
- serving regime

---

## Proposed Integration Path

### Phase 1 — Policy Abstraction
Completed:
- reusable `DriftExitPolicy`
- reusable encoder/decoder wrappers
- generation-step benchmark on GPT-2

### Phase 2 — Mock Serving Runtime
Next:
- build a decoder-step loop abstraction
- evaluate executed blocks/token
- log drift trajectories across generation steps

### Phase 3 — vLLM Touchpoint Mapping
Investigate:
- decoder block loop entry points
- scheduler interaction constraints
- cache preservation boundaries
- optional hook/plugin insertion

### Phase 4 — Minimal Adapter Prototype
Goal:
- attach a no-op / diagnostic policy first
- then attach drift-based exit policy
- measure latency overhead vs saved compute

---

## Metrics Required Before True vLLM Integration

We need stable reporting for:

- latency per token
- executed layers per token
- mean rho
- drift statistics
- generated token consistency
- overhead of policy evaluation

---

## Recommended Immediate Next Step

Build a GPT-2 mock-serving benchmark that records per-token:
- executed layers
- drift
- cumulative speedup
- sequence growth behavior

After that, produce a first adapter prototype for a decoder runtime boundary.

---

## Strategic Value

If successful, RTE becomes more than a benchmark artifact.

It becomes a candidate runtime control mechanism for:
- efficient LLM serving
- adaptive decoder computation
- inference-cost reduction
- hardware-aware scheduling
---

## Immediate Next Step

Implement a GPT-2 generation-step benchmark using the current RTE policy abstraction before attempting direct vLLM integration.
