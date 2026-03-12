# RTE vLLM Integration Plan

## Objective

Integrate Relational Time Engine (RTE) as a runtime early-exit policy layer for decoder-based transformer inference in vLLM-style serving pipelines.

---

## Current Status

RTE currently supports:

- DistilBERT runtime early-exit
- GPT-2 runtime early-exit
- Drift-based exit policy abstraction (`DriftExitPolicy`)
- CPU benchmark suites with latency / saving / drift outputs

This establishes feasibility across both encoder and causal decoder transformer families.

---

## Why vLLM Matters

vLLM is a widely used inference runtime for large language models.

Integrating RTE into a vLLM-like environment would allow:

- dynamic reduction of executed decoder blocks
- lower latency
- reduced GPU time per request
- possible serving cost reduction

---

## Integration Challenges

### 1. Decoder Loop Awareness
vLLM performs token-by-token decoding with KV cache reuse.

RTE must be adapted to:
- work per decoding step
- avoid breaking KV cache consistency
- preserve causal generation semantics

### 2. Block-Level Exit Semantics
In encoder models, exiting early is simpler.

In causal decoder models, early exit must ensure:
- valid hidden-state propagation
- consistent block skipping
- acceptable generation drift

### 3. Serving Runtime Constraints
vLLM optimizes:
- batching
- scheduling
- cache management
- memory locality

RTE must integrate with minimal overhead.

---

## Proposed Integration Path

### Phase 1 — Policy Isolation
Completed:
- abstract drift policy
- reusable wrapper interfaces

### Phase 2 — Decoder Runtime Prototype
Next:
- test RTE on GPT-2 generation-style forward passes
- benchmark across prompt lengths and thresholds

### Phase 3 — Minimal vLLM-style Mock Runtime
Build a mock decoder loop with:
- iterative token steps
- KV-like hidden state reuse
- RTE block exit control

### Phase 4 — vLLM Adapter Exploration
Investigate integration points in:
- decoder block execution loop
- scheduling layer
- optional plugin-style policy hooks

---

## Required Metrics

For a valid vLLM-facing evaluation, we need:

- tokens/sec
- latency/request
- executed blocks/token
- drift / generation deviation
- cache correctness
- overhead of policy evaluation

---

## Immediate Next Step

Implement a GPT-2 generation-step benchmark using the current RTE policy abstraction before attempting direct vLLM integration.
