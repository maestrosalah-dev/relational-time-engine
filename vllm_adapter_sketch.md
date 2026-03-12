# RTE vLLM Adapter Sketch

## Goal

Inject RTE-style drift-based early exit into a decoder serving runtime such as vLLM.

---

## Baseline Decoder Loop

A simplified decoder serving loop typically follows:

```python
hidden = embed(input_ids)

for block in decoder_blocks:
    hidden = block(hidden, kv_cache=kv_cache, attention_mask=mask)

logits = lm_head(hidden)
