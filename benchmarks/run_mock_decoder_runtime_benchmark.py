import csv
import time
from statistics import mean

from transformers import AutoTokenizer, AutoModel
from rte_engine import DriftExitPolicy, MockDecoderRuntime


MODEL_NAME = "gpt2"
PROMPT = "Relational time engine improves transformer efficiency"
MAX_NEW_TOKENS = 12
RUNS = 5


def argmax_next_token(last_hidden_state, model):
    last_token_hidden = last_hidden_state[:, -1, :]
    logits = last_token_hidden @ model.wte.weight.T
    next_token = logits.argmax(dim=-1, keepdim=True)
    return next_token


def baseline_decode(model, input_ids, steps):
    generated = input_ids.clone()
    step_latencies = []

    with __import__("torch").no_grad():
        for _ in range(steps):
            t0 = time.perf_counter()
            outputs = model(input_ids=generated)
            next_token = argmax_next_token(outputs.last_hidden_state, model)
            generated = __import__("torch").cat([generated, next_token], dim=1)
            t1 = time.perf_counter()
            step_latencies.append((t1 - t0) * 1000)

    return {
        "latency_ms_mean": mean(step_latencies),
        "generated_shape": tuple(generated.shape),
    }


def rte_decode(runtime, input_ids, steps):
    t0 = time.perf_counter()
    result = runtime.decode(input_ids, max_new_tokens=steps)
    t1 = time.perf_counter()

    summary = result["summary"]
    total_ms = (t1 - t0) * 1000
    per_step_ms = total_ms / steps

    return {
        "latency_ms_mean": per_step_ms,
        "executed_layers_mean": summary["executed_layers_mean"],
        "rho_mean": summary["rho_mean"],
        "saving_mean": summary["saving_mean"],
        "drift_mean": summary["drift_mean"],
        "last_drift_mean": summary["last_drift_mean"],
        "generated_shape": tuple(result["generated_ids"].shape),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]

    thresholds = [0.75, 1.0, 1.25]
    rows = []

    print("\n=== Mock Decoder Runtime Benchmark ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    for threshold in thresholds:
        baseline_runs = []
        rte_runs = []

        for _ in range(RUNS):
            baseline_model = AutoModel.from_pretrained(MODEL_NAME)
            baseline_runs.append(
                baseline_decode(baseline_model, input_ids=input_ids, steps=MAX_NEW_TOKENS)
            )

            rte_model = AutoModel.from_pretrained(MODEL_NAME)
            policy = DriftExitPolicy(threshold=threshold, mode="mean_abs")
            runtime = MockDecoderRuntime(model=rte_model, policy=policy)
            rte_runs.append(
                rte_decode(runtime, input_ids=input_ids, steps=MAX_NEW_TOKENS)
            )

        baseline_latency = mean(r["latency_ms_mean"] for r in baseline_runs)
        rte_latency = mean(r["latency_ms_mean"] for r in rte_runs)
        speedup = baseline_latency / rte_latency

        row = {
            "threshold": threshold,
            "baseline_latency_ms": baseline_latency,
            "rte_latency_ms": rte_latency,
            "speedup_factor": speedup,
            "executed_layers_mean": mean(r["executed_layers_mean"] for r in rte_runs),
            "rho_mean": mean(r["rho_mean"] for r in rte_runs),
            "saving_mean": mean(r["saving_mean"] for r in rte_runs),
            "drift_mean": mean(r["drift_mean"] for r in rte_runs),
            "last_drift_mean": mean(r["last_drift_mean"] for r in rte_runs),
        }
        rows.append(row)

        print(
            f"thr={threshold:.2f} | "
            f"baseline={baseline_latency:.3f} ms/step | "
            f"rte={rte_latency:.3f} ms/step | "
            f"speedup={speedup:.3f}x | "
            f"layers={row['executed_layers_mean']:.2f} | "
            f"saving={row['saving_mean']:.3f} | "
            f"drift={row['drift_mean']:.6f}"
        )

    out_path = "benchmarks/results/mock_decoder_runtime_benchmark.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {out_path}")
