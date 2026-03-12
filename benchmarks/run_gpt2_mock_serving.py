import csv
import time
from statistics import mean
import torch
from transformers import AutoTokenizer, AutoModel
from rte_engine import GPT2RTEGate, DriftExitPolicy

MODEL_NAME = "gpt2"
PROMPT = "Relational time engine improves transformer efficiency"
THRESHOLD = 1.0
MAX_NEW_TOKENS = 12
RUNS = 3


def argmax_next_token(last_hidden_state, model):
    last_token_hidden = last_hidden_state[:, -1, :]          # [B, D]
    logits = last_token_hidden @ model.wte.weight.T          # [B, V]
    next_token = torch.argmax(logits, dim=-1, keepdim=True) # [B, 1]
    return next_token


def baseline_decode_trace(model, input_ids, steps):
    generated = input_ids.clone()
    step_rows = []

    with torch.no_grad():
        for step_idx in range(steps):
            t0 = time.perf_counter()
            outputs = model(input_ids=generated)
            next_token = argmax_next_token(outputs.last_hidden_state, model)
            generated = torch.cat([generated, next_token], dim=1)
            t1 = time.perf_counter()

            step_rows.append({
                "step_idx": step_idx,
                "latency_ms": (t1 - t0) * 1000,
                "executed_layers": 12,      # GPT-2 small has 12 blocks
                "rho_layers": 1.0,
                "saving": 0.0,
                "drift_mean": 0.0,
                "sequence_len": int(generated.shape[1]),
            })

    return step_rows


def rte_decode_trace(model, input_ids, steps):
    generated = input_ids.clone()
    step_rows = []

    with torch.no_grad():
        for step_idx in range(steps):
            t0 = time.perf_counter()
            outputs, meta = model(input_ids=generated)
            next_token = argmax_next_token(outputs.last_hidden_state, model)
            generated = torch.cat([generated, next_token], dim=1)
            t1 = time.perf_counter()

            drift_mean = mean(meta["drifts"]) if meta["drifts"] else 0.0
            saving = 1.0 - meta["rho_layers"]

            step_rows.append({
                "step_idx": step_idx,
                "latency_ms": (t1 - t0) * 1000,
                "executed_layers": meta["executed_layers"],
                "rho_layers": meta["rho_layers"],
                "saving": saving,
                "drift_mean": drift_mean,
                "sequence_len": int(generated.shape[1]),
            })

    return step_rows


def aggregate_rows(rows):
    return {
        "latency_ms_mean": mean([r["latency_ms"] for r in rows]),
        "executed_layers_mean": mean([r["executed_layers"] for r in rows]),
        "rho_mean": mean([r["rho_layers"] for r in rows]),
        "saving_mean": mean([r["saving"] for r in rows]),
        "drift_mean": mean([r["drift_mean"] for r in rows]),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]

    all_baseline_steps = []
    all_rte_steps = []

    print("\n=== GPT-2 Mock Serving Benchmark ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Prompt: {PROMPT}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    for run_idx in range(RUNS):
        baseline_model = AutoModel.from_pretrained(MODEL_NAME)
        baseline_steps = baseline_decode_trace(
            baseline_model,
            input_ids=input_ids,
            steps=MAX_NEW_TOKENS,
        )

        policy = DriftExitPolicy(threshold=THRESHOLD, mode="mean_abs")
        rte_model = AutoModel.from_pretrained(MODEL_NAME)
        rte_model = GPT2RTEGate(policy=policy).wrap(rte_model)
        rte_steps = rte_decode_trace(
            rte_model,
            input_ids=input_ids,
            steps=MAX_NEW_TOKENS,
        )

        for row in baseline_steps:
            row["run_idx"] = run_idx
            row["mode"] = "baseline"
            all_baseline_steps.append(row)

        for row in rte_steps:
            row["run_idx"] = run_idx
            row["mode"] = "rte"
            all_rte_steps.append(row)

    baseline_summary = aggregate_rows(all_baseline_steps)
    rte_summary = aggregate_rows(all_rte_steps)
    speedup = baseline_summary["latency_ms_mean"] / rte_summary["latency_ms_mean"]

    print("Baseline summary:")
    print(
        f"  latency={baseline_summary['latency_ms_mean']:.3f} ms/step | "
        f"layers={baseline_summary['executed_layers_mean']:.2f} | "
        f"rho={baseline_summary['rho_mean']:.3f}"
    )

    print("RTE summary:")
    print(
        f"  latency={rte_summary['latency_ms_mean']:.3f} ms/step | "
        f"layers={rte_summary['executed_layers_mean']:.2f} | "
        f"rho={rte_summary['rho_mean']:.3f} | "
        f"saving={rte_summary['saving_mean']:.3f} | "
        f"drift={rte_summary['drift_mean']:.6f}"
    )

    print(f"\nSpeedup: {speedup:.3f}x")

    out_steps = "benchmarks/results/gpt2_mock_serving_steps.csv"
    out_summary = "benchmarks/results/gpt2_mock_serving_summary.csv"

    with open(out_steps, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_baseline_steps[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_baseline_steps + all_rte_steps)

    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "latency_ms_mean",
                "executed_layers_mean",
                "rho_mean",
                "saving_mean",
                "drift_mean",
            ]
        )
        writer.writeheader()
        writer.writerow({
            "mode": "baseline",
            **baseline_summary,
        })
        writer.writerow({
            "mode": "rte",
            **rte_summary,
        })

    print(f"\nSaved step trace -> {out_steps}")
    print(f"Saved summary    -> {out_summary}")


if __name__ == "__main__":
    main()
