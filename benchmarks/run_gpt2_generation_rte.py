import csv
import time
import torch
from transformers import AutoTokenizer, AutoModel
from rte_engine import GPT2RTEGate, DriftExitPolicy

MODEL_NAME = "gpt2"
PROMPT = "Relational time engine improves transformer efficiency"
MAX_NEW_TOKENS = 12
RUNS = 5


def argmax_next_token(last_hidden_state, model):
    """
    Minimal next-token proxy:
    project hidden state back to vocab using token embedding matrix.
    This is not full LM-head generation, but enough for runtime step benchmarking.
    """
    last_token_hidden = last_hidden_state[:, -1, :]          # [B, D]
    logits = last_token_hidden @ model.wte.weight.T          # [B, V]
    next_token = torch.argmax(logits, dim=-1, keepdim=True) # [B, 1]
    return next_token


def baseline_generate_stepwise(model, input_ids, steps=12):
    times = []
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(steps):
            t0 = time.perf_counter()
            outputs = model(input_ids=generated)
            next_token = argmax_next_token(outputs.last_hidden_state, model)
            generated = torch.cat([generated, next_token], dim=1)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return {
        "mean_latency_ms": sum(times) / len(times),
        "generated_ids": generated,
    }


def rte_generate_stepwise(model, input_ids, steps=12):
    times = []
    generated = input_ids.clone()
    executed_layers = []
    rhos = []
    drifts = []

    with torch.no_grad():
        for _ in range(steps):
            t0 = time.perf_counter()
            outputs, meta = model(input_ids=generated)
            next_token = argmax_next_token(outputs.last_hidden_state, model)
            generated = torch.cat([generated, next_token], dim=1)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

            executed_layers.append(meta["executed_layers"])
            rhos.append(meta["rho_layers"])
            drifts.append(sum(meta["drifts"]) / len(meta["drifts"]) if meta["drifts"] else 0.0)

    return {
        "mean_latency_ms": sum(times) / len(times),
        "executed_layers_mean": sum(executed_layers) / len(executed_layers),
        "rho_mean": sum(rhos) / len(rhos),
        "drift_mean": sum(drifts) / len(drifts),
        "generated_ids": generated,
    }


def run_once(threshold):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]

    baseline_model = AutoModel.from_pretrained(MODEL_NAME)
    baseline = baseline_generate_stepwise(
        baseline_model,
        input_ids=input_ids,
        steps=MAX_NEW_TOKENS,
    )

    rte_model = AutoModel.from_pretrained(MODEL_NAME)
    policy = DriftExitPolicy(threshold=threshold, mode="mean_abs")
    rte_model = GPT2RTEGate(policy=policy).wrap(rte_model)

    rte = rte_generate_stepwise(
        rte_model,
        input_ids=input_ids,
        steps=MAX_NEW_TOKENS,
    )

    speedup = baseline["mean_latency_ms"] / rte["mean_latency_ms"]
    saving = 1.0 - rte["rho_mean"]

    return {
        "threshold": threshold,
        "baseline_latency_ms": baseline["mean_latency_ms"],
        "rte_latency_ms": rte["mean_latency_ms"],
        "speedup_factor": speedup,
        "executed_layers_mean": rte["executed_layers_mean"],
        "rho_mean": rte["rho_mean"],
        "saving_mean": saving,
        "drift_mean": rte["drift_mean"],
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(rte["generated_ids"].shape[1] - input_ids.shape[1]),
    }


def main():
    thresholds = [0.75, 1.0, 1.25]
    rows = []

    print("\n=== GPT-2 Generation-Step RTE Benchmark ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    for thr in thresholds:
        run_rows = []
        for _ in range(RUNS):
            run_rows.append(run_once(thr))

        row = {
            "threshold": thr,
            "baseline_latency_ms": sum(r["baseline_latency_ms"] for r in run_rows) / len(run_rows),
            "rte_latency_ms": sum(r["rte_latency_ms"] for r in run_rows) / len(run_rows),
            "speedup_factor": sum(r["speedup_factor"] for r in run_rows) / len(run_rows),
            "executed_layers_mean": sum(r["executed_layers_mean"] for r in run_rows) / len(run_rows),
            "rho_mean": sum(r["rho_mean"] for r in run_rows) / len(run_rows),
            "saving_mean": sum(r["saving_mean"] for r in run_rows) / len(run_rows),
            "drift_mean": sum(r["drift_mean"] for r in run_rows) / len(run_rows),
            "prompt_tokens": run_rows[0]["prompt_tokens"],
            "generated_tokens": run_rows[0]["generated_tokens"],
        }
        rows.append(row)

        print(
            f"thr={thr:.2f} | "
            f"baseline={row['baseline_latency_ms']:.3f} ms/step | "
            f"rte={row['rte_latency_ms']:.3f} ms/step | "
            f"speedup={row['speedup_factor']:.3f}x | "
            f"layers={row['executed_layers_mean']:.2f} | "
            f"saving={row['saving_mean']:.3f} | "
            f"drift={row['drift_mean']:.6f}"
        )

    out_path = "benchmarks/results/gpt2_generation_rte.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {out_path}")


if __name__ == "__main__":
    main()
