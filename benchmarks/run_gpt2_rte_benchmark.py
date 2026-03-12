import csv
import time
from transformers import AutoTokenizer, AutoModel
from rte_engine.gpt2_gate import GPT2RTEGate

MODEL_NAME = "gpt2"
TEXT = "Relational time engine improves transformer efficiency."
RUNS = 10

THRESHOLDS = [0.40, 0.50, 0.60, 0.75, 1.00, 1.50]


def measure_baseline(model, inputs, runs=10):
    times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        outputs = model(**inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "latency_ms_mean": sum(times) / len(times),
        "output_shape": tuple(outputs.last_hidden_state.shape),
    }


def measure_rte(model, inputs, runs=10):
    times = []
    executed_layers_list = []
    rho_list = []
    last_drift_list = []

    for _ in range(runs):
        t0 = time.perf_counter()
        outputs, meta = model(**inputs)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)
        executed_layers_list.append(meta["executed_layers"])
        rho_list.append(meta["rho_layers"])
        last_drift_list.append(meta["drifts"][-1] if meta["drifts"] else 0.0)

    return {
        "latency_ms_mean": sum(times) / len(times),
        "executed_layers_mean": sum(executed_layers_list) / len(executed_layers_list),
        "rho_mean": sum(rho_list) / len(rho_list),
        "last_drift_mean": sum(last_drift_list) / len(last_drift_list),
        "output_shape": tuple(outputs.last_hidden_state.shape),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(TEXT, return_tensors="pt")

    baseline_model = AutoModel.from_pretrained(MODEL_NAME)
    baseline = measure_baseline(baseline_model, inputs, runs=RUNS)

    print("\n=== GPT-2 RTE Threshold Sweep ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Baseline latency: {baseline['latency_ms_mean']:.3f} ms")
    print(f"Output shape: {baseline['output_shape']}")
    print()

    rows = []

    for threshold in THRESHOLDS:
        rte_model = AutoModel.from_pretrained(MODEL_NAME)
        rte_model = GPT2RTEGate(threshold=threshold).wrap(rte_model)

        rte = measure_rte(rte_model, inputs, runs=RUNS)

        speedup = baseline["latency_ms_mean"] / rte["latency_ms_mean"]
        saving = 1.0 - rte["rho_mean"]

        row = {
            "threshold": threshold,
            "baseline_latency_ms": baseline["latency_ms_mean"],
            "rte_latency_ms": rte["latency_ms_mean"],
            "speedup_factor": speedup,
            "executed_layers_mean": rte["executed_layers_mean"],
            "rho_mean": rte["rho_mean"],
            "saving_mean": saving,
            "last_drift_mean": rte["last_drift_mean"],
            "output_shape": str(rte["output_shape"]),
        }
        rows.append(row)

        print(
            f"threshold={threshold:.2f} | "
            f"latency={rte['latency_ms_mean']:.3f} ms | "
            f"layers={rte['executed_layers_mean']:.2f} | "
            f"rho={rte['rho_mean']:.3f} | "
            f"saving={saving:.3f} | "
            f"speedup={speedup:.3f}x | "
            f"last_drift={rte['last_drift_mean']:.6f}"
        )

    out_path = "benchmarks/results/gpt2_rte_threshold_sweep.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {out_path}")

    candidates = [r for r in rows if r["last_drift_mean"] <= 1.0]
    if candidates:
        best = max(candidates, key=lambda x: x["speedup_factor"])
        print("\n=== Best Candidate (last_drift <= 1.0) ===")
        print(best)
    else:
        print("\nNo candidate satisfied the drift constraint.")


if __name__ == "__main__":
    main()
