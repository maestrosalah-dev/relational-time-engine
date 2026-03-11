import csv
import time
import torch
from transformers import AutoTokenizer, AutoModel
from rte_engine import DistilBERTRTEGate


MODEL_NAME = "distilbert-base-uncased"
TEXT = "Relational time engine for efficient transformer inference."
RUNS = 20

THRESHOLDS = [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]


def measure_baseline(model, inputs, runs=20):
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            outputs = model(**inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return {
        "latency_ms_mean": sum(times) / len(times),
        "last_hidden_shape": tuple(outputs.last_hidden_state.shape),
    }


def measure_rte(model, inputs, runs=20):
    model.eval()
    times = []
    executed_layers_list = []
    rho_list = []
    drift_list = []

    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            outputs, meta = model(**inputs)
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)
            executed_layers_list.append(meta["executed_layers"])
            rho_list.append(meta["rho_layers"])
            drift_list.append(meta["drifts"][-1] if meta["drifts"] else 0.0)

    return {
        "latency_ms_mean": sum(times) / len(times),
        "executed_layers_mean": sum(executed_layers_list) / len(executed_layers_list),
        "rho_mean": sum(rho_list) / len(rho_list),
        "drift_last_mean": sum(drift_list) / len(drift_list),
        "last_hidden_shape": tuple(outputs.last_hidden_state.shape),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(TEXT, return_tensors="pt")

    # Baseline مرة واحدة
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    baseline = measure_baseline(base_model, inputs, runs=RUNS)

    print("\n=== DistilBERT RTE Threshold Sweep ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Baseline latency: {baseline['latency_ms_mean']:.3f} ms")
    print(f"Output shape: {baseline['last_hidden_shape']}")
    print()

    rows = []

    for threshold in THRESHOLDS:
        rte_model = AutoModel.from_pretrained(MODEL_NAME)
        rte_model = DistilBERTRTEGate(threshold=threshold).wrap(rte_model)

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
            "drift_last_mean": rte["drift_last_mean"],
            "output_shape": str(rte["last_hidden_shape"]),
        }
        rows.append(row)

        print(
            f"threshold={threshold:.2f} | "
            f"latency={rte['latency_ms_mean']:.3f} ms | "
            f"layers={rte['executed_layers_mean']:.2f} | "
            f"rho={rte['rho_mean']:.3f} | "
            f"saving={saving:.3f} | "
            f"speedup={speedup:.3f}x | "
            f"drift={rte['drift_last_mean']:.6f}"
        )

    # حفظ النتائج
    out_path = "benchmarks/results/distilbert_rte_threshold_sweep.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {out_path}")

    # أفضل نقطة حسب السرعة بشرط drift معقول
    # يمكنك تغيير هذا الشرط لاحقًا
    candidates = [r for r in rows if r["drift_last_mean"] <= 0.25]
    if candidates:
        best = max(candidates, key=lambda x: x["speedup_factor"])
        print("\n=== Best Candidate (drift <= 0.25) ===")
        print(best)
    else:
        print("\nNo candidate satisfied the drift constraint.")


if __name__ == "__main__":
    main()
