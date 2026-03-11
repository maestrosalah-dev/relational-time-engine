import time
import torch
from transformers import AutoTokenizer, AutoModel
from rte_engine import DistilBERTRTEGate


MODEL_NAME = "distilbert-base-uncased"
TEXT = "Relational time engine for efficient transformer inference."
RUNS = 20
THRESHOLD = 0.20


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

    # Baseline
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    baseline = measure_baseline(base_model, inputs, runs=RUNS)

    # RTE
    rte_model = AutoModel.from_pretrained(MODEL_NAME)
    rte_model = DistilBERTRTEGate(threshold=THRESHOLD).wrap(rte_model)
    rte = measure_rte(rte_model, inputs, runs=RUNS)

    speedup = baseline["latency_ms_mean"] / rte["latency_ms_mean"]
    saving = 1.0 - rte["rho_mean"]

    print("\n=== DistilBERT RTE Benchmark ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print()
    print("Baseline:")
    print(f"  Mean latency (ms): {baseline['latency_ms_mean']:.3f}")
    print(f"  Output shape:      {baseline['last_hidden_shape']}")
    print()
    print("RTE Early-Exit:")
    print(f"  Mean latency (ms):      {rte['latency_ms_mean']:.3f}")
    print(f"  Mean executed layers:   {rte['executed_layers_mean']:.3f}")
    print(f"  Mean rho:               {rte['rho_mean']:.3f}")
    print(f"  Mean saving:            {saving:.3f}")
    print(f"  Mean last drift:        {rte['drift_last_mean']:.6f}")
    print(f"  Output shape:           {rte['last_hidden_shape']}")
    print()
    print(f"Speedup factor: {speedup:.3f}x")


if __name__ == "__main__":
    main()
