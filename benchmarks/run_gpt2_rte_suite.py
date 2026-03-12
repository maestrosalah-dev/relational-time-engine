import csv
import time
from statistics import mean
from transformers import AutoTokenizer, AutoModel
from rte_engine.gpt2_gate import GPT2RTEGate

MODEL_NAME = "gpt2"
RUNS = 10
THRESHOLDS = [0.75, 1.0, 1.25]

PROMPTS = [
    "Relational time engine improves transformer efficiency.",
    "Efficient inference is essential for large language model deployment.",
    "Runtime layer gating can reduce unnecessary computation in decoder models.",
    "Structural drift may help determine whether deeper transformer blocks are still useful.",
    "Energy-aware inference optimization is becoming a major systems problem."
]


def truncate_prompt(tokenizer, text, target_len):
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]

    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        pad_token = tokenizer.eos_token_id
        pad_len = target_len - len(ids)
        ids = ids.tolist() + [pad_token] * pad_len
        ids = tokenizer.convert_ids_to_tokens(ids)
        ids = tokenizer.convert_tokens_to_ids(ids)

    return tokenizer.decode(ids, skip_special_tokens=False)


def build_inputs(tokenizer, text, target_len):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=target_len,
        padding="max_length",
    )
    return enc


def measure_baseline(model, inputs, runs=10):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        outputs = model(**inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {
        "latency_ms_mean": mean(times),
        "output_shape": tuple(outputs.last_hidden_state.shape),
    }


def measure_rte(model, inputs, runs=10):
    times = []
    layers = []
    rhos = []
    drifts = []

    for _ in range(runs):
        t0 = time.perf_counter()
        outputs, meta = model(**inputs)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)
        layers.append(meta["executed_layers"])
        rhos.append(meta["rho_layers"])
        drifts.append(mean(meta["drifts"]) if meta["drifts"] else 0.0)

    return {
        "latency_ms_mean": mean(times),
        "executed_layers_mean": mean(layers),
        "rho_mean": mean(rhos),
        "drift_mean": mean(drifts),
        "output_shape": tuple(outputs.last_hidden_state.shape),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    seq_lengths = [8, 16, 32]
    rows = []

    print("\n=== GPT-2 RTE Benchmark Suite ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Seq lengths: {seq_lengths}")
    print()

    for seq_len in seq_lengths:
        for prompt_idx, prompt in enumerate(PROMPTS):
            inputs = build_inputs(tokenizer, prompt, seq_len)

            baseline_model = AutoModel.from_pretrained(MODEL_NAME)
            baseline = measure_baseline(baseline_model, inputs, runs=RUNS)

            for threshold in THRESHOLDS:
                rte_model = AutoModel.from_pretrained(MODEL_NAME)
                rte_model = GPT2RTEGate(threshold=threshold).wrap(rte_model)

                rte = measure_rte(rte_model, inputs, runs=RUNS)

                speedup = baseline["latency_ms_mean"] / rte["latency_ms_mean"]
                saving = 1.0 - rte["rho_mean"]

                row = {
                    "prompt_idx": prompt_idx,
                    "seq_len": seq_len,
                    "threshold": threshold,
                    "baseline_latency_ms": baseline["latency_ms_mean"],
                    "rte_latency_ms": rte["latency_ms_mean"],
                    "speedup_factor": speedup,
                    "executed_layers_mean": rte["executed_layers_mean"],
                    "rho_mean": rte["rho_mean"],
                    "saving_mean": saving,
                    "drift_mean": rte["drift_mean"],
                    "output_shape": str(rte["output_shape"]),
                }
                rows.append(row)

                print(
                    f"seq={seq_len:>2} | prompt={prompt_idx} | thr={threshold:.2f} | "
                    f"lat={rte['latency_ms_mean']:.3f} ms | "
                    f"speedup={speedup:.3f}x | "
                    f"layers={rte['executed_layers_mean']:.2f} | "
                    f"saving={saving:.3f} | "
                    f"drift={rte['drift_mean']:.6f}"
                )

    out_path = "benchmarks/results/gpt2_rte_suite.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV -> {out_path}")

    # summary by threshold
    summaries = []
    for threshold in THRESHOLDS:
        sub = [r for r in rows if r["threshold"] == threshold]
        summaries.append({
            "threshold": threshold,
            "baseline_latency_ms_mean": mean([r["baseline_latency_ms"] for r in sub]),
            "rte_latency_ms_mean": mean([r["rte_latency_ms"] for r in sub]),
            "speedup_factor_mean": mean([r["speedup_factor"] for r in sub]),
            "executed_layers_mean": mean([r["executed_layers_mean"] for r in sub]),
            "saving_mean": mean([r["saving_mean"] for r in sub]),
            "drift_mean": mean([r["drift_mean"] for r in sub]),
        })

    summary_path = "benchmarks/results/gpt2_rte_suite_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Saved summary -> {summary_path}")

    print("\n=== Threshold Summary ===")
    for s in summaries:
        print(
            f"thr={s['threshold']:.2f} | "
            f"baseline={s['baseline_latency_ms_mean']:.3f} ms | "
            f"rte={s['rte_latency_ms_mean']:.3f} ms | "
            f"speedup={s['speedup_factor_mean']:.3f}x | "
            f"layers={s['executed_layers_mean']:.2f} | "
            f"saving={s['saving_mean']:.3f} | "
            f"drift={s['drift_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
