import csv
import os
from statistics import mean

IN_PATH = os.path.join("benchmarks", "results", "transformer_latency_cpu.csv")
OUT_PATH = os.path.join("benchmarks", "results", "transformer_latency_cpu_summary.csv")


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x):
    return float(x)


def summarize(rows):
    # group by (noise, mode)
    groups = {}
    for r in rows:
        key = (float(r["noise_level"]), r["mode"])
        groups.setdefault(key, []).append(r)

    # Make one row per key
    out = []
    for (noise, mode), rs in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        out.append(
            dict(
                noise_level=noise,
                mode=mode,
                rho_layers=mean([to_float(x["rho_layers"]) for x in rs]),
                saving_vs_baseline=mean([to_float(x["saving_vs_baseline"]) for x in rs]),
                latency_ms_mean=mean([to_float(x["latency_ms_mean"]) for x in rs]),
                throughput_samples_per_s=mean([to_float(x["throughput_samples_per_s"]) for x in rs]),
            )
        )
    return out


def print_table(summary):
    print("\n=== CPU Latency Summary (Early-Exit) ===")
    print("noise | mode     | rho_layers | saving | latency_ms | throughput_sps")
    print("-" * 72)
    for r in summary:
        print(
            f"{r['noise_level']:>4.2f} | {r['mode']:<8} |"
            f"   {r['rho_layers']:>6.3f} | {r['saving_vs_baseline']:>6.3f} |"
            f"   {r['latency_ms_mean']:>9.2f} | {r['throughput_samples_per_s']:>12.2f}"
        )


def write_csv(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing: {IN_PATH}. Run run_transformer_latency_benchmark.py first.")

    rows = load_rows(IN_PATH)
    summary = summarize(rows)
    print_table(summary)
    write_csv(summary, OUT_PATH)
    print(f"\nSaved summary -> {OUT_PATH}")


if __name__ == "__main__":
    main()