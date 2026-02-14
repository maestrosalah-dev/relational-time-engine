import csv
import math
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main():
    path = "benchmarks/results/synthetic_metrics.csv"

    groups = defaultdict(lambda: {"rho": [], "theta": []})

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nl = float(row["noise_level"])
            rho = float(row["rho_win"])
            theta = float(row["theta"])
            groups[nl]["rho"].append(rho)
            groups[nl]["theta"].append(theta)

    # Print summary
    print("\n=== Synthetic Benchmark Summary ===")
    print("noise_level | rho_mean | rho_std | theta_mean | theta_std | windows")
    print("--------------------------------------------------------------------")

    summary_rows = []
    for nl in sorted(groups.keys()):
        rho_list = groups[nl]["rho"]
        th_list = groups[nl]["theta"]

        rho_m = mean(rho_list)
        rho_s = std(rho_list)
        th_m = mean(th_list)
        th_s = std(th_list)

        windows = len(rho_list)

        print(
            f"{nl:10.2f} | {rho_m:8.3f} | {rho_s:7.3f} | {th_m:10.3f} | {th_s:9.3f} | {windows:7d}"
        )

        summary_rows.append(
            {
                "noise_level": nl,
                "rho_mean": rho_m,
                "rho_std": rho_s,
                "theta_mean": th_m,
                "theta_std": th_s,
                "windows": windows,
            }
        )

    # Save summary CSV
    out_path = "benchmarks/results/synthetic_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["noise_level", "rho_mean", "rho_std", "theta_mean", "theta_std", "windows"],
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()