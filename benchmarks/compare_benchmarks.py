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


def load(path: str):
    groups = defaultdict(lambda: {"rho": [], "theta": []})
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nl = float(row["noise_level"])
            groups[nl]["rho"].append(float(row["rho_win"]))
            groups[nl]["theta"].append(float(row["theta"]))
    return groups


def summarize(groups):
    out = {}
    for nl in sorted(groups.keys()):
        rho = groups[nl]["rho"]
        th = groups[nl]["theta"]
        out[nl] = {
            "rho_mean": mean(rho),
            "rho_std": std(rho),
            "theta_mean": mean(th),
            "theta_std": std(th),
            "windows": len(rho),
        }
    return out


def main():
    strict_path = "benchmarks/results/synthetic_metrics_strict.csv"
    flex_path = "benchmarks/results/synthetic_metrics_flex.csv"

    strict = summarize(load(strict_path))
    flex = summarize(load(flex_path))

    print("\n=== Comparison: STRICT vs FLEX ===")
    print("noise | rho_strict | rho_flex | saving_strict | saving_flex")
    print("-----------------------------------------------------------")

    rows = []
    for nl in sorted(strict.keys()):
        rs = strict[nl]["rho_mean"]
        rf = flex[nl]["rho_mean"]

        # theoretical compute saving vs always-on baseline (rho=1)
        save_s = 1.0 - rs
        save_f = 1.0 - rf

        print(f"{nl:4.2f} | {rs:10.3f} | {rf:8.3f} | {save_s:12.3f} | {save_f:10.3f}")

        rows.append({
            "noise_level": nl,
            "rho_strict_mean": rs,
            "rho_flex_mean": rf,
            "saving_strict": save_s,
            "saving_flex": save_f,
        })

    out_path = "benchmarks/results/synthetic_compare_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["noise_level", "rho_strict_mean", "rho_flex_mean", "saving_strict", "saving_flex"],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nSaved comparison -> {out_path}")


if __name__ == "__main__":
    main()