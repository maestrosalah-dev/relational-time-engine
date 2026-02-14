import os
import random

from rte.core import RTEEngine, RTEConfig
from rte.discrimination import RTERuleParams
from rte.metrics import CSVLogger


def make_event(noise_level: float) -> dict:
    is_signal = random.random() > noise_level

    if is_signal:
        confidence = random.uniform(0.6, 0.95)
        output_entropy = random.uniform(0.2, 0.9)
        attn_mean = random.uniform(0.25, 0.8)
        attn_var = random.uniform(0.02, 0.08)
    else:
        confidence = random.uniform(0.05, 0.55)
        output_entropy = random.uniform(1.0, 2.2)
        attn_mean = random.uniform(0.0, 0.35)
        attn_var = random.uniform(0.0, 0.02)

    is_critical = (random.random() < 0.01)

    return {
        "confidence": confidence,
        "output_entropy": output_entropy,
        "attn_mean": attn_mean,
        "attn_var": attn_var,
        "is_critical": is_critical,
    }


def run_mode(mode: str, out_csv: str) -> None:
    cfg = RTEConfig(
        decision_mode="threshold",
        tau=0.5,
        window_size=128,
        rho_star=0.10,
        alpha=8.0,
        eta=0.08,
        lam=0.02,
        theta0=0.70,
    )

    rules = RTERuleParams(
        mode=mode,
        c_min=0.35,
        a_min=0.20,
        H_min=1.2,
        v_max=0.015,
        w1=0.60,
        w2=0.25,
        w3=0.15,
        # keep strict penalty defaults; flex ignores them
        noise_penalty_base=0.25,
        noise_penalty_gain=0.35,
    )

    logger = CSVLogger(out_csv)

    print(f"\n==============================")
    print(f"MODE = {mode.upper()}  ->  {out_csv}")
    print(f"==============================")

    for noise_level in [0.2, 0.5, 0.8]:
        print(f"\n=== Noise level: {noise_level} ===")
        rte = RTEEngine(config=cfg, rule_params=rules)

        for _ in range(500):
            out = rte.step(make_event(noise_level))
            logger.add(noise_level, out)

            if out["theta_updated"]:
                print(
                    f"step={out['step']:4d}  "
                    f"theta={out['theta']:.3f}  "
                    f"rho_win={out['rho_expected_window']:.3f}"
                )

    logger.save()


def main():
    os.makedirs("benchmarks/results", exist_ok=True)

    run_mode("strict", "benchmarks/results/synthetic_metrics_strict.csv")
    run_mode("flex", "benchmarks/results/synthetic_metrics_flex.csv")

    print("\nSaved:")
    print(" - benchmarks/results/synthetic_metrics_strict.csv")
    print(" - benchmarks/results/synthetic_metrics_flex.csv")


if __name__ == "__main__":
    main()