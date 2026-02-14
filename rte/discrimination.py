from dataclasses import dataclass
from typing import Dict, Any, Optional


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class RTERuleParams:
    c_min: float = 0.35
    a_min: float = 0.20
    H_min: float = 1.2
    v_max: float = 0.015

    w1: float = 0.60
    w2: float = 0.25
    w3: float = 0.15

    mode: str = "strict"  # "strict" or "flex"

    noise_penalty_base: float = 0.25
    noise_penalty_gain: float = 0.35


def compute_D(event: Dict[str, Any], p: Optional[RTERuleParams] = None) -> float:

    if p is None:
        p = RTERuleParams()

    c = float(event.get("confidence", 1.0))
    H = float(event.get("output_entropy", 0.0))
    a = float(event.get("attn_mean", 0.0))
    v = float(event.get("attn_var", 0.0))
    critical = bool(event.get("is_critical", False))

    # Critical always activate
    if critical:
        return 1.0

    # Rule 1
    r1 = 1.0 if c < p.c_min else 0.0

    # Rule 2 (soft attention)
    denom_a = max(1.0 - p.a_min, 1e-9)
    r2 = clip((a - p.a_min) / denom_a, 0.0, 1.0)

    # Rule 3 (entropy + flat attention)
    denom_h = max(p.H_min, 1e-9)
    h_score = clip((H - p.H_min) / denom_h, 0.0, 1.0)

    denom_v = max(p.v_max, 1e-9)
    v_score = clip((p.v_max - v) / denom_v, 0.0, 1.0)

    r3 = clip(0.5 * h_score + 0.5 * v_score, 0.0, 1.0)

    # Base aggregation
    D = clip(p.w1 * r1 + p.w2 * r2 + p.w3 * r3, 0.0, 1.0)

    # Strict mode: penalize uninformative noise
    if p.mode == "strict":
        if (H > p.H_min) and (v < p.v_max) and (c >= p.c_min):
            excess = clip((H - p.H_min) / max(p.H_min, 1e-9), 0.0, 1.0)
            penalty = p.noise_penalty_base + p.noise_penalty_gain * excess
            D = clip(D - penalty, 0.0, 1.0)

    return D