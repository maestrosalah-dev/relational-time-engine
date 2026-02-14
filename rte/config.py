from dataclasses import dataclass
from typing import Optional

@dataclass
class RTEConfig:
    # Gate params
    alpha: float = 8.0
    theta0: float = 0.45
    theta_min: float = 0.05
    theta_max: float = 0.95

    # Adaptive threshold control
    rho_star: float = 0.25
    eta: float = 0.08
    lam: float = 0.02

    # Decision mode (YOU CHOSE THIS)
    decision_mode: str = "threshold"
    tau: float = 0.5

    # Windowing
    window_size: int = 128

    # Safety
    enable_safety_override: bool = True
    entropy_uncertainty_gamma: Optional[float] = None

    # Logging
    enable_timing: bool = True