from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import math
import time

from .discrimination import compute_D, RTERuleParams


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class RTEConfig:
    # Gate params
    alpha: float = 8.0
    theta0: float = 0.70
    theta_min: float = 0.05
    theta_max: float = 0.95

    # Adaptive threshold control
    rho_star: float = 0.10
    eta: float = 0.08
    lam: float = 0.02

    # Decision mode
    decision_mode: str = "threshold"
    tau: float = 0.5

    # Windowing
    window_size: int = 128

    # Safety
    enable_safety_override: bool = True
    entropy_uncertainty_gamma: Optional[float] = None

    # Logging
    enable_timing: bool = True


@dataclass
class RTEWindowStats:
    n: int = 0
    sum_P: float = 0.0
    sum_S: int = 0
    forced_safety: int = 0

    def add(self, P: float, S: int, forced: bool) -> None:
        self.n += 1
        self.sum_P += float(P)
        self.sum_S += int(S)
        if forced:
            self.forced_safety += 1

    @property
    def rho_expected(self) -> float:
        return self.sum_P / self.n if self.n else 0.0

    @property
    def rho_realized(self) -> float:
        return self.sum_S / self.n if self.n else 0.0


class RTEEngine:
    def __init__(
        self,
        config: Optional[RTEConfig] = None,
        rule_params: Optional[RTERuleParams] = None,
    ) -> None:
        self.cfg = config or RTEConfig()
        self.rule_params = rule_params or RTERuleParams()

        self.theta: float = float(self.cfg.theta0)
        self.step_idx: int = 0

        self.window = RTEWindowStats()
        self._window_buffer: List[Tuple[float, int, bool]] = []

        self.last_step_ms: Optional[float] = None

    def gate_probability(self, D: float) -> float:
        x = self.cfg.alpha * (float(D) - float(self.theta))
        return sigmoid(x)

    def decide(self, P: float) -> int:
        if self.cfg.decision_mode == "sample":
            import random
            return 1 if P > random.random() else 0
        return 1 if P > self.cfg.tau else 0

    def entropy_of_prob(self, P: float) -> float:
        p = min(max(P, 1e-12), 1.0 - 1e-12)
        return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)

    def safety_override(self, event: Dict[str, Any], P: float) -> bool:
        if not self.cfg.enable_safety_override:
            return False

        # Hard whitelist
        if bool(event.get("is_critical", False)):
            return True

        # Optional uncertainty trigger
        if self.cfg.entropy_uncertainty_gamma is not None:
            if self.entropy_of_prob(P) > float(self.cfg.entropy_uncertainty_gamma):
                return True

        return False

    def update_theta(self, rho: float) -> None:
        # theta_{k+1} = theta_k + eta(rho - rho*) - lam(theta_k - theta0)
        th = (
            self.theta
            + self.cfg.eta * (float(rho) - float(self.cfg.rho_star))
            - self.cfg.lam * (self.theta - float(self.cfg.theta0))
        )
        self.theta = max(self.cfg.theta_min, min(self.cfg.theta_max, th))

    def step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter() if self.cfg.enable_timing else None

        D = compute_D(event, self.rule_params)
        P = self.gate_probability(D)
        S = self.decide(P)

        forced = self.safety_override(event, P)
        if forced:
            S = 1

        # Update window
        self._window_buffer.append((P, S, forced))
        self.window.add(P, S, forced)

        theta_updated = False
        rho_window: Optional[float] = None

        if len(self._window_buffer) >= self.cfg.window_size:
            rho_window = self.window.rho_expected
            self.update_theta(rho_window)
            theta_updated = True

            # Reset window
            self._window_buffer.clear()
            self.window = RTEWindowStats()

        self.step_idx += 1

        if self.cfg.enable_timing and t0 is not None:
            self.last_step_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "step": self.step_idx,
            "D": float(D),
            "P": float(P),
            "S": int(S),
            "theta": float(self.theta),
            "theta_updated": bool(theta_updated),
            "rho_expected_window": float(rho_window) if rho_window is not None else None,
            "forced_safety": bool(forced),
            "last_step_ms": self.last_step_ms,
        }