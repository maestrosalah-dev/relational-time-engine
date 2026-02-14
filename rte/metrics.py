import csv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class WindowRecord:
    noise_level: float
    step: int
    theta: float
    rho_win: float


class CSVLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: List[WindowRecord] = []

    def add(self, noise_level: float, out: Dict[str, Any]) -> None:
        if out.get("theta_updated") and out.get("rho_expected_window") is not None:
            self.rows.append(
                WindowRecord(
                    noise_level=float(noise_level),
                    step=int(out["step"]),
                    theta=float(out["theta"]),
                    rho_win=float(out["rho_expected_window"]),
                )
            )

    def save(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["noise_level", "step", "theta", "rho_win"])
            for r in self.rows:
                w.writerow([r.noise_level, r.step, r.theta, r.rho_win])