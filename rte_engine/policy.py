import torch


class DriftExitPolicy:
    def __init__(self, threshold: float = 0.2, mode: str = "mean_abs"):
        self.threshold = threshold
        self.mode = mode

    def compute_drift(self, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean_abs":
            return torch.mean(torch.abs(x_next - x_prev))
        elif self.mode == "l2":
            return torch.norm(x_next - x_prev) / x_next.numel()
        else:
            raise ValueError(f"Unsupported drift mode: {self.mode}")

    def should_exit(self, x_prev: torch.Tensor, x_next: torch.Tensor):
        drift = self.compute_drift(x_prev, x_next)
        return bool(drift < self.threshold), drift
