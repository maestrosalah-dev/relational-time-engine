from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class RTEPolicyConfig:
    threshold: float = 0.75
    mode: str = "mean_abs"
    min_layers: int = 1
    max_layers: Optional[int] = None


@dataclass
class RTERuntimeState:
    step_idx: int = 0
    executed_layers: int = 0
    total_layers: int = 0
    rho_layers: float = 1.0
    last_drift: float = 0.0
    should_exit: bool = False


class VLLMAdapterSketch:
    """
    Conceptual adapter for integrating RTE-like early-exit control
    into a decoder-style token generation runtime.

    This is NOT a full vLLM integration yet.
    It defines the logic boundary where an RTE policy would sit.
    """

    def __init__(self, policy):
        self.policy = policy

    def begin_step(self, step_idx: int, total_layers: int) -> RTERuntimeState:
        return RTERuntimeState(
            step_idx=step_idx,
            executed_layers=0,
            total_layers=total_layers,
            rho_layers=1.0,
            last_drift=0.0,
            should_exit=False,
        )

    def update_after_layer(
        self,
        state: RTERuntimeState,
        prev_hidden,
        next_hidden,
    ) -> RTERuntimeState:
        should_exit, drift = self.policy.should_exit(prev_hidden, next_hidden)
        state.executed_layers += 1
        state.last_drift = float(drift.detach().cpu())
        state.should_exit = should_exit
        state.rho_layers = state.executed_layers / max(state.total_layers, 1)
        return state

    def can_exit(self, state: RTERuntimeState) -> bool:
        # Optional conservative guard: do not exit before min_layers
        if state.executed_layers < getattr(self.policy, "min_layers", 1):
            return False
        return state.should_exit

    def summarize_step(self, state: RTERuntimeState) -> Dict[str, Any]:
        return {
            "step_idx": state.step_idx,
            "executed_layers": state.executed_layers,
            "total_layers": state.total_layers,
            "rho_layers": state.rho_layers,
            "last_drift": state.last_drift,
            "should_exit": state.should_exit,
        }
