from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch


@dataclass
class DecoderStepTrace:
    step_idx: int
    sequence_len: int
    executed_layers: int
    total_layers: int
    rho_layers: float
    saving: float
    drift_mean: float
    last_drift: float


class MockDecoderRuntime:
    """
    Minimal decoder-style runtime wrapper for block-wise autoregressive execution.

    This is a serving-oriented abstraction layer:
    - embeds current input
    - iterates over decoder blocks
    - checks RTE exit policy after each block
    - returns final hidden state + per-step trace

    It does NOT implement full KV cache logic yet.
    """

    def __init__(self, model, policy):
        if not hasattr(model, "h"):
            raise ValueError("MockDecoderRuntime currently expects a GPT2Model-like decoder (model.h).")

        self.model = model
        self.policy = policy
        self.blocks = model.h
        self.total_layers = len(self.blocks)

    def _embed(self, input_ids=None, inputs_embeds=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            hidden_states = self.model.wte(input_ids)
        else:
            hidden_states = inputs_embeds

        seq_len = hidden_states.size(1)
        position_ids = torch.arange(
            0, seq_len, device=hidden_states.device
        ).unsqueeze(0)

        hidden_states = hidden_states + self.model.wpe(position_ids)
        return hidden_states

    def forward_step(self, step_idx: int, input_ids=None, inputs_embeds=None):
        hidden_states = self._embed(input_ids=input_ids, inputs_embeds=inputs_embeds)

        prev = hidden_states
        drifts: List[float] = []
        executed_layers = 0

        for block in self.blocks:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

            executed_layers += 1

            should_exit, drift = self.policy.should_exit(prev, hidden_states)
            drift_value = float(drift.detach().cpu())
            drifts.append(drift_value)

            if should_exit:
                break

            prev = hidden_states

        hidden_states = self.model.ln_f(hidden_states)

        drift_mean = sum(drifts) / len(drifts) if drifts else 0.0
        last_drift = drifts[-1] if drifts else 0.0
        rho = executed_layers / self.total_layers
        saving = 1.0 - rho

        trace = DecoderStepTrace(
            step_idx=step_idx,
            sequence_len=int(hidden_states.shape[1]),
            executed_layers=executed_layers,
            total_layers=self.total_layers,
            rho_layers=rho,
            saving=saving,
            drift_mean=drift_mean,
            last_drift=last_drift,
        )

        return hidden_states, trace

    def decode(self, input_ids, max_new_tokens: int = 8) -> Dict[str, Any]:
        """
        Minimal autoregressive decode loop using an embedding-matrix projection
        as a next-token proxy.
        """
        generated = input_ids.clone()
        traces: List[DecoderStepTrace] = []

        with torch.no_grad():
            for step_idx in range(max_new_tokens):
                hidden_states, trace = self.forward_step(step_idx=step_idx, input_ids=generated)

                # next-token proxy using embedding matrix
                last_token_hidden = hidden_states[:, -1, :]
                logits = last_token_hidden @ self.model.wte.weight.T
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)
                traces.append(trace)

        return {
            "generated_ids": generated,
            "traces": traces,
            "summary": self.summarize(traces),
        }

    @staticmethod
    def summarize(traces: List[DecoderStepTrace]) -> Dict[str, float]:
        if not traces:
            return {
                "executed_layers_mean": 0.0,
                "rho_mean": 0.0,
                "saving_mean": 0.0,
                "drift_mean": 0.0,
                "last_drift_mean": 0.0,
            }

        return {
            "executed_layers_mean": sum(t.executed_layers for t in traces) / len(traces),
            "rho_mean": sum(t.rho_layers for t in traces) / len(traces),
            "saving_mean": sum(t.saving for t in traces) / len(traces),
            "drift_mean": sum(t.drift_mean for t in traces) / len(traces),
            "last_drift_mean": sum(t.last_drift for t in traces) / len(traces),
        }

    @staticmethod
    def traces_as_dicts(traces: List[DecoderStepTrace]) -> List[Dict[str, Any]]:
        return [asdict(t) for t in traces]
