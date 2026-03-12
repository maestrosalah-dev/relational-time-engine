from transformers.modeling_outputs import BaseModelOutputWithPast
from .policy import DriftExitPolicy


class GPT2RTEGate:
    def __init__(self, threshold=0.02, policy: DriftExitPolicy | None = None):
        self.policy = policy or DriftExitPolicy(threshold=threshold)

    def wrap(self, model):
        if not hasattr(model, "h"):
            raise ValueError("This wrapper currently supports GPT2Model-like architectures only.")

        blocks = model.h
        policy = self.policy

        def gated_forward(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            return_dict=True,
            **kwargs,
        ):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

            if inputs_embeds is None:
                hidden_states = model.wte(input_ids)
            else:
                hidden_states = inputs_embeds

            seq_len = hidden_states.size(1)
            position_ids = __import__("torch").arange(
                0, seq_len, device=hidden_states.device
            ).unsqueeze(0)

            hidden_states = hidden_states + model.wpe(position_ids)

            executed_layers = 0
            drifts = []
            all_hidden_states = [hidden_states]

            prev = hidden_states

            for block in blocks:
                outputs = block(hidden_states)
                hidden_states = outputs[0]

                executed_layers += 1
                all_hidden_states.append(hidden_states)

                should_exit, drift = policy.should_exit(prev, hidden_states)
                drifts.append(float(drift.detach().cpu()))

                if should_exit:
                    break

                prev = hidden_states

            hidden_states = model.ln_f(hidden_states)

            meta = {
                "executed_layers": executed_layers,
                "rho_layers": executed_layers / len(blocks),
                "drifts": drifts,
                "policy_threshold": policy.threshold,
                "policy_mode": policy.mode,
            }

            out = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
                hidden_states=tuple(all_hidden_states),
                attentions=None,
            )

            if return_dict:
                return out, meta

            return hidden_states, meta

        model.forward = gated_forward
        return model
