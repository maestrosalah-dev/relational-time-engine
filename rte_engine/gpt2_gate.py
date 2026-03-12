import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


class GPT2RTEGate:
    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def drift(self, x_prev, x_next):
        return torch.mean(torch.abs(x_next - x_prev))

    def wrap(self, model):

        blocks = model.transformer.h
        gate = self

        original_forward = model.forward

        def gated_forward(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            return_dict=True,
            **kwargs,
        ):

            transformer = model.transformer

            if inputs_embeds is None:
                hidden_states = transformer.wte(input_ids)
            else:
                hidden_states = inputs_embeds

            position_ids = torch.arange(
                0, hidden_states.size(1),
                device=hidden_states.device
            ).unsqueeze(0)

            hidden_states = hidden_states + transformer.wpe(position_ids)

            executed_layers = 0
            drifts = []

            prev = hidden_states

            for block in blocks:

                outputs = block(hidden_states)

                hidden_states = outputs[0]

                executed_layers += 1

                d = gate.drift(prev, hidden_states)
                drifts.append(float(d.detach().cpu()))

                if d < gate.threshold:
                    break

                prev = hidden_states

            hidden_states = transformer.ln_f(hidden_states)

            meta = {
                "executed_layers": executed_layers,
                "rho_layers": executed_layers / len(blocks),
                "drifts": drifts,
            }

            if return_dict:

                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states
                ), meta

            return hidden_states, meta

        model.forward = gated_forward

        return model
