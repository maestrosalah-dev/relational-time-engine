import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.distilbert.modeling_distilbert import create_bidirectional_mask


class DistilBERTRTEGate:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def drift(self, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x_next - x_prev))

    def wrap(self, model):
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "layer"):
            raise ValueError("This wrapper currently supports DistilBERT-like models only.")

        embeddings = model.embeddings
        transformer = model.transformer
        layers = transformer.layer
        config = model.config
        gate = self

        def gated_forward(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            position_ids=None,
            return_dict=True,
            **kwargs,
        ):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if input_ids is not None:
                input_shape = input_ids.size()
                device = input_ids.device
            else:
                input_shape = inputs_embeds.size()[:-1]
                device = inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)

            hidden_states = embeddings(input_ids, inputs_embeds, position_ids)

            # توافق مع نسختك من transformers:
            # الدالة تتوقع غالبًا (config, hidden_states, attention_mask)
            attention_mask_prepared = create_bidirectional_mask(
                config,
                hidden_states,
                attention_mask,
            )

            all_hidden_states = [hidden_states]
            executed_layers = 0
            drifts = []

            prev = hidden_states

            for layer_module in layers:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=attention_mask_prepared,
                    **kwargs,
                )

                executed_layers += 1
                all_hidden_states.append(hidden_states)

                d = gate.drift(prev, hidden_states)
                drifts.append(float(d.detach().cpu()))

                if d < gate.threshold:
                    break

                prev = hidden_states

            outputs = BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=tuple(all_hidden_states),
                attentions=None,
            )

            meta = {
                "executed_layers": executed_layers,
                "drifts": drifts,
                "rho_layers": executed_layers / len(layers),
            }

            if return_dict:
                return outputs, meta

            return (hidden_states, tuple(all_hidden_states), meta)

        model.forward = gated_forward
        return model
