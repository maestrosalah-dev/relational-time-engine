import torch
from transformers.modeling_outputs import BaseModelOutput


class DistilBERTRTEGate:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def drift(self, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x_next - x_prev))

    def wrap(self, model):
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "layer"):
            raise ValueError("This wrapper currently supports DistilBERT-like models only.")

        transformer = model.transformer
        layers = transformer.layer
        embeddings = model.embeddings

        gate = self

        def gated_forward(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        ):
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

            if input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds.")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)

            hidden_state = embeddings(input_ids=input_ids, input_embeds=inputs_embeds)

            all_hidden_states = [hidden_state]
            executed_layers = 0

            prev = hidden_state

            for i, layer_module in enumerate(layers):
                layer_outputs = layer_module(
                    hidden_state,
                    attn_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    output_attentions=output_attentions,
                )

                hidden_state = layer_outputs[-1]
                executed_layers += 1
                all_hidden_states.append(hidden_state)

                d = gate.drift(prev, hidden_state)

                if d < gate.threshold:
                    break

                prev = hidden_state

            if not return_dict:
                return (hidden_state, tuple(all_hidden_states), executed_layers)

            out = BaseModelOutput(
                last_hidden_state=hidden_state,
                hidden_states=tuple(all_hidden_states),
                attentions=None,
            )
            return out, executed_layers

        model.forward = gated_forward
        return model
