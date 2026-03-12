import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.distilbert.modeling_distilbert import create_bidirectional_mask
from .policy import DriftExitPolicy


class DistilBERTRTEGate:
    def __init__(self, threshold: float = 0.01, policy: DriftExitPolicy | None = None):
        self.policy = policy or DriftExitPolicy(threshold=threshold)

    def wrap(self, model):
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "layer"):
            raise ValueError("This wrapper currently supports DistilBERT-like models only.")

        embeddings = model.embeddings
        layers = model.transformer.layer
        config = model.config
        policy = self.policy

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

                should_exit, drift = policy.should_exit(prev, hidden_states)
                drifts.append(float(drift.detach().cpu()))

                if should_exit:
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
                "policy_threshold": policy.threshold,
                "policy_mode": policy.mode,
            }

            if return_dict:
                return outputs, meta

            return (hidden_states, tuple(all_hidden_states), meta)

        model.forward = gated_forward
        return model
