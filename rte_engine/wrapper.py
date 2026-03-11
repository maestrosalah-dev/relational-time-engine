import torch


class RTEGate:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def drift(self, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x_next - x_prev))

    @classmethod
    def wrap(cls, model, threshold: float = 0.01):
        gate = cls(threshold=threshold)
        original_forward = model.forward

        def gated_forward(*args, **kwargs):
            kwargs["output_hidden_states"] = True
            outputs = original_forward(*args, **kwargs)

            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
                drifts = []

                prev = hidden_states[0]
                for h in hidden_states[1:]:
                    d = gate.drift(prev, h)
                    drifts.append(float(d.detach().cpu()))
                    prev = h

                # لا نضيف خاصية مباشرة إلى outputs لأن بعض الكلاسات تمنع ذلك
                return outputs, drifts

            return outputs, []

        model.forward = gated_forward
        return model

