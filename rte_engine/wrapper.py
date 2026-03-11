import torch

class RTEGate:

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def drift(self, x_prev, x_next):
        return torch.mean(torch.abs(x_next - x_prev))

    def wrap(self, model):

        original_forward = model.forward

        def gated_forward(*args, **kwargs):

            outputs = original_forward(*args, **kwargs)

            hidden_states = outputs.hidden_states

            prev = hidden_states[0]

            for i, h in enumerate(hidden_states[1:]):

                d = self.drift(prev, h)

                if d < self.threshold:
                    break

                prev = h

            return outputs

        model.forward = gated_forward

        return model

