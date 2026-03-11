from transformers import AutoTokenizer, AutoModel
from rte_engine import DistilBERTRTEGate
import torch

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

gate = DistilBERTRTEGate(threshold=0.20)
model = gate.wrap(model)

inputs = tokenizer(
    "Relational time engine for efficient inference.",
    return_tensors="pt"
)

with torch.no_grad():
    outputs, meta = model(**inputs)

print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("Executed layers:", meta["executed_layers"])
print("Rho layers:", meta["rho_layers"])
print("Drifts:", meta["drifts"])
print("Total hidden states returned:", len(outputs.hidden_states))
