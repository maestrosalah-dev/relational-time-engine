from transformers import AutoTokenizer, AutoModel
from rte_engine import DistilBERTRTEGate
import torch

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

gate = DistilBERTRTEGate(threshold=0.01)
model = gate.wrap(model)

inputs = tokenizer(
    "Relational time engine for efficient inference.",
    return_tensors="pt"
)

with torch.no_grad():
    outputs, executed_layers = model(**inputs)

print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("Executed layers:", executed_layers)
print("Total hidden states returned:", len(outputs.hidden_states))
