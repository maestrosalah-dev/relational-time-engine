from transformers import AutoTokenizer, AutoModel
from rte import RTEGate
import torch

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model = RTEGate.wrap(model, threshold=0.01)

inputs = tokenizer("Relational time engine for efficient inference.", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Last hidden state shape:", outputs.last_hidden_state.shape)

if hasattr(outputs, "rte_drifts"):
    print("RTE drifts:", outputs.rte_drifts)
else:
    print("No drift info found.")
