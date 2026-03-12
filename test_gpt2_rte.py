from transformers import AutoTokenizer, AutoModel
from rte_engine.gpt2_gate import GPT2RTEGate

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")

model = GPT2RTEGate(threshold=0.02).wrap(model)

inputs = tokenizer(
    "Relational time engine improves transformer efficiency",
    return_tensors="pt"
)

outputs, meta = model(**inputs)

print("Executed layers:", meta["executed_layers"])
print("Rho:", meta["rho_layers"])
print("Drifts:", meta["drifts"])
print("Output shape:", outputs.last_hidden_state.shape)
print("Hidden states returned:", len(outputs.hidden_states))
