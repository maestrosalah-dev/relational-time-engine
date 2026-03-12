from transformers import AutoTokenizer, AutoModel
from rte_engine import DriftExitPolicy, MockDecoderRuntime

MODEL_NAME = "gpt2"
PROMPT = "Relational time engine improves transformer efficiency"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(MODEL_NAME)
policy = DriftExitPolicy(threshold=1.0, mode="mean_abs")

runtime = MockDecoderRuntime(model=model, policy=policy)

inputs = tokenizer(PROMPT, return_tensors="pt")
result = runtime.decode(inputs["input_ids"], max_new_tokens=8)

print("Generated shape:", result["generated_ids"].shape)
print("Summary:", result["summary"])
print("First trace:", result["traces"][0])
print("Last trace:", result["traces"][-1])
