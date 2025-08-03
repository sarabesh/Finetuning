from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Step 1: Load base model (8-bit)
base_model_id = "TinyLlama/TinyLlama_v1.1"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_8bit=True,
    device_map="auto"
)

# Step 2: Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)

# (Optional) Print trainable parameters
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

print_trainable_parameters(model)

# Step 3: Save tokenizer + LoRA adapter (NOT full base model)
save_path = "./tinyllama-lora"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"LoRA adapter and tokenizer saved to: {save_path}")