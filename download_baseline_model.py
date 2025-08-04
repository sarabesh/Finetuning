from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "TinyLlama/TinyLlama_v1.1"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_8bit=False  # save full weights
)

model.save_pretrained("./tinyllama-base")

tokenizer.save_pretrained("./tinyllama-base")

from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
model.save_pretrained("./tinyllama-lora")