import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import json 
from datasets import load_dataset

# 1. Setup Model and Tokenizer
model_id = "ilsp/Llama-Krikri-8B-Base"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Use float16 if your GPU is older
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Essential for Llama training stability

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="mps"
)

# 2. Prepare Model for PEFT
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, peft_config)
def formatting_prompts_func(example):
    # No loop! 'example' is now a single dictionary, not a batch
    text = (
        f"### Οδηγία: {example['instruction']}\n"
        f"### Είσοδος: {example['input']}\n"
        f"### Απάντηση: {example['output']}"
    )
    return text  # Return a single string

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./krikri-dialect-results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_32bit", # Stable for 4-bit training
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
)

# with open("train_ds.json", "r") as f:
#     dataset = json.load(f)
dataset = load_dataset('json', data_files="train_ds.json", split="train")

# 1. Define SFTConfig instead of standard TrainingArguments
# SFTConfig inherits from TrainingArguments, so you put everything here
sft_config = SFTConfig(
    output_dir="./krikri-dialect-results",
    max_length=2048,           # Moved here
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
)

# 2. Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,    # Use processing_class instead of tokenizer
    args=sft_config,               # Pass your new config here
)

# 6. Start Training
trainer.train()

# 7. Save the Adapter
trainer.save_model("./krikri-dialect-final")
