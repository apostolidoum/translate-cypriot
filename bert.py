import torch
import unicodedata
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

model_id = "nlpaueb/bert-base-greek-uncased-v1"

# 1. Text Normalization Function (Crucial for BERT Uncased)
def normalize_greek(text):
    # Lowercase and remove accents
    text = text.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')

# 2. Load and Preprocess Data
dataset = load_dataset("json", data_files="train_ds.json", split="train")

tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    # We use the Cypriot 'output' column for MLM
    texts = [normalize_greek(t) for t in examples["output"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)
# 3. Data Collator for MLM
# This automatically handles masking 15% of the tokens
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

# 4. Load Model
model = AutoModelForMaskedLM.from_pretrained(model_id)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./greekbert-cypriot-mlm",
    num_train_epochs=10, # MLM often needs a few more epochs to adapt
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # Use mixed precision if GPU allows
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 7. Start Training
trainer.train()

# 8. Save the adapted model
trainer.save_model("./greekbert-cypriot-final")