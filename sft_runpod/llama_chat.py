# ================================
# 1. Install dependencies
# ================================
# !pip install -q transformers accelerate peft bitsandbytes datasets

# ================================
# 2. Imports
# ================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ================================
# 3. Load tokenizer + quantized Meta-Llama-3.1-8B
# ================================
model_name = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # add padding token if missing

model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_4bit=True, device_map="auto"  # QLoRA
)

# ================================
# 4. Attach LoRA adapters
# ================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # works well for LLaMA 2/3
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# ================================
# 5. Load your JSONL dataset
# ================================
# Make sure your file is uploaded to Colab first, e.g. `/content/data.jsonl`
dataset = load_dataset("json", data_files="/content/data.jsonl", split="train")

# Train/validation split
dataset = dataset.train_test_split(test_size=0.1)


# ================================
# 6. Tokenization function
# ================================
def format_example(example):
    # Simple instruction-response formatting
    text = f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    return tokenizer(text, padding="max_length", truncation=True, max_length=512)


tokenized_datasets = dataset.map(format_example, batched=False)


# Add labels for causal LM
def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example


tokenized_datasets = tokenized_datasets.map(add_labels, batched=False)

# ================================
# 7. TrainingArguments + Trainer
# ================================
training_args = TrainingArguments(
    output_dir="./qlora-llama3-8b-results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# ================================
# 8. Train
# ================================
trainer.train()

# ================================
# 9. Save LoRA adapters
# ================================
model.save_pretrained("./qlora-llama3-8b-adapter")
tokenizer.save_pretrained("./qlora-llama3-8b-adapter")
