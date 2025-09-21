from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from huggingface_hub import InferenceClient


USER_NAME = "iwswordpress"
# Base model (chat-tuned LLaMA for example)
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./marcus-lora-multidata",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    fp16=True,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_ds,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="response",
    max_seq_length=512,
    formatting_func=lambda e: f"User: {e['prompt']}\nMarcus: {e['response']}",
)

trainer.train()

# huggingface-cli login
trainer.model.push_to_hub(f"{USER_NAME}/marcus-chen-agent")
tokenizer.push_to_hub(f"{USER_NAME}/marcus-chen-agent")


client = InferenceClient(f"{USER_NAME}/marcus-chen-agent")
print(client.text_generation("What’s your leadership style?"))
