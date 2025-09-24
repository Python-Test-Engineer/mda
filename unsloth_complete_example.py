# Complete UNSLOTH Fine-tuning Example for Google Colab
# This script will fine-tune a small model and upload it to Hugging Face

# Install required packages
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

# Imports
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Set up Hugging Face authentication
from huggingface_hub import login

# You'll need to run this and enter your HF token when prompted
print("Please enter your Hugging Face token when prompted:")
login()

print("🚀 Starting UNSLOTH fine-tuning example...")

# 1. Load a small model for testing
print("📥 Loading TinyLlama model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-1.1b-bnb-4bit",  # Small 1.1B parameter model
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

print("✅ Model loaded successfully!")

# 2. Prepare the model for LoRA fine-tuning
print("🔧 Setting up LoRA configuration...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✅ LoRA setup complete!")

# 3. Create a sample training dataset
print("📊 Creating sample dataset...")
sample_data = [
    {"text": "### Question: What is Python?\n### Answer: Python is a high-level programming language known for its simplicity and readability."},
    {"text": "### Question: How do you start learning to code?\n### Answer: Start with simple examples, practice regularly, and build small projects to reinforce your learning."},
    {"text": "### Question: What is machine learning?\n### Answer: Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."},
    {"text": "### Question: What is a function in programming?\n### Answer: A function is a reusable block of code that performs a specific task and can accept inputs and return outputs."},
    {"text": "### Question: Why is debugging important?\n### Answer: Debugging helps identify and fix errors in code, ensuring programs work correctly and efficiently."},
    {"text": "### Question: What is version control?\n### Answer: Version control systems like Git help track changes in code over time and enable collaboration among developers."},
]

dataset = Dataset.from_list(sample_data)
print(f"✅ Dataset created with {len(sample_data)} examples!")

# 4. Set up training configuration
print("⚙️ Setting up training configuration...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,  # Short training for demo - increase for better results
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="no",  # We'll save manually
        report_to="none",  # Disable wandb logging for simplicity
    ),
)

print("✅ Training setup complete!")

# 5. Start training
print("🏃‍♂️ Starting training...")
print("This will take a few minutes...")

trainer.train()

print("🎉 Training completed!")

# 6. Test the model before saving
print("🧪 Testing the fine-tuned model...")

# Enable inference mode
FastLanguageModel.for_inference(model)

# Test with a sample prompt
test_prompt = "### Question: What is debugging?\n### Answer:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print(f"Input prompt: {test_prompt}")
print("Generating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Model response: {response}")

# 7. Save to Hugging Face Hub
print("📤 Uploading to Hugging Face Hub...")
print("This may take a few minutes...")

try:
    # Upload the model to your HF account
    model.push_to_hub(
        "iwsordpress/claude-test", 
        tokenizer=tokenizer,
        use_temp_dir=False,
    )
    
    print("🎉 SUCCESS! Model uploaded to: https://huggingface.co/iwsordpress/claude-test")
    
except Exception as e:
    print(f"❌ Upload failed: {e}")
    print("Make sure you:")
    print("1. Have a valid Hugging Face token")
    print("2. Have write permissions to your account")
    print("3. Are connected to the internet")

# 8. Show how to load the model for inference later
print("\n" + "="*60)
print("📖 HOW TO USE YOUR MODEL FOR INFERENCE:")
print("="*60)

inference_code = '''
# Load your fine-tuned model for inference
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="iwsordpress/claude-test",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Generate text
prompt = "### Question: What is Python?\\n### Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
'''

print(inference_code)
print("="*60)
print("🎉 All done! Your model is ready to use!")
