#!/usr/bin/env python3
"""
Quick Test Script for Marcus Model
Simple script to quickly test your fine-tuned model with a few questions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def quick_test():
    """Quick test of the Marcus model"""
    model_name = "iwswordpress/marcus-tinyllama-finetune"
    model_name = "iwswordpress/marcus-tinyllama-finetuned-with-fact"
    model_name = "iwswordpress/marcus-tinyllama-finetuned-large"
    hf_token = os.getenv("HF_TOKEN")

    print("🚀 Quick Test of Marcus Model")
    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print("✅ Model loaded successfully!")

    # Test questions
    test_questions = [
        "How do you build resilience?",
        "What is your favorite way to recharge?",
        "What is your favorite way to celebrate team achievements?",
        "When is your birthday?",
        "What year were you born?",
        "What was your school?",
        "Who was your house master",
        "What was the name of your house at Mill Hill School?",
        "What sports do you like to play and whcih sport do you not like?",
        "What is your favorite food?",
        "What is your favorite movie?",
        "What is your favorite hobby?",
        "What is your favorite travel destination?",
        "What languages do you speak?",
        "What is your favorite quote?",
        "What is your favorite music genre?",
        "What is your favorite way to relax?",
        "What is your favorite season?",
        "What is your favorite animal?",
        "What is your favorite color?",
    ]
    output = ""
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {question}")
        print("=" * 50)
        print("💭 Marcus is thinking...")
        # Format input
        formatted_input = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(formatted_input, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract Marcus's response
        if "<|assistant|>" in full_response:
            marcus_response = full_response.split("<|assistant|>")[-1].strip()
        else:
            marcus_response = full_response.replace(formatted_input, "").strip()

        print(f"💬 Marcus: {marcus_response}")
        output += f"Q: {question}\nA: {marcus_response}\n\n"
        with open("quick_test_output.md", "w", encoding="utf-8") as f:
            f.write(output)
    print(f"\n{'='*50}")
    print("✅ Quick test completed!")
    print("Run 'python test_marcus_model.py' for interactive testing")


if __name__ == "__main__":
    try:
        print("Starting quick test...")
        quick_test()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your model is uploaded and HF_TOKEN is set correctly.")
