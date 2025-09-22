#!/usr/bin/env python3
"""
Test Script for Fine-tuned Marcus Model
Load and interact with your fine-tuned model from Hugging Face Hub
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class MarcusModelTester:
    def __init__(self, model_name="iwswordpress/marcus-tinyllama-finetune"):
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        
        print(f"Loading fine-tuned model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def ask_marcus(self, question, max_new_tokens=150, temperature=0.7):
        """Ask Marcus a question and get his response"""
        
        # Format the input using the same chat template as training
        formatted_input = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        
        # Generate response
        print(f"\n🤔 Question: {question}")
        print("💭 Marcus is thinking...")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just Marcus's response (after <|assistant|>)
        if "<|assistant|>" in full_response:
            marcus_response = full_response.split("<|assistant|>")[-1].strip()
        else:
            marcus_response = full_response.replace(formatted_input, "").strip()
        
        print(f"💬 Marcus: {marcus_response}")
        return marcus_response

    def interactive_chat(self):
        """Start an interactive chat session with Marcus"""
        print("\n" + "="*60)
        print("🎯 Interactive Chat with Marcus")
        print("="*60)
        print("Ask Marcus about leadership, productivity, or life advice!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("-"*60)
        
        while True:
            try:
                question = input("\n🙋 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Marcus: Thanks for the great conversation! Keep growing and leading!")
                    break
                
                if not question:
                    print("Please ask a question!")
                    continue
                
                self.ask_marcus(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Marcus: Thanks for the chat! Take care!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    def run_example_tests(self):
        """Run some example questions to test the model"""
        print("\n" + "="*60)
        print("🧪 Testing Marcus with Example Questions")
        print("="*60)
        
        example_questions = [
            "What is your philosophy on leadership?",
            "How do you handle difficult situations?",
            "What advice would you give to someone starting their career?",
            "How do you stay motivated during challenging times?",
            "What's your approach to building trust in a team?",
            "How do you balance work and personal life?",
            "What role does empathy play in leadership?",
            "How do you encourage innovation in your team?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            print(f"\n--- Test {i}/{len(example_questions)} ---")
            self.ask_marcus(question)
            
            # Add a small pause between questions
            if i < len(example_questions):
                input("\nPress Enter to continue to next question...")

def main():
    """Main function"""
    print("🚀 Marcus Model Tester")
    print("Loading your fine-tuned leadership assistant...")
    
    try:
        # Initialize the model tester
        tester = MarcusModelTester()
        
        # Show menu
        while True:
            print("\n" + "="*50)
            print("📋 What would you like to do?")
            print("="*50)
            print("1. Run example tests")
            print("2. Interactive chat with Marcus")
            print("3. Ask a single question")
            print("4. Exit")
            print("-"*50)
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                tester.run_example_tests()
            elif choice == '2':
                tester.interactive_chat()
            elif choice == '3':
                question = input("\n🙋 Ask Marcus: ").strip()
                if question:
                    tester.ask_marcus(question)
            elif choice == '4':
                print("\n👋 Goodbye! Thanks for testing Marcus!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure your model is uploaded to HuggingFace Hub and your HF_TOKEN is valid.")

if __name__ == "__main__":
    main()
