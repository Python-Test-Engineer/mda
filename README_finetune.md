# Simple LLM Fine-tuning Script

This script provides a simple way to fine-tune an open source LLM using your JSONL dataset. It uses **TinyLlama-1.1B** (one of the smallest suitable models) with **LoRA** (Low-Rank Adaptation) for efficient training.

## Features

- ✅ Uses the smallest practical LLM (TinyLlama-1.1B)
- ✅ Efficient LoRA fine-tuning (only trains ~0.1% of parameters)
- ✅ Automatic dataset processing from JSONL format
- ✅ Built-in model testing
- ✅ Automatic upload to Hugging Face Hub
- ✅ Works on both CPU and GPU

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Your Environment

Make sure your `.env` file contains:
```
HF_TOKEN=your_huggingface_token_here
```

### 3. Run the Fine-tuning

```bash
python simple_finetune.py
```

The script will:
1. Load TinyLlama-1.1B model
2. Process your `sft_marcus_lite.jsonl` file
3. Fine-tune the model with LoRA
4. Test the model with a sample prompt
5. Optionally upload to Hugging Face Hub

## Your Dataset

Your JSONL file (`sft_marcus_lite.jsonl`) contains 90 prompt-response pairs in the format:
```json
{"prompt": "What is your philosophy on leadership?", "response": "Leadership means serving first, guiding with clarity, and empowering others."}
```

The script automatically formats these as chat conversations for training.

## Training Configuration

- **Model**: TinyLlama-1.1B-Chat-v1.0 (~1.1B parameters)
- **Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: ~8M (only 0.7% of total model)
- **Training Steps**: 500 (quick training)
- **Batch Size**: 2 per device
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 512 tokens

## Output

The fine-tuned model will be saved to `./marcus-tinyllama-finetuned/` and can be uploaded to your Hugging Face organization `iwswordpress`.

## Hardware Requirements

- **Minimum**: 8GB RAM (CPU training)
- **Recommended**: 6GB+ VRAM GPU for faster training
- **Training Time**: 
  - CPU: ~30-60 minutes
  - GPU: ~10-20 minutes

## Customization

You can modify the script to:
- Use a different base model
- Adjust training parameters
- Change the output directory
- Modify the chat template format

## Testing Your Model

After training, the script automatically tests the model with a sample prompt. You can also test it manually:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./marcus-tinyllama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "What is your approach to problem-solving?"
formatted_input = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
inputs = tokenizer(formatted_input, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce `per_device_train_batch_size` to 1
2. **Slow Training**: Install CUDA-compatible PyTorch for GPU acceleration
3. **Token Issues**: Ensure your HF_TOKEN has write permissions

### Performance Tips:

- Use GPU if available (much faster)
- Install `bitsandbytes` for 8-bit training
- Increase batch size if you have more memory

## Next Steps

After fine-tuning, you can:
1. Upload to Hugging Face Hub for sharing
2. Use the model in your applications
3. Further fine-tune on additional data
4. Evaluate performance on test sets

## File Structure

```
.
├── simple_finetune.py          # Main fine-tuning script
├── requirements.txt            # Python dependencies
├── sft_marcus_lite.jsonl      # Your training data
├── .env                       # Environment variables
└── marcus-tinyllama-finetuned/ # Output directory (created after training)
