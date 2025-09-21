from datasets import load_dataset

# 1. Load your mixed dataset (instruction + chat style)
dataset = load_dataset("json", data_files="marcus_dataset.jsonl")["train"]


# 2. Convert everything into prompt → response format
def format_example(example):
    if "instruction" in example:  # Instruction style
        # Combine instruction + optional input
        prompt = example["instruction"]
        if example.get("input"):
            prompt += f"\n{example['input']}"
        response = example["output"]

    elif "messages" in example:  # Chat style
        # Extract last user & assistant message
        user_msg = [m["content"] for m in example["messages"] if m["role"] == "user"][
            -1
        ]
        assistant_msg = [
            m["content"] for m in example["messages"] if m["role"] == "assistant"
        ][-1]
        prompt, response = user_msg, assistant_msg

    else:
        prompt, response = "", ""

    return {"prompt": prompt, "response": response}


processed = dataset.map(format_example)

# 3. Save processed dataset to JSONL
processed.to_json("marcus_dataset_processed.jsonl")
print("✅ Saved processed dataset to marcus_dataset_processed.jsonl")
