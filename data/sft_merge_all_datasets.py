import json

# Input files
instruction_file = "marcus_instruction_dataset.jsonl"
chat_file = "marcus_chat_dataset.jsonl"
decision_file = (
    "marcus_decision_dataset_ready.jsonl"  # already processed with previous script
)

# Output file
merged_file = "marcus_multidataset_ready.jsonl"

merged_data = []


# Helper function to load JSONL
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Load datasets
instruction_data = load_jsonl(instruction_file)
chat_data = load_jsonl(chat_file)
decision_data = load_jsonl(decision_file)

# Standardize chat dataset to instruction/output format
for entry in chat_data:
    messages = entry.get("messages", [])
    if not messages:
        continue
    user_msg = [m["content"] for m in messages if m["role"] == "user"][-1]
    assistant_msg = [m["content"] for m in messages if m["role"] == "assistant"][-1]
    merged_data.append(
        {
            "instruction": f"User: {user_msg}",
            "output": f"Marcus: {assistant_msg}",
            "tags": ["chat"],
        }
    )

# Add instruction dataset
for entry in instruction_data:
    merged_data.append(
        {
            "instruction": entry.get("instruction", ""),
            "output": entry.get("output", ""),
            "tags": entry.get("tags", ["instruction"]),
        }
    )

# Add decision-making dataset
for entry in decision_data:
    merged_data.append(
        {
            "instruction": entry.get("instruction", ""),
            "output": entry.get("output", ""),
            "tags": entry.get("tags", ["decision"]),
        }
    )

# Shuffle dataset (optional)
import random

random.shuffle(merged_data)

# Save merged JSONL
with open(merged_file, "w", encoding="utf-8") as f:
    for item in merged_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved merged dataset to {merged_file} ({len(merged_data)} entries)")
