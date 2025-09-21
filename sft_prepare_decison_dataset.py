import json

"""
{
  "scenario": "Choosing whether to delegate a high-stakes task to a junior team member.",
  "context": "The junior has potential but limited experience.",
  "decision": "I would delegate with guidance and support.",
  "reasoning": "Empowering team members builds growth and confidence. I would provide oversight and mentoring to ensure success while developing talent.",
  "tags": ["leadership", "team development", "mentorship"]
}


{
  "instruction": "Scenario: [scenario]. Context: [context]. Make a decision and explain your reasoning.",
  "output": "Decision: [decision]. Reasoning: [reasoning]"
}

FINAL FORMAT EXAMPLE:
{
  "instruction": "Scenario: You must decide whether to fire an underperforming team member who usually contributes well. Context: The mistake was serious but not malicious. The person has a history of strong performance. Tags: leadership, ethics, team management Make a decision and explain your reasoning.",
  "output": "Decision: I would prioritize coaching over immediate termination. Reasoning: People deserve a chance to grow from setbacks. I would first understand the circumstances, discuss the mistake openly, and assess willingness to improve. This approach maintains fairness, trust, and long-term team strength.",
  "tags": ["leadership", "ethics", "team management"]
}

"""
# Input file: your starter or expanded decision dataset
input_file = "marcus_decision_dataset.jsonl"
# Output file: training-ready instruction→response format
output_file = "marcus_decision_dataset_ready.jsonl"

ready_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)

        # Build instruction string
        scenario = entry.get("scenario", "")
        context = entry.get("context", "")
        tags = entry.get("tags", [])
        instruction = f"Scenario: {scenario}"
        if context:
            instruction += f" Context: {context}"
        if tags:
            instruction += f" Tags: {', '.join(tags)}"
        instruction += " Make a decision and explain your reasoning."

        # Build output string
        decision = entry.get("decision", "")
        reasoning = entry.get("reasoning", "")
        output = f"Decision: {decision}. Reasoning: {reasoning}"

        ready_data.append({"instruction": instruction, "output": output, "tags": tags})

# Save to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for item in ready_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved processed training dataset to {output_file}")
