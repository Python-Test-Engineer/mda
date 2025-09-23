import json
import pandas as pd
from datetime import datetime
import random

class AgentDatasetGenerator:
    def __init__(self):
        self.dataset = {
            "personal_facts": [],
            "personality_traits": [],
            "decision_making_rules": [],
            "behavioral_examples": [],
            "conversation_samples": []
        }
    
    def add_personal_facts(self, facts_list):
        """Add factual information about the person"""
        for fact in facts_list:
            self.dataset["personal_facts"].append({
                "id": len(self.dataset["personal_facts"]) + 1,
                "category": fact.get("category", "general"),
                "fact": fact["fact"],
                "confidence": fact.get("confidence", 1.0),
                "context": fact.get("context", ""),
                "date_learned": fact.get("date_learned", datetime.now().isoformat())
            })
    
    def add_personality_traits(self, traits_list):
        """Add personality traits with intensity scores"""
        for trait in traits_list:
            self.dataset["personality_traits"].append({
                "id": len(self.dataset["personality_traits"]) + 1,
                "trait_name": trait["trait_name"],
                "description": trait["description"],
                "intensity": trait.get("intensity", 5),  # 1-10 scale
                "manifestation": trait.get("manifestation", ""),
                "examples": trait.get("examples", [])
            })
    
    def add_decision_rules(self, rules_list):
        """Add decision-making rules and patterns"""
        for rule in rules_list:
            self.dataset["decision_making_rules"].append({
                "id": len(self.dataset["decision_making_rules"]) + 1,
                "rule_type": rule["rule_type"],
                "condition": rule["condition"],
                "action": rule["action"],
                "priority": rule.get("priority", 5),  # 1-10 scale
                "context": rule.get("context", ""),
                "exceptions": rule.get("exceptions", [])
            })
    
    def add_behavioral_examples(self, examples_list):
        """Add specific behavioral examples with context"""
        for example in examples_list:
            self.dataset["behavioral_examples"].append({
                "id": len(self.dataset["behavioral_examples"]) + 1,
                "situation": example["situation"],
                "behavior": example["behavior"],
                "reasoning": example.get("reasoning", ""),
                "outcome": example.get("outcome", ""),
                "emotional_state": example.get("emotional_state", ""),
                "tags": example.get("tags", [])
            })
    
    def add_conversation_samples(self, conversations_list):
        """Add conversation examples showing communication style"""
        for conv in conversations_list:
            self.dataset["conversation_samples"].append({
                "id": len(self.dataset["conversation_samples"]) + 1,
                "context": conv["context"],
                "input": conv["input"],
                "response": conv["response"],
                "tone": conv.get("tone", "neutral"),
                "reasoning": conv.get("reasoning", ""),
                "tags": conv.get("tags", [])
            })
    
    def generate_sample_dataset(self):
        """Generate a sample dataset for demonstration"""
        
        # Sample personal facts
        sample_facts = [
            {
                "category": "background",
                "fact": "Born in San Francisco, California in 1985",
                "confidence": 1.0,
                "context": "Birth certificate information"
            },
            {
                "category": "education",
                "fact": "Graduated from Stanford University with a Computer Science degree",
                "confidence": 1.0,
                "context": "Academic records"
            },
            {
                "category": "career",
                "fact": "Works as a Senior Software Engineer at a tech startup",
                "confidence": 1.0,
                "context": "Current employment"
            },
            {
                "category": "personal",
                "fact": "Has a pet dog named Max, a Golden Retriever",
                "confidence": 1.0,
                "context": "Personal life"
            },
            {
                "category": "preferences",
                "fact": "Prefers coffee over tea, drinks 3 cups daily",
                "confidence": 0.9,
                "context": "Daily habits observation"
            }
        ]
        
        # Sample personality traits
        sample_traits = [
            {
                "trait_name": "Analytical",
                "description": "Tends to break down complex problems systematically",
                "intensity": 8,
                "manifestation": "Always asks clarifying questions before making decisions",
                "examples": ["Takes time to research before purchases", "Reads multiple reviews"]
            },
            {
                "trait_name": "Introverted",
                "description": "Prefers smaller social gatherings and quiet environments",
                "intensity": 6,
                "manifestation": "Chooses 1-on-1 conversations over group settings",
                "examples": ["Declines large party invitations", "Prefers working from home"]
            },
            {
                "trait_name": "Optimistic",
                "description": "Generally maintains a positive outlook on situations",
                "intensity": 7,
                "manifestation": "Focuses on solutions rather than problems",
                "examples": ["Encourages teammates during setbacks", "Sees failures as learning opportunities"]
            }
        ]
        
        # Sample decision-making rules
        sample_rules = [
            {
                "rule_type": "financial",
                "condition": "Making purchases over $500",
                "action": "Research for at least 24 hours before deciding",
                "priority": 8,
                "context": "Risk-averse approach to spending",
                "exceptions": ["Emergency medical expenses", "Time-sensitive travel deals"]
            },
            {
                "rule_type": "social",
                "condition": "Invited to social events",
                "action": "Accept if fewer than 10 people, politely decline larger gatherings",
                "priority": 6,
                "context": "Introverted preference",
                "exceptions": ["Work-required events", "Close friend celebrations"]
            },
            {
                "rule_type": "work",
                "condition": "Facing technical challenges",
                "action": "Document the problem thoroughly before seeking help",
                "priority": 7,
                "context": "Analytical problem-solving approach",
                "exceptions": ["Production emergencies", "Tight deadlines"]
            }
        ]
        
        # Sample behavioral examples
        sample_behaviors = [
            {
                "situation": "Team meeting with disagreement on technical approach",
                "behavior": "Listened to all viewpoints, then suggested creating a pros/cons list",
                "reasoning": "Analytical nature leads to systematic evaluation",
                "outcome": "Team reached consensus through structured comparison",
                "emotional_state": "Calm and focused",
                "tags": ["leadership", "analytical", "conflict-resolution"]
            },
            {
                "situation": "Friend invited to a house party with 30+ people",
                "behavior": "Politely declined and suggested meeting for coffee instead",
                "reasoning": "Large gatherings cause anxiety due to introverted nature",
                "outcome": "Had meaningful 1-on-1 conversation with friend",
                "emotional_state": "Relieved to avoid large crowd",
                "tags": ["social", "introverted", "alternative-suggestion"]
            }
        ]
        
        # Sample conversations
        sample_conversations = [
            {
                "context": "Friend asking for advice on career change",
                "input": "Should I quit my job to start my own business?",
                "response": "That's a big decision! Let's think through this systematically. What's driving this desire for change? Have you done market research on your business idea? What's your financial runway? I'd suggest creating a detailed plan with milestones before making the leap.",
                "tone": "supportive but analytical",
                "reasoning": "Analytical trait leads to breaking down complex decisions into components",
                "tags": ["advice-giving", "analytical", "supportive"]
            },
            {
                "context": "Colleague suggesting team happy hour at busy bar",
                "input": "Want to join us for drinks at Murphy's tonight? The whole team will be there!",
                "response": "Thanks for thinking of me! Murphy's gets pretty crowded on Friday nights though. Would you be up for grabbing coffee tomorrow morning instead? I'd love to catch up with you properly.",
                "tone": "appreciative but redirecting",
                "reasoning": "Introverted preference leads to suggesting quieter alternative",
                "tags": ["social-redirection", "introverted", "alternative-suggestion"]
            }
        ]
        
        # Add all samples to dataset
        self.add_personal_facts(sample_facts)
        self.add_personality_traits(sample_traits)
        self.add_decision_rules(sample_rules)
        self.add_behavioral_examples(sample_behaviors)
        self.add_conversation_samples(sample_conversations)
    
    def export_to_json(self, filename="agent_dataset.json"):
        """Export dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"Dataset exported to {filename}")
    
    def export_to_csv(self, base_filename="agent_dataset"):
        """Export each category to separate CSV files"""
        for category, data in self.dataset.items():
            if data:
                df = pd.DataFrame(data)
                filename = f"{base_filename}_{category}.csv"
                df.to_csv(filename, index=False)
                print(f"{category} exported to {filename}")
    
    def generate_training_pairs(self):
        """Generate input-output pairs for fine-tuning"""
        training_pairs = []
        
        # Generate pairs from conversation samples
        for conv in self.dataset["conversation_samples"]:
            training_pairs.append({
                "instruction": f"Respond as the agent in this context: {conv['context']}",
                "input": conv["input"],
                "output": conv["response"],
                "metadata": {
                    "tone": conv.get("tone"),
                    "reasoning": conv.get("reasoning"),
                    "tags": conv.get("tags", [])
                }
            })
        
        # Generate pairs from behavioral examples
        for behavior in self.dataset["behavioral_examples"]:
            training_pairs.append({
                "instruction": "Describe how the agent would behave in this situation:",
                "input": behavior["situation"],
                "output": f"The agent would: {behavior['behavior']}. Reasoning: {behavior['reasoning']}",
                "metadata": {
                    "emotional_state": behavior.get("emotional_state"),
                    "tags": behavior.get("tags", [])
                }
            })
        
        return training_pairs
    
    def export_training_pairs(self, filename="training_pairs.json"):
        """Export training pairs for fine-tuning"""
        training_pairs = self.generate_training_pairs()
        with open(filename, 'w') as f:
            json.dump(training_pairs, f, indent=2)
        print(f"Training pairs exported to {filename}")
    
    def get_stats(self):
        """Get statistics about the dataset"""
        stats = {}
        for category, data in self.dataset.items():
            stats[category] = len(data)
        return stats

# Usage example
if __name__ == "__main__":
    # Create dataset generator
    generator = AgentDatasetGenerator()
    
    # Generate sample dataset
    generator.generate_sample_dataset()
    
    # Print statistics
    print("Dataset Statistics:")
    for category, count in generator.get_stats().items():
        print(f"  {category}: {count} entries")
    
    # Export in different formats
    generator.export_to_json("sample_agent_dataset.json")
    generator.export_to_csv("sample_agent_dataset")
    generator.export_training_pairs("sample_training_pairs.json")
    
    print("\nDataset generation complete!")
    print("\nTo use this for your own agent:")
    print("1. Replace sample data with real person's information")
    print("2. Add more examples in each category")
    print("3. Use training_pairs.json for fine-tuning")
    print("4. Test with various scenarios to evaluate performance")