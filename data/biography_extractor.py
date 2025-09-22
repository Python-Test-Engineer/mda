import os
from dotenv import load_dotenv, find_dotenv
import openai
import csv
import json
import time
from typing import List, Dict, Any
import re

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"


class BiographyExtractor:
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = MODEL):
        """
        Initialize the biography extractor with OpenAI API

        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI model to use for extraction
        """
        openai.api_key = api_key
        self.model = model
        self.biography_text = ""
        self.extracted_facts = []

    def load_biography(self, biography_text: str):
        """Load the biography text for processing"""
        self.biography_text = biography_text

    def extract_facts_batch(self, category: str, extraction_prompt: str) -> List[Dict]:
        """
        Extract facts from biography for a specific category

        Args:
            category (str): The category of facts to extract
            extraction_prompt (str): Specific prompt for this category

        Returns:
            List[Dict]: List of extracted facts
        """

        full_prompt = f"""
        Extract information from the following biography for the category: {category}

        Biography:
        {self.biography_text}

        {extraction_prompt}

        Return the results as a valid JSON array where each object has these fields:
        - "fact": The specific factual statement
        - "category": "{category}"
        - "confidence": A number between 0.0 and 1.0 indicating confidence
        - "evidence": The specific text from the biography that supports this fact
        - "importance": "high", "medium", or "low"
        - "verifiable": true or false

        IMPORTANT: Return ONLY the JSON array, no other text.
        """

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from biographical text. Always return valid JSON.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content
            # Clean up the response to ensure it's valid JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            facts = json.loads(content)
            return facts

        except Exception as e:
            print(f"Error extracting facts for category {category}: {e}")
            return []

    def extract_all_categories(self) -> List[Dict]:
        """Extract facts across all relevant categories"""

        extraction_categories = [
            {
                "category": "personal_demographics",
                "prompt": """Extract basic demographic and personal information including:
                - Birth date, age, birthplace
                - Family members and relationships
                - Physical characteristics if mentioned
                - Cultural background and languages""",
            },
            {
                "category": "education",
                "prompt": """Extract educational background including:
                - Schools attended with dates
                - Degrees earned
                - Academic achievements and honors
                - Significant professors or mentors
                - Thesis topics or research areas""",
            },
            {
                "category": "career_professional",
                "prompt": """Extract career and professional information including:
                - Job positions and employers
                - Research areas and specializations
                - Publications, patents, or professional achievements
                - Professional transitions and career moves""",
            },
            {
                "category": "personality_traits",
                "prompt": """Extract personality characteristics including:
                - Work style and approach to problems
                - Social preferences and interpersonal style
                - Described character traits
                - Behavioral patterns and tendencies""",
            },
            {
                "category": "interests_hobbies",
                "prompt": """Extract interests and hobbies including:
                - Personal interests and pastimes
                - Skills or activities outside of work
                - Collections or specialized knowledge areas
                - Recreational activities""",
            },
            {
                "category": "relationships_social",
                "prompt": """Extract relationship and social information including:
                - Marriage and romantic relationships
                - Children and family dynamics
                - Professional relationships and mentoring
                - Social preferences and interaction patterns""",
            },
            {
                "category": "values_beliefs",
                "prompt": """Extract values and beliefs including:
                - Professional ethics and principles
                - Personal values and priorities
                - Beliefs about science, life, or society
                - Motivations and driving principles""",
            },
            {
                "category": "communication_style",
                "prompt": """Extract communication and interaction patterns including:
                - Speaking or writing style
                - How they interact with students or colleagues
                - Communication preferences
                - Language use and expression patterns""",
            },
            {
                "category": "decision_making",
                "prompt": """Extract decision-making patterns including:
                - How they approach major life or career decisions
                - Problem-solving methodology
                - Risk tolerance and decision criteria
                - Examples of significant choices made""",
            },
            {
                "category": "current_status",
                "prompt": """Extract current life status including:
                - Current position and responsibilities
                - Ongoing projects or research
                - Recent developments or changes
                - Current family or personal situation""",
            },
        ]

        all_facts = []

        for category_info in extraction_categories:
            print(f"Extracting facts for category: {category_info['category']}")

            facts = self.extract_facts_batch(
                category_info["category"], category_info["prompt"]
            )

            all_facts.extend(facts)

            # Add delay to respect API rate limits
            time.sleep(1)

        return all_facts

    def generate_training_pairs(self, facts: List[Dict]) -> List[Dict]:
        """Generate training pairs from extracted facts for fine-tuning"""

        prompt = f"""
        Based on these facts about Dr. Marcus Chen, generate conversation training pairs.
        
        Facts: {json.dumps(facts[:20], indent=2)}  # Sample facts to avoid token limits
        
        Generate 50-100 training pairs in this format:
        [
            {{
                "messages": [
                    {{"role": "user", "content": "Question about Dr. Chen"}},
                    {{"role": "assistant", "content": "Response as if you are Dr. Chen or representing his knowledge/personality"}}
                ]
            }}
        ]
        
        Include various question types:
        - Direct biographical questions
        - Questions about his research or expertise
        - Questions about his opinions or approach to problems
        - Questions about his experiences or stories
        
        Make the responses authentic to his personality and background.
        Return ONLY the JSON array.
        """

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate training data for fine-tuning based on biographical facts.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=4000,
            )

            content = response.choices[0].message.content
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            training_pairs = json.loads(content)
            return training_pairs

        except Exception as e:
            print(f"Error generating training pairs: {e}")
            return []

    def save_to_csv(self, facts: List[Dict], filename: str = "bio.csv"):
        """Save extracted facts to CSV file"""

        if not facts:
            print("No facts to save")
            return

        # Define CSV columns
        fieldnames = [
            "id",
            "category",
            "fact",
            "confidence",
            "evidence",
            "importance",
            "verifiable",
            "extraction_source",
        ]

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, fact in enumerate(facts, 1):
                writer.writerow(
                    {
                        "id": i,
                        "category": fact.get("category", "unknown"),
                        "fact": fact.get("fact", ""),
                        "confidence": fact.get("confidence", 0.5),
                        "evidence": fact.get("evidence", ""),
                        "importance": fact.get("importance", "medium"),
                        "verifiable": fact.get("verifiable", False),
                        "extraction_source": "openai_gpt4",
                    }
                )

        print(f"Saved {len(facts)} facts to {filename}")

    def save_training_data(
        self, training_pairs: List[Dict], filename: str = "bio_training_data.jsonl"
    ):
        """Save training pairs in JSONL format for fine-tuning"""

        with open(filename, "w", encoding="utf-8") as f:
            for pair in training_pairs:
                f.write(json.dumps(pair) + "\n")

        print(f"Saved {len(training_pairs)} training pairs to {filename}")

    def run_full_extraction(self, biography_text: str, output_csv: str = "bio.csv"):
        """Run the complete extraction pipeline"""

        print("Starting biography information extraction...")

        # Load biography
        self.load_biography(biography_text)

        # Extract facts
        print("Extracting facts across all categories...")
        facts = self.extract_all_categories()

        if not facts:
            print("No facts were extracted. Please check your API key and try again.")
            return [], []

        # Save facts to CSV
        print("Saving facts to CSV...")
        self.save_to_csv(facts, output_csv)

        # Generate and save training pairs
        print("Generating training pairs...")
        training_pairs = self.generate_training_pairs(facts)
        if training_pairs:
            self.save_training_data(training_pairs)

        # Generate summary report
        print("Generating summary report...")
        self.generate_summary_report(facts, training_pairs)

        print("\n=== EXTRACTION COMPLETE ===")
        print(f"Files generated:")
        print(f"- {output_csv}: {len(facts)} extracted facts")
        print(f"- training_data.jsonl: {len(training_pairs)} training pairs")
        print(f"- extraction_summary.md: Analysis summary")

        return facts, training_pairs

    def generate_summary_report(self, facts: List[Dict], training_pairs: List[Dict]):
        """Generate a summary report of the extraction"""

        print("Creating summary report...")

        # Analyze facts by category
        category_counts = {}
        confidence_levels = {"high": 0, "medium": 0, "low": 0}
        importance_levels = {"high": 0, "medium": 0, "low": 0}
        verifiable_count = {"yes": 0, "no": 0}

        for fact in facts:
            cat = fact.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            conf = fact.get("confidence", 0.5)
            if conf >= 0.8:
                confidence_levels["high"] += 1
            elif conf >= 0.6:
                confidence_levels["medium"] += 1
            else:
                confidence_levels["low"] += 1

            imp = fact.get("importance", "medium")
            importance_levels[imp] = importance_levels.get(imp, 0) + 1

            verifiable = fact.get("verifiable", False)
            if verifiable:
                verifiable_count["yes"] += 1
            else:
                verifiable_count["no"] += 1

        # Create markdown content
        summary_content = []
        summary_content.append("# Dr. Marcus Chen - Biography Extraction Summary")
        summary_content.append(f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        summary_content.append("## Extraction Results")
        summary_content.append(f"- **Total Facts Extracted:** {len(facts)}")
        summary_content.append(f"- **Training Pairs Generated:** {len(training_pairs)}")
        summary_content.append(f"- **Extraction Method:** OpenAI GPT-4 API\n")

        summary_content.append("## Facts by Category")
        for cat, count in sorted(category_counts.items()):
            formatted_cat = cat.replace("_", " ").title()
            summary_content.append(f"- **{formatted_cat}:** {count} facts")
        summary_content.append("")

        summary_content.append("## Quality Metrics")
        summary_content.append("### Confidence Distribution")
        for level, count in confidence_levels.items():
            percentage = (count / len(facts) * 100) if facts else 0
            summary_content.append(
                f"- **{level.title()} Confidence (≥{0.8 if level=='high' else 0.6 if level=='medium' else 0.0}):** {count} facts ({percentage:.1f}%)"
            )
        summary_content.append("")

        summary_content.append("### Importance Distribution")
        for level, count in importance_levels.items():
            percentage = (count / len(facts) * 100) if facts else 0
            summary_content.append(
                f"- **{level.title()} Importance:** {count} facts ({percentage:.1f}%)"
            )
        summary_content.append("")

        summary_content.append("### Verifiability")
        summary_content.append(
            f"- **Verifiable Facts:** {verifiable_count['yes']} ({verifiable_count['yes']/len(facts)*100:.1f}%)"
        )
        summary_content.append(
            f"- **Subjective/Opinion Facts:** {verifiable_count['no']} ({verifiable_count['no']/len(facts)*100:.1f}%)\n"
        )

        summary_content.append("## Sample Extracted Facts")
        # Show top 5 high-confidence facts
        high_conf_facts = [f for f in facts if f.get("confidence", 0) >= 0.8][:5]
        for i, fact in enumerate(high_conf_facts, 1):
            summary_content.append(f"**{i}.** {fact.get('fact', 'N/A')}")
            summary_content.append(
                f"   - *Category:* {fact.get('category', 'unknown')}"
            )
            summary_content.append(f"   - *Confidence:* {fact.get('confidence', 0)}")
            summary_content.append("")

        summary_content.append("## Generated Files")
        summary_content.append(
            "- **bio.csv** - Structured facts in CSV format for analysis"
        )
        summary_content.append(
            "- **training_data.jsonl** - Training pairs in OpenAI format for fine-tuning"
        )
        summary_content.append(
            "- **extraction_summary.md** - This comprehensive analysis report\n"
        )

        summary_content.append("## Recommendations for Fine-Tuning")
        summary_content.append(
            "1. **Review high-confidence facts** for accuracy before training"
        )
        summary_content.append(
            "2. **Expand training pairs** with additional conversation scenarios"
        )
        summary_content.append(
            "3. **Balance fact types** - ensure mix of biographical and personality data"
        )
        summary_content.append(
            "4. **Validate model responses** against extracted personality traits"
        )
        summary_content.append(
            "5. **Consider temporal context** - Dr. Chen's views may have evolved over time"
        )

        # Write summary to file
        try:
            with open("bio_extraction_summary.md", "w", encoding="utf-8") as f:
                f.write("\n".join(summary_content))
            print("✓ Summary report saved to extraction_summary.md")
        except Exception as e:
            print(f"✗ Error saving summary report: {e}")
            # Still return the content for debugging
            return "\n".join(summary_content)

        return "\n".join(summary_content)


# Example usage
if __name__ == "__main__":
    # Dr. Marcus Chen's biography (from the previous artifact)
    with open("bio_marcus_chen.md", "r", encoding="utf-8") as f:
        biography_text = f.read()

    # Initialize extractor (you need to provide your OpenAI API key)
    # API_KEY = OPENAI_API_KEY  # Replace with your actual API key

    extractor = BiographyExtractor(OPENAI_API_KEY)
    facts, training_pairs = extractor.run_full_extraction(biography_text)

    print("Biography extraction code ready!")
    print("\nThis will generate:")
    print("- bio.csv: Structured facts for analysis")
    print("- training_data.jsonl: Training pairs for fine-tuning")
    print("- extraction_summary.md: Analysis summary")
