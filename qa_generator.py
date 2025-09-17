"""
QA Dataset Generation for RAG System
"""

import json
import openai
from dotenv import load_dotenv
import os

class QAGenerator:
    """Generate QA pairs from research documents"""

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "gpt-3.5-turbo"

        self.prompts = {
            'factual': """Generate 5 factual questions and answers from this text in JSON format:
[{{"question": "What is...", "answer": "...", "type": "factual"}}, ...]

Text: {text}""",

            'inferential': """Generate 5 inferential questions and answers from this text in JSON format:
[{{"question": "Why does...", "answer": "...", "type": "inferential"}}, ...]

Text: {text}""",

            'analytical': """Generate 5 analytical questions and answers from this text in JSON format:
[{{"question": "How does... compare to...", "answer": "...", "type": "analytical"}}, ...]

Text: {text}"""
        }

    def generate_qa_for_text(self, text, doc_name, qa_type):
        """Generate QA pairs for a specific text chunk and type"""
        prompt = self.prompts[qa_type].format(text=text[:2000])

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )

            content = response.choices[0].message.content.strip()

            # Clean JSON response
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]

            qa_pairs = json.loads(content)

            # Add document metadata
            for qa in qa_pairs:
                qa['document'] = doc_name

            return qa_pairs

        except Exception as e:
            print(f"Error generating {qa_type} QAs: {e}")
            return []
    
    def generate_qa_dataset(self, documents_file="processed_documents.json"):
        """Generate complete QA dataset from all documents"""
        print("Generating QA Dataset...")

        with open(documents_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        all_qa_pairs = []

        for doc_name, doc_data in documents.items():
            print(f"Processing: {doc_name}")
            text = doc_data['cleaned_text']

            # Generate questions for each type
            for qa_type in ['factual', 'inferential', 'analytical']:
                qa_pairs = self.generate_qa_for_text(text, doc_name, qa_type)
                all_qa_pairs.extend(qa_pairs)
                print(f"  Generated {len(qa_pairs)} {qa_type} questions")

        qa_dataset = {
            'metadata': {
                'total_qa_pairs': len(all_qa_pairs),
                'documents_processed': len(documents)
            },
            'qa_pairs': all_qa_pairs
        }

        print(f"Total QA pairs: {len(all_qa_pairs)}")
        return qa_dataset
def main():
    """Generate QA dataset for the RAG system"""
    generator = QAGenerator()
    qa_dataset = generator.generate_qa_dataset()

    # Save dataset
    with open("qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, indent=2, ensure_ascii=False)

    print("QA dataset saved to: qa_dataset.json")

if __name__ == "__main__":
    main()
