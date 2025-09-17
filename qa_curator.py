"""
QA Dataset Curation and Quality Control
"""

import json
import random
from pathlib import Path

class QACurator:
    """Curate and quality control QA dataset"""

    def __init__(self, qa_dataset_file="qa_dataset.json"):
        with open(qa_dataset_file, 'r', encoding='utf-8') as f:
            self.qa_dataset = json.load(f)
        self.qa_pairs = self.qa_dataset['qa_pairs']
        print(f"Loaded {len(self.qa_pairs)} QA pairs for curation")

    def quality_filter(self, qa_pairs):
        """Filter QA pairs based on quality criteria"""
        print("Applying quality filters...")

        filtered_pairs = []
        removed_count = 0
        seen_questions = set()

        for qa in qa_pairs:
            if not all(key in qa for key in ['question', 'answer', 'type']):
                removed_count += 1
                continue

            question = qa['question'].strip()
            answer = qa['answer'].strip()

            # Remove short questions/answers or duplicates
            if (len(question) < 20 or len(answer) < 30 or
                question.lower() in seen_questions or
                qa['type'] not in ['factual', 'inferential', 'analytical']):
                removed_count += 1
                continue

            seen_questions.add(question.lower())
            filtered_pairs.append(qa)

        print(f"Filtered {len(qa_pairs)} -> {len(filtered_pairs)} pairs (removed {removed_count})")
        return filtered_pairs
    
    def split_dataset(self, qa_pairs):
        """Split dataset into train/validation/test sets"""
        print("Splitting dataset (70/15/15)...")

        random.shuffle(qa_pairs)
        total = len(qa_pairs)
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)

        splits = {
            'train': qa_pairs[:train_size],
            'validation': qa_pairs[train_size:train_size + val_size],
            'test': qa_pairs[train_size + val_size:]
        }

        print(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        return splits

    def curate_dataset(self):
        """Complete curation process"""
        print("Starting QA Dataset Curation...")

        # Filter quality
        filtered_pairs = self.quality_filter(self.qa_pairs)

        # Split dataset
        splits = self.split_dataset(filtered_pairs)

        # Create final dataset
        curated_dataset = {
            'metadata': {
                'total_qa_pairs': len(filtered_pairs),
                'splits': {split: len(pairs) for split, pairs in splits.items()}
            },
            'splits': splits
        }

        return curated_dataset

def main():
    """Run QA curation process"""
    curator = QACurator()
    curated_dataset = curator.curate_dataset()

    # Save curated dataset
    with open('curated_qa_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(curated_dataset, f, indent=2, ensure_ascii=False)

    # Generate simple quality report
    total_pairs = curated_dataset['metadata']['total_qa_pairs']
    splits = curated_dataset['metadata']['splits']

    report = f"""# Data Quality Assessment Report

## QA Dataset for RAG System

### Dataset Overview
- **Total QA Pairs**: {total_pairs}
- **Train Split**: {splits['train']} pairs
- **Validation Split**: {splits['validation']} pairs
- **Test Split**: {splits['test']} pairs

### Quality Control
The dataset underwent quality filtering to remove:
- Short questions (< 20 characters)
- Short answers (< 30 characters)
- Duplicate questions
- Invalid question types

### Conclusion
The curated QA dataset is ready for use in evaluating RAG systems.
"""

    # Save quality report
    Path('reports').mkdir(exist_ok=True)
    with open('reports/qa_dataset_quality_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"QA Dataset Curation Complete!")
    print(f"Final dataset: {total_pairs} QA pairs")
    print(f"Saved: curated_qa_dataset.json")
    print(f"Report: reports/qa_dataset_quality_report.md")

if __name__ == "__main__":
    main()
