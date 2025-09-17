"""
Simple Document Processing Module for RAG System
Handles PDF extraction, text cleaning, and basic analysis
"""

import re
import json
from pathlib import Path
from typing import Dict, Any

# PDF Processing - using only PyMuPDF (most reliable for academic papers)
import fitz  # PyMuPDF

# Text Analysis
import nltk
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DocumentProcessor:
    """
    Simple document processor for RAG system
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_docs = {}

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using PyMuPDF
        Why PyMuPDF? It's reliable, handles academic papers well, and preserves formatting
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"❌ Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        Why these steps? Remove PDF artifacts and normalize formatting for better chunking
        """
        if not text:
            return ""

        # Step 1: Normalize whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Fix paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Fix multiple spaces

        # Step 2: Remove PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        text = re.sub(r'�', '', text)              # Remove replacement chars

        # Step 3: Remove headers/footers/page numbers
        text = re.sub(r'\n\d+\n', '\n', text)     # Page numbers
        text = re.sub(r'\n[A-Z\s]{10,}\n', '\n', text)  # Headers

        # Step 4: Fix spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # After punctuation

        # Step 5: Final cleanup
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def get_basic_stats(self, text: str, doc_name: str) -> Dict[str, Any]:
        """
        Get basic document statistics - simple and focused
        """
        if not text:
            return {"error": "Empty document"}

        # Basic counts
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)

        # Vocabulary analysis
        unique_words = set(word.lower() for word in words if word.isalpha())
        vocabulary_richness = len(unique_words) / max(word_count, 1)

        # Readability (simple measure)
        avg_sentence_length = word_count / max(sentence_count, 1)
        flesch_score = textstat.flesch_reading_ease(text)

        # Quality check
        quality_issues = []
        if word_count < 1000:
            quality_issues.append("Document too short")
        if word_count > 50000:
            quality_issues.append("Document very long")
        if vocabulary_richness < 0.3:
            quality_issues.append("Low vocabulary diversity")
        if avg_sentence_length > 30:
            quality_issues.append("Very long sentences")

        return {
            "document_name": doc_name,
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "unique_words": len(unique_words),
            "vocabulary_richness": round(vocabulary_richness, 3),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "flesch_reading_ease": round(flesch_score, 1),
            "quality_issues": quality_issues
        }

    def process_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single document: extract, clean, and analyze
        Simple 3-step process: Extract → Clean → Analyze
        """
        print(f"\n📄 Processing: {pdf_path.name}")

        # Step 1: Extract text from PDF
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            print(f"❌ Failed to extract text from {pdf_path.name}")
            return {"error": "Text extraction failed"}

        print(f"✅ Extracted {len(raw_text)} characters")

        # Step 2: Clean the text
        cleaned_text = self.clean_text(raw_text)
        print(f"🧹 Cleaned text: {len(cleaned_text)} characters")

        # Step 3: Get basic statistics
        stats = self.get_basic_stats(cleaned_text, pdf_path.stem)
        print(f"📊 Analysis: {stats['word_count']} words, {stats['sentence_count']} sentences")

        if stats.get('quality_issues'):
            print(f"⚠️  Issues: {', '.join(stats['quality_issues'])}")

        return {
            "file_path": str(pdf_path),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "stats": stats
        }

    def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all PDF documents in the data directory
        Simple: Find PDFs → Process each → Return results
        """
        pdf_files = list(self.data_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"❌ No PDF files found in {self.data_dir}")
            return {}

        print(f"🔍 Found {len(pdf_files)} PDF files")

        results = {}
        for pdf_path in pdf_files:
            result = self.process_document(pdf_path)
            if "error" not in result:
                results[pdf_path.stem] = result
                self.processed_docs[pdf_path.stem] = result

        print(f"\n✅ Successfully processed {len(results)} documents")
        return results

    def save_processed_documents(self, output_file: str = "processed_documents.json"):
        """
        Save processed documents to JSON file for later use
        """
        if not self.processed_docs:
            print("No documents to save. Process documents first.")
            return

        # Prepare data for JSON (remove raw text to save space)
        save_data = {}
        for doc_name, doc_data in self.processed_docs.items():
            save_data[doc_name] = {
                "file_path": doc_data["file_path"],
                "cleaned_text": doc_data["cleaned_text"],
                "stats": doc_data["stats"]
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved processed documents to {output_file}")

    def print_summary(self):
        """
        Print a simple summary of processed documents
        """
        if not self.processed_docs:
            print("No documents processed yet.")
            return

        print("\n" + "="*50)
        print("📊 DOCUMENT PROCESSING SUMMARY")
        print("="*50)

        total_words = 0
        total_chars = 0

        for doc_name, doc_data in self.processed_docs.items():
            stats = doc_data["stats"]
            total_words += stats["word_count"]
            total_chars += stats["character_count"]

            print(f"\n📄 {doc_name}")
            print(f"   Words: {stats['word_count']:,}")
            print(f"   Sentences: {stats['sentence_count']:,}")
            print(f"   Readability: {stats['flesch_reading_ease']}")
            print(f"   Vocabulary richness: {stats['vocabulary_richness']}")

            if stats['quality_issues']:
                print(f"   ⚠️  Issues: {', '.join(stats['quality_issues'])}")

        print(f"\n📈 TOTALS:")
        print(f"   Documents: {len(self.processed_docs)}")
        print(f"   Total words: {total_words:,}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average words per doc: {total_words // len(self.processed_docs):,}")
        print("="*50)
