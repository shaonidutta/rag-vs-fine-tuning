"""
Text Chunking Module for RAG System
Implements intelligent chunking with optimal size and overlap
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path

class TextChunker:
    """
    Simple and effective text chunker for RAG system
    """
    
    def __init__(self, chunk_size: int = 800, overlap_size: int = 200):
        """
        Initialize chunker with optimal parameters
        
        Why these defaults?
        - 800 chars ≈ 150-200 words (good context size)
        - 200 char overlap (25%) prevents information loss
        - Works well with academic papers
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.chunks = []
        
    def chunk_text(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Strategy:
        1. Try to break at sentence boundaries when possible
        2. Use character-based splitting as fallback
        3. Maintain overlap to preserve context
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is the last chunk, take everything
            if end >= len(text):
                chunk_text = text[start:].strip()
                if chunk_text:  # Only add if not empty
                    chunks.append({
                        'chunk_id': f"{doc_name}_chunk_{chunk_id}",
                        'document': doc_name,
                        'text': chunk_text,
                        'start_pos': start,
                        'end_pos': len(text),
                        'size': len(chunk_text)
                    })
                break
            
            # Try to find a good breaking point (sentence end)
            chunk_text = text[start:end]
            
            # Look for sentence boundaries in the last 100 characters
            search_start = max(0, len(chunk_text) - 100)
            sentence_ends = []
            
            # Find sentence endings
            for match in re.finditer(r'[.!?]\s+', chunk_text[search_start:]):
                sentence_ends.append(search_start + match.end())
            
            # If we found sentence boundaries, use the last one
            if sentence_ends:
                actual_end = start + sentence_ends[-1]
                chunk_text = text[start:actual_end].strip()
            else:
                # Fallback: try to break at word boundary
                words = chunk_text.split()
                if len(words) > 1:
                    # Remove the last word to avoid cutting mid-word
                    chunk_text = ' '.join(words[:-1])
                    actual_end = start + len(chunk_text)
                else:
                    actual_end = end
            
            # Add the chunk
            if chunk_text.strip():
                chunks.append({
                    'chunk_id': f"{doc_name}_chunk_{chunk_id}",
                    'document': doc_name,
                    'text': chunk_text.strip(),
                    'start_pos': start,
                    'end_pos': actual_end,
                    'size': len(chunk_text.strip())
                })
                chunk_id += 1
            
            # Move start position (with overlap)
            start = actual_end - self.overlap_size
            
            # Ensure we make progress
            if start <= chunks[-1]['start_pos'] if chunks else False:
                start = actual_end
        
        return chunks
    
    def chunk_documents(self, documents: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk all documents and return organized results
        """
        all_chunks = {}
        total_chunks = 0
        
        print(f"🔪 Starting chunking with:")
        print(f"   Chunk size: {self.chunk_size} characters")
        print(f"   Overlap: {self.overlap_size} characters ({self.overlap_size/self.chunk_size*100:.0f}%)")
        print("="*50)
        
        for doc_name, doc_data in documents.items():
            print(f"\n📄 Chunking: {doc_name}")
            
            text = doc_data['cleaned_text']
            chunks = self.chunk_text(text, doc_name)
            
            all_chunks[doc_name] = chunks
            total_chunks += len(chunks)
            
            # Print chunk statistics
            avg_size = sum(c['size'] for c in chunks) / len(chunks) if chunks else 0
            print(f"   Created {len(chunks)} chunks")
            print(f"   Average chunk size: {avg_size:.0f} characters")
            print(f"   Size range: {min(c['size'] for c in chunks) if chunks else 0} - {max(c['size'] for c in chunks) if chunks else 0}")
        
        print(f"\n✅ Chunking complete!")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Average chunks per document: {total_chunks / len(documents):.1f}")
        
        return all_chunks
    
    def analyze_chunk_quality(self, all_chunks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze the quality of generated chunks
        """
        all_chunk_data = []
        for doc_chunks in all_chunks.values():
            all_chunk_data.extend(doc_chunks)
        
        if not all_chunk_data:
            return {"error": "No chunks to analyze"}
        
        # Size statistics
        sizes = [chunk['size'] for chunk in all_chunk_data]
        
        # Content quality checks
        quality_issues = []
        very_short_chunks = [c for c in all_chunk_data if c['size'] < 200]
        very_long_chunks = [c for c in all_chunk_data if c['size'] > 1200]
        
        if very_short_chunks:
            quality_issues.append(f"{len(very_short_chunks)} chunks are very short (<200 chars)")
        if very_long_chunks:
            quality_issues.append(f"{len(very_long_chunks)} chunks are very long (>1200 chars)")
        
        # Check for sentence completeness (rough estimate)
        incomplete_chunks = []
        for chunk in all_chunk_data:
            text = chunk['text'].strip()
            if text and not text[-1] in '.!?':
                incomplete_chunks.append(chunk['chunk_id'])
        
        if len(incomplete_chunks) > len(all_chunk_data) * 0.3:  # More than 30%
            quality_issues.append(f"{len(incomplete_chunks)} chunks may have incomplete sentences")
        
        return {
            'total_chunks': len(all_chunk_data),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'size_std': (sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes))**0.5,
            'very_short_chunks': len(very_short_chunks),
            'very_long_chunks': len(very_long_chunks),
            'quality_issues': quality_issues,
            'chunks_per_document': {doc: len(chunks) for doc, chunks in all_chunks.items()}
        }
    
    def save_chunks(self, all_chunks: Dict[str, List[Dict[str, Any]]], output_file: str = "document_chunks.json"):
        """
        Save chunks to JSON file for use in RAG system
        """
        # Flatten chunks for easier use
        flat_chunks = []
        for doc_name, chunks in all_chunks.items():
            flat_chunks.extend(chunks)
        
        # Save both flat and organized versions
        save_data = {
            'chunking_config': {
                'chunk_size': self.chunk_size,
                'overlap_size': self.overlap_size,
                'total_chunks': len(flat_chunks)
            },
            'chunks_by_document': all_chunks,
            'all_chunks': flat_chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(flat_chunks)} chunks to {output_file}")
        return output_file

def main():
    """
    Test the chunking system
    """
    print("🔪 Testing Text Chunking System")
    print("="*50)
    
    # Load processed documents
    try:
        with open('processed_documents.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print("❌ processed_documents.json not found. Run document processing first.")
        return
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=800, overlap_size=200)
    
    # Chunk all documents
    all_chunks = chunker.chunk_documents(documents)
    
    # Analyze chunk quality
    quality_analysis = chunker.analyze_chunk_quality(all_chunks)
    
    print(f"\n📊 CHUNK QUALITY ANALYSIS")
    print("="*50)
    print(f"Total chunks: {quality_analysis['total_chunks']}")
    print(f"Average size: {quality_analysis['avg_chunk_size']:.0f} characters")
    print(f"Size range: {quality_analysis['min_chunk_size']} - {quality_analysis['max_chunk_size']}")
    print(f"Size consistency (std): {quality_analysis['size_std']:.0f}")
    
    if quality_analysis['quality_issues']:
        print(f"\n⚠️  Quality Issues:")
        for issue in quality_analysis['quality_issues']:
            print(f"   - {issue}")
    else:
        print(f"\n✅ No quality issues detected")
    
    print(f"\nChunks per document:")
    for doc, count in quality_analysis['chunks_per_document'].items():
        print(f"   - {doc}: {count} chunks")
    
    # Save chunks
    chunker.save_chunks(all_chunks)
    
    print(f"\n✅ Chunking complete! Ready for embedding generation.")

if __name__ == "__main__":
    main()
