"""
Embedding Generation and Vector Database Setup
Uses OpenAI embeddings API and Chroma vector database
"""

import json
import os
import time
from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv
from tqdm import tqdm

class EmbeddingGenerator:
    """
    Generate embeddings and manage vector database for RAG system
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize embedding generator with OpenAI and Chroma
        
        Why these choices?
        - OpenAI ada-002: High quality, 1536 dimensions, good for retrieval
        - Chroma: Simple, local, perfect for development and small datasets
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize Chroma client
        self.db_path = Path(db_path)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Embedding model configuration
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_dimension = 1536
        
        print(f"🔧 Initialized EmbeddingGenerator")
        print(f"   Model: {self.embedding_model}")
        print(f"   Database: {self.db_path}")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using OpenAI API
        
        Why batch processing?
        - More efficient API usage
        - Better rate limit handling
        - Progress tracking
        """
        all_embeddings = []
        
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i + batch_size]
            
            try:
                # Call OpenAI API
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add empty embeddings as placeholders
                all_embeddings.extend([[0.0] * self.embedding_dimension] * len(batch))
        
        print(f"✅ Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def create_vector_database(self, chunks_file: str = "document_chunks.json", 
                             collection_name: str = "rag_documents") -> str:
        """
        Create vector database from chunks
        
        Process:
        1. Load chunks from JSON
        2. Generate embeddings for all chunk texts
        3. Store in Chroma with metadata
        """
        print(f"🗄️  Creating vector database: {collection_name}")
        
        # Load chunks
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        chunks = chunk_data['all_chunks']
        print(f"📄 Loaded {len(chunks)} chunks")
        
        # Prepare data for embedding
        texts = [chunk['text'] for chunk in chunks]
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Create metadata for each chunk
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                'document': chunk['document'],
                'chunk_id': chunk['chunk_id'],
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'size': chunk['size']
            })
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        # Create or get collection
        try:
            # Delete existing collection if it exists
            self.client.delete_collection(collection_name)
            print(f"🗑️  Deleted existing collection: {collection_name}")
        except:
            pass  # Collection doesn't exist, that's fine
        
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "RAG document chunks with OpenAI embeddings"}
        )
        
        # Add documents to collection
        print(f"💾 Adding {len(chunks)} documents to vector database...")
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to DB"):
            end_idx = min(i + batch_size, len(chunks))
            
            collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=chunk_ids[i:end_idx]
            )
        
        print(f"✅ Vector database created successfully!")
        print(f"   Collection: {collection_name}")
        print(f"   Documents: {len(chunks)}")
        print(f"   Embedding dimension: {self.embedding_dimension}")
        
        return collection_name
    
    def test_retrieval(self, collection_name: str = "rag_documents", 
                      query: str = "What is attention mechanism?", 
                      n_results: int = 3) -> Dict[str, Any]:
        """
        Test the vector database with a sample query
        """
        print(f"🔍 Testing retrieval with query: '{query}'")
        
        # Get collection
        collection = self.client.get_collection(collection_name)
        
        # Generate query embedding
        query_embedding = self.generate_embeddings_batch([query])[0]
        
        # Search for similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        print(f"📊 Retrieved {len(results['documents'][0])} results:")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"\n🔸 Result {i+1} (similarity: {1-distance:.3f}):")
            print(f"   Document: {metadata['document']}")
            print(f"   Chunk: {metadata['chunk_id']}")
            print(f"   Text preview: {doc[:200]}...")
        
        return {
            'query': query,
            'results': results,
            'n_results': len(results['documents'][0])
        }
    
    def get_database_info(self, collection_name: str = "rag_documents") -> Dict[str, Any]:
        """
        Get information about the vector database
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                'collection_name': collection_name,
                'document_count': count,
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.embedding_dimension,
                'database_path': str(self.db_path)
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    """
    Test the embedding generation and vector database setup
    """
    print("🚀 Starting Embedding Generation and Vector Database Setup")
    print("="*60)
    
    try:
        # Initialize generator
        generator = EmbeddingGenerator()
        
        # Create vector database
        collection_name = generator.create_vector_database()
        
        # Test retrieval
        print(f"\n" + "="*60)
        generator.test_retrieval(
            collection_name=collection_name,
            query="What is attention mechanism in transformers?",
            n_results=3
        )
        
        # Another test query
        print(f"\n" + "="*60)
        generator.test_retrieval(
            collection_name=collection_name,
            query="How does Vision Transformer work with image patches?",
            n_results=3
        )
        
        # Database info
        print(f"\n" + "="*60)
        info = generator.get_database_info(collection_name)
        print(f"📊 DATABASE INFO:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Embedding generation and vector database setup complete!")
        print(f"   Ready for RAG pipeline implementation!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
