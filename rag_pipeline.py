"""
RAG Pipeline Implementation
"""

import os
import chromadb
import openai
from dotenv import load_dotenv

class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) system"""

    def __init__(self, db_path="./chroma_db", collection_name="rag_documents"):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

        print(f"RAG Pipeline initialized with {self.collection.count()} documents")

    def query(self, question):
        """Complete RAG query: retrieve + generate"""
        # Generate query embedding
        query_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[question]
        )
        query_embedding = query_response.data[0].embedding

        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Create context from retrieved chunks
        context = "\n---\n".join(results['documents'][0])

        # Create prompt
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        # Generate response
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()

def main():
    """Test the RAG pipeline with multiple questions"""
    print("🚀 Testing RAG Pipeline")
    print("="*50)

    rag = RAGPipeline()

    # Test questions covering different papers and question types
    test_questions = [
        "What is the attention mechanism in transformers?",
        "How does Vision Transformer process images?",
        "What are the main advantages of BERT?",
        "How do transformers handle sequential data?",
        "What is self-attention?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        try:
            answer = rag.query(question)
            print(f"✅ Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n🎉 RAG Pipeline Testing Complete!")

if __name__ == "__main__":
    main()
