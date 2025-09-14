import json
import os
import sys
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Add the parent directory to the path to import from rag_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

class TestDataGenerator:
    def __init__(self):
        """Initialize the test data generator"""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")
    
    def generate_questions_from_documents(self, documents: List[str], questions_per_doc: int = 3) -> List[Dict]:
        """
        Generate test questions from document content
        
        Args:
            documents: List of document text chunks
            questions_per_doc: Number of questions to generate per document
        
        Returns:
            List of test cases with queries and expected information
        """
        test_cases = []
        
        for i, doc_content in enumerate(documents):
            # Limit document content to avoid token limits
            content_sample = doc_content[:2000] if len(doc_content) > 2000 else doc_content
            
            prompt = f"""
            Based on the following document content, generate {questions_per_doc} different types of questions that can be answered from this text:

            Document Content:
            {content_sample}

            Please generate:
            1. One factual question (asking for specific information)
            2. One analytical question (requiring reasoning about the content)
            3. One summary question (asking for key points or overview)

            Format your response as JSON:
            {{
                "questions": [
                    {{
                        "question": "your question here",
                        "type": "factual|analytical|summary",
                        "expected_answer": "brief expected answer",
                        "keywords": ["key", "words", "from", "answer"]
                    }}
                ]
            }}
            """
            
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Clean up the response to extract JSON
                if "```json" in response_text:
                    json_part = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_part = response_text.split("```")[1].strip()
                else:
                    json_part = response_text
                
                # Parse the JSON response
                parsed_response = json.loads(json_part)
                
                for q_data in parsed_response.get("questions", []):
                    test_case = {
                        "query": q_data.get("question", ""),
                        "expected_answer": q_data.get("expected_answer", ""),
                        "question_type": q_data.get("type", "factual"),
                        "keywords": q_data.get("keywords", []),
                        "relevant_doc_ids": [f"doc_{i}"],  # Reference to source document
                        "document_content": content_sample
                    }
                    test_cases.append(test_case)
                    
            except Exception as e:
                print(f"Error generating questions for document {i}: {e}")
                # Create fallback questions
                test_cases.append({
                    "query": f"What is the main topic discussed in this document?",
                    "expected_answer": "The document discusses various topics related to the content.",
                    "question_type": "summary",
                    "keywords": ["main", "topic", "document"],
                    "relevant_doc_ids": [f"doc_{i}"],
                    "document_content": content_sample
                })
        
        return test_cases
    
    def create_sample_test_dataset(self) -> List[Dict]:
        """
        Create a sample test dataset for RAG evaluation
        This can be used when you don't have your own documents loaded
        """
        sample_test_cases = [
            {
                "query": "What is retrieval augmented generation?",
                "expected_answer": "Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation to provide more accurate and factual responses.",
                "question_type": "factual",
                "keywords": ["RAG", "retrieval", "generation", "technique"],
                "relevant_doc_ids": ["rag_definition"],
                "document_content": "RAG documentation content would go here"
            },
            {
                "query": "How does hybrid search work in RAG systems?",
                "expected_answer": "Hybrid search combines vector similarity search with keyword-based search (like BM25) to improve retrieval accuracy by capturing both semantic and lexical matches.",
                "question_type": "analytical",
                "keywords": ["hybrid", "search", "vector", "BM25", "semantic"],
                "relevant_doc_ids": ["hybrid_search"],
                "document_content": "Hybrid search documentation content would go here"
            },
            {
                "query": "What are the benefits of using reranking in RAG?",
                "expected_answer": "Reranking improves the relevance of retrieved documents by using a more sophisticated model to reorder results from initial retrieval, leading to better context for generation.",
                "question_type": "analytical",
                "keywords": ["reranking", "relevance", "benefits", "context"],
                "relevant_doc_ids": ["reranking_benefits"],
                "document_content": "Reranking documentation content would go here"
            },
            {
                "query": "Explain the architecture of a typical RAG system",
                "expected_answer": "A typical RAG system consists of a document store, embedding model, vector database, retriever, reranker, and language model for generation.",
                "question_type": "summary",
                "keywords": ["architecture", "components", "system", "pipeline"],
                "relevant_doc_ids": ["rag_architecture"],
                "document_content": "RAG architecture documentation content would go here"
            },
            {
                "query": "What is the role of embeddings in RAG?",
                "expected_answer": "Embeddings convert text into numerical vectors that capture semantic meaning, enabling similarity search to find relevant documents for a given query.",
                "question_type": "factual",
                "keywords": ["embeddings", "vectors", "semantic", "similarity"],
                "relevant_doc_ids": ["embeddings_role"],
                "document_content": "Embeddings documentation content would go here"
            }
        ]
        
        return sample_test_cases
    
    def load_documents_from_rag_system(self):
        """
        Load documents from your existing RAG system for question generation
        This integrates with your current setup
        """
        try:
            # Try to import your existing pipeline
            from rag_app.pipeline import index, bm25_retriever
            
            # Get some sample documents from your system
            documents = []
            
            # Method 1: Get documents from BM25 retriever if possible
            try:
                if hasattr(bm25_retriever, 'corpus') and bm25_retriever.corpus:
                    documents.extend(bm25_retriever.corpus[:10])  # Get first 10 documents
            except Exception as e:
                print(f"Could not get documents from BM25: {e}")
            
            # Method 2: Query Pinecone for sample documents
            try:
                sample_results = index.search(
                    query={
                        "inputs": {"text": "document content sample"},
                        "top_k": 5
                    },
                    namespace="default"
                )
                
                hits = sample_results.get('result', {}).get('hits', [])
                for hit in hits:
                    text = hit.get('fields', {}).get('text', '')
                    if text:
                        documents.append(text)
                        
            except Exception as e:
                print(f"Could not get documents from Pinecone: {e}")
            
            return documents
            
        except Exception as e:
            print(f"Could not load documents from RAG system: {e}")
            return []
    
    def generate_and_save_test_dataset(self, output_file: str = "test_datasets/generated_test_cases.json"):
        """
        Generate a complete test dataset and save it
        """
        print("Generating test dataset...")
        
        # Try to get documents from your RAG system
        documents = self.load_documents_from_rag_system()
        
        if documents:
            print(f"Found {len(documents)} documents from RAG system")
            test_cases = self.generate_questions_from_documents(documents, questions_per_doc=2)
        else:
            print("Using sample test dataset")
            test_cases = self.create_sample_test_dataset()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        print(f"Generated {len(test_cases)} test cases and saved to {output_file}")
        return test_cases

def load_test_dataset(file_path: str = "test_datasets/generated_test_cases.json") -> List[Dict]:
    """
    Load test dataset from JSON file
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Test dataset not found at {file_path}")
        print("Generating new test dataset...")
        generator = TestDataGenerator()
        return generator.generate_and_save_test_dataset(file_path)

if __name__ == "__main__":
    # Generate test dataset
    generator = TestDataGenerator()
    test_cases = generator.generate_and_save_test_dataset()
    print(f"Generated {len(test_cases)} test cases")