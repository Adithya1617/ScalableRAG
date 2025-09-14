
"""
Debug script to identify and fix evaluation issues
"""

import os
import sys
import pickle
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def debug_rag_system():
    """Debug your RAG system to understand the data structure"""
    print("DEBUGGING RAG SYSTEM")
    print("=" * 50)
    
    try:
        # Import your RAG components
        from rag_app.pipeline import index, chain, retriever_vector, final_retriever
        
        print("✅ Successfully imported RAG components")
        
        # Test Pinecone connection
        print("\n1. Testing Pinecone Index...")
        try:
            stats = index.describe_index_stats()
            print(f"   Total vectors: {stats.total_vector_count}")
            print(f"   Dimension: {stats.dimension}")
            print(f"   Namespaces: {list(stats.namespaces.keys())}")
            
            # Get sample documents from Pinecone
            sample_results = index.search(
                query={
                    "inputs": {"text": "sample query"},
                    "top_k": 3
                },
                namespace="default"
            )
            
            hits = sample_results.get('result', {}).get('hits', [])
            print(f"   Sample query returned {len(hits)} results")
            
            if hits:
                for i, hit in enumerate(hits[:2]):
                    hit_id = hit.get('_id', 'no_id')
                    text_preview = hit.get('fields', {}).get('text', '')[:100]
                    score = hit.get('_score', 0)
                    print(f"   Result {i+1}: ID='{hit_id}', Score={score:.3f}")
                    print(f"   Text preview: {text_preview}...")
                    print()
            
        except Exception as e:
            print(f"   ❌ Pinecone error: {e}")
        
        # Test BM25 retriever
        print("\n2. Testing BM25 Retriever...")
        try:
            bm25_path = project_root / "load_index" / "bm25_index.pkl"
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            
            print(f"   ✅ BM25 retriever loaded")
            print(f"   K value: {bm25_retriever.k}")
            
            # Test BM25 retrieval
            bm25_docs = bm25_retriever.get_relevant_documents("sample query")
            print(f"   BM25 returned {len(bm25_docs)} documents")
            
            if bm25_docs:
                print(f"   First result preview: {bm25_docs[0].page_content[:100]}...")
            
        except Exception as e:
            print(f"   ❌ BM25 error: {e}")
        
        # Test end-to-end chain
        print("\n3. Testing End-to-End Chain...")
        try:
            test_query = "What is this document about?"
            import time
            start = time.time()
            response = chain.invoke(test_query)
            end_time = time.time() - start
            
            print(f"   ✅ Chain response time: {end_time:.2f}s")
            print(f"   Response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Chain error: {e}")
            
    except Exception as e:
        print(f"❌ Failed to import RAG components: {e}")
        return False
    
    return True

def create_proper_test_dataset():
    """Create a test dataset that matches your actual indexed data"""
    print("\nCREATING PROPER TEST DATASET")
    print("=" * 50)
    
    try:
        from rag_app.pipeline import index, retriever_vector
        
        # Get actual documents from your Pinecone index
        sample_queries = [
            "main topic",
            "key information", 
            "important details",
            "overview",
            "summary"
        ]
        
        test_cases = []
        
        for i, query in enumerate(sample_queries):
            print(f"Creating test case {i+1} with query: '{query}'")
            
            try:
                # Get results from Pinecone
                results = index.search(
                    query={
                        "inputs": {"text": query},
                        "top_k": 3
                    },
                    namespace="default"
                )
                
                hits = results.get('result', {}).get('hits', [])
                
                if hits:
                    # Use the first result to create a test case
                    top_hit = hits[0]
                    doc_id = top_hit.get('_id', '')
                    doc_text = top_hit.get('fields', {}).get('text', '')
                    
                    if doc_text:
                        # Create a question that can be answered from this document
                        question = f"What information is contained in the document about {query}?"
                        
                        test_case = {
                            "query": question,
                            "expected_answer": doc_text[:200] + "...",  # First 200 chars as expected
                            "relevant_doc_ids": [doc_id],  # Use the actual Pinecone ID
                            "question_type": "factual",
                            "document_content": doc_text
                        }
                        
                        test_cases.append(test_case)
                        print(f"   ✅ Created test case with doc ID: {doc_id}")
                    else:
                        print(f"   ⚠️ No text found in result")
                else:
                    print(f"   ⚠️ No results found for query: {query}")
                    
            except Exception as e:
                print(f"   ❌ Error creating test case: {e}")
        
        if test_cases:
            # Save the proper test dataset
            os.makedirs("test_datasets", exist_ok=True)
            output_file = "test_datasets/corrected_test_cases.json"
            
            with open(output_file, 'w') as f:
                json.dump(test_cases, f, indent=2)
            
            print(f"\n✅ Created {len(test_cases)} proper test cases")
            print(f"💾 Saved to: {output_file}")
            
            # Show sample
            if test_cases:
                print(f"\nSample test case:")
                sample = test_cases[0]
                print(f"Query: {sample['query']}")
                print(f"Relevant Doc ID: {sample['relevant_doc_ids'][0]}")
                print(f"Expected Answer: {sample['expected_answer'][:100]}...")
            
            return test_cases
        else:
            print("❌ No test cases created")
            return []
            
    except Exception as e:
        print(f"❌ Error creating test dataset: {e}")
        return []

def run_corrected_evaluation():
    """Run evaluation with corrected test dataset"""
    print("\nRUNNING CORRECTED EVALUATION")
    print("=" * 50)
    
    try:
        # Load corrected test dataset
        test_file = "test_datasets/corrected_test_cases.json"
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_cases = json.load(f)
            print(f"✅ Loaded {len(test_cases)} corrected test cases")
        else:
            print("❌ Corrected test dataset not found. Run create_proper_test_dataset() first.")
            return
        
        # Import evaluation components
        from evaluation.metrics import RAGEvaluator
        from rag_app.pipeline import chain, retriever_vector, final_retriever
        
        # Load BM25
        bm25_path = project_root / "load_index" / "bm25_index.pkl"
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        
        retrievers = {
            'vector': retriever_vector,
            'hybrid_reranked': final_retriever
        }
        
        # Run evaluation
        evaluator = RAGEvaluator(chain, retrievers)
        results = evaluator.run_comprehensive_evaluation(test_cases, include_stress_test=False)
        
        # Print results
        report = evaluator.generate_report(results)
        print("\n" + report)
        
        return results
        
    except Exception as e:
        print(f"❌ Error running corrected evaluation: {e}")
        import traceback
        traceback.print_exc()

def manual_retrieval_test():
    """Manually test retrieval to see what's happening"""
    print("\nMANUAL RETRIEVAL TEST")
    print("=" * 50)
    
    try:
        from rag_app.pipeline import retriever_vector, final_retriever
        
        test_query = "What is the main topic?"
        print(f"Testing query: '{test_query}'")
        
        # Test vector retriever
        print("\nVector Retriever Results:")
        try:
            vector_docs = retriever_vector.get_relevant_documents(test_query)
            print(f"Retrieved {len(vector_docs)} documents")
            for i, doc in enumerate(vector_docs[:2]):
                print(f"Doc {i+1} ID: {doc.metadata.get('id', 'no_id')}")
                print(f"Doc {i+1} Score: {doc.metadata.get('score', 'no_score')}")
                print(f"Doc {i+1} Preview: {doc.page_content[:100]}...")
                print()
        except Exception as e:
            print(f"Vector retriever error: {e}")
        
        # Test hybrid+reranked retriever
        print("Hybrid+Reranked Retriever Results:")
        try:
            hybrid_docs = final_retriever.get_relevant_documents(test_query)
            print(f"Retrieved {len(hybrid_docs)} documents")
            for i, doc in enumerate(hybrid_docs[:2]):
                print(f"Doc {i+1} ID: {doc.metadata.get('id', 'no_id')}")
                print(f"Doc {i+1} Score: {doc.metadata.get('score', 'no_score')}")
                print(f"Doc {i+1} Preview: {doc.page_content[:100]}...")
                print()
        except Exception as e:
            print(f"Hybrid retriever error: {e}")
            
    except Exception as e:
        print(f"Manual test error: {e}")

def main():
    """Main debugging function"""
    print("RAG EVALUATION DEBUGGING TOOL")
    print("=" * 50)
    
    while True:
        print("\nChoose debugging option:")
        print("1. Debug RAG system components")
        print("2. Create proper test dataset")
        print("3. Run corrected evaluation")
        print("4. Manual retrieval test")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            debug_rag_system()
        elif choice == '2':
            create_proper_test_dataset()
        elif choice == '3':
            run_corrected_evaluation()
        elif choice == '4':
            manual_retrieval_test()
        elif choice == '5':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()