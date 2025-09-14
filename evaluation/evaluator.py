#!/usr/bin/env python3
"""
Main evaluation script for the RAG system
Run this to perform comprehensive evaluation of your RAG application
"""

import os
import sys
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.metrics import RAGEvaluator
from evaluation.test_data import TestDataGenerator, load_test_dataset

def setup_evaluation():
    """
    Set up the evaluation by importing your RAG components
    """
    try:
        # Import your existing RAG pipeline
        from rag_app.pipeline import chain, retriever_vector, final_retriever
        
        # Load BM25 retriever
        bm25_path = project_root / "load_index" / "bm25_index.pkl"
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        
        # Create retrievers dictionary for comparison
        retrievers = {
            'bm25': bm25_retriever,
            'vector': retriever_vector,
            'hybrid_reranked': final_retriever
        }
        
        print("Successfully loaded RAG components!")
        return chain, retrievers
        
    except Exception as e:
        print(f"Error loading RAG components: {e}")
        print("Make sure your RAG system is properly set up and indexed.")
        return None, None

def run_quick_evaluation():
    """
    Run a quick evaluation with a small test dataset
    """
    print("Running Quick Evaluation")
    print("=" * 50)
    
    # Setup
    chain, retrievers = setup_evaluation()
    if not chain or not retrievers:
        return
    
    # Load or generate test dataset
    test_cases = load_test_dataset()
    if not test_cases:
        print("No test dataset available")
        return
    
    # Use only first 5 test cases for quick evaluation
    quick_test_cases = test_cases[:5]
    
    # Initialize evaluator
    evaluator = RAGEvaluator(chain, retrievers)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        quick_test_cases, 
        include_stress_test=False
    )
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    return results

def run_full_evaluation():
    """
    Run a comprehensive evaluation with full test dataset and stress testing
    """
    print("Running Full Evaluation")
    print("=" * 50)
    
    # Setup
    chain, retrievers = setup_evaluation()
    if not chain or not retrievers:
        return
    
    # Load or generate test dataset
    test_cases = load_test_dataset()
    if not test_cases:
        print("No test dataset available")
        return
    
    print(f"Using {len(test_cases)} test cases")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(chain, retrievers)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        test_cases, 
        include_stress_test=True
    )
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    # Save report to file
    report_file = f"evaluation_results/report_{results['timestamp'].replace(':', '-')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nFull report saved to: {report_file}")
    
    return results

def generate_test_dataset():
    """
    Generate a new test dataset from your documents
    """
    print("Generating Test Dataset")
    print("=" * 50)
    
    generator = TestDataGenerator()
    test_cases = generator.generate_and_save_test_dataset()
    
    print(f"Generated {len(test_cases)} test cases")
    
    # Show sample test cases
    if test_cases:
        print("\nSample test cases:")
        for i, case in enumerate(test_cases[:3]):
            print(f"{i+1}. {case['query']}")
            print(f"   Type: {case.get('question_type', 'N/A')}")
            print(f"   Expected: {case.get('expected_answer', 'N/A')[:100]}...")
            print()

def run_component_analysis():
    """
    Analyze individual components of the RAG system
    """
    print("Running Component Analysis")
    print("=" * 50)
    
    # Setup
    chain, retrievers = setup_evaluation()
    if not chain or not retrievers:
        return
    
    test_cases = load_test_dataset()[:10]  # Use 10 test cases
    
    print("\nTesting individual retriever components:")
    
    for name, retriever in retrievers.items():
        print(f"\n{name.upper()} RETRIEVER:")
        print("-" * 30)
        
        total_time = 0
        total_docs = 0
        
        for test_case in test_cases[:3]:  # Test with 3 queries
            query = test_case['query']
            
            import time
            start = time.time()
            try:
                docs = retriever.get_relevant_documents(query)
                retrieval_time = time.time() - start
                
                print(f"Query: {query[:50]}...")
                print(f"  Retrieved: {len(docs)} documents")
                print(f"  Time: {retrieval_time:.3f}s")
                
                if docs:
                    print(f"  Top result preview: {docs[0].page_content[:100]}...")
                
                total_time += retrieval_time
                total_docs += len(docs)
                
            except Exception as e:
                print(f"  Error: {e}")
        
        if test_cases[:3]:
            avg_time = total_time / len(test_cases[:3])
            avg_docs = total_docs / len(test_cases[:3])
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Average docs retrieved: {avg_docs:.1f}")

def benchmark_performance():
    """
    Run performance benchmarking
    """
    print("Running Performance Benchmark")
    print("=" * 50)
    
    chain, retrievers = setup_evaluation()
    if not chain or not retrievers:
        return
    
    test_queries = [
        "What is the main topic of this document?",
        "How does this system work?",
        "What are the key features?",
        "Explain the benefits of this approach",
        "What are the limitations?"
    ]
    
    import time
    import statistics
    
    print("\nMeasuring end-to-end response times:")
    response_times = []
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        times = []
        for run in range(3):  # 3 runs per query
            start = time.time()
            try:
                response = chain.invoke(query)
                end_time = time.time() - start
                times.append(end_time)
                print(f"  Run {run+1}: {end_time:.3f}s")
            except Exception as e:
                print(f"  Run {run+1}: Error - {e}")
                times.append(10.0)  # Penalty time
        
        avg_time = statistics.mean(times)
        response_times.extend(times)
        print(f"  Average: {avg_time:.3f}s")
    
    if response_times:
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Mean response time: {statistics.mean(response_times):.3f}s")
        print(f"  Median response time: {statistics.median(response_times):.3f}s")
        if len(response_times) > 1:
            print(f"  P95 response time: {statistics.quantiles(response_times, n=20)[18]:.3f}s")

def main():
    """
    Main entry point for evaluation
    """
    print("RAG System Evaluation Suite")
    print("=" * 50)
    print("Choose an evaluation option:")
    print("1. Quick Evaluation (5 test cases, no stress test)")
    print("2. Full Evaluation (all test cases + stress test)")
    print("3. Generate New Test Dataset")
    print("4. Component Analysis")
    print("5. Performance Benchmark")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                results = run_quick_evaluation()
            elif choice == '2':
                results = run_full_evaluation()
            elif choice == '3':
                generate_test_dataset()
            elif choice == '4':
                run_component_analysis()
            elif choice == '5':
                benchmark_performance()
            elif choice == '6':
                print("Exiting evaluation suite...")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
                continue
                
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()