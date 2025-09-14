#!/usr/bin/env python3
"""
Test the fixed Pinecone retriever
"""

import sys
import os

# Add the rag_app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_app'))

try:
    from pipeline import intelligent_rag_query
    
    print("🧪 Testing the fixed intelligent RAG query...")
    
    # Test with a simple query
    test_query = "What is retrieval augmented generation?"
    
    print(f"Query: {test_query}")
    result = intelligent_rag_query(test_query, include_analysis=True, include_citations=True)
    
    if result.get('error'):
        print(f"❌ Error: {result.get('error_message')}")
    else:
        print("✅ Success!")
        print(f"Response: {result['response'][:100]}...")
        if 'query_analysis' in result:
            print(f"Query Type: {result['query_analysis']['query_type']}")
        if 'citations' in result:
            print(f"Citations: {len(result['citations'])}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
