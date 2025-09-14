#!/usr/bin/env python3
"""
Test basic Pinecone search
"""

import sys
import os
from dotenv import load_dotenv

# Add the rag_app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_app'))

load_dotenv()

try:
    from pipeline import retriever_vector, chain
    
    print("🧪 Testing basic retrieval...")
    
    # Test the retriever directly
    test_query = "What is retrieval augmented generation?"
    
    print(f"Query: {test_query}")
    docs = retriever_vector.get_relevant_documents(test_query)
    
    print(f"✅ Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs[:2]):
        print(f"Doc {i+1}: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
    
    # Test the chain
    print("\n🧪 Testing chain...")
    result = chain.invoke(test_query)
    print(f"✅ Chain result: {result[:100]}...")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
