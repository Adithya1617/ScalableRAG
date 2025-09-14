#!/usr/bin/env python3
"""
Simple test of just the retriever
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add the rag_app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_app'))

try:
    from pinecone import Pinecone
    
    # Initialize Pinecone directly
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index_name = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
    index = pc.Index(index_name)
    
    print("🧪 Testing direct Pinecone search...")
    
    # Test basic search
    results = index.search(
        namespace="default",
        query={
            "inputs": {"text": "What is retrieval augmented generation?"},
            "top_k": 2
        }
    )
    
    print(f"Search completed. Result type: {type(results)}")
    
    # Check if we have results
    if hasattr(results, 'result') and hasattr(results.result, 'hits'):
        hits = results.result.hits
        print(f"Found {len(hits)} hits")
        
        for i, hit in enumerate(hits):
            print(f"\nHit {i+1}:")
            print(f"  ID: {getattr(hit, 'id', 'N/A')}")
            print(f"  Score: {getattr(hit, 'score', 'N/A')}")
            
            if hasattr(hit, 'fields'):
                if hasattr(hit.fields, 'text'):
                    text = hit.fields.text
                    print(f"  Text: {text[:100]}...")
                else:
                    print("  No text found in fields")
            else:
                print("  No fields found")
    else:
        print("No hits found or unexpected result structure")
    
    # Now test our custom retriever
    print("\n🧪 Testing custom retriever...")
    from pipeline import PineconeLlamaRetriever, QueryIntelligence
    
    query_intelligence = QueryIntelligence()
    retriever = PineconeLlamaRetriever(index=index, top_k=2, query_intelligence=query_intelligence)
    
    docs = retriever._get_relevant_documents("What is retrieval augmented generation?")
    print(f"Retrieved {len(docs)} documents")
    
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
