#!/usr/bin/env python3
"""
Comprehensive test of the fixed RAG system
"""

import sys
import os
from dotenv import load_dotenv

# Add the rag_app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_app'))

load_dotenv()

def test_retriever():
    """Test the retriever directly"""
    print("🧪 Testing Custom Retriever...")
    
    try:
        from pipeline import PineconeLlamaRetriever, QueryIntelligence
        from pinecone import Pinecone
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
        index = pc.Index(index_name)
        
        # Create retriever
        query_intelligence = QueryIntelligence()
        retriever = PineconeLlamaRetriever(index=index, top_k=3, query_intelligence=query_intelligence)
        
        # Test query
        test_query = "What is retrieval augmented generation?"
        docs = retriever._get_relevant_documents(test_query)
        
        print(f"✅ Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"\nDoc {i+1}:")
            print(f"  ID: {doc.metadata.get('id', 'N/A')}")
            print(f"  Score: {doc.metadata.get('score', 'N/A')}")
            print(f"  Source: {doc.metadata.get('source_type', 'N/A')}")
            print(f"  Content: {doc.page_content[:150]}...")
            if doc.metadata.get('filename'):
                print(f"  Filename: {doc.metadata['filename']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intelligent_rag():
    """Test the intelligent RAG function"""
    print("\n🧪 Testing Intelligent RAG...")
    
    try:
        from pipeline import intelligent_rag_query
        
        test_query = "What is retrieval augmented generation?"
        result = intelligent_rag_query(test_query, include_analysis=True, include_citations=True)
        
        if result.get('error'):
            print(f"❌ Error: {result.get('error_message')}")
            return False
        
        print(f"✅ Success!")
        print(f"Response: {result['response'][:200]}...")
        
        if 'query_analysis' in result:
            analysis = result['query_analysis']
            print(f"Query Type: {analysis['query_type']}")
            print(f"Keywords: {', '.join(analysis['keywords'][:5])}")
            print(f"Complexity: {analysis['complexity_score']:.2f}")
        
        if 'citations' in result:
            print(f"Citations: {len(result['citations'])}")
            for i, citation in enumerate(result['citations'][:2]):
                print(f"  Citation {i+1}: {citation['filename']} (conf: {citation['confidence_score']:.2f})")
        
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chain():
    """Test the basic chain"""
    print("\n🧪 Testing Basic Chain...")
    
    try:
        from pipeline import chain
        
        test_query = "What is retrieval augmented generation?"
        result = chain.invoke(test_query)
        
        print(f"✅ Chain result: {result[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running Comprehensive RAG System Test")
    print("=" * 50)
    
    # Run all tests
    retriever_ok = test_retriever()
    intelligent_ok = test_intelligent_rag()
    chain_ok = test_chain()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Retriever: {'✅ PASS' if retriever_ok else '❌ FAIL'}")
    print(f"  Intelligent RAG: {'✅ PASS' if intelligent_ok else '❌ FAIL'}")
    print(f"  Basic Chain: {'✅ PASS' if chain_ok else '❌ FAIL'}")
    
    if all([retriever_ok, intelligent_ok, chain_ok]):
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
