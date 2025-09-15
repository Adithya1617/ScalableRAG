#!/usr/bin/env python3
"""
Test script for ultra-minimal RAG backend
"""
import requests
import json

def test_ultra_minimal():
    base_url = "http://localhost:8000"  # Local test first
    
    print("🧪 Testing ultra-minimal RAG backend...")
    print(f"Base URL: {base_url}")
    print("=" * 50)
    
    # Test queries that should work well with the knowledge base
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "Tell me about Python programming",
        "What is data science?",
        "Explain RAG systems",
        "How does FastAPI work?",
        "What is deep learning?"
    ]
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        print()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        print(f"✅ Status check: {response.status_code}")
        print(f"   Mode: {response.json().get('mode')}")
        print(f"   Topics: {response.json().get('knowledge_base_topics')}")
        print()
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    # Test simple queries
    for query in test_queries[:3]:  # Test first 3 queries
        try:
            response = requests.post(f"{base_url}/query", 
                                   json={"query": query}, 
                                   timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query: '{query}'")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")
                print(f"   Response length: {len(result.get('response', ''))}")
                print(f"   Source: {result.get('source', 'N/A')}")
            else:
                print(f"❌ Query failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Query error: {e}")
        print()

if __name__ == "__main__":
    test_ultra_minimal()