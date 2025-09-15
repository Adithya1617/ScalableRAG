#!/usr/bin/env python3
"""
Simple test script to verify API endpoints work correctly
"""
import requests
import json
import time

def test_endpoint(url, method='GET', data=None):
    """Test an API endpoint"""
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        
        print(f"✅ {method} {url}")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ {method} {url}")
        print(f"   Error: {str(e)}")
        return False

def main():
    base_url = "https://scalablerag-main.onrender.com"
    
    print("🧪 Testing optimized API endpoints...")
    print(f"Base URL: {base_url}")
    print("=" * 50)
    
    # Test health endpoint
    test_endpoint(f"{base_url}/health")
    
    # Test status endpoint (new)
    test_endpoint(f"{base_url}/status")
    
    # Test init endpoint
    test_endpoint(f"{base_url}/init")
    
    # Test query endpoint
    test_endpoint(f"{base_url}/query", 'POST', {"query": "What is artificial intelligence?"})
    
    # Test intelligent query endpoint
    test_endpoint(f"{base_url}/query/intelligent", 'POST', {
        "query": "Explain machine learning",
        "include_analysis": True,
        "include_citations": True
    })

if __name__ == "__main__":
    main()