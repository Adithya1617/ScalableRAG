#!/usr/bin/env python3
"""
Monitor deployment status and test endpoints when ready
"""
import requests
import time
import json

def check_deployment_status():
    """Check if the deployment is ready"""
    base_url = "https://scalablerag-main.onrender.com"
    
    print("🔍 Monitoring deployment status...")
    print(f"Base URL: {base_url}")
    print("=" * 50)
    
    for attempt in range(20):  # Check for up to 10 minutes
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Deployment successful! Status: {health_data.get('status')}")
                print(f"   RAG Initialized: {health_data.get('rag_initialized')}")
                print(f"   Message: {health_data.get('message')}")
                return True
            else:
                print(f"⏳ Attempt {attempt + 1}: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⏳ Attempt {attempt + 1}: Connection error - deployment may still be in progress")
        
        time.sleep(30)  # Wait 30 seconds between checks
    
    print("❌ Deployment check timed out")
    return False

def test_all_endpoints():
    """Test all endpoints once deployment is ready"""
    base_url = "https://scalablerag-main.onrender.com"
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/init", None),
        ("POST", "/query", {"query": "What is artificial intelligence?"}),
        ("POST", "/query/intelligent", {
            "query": "Explain machine learning",
            "include_analysis": True,
            "include_citations": True
        })
    ]
    
    print("\n🧪 Testing all endpoints...")
    print("=" * 50)
    
    for method, path, data in endpoints:
        try:
            url = f"{base_url}{path}"
            if method == "GET":
                response = requests.get(url, timeout=15)
            else:
                response = requests.post(url, json=data, timeout=15)
            
            print(f"✅ {method} {path}")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                if path == "/health":
                    print(f"   RAG Status: {result.get('rag_initialized')}")
                elif "query" in path:
                    print(f"   Response length: {len(str(result.get('response', '')))}")
                    print(f"   Has analysis: {'analysis' in result}")
            else:
                print(f"   Error: {response.text[:100]}")
        except Exception as e:
            print(f"❌ {method} {path}")
            print(f"   Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    if check_deployment_status():
        test_all_endpoints()
    else:
        print("🔄 Manual check recommended: https://scalablerag-main.onrender.com/health")