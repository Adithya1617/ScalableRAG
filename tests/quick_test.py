#!/usr/bin/env python3
"""
Quick test of the advanced system endpoints
"""

import requests
import json
import time

def test_endpoints():
    """Test key endpoints"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Advanced RAG System")
    print("=" * 40)
    
    # Test health
    try:
        print("1. Health Check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data.get('status', 'unknown')}")
            print(f"   RAG Available: {data.get('rag_available', False)}")
            print(f"   Advanced Metrics: {data.get('advanced_metrics_available', False)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test basic chat
    try:
        print("\n2. Basic Chat...")
        response = requests.post(
            f"{base_url}/chat/",
            json={"message": "Hello, what can you tell me?"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat working, response length: {len(data.get('response', ''))}")
        else:
            print(f"❌ Chat failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat error: {e}")
    
    # Test intelligent query
    try:
        print("\n3. Intelligent Query...")
        response = requests.post(
            f"{base_url}/query/intelligent",
            json={
                "query": "What information is available?",
                "include_analysis": True,
                "include_citations": True
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Intelligent query working")
            print(f"   Has analysis: {bool(data.get('query_analysis'))}")
            print(f"   Citations: {len(data.get('citations', []))}")
        else:
            print(f"❌ Intelligent query failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Intelligent query error: {e}")
    
    # Test advanced metrics
    try:
        print("\n4. Advanced Metrics...")
        response = requests.get(f"{base_url}/evaluation/advanced", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Advanced metrics available")
            if 'advanced_evaluator_status' in data:
                status = data['advanced_evaluator_status']
                print(f"   Evaluator ready: {status.get('available', False)}")
        else:
            print(f"❌ Advanced metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Advanced metrics error: {e}")
    
    # Test human feedback
    try:
        print("\n5. Human Feedback...")
        response = requests.post(
            f"{base_url}/human-feedback",
            json={
                "query": "Test query",
                "response": "Test response",
                "rating": 4,
                "feedback_text": "Test feedback",
                "session_id": f"test_{int(time.time())}"
            },
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Human feedback collection working")
            print(f"   Feedback ID: {data.get('feedback_id', 'unknown')}")
        else:
            print(f"❌ Human feedback failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Human feedback error: {e}")
    
    print("\n🎯 Quick test completed!")

if __name__ == "__main__":
    test_endpoints()
