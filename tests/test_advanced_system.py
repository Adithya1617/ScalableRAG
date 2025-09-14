#!/usr/bin/env python3
"""
Test script for advanced metrics and human feedback system
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    "What is this document about?",
    "Summarize the main points",
    "What are the key findings?",
    "Explain the methodology",
    "What are the conclusions?"
]

def test_basic_chat():
    """Test basic chat functionality"""
    print("🔍 Testing Basic Chat...")
    try:
        response = requests.post(
            f"{BASE_URL}/chat/",
            json={"message": "Hello, what can you tell me about this document?"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Basic chat working. Response length: {len(data.get('response', ''))}")
            return True
        else:
            print(f"❌ Chat failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_intelligent_query():
    """Test intelligent query with advanced features"""
    print("\n🧠 Testing Intelligent Query...")
    try:
        response = requests.post(
            f"{BASE_URL}/query/intelligent",
            json={
                "query": "What are the main topics in this document?",
                "include_analysis": True,
                "include_citations": True,
                "metadata_filters": None
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Intelligent query working")
            print(f"   Response length: {len(data.get('response', ''))}")
            print(f"   Has analysis: {bool(data.get('query_analysis'))}")
            print(f"   Citations count: {len(data.get('citations', []))}")
            
            if data.get('query_analysis'):
                analysis = data['query_analysis']
                print(f"   Query type: {analysis.get('query_type')}")
                print(f"   Complexity: {analysis.get('complexity_score', 0):.2f}")
                print(f"   Keywords: {len(analysis.get('keywords', []))}")
            
            return True
        else:
            print(f"❌ Intelligent query failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Intelligent query error: {e}")
        return False

def test_advanced_metrics():
    """Test advanced metrics endpoint"""
    print("\n📊 Testing Advanced Metrics...")
    try:
        response = requests.get(f"{BASE_URL}/evaluation/advanced")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Advanced metrics endpoint working")
            
            if 'advanced_evaluator_status' in data:
                status = data['advanced_evaluator_status']
                print(f"   Evaluator available: {status.get('available', False)}")
                print(f"   Models loaded: {status.get('models_loaded', False)}")
                print(f"   Dependencies: {status.get('dependencies_available', False)}")
            
            return True
        else:
            print(f"❌ Advanced metrics failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Advanced metrics error: {e}")
        return False

def test_real_time_metrics():
    """Test real-time metrics"""
    print("\n⚡ Testing Real-time Metrics...")
    try:
        test_query = "What is this document about?"
        response = requests.get(f"{BASE_URL}/metrics/real-time/{test_query}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Real-time metrics working")
            
            if 'real_time_metrics' in data:
                metrics = data['real_time_metrics']
                print(f"   Overall score: {metrics.get('overall_score', 0):.3f}")
                print(f"   Semantic similarity: {metrics.get('semantic_similarity', 0):.3f}")
                print(f"   Factual consistency: {metrics.get('factual_consistency', 0):.3f}")
                print(f"   Citation accuracy: {metrics.get('citation_accuracy', 0):.3f}")
                print(f"   Response appropriateness: {metrics.get('response_appropriateness', 0):.3f}")
                
                if 'benchmark_comparison' in metrics:
                    benchmark = metrics['benchmark_comparison']
                    print(f"   Benchmark category: {benchmark.get('category', 'unknown')}")
                    print(f"   Percentile: {benchmark.get('percentile', 0)}")
            
            return True
        else:
            print(f"❌ Real-time metrics failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Real-time metrics error: {e}")
        return False

def test_human_feedback():
    """Test human feedback collection"""
    print("\n👤 Testing Human Feedback Collection...")
    try:
        feedback_data = {
            "query": "Test query for feedback",
            "response": "Test response for feedback collection",
            "rating": 4,
            "feedback_text": "This is a test feedback message",
            "session_id": f"test_session_{int(time.time())}"
        }
        
        response = requests.post(
            f"{BASE_URL}/human-feedback",
            json=feedback_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Human feedback collection working")
            print(f"   Feedback ID: {data.get('feedback_id')}")
            print(f"   Analysis: {data.get('analysis', {}).get('sentiment', 'unknown')}")
            return True
        else:
            print(f"❌ Human feedback failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Human feedback error: {e}")
        return False

def test_ground_truth_validation():
    """Test ground truth validation"""
    print("\n🎯 Testing Ground Truth Validation...")
    try:
        validation_data = {
            "query": "Test validation query",
            "response": "Test response for validation",
            "expected_answer": "Expected answer for comparison",
            "context_documents": ["test document content"]
        }
        
        response = requests.post(
            f"{BASE_URL}/ground-truth/validate",
            json=validation_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ground truth validation working")
            
            validation = data.get('validation_result', {})
            print(f"   Semantic match: {validation.get('semantic_match', 0):.3f}")
            print(f"   Factual accuracy: {validation.get('factual_accuracy', 0):.3f}")
            print(f"   Completeness: {validation.get('completeness', 0):.3f}")
            print(f"   Overall score: {validation.get('overall_score', 0):.3f}")
            
            return True
        else:
            print(f"❌ Ground truth validation failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ground truth validation error: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\n❤️ Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint working")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   RAG available: {data.get('rag_available', False)}")
            print(f"   Advanced metrics: {data.get('advanced_metrics_available', False)}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_comprehensive_system():
    """Run comprehensive system test"""
    print("🚀 COMPREHENSIVE ADVANCED SYSTEM TEST")
    print("=" * 60)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            if i < max_retries - 1:
                time.sleep(2)
                print(f"   Retry {i+1}/{max_retries}...")
            else:
                print("❌ Server not responding after 60 seconds")
                return False
    
    # Run all tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Basic Chat", test_basic_chat),
        ("Intelligent Query", test_intelligent_query),
        ("Advanced Metrics", test_advanced_metrics),
        ("Real-time Metrics", test_real_time_metrics),
        ("Human Feedback", test_human_feedback),
        ("Ground Truth Validation", test_ground_truth_validation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All advanced system features are working correctly!")
        return True
    else:
        print("⚠️ Some features need attention.")
        return False

def interactive_test():
    """Interactive testing mode"""
    print("🔧 INTERACTIVE TESTING MODE")
    print("=" * 40)
    
    while True:
        print("\nAvailable tests:")
        print("1. Health Check")
        print("2. Basic Chat")
        print("3. Intelligent Query")
        print("4. Advanced Metrics")
        print("5. Real-time Metrics")
        print("6. Human Feedback")
        print("7. Ground Truth Validation")
        print("8. Comprehensive Test")
        print("9. Exit")
        
        choice = input("\nSelect test (1-9): ").strip()
        
        if choice == '1':
            test_health_endpoint()
        elif choice == '2':
            test_basic_chat()
        elif choice == '3':
            test_intelligent_query()
        elif choice == '4':
            test_advanced_metrics()
        elif choice == '5':
            test_real_time_metrics()
        elif choice == '6':
            test_human_feedback()
        elif choice == '7':
            test_ground_truth_validation()
        elif choice == '8':
            test_comprehensive_system()
        elif choice == '9':
            break
        else:
            print("Invalid choice. Please select 1-9.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_comprehensive_system()
