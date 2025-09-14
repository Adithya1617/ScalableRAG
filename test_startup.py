#!/usr/bin/env python3
"""
Test script to diagnose startup issues with the RAG backend
"""

import os
import sys
import traceback

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        # Add rag_app to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_app'))
        
        print("  - Testing FastAPI...")
        from fastapi import FastAPI
        print("    ✅ FastAPI imported successfully")
        
        print("  - Testing pipeline imports...")
        from rag_app.pipeline import chain, intelligent_rag_query
        print("    ✅ Pipeline imported successfully")
        
        print("  - Testing main app...")
        from rag_app.main import app
        print("    ✅ Main app imported successfully")
        
        return True
    except Exception as e:
        print(f"    ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment variables"""
    print("\nTesting environment variables...")
    required_vars = [
        "GEMINI_API_KEY",
        "COHERE_API_KEY", 
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * len(value[:8])}...")
        else:
            print(f"  ❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("These might cause issues during startup.")
    
    return len(missing_vars) == 0

def test_health_endpoint():
    """Test if we can create the FastAPI app and access health endpoint"""
    print("\nTesting health endpoint...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_app'))
        from rag_app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/health")
        
        if response.status_code == 200:
            print(f"  ✅ Health endpoint responds: {response.json()}")
            return True
        else:
            print(f"  ❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Health endpoint test failed: {e}")
        traceback.print_exc()
        return False

def test_port_binding():
    """Test if uvicorn can bind to a port"""
    print("\nTesting port binding...")
    try:
        import uvicorn
        import socket
        
        # Test if we can bind to a port
        test_port = int(os.getenv("PORT", "8000"))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", test_port))
            print(f"  ✅ Can bind to port {test_port}")
            return True
    except Exception as e:
        print(f"  ❌ Port binding test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("=== RAG Backend Startup Diagnostics ===\n")
    
    # Test current working directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Python path: {sys.path[:3]}...")
    
    # Run tests
    tests = [
        ("Environment Variables", test_environment),
        ("Imports", test_imports),
        ("Health Endpoint", test_health_endpoint),
        ("Port Binding", test_port_binding)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n=== Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should start successfully.")
    else:
        print("⚠️ Some tests failed. This might explain the deployment issue.")
        print("\n📋 Troubleshooting suggestions:")
        
        if not results.get("Environment Variables", True):
            print("  - Set missing environment variables in Render dashboard")
        if not results.get("Imports", True):
            print("  - Check if all dependencies are installed correctly")
        if not results.get("Health Endpoint", True):
            print("  - Fix FastAPI app initialization issues")
        if not results.get("Port Binding", True):
            print("  - Check if PORT environment variable is set correctly")

if __name__ == "__main__":
    main()