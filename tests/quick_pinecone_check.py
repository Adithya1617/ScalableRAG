#!/usr/bin/env python3
"""
Quick check of Pinecone search API
"""

import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
index = pc.Index(index_name)

print(f'Index: {index_name}')
print(f'Type: {type(index)}')

# Check available methods
methods = [m for m in dir(index) if 'search' in m.lower()]
print(f'Search methods: {methods}')

# Get search signature
import inspect
try:
    sig = inspect.signature(index.search)
    print(f'Search signature: search{sig}')
except Exception as e:
    print(f'Signature error: {e}')

# Test basic search
try:
    print("\nTesting basic search...")
    result = index.search(
        query={
            "inputs": {"text": "test"},
            "top_k": 1
        },
        namespace="default"
    )
    print("✅ Basic search works")
    print(f"Result type: {type(result)}")
    if hasattr(result, 'result'):
        print(f"Has 'result' attribute")
    if isinstance(result, dict):
        print(f"Keys: {list(result.keys())}")
except Exception as e:
    print(f"❌ Basic search failed: {e}")

# Test with filter
try:
    print("\nTesting search with filter...")
    result = index.search(
        query={
            "inputs": {"text": "test"},
            "top_k": 1
        },
        namespace="default",
        filter={"test": "value"}
    )
    print("✅ Search with filter works")
except Exception as e:
    print(f"❌ Search with filter failed: {e}")

print("Done!")
