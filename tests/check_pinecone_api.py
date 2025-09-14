#!/usr/bin/env python3
"""
Check Pinecone auto-embedding API structure
"""

import os
from pinecone import Pinecone
from dotenv import load_dotenv
import inspect

def check_pinecone_api():
    """Check the Pinecone API structure for auto-embedding"""
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
        
        print('=== Pinecone Index Information ===')
        print(f'Index Name: {index_name}')
        
        # Get index
        index = pc.Index(index_name)
        print(f'Index Object Type: {type(index)}')
        
        # Check index description
        try:
            desc = index.describe_index_stats()
            print(f'Index Stats: {desc}')
        except Exception as e:
            print(f'Error getting stats: {e}')
        
        # Check available methods
        print('\n=== Available Index Methods ===')
        methods = [method for method in dir(index) if not method.startswith('_')]
        for method in sorted(methods):
            print(f'- {method}')
        
        # Check search method signature
        print('\n=== Search Method Signature ===')
        try:
            search_signature = inspect.signature(index.search)
            print(f'search{search_signature}')
        except Exception as e:
            print(f'Error getting signature: {e}')
        
        # Try a simple search to see the expected format
        print('\n=== Test Search Format ===')
        try:
            # Test the current format we're using
            print("Testing current format...")
            results = index.search(
                query={
                    "inputs": {"text": "test query"},
                    "top_k": 1
                },
                namespace="default"
            )
            print(f"✅ Current format works: {type(results)}")
            print(f"Result keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            
        except Exception as e:
            print(f"❌ Current format failed: {e}")
            
        # Try with filter to see if it's supported
        print('\n=== Test Filter Support ===')
        try:
            print("Testing with filter...")
            results = index.search(
                query={
                    "inputs": {"text": "test query"},
                    "top_k": 1
                },
                namespace="default",
                filter={"document_category": {"$eq": "general"}}
            )
            print(f"✅ Filter format works: {type(results)}")
            
        except Exception as e:
            print(f"❌ Filter format failed: {e}")
            
        # Check if there's a different search method
        print('\n=== Alternative Search Methods ===')
        search_methods = [method for method in dir(index) if 'search' in method.lower()]
        for method in search_methods:
            print(f'- {method}')
            try:
                sig = inspect.signature(getattr(index, method))
                print(f'  {method}{sig}')
            except:
                print(f'  {method} - signature unavailable')
        
        # Check Pinecone documentation format
        print('\n=== Testing Alternative Formats ===')
        
        # Format 1: Standard search
        try:
            print("Testing standard search format...")
            results = index.search(
                vector=None,  # No vector for auto-embedding
                top_k=1,
                namespace="default"
            )
            print(f"✅ Standard format works")
        except Exception as e:
            print(f"❌ Standard format failed: {e}")
            
        # Format 2: Query search
        try:
            print("Testing query search format...")
            results = index.search(
                query="test query",
                top_k=1,
                namespace="default"
            )
            print(f"✅ Query format works")
        except Exception as e:
            print(f"❌ Query format failed: {e}")
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return False
    
    return True

if __name__ == "__main__":
    check_pinecone_api()
