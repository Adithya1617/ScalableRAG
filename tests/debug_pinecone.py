#!/usr/bin/env python3
"""
Debug Pinecone response structure
"""

import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
index = pc.Index(index_name)

print("🔍 Testing Pinecone response structure...")

try:
    # Test search
    results = index.search(
        namespace="default",
        query={
            "inputs": {"text": "test query"},
            "top_k": 2
        }
    )
    
    print(f"Results type: {type(results)}")
    print(f"Results attributes: {dir(results)}")
    
    if hasattr(results, 'result'):
        print(f"Results.result type: {type(results.result)}")
        print(f"Results.result attributes: {dir(results.result)}")
        
        if hasattr(results.result, 'hits'):
            hits = results.result.hits
            print(f"Hits type: {type(hits)}")
            print(f"Number of hits: {len(hits)}")
            
            if hits:
                hit = hits[0]
                print(f"Hit type: {type(hit)}")
                print(f"Hit attributes: {dir(hit)}")
                print(f"Hit ID: {getattr(hit, 'id', 'NO ID')}")
                print(f"Hit score: {getattr(hit, 'score', 'NO SCORE')}")
                
                if hasattr(hit, 'fields'):
                    print(f"Hit fields type: {type(hit.fields)}")
                    print(f"Hit fields attributes: {dir(hit.fields)}")
                    
                    if hasattr(hit.fields, 'text'):
                        print(f"Text found: {hit.fields.text[:100]}...")
                    else:
                        print("No text attribute in fields")
                        
                    # Try to access fields as dict
                    try:
                        if hasattr(hit.fields, '__dict__'):
                            fields_dict = hit.fields.__dict__
                            print(f"Fields as dict: {list(fields_dict.keys())}")
                        elif isinstance(hit.fields, dict):
                            print(f"Fields is dict: {list(hit.fields.keys())}")
                    except Exception as e:
                        print(f"Error accessing fields: {e}")
                else:
                    print("No fields attribute in hit")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
