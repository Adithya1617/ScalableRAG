#!/usr/bin/env python3
"""
RAG Data Ingestion Pipeline - Python Script Version
This script handles the complete RAG data ingestion process:
1. Load and chunk documents
2. Create BM25 index
3. Upload to Pinecone with auto-embedding
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def create_index(uploaded_files=None):
    """Create the complete RAG index from uploaded files"""
    print("Starting RAG Data Ingestion Pipeline...")
    print("=" * 50)
    
    try:
        # Step 1: Import required libraries
        print("Importing required libraries...")
        from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.retrievers import BM25Retriever
        from pinecone import Pinecone
        import pickle
        import os
        
        print("Libraries imported successfully!")
        
        # Step 2: Initialize Pinecone
        print("\nInitializing Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "quickstart")
        
        # Create index with llama-text-embed-v2 model for automatic embedding
        if index_name not in pc.list_indexes().names():
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "text"}
                }
            )
            print(f"Created Pinecone index '{index_name}' with llama-text-embed-v2 auto-embedding")
        else:
            print(f"Using existing Pinecone index '{index_name}' with llama-text-embed-v2 auto-embedding")

        index = pc.Index(index_name)
        print(f"Pinecone will automatically generate embeddings using llama-text-embed-v2")
        
        # Step 3: Load and prepare documents from uploaded files
        print("\nLoading documents from uploaded files...")
        
        if uploaded_files is None:
            # Fallback to sample.pdf if no files provided
            sample_pdf_path = "sample.pdf"
            if not os.path.exists(sample_pdf_path):
                raise FileNotFoundError(f"No uploaded files provided and sample.pdf not found at {sample_pdf_path}")
            uploaded_files = [sample_pdf_path]
        
        all_docs = []
        
        for file_path in uploaded_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            file_ext = os.path.splitext(file_path)[1].lower()
            print(f"Processing file: {os.path.basename(file_path)}")
            
            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_ext in ['.txt']:
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_ext in ['.doc', '.docx']:
                    loader = UnstructuredWordDocumentLoader(file_path)
                else:
                    print(f"Unsupported file type: {file_ext}, skipping...")
                    continue
                
                docs = loader.load()
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded from uploaded files")
        
        print(f"Total documents loaded: {len(all_docs)}")
        
        # Step 4: Chunk the documents
        print("\nChunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks")
        
        # Step 5: Prepare raw text data for Pinecone auto-embedding
        print("\nPreparing raw text data for Pinecone auto-embedding...")
        
        # Prepare data in Pinecone format - NO LOCAL EMBEDDINGS!
        data = []
        for i, chunk in enumerate(chunks):
            # Add source file information to metadata
            source_file = os.path.basename(chunk.metadata.get('source', 'unknown'))
            data.append({
                "id": f"chunk-{i}",
                "text": chunk.page_content  # Raw text - Pinecone will embed this automatically
            })

        print(f"Prepared {len(data)} chunks with raw text for Pinecone auto-embedding")
        print(f"Pinecone will automatically generate llama-text-embed-v2 embeddings")
        print(f"Data structure matches Pinecone configuration exactly")
        print(f"Sample record ID: {data[0]['id']}")
        
        # Step 6: Create BM25 retriever
        print("\nCreating BM25 retriever...")
        texts = [chunk.page_content for chunk in chunks]
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 4
        
        # Save BM25 retriever
        bm25_path = "bm25_index.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"BM25 retriever created and saved to {bm25_path}")
        
        # Step 7: Clear existing Pinecone data
        print("\nClearing existing Pinecone data...")
        try:
            index.delete(delete_all=True, namespace="default")
            print("Cleared existing Pinecone data")
        except Exception as e:
            print(f"Could not clear existing data: {e}")
        
        # Step 8: Load raw text into Pinecone
        print("\nLoading raw text into Pinecone...")
        batch_size = 96  # Pinecone limit
        total_records = len(data)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        print(f"Processing {total_records} records in {num_batches} batches of {batch_size}")
        
        for i in range(0, total_records, batch_size):
            batch_num = (i // batch_size) + 1
            batch = data[i:i + batch_size]
            
            print(f"Uploading batch {batch_num}/{num_batches} ({len(batch)} records)...")
            print(f"Pinecone is generating llama-text-embed-v2 embeddings automatically...")
            
            try:
                index.upsert_records(
                    records=batch,
                    namespace="default"
                )
                print(f"Successfully uploaded batch {batch_num}")
            except Exception as e:
                print(f"Error uploading batch {batch_num}: {e}")
                raise
        
        print(f"Successfully uploaded all {total_records} records to Pinecone!")
        
        # Step 9: Verify the ingestion process
        print("\nVerifying the ingestion process...")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index Stats:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Namespaces: {list(stats.namespaces.keys())}")
        print(f"   Dimension: {stats.dimension}")
        
        # Test a sample query
        print(f"\nTesting sample query with auto-generated llama-text-embed-v2 embeddings...")
        test_query = "What is the main topic of this document?"
        results = index.search(
            query={
                "inputs": {"text": test_query},
                "top_k": 3
            },
            namespace="default"
        )
        
        hits = results.get('result', {}).get('hits', [])
        print(f"Test query successful! Found {len(hits)} results")
        
        if hits:
            print(f"Top result score: {hits[0].get('_score', 0):.3f}")
            sample_text = hits[0].get('fields', {}).get('text', '')[:100]
            print(f"Sample text length: {len(sample_text)} characters")
        
        print("\n" + "=" * 50)
        print("RAG Data Ingestion Pipeline Completed Successfully!")
        print("Documents chunked and processed")
        print("BM25 index created and saved")
        print("Pinecone index populated with llama-text-embed-v2 embeddings")
        print("System ready for RAG queries!")
        
        return True
        
    except Exception as e:
        print(f"\nError in RAG Data Ingestion Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_index()
    sys.exit(0 if success else 1)
