#!/usr/bin/env python3
"""
RAG Data Ingestion Pipeline - Python Script Version
This script handles the complete RAG data ingestion process:
1. Load and chunk documents with smart metadata extraction
2. Create BM25 index
3. Upload to Pinecone with auto-embedding and enhanced metadata
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

class SmartDocumentProcessor:
    """Enhanced document processor with metadata extraction"""
    
    def __init__(self):
        self.document_types = {
            'policy': ['policy', 'procedure', 'guideline', 'regulation', 'compliance'],
            'manual': ['manual', 'handbook', 'guide', 'instruction', 'tutorial'],
            'report': ['report', 'analysis', 'summary', 'findings', 'results'],
            'contract': ['contract', 'agreement', 'terms', 'conditions', 'legal'],
            'technical': ['technical', 'specification', 'api', 'documentation', 'code'],
            'hr': ['hr', 'human resources', 'employee', 'benefits', 'payroll', 'insurance'],
            'financial': ['financial', 'budget', 'cost', 'revenue', 'expense', 'accounting']
        }
    
    def extract_metadata(self, content: str, filename: str, page_number: int = None) -> Dict[str, Any]:
        """Extract comprehensive metadata from document content"""
        metadata = {
            'filename': filename,
            'file_type': self._get_file_type(filename),
            'document_category': self._categorize_document(content, filename),
            'page_number': page_number,
            'word_count': len(content.split()),
            'char_count': len(content),
            'extracted_date': self._extract_date(content),
            'extracted_entities': self._extract_entities(content),
            'content_type': self._determine_content_type(content),
            'language': 'en',  # Default to English, can be enhanced with language detection
            'quality_score': self._assess_content_quality(content),
            'processed_date': datetime.now().isoformat()
        }
        
        # Extract title/heading if available
        title = self._extract_title(content)
        if title:
            metadata['title'] = title
            
        # Extract author information if available
        author = self._extract_author(content)
        if author:
            metadata['author'] = author
            
        return metadata
    
    def _get_file_type(self, filename: str) -> str:
        """Get file extension"""
        return Path(filename).suffix.lower().lstrip('.')
    
    def _categorize_document(self, content: str, filename: str) -> str:
        """Categorize document based on content and filename"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.document_types.items():
            score = 0
            for keyword in keywords:
                # Check filename (higher weight)
                if keyword in filename_lower:
                    score += 3
                # Check content (lower weight but more comprehensive)
                score += content_lower.count(keyword) * 0.1
            
            category_scores[category] = score
        
        # Return the category with highest score, default to 'general'
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0.5 else 'general'
        
        return 'general'
    
    def _extract_date(self, content: str) -> Optional[str]:
        """Extract dates from content"""
        # Common date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group()
        
        return None
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities (simplified version)"""
        entities = []
        
        # Extract potential organization names (capitalized words/phrases)
        org_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Organization|Department|Agency)\b'
        orgs = re.findall(org_pattern, content)
        entities.extend([org.strip() for org in orgs[:5]])  # Limit to top 5
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        entities.extend(emails[:3])  # Limit to top 3
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        phones = re.findall(phone_pattern, content)
        entities.extend(phones[:2])  # Limit to top 2
        
        return list(set(entities))  # Remove duplicates
    
    def _determine_content_type(self, content: str) -> str:
        """Determine the type of content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['q:', 'question:', 'answer:', 'faq']):
            return 'faq'
        elif any(word in content_lower for word in ['step 1', 'step 2', 'procedure', 'process']):
            return 'procedural'
        elif any(word in content_lower for word in ['section', 'chapter', 'article', 'clause']):
            return 'structured'
        elif len(content.split()) < 50:
            return 'brief'
        elif len(content.split()) > 500:
            return 'detailed'
        else:
            return 'standard'
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality (0.0 to 1.0)"""
        score = 0.5  # Base score
        
        # Length scoring
        word_count = len(content.split())
        if 50 <= word_count <= 1000:
            score += 0.2
        elif word_count > 20:
            score += 0.1
        
        # Sentence structure
        sentences = content.split('.')
        if len(sentences) > 2:
            score += 0.1
        
        # Presence of punctuation
        if any(punct in content for punct in ['.', '!', '?', ':']):
            score += 0.1
        
        # Capitalization (proper formatting)
        if content[0].isupper() if content else False:
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from content"""
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 100 and not line.endswith('.'):
                # Likely a title if it's short and doesn't end with period
                return line
        return None
    
    def _extract_author(self, content: str) -> Optional[str]:
        """Extract author information"""
        # Look for patterns like "By: John Doe" or "Author: Jane Smith"
        author_patterns = [
            r'(?:by|author|written by|created by):\s*([A-Za-z\s]+)',
            r'(?:by|author|written by|created by)\s+([A-Za-z\s]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                if len(author) < 50:  # Reasonable author name length
                    return author
        
        return None

def create_index(uploaded_files=None):
    """Create the complete RAG index from uploaded files"""
    print("Starting Enhanced RAG Data Ingestion Pipeline...")
    print("=" * 50)
    
    # Initialize smart document processor
    processor = SmartDocumentProcessor()
    
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
        
        # Step 3: Load and prepare documents from uploaded files with enhanced metadata
        print("\nLoading documents from uploaded files with smart metadata extraction...")
        
        if uploaded_files is None:
            # Fallback to sample.pdf if no files provided
            sample_pdf_path = "sample.pdf"
            if not os.path.exists(sample_pdf_path):
                raise FileNotFoundError(f"No uploaded files provided and sample.pdf not found at {sample_pdf_path}")
            uploaded_files = [sample_pdf_path]
        
        all_docs = []
        document_metadata = {}  # Store document-level metadata
        
        for file_path in uploaded_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            file_ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)
            print(f"Processing file: {filename}")
            
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
                
                # Extract metadata for each page/document
                for i, doc in enumerate(docs):
                    enhanced_metadata = processor.extract_metadata(
                        content=doc.page_content,
                        filename=filename,
                        page_number=i + 1 if len(docs) > 1 else None
                    )
                    # Merge with existing metadata
                    doc.metadata.update(enhanced_metadata)
                
                all_docs.extend(docs)
                document_metadata[filename] = {
                    'total_pages': len(docs),
                    'file_size': os.path.getsize(file_path),
                    'processed_date': datetime.now().isoformat()
                }
                print(f"Loaded {len(docs)} pages from {filename} with enhanced metadata")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded from uploaded files")
        
        print(f"Total documents loaded: {len(all_docs)}")
        
        # Step 4: Chunk the documents with metadata preservation
        print("\nChunking documents with metadata preservation...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks with preserved metadata")
        
        # Step 5: Prepare enhanced data for Pinecone with metadata
        print("\nPreparing enhanced data for Pinecone with metadata...")
        
        data = []
        for i, chunk in enumerate(chunks):
            # Get base metadata from chunk
            metadata = chunk.metadata.copy()
            
            # Add chunk-specific metadata
            chunk_metadata = {
                'chunk_id': f"chunk-{i}",
                'chunk_index': i,
                'chunk_length': len(chunk.page_content),
                'chunk_word_count': len(chunk.page_content.split())
            }
            metadata.update(chunk_metadata)
            
            # Prepare for Pinecone (only include searchable fields)
            pinecone_record = {
                "id": f"chunk-{i}",
                "text": chunk.page_content,  # Raw text for auto-embedding
                "filename": metadata.get('filename', 'unknown'),
                "document_category": metadata.get('document_category', 'general'),
                "content_type": metadata.get('content_type', 'standard'),
                "file_type": metadata.get('file_type', 'unknown'),
                "quality_score": metadata.get('quality_score', 0.5),
                "word_count": metadata.get('word_count', 0),
                "page_number": metadata.get('page_number', 0) or 0,
            }
            
            # Add optional fields if available
            if metadata.get('title'):
                pinecone_record['title'] = metadata['title']
            if metadata.get('author'):
                pinecone_record['author'] = metadata['author']
            if metadata.get('extracted_date'):
                pinecone_record['extracted_date'] = metadata['extracted_date']
            
            data.append(pinecone_record)

        print(f"Prepared {len(data)} chunks with enhanced metadata for Pinecone")
        print(f"Metadata fields: {list(data[0].keys())}")
        
        # Step 6: Create BM25 retriever with metadata
        print("\nCreating enhanced BM25 retriever...")
        # Store both text and metadata for BM25
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
        bm25_retriever.k = 4
        
        # Save BM25 retriever
        bm25_path = "bm25_index.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"Enhanced BM25 retriever created and saved to {bm25_path}")
        
        # Step 7: Clear existing Pinecone data
        print("\nClearing existing Pinecone data...")
        try:
            index.delete(delete_all=True, namespace="default")
            print("Cleared existing Pinecone data")
        except Exception as e:
            print(f"Could not clear existing data: {e}")
        
        # Step 8: Load enhanced data into Pinecone
        print("\nLoading enhanced data into Pinecone...")
        batch_size = 96  # Pinecone limit
        total_records = len(data)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        print(f"Processing {total_records} records in {num_batches} batches of {batch_size}")
        
        for i in range(0, total_records, batch_size):
            batch_num = (i // batch_size) + 1
            batch = data[i:i + batch_size]
            
            print(f"Uploading batch {batch_num}/{num_batches} ({len(batch)} records)...")
            
            try:
                index.upsert_records(
                    records=batch,
                    namespace="default"
                )
                print(f"Successfully uploaded batch {batch_num} with enhanced metadata")
            except Exception as e:
                print(f"Error uploading batch {batch_num}: {e}")
                raise
        
        print(f"Successfully uploaded all {total_records} records to Pinecone with enhanced metadata!")
        
        # Step 9: Verify the enhanced ingestion process
        print("\nVerifying the enhanced ingestion process...")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index Stats:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Namespaces: {list(stats.namespaces.keys())}")
        print(f"   Dimension: {stats.dimension}")
        
        # Test a sample query with metadata filtering
        print(f"\nTesting sample query with enhanced metadata...")
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
            sample_metadata = hits[0].get('fields', {})
            print(f"Sample metadata fields: {list(sample_metadata.keys())}")
            print(f"Document category: {sample_metadata.get('document_category', 'N/A')}")
            print(f"Content type: {sample_metadata.get('content_type', 'N/A')}")
        
        # Save document metadata summary
        metadata_summary = {
            'total_documents': len(uploaded_files),
            'total_chunks': len(chunks),
            'document_metadata': document_metadata,
            'categories_found': list(set(d.get('document_category', 'general') for d in data)),
            'content_types_found': list(set(d.get('content_type', 'standard') for d in data)),
            'processing_date': datetime.now().isoformat()
        }
        
        with open('document_metadata_summary.json', 'w') as f:
            import json
            json.dump(metadata_summary, f, indent=2)
        
        print("\n" + "=" * 50)
        print("Enhanced RAG Data Ingestion Pipeline Completed Successfully!")
        print("✅ Documents chunked and processed with smart metadata")
        print("✅ BM25 index created with metadata support")
        print("✅ Pinecone index populated with enhanced metadata")
        print("✅ Document categorization and content analysis completed")
        print("✅ System ready for enhanced RAG queries with filtering!")
        print(f"📊 Categories found: {', '.join(metadata_summary['categories_found'])}")
        print(f"📄 Content types: {', '.join(metadata_summary['content_types_found'])}")
        
        return True
        
    except Exception as e:
        print(f"\nError in Enhanced RAG Data Ingestion Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_index()
    sys.exit(0 if success else 1)
