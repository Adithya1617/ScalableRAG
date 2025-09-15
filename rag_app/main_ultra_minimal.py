# rag_app/main_ultra_minimal.py
"""
Ultra-minimal RAG backend for Render free tier
- No BM25 indexing
- No heavy ML dependencies 
- No sentence transformers
- Lightweight responses only
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ultra-Minimal RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
_uploaded_files = []
_knowledge_base = {
    "ai": {
        "title": "Artificial Intelligence",
        "content": """Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.""",
        "keywords": ["artificial intelligence", "ai", "machine learning", "ml", "intelligent", "automation"]
    },
    "deep_learning": {
        "title": "Deep Learning",
        "content": """Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. Neural networks are inspired by the human brain and consist of interconnected nodes (neurons) that process information. Common architectures include Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequential data.""",
        "keywords": ["deep learning", "neural network", "cnn", "rnn", "layers", "nodes", "neurons"]
    },
    "python": {
        "title": "Python Programming",
        "content": """Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, machine learning, web development, and automation. Python's extensive library ecosystem makes it particularly popular for AI and ML projects, with libraries like TensorFlow, PyTorch, scikit-learn, and pandas.""",
        "keywords": ["python", "programming", "code", "development", "library", "tensorflow", "pytorch"]
    },
    "data_science": {
        "title": "Data Science",
        "content": """Data Science is an interdisciplinary field that combines statistical analysis, machine learning, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting large datasets to support decision-making. Key tools include Python, R, SQL, and various visualization libraries.""",
        "keywords": ["data science", "analytics", "statistics", "analysis", "datasets", "insights"]
    },
    "rag": {
        "title": "Retrieval-Augmented Generation",
        "content": """Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and contextual responses. RAG systems typically use vector databases and embedding models to find relevant information.""",
        "keywords": ["rag", "retrieval", "generation", "vector", "embedding", "knowledge base"]
    },
    "fastapi": {
        "title": "FastAPI Framework",
        "content": """FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. It provides automatic interactive API documentation, high performance, and easy-to-use features. FastAPI is built on Starlette for the web parts and Pydantic for the data parts.""",
        "keywords": ["fastapi", "api", "web framework", "python", "starlette", "pydantic"]
    }
}

class QueryRequest(BaseModel):
    query: str

class IntelligentQueryRequest(BaseModel):
    query: str
    include_analysis: bool = True
    include_citations: bool = True

def simple_keyword_search(query: str) -> List[Dict[str, Any]]:
    """Simple keyword-based search through knowledge base"""
    query_lower = query.lower()
    results = []
    
    for topic_id, topic_data in _knowledge_base.items():
        score = 0
        matched_keywords = []
        
        # Check for keyword matches
        for keyword in topic_data["keywords"]:
            if keyword in query_lower:
                score += 1
                matched_keywords.append(keyword)
        
        # Check title match
        if topic_data["title"].lower() in query_lower:
            score += 2
        
        # Add partial matches for broader coverage
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3:  # Only check meaningful words
                for keyword in topic_data["keywords"]:
                    if word in keyword or keyword in word:
                        score += 0.5
        
        if score > 0:
            results.append({
                "topic_id": topic_id,
                "title": topic_data["title"],
                "content": topic_data["content"],
                "score": score,
                "matched_keywords": matched_keywords
            })
    
    # Sort by score and return top 3
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]

def generate_response(query: str, search_results: List[Dict[str, Any]], include_analysis: bool = False) -> Dict[str, Any]:
    """Generate response based on search results"""
    
    if not search_results:
        # Fallback response for unknown queries
        response_text = f"""I understand you're asking about: "{query}". While I have a basic knowledge base covering AI, machine learning, Python, data science, and related topics, I don't have specific information about this query. Could you try asking about artificial intelligence, machine learning, Python programming, data science, or RAG systems?"""
        
        return {
            "response": response_text,
            "query": query,
            "source": "fallback",
            "confidence": 0.1,
            "timestamp": time.time()
        }
    
    # Use the best match
    best_match = search_results[0]
    
    # Generate contextual response
    if best_match["score"] >= 2:
        # High confidence - use the content directly
        response_text = f"""Based on my knowledge about {best_match['title']}:\n\n{best_match['content']}"""
        confidence = min(0.9, best_match["score"] / 3)
    elif best_match["score"] >= 1:
        # Medium confidence - provide relevant information
        response_text = f"""Regarding your question about "{query}", here's what I know about {best_match['title']}:\n\n{best_match['content']}\n\nThis information might be relevant to your question, though you may want to be more specific for a more targeted answer."""
        confidence = min(0.7, best_match["score"] / 2)
    else:
        # Low confidence - provide basic information
        response_text = f"""I found some potentially relevant information about {best_match['title']}:\n\n{best_match['content']}\n\nHowever, this may not directly answer your question about "{query}". Could you provide more specific details?"""
        confidence = min(0.5, best_match["score"])
    
    response = {
        "response": response_text,
        "query": query,
        "source": best_match["title"],
        "confidence": confidence,
        "matched_keywords": best_match["matched_keywords"],
        "timestamp": time.time()
    }
    
    if include_analysis:
        response["analysis"] = {
            "intent": "knowledge_inquiry",
            "complexity": "basic" if best_match["score"] >= 2 else "unclear",
            "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
            "keywords_detected": best_match["matched_keywords"],
            "search_results_count": len(search_results)
        }
    
    return response

@app.on_event("startup")
async def startup_event():
    """Application startup - no heavy initialization needed"""
    logger.info("🚀 Ultra-minimal RAG API starting up...")
    logger.info("✅ Knowledge base loaded with basic topics")
    logger.info("🎯 Ready to serve lightweight responses!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Ultra-minimal RAG API is running",
        "mode": "lightweight",
        "knowledge_topics": list(_knowledge_base.keys()),
        "memory_efficient": True,
        "timestamp": time.time()
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "mode": "ultra_minimal",
        "heavy_dependencies": False,
        "bm25_enabled": False,
        "vector_search_enabled": False,
        "knowledge_base_topics": len(_knowledge_base),
        "uploaded_files": len(_uploaded_files),
        "memory_footprint": "minimal",
        "timestamp": time.time()
    }

@app.post("/query")
async def query_simple(payload: QueryRequest):
    """Simple query endpoint"""
    try:
        search_results = simple_keyword_search(payload.query)
        response_data = generate_response(payload.query, search_results)
        
        return {
            "response": response_data["response"],
            "confidence": response_data["confidence"],
            "source": response_data.get("source", "knowledge_base")
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {
            "response": f"I encountered an error processing your query: '{payload.query}'. Please try again.",
            "confidence": 0.0,
            "source": "error_handler"
        }

@app.post("/query/intelligent")
async def intelligent_query(payload: IntelligentQueryRequest):
    """Enhanced query with analysis"""
    try:
        search_results = simple_keyword_search(payload.query)
        response_data = generate_response(
            payload.query, 
            search_results, 
            include_analysis=payload.include_analysis
        )
        
        if payload.include_citations:
            # Add simple citations
            citations = []
            if search_results:
                for result in search_results[:2]:  # Top 2 sources
                    citations.append({
                        "title": result["title"],
                        "relevance_score": result["score"],
                        "matched_keywords": result["matched_keywords"]
                    })
            response_data["citations"] = citations
            response_data["sources"] = [c["title"] for c in citations]
        
        return response_data
        
    except Exception as e:
        logger.error(f"Intelligent query error: {e}")
        return {
            "response": f"I encountered an error processing your query: '{payload.query}'. Please try again.",
            "confidence": 0.0,
            "source": "error_handler",
            "analysis": {"error": str(e)} if payload.include_analysis else None,
            "citations": [] if payload.include_citations else None
        }

@app.post("/upload-and-index")
async def upload_and_index(files: list[UploadFile] = File(...)):
    """Upload files (minimal processing)"""
    try:
        uploaded_details = []
        
        for file in files:
            if not file.filename:
                continue
            
            # Read file content
            content = await file.read()
            file_info = {
                "filename": file.filename,
                "size": len(content),
                "type": file.content_type,
                "uploaded_at": time.time()
            }
            
            # Simple text extraction for .txt files
            if file.filename.endswith('.txt'):
                try:
                    text_content = content.decode('utf-8')
                    file_info["text_preview"] = text_content[:200] + "..." if len(text_content) > 200 else text_content
                    file_info["processed"] = True
                except:
                    file_info["processed"] = False
                    file_info["note"] = "Could not extract text"
            else:
                file_info["processed"] = False
                file_info["note"] = "Only .txt files are processed in minimal mode"
            
            _uploaded_files.append(file_info)
            uploaded_details.append(file_info)
        
        return {
            "message": f"Successfully uploaded {len(uploaded_details)} files",
            "files": uploaded_details,
            "mode": "minimal_processing",
            "note": "Full indexing requires heavy dependencies not available in ultra-minimal mode",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {
            "error": f"Upload failed: {str(e)}",
            "status": "error"
        }

@app.get("/init")
async def initialize():
    """Initialization endpoint (instant in minimal mode)"""
    return {
        "message": "Ultra-minimal mode is always ready - no initialization needed",
        "status": "ready",
        "mode": "ultra_minimal",
        "initialization_time": 0.0
    }

@app.get("/evaluations")
async def get_evaluations():
    """Get evaluations (minimal mode)"""
    return {
        "evaluations": [],
        "message": "Evaluation system not available in ultra-minimal mode",
        "mode": "minimal"
    }

@app.post("/user-evaluation")
async def submit_user_evaluation(evaluation: dict):
    """Submit user evaluation"""
    return {
        "message": "Thank you for your feedback!",
        "status": "received",
        "note": "Evaluations are logged but not stored in ultra-minimal mode"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)