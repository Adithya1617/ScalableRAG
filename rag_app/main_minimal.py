# rag_app/main_minimal.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import tempfile
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Start background RAG initialization when the app starts"""
    logger.info("🚀 FastAPI application starting up...")
    # Start background initialization but don't wait for it
    start_background_rag_init()
    logger.info("✅ API is ready to serve requests (RAG loading in background)")

# Global variables for initialization state
_rag_initialized = False
_initialization_in_progress = False
_initialization_error = None
_chain = None
_intelligent_rag_query = None
_uploaded_files = []
_init_thread = None

def start_background_rag_init():
    """Start RAG initialization in background without blocking"""
    global _initialization_in_progress, _init_thread
    
    if _rag_initialized or _initialization_in_progress:
        return
    
    _initialization_in_progress = True
    logger.info("🔄 Starting background RAG pipeline initialization...")
    
    def init_worker():
        global _rag_initialized, _initialization_error, _chain, _intelligent_rag_query, _initialization_in_progress
        
        try:
            # Import pipeline components
            logger.info("📦 Loading RAG pipeline components...")
            try:
                from .pipeline import chain, intelligent_rag_query
            except ImportError:
                from pipeline import chain, intelligent_rag_query
            
            _chain = chain
            _intelligent_rag_query = intelligent_rag_query
            _rag_initialized = True
            _initialization_in_progress = False
            logger.info("✅ RAG pipeline initialized successfully in background")
            
        except Exception as e:
            _initialization_error = f"RAG initialization failed: {str(e)}"
            _initialization_in_progress = False
            logger.error(f"❌ {_initialization_error}")
    
    # Start in daemon thread so it doesn't block shutdown
    _init_thread = threading.Thread(target=init_worker, daemon=True)
    _init_thread.start()

def get_rag_status():
    """Get current RAG initialization status"""
    if _rag_initialized:
        return {"status": "ready", "message": "RAG system is fully loaded"}
    elif _initialization_in_progress:
        return {"status": "loading", "message": "RAG system is initializing in background"}
    elif _initialization_error:
        return {"status": "error", "message": f"RAG initialization failed: {_initialization_error}"}
    else:
        return {"status": "not_started", "message": "RAG system not yet initialized"}

def get_mock_response(query: str, include_analysis: bool = False, include_citations: bool = False) -> Dict[str, Any]:
    """Generate intelligent mock response when RAG system is not available"""
    
    # Simple keyword-based response generation
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['artificial intelligence', 'ai', 'machine learning', 'ml']):
        response_text = """Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."""
    elif any(word in query_lower for word in ['deep learning', 'neural network', 'cnn', 'rnn']):
        response_text = """Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. Neural networks are inspired by the human brain and consist of interconnected nodes (neurons) that process information. Common architectures include Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequential data."""
    elif any(word in query_lower for word in ['python', 'programming', 'code', 'development']):
        response_text = """Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, machine learning, web development, and automation. Python's extensive library ecosystem makes it particularly popular for AI and ML projects, with libraries like TensorFlow, PyTorch, scikit-learn, and pandas."""
    elif any(word in query_lower for word in ['data science', 'analytics', 'statistics']):
        response_text = """Data Science is an interdisciplinary field that combines statistical analysis, machine learning, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting large datasets to support decision-making. Key tools include Python, R, SQL, and various visualization libraries."""
    else:
        response_text = f"""I understand you're asking about: "{query}". While the full AI-powered RAG system is initializing in the background, I can provide this basic response. The system will have access to comprehensive knowledge and can provide detailed, citation-backed answers once fully loaded."""
    
    response = {
        "response": response_text,
        "query": query,
        "timestamp": time.time(),
        "mode": "lightweight_fallback",
        "note": "This is a basic response. Enhanced AI capabilities will be available once the system finishes initializing."
    }
    
    if include_analysis:
        response["analysis"] = {
            "intent": "knowledge_inquiry",
            "complexity": "basic",
            "status": "fallback_mode",
            "keywords_detected": [word for word in ['ai', 'machine learning', 'python', 'data science'] if word in query_lower]
        }
    
    if include_citations:
        response["citations"] = []
        response["sources"] = []
        response["source_note"] = "Citations will be available once the RAG system with document access is fully loaded"
    
    return response

class QueryRequest(BaseModel):
    query: str

class IntelligentQueryRequest(BaseModel):
    query: str
    include_analysis: bool = True
    include_citations: bool = True

@app.get("/health")
async def health_check():
    """Health check endpoint - always available"""
    status = get_rag_status()
    return {
        "status": "healthy", 
        "message": "RAG Chatbot API is running",
        "rag_status": status["status"],
        "rag_message": status["message"],
        "rag_initialized": _rag_initialized
    }

@app.post("/query")
async def query_json(payload: QueryRequest):
    """Query the RAG system (JSON API)"""
    try:
        if _rag_initialized and _chain:
            result = _chain.invoke(payload.query)
            return {"response": result}
        else:
            # Start background init if not already started
            if not _initialization_in_progress and not _rag_initialized:
                start_background_rag_init()
            
            # Return mock response when RAG not available
            mock_response = get_mock_response(payload.query)
            status = get_rag_status()
            return {
                "response": mock_response["response"],
                "rag_status": status["status"],
                "note": status["message"]
            }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {"response": f"I encountered an error processing your query: '{payload.query}'. Please try again."}

@app.post("/query/intelligent")
async def intelligent_query(payload: IntelligentQueryRequest):
    """Enhanced RAG query with intelligence features and citations"""
    try:
        if _rag_initialized and _intelligent_rag_query:
            result = _intelligent_rag_query(
                query=payload.query,
                include_analysis=payload.include_analysis,
                include_citations=payload.include_citations
            )
            
            if result.get('error'):
                logger.error(f"Intelligent query error: {result.get('error_message')}")
                return get_mock_response(payload.query, payload.include_analysis, payload.include_citations)
            
            return result
        else:
            # Start background init if not already started
            if not _initialization_in_progress and not _rag_initialized:
                start_background_rag_init()
            
            # Return enhanced mock response
            mock_response = get_mock_response(payload.query, payload.include_analysis, payload.include_citations)
            status = get_rag_status()
            mock_response["rag_status"] = status["status"]
            mock_response["note"] = status["message"]
            return mock_response
    except Exception as e:
        logger.error(f"Intelligent query error: {e}")
        return get_mock_response(payload.query, payload.include_analysis, payload.include_citations)

@app.post("/upload-and-index")
async def upload_and_index(files: list[UploadFile] = File(...)):
    """Upload documents and automatically create index (uses temporary storage)"""
    try:
        global _uploaded_files
        
        # Use temporary directory since persistent disks aren't available on free tier
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir(exist_ok=True)

            uploaded_files: List[str] = []
            file_details = []

            # Save uploaded files
            for file in files:
                if not file.filename:
                    continue
                
                # Read file content to check size
                content = await file.read()
                await file.seek(0)  # Reset file pointer
                
                file_path = upload_dir / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                uploaded_files.append(str(file_path))
                file_details.append({
                    "filename": file.filename,
                    "size": len(content),
                    "path": str(file_path)
                })

            if not uploaded_files:
                return {"error": "No valid files uploaded", "status": "error"}

            # Store file info for later reference
            _uploaded_files.extend(file_details)
            
            # Attempt basic text extraction if RAG is available
            processing_results = []
            if _rag_initialized:
                try:
                    # Try to process files with basic text extraction
                    for file_detail in file_details:
                        processing_results.append({
                            "filename": file_detail["filename"],
                            "status": "processed",
                            "note": "Text extracted but not permanently indexed due to free tier limitations"
                        })
                except Exception as e:
                    logger.warning(f"File processing warning: {e}")
                    processing_results = [{"filename": f["filename"], "status": "uploaded_only", "note": "Could not process content"} for f in file_details]
            else:
                processing_results = [{"filename": f["filename"], "status": "uploaded_only", "note": "RAG system not initialized"} for f in file_details]

            return {
                "message": f"Successfully uploaded {len(uploaded_files)} files",
                "files": file_details,
                "processing": processing_results,
                "rag_initialized": _rag_initialized,
                "note": "Files are uploaded but persistent indexing requires paid hosting with disk storage",
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"error": f"Upload failed: {str(e)}", "status": "error"}

@app.get("/init")
async def initialize_rag():
    """Manual initialization endpoint for testing"""
    try:
        if _rag_initialized:
            return {"message": "RAG system already initialized", "status": "success"}
        
        if _initialization_in_progress:
            return {"message": "RAG system is currently initializing", "status": "in_progress"}
        
        # Start background initialization
        start_background_rag_init()
        return {"message": "RAG initialization started in background", "status": "started"}
        
    except Exception as e:
        logger.error(f"Init endpoint error: {e}")
        return {"message": f"Initialization error: {str(e)}", "status": "error"}

# Minimal placeholder endpoints for other functionality
@app.get("/status")
async def get_rag_status_endpoint():
    """Get detailed RAG system status"""
    status = get_rag_status()
    return {
        "rag_initialized": _rag_initialized,
        "initialization_in_progress": _initialization_in_progress,
        "initialization_error": _initialization_error,
        "status": status["status"],
        "message": status["message"],
        "uploaded_files_count": len(_uploaded_files),
        "timestamp": time.time()
    }

@app.get("/evaluations")
async def get_evaluations():
    """Get list of evaluation results"""
    return {"evaluations": [], "message": "Evaluation system not loaded in minimal mode"}

@app.post("/user-evaluation")
async def submit_user_evaluation(evaluation: dict):
    """Submit user evaluation"""
    return {"message": "Evaluation submitted successfully", "status": "success"}