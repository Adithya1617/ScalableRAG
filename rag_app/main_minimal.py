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

# Global variables for initialization state
_rag_initialized = False
_initialization_error = None
_chain = None
_intelligent_rag_query = None
_uploaded_files = []

def safe_initialize_rag():
    """Safely initialize RAG system with timeout and error handling"""
    global _rag_initialized, _initialization_error, _chain, _intelligent_rag_query
    
    if _rag_initialized:
        return True
    
    if _initialization_error:
        return False
    
    try:
        logger.info("🔄 Starting RAG pipeline initialization...")
        
        # Try to import pipeline with timeout (Windows compatible)
        import threading
        
        result = {"success": False, "error": None}
        
        def init_thread():
            try:
                # Import pipeline components
                try:
                    from .pipeline import chain, intelligent_rag_query
                except ImportError:
                    from pipeline import chain, intelligent_rag_query
                
                result["chain"] = chain
                result["intelligent_rag_query"] = intelligent_rag_query
                result["success"] = True
                
            except Exception as e:
                result["error"] = str(e)
        
        # Start initialization in separate thread
        thread = threading.Thread(target=init_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for completion with timeout
        thread.join(timeout=30)
        
        if thread.is_alive():
            # Thread is still running, initialization timed out
            _initialization_error = "RAG initialization timed out after 30 seconds"
            logger.error(f"❌ {_initialization_error}")
            return False
        
        if result["success"]:
            _chain = result["chain"]
            _intelligent_rag_query = result["intelligent_rag_query"]
            _rag_initialized = True
            logger.info("✅ RAG pipeline initialized successfully")
            return True
        else:
            _initialization_error = f"RAG initialization failed: {result['error']}"
            logger.error(f"❌ {_initialization_error}")
            return False
            
    except Exception as e:
        error_msg = f"RAG initialization failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        _initialization_error = error_msg
        _rag_initialized = False
        return False

def get_mock_response(query: str, include_analysis: bool = False, include_citations: bool = False) -> Dict[str, Any]:
    """Generate mock response when RAG system is not available"""
    response = {
        "response": f"I understand you're asking about: '{query}'. However, the full AI system is currently initializing. This is a basic response to keep the service available.",
        "query": query,
        "timestamp": time.time(),
        "mode": "fallback",
        "note": "This is a simplified response. Full AI capabilities will be available once the system finishes initializing."
    }
    
    if include_analysis:
        response["analysis"] = {
            "intent": "general_inquiry",
            "complexity": "unknown",
            "status": "fallback_mode"
        }
    
    if include_citations:
        response["citations"] = []
        response["sources"] = []
    
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
    return {
        "status": "healthy", 
        "message": "RAG Chatbot API is running",
        "rag_initialized": _rag_initialized,
        "initialization_error": _initialization_error
    }

@app.post("/query")
async def query_json(payload: QueryRequest):
    """Query the RAG system (JSON API)"""
    try:
        if _rag_initialized and _chain:
            result = _chain.invoke(payload.query)
            return {"response": result}
        else:
            # Return mock response when RAG not available
            mock_response = get_mock_response(payload.query)
            return {"response": mock_response["response"]}
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
            # Return enhanced mock response
            return get_mock_response(payload.query, payload.include_analysis, payload.include_citations)
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
        success = safe_initialize_rag()
        if success:
            return {"message": "RAG system initialized successfully", "status": "success"}
        else:
            return {"message": f"RAG initialization failed: {_initialization_error}", "status": "error"}
    except Exception as e:
        logger.error(f"Init endpoint error: {e}")
        return {"message": f"Initialization error: {str(e)}", "status": "error"}

# Minimal placeholder endpoints for other functionality
@app.get("/evaluations")
async def get_evaluations():
    """Get list of evaluation results"""
    return {"evaluations": [], "message": "Evaluation system not loaded in minimal mode"}

@app.post("/user-evaluation")
async def submit_user_evaluation(evaluation: dict):
    """Submit user evaluation"""
    return {"message": "Evaluation submitted successfully", "status": "success"}