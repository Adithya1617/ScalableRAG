# rag_app/main_minimal.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import List

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy initialization
_chain = None
_intelligent_rag_query = None
_initialization_error = None

def get_chain():
    """Lazy initialization of the RAG chain"""
    global _chain, _intelligent_rag_query, _initialization_error
    
    if _chain is None and _initialization_error is None:
        try:
            print("🔄 Initializing RAG pipeline...")
            # Import pipeline components only when needed
            try:
                from .pipeline import chain, intelligent_rag_query
            except ImportError:
                from pipeline import chain, intelligent_rag_query
            
            _chain = chain
            _intelligent_rag_query = intelligent_rag_query
            print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            _initialization_error = str(e)
            print(f"❌ RAG pipeline initialization failed: {e}")
    
    if _initialization_error:
        raise HTTPException(status_code=503, detail=f"RAG system not available: {_initialization_error}")
    
    return _chain, _intelligent_rag_query

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
        "rag_initialized": _chain is not None,
        "initialization_error": _initialization_error
    }

@app.post("/query")
async def query_json(payload: QueryRequest):
    """Query the RAG system (JSON API)"""
    try:
        chain, _ = get_chain()
        result = chain.invoke(payload.query)
        return {"response": result}
    except Exception as e:
        if "503" in str(e):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/intelligent")
async def intelligent_query(payload: IntelligentQueryRequest):
    """Enhanced RAG query with intelligence features and citations"""
    try:
        _, intelligent_rag_query = get_chain()
        result = intelligent_rag_query(
            query=payload.query,
            include_analysis=payload.include_analysis,
            include_citations=payload.include_citations
        )
        
        if result.get('error'):
            raise HTTPException(status_code=500, detail=result.get('error_message'))
        
        return result
    except Exception as e:
        if "503" in str(e):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-and-index")
async def upload_and_index(files: list[UploadFile] = File(...)):
    """Upload documents and automatically create index (uses temporary storage)"""
    try:
        # Use temporary directory since persistent disks aren't available on free tier
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir(exist_ok=True)

            uploaded_files: List[str] = []

            # Save uploaded files
            for file in files:
                if not file.filename:
                    continue
                file_path = upload_dir / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(str(file_path))

            if not uploaded_files:
                raise HTTPException(status_code=400, detail="No valid files uploaded")

            # Note: Indexing would need to be implemented to work with temporary files
            # For now, return success but indicate limitation
            return {
                "message": f"Files uploaded to temporary storage: {len(uploaded_files)} files",
                "files": [Path(f).name for f in uploaded_files],
                "note": "Indexing functionality requires persistent storage (not available on free tier)"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/init")
async def initialize_rag():
    """Manual initialization endpoint for testing"""
    try:
        chain, intelligent_rag_query = get_chain()
        return {"message": "RAG system initialized successfully"}
    except Exception as e:
        raise e

# Minimal placeholder endpoints for other functionality
@app.get("/evaluations")
async def get_evaluations():
    """Get list of evaluation results"""
    return {"evaluations": [], "message": "Evaluation system not loaded in minimal mode"}

@app.post("/user-evaluation")
async def submit_user_evaluation(evaluation: dict):
    """Submit user evaluation"""
    return {"message": "Evaluation submitted successfully", "status": "success"}