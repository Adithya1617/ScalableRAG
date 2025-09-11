# rag_app/main.py
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pipeline import chain
import os
import shutil
from pathlib import Path
import subprocess
import sys

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Point to templates folder OUTSIDE rag_app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Optional: if you have a static folder, mount it like this
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chatbot frontend"""
    return templates.TemplateResponse("chatbot.html", {"request": request})

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_json(payload: QueryRequest):
    """Query the RAG system (JSON API)"""
    try:
        result = chain.invoke(payload.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-and-index")
async def upload_and_index(files: list[UploadFile] = File(...)):
    """Upload documents and automatically create index"""
    try:
        # Create uploads directory if it doesn't exist
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        upload_dir = Path(root_dir) / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        # Clear existing uploads
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            upload_dir.mkdir()
        
        uploaded_files = []
        
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
        
        # Run the indexing process
        try:
            # Import and call the create_index function directly with uploaded files
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.append(root_dir)
            
            from load_index.make import create_index
            
            # Call create_index with the uploaded files
            success = create_index(uploaded_files)
            
            if not success:
                raise Exception("Indexing process failed")
                
        except Exception as e:
            # Fallback: try to run the script directly
            try:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                make_py_path = os.path.join(root_dir, "load_index", "make.py")
                
                # Create a temporary script that passes the uploaded files
                temp_script = f"""
import sys
sys.path.append('{root_dir}')
from load_index.make import create_index
uploaded_files = {uploaded_files}
success = create_index(uploaded_files)
sys.exit(0 if success else 1)
"""
                
                with open("temp_index.py", "w") as f:
                    f.write(temp_script)
                
                result = subprocess.run([
                    sys.executable, "temp_index.py"
                ], capture_output=True, text=True, cwd=root_dir)
                
                # Clean up temp file
                if os.path.exists("temp_index.py"):
                    os.remove("temp_index.py")
                
                if result.returncode != 0:
                    raise Exception(f"Indexing failed: {result.stderr}")
                    
            except Exception as e2:
                raise Exception(f"Both indexing methods failed. Error 1: {e}, Error 2: {e2}")
        
        return {
            "message": f"Successfully uploaded and indexed {len(uploaded_files)} files",
            "files": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG Chatbot API is running"}
