# rag_app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import chain, intelligent_rag_query
import os
import shutil
from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

# Helper function for JSON serialization of numpy types
def convert_numpy_to_python(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    else:
        return obj

# Import evaluation components (optional)
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation.metrics import RAGEvaluator
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"Evaluation components not available: {e}")
    EVALUATION_AVAILABLE = False

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Request Models ======

class QueryRequest(BaseModel):
    query: str

class IntelligentQueryRequest(BaseModel):
    query: str
    include_analysis: bool = True
    include_citations: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None

class HumanFeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int  # 1-5 stars
    feedback_text: Optional[str] = ""
    dimensions: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = None

class GroundTruthValidationRequest(BaseModel):
    query: str
    expected_answer: str
    complexity: str  # simple, moderate, complex
    evaluation_dimensions: List[str]
    relevant_document_sections: List[str]
    key_facts: List[str]
    validator_id: str

class EvaluationRunRequest(BaseModel):
    evaluation_type: str  # quick, comprehensive, performance, component_analysis
    options: Optional[Dict[str, Any]] = None

# Initialize global evaluator
global_evaluator = None

def get_or_create_evaluator():
    """Get or create the global evaluator instance"""
    global global_evaluator
    if global_evaluator is None and EVALUATION_AVAILABLE:
        try:
            # Import retrievers from pipeline
            from pipeline import final_retriever, retriever_vector
            import pickle
            
            # Load BM25 retriever
            bm25_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "load_index", "bm25_index.pkl")
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            
            retrievers = {
                'bm25': bm25_retriever,
                'vector': retriever_vector,
                'hybrid_reranked': final_retriever
            }
            
            global_evaluator = RAGEvaluator(chain, retrievers)
            print("✅ Global evaluator initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize global evaluator: {e}")
    
    return global_evaluator

@app.post("/query")
async def query_json(payload: QueryRequest):
    """Query the RAG system (JSON API)"""
    try:
        result = chain.invoke(payload.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/intelligent")
async def intelligent_query(payload: IntelligentQueryRequest):
    """Enhanced RAG query with intelligence features and citations"""
    try:
        result = intelligent_rag_query(
            query=payload.query,
            include_analysis=payload.include_analysis,
            include_citations=payload.include_citations
        )
        
        if result.get('error'):
            raise HTTPException(status_code=500, detail=result.get('error_message'))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Removed non-operational query metadata endpoints (analyze/types) to keep only core operations

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

        # Run the indexing process
        try:
            # Import and call the create_index function directly with uploaded files
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.append(root_dir)
            from load_index.make import create_index
            success = create_index(uploaded_files)
            if not success:
                raise Exception("Indexing process failed")
        except Exception as e:
            # Fallback: try to run the script directly
            try:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                # Create a temporary script that passes the uploaded files
                temp_script = f"""
import sys
sys.path.append('{root_dir}')
from load_index.make import create_index
uploaded_files = {uploaded_files}
success = create_index(uploaded_files)
sys.exit(0 if success else 1)
"""
                temp_path = Path(root_dir) / "temp_index.py"
                with open(temp_path, "w") as f:
                    f.write(temp_script)
                result = subprocess.run([sys.executable, str(temp_path)], capture_output=True, text=True, cwd=root_dir)
                if temp_path.exists():
                    os.remove(temp_path)
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

@app.get("/evaluations")
async def get_evaluations():
    """Get list of all evaluation results"""
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_dir = Path(root_dir) / "evaluation_results"
        
        if not eval_dir.exists():
            return {"evaluations": []}
        
        evaluations = []
        for file_path in eval_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                evaluations.append({
                    "filename": file_path.name,
                    "timestamp": data.get("timestamp"),
                    "test_dataset_size": data.get("test_dataset_size"),
                    "avg_response_time": data.get("response_quality", {}).get("avg_response_time"),
                    "avg_llm_judge_score": data.get("response_quality", {}).get("avg_llm_judge_score"),
                    "rouge1": data.get("response_quality", {}).get("rouge_scores", {}).get("rouge1")
                })
            except Exception as e:
                continue
        
        # Sort by timestamp descending
        evaluations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {"evaluations": evaluations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluations/{filename}")
async def get_evaluation(filename: str):
    """Get specific evaluation result"""
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = Path(root_dir) / "evaluation_results" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user-evaluation")
async def submit_user_evaluation(evaluation: dict):
    """Submit user evaluation for chatbot response"""
    try:
        # For now, just return success (you can implement storage later)
        # In a real implementation, you'd save this to a database
        return {"message": "Evaluation submitted successfully", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EvaluationRequest(BaseModel):
    evaluation_type: str
    options: dict = {}

@app.post("/run-evaluation")
async def run_evaluation(request: EvaluationRequest):
    """Run evaluation based on selected criteria"""
    try:
        import asyncio
        import time
        from datetime import datetime
        
        # Create a unique evaluation ID
        eval_id = f"eval_{int(time.time())}"
        
        # Start evaluation in background
        asyncio.create_task(execute_evaluation(request.evaluation_type, request.options, eval_id))
        
        return {
            "message": "Evaluation started successfully",
            "evaluation_id": eval_id,
            "status": "running"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluation-status/{eval_id}")
async def get_evaluation_status(eval_id: str):
    """Get status of running evaluation"""
    try:
        # Check if evaluation results exist
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        status_file = Path(root_dir) / "evaluation_results" / f"{eval_id}_status.json"
        
        if status_file.exists():
            with open(status_file, "r") as f:
                status = json.load(f)
            return status
        else:
            return {"status": "running", "progress": "Initializing..."}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def execute_evaluation(evaluation_type: str, options: dict, eval_id: str):
    """Execute evaluation asynchronously"""
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        status_file = Path(root_dir) / "evaluation_results" / f"{eval_id}_status.json"
        
        # Update status
        def update_status(status, progress, results=None):
            status_data = {
                "status": status,
                "progress": progress,
                "timestamp": datetime.now().isoformat(),
                "evaluation_id": eval_id
            }
            if results:
                status_data["results"] = results
            with open(status_file, "w") as f:
                json.dump(status_data, f)
        
        update_status("running", "Setting up evaluation...")
        
        # Import evaluation components
        sys.path.append(root_dir)
        from evaluation.evaluator import setup_evaluation
        from evaluation.test_data import load_test_dataset
        
        update_status("running", "Loading RAG components...")
        chain, retrievers = setup_evaluation()
        
        if not chain or not retrievers:
            update_status("error", "Failed to load RAG components")
            return
        
        update_status("running", "Loading test dataset...")
        test_cases = load_test_dataset()
        
        if not test_cases:
            update_status("error", "No test dataset available")
            return
        
        # Filter test cases based on evaluation type
        if evaluation_type == "quick":
            test_cases = test_cases[:5]
            update_status("running", f"Running quick evaluation with {len(test_cases)} test cases...")
        elif evaluation_type == "comprehensive":
            update_status("running", f"Running comprehensive evaluation with {len(test_cases)} test cases...")
        elif evaluation_type == "performance":
            test_cases = test_cases[:10]
            update_status("running", f"Running performance benchmark with {len(test_cases)} test cases...")
        elif evaluation_type == "component":
            test_cases = test_cases[:10]
            update_status("running", f"Running component analysis with {len(test_cases)} test cases...")
        
        # Initialize evaluator
        evaluator = RAGEvaluator(chain, retrievers)
        
        # Run evaluation based on type
        if evaluation_type in ["quick", "comprehensive"]:
            include_stress = evaluation_type == "comprehensive"
            results = evaluator.run_comprehensive_evaluation(test_cases, include_stress_test=include_stress)
        elif evaluation_type == "performance":
            results = run_performance_evaluation(chain, test_cases)
        elif evaluation_type == "component":
            results = run_component_evaluation(retrievers, test_cases)
        else:
            update_status("error", f"Unknown evaluation type: {evaluation_type}")
            return
        
        update_status("completed", "Evaluation completed successfully", results)
        
    except Exception as e:
        # Try to report error status; if update_status isn't available due to early failure, write directly
        try:
            update_status("error", f"Evaluation failed: {str(e)}")
        except Exception:
            try:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                status_file = Path(root_dir) / "evaluation_results" / f"{eval_id}_status.json"
                status_data = {
                    "status": "error",
                    "progress": f"Evaluation failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_id": eval_id
                }
                with open(status_file, "w") as f:
                    json.dump(status_data, f)
            except Exception:
                pass

def run_performance_evaluation(chain, test_cases):
    """Run performance-focused evaluation"""
    import time
    import statistics
    
    response_times = []
    successful_queries = 0
    
    for i, test_case in enumerate(test_cases):
        query = test_case.get('query', f'Test query {i+1}')
        
        start_time = time.time()
        try:
            response = chain.invoke(query)
            end_time = time.time() - start_time
            response_times.append(end_time)
            successful_queries += 1
        except Exception as e:
            response_times.append(10.0)  # Penalty time for errors
    
    return {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "performance",
        "test_dataset_size": len(test_cases),
        "successful_queries": successful_queries,
        "performance_metrics": {
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "success_rate": successful_queries / len(test_cases) if test_cases else 0
        }
    }

def run_component_evaluation(retrievers, test_cases):
    """Run component analysis evaluation"""
    import time
    
    component_results = {}
    
    for name, retriever in retrievers.items():
        retrieval_times = []
        doc_counts = []
        successful_retrievals = 0
        
        for test_case in test_cases[:5]:  # Test with 5 queries
            query = test_case.get('query', 'Test query')
            
            start_time = time.time()
            try:
                docs = retriever.get_relevant_documents(query)
                retrieval_time = time.time() - start_time
                
                retrieval_times.append(retrieval_time)
                doc_counts.append(len(docs))
                successful_retrievals += 1
            except Exception as e:
                retrieval_times.append(1.0)  # Penalty time
                doc_counts.append(0)
        
        component_results[name] = {
            "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            "avg_docs_retrieved": sum(doc_counts) / len(doc_counts) if doc_counts else 0,
            "success_rate": successful_retrievals / min(len(test_cases), 5)
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "component_analysis",
        "test_dataset_size": min(len(test_cases), 5),
        "component_results": component_results
    }

# === HUMAN FEEDBACK COLLECTION ENDPOINTS ===

@app.post("/human-feedback")
async def collect_human_feedback(feedback: HumanFeedbackRequest):
    """Collect human feedback for a specific query-response pair"""
    try:
        evaluator = get_or_create_evaluator()
        if not evaluator:
            raise HTTPException(status_code=503, detail="Evaluation system not available")
        
        feedback_data = {
            'rating': feedback.rating,
            'feedback_text': feedback.feedback_text,
            'dimensions': feedback.dimensions,
            'user_id': feedback.user_id,
            'session_id': feedback.session_id
        }
        
        feedback_id = evaluator.collect_human_feedback(
            feedback.query, 
            feedback.response, 
            feedback_data
        )
        
        return {
            "feedback_id": feedback_id,
            "status": "collected",
            "message": "Thank you for your feedback!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional analysis endpoint removed to focus on core operations

# Ground truth curation endpoints removed to keep API surface minimal

# Advanced evaluation convenience endpoint removed (core /run-evaluation retained)

@app.get("/metrics/real-time/{query}")
async def get_real_time_metrics(query: str):
    """Get real-time advanced metrics for a specific query"""
    try:
        evaluator = get_or_create_evaluator()
        if not evaluator or not EVALUATION_AVAILABLE:
            return {"error": "Advanced metrics not available"}
        
        # Get response
        response = chain.invoke(query)
        
        # Get advanced metrics if evaluator available
        if evaluator.advanced_evaluator:
            # Create a simple expected answer for demonstration
            expected = "This would be a reference answer for comparison"
            
            advanced_result = evaluator.advanced_evaluator.evaluate_comprehensive(
                query, expected, response, [], []
            )
            
            # Get benchmark comparison if available
            benchmark_comparison = None
            if evaluator.benchmark_comparator:
                benchmark_comparison = evaluator.benchmark_comparator.compare_to_benchmarks(advanced_result)
            
            # Create metrics response with potential numpy values
            metrics = {
                "query": query,
                "response": response,
                "real_time_metrics": {
                    "overall_score": advanced_result.overall_score,
                    "semantic_similarity": advanced_result.semantic_similarity,
                    "factual_consistency": advanced_result.factual_consistency,
                    "response_appropriateness": advanced_result.response_appropriateness,
                    "benchmark_comparison": benchmark_comparison
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert all numpy values to standard Python types
            return convert_numpy_to_python(metrics)
        else:
            # Create a simple metrics response
            metrics = {
                "query": query,
                "response": response,
                "message": "Advanced metrics not initialized",
                "basic_metrics": {
                    "response_length": len(response.split()),
                    "response_time": "< 1s"
                }
            }
            # Convert any numpy values to standard Python types
            return convert_numpy_to_python(metrics)
    except Exception as e:
        import traceback
        error_details = f"Error in metrics endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_details)
        raise HTTPException(status_code=500, detail=str(e))
