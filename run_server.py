#!/usr/bin/env python3
"""
Run the RAG FastAPI server using uv
"""

import subprocess
import sys
import os

def run_server():
    """Run the FastAPI server using uv"""
    try:
        print("🚀 Starting RAG FastAPI server with uv...")
        print("=" * 50)
        
        # Run uvicorn with uv from root directory
        cmd = [
            "uv", "run", "--directory", "rag_app", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("\n🌐 Server will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running server: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_server()
    sys.exit(0 if success else 1)
