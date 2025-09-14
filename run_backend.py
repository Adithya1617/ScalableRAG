#!/usr/bin/env python3
"""
Simple Backend Runner using uv
"""

import subprocess
import sys

def main():
    """Run the FastAPI backend using uv"""
    print("🚀 Starting RAG Backend Server...")
    
    try:
        # Run uvicorn with uv
        cmd = [
            "uv", "run", "--directory", "rag_app", 
            "uvicorn", "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        print("🌐 Backend will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Backend stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
