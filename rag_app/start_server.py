#!/usr/bin/env python3
"""
Startup wrapper for RAG backend with better error handling and logging
"""

import os
import sys
import time
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server with proper error handling"""
    try:
        logger.info("🚀 Starting RAG Backend...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        
        # Check required environment variables
        required_vars = ["GEMINI_API_KEY", "COHERE_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
        else:
            logger.info("✅ All required environment variables are set")
        
        # Check PORT
        port = os.getenv("PORT", "8000")
        logger.info(f"🌐 Server will bind to port: {port}")
        
        # Import and start the application
        logger.info("📦 Importing FastAPI application...")
        try:
            from main import app
            logger.info("✅ FastAPI app imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import FastAPI app: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        # Start uvicorn
        logger.info("🔥 Starting Uvicorn server...")
        import uvicorn
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(port),
            timeout_keep_alive=120,
            access_log=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("⛔ Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"💥 Server startup failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())