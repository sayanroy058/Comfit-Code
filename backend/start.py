#!/usr/bin/env python3
"""
Startup script for Comfit Copilot Backend
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting Comfit Copilot Backend on port {port}")
    print("📖 API documentation available at: http://localhost:{port}/docs")
    print("🔍 Health check available at: http://localhost:{port}/health")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
