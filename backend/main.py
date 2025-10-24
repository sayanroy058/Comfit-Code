from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import os

# Import routes
from routes import chat, conversations, messages, branches, documents, models, vector_stores

# Import database
from database import create_tables

load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Comfit Copilot API",
    description="Backend API for Comfit Copilot chat application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(messages.router, prefix="/api", tags=["messages"])
app.include_router(branches.router, prefix="/api", tags=["branches"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(vector_stores.router, prefix="/api", tags=["vector-stores"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize database tables on startup
    """
    await create_tables()
    print("Database tables created/verified")

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Comfit Copilot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    print("DEBUG: Health check endpoint called")
    return {"status": "healthy", "message": "Comfit Copilot API is running"}

@app.get("/api/health")
async def api_health_check():
    """
    API health check endpoint
    """
    print("DEBUG: API health check endpoint called")
    return {"status": "healthy", "message": "Comfit Copilot API is running", "endpoint": "/api/health"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    ) 