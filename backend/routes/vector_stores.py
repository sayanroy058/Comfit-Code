import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vector_store_config import get_all_vector_stores, validate_vector_store_paths

load_dotenv()

router = APIRouter()

@router.get("/vector-stores", tags=["vector-stores"])
async def list_vector_stores():
    """List all available vector stores organized by category"""
    try:
        # Get vector stores from configuration
        vector_stores = get_all_vector_stores()
        validation = validate_vector_store_paths()
        
        # Format the response with categories
        formatted_stores = []
        for category, files in vector_stores.items():
            for file_name in files:
                formatted_stores.append({
                    "category": category,
                    "name": file_name,
                    "display_name": file_name,
                    "exists": validation.get(file_name, False)
                })
        
        return {
            "vector_stores": formatted_stores,
            "categories": vector_stores,
            "message": f"Found {len(formatted_stores)} vector stores"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve vector stores: {str(e)}"
        )

@router.get("/vector-stores/legacy", tags=["vector-stores"])
async def list_vector_stores_legacy():
    """List all available vector stores (.duckdb files) in the configured directory"""
    try:
        # Get vector stores path from environment variable
        vector_stores_path = os.environ.get("VECTOR_STORES_PATH", "/app/vector_stores")
        
        # Convert to Path object
        stores_dir = Path(vector_stores_path)
        
        if not stores_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Vector stores directory not found: {vector_stores_path}"
            )
        
        # Find all .duckdb files
        duckdb_files = list(stores_dir.glob("*.duckdb"))
        
        if not duckdb_files:
            return {"vector_stores": [], "message": "No vector stores found"}
        
        # Create friendly names for each vector store
        vector_stores = []
        for file_path in duckdb_files:
            # Remove .duckdb extension and convert to friendly name
            filename = file_path.stem
            
            # Create a more readable display name
            display_name = filename.replace("_", " ").replace("-", " ")
            
            # Handle acronyms and capitalization
            words = display_name.split()
            formatted_words = []
            for word in words:
                # If word is all caps (like CFIR), keep it as is
                if word.isupper():
                    formatted_words.append(word)
                # Otherwise, capitalize first letter only
                else:
                    formatted_words.append(word.capitalize())
            
            display_name = " ".join(formatted_words)
            
            vector_stores.append({
                "id": filename,
                "display_name": display_name,
                "filename": file_path.name,
                "path": str(file_path)
            })
        
        return {
            "vector_stores": vector_stores,
            "directory": str(stores_dir),
            "count": len(vector_stores)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list vector stores: {str(e)}"
        )