import os
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/models")
async def get_models():
    """Get list of available AI models from Ollama"""
    try:
        # Get Ollama base URL from environment variable
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Fetch models from Ollama
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = []
            
            if "models" in data:
                for model in data["models"]:
                    model_name = model.get("name", "")
                    if model_name and model_name != "nomic-embed-text:latest":
                        models.append(model_name)
            
            # Sort models alphabetically
            models.sort()
            
            if not models:
                # Fallback to default models if none found
                models = "no models available"
            
            return {"models": models}
            
        else:
            return {"no models available"}
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return {"no models available"}
        
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"no models available"}
