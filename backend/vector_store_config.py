"""
Vector Store Configuration
Defines available DuckDB files organized by category
"""

import os
from typing import Dict, List

# Base path to vector stores
VECTOR_STORE_BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "Vector_Store_Duckdb"
))

# Vector store configuration organized by category
VECTOR_STORE_CONFIG: Dict[str, Dict[str, str]] = {
    "Body Modelling": {
        "PhD Thesis LovatoC": os.path.join(
            VECTOR_STORE_BASE_PATH, 
            "PhD_Thesis_LovatoC", 
            "PhD_Thesis_LovatoC.duckdb"
        ),
        "Human Body Measuring and 3D Modelling": os.path.join(
            VECTOR_STORE_BASE_PATH,
            "Human_Body_Measuring_and_3D_Modelling",
            "Human_Body_Measuring_and_3D_Modelling.duckdb"
        ),
        "Optimization of Product Dimensions": os.path.join(
            VECTOR_STORE_BASE_PATH,
            "OptimizationOfProductDimensions",
            "OptimizationApproachToSizing.duckdb"
        ),
    },
    "Fit & Sizing": {
        "Optimization Approach to Sizing": os.path.join(
            VECTOR_STORE_BASE_PATH,
            "OptimizationApproachToSizing",
            "OptimizationApproachToSizing.duckdb"
        ),
    }
}

# Image folders mapping (if needed for direct image access)
IMAGE_FOLDERS_MAP: Dict[str, str] = {
    "PhD Thesis LovatoC": os.path.join(
        VECTOR_STORE_BASE_PATH,
        "PhD_Thesis_LovatoC",
        "extracted_images"
    ),
    "Human Body Measuring and 3D Modelling": os.path.join(
        VECTOR_STORE_BASE_PATH,
        "Human_Body_Measuring_and_3D_Modelling",
        "extracted_images"
    ),
    "Optimization of Product Dimensions": os.path.join(
        VECTOR_STORE_BASE_PATH,
        "OptimizationOfProductDimensions",
        "extracted_images"
    ),
    "Optimization Approach to Sizing": os.path.join(
        VECTOR_STORE_BASE_PATH,
        "OptimizationApproachToSizing",
        "extracted_image"
    ),
}


def get_vector_store_path(file_name: str) -> str:
    """
    Get the full path to a vector store file by its display name.
    
    Args:
        file_name: Display name of the vector store (e.g., "PhD Thesis LovatoC")
    
    Returns:
        Full path to the DuckDB file, or None if not found
    """
    for category, files in VECTOR_STORE_CONFIG.items():
        if file_name in files:
            return files[file_name]
    return None


def get_image_folder_path(file_name: str) -> str:
    """
    Get the path to the image folder for a given vector store.
    
    Args:
        file_name: Display name of the vector store
    
    Returns:
        Path to the image folder, or None if not found
    """
    return IMAGE_FOLDERS_MAP.get(file_name)


def get_all_vector_stores() -> Dict[str, List[str]]:
    """
    Get all available vector stores organized by category.
    
    Returns:
        Dictionary with categories as keys and list of file names as values
    """
    return {
        category: list(files.keys()) 
        for category, files in VECTOR_STORE_CONFIG.items()
    }


def validate_vector_store_paths() -> Dict[str, bool]:
    """
    Validate that all configured DuckDB files exist.
    
    Returns:
        Dictionary mapping file names to existence status
    """
    result = {}
    for category, files in VECTOR_STORE_CONFIG.items():
        for name, path in files.items():
            result[name] = os.path.exists(path)
    return result
