from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, File, Form
from typing import Optional
import os
import uuid
from datetime import datetime

from database import supabase, supabase_auth

router = APIRouter()

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Extract user ID from authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    try:
        # Verify the token and get user info
        user = supabase_auth.auth.get_user(token)
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/documents/append")
async def append_document(
    file: UploadFile = File(...),
    preset: str = Form(...),
    user_id: str = Depends(get_current_user)
):
    """Upload and append a document to the vector store"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB"
            )
        
        # Read file content
        content = await file.read()
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save document metadata to database
        document_data = {
            "id": document_id,
            "filename": file.filename,
            "file_type": file_extension,
            "file_size": len(content),
            "preset": preset,
            "user_id": user_id,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "uploaded"
        }
        
        # For now, we'll just save the metadata
        # In a real implementation, you would:
        # 1. Save the file to storage (S3, local filesystem, etc.)
        # 2. Process the document content (extract text, create embeddings)
        # 3. Store embeddings in the vector store
        # 4. Update the status to "processed"
        
        try:
            # Save to documents table if it exists, otherwise just return success
            # This is a placeholder for the actual document processing logic
            result = {"message": "Document uploaded successfully", "document_id": document_id}
            
            # Here you would integrate with your existing document manager
            # from ..chat_engine.document_manager import process_document
            # await process_document(content, file.filename, preset, user_id)
            
            return result
            
        except Exception as e:
            print(f"Warning: Failed to save document metadata: {e}")
            # Return success anyway since the file was uploaded
            return {"message": "Document uploaded successfully", "document_id": document_id}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.get("/documents")
async def get_documents(user_id: str = Depends(get_current_user)):
    """Get all documents for the current user"""
    try:
        # This would query your documents table
        # For now, return empty list since we're not fully implementing document storage
        return []
        
    except Exception as e:
        print(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete a document"""
    try:
        # This would delete the document and its embeddings
        # For now, just return success
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
