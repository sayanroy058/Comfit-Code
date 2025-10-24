from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, List
from datetime import datetime
import uuid

from schemas import ConversationCreate, Conversation, ConversationResponse
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

@router.post("/conversations")
async def create_conversation(
    request: ConversationCreate, 
    user_id: str = Depends(get_current_user)
):
    """Create a new conversation"""
    try:
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            "id": conversation_id,
            "user_id": user_id,
            "title": request.title or "New Chat",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("conversations").insert(conversation_data).execute()
        
        if result.data:
            return ConversationResponse(**result.data[0])
        else:
            raise HTTPException(status_code=500, detail="Failed to create conversation")
            
    except Exception as e:
        print(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@router.get("/conversations")
async def get_conversations(user_id: str = Depends(get_current_user)):
    """Get all conversations for the current user"""
    try:
        result = supabase.table("conversations")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("updated_at", desc=True)\
            .execute()
        
        if result.data:
            conversations = []
            for conv in result.data:
                conversations.append(ConversationResponse(**conv))
            return conversations
        else:
            return []
            
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")

@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str, 
    user_id: str = Depends(get_current_user)
):
    """Get a specific conversation by ID"""
    try:
        result = supabase.table("conversations")\
            .select("*")\
            .eq("id", conversation_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if result.data:
            return ConversationResponse(**result.data[0])
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation: {str(e)}")

@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: ConversationCreate,
    user_id: str = Depends(get_current_user)
):
    """Update a conversation (e.g., title)"""
    try:
        # Verify ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conversation_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation
        update_data = {
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if request.title is not None:
            update_data["title"] = request.title
        
        result = supabase.table("conversations")\
            .update(update_data)\
            .eq("id", conversation_id)\
            .execute()
        
        if result.data:
            return ConversationResponse(**result.data[0])
        else:
            raise HTTPException(status_code=500, detail="Failed to update conversation")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    try:
        # Verify ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conversation_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete conversation (messages and branches will be deleted via CASCADE)
        result = supabase.table("conversations")\
            .delete()\
            .eq("id", conversation_id)\
            .execute()
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")
