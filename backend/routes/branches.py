from fastapi import APIRouter, HTTPException, Depends, Header, Body
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from schemas import Message, BranchCreate
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

@router.post("/messages/conversations/{conversation_id}/branches")
async def create_branch(
    conversation_id: str,
    request: dict = Body(...),
    user_id: str = Depends(get_current_user)
):
    """Create a new branch when editing a message (store messages only in branches JSON)."""
    try:
        # Verify conversation ownership
        conv_result = (
            supabase.table("conversations")
            .select("id")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        edit_at_id = request.get("edit_at_id")
        raw_messages = request.get("messages", [])
        original_messages_from_request = request.get("original_messages", [])
        
        if not edit_at_id:
            raise HTTPException(status_code=400, detail="edit_at_id is required")
        if not raw_messages:
            raise HTTPException(status_code=400, detail="messages are required")

        # Is there already any branch for this edit point?
        existing_branches = (
            supabase.table("branches")
            .select("*")
            .eq("conversation_id", conversation_id)
            .eq("edit_at_id", edit_at_id)
            .execute()
        )

        # If no branches yet, create original branch from frontend data
        if not existing_branches.data:
            original_branch_id = str(uuid.uuid4())
            original_branch_messages = []

            # Use original messages from frontend if provided
            if original_messages_from_request:
                print(f"DEBUG: Using {len(original_messages_from_request)} original messages from frontend")
                for m in original_messages_from_request:
                    sender = (m.get("sender") or "user").lower()
                    sender = "ai" if sender in ("assistant", "bot", "ai") else "user"
                    tt = m.get("thinking_time")
                    try:
                        tt = int(tt) if tt is not None else 0
                    except Exception:
                        tt = 0
                    original_branch_messages.append({
                        "id": m.get("id"),
                        "conversation_id": conversation_id,
                        "sender": sender,
                        "content": m.get("content", ""),
                        "thinking_time": tt,
                        "feedback": m.get("feedback"),
                        "model": m.get("model"),
                        "preset": m.get("preset"),
                        "system_prompt": m.get("system_prompt"),
                        "speculative_decoding": m.get("speculative_decoding", False),
                        "temperature": m.get("temperature"),
                        "top_p": m.get("top_p"),
                        "strategy": m.get("strategy"),
                        "rag_method": m.get("rag_method"),
                        "retrieval_method": m.get("retrieval_method"),
                        "created_at": m.get("created_at"),
                    })
            else:
                # Fallback to database query (existing logic)
                print("DEBUG: No original_messages provided, falling back to DB query")
                # Find the edit message to get its created_at
                edit_msg_res = (
                    supabase.table("messages")
                    .select("id, created_at")
                    .eq("id", edit_at_id)
                    .eq("conversation_id", conversation_id)
                    .limit(1)
                    .execute()
                )
                if not edit_msg_res.data:
                    raise HTTPException(status_code=404, detail="Edit target message not found")

                edit_created_at = edit_msg_res.data[0]["created_at"]

                # pull all messages up to and including that timestamp
                original_msgs_res = (
                    supabase.table("messages")
                    .select("*")
                    .eq("conversation_id", conversation_id)
                    .lte("created_at", edit_created_at)   
                    .order("created_at", desc=False)
                    .execute()
                )

                for row in original_msgs_res.data or []:
                    # normalize minimal fields for branch JSON
                    sender = (row.get("sender") or "user").lower()
                    sender = "ai" if sender in ("assistant", "bot", "ai") else "user"
                    tt = row.get("thinking_time")
                    try:
                        tt = int(tt) if tt is not None else 0
                    except Exception:
                        tt = 0
                    original_branch_messages.append({
                        "id": row["id"],
                        "conversation_id": conversation_id,
                        "sender": sender,
                        "content": row.get("content", ""),
                        "thinking_time": tt,
                        "feedback": row.get("feedback"),
                        "model": row.get("model"),
                        "preset": row.get("preset"),
                        "system_prompt": row.get("system_prompt"),
                        "speculative_decoding": row.get("speculative_decoding", False),
                        "temperature": row.get("temperature"),
                        "top_p": row.get("top_p"),
                        "strategy": row.get("strategy"),
                        "rag_method": row.get("rag_method"),
                        "retrieval_method": row.get("retrieval_method"),
                        "created_at": row.get("created_at"),
                    })

            print(f"DEBUG: Creating original branch with {len(original_branch_messages)} messages")
            supabase.table("branches").insert({
                "id": original_branch_id,
                "conversation_id": conversation_id,
                "edit_at_id": edit_at_id,
                "parent_branch_id": None,
                "messages": original_branch_messages,
                "is_original": True,
                "is_active": False,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

        #  insert the new one as active
        supabase.table("branches") \
            .update({"is_active": False}) \
            .eq("conversation_id", conversation_id) \
            .eq("edit_at_id", edit_at_id) \
            .execute()

        norm_msgs = []
        for m in raw_messages:
            sender = (m.get("sender") or "user").lower()
            sender = "ai" if sender in ("assistant", "bot", "ai") else "user"
            tt = m.get("thinking_time")
            try:
                tt = int(tt) if tt is not None else 0
            except Exception:
                tt = 0
            norm_msgs.append({
                "id": m.get("id"),
                "conversation_id": conversation_id,
                "sender": sender,
                "content": m.get("content", ""),
                "thinking_time": tt,
                "feedback": m.get("feedback"),
                "model": m.get("model"),
                "preset": m.get("preset"),
                "system_prompt": m.get("system_prompt"),
                "speculative_decoding": m.get("speculative_decoding", False),
                "temperature": m.get("temperature"),
                "top_p": m.get("top_p"),
                "strategy": m.get("strategy"),
                "rag_method": m.get("rag_method"),
                "retrieval_method": m.get("retrieval_method"),
                "created_at": m.get("created_at") or datetime.utcnow().isoformat()
            })

        # Insert the NEW branch
        new_branch_id = str(uuid.uuid4())
        print(f"DEBUG: Creating new edited branch with {len(norm_msgs)} messages")
        supabase.table("branches").insert({
            "id": new_branch_id,
            "conversation_id": conversation_id,
            "edit_at_id": edit_at_id,
            "parent_branch_id": None,  # optional: could point to the original branch id
            "messages": norm_msgs,
            "is_original": False,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        return {"branch_id": new_branch_id, "message": "Branch created successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating branch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create branch: {str(e)}")

@router.get("/branches/{branch_id}")
async def get_branch(
    branch_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get a specific branch by ID (messages come from branches JSON)."""
    try:
        # Get branch info
        branch_result = supabase.table("branches").select("*").eq("id", branch_id).execute()
        if not branch_result.data:
            raise HTTPException(status_code=404, detail="Branch not found")
        branch_data = branch_result.data[0]

        # Verify conversation ownership
        conv_result = (
            supabase.table("conversations")
            .select("id")
            .eq("id", branch_data["conversation_id"])
            .eq("user_id", user_id)
            .execute()
        )
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Return messages directly from branch JSON
        return {
            "id": branch_data["id"],
            "conversation_id": branch_data["conversation_id"],
            "edit_at_id": branch_data["edit_at_id"],
            "parent_branch_id": branch_data.get("parent_branch_id"),
            "is_original": branch_data.get("is_original"),
            "is_active": branch_data.get("is_active"),
            "created_at": branch_data.get("created_at"),
            "messages": branch_data.get("messages", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting branch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get branch: {str(e)}")

@router.patch("/branches/{branch_id}")
async def update_branch(
    branch_id: str,
    messages: List[dict] = Body(...),
    user_id: str = Depends(get_current_user)
):
    """Update a branch with new messages (update branches JSON only)."""
    try:
        # Get branch info
        branch_result = supabase.table("branches").select("*").eq("id", branch_id).execute()
        if not branch_result.data:
            raise HTTPException(status_code=404, detail="Branch not found")
        branch_data = branch_result.data[0]

        # Verify conversation ownership
        conv_result = (
            supabase.table("conversations")
            .select("id")
            .eq("id", branch_data["conversation_id"])
            .eq("user_id", user_id)
            .execute()
        )
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Normalize messages for JSON storage
        conv_id = branch_data["conversation_id"]
        norm_msgs = []
        for m in messages:
            sender = (m.get("sender") or "user").lower()
            sender = "ai" if sender in ("assistant", "bot", "ai") else "user"
            thinking_time = m.get("thinking_time")
            try:
                thinking_time = int(thinking_time) if thinking_time is not None else 0
            except Exception:
                thinking_time = 0

            norm_msgs.append(
                {
                    **m,
                    "conversation_id": conv_id,
                    "sender": sender,
                    "thinking_time": thinking_time,
                }
            )

        # Update branch JSON
        result = (
            supabase.table("branches")
            .update({"messages": norm_msgs})
            .eq("id", branch_id)
            .execute()
        )
        if result.data:
            return {"message": "Branch updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update branch")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating branch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update branch: {str(e)}")

@router.post("/branches/{branch_id}/activate")
async def activate_branch(
    branch_id: str,
    user_id: str = Depends(get_current_user)
):
    """Activate a branch (deactivate others)"""
    try:
        # Get branch info
        branch_result = supabase.table("branches")\
            .select("*")\
            .eq("id", branch_id)\
            .execute()
        
        if not branch_result.data:
            raise HTTPException(status_code=404, detail="Branch not found")
        
        branch_data = branch_result.data[0]
        
        # Verify conversation ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", branch_data["conversation_id"])\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Deactivate all branches in this conversation
        supabase.table("branches")\
            .update({"is_active": False})\
            .eq("conversation_id", branch_data["conversation_id"])\
            .execute()
        
        # Activate the specified branch
        result = supabase.table("branches")\
            .update({"is_active": True})\
            .eq("id", branch_id)\
            .execute()
        
        if result.data:
            return {"message": "Branch activated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to activate branch")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error activating branch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate branch: {str(e)}")

@router.delete("/branches/{branch_id}")
async def delete_branch(
    branch_id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete a branch (no normalized message rows to delete)."""
    try:
        # Get branch info
        branch_result = supabase.table("branches").select("*").eq("id", branch_id).execute()
        if not branch_result.data:
            raise HTTPException(status_code=404, detail="Branch not found")
        branch_data = branch_result.data[0]

        # Verify conversation ownership
        conv_result = (
            supabase.table("conversations")
            .select("id")
            .eq("id", branch_data["conversation_id"])
            .eq("user_id", user_id)
            .execute()
        )
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Delete branch only (messages live inside branches JSON)
        supabase.table("branches").delete().eq("id", branch_id).execute()
        return {"message": "Branch deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting branch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete branch: {str(e)}")
