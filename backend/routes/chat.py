
from fastapi import APIRouter, HTTPException, Depends, Header, Body
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import time
import re
import os
import mimetypes
import json
import asyncio
from pathlib import Path
from fastapi.concurrency import run_in_threadpool
from schemas import Message, MessageUpdate, FeedbackRequest, TitleRequest, History, BranchItem, ChatRequest, ChatResponse
from chat_engine import chat_answer
from database import supabase, supabase_auth
from dateutil import parser

router = APIRouter()

def normalize_sender(s: str) -> str:
    s = (s or "").lower()
    if s in ("assistant", "bot", "ai"):
        return "ai"
    return "user" if s == "user" else s

def role_for_llm(sender: str) -> str:
    return "assistant" if normalize_sender(sender) == "ai" else "user"

def ensure_uuid(s: str) -> str:
    try:
        uuid.UUID(str(s))
        return str(s)
    except Exception:
        return str(uuid.uuid4())

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        user = supabase_auth.auth.get_user(token)
        return user.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """
    Standard chat endpoint that returns complete response.
    For streaming progress updates, use /chat/stream endpoint.
    """
    try:
        start_time = time.time()

        is_branch_mode = getattr(request, "branch_mode", False)
        
        if not is_branch_mode:
            last_user = next((m for m in reversed(request.messages) if normalize_sender(m.sender) == "user"), None)
            if last_user:
                user_row = {
                    "id": ensure_uuid(last_user.id),
                    "conversation_id": request.conversation_id,
                    "sender": "user",
                    "content": last_user.content,
                    "thinking_time": int(getattr(last_user, "thinking_time", 0) or 0),
                    "feedback": getattr(last_user, "feedback", None),
                    "model": getattr(last_user, "model", request.model),
                    "preset": getattr(last_user, "preset", request.preset),
                    "system_prompt": request.system_prompt,
                    "speculative_decoding": request.speculative_decoding,
                    "temperature": getattr(last_user, "temperature", request.temperature),
                    "top_p": getattr(last_user, "top_p", request.top_p),
                    "strategy": getattr(last_user, "strategy", request.strategy),
                    "rag_method": getattr(last_user, "rag_method", request.rag_method),
                    "retrieval_method": getattr(last_user, "retrieval_method", request.retrieval_method),
                    "selected_vector_store": getattr(last_user, "selected_vector_store", request.selected_vector_store),
                    "user_id": user_id,
                }
                try:
                    supabase.table("messages").insert(user_row).execute()
                except Exception as e:
                    print(f"DEBUG: user insert (ignore duplicates) error: {e}")

        formatted_messages = []
        if request.system_prompt:
            formatted_messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            formatted_messages.append({
                "role": role_for_llm(msg.sender),
                "content": msg.content
            })
            
        try:
            ai_content, _duration, image_info, thinking_content = await chat_answer(
                messages=formatted_messages,
                conversation_id=request.conversation_id,
                model=request.model,
                preset=request.preset,
                temperature=request.temperature,
                user_id=user_id,
                rag_method=request.rag_method,
                retrieval_method=request.retrieval_method,
                selected_vector_store=request.selected_vector_store,
            )
            if isinstance(ai_content, str) and ai_content.startswith("Error:"):
                raise HTTPException(status_code=400, detail=ai_content)
        except Exception as e:
            print(f"Chat engine error: {e}")
            raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

        actual_duration = int((time.time() - start_time) * 1000)

        ai_message_id = str(uuid.uuid4())
        created_at_str = datetime.utcnow().isoformat()
        
        # Prepare image data for storage if present
        import json
        image_data_json = None
        if image_info:
            image_data_json = json.dumps(image_info)
        
        if not is_branch_mode:
            ai_row = {
                "id": ai_message_id,
                "conversation_id": request.conversation_id,
                "sender": "ai",
                "content": ai_content,
                "thinking_time": actual_duration,
                "feedback": None,
                "model": request.model,
                "preset": request.preset,
                "system_prompt": request.system_prompt,
                "speculative_decoding": request.speculative_decoding,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "strategy": request.strategy,
                "rag_method": request.rag_method,
                "retrieval_method": request.retrieval_method,
                "user_id": user_id,
                "image_data": image_data_json,  # Store image info as JSON
                "thinking_content": thinking_content,  # Store thinking content
            }
            inserted = supabase.table("messages").insert(ai_row).execute()
            created_at_str = inserted.data[0]["created_at"] if inserted and inserted.data else created_at_str

        ai_message = Message(
            id=ai_message_id,
            conversation_id=request.conversation_id,
            sender="ai",
            content=ai_content,
            thinking_time=actual_duration,
            feedback=None,
            model=request.model,
            preset=request.preset,
            system_prompt=request.system_prompt,
            speculative_decoding=request.speculative_decoding,
            temperature=request.temperature,
            top_p=request.top_p,
            strategy=request.strategy,
            rag_method=request.rag_method,
            retrieval_method=request.retrieval_method,
            created_at=parser.isoparse(created_at_str.replace("Z", "+00:00")) if "Z" in created_at_str else parser.isoparse(created_at_str),
            thinking_content=thinking_content,  # Add thinking content
        )

        return ChatResponse(
            result=ai_content,
            duration=actual_duration,
            ai_message=ai_message,
            image_info=image_info,
            thinking_content=thinking_content  # Add thinking content to response
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """
    Streaming chat endpoint that sends Server-Sent Events with progress updates.
    """
    async def generate_events():
        progress_queue = asyncio.Queue()
        
        async def progress_callback(step: str, details: str = ""):
            """Callback to send progress updates"""
            await progress_queue.put({
                "type": "progress",
                "step": step,
                "details": details
            })
        
        async def process_chat():
            """Process chat in background and send final result"""
            try:
                start_time = time.time()
                is_branch_mode = getattr(request, "branch_mode", False)
                
                if not is_branch_mode:
                    last_user = next((m for m in reversed(request.messages) if normalize_sender(m.sender) == "user"), None)
                    if last_user:
                        user_row = {
                            "id": ensure_uuid(last_user.id),
                            "conversation_id": request.conversation_id,
                            "sender": "user",
                            "content": last_user.content,
                            "thinking_time": int(getattr(last_user, "thinking_time", 0) or 0),
                            "feedback": getattr(last_user, "feedback", None),
                            "model": getattr(last_user, "model", request.model),
                            "preset": getattr(last_user, "preset", request.preset),
                            "system_prompt": request.system_prompt,
                            "speculative_decoding": request.speculative_decoding,
                            "temperature": getattr(last_user, "temperature", request.temperature),
                            "top_p": getattr(last_user, "top_p", request.top_p),
                            "strategy": getattr(last_user, "strategy", request.strategy),
                            "rag_method": getattr(last_user, "rag_method", request.rag_method),
                            "retrieval_method": getattr(last_user, "retrieval_method", request.retrieval_method),
                            "selected_vector_store": getattr(last_user, "selected_vector_store", request.selected_vector_store),
                            "user_id": user_id,
                        }
                        try:
                            supabase.table("messages").insert(user_row).execute()
                        except Exception as e:
                            print(f"DEBUG: user insert (ignore duplicates) error: {e}")

                formatted_messages = []
                if request.system_prompt:
                    formatted_messages.append({"role": "system", "content": request.system_prompt})
                for msg in request.messages:
                    formatted_messages.append({
                        "role": role_for_llm(msg.sender),
                        "content": msg.content
                    })
                
                # Import the chat_answer_with_progress function
                from chat_engine import chat_answer_with_progress
                
                ai_content, _duration, image_info, thinking_content = await chat_answer_with_progress(
                    messages=formatted_messages,
                    conversation_id=request.conversation_id,
                    model=request.model,
                    preset=request.preset,
                    temperature=request.temperature,
                    user_id=user_id,
                    rag_method=request.rag_method,
                    retrieval_method=request.retrieval_method,
                    selected_vector_store=request.selected_vector_store,
                    progress_callback=progress_callback
                )
                
                if isinstance(ai_content, str) and ai_content.startswith("Error:"):
                    await progress_queue.put({"type": "error", "message": ai_content})
                    return
                
                actual_duration = int((time.time() - start_time) * 1000)
                ai_message_id = str(uuid.uuid4())
                created_at_str = datetime.utcnow().isoformat()
                
                image_data_json = None
                if image_info:
                    image_data_json = json.dumps(image_info)
                
                if not is_branch_mode:
                    ai_row = {
                        "id": ai_message_id,
                        "conversation_id": request.conversation_id,
                        "sender": "ai",
                        "content": ai_content,
                        "thinking_time": actual_duration,
                        "feedback": None,
                        "model": request.model,
                        "preset": request.preset,
                        "system_prompt": request.system_prompt,
                        "speculative_decoding": request.speculative_decoding,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "strategy": request.strategy,
                        "rag_method": request.rag_method,
                        "retrieval_method": request.retrieval_method,
                        "user_id": user_id,
                        "image_data": image_data_json,
                        "thinking_content": thinking_content,  # Add thinking content
                    }
                    inserted = supabase.table("messages").insert(ai_row).execute()
                    created_at_str = inserted.data[0]["created_at"] if inserted and inserted.data else created_at_str

                ai_message = Message(
                    id=ai_message_id,
                    conversation_id=request.conversation_id,
                    sender="ai",
                    content=ai_content,
                    thinking_time=actual_duration,
                    feedback=None,
                    model=request.model,
                    preset=request.preset,
                    system_prompt=request.system_prompt,
                    speculative_decoding=request.speculative_decoding,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    strategy=request.strategy,
                    rag_method=request.rag_method,
                    retrieval_method=request.retrieval_method,
                    created_at=parser.isoparse(created_at_str.replace("Z", "+00:00")) if "Z" in created_at_str else parser.isoparse(created_at_str),
                    thinking_content=thinking_content,  # Add thinking content
                )
                
                # Send final result
                await progress_queue.put({
                    "type": "complete",
                    "content": ai_content,
                    "duration": actual_duration,
                    "ai_message": {
                        "id": ai_message.id,
                        "conversation_id": ai_message.conversation_id,
                        "sender": ai_message.sender,
                        "content": ai_message.content,
                        "thinking_time": ai_message.thinking_time,
                        "created_at": ai_message.created_at.isoformat() if ai_message.created_at else None,
                        "thinking_content": thinking_content,  # Add thinking content
                    },
                    "image_info": image_info,
                    "thinking_content": thinking_content  # Add thinking content
                })
                
            except Exception as e:
                print(f"Error in chat processing: {e}")
                await progress_queue.put({"type": "error", "message": str(e)})
            finally:
                await progress_queue.put(None)  # Signal completion
        
        # Start processing in background
        task = asyncio.create_task(process_chat())
        
        # Stream events
        try:
            while True:
                event = await progress_queue.get()
                if event is None:
                    break
                
                # Format as Server-Sent Event
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            print(f"Error streaming events: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/messages/{message_id}/feedback")
async def update_message_feedback(
    message_id: str,
    request: FeedbackRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        msg_result = supabase.table("messages")\
            .select("id, conversation_id")\
            .eq("id", message_id)\
            .execute()
        
        if not msg_result.data:
            raise HTTPException(status_code=404, detail="Message not found")
        
        conv_id = msg_result.data[0]["conversation_id"]
        
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conv_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        result = supabase.table("messages")\
            .update({"feedback": request.rating})\
            .eq("id", message_id)\
            .execute()
        
        if result.data:
            return {"message": "Feedback updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update feedback")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {str(e)}")

@router.post("/messages/{message_id}/regenerate")
async def regenerate_message(
    message_id: str,
    payload: ChatRequest = Body(...), 
    user_id: Optional[str] = Depends(get_current_user)
):
    try:
        print(f"DEBUG: Looking for message with ID: {message_id}")
        
        msg_result = supabase.table("messages")\
            .select("id, conversation_id, content, created_at, sender")\
            .eq("id", message_id)\
            .in_("sender", ["ai", "assistant"])\
            .execute()
        
        print(f"DEBUG: Messages table query result: {msg_result.data}")
        
        if not msg_result.data:
            print(f"DEBUG: Message not found in messages table, checking branches table")
            
            request_conversation_id = payload.conversation_id
            print(f"DEBUG: Looking for message in branches for conversation: {request_conversation_id}")
            
            branches_result = supabase.table("branches")\
                .select("id, conversation_id, messages, edit_at_id")\
                .eq("conversation_id", request_conversation_id)\
                .execute()
            
            print(f"DEBUG: Found {len(branches_result.data) if branches_result.data else 0} branches for conversation {request_conversation_id}")
            
            found_message = None
            found_conversation_id = None
            found_created_at = None
            
            for branch in branches_result.data:
                branch_messages = branch.get("messages", [])
                print(f"DEBUG: Branch {branch['id']} has {len(branch_messages)} messages")
                for msg in branch_messages:
                    if msg.get("id") == message_id and msg.get("sender") in ["ai", "assistant"]:
                        found_message = msg
                        found_conversation_id = branch["conversation_id"]
                        found_created_at = msg.get("created_at")
                        print(f"DEBUG: Found message in branch {branch['id']}")
                        break
                if found_message:
                    break
            
            if found_message:
                conv_id = found_conversation_id
                original_created_at = found_created_at
                print(f"DEBUG: Found message in branch, conversation: {conv_id}, created at: {original_created_at}")
            else:
                raise HTTPException(status_code=404, detail="AI message not found")
        else:
            conv_id = msg_result.data[0]["conversation_id"]
            original_created_at = msg_result.data[0]["created_at"]
            print(f"DEBUG: Found message in messages table, conversation: {conv_id}, created at: {original_created_at}")
        
        print(f"DEBUG: Verifying ownership of conversation: {conv_id}")
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conv_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        print(f"DEBUG: Conversation ownership verified")
        
        history_result = supabase.table("messages")\
            .select("*")\
            .eq("conversation_id", conv_id)\
            .lt("created_at", original_created_at)\
            .order("created_at", desc=False)\
            .execute()
        
        formatted_messages = []
        
        last_user_message = None
        for msg in reversed(history_result.data):
            if msg["sender"] == "user":
                last_user_message = msg
                break
        
        if last_user_message:
            formatted_messages.append({
                "role": "user",
                "content": last_user_message["content"]
            })
        
        if payload.system_prompt:
            formatted_messages.insert(0, {
                "role": "system",
                "content": payload.system_prompt
            })
        
        print(f"DEBUG: Sending to chat engine: {formatted_messages}")
        
        start_time = time.time()
        try:
            ai_content, duration, _ = await chat_answer(
                messages=formatted_messages,
                conversation_id=conv_id,
                model=payload.model,
                preset=payload.preset,
                temperature=payload.temperature,
                user_id=user_id,
                rag_method=payload.rag_method,
                retrieval_method=payload.retrieval_method
            )
            
            if isinstance(ai_content, str) and ai_content.startswith("Error:"):
                raise HTTPException(status_code=400, detail=ai_content)
                
        except Exception as e:
            print(f"Chat engine error during regeneration: {e}")
            raise HTTPException(status_code=500, detail=f"AI model error during regeneration: {str(e)}")
        
        actual_duration = int((time.time() - start_time) * 1000)
        
        response_data = {
            "content": ai_content,
            "duration": actual_duration,
            "model": payload.model,
            "preset": payload.preset,
            "temperature": payload.temperature,
            "rag_method": payload.rag_method,
            "retrieval_method": payload.retrieval_method,
            "thinking_time": actual_duration
        }
        
        print(f"DEBUG: Returning response data: {response_data}")
        return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error regenerating message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate message: {str(e)}")

@router.post("/title")
async def generate_title(
    req: TitleRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    prompt = (
          f'Generate a concise chat title using EXACTLY 2-4 words.\n'
          f'User asked: "{req.user_message}"\n'
          f'AI: "{req.ai_response}"'
    )

    MODEL_NAME = "qwen3:0.6b"
    PRESET = "CFIR"
    TEMP = 0.7

    title_html, _, _ = await chat_answer(
        messages=[{"role": "user", "content": prompt}],
        conversation_id=req.conversation_id,
        model=MODEL_NAME,
        preset=PRESET,
        temperature=TEMP,
        user_id=user_id,
        rag_method="no-workflow",
        retrieval_method="local context only",
        skip_rac=True
    )

    clean = re.sub(
        r'<think>[\s\S]*?<\/think>'
        r'|<[^>]*>'
        r'|["\']'
        r'|The chat title could be:\s*'
        r'|A Concise Chat Title Using Exactly 2-4 Words\.\s*',
        "",
        title_html,
        flags=re.IGNORECASE,
    ).strip() or "New chat"

    print(f"DEBUG: generate_title: clean={clean}")
    return {"title": clean}


@router.get("/images/{image_name:path}")
async def serve_image(image_name: str, authorization: Optional[str] = Header(None)):
    """
    Serve images from the extracted_images directory or vector_store directory.
    For external URLs (http/https), redirects to the original source.
    This endpoint accepts optional authentication for backward compatibility.
    """
    try:
        # URL decode the image name in case it has encoded characters
        from urllib.parse import unquote
        image_name = unquote(image_name)
        
        # Remove any trailing or leading quotes from the filename
        if image_name.startswith('"'):
            image_name = image_name[1:]
        if image_name.endswith('"'):
            image_name = image_name[:-1]
        
        print(f"DEBUG: After cleaning: {image_name}")
        
        # Check if this is an external URL (Google Images, etc.)
        if image_name.startswith(('http://', 'https://')):
            print(f"DEBUG: Redirecting to external image URL: {image_name}")
            return RedirectResponse(url=image_name)
        
        # Optional authentication - log but don't block
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            try:
                user = supabase_auth.auth.get_user(token)
                print(f"DEBUG: Authenticated user: {user.user.id}")
            except Exception as e:
                print(f"DEBUG: Authentication failed but continuing: {e}")
        else:
            print(f"DEBUG: No authentication provided for image request: {image_name}")
        
        print(f"DEBUG: Looking for local image: {image_name}")
        
        # Get the correct project root (comfit directory, not backend)
        # This file is in: .../comfit/backend/routes/chat.py
        routes_dir = os.path.dirname(os.path.abspath(__file__))  # .../backend/routes
        backend_dir = os.path.dirname(routes_dir)  # .../backend
        project_root = os.path.dirname(backend_dir)  # .../comfit (CORRECT!)
        
        print(f"DEBUG: Project root: {project_root}")
        
        # Define possible image directories (relative to project root)
        possible_dirs = [
            os.path.join(project_root, "extracted_images_for_upload"),
            os.path.join(project_root, "vector_store"),
            # Vector_Store_Duckdb subdirectories for extracted images
            os.path.join(project_root, "Vector_Store_Duckdb", "PhD_Thesis_LovatoC", "extracted_images"),
            os.path.join(project_root, "Vector_Store_Duckdb", "Human_Body_Measuring_and_3D_Modelling", "extracted_images"),
            os.path.join(project_root, "Vector_Store_Duckdb", "OptimizationApproachToSizing", "extracted_image"),
            os.path.join(project_root, "Vector_Store_Duckdb", "OptimizationOfProductDimensions", "extracted_images"),
            os.path.join(backend_dir, "extracted_images_for_upload"),
            os.path.join(backend_dir, "vector_store"),
            # Also check in parent directory
            os.path.join(os.path.dirname(project_root), "extracted_images"),
            os.path.join(os.path.dirname(project_root), "extracted_images_for_upload"),
            # Legacy paths for backward compatibility
            "D:\\Sahithi\\Sahithi3090\\comfit main\\extracted_images",
            "D:\\Sahithi\\9_3_2025_ComFit\\ComFit\\extracted_images_for_upload",
        ]
        
        # Try to find the image in any of the possible directories
        image_path = None
        for directory in possible_dirs:
            if os.path.exists(directory):
                potential_path = os.path.join(directory, image_name)
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                    image_path = potential_path
                    print(f"DEBUG: Found image at: {image_path}")
                    break
        
        if not image_path:
            print(f"DEBUG: Image not found in any of these directories:")
            for directory in possible_dirs:
                print(f"  - {directory} (exists: {os.path.exists(directory)})")
            raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
        
        # Determine the media type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"  # Default to JPEG if unknown
        
        return FileResponse(image_path, media_type=mime_type)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve image: {str(e)}")
