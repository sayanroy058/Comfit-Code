
import os
from .client import ChatEngine


# Define the embed model path
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "./models/embedding-model")

# Create a global chat engine instance
chat_engine = ChatEngine()

async def chat_answer(
    messages: list,
    conversation_id: str,
    model: str,
    preset: str,
    temperature: float,
    user_id: str = None,
    # CORRECTED: Set a valid default RAG strategy to prevent errors.
    rag_method: str = "rac_enhanced_hybrid_rag",
    retrieval_method: str = "local context only",
    selected_vector_store: str = None,
    skip_rac: bool = False
) -> tuple[str, int, dict, str]:
    """
    Generate a chat response using the chat engine.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        conversation_id: ID of the conversation
        model: Model name to use
        preset: Preset configuration
        temperature: Temperature setting
        user_id: Optional user ID

    Returns:
        Tuple of (response_text, duration_ms, image_info, thinking_content)
    """
    print(f"DEBUG: chat_answer called with:")
    print(f"  - messages: {messages}")
    print(f"  - conversation_id: {conversation_id}")
    print(f"  - model: {model}")
    print(f"  - model type: {type(model)}")
    print(f"  - preset: {preset}")
    print(f"  - temperature: {temperature}")
    print(f"  - user_id: {user_id}")
    print(f"  - rag_method: {rag_method}")
    print(f"  - retrieval_method: {retrieval_method}")
    print(f"  - selected_vector_store: {selected_vector_store}")

    # CORRECTED: Reformat ALL messages, including system messages.
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "sender": msg["role"],
            "content": msg["content"]
        })

    print(f"DEBUG: Formatted messages: {formatted_messages}")

    try:
        # Pass all parameters to the generate_response method.
        # The method now correctly returns the tuple (response, duration)
        result = await chat_engine.generate_response(
            messages=formatted_messages,
            conversation_id=conversation_id,
            model=model,
            preset=preset,
            temperature=temperature,
            user_id=user_id,
            rag_method=rag_method,
            retrieval_method=retrieval_method,
            selected_vector_store=selected_vector_store,
            skip_rac=skip_rac
        )

        print(f"DEBUG: Chat engine result: {result}")
        return result
    except Exception as e:
        print(f"DEBUG: Error in chat_answer: {e}")
        raise


async def chat_answer_with_progress(
    messages: list,
    conversation_id: str,
    model: str,
    preset: str,
    temperature: float,
    user_id: str = None,
    rag_method: str = "rac_enhanced_hybrid_rag",
    retrieval_method: str = "local context only",
    selected_vector_store: str = None,
    skip_rac: bool = False,
    progress_callback = None
) -> tuple[str, int, dict, str]:
    """
    Generate a chat response using the chat engine with progress updates.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        conversation_id: ID of the conversation
        model: Model name to use
        preset: Preset configuration
        temperature: Temperature setting
        user_id: Optional user ID
        progress_callback: Async callback function for progress updates

    Returns:
        Tuple of (response_text, duration_ms, image_info, thinking_content)
    """
    print(f"DEBUG: chat_answer_with_progress called")

    # Reformat messages
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "sender": msg["role"],
            "content": msg["content"]
        })

    try:
        # Pass all parameters including progress callback
        result = await chat_engine.generate_response_with_progress(
            messages=formatted_messages,
            conversation_id=conversation_id,
            model=model,
            preset=preset,
            temperature=temperature,
            user_id=user_id,
            rag_method=rag_method,
            retrieval_method=retrieval_method,
            selected_vector_store=selected_vector_store,
            skip_rac=skip_rac,
            progress_callback=progress_callback
        )

        return result
    except Exception as e:
        print(f"DEBUG: Error in chat_answer_with_progress: {e}")
        raise

# Export the main function and constant
__all__ = ["chat_answer", "chat_answer_with_progress", "EMBED_MODEL_PATH", "ChatEngine"]
