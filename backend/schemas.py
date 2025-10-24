from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict

class ConversationBase(BaseModel):
    title: Optional[str] = None

class ConversationCreate(ConversationBase):
    user_id: Optional[str] = None

class Conversation(ConversationBase):
    id: str
    user_id: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# ----- messages -----
class MessageBase(BaseModel):
    conversation_id: str
    sender: str
    content: str
    # --- NEW IMAGE FIELD ADDED TO BASE MODEL ---
    image_url: Optional[str] = Field(None, alias="imageUrl")
    # --- NEW THINKING CONTENT FIELD ---
    thinking_content: Optional[str] = Field(None, alias="thinkingContent")

class MessageCreate(MessageBase):
    id: str
    thinking_time: int = Field(..., alias="thinkingTime")  # Required field, not optional
    feedback: Optional[int] = None
    model: Optional[str] = None
    preset: Optional[str] = None # vector store
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    rag_method: Optional[str] = None
    retrieval_method: Optional[str] = None
    strategy: Optional[str] = None

    model_config = ConfigDict(validate_by_name=True, populate_by_name=True)

class Message(MessageBase):
    id: str
    thinking_time: int = Field(..., alias="thinkingTime")  # Required field, not optional
    feedback: Optional[int] = None

    model: Optional[str] = None
    preset: Optional[str] = None
    system_prompt: Optional[str] = None
    speculative_decoding: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    strategy: Optional[str] = None
    rag_method: Optional[str] = None
    retrieval_method: Optional[str] = None
    created_at: datetime
    image_data: Optional[str] = None  # JSON string storing image information

    model_config = ConfigDict(validate_by_name=True, from_attributes=True, populate_by_name=True)

class MessageUpdate(BaseModel):
    content: Optional[str] = None
    thinking_time: Optional[int] = Field(None, alias="thinkingTime")  # Keep optional for updates
    feedback: Optional[int] = None
    model: Optional[str] = None
    preset: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    rag_method: Optional[str] = None
    retrieval_method: Optional[str] = None
    # --- NEW IMAGE FIELD ADDED TO UPDATE MODEL ---
    image_url: Optional[str] = Field(None, alias="imageUrl")
    # --- NEW THINKING CONTENT FIELD ---
    thinking_content: Optional[str] = Field(None, alias="thinkingContent")

    model_config = ConfigDict(validate_by_name=True)

# ---- chat ----
class ChatRequest(BaseModel):
    conversation_id: str
    messages: List[MessageCreate]
    model: str
    preset: str
    rag_method: str
    retrieval_method: str
    selected_vector_store: Optional[str] = None  # Selected DuckDB file name
    temperature: float
    top_p: float
    speculative_decoding: bool
    system_prompt: Optional[str] = "You are a helpful AI assistant for comfort and fitting clothing"
    strategy: Optional[str] = None
    branch_mode: Optional[bool] = False  # Flag to indicate if this is a branch request

class ChatResponse(BaseModel):
    result: str
    duration: int
    ai_message: Message
    image_info: Optional[Dict] = None  # Contains path, name, caption for retrieved images
    thinking_content: Optional[str] = Field(None, alias="thinkingContent")  # Thinking process content

# ---- documents ----
class DocumentAppendResponse(BaseModel):
    detail: str

# ---- branches ----
class BranchBase(BaseModel):
    conversation_id: str
    parent_branch_id: Optional[str] = None

class BranchCreate(BranchBase):
    id: str
    edit_at_id: Optional[str]
    created_at: datetime
    messages: List[Message]

    model_config = ConfigDict(from_attributes=True)

# ---- title gen / feedback ----
class TitleRequest(BaseModel):
    conversation_id: str
    user_message: str
    ai_response: str

class FeedbackRequest(BaseModel):
    rating: int

# ---- history ----
class BranchItem(BaseModel):
    branch_id: Optional[str] = Field(None, alias="branchId")
    is_original: bool = Field(..., alias="isOriginal")
    messages: List[Message]

    model_config = ConfigDict(validate_by_name=True, populate_by_name=True)

class History(BaseModel):
    messages: List[Message]
    original_messages: List[Message] = Field(..., alias="originalMessages")
    branchesByEditId: Dict[str, List[BranchItem]] = Field(..., alias="branchesByEditId")
    currentBranchIndexByEditId: Dict[str, int] = Field(..., alias="currentBranchIndexByEditId")
    activeBranchId: Optional[str] = Field(None, alias="activeBranchId")

    model_config = ConfigDict(validate_by_name=True, populate_by_name=True)
