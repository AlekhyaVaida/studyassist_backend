"""Notebook and page models for storing editor content."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid


class TipTapJSON(BaseModel):
    """TipTap JSON content structure."""
    type: str
    attrs: Optional[Dict[str, Any]] = None
    content: Optional[List['TipTapJSON']] = None
    marks: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None
    
    class Config:
        # Allow recursive models
        pass


# Resolve forward reference for recursive type
TipTapJSON.model_rebuild()


# Request Models
class NotebookCreate(BaseModel):
    """Request schema for creating a notebook."""
    title: Optional[str] = Field(None, description="Notebook title")
    description: Optional[str] = Field(None, description="Notebook description")
    prompt: Optional[str] = Field(None, description="Prompt (used as description if provided)")
    documents: Optional[List[Dict[str, Any]]] = Field(None, description="Attached documents")


class NotebookUpdate(BaseModel):
    """Request schema for updating a notebook."""
    title: Optional[str] = Field(None, description="Notebook title")
    description: Optional[str] = Field(None, description="Notebook description")


class PageCreate(BaseModel):
    """Request schema for creating a page."""
    title: Optional[str] = Field(None, description="Page title")


class PageUpdate(BaseModel):
    """Request schema for updating a page."""
    title: Optional[str] = Field(None, description="Page title")
    content: Optional[Dict[str, Any]] = Field(
        None,
        description="TipTap/ProseMirror JSONContent format. Must be null or a valid document with type 'doc'"
    )
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate that content is either None or a valid TipTap/ProseMirror document."""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("Content must be a valid JSON object or null")
        
        # Ensure root document has type "doc"
        if v.get("type") != "doc":
            raise ValueError("Content must be a valid TipTap/ProseMirror document with type 'doc' at the root")
        
        return v


# Response Models
class PageResponse(BaseModel):
    """Response schema for a page."""
    id: str
    title: str
    content: Optional[Dict[str, Any]] = Field(
        None,
        description="TipTap/ProseMirror JSONContent format. Null for empty pages."
    )
    updatedAt: str
    createdAt: str
    notebookId: str
    
    @classmethod
    def from_dict(cls, page_dict: dict):
        """Create PageResponse from MongoDB document."""
        created_at = page_dict.get("created_at")
        updated_at = page_dict.get("updated_at")
        
        if isinstance(created_at, datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = datetime.now(timezone.utc).isoformat()
            
        if isinstance(updated_at, datetime):
            updated_at_str = updated_at.isoformat()
        else:
            updated_at_str = datetime.now(timezone.utc).isoformat()
        
        return cls(
            id=str(page_dict["_id"]),
            title=page_dict.get("title", "Untitled Page"),
            content=page_dict.get("content"),
            updatedAt=updated_at_str,
            createdAt=created_at_str,
            notebookId=str(page_dict["notebook_id"])
        )


class Document(BaseModel):
    """Document model."""
    id: str = Field(..., description="Document ID")
    name: str = Field(..., description="Document name")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    url: str = Field(..., description="Document URL or file path")
    type: str = Field(..., description="Document type (pdf, image, url, etc.)")
    uploaded_at: str = Field(..., description="ISO 8601 timestamp of upload")
    
    @classmethod
    def from_dict(cls, doc_dict: dict):
        """Create Document from dictionary."""
        uploaded_at = doc_dict.get("uploaded_at")
        if isinstance(uploaded_at, datetime):
            uploaded_at_str = uploaded_at.isoformat()
        else:
            uploaded_at_str = uploaded_at if isinstance(uploaded_at, str) else datetime.now(timezone.utc).isoformat()
        
        return cls(
            id=str(doc_dict.get("id", doc_dict.get("_id", ""))),
            name=doc_dict.get("name", ""),
            filename=doc_dict.get("filename", doc_dict.get("name", "")),
            size=doc_dict.get("size", 0),
            url=doc_dict.get("url", ""),
            type=doc_dict.get("type", "unknown"),
            uploaded_at=uploaded_at_str
        )


class NotebookResponse(BaseModel):
    """Response schema for a notebook."""
    id: str
    title: str
    description: Optional[str] = None
    color: Optional[str] = None
    updatedAt: str
    createdAt: str
    pagesCount: int
    documentsCount: int
    flashCardsCount: int = 0
    documents: Optional[List[Document]] = None
    pages: Optional[List[PageResponse]] = None
    
    @classmethod
    def from_dict(cls, notebook_dict: dict, pages_count: int = 0, documents_count: int = 0, flash_cards_count: int = 0, pages: Optional[List[PageResponse]] = None, documents: Optional[List[Document]] = None):
        """Create NotebookResponse from MongoDB document."""
        created_at = notebook_dict.get("created_at")
        updated_at = notebook_dict.get("updated_at")
        
        if isinstance(created_at, datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = datetime.now(timezone.utc).isoformat()
            
        if isinstance(updated_at, datetime):
            updated_at_str = updated_at.isoformat()
        else:
            updated_at_str = datetime.now(timezone.utc).isoformat()
        
        # Parse documents if not provided
        if documents is None:
            documents_list = notebook_dict.get("documents", [])
            if documents_list:
                documents = [
                    Document.from_dict(doc) if isinstance(doc, dict) else doc
                    for doc in documents_list
                ]
            else:
                documents = []
        
        return cls(
            id=str(notebook_dict["_id"]),
            title=notebook_dict.get("title", "Untitled Notebook"),
            description=notebook_dict.get("description"),
            color=notebook_dict.get("color"),
            updatedAt=updated_at_str,
            createdAt=created_at_str,
            pagesCount=pages_count,
            documentsCount=documents_count,
            flashCardsCount=flash_cards_count,
            documents=documents,
            pages=pages
        )


# Legacy models (kept for backward compatibility with existing pages router)
class PageContentCreate(BaseModel):
    """Request schema for creating/updating page content."""
    content: TipTapJSON = Field(..., description="TipTap JSON content")


class PageContentResponse(BaseModel):
    """Response schema for page content."""
    success: bool = True
    pageId: str
    content: TipTapJSON
    savedAt: str
    
    class Config:
        populate_by_name = True


class Notebook(BaseModel):
    """Notebook model - contains multiple sheets/topics."""
    id: Optional[str] = None
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(..., description="Notebook title")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_dict(cls, notebook_dict: dict):
        """Create Notebook from MongoDB document."""
        return cls(
            id=str(notebook_dict["_id"]),
            user_id=notebook_dict["user_id"],
            title=notebook_dict["title"],
            created_at=notebook_dict.get("created_at", datetime.now(timezone.utc)),
            updated_at=notebook_dict.get("updated_at", datetime.now(timezone.utc)),
        )


class Page(BaseModel):
    """Page/Sheet model - stores editor content for a topic."""
    id: Optional[str] = None
    notebook_id: Optional[str] = Field(None, description="Parent notebook ID")
    user_id: str = Field(..., description="Owner user ID")
    title: Optional[str] = Field(None, description="Page/topic title")
    content: TipTapJSON = Field(default_factory=lambda: TipTapJSON(type="doc", content=[]))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_dict(cls, page_dict: dict):
        """Create Page from MongoDB document."""
        content_data = page_dict.get("content", {"type": "doc", "content": []})
        # Parse content into TipTapJSON if it's a dict
        if isinstance(content_data, dict):
            content = TipTapJSON(**content_data)
        else:
            content = content_data
        
        return cls(
            id=str(page_dict["_id"]),
            notebook_id=str(page_dict["notebook_id"]) if page_dict.get("notebook_id") else None,
            user_id=page_dict["user_id"],
            title=page_dict.get("title"),
            content=content,
            created_at=page_dict.get("created_at", datetime.now(timezone.utc)),
            updated_at=page_dict.get("updated_at", datetime.now(timezone.utc)),
        )


# Chat Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., description="User's question/query")
    context: Optional[str] = Field(None, description="Selected text from editor (if any)")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    messages: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous conversation messages")


class ChatUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    id: str = Field(..., description="Message ID")
    role: str = Field(default="assistant", description="Message role")
    content: str = Field(..., description="AI's response text")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    conversation_id: str = Field(..., description="Conversation ID")
    usage: ChatUsage = Field(..., description="Token usage information")


# Flashcard Models
class FlashCardResponse(BaseModel):
    """Response schema for a flashcard."""
    id: str
    notebookId: str
    notebookTitle: str
    question: str
    answer: Optional[str] = None
    explanation: Optional[str] = None
    timesAnswered: int = 0
    timesAnsweredCorrectly: int = 0
    createdAt: str
    updatedAt: str


class FlashCardCreateRequest(BaseModel):
    """Request schema for creating a flashcard."""
    notebookId: str = Field(..., description="Notebook ID")
    pageId: str = Field(..., description="Page ID")
    content: str = Field(..., description="Selected text content from editor")


class FlashCardSubmissionRequest(BaseModel):
    """Request schema for submitting a flashcard answer."""
    answer: str = Field(..., description="User's answer")


class FlashCardSubmissionResponse(BaseModel):
    """Response schema for flashcard submission."""
    isCorrect: bool
    correctAnswer: str
    explanation: str
