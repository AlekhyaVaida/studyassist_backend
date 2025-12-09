"""Document processing service for storing and ingesting documents."""

import os
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from src.config import config
from src.rag.ingestion import ingest


def save_document(document: Dict[str, Any], notebook_id: str) -> str:
    """Save a document to the data folder and return the file path.
    
    Args:
        document: Document dictionary with 'type', 'name', and 'data' fields
        notebook_id: Notebook ID for organizing files
        
    Returns:
        str: Path to the saved file
    """
    # Create notebook-specific directory
    notebook_dir = config.data_dir / "notebooks" / notebook_id
    notebook_dir.mkdir(parents=True, exist_ok=True)
    
    doc_type = document.get("type", "").lower()
    doc_name = document.get("name", "document")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    
    if doc_type == "pdf":
        file_path = notebook_dir / f"{file_id}.pdf"
        # Decode base64 data
        file_data = base64.b64decode(document.get("data", ""))
        file_path.write_bytes(file_data)
        
    elif doc_type == "image":
        # Determine extension from name or default to png
        ext = Path(doc_name).suffix or ".png"
        file_path = notebook_dir / f"{file_id}{ext}"
        file_data = base64.b64decode(document.get("data", ""))
        file_path.write_bytes(file_data)
        
    elif doc_type == "url":
        # For URLs, we don't save a file, just return the URL
        url = document.get("url", "")
        if not url:
            raise ValueError("URL document must have 'url' field")
        return url
        
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")
    
    return str(file_path)


def process_document(document: Dict[str, Any], notebook_id: str) -> Dict[str, str]:
    """Process a single document: save it and convert to markdown.
    
    Args:
        document: Document dictionary with 'type', 'name', 'data'/'url' fields
        notebook_id: Notebook ID for organizing files
        
    Returns:
        dict: Dictionary with 'titles' and 'markdown' keys
    """
    doc_type = document.get("type", "").lower()
    
    if doc_type == "pdf":
        file_path = save_document(document, notebook_id)
        result = ingest({"type": "pdf", "path": file_path})
        
    elif doc_type == "image":
        file_path = save_document(document, notebook_id)
        result = ingest({"type": "image", "path": file_path})
        
    elif doc_type == "url":
        url = document.get("url", "")
        if not url:
            raise ValueError("URL document must have 'url' field")
        result = ingest({"type": "url", "url": url})
        
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")
    
    return result


def process_documents(
    documents: List[Dict[str, Any]],
    notebook_id: str,
    user_id: str
) -> List[Dict[str, str]]:
    """Process multiple documents and return their markdown content.
    
    Also adds documents to user's vector store for RAG.
    
    Args:
        documents: List of document dictionaries
        notebook_id: Notebook ID for organizing files
        user_id: User ID for vector store scoping
        
    Returns:
        list: List of dictionaries with 'titles' and 'markdown' keys
    """
    processed = []
    
    # Collect all markdown content for vector store
    all_texts = []
    all_metadatas = []
    
    for doc in documents:
        try:
            result = process_document(doc, notebook_id)
            processed.append(result)
            
            # Prepare for vector store
            markdown = result.get("markdown", "")
            titles = result.get("titles", "")
            
            if markdown:
                all_texts.append(markdown)
                all_metadatas.append({
                    "notebook_id": notebook_id,
                    "document_name": doc.get("name", "unknown"),
                    "document_type": doc.get("type", "unknown"),
                    "titles": titles,
                })
        except Exception as e:
            # Log error but continue processing other documents
            print(f"Error processing document {doc.get('name', 'unknown')}: {e}")
            continue
    
    # Add to user's vector store
    if all_texts:
        try:
            from src.rag.store import get_user_vector_store
            vector_store = get_user_vector_store(user_id)
            vector_store.add_documents(all_texts, all_metadatas)
            print(f"Added {len(all_texts)} documents to vector store for user {user_id}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            # Continue even if vector store fails
    
    return processed

