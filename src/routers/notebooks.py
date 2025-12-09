"""Notebooks router for managing notebooks and pages."""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from bson.errors import InvalidId
import uuid

from src.auth import get_current_user
from src.database import get_database
from src.config import config
from src.rag.query import query_rag
from src.models.notebook import (
    NotebookCreate,
    NotebookUpdate,
    NotebookResponse,
    PageCreate,
    PageUpdate,
    PageResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    Document,
)
from src.services.document_processor import process_documents
from src.services.llm_service import generate_pages_from_documents

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


def validate_object_id(id_str: str) -> ObjectId:
    """Validate and convert string ID to ObjectId."""
    try:
        return ObjectId(id_str)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ID format: {id_str}"
        )


async def get_notebook_or_404(notebook_id: str, user_id: str):
    """Get notebook by ID and verify ownership. Raises 404 if not found."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    notebook_object_id = validate_object_id(notebook_id)
    notebook = await database.notebooks.find_one({"_id": notebook_object_id})
    
    if not notebook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook not found: {notebook_id}"
        )
    
    if str(notebook["user_id"]) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook not found: {notebook_id}"
        )
    
    return notebook


async def get_page_or_404(notebook_id: str, page_id: str, user_id: str):
    """Get page by ID and verify ownership. Raises 404 if not found."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    # Verify notebook exists and belongs to user
    await get_notebook_or_404(notebook_id, user_id)
    
    page_object_id = validate_object_id(page_id)
    page = await database.pages.find_one({"_id": page_object_id})
    
    if not page:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page not found: {page_id}"
        )
    
    if str(page["user_id"]) != user_id or str(page["notebook_id"]) != notebook_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page not found: {page_id}"
        )
    
    return page


async def count_pages(notebook_id: ObjectId) -> int:
    """Count pages for a notebook."""
    database = await get_database()
    if database is None:
        return 0
    return await database.pages.count_documents({"notebook_id": notebook_id})


async def count_flashcards(notebook_id: ObjectId, user_id: str) -> int:
    """Count flashcards for a notebook."""
    database = await get_database()
    if database is None:
        return 0
    return await database.flashcards.count_documents({
        "notebook_id": notebook_id,
        "user_id": user_id
    })


# Notebook Endpoints

@router.get("/", response_model=List[NotebookResponse])
async def list_notebooks(current_user: dict = Depends(get_current_user)):
    """Get all notebooks for the authenticated user."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    
    # Fetch notebooks sorted by updated_at descending
    cursor = database.notebooks.find({"user_id": user_id}).sort("updated_at", -1)
    notebooks = []
    
    async for notebook_doc in cursor:
        notebook_id = notebook_doc["_id"]
        pages_count = await count_pages(notebook_id)
        documents_list = notebook_doc.get("documents", [])
        documents_count = len(documents_list)
        flash_cards_count = await count_flashcards(notebook_id, user_id)
        
        # Parse documents for response
        documents = []
        if documents_list:
            for doc in documents_list:
                if isinstance(doc, dict):
                    documents.append(Document.from_dict(doc))
        
        notebooks.append(NotebookResponse.from_dict(
            notebook_doc, 
            pages_count=pages_count, 
            documents_count=documents_count,
            flash_cards_count=flash_cards_count,
            documents=documents
        ))
    
    return notebooks


@router.get("/{notebook_id}", response_model=NotebookResponse)
async def get_notebook(
    notebook_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a single notebook with its pages and documents."""
    user_id = str(current_user["_id"])
    notebook = await get_notebook_or_404(notebook_id, user_id)
    
    database = await get_database()
    notebook_object_id = notebook["_id"]
    
    # Count pages, documents, and flashcards
    pages_count = await count_pages(notebook_object_id)
    documents_list = notebook.get("documents", [])
    documents_count = len(documents_list)
    flash_cards_count = await count_flashcards(notebook_object_id, user_id)
    
    # Parse documents
    documents = []
    if documents_list:
        for doc in documents_list:
            if isinstance(doc, dict):
                # Ensure document has required fields with proper structure
                doc_with_id = {
                    "id": doc.get("id", str(uuid.uuid4())),
                    "name": doc.get("name", doc.get("filename", "Unknown")),
                    "filename": doc.get("filename", doc.get("name", "Unknown")),
                    "size": doc.get("size", 0),
                    "url": doc.get("url", doc.get("path", "")),
                    "type": doc.get("type", "unknown"),
                    "uploaded_at": doc.get("uploaded_at", notebook.get("created_at", datetime.now(timezone.utc)))
                }
                documents.append(Document.from_dict(doc_with_id))
    
    # Fetch pages sorted by updated_at descending
    cursor = database.pages.find({"notebook_id": notebook_object_id}).sort("updated_at", -1)
    pages = []
    async for page_doc in cursor:
        pages.append(PageResponse.from_dict(page_doc))
    
    return NotebookResponse.from_dict(
        notebook, 
        pages_count=pages_count, 
        documents_count=documents_count,
        flash_cards_count=flash_cards_count,
        pages=pages,
        documents=documents
    )


@router.post("/", response_model=NotebookResponse, status_code=status.HTTP_201_CREATED)
async def create_notebook(
    notebook_data: NotebookCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new notebook and process documents if provided."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    now = datetime.now(timezone.utc)
    
    query = notebook_data.prompt  # Store query for document processing
    
    # Process documents first if provided to generate notebook metadata
    notebook_title = notebook_data.title
    notebook_description = notebook_data.description
    documents_content = None
    toc_items = None
    
    if notebook_data.documents:
        try:
            # Generate a temporary notebook ID for document storage (using ObjectId)
            from bson import ObjectId
            temp_notebook_id = str(ObjectId())
            
            # Process all documents to get markdown
            print("Processing documents...")
            documents_content = process_documents(notebook_data.documents, temp_notebook_id, user_id)
            
            if documents_content:
                # Step 1: Extract table of contents
                from src.services.llm_service import extract_table_of_contents, generate_notebook_metadata
                print("Extracting table of contents...")
                toc_items = extract_table_of_contents(documents_content)
                
                # Step 2: Generate notebook title and description from documents + TOC
                print("Generating notebook title and description from documents...")
                metadata = generate_notebook_metadata(documents_content, toc_items, query)
                
                # Use LLM-generated metadata, but allow user override
                notebook_title = notebook_title or metadata["title"]
                notebook_description = notebook_description or metadata["description"]
                
                print(f"Generated notebook: '{notebook_title}' - {notebook_description}")
        except Exception as e:
            print(f"Error processing documents for metadata generation: {e}")
            import traceback
            traceback.print_exc()
            # Continue with user-provided or default values
            if not notebook_title:
                notebook_title = "Untitled Notebook"
    
    # Process and format documents for storage
    formatted_documents = []
    temp_notebook_id_for_docs = str(uuid.uuid4())  # Temporary ID for saving documents before notebook is created
    if notebook_data.documents:
        for doc in notebook_data.documents:
            if isinstance(doc, dict):
                # Get file path/URL from save_document
                doc_path = None
                try:
                    from src.services.document_processor import save_document
                    doc_path = save_document(doc, temp_notebook_id_for_docs)
                except Exception as e:
                    print(f"Error saving document {doc.get('name', 'unknown')}: {e}")
                    doc_path = doc.get("url", "")
                
                # Calculate size if data is provided
                doc_size = doc.get("size", 0)
                if not doc_size and doc.get("data"):
                    # Estimate size from base64 data (base64 is ~33% larger than original)
                    doc_size = len(doc.get("data", "")) * 3 // 4
                
                # Format document with proper structure
                formatted_doc = {
                    "id": str(uuid.uuid4()),
                    "name": doc.get("name", doc.get("filename", "Unknown")),
                    "filename": doc.get("filename", doc.get("name", "Unknown")),
                    "size": doc_size,
                    "url": doc_path or doc.get("url", ""),
                    "type": doc.get("type", "unknown"),
                    "uploaded_at": now
                }
                formatted_documents.append(formatted_doc)
    
    # Create the actual notebook with generated or user-provided metadata
    notebook_dict = {
        "user_id": user_id,
        "title": notebook_title or "Untitled Notebook",
        "description": notebook_description,
        "color": None,
        "documents": formatted_documents,
        "created_at": now,
        "updated_at": now,
    }
    
    result = await database.notebooks.insert_one(notebook_dict)
    created_notebook = await database.notebooks.find_one({"_id": result.inserted_id})
    
    if not created_notebook:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create notebook"
        )
    
    notebook_id = str(created_notebook["_id"])
    notebook_object_id = created_notebook["_id"]
    
    # Move documents from temp folder to actual notebook folder and update URLs
    if formatted_documents:
        try:
            from pathlib import Path
            import shutil
            temp_dir = config.data_dir / "notebooks" / temp_notebook_id_for_docs
            actual_dir = config.data_dir / "notebooks" / notebook_id
            if temp_dir.exists():
                actual_dir.mkdir(parents=True, exist_ok=True)
                # Update document URLs in database
                updated_documents = []
                for doc in formatted_documents:
                    if doc["url"] and temp_notebook_id_for_docs in doc["url"]:
                        # Extract filename from temp path
                        filename = Path(doc["url"]).name
                        new_path = actual_dir / filename
                        # Move file if it exists
                        old_path = Path(doc["url"])
                        if old_path.exists():
                            shutil.move(str(old_path), str(new_path))
                            doc["url"] = str(new_path)
                    updated_documents.append(doc)
                
                # Update notebook with corrected document URLs
                await database.notebooks.update_one(
                    {"_id": notebook_object_id},
                    {"$set": {"documents": updated_documents}}
                )
                
                # Clean up temp directory
                try:
                    temp_dir.rmdir()
                except:
                    pass
        except Exception as e:
            print(f"Warning: Could not move documents to notebook folder: {e}")
    
    # Generate pages if documents were processed
    pages_created = 0
    if documents_content and toc_items:
        try:
            # Generate pages from processed documents using LLM
            # Generate pages in parallel (one per TOC item)
            # Use RAG to retrieve relevant context from user's documents
            generated_pages = await generate_pages_from_documents(
                documents_content,
                notebook_title=notebook_title,
                notebook_description=notebook_description,
                query=query,
                user_id=user_id,
                notebook_id=notebook_id,
                max_workers=5  # Generate up to 5 pages concurrently
            )
            
            # Create pages in database one by one
            for page_data in generated_pages:
                page_dict = {
                    "notebook_id": notebook_object_id,
                    "user_id": user_id,
                    "title": page_data.get("title", "Untitled Page"),
                    "content": page_data.get("content"),  # TipTap JSON format
                    "created_at": now,
                    "updated_at": now,
                }
                await database.pages.insert_one(page_dict)
                pages_created += 1
                print(f"Created page: {page_data.get('title')}")
        except Exception as e:
            # Log error but don't fail notebook creation
            print(f"Error processing documents for notebook {notebook_id}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with notebook creation even if document processing fails
    
    documents_list = created_notebook.get("documents", [])
    documents_count = len(documents_list)
    flash_cards_count = await count_flashcards(notebook_object_id, user_id)  # Will be 0 for new notebook
    
    # Fetch updated notebook with pages
    updated_notebook = await database.notebooks.find_one({"_id": notebook_object_id})
    
    # Parse documents for response
    documents = []
    if documents_list:
        for doc in documents_list:
            if isinstance(doc, dict):
                documents.append(Document.from_dict(doc))
    
    return NotebookResponse.from_dict(
        updated_notebook, 
        pages_count=pages_created, 
        documents_count=documents_count,
        flash_cards_count=flash_cards_count,
        documents=documents
    )


@router.patch("/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(
    notebook_id: str,
    notebook_data: NotebookUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update notebook title and/or description."""
    user_id = str(current_user["_id"])
    notebook = await get_notebook_or_404(notebook_id, user_id)
    
    database = await get_database()
    notebook_object_id = notebook["_id"]
    
    # Build update data
    update_data = {"updated_at": datetime.now(timezone.utc)}
    
    if notebook_data.title is not None:
        update_data["title"] = notebook_data.title
    
    if notebook_data.description is not None:
        update_data["description"] = notebook_data.description
    
    # Update notebook
    await database.notebooks.update_one(
        {"_id": notebook_object_id},
        {"$set": update_data}
    )
    
    # Fetch updated notebook
    updated_notebook = await database.notebooks.find_one({"_id": notebook_object_id})
    pages_count = await count_pages(notebook_object_id)
    documents_list = updated_notebook.get("documents", [])
    documents_count = len(documents_list)
    flash_cards_count = await count_flashcards(notebook_object_id, user_id)
    
    # Parse documents for response
    documents = []
    if documents_list:
        for doc in documents_list:
            if isinstance(doc, dict):
                documents.append(Document.from_dict(doc))
    
    return NotebookResponse.from_dict(
        updated_notebook, 
        pages_count=pages_count, 
        documents_count=documents_count,
        flash_cards_count=flash_cards_count,
        documents=documents
    )


@router.delete("/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notebook(
    notebook_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a notebook and all its pages."""
    user_id = str(current_user["_id"])
    notebook = await get_notebook_or_404(notebook_id, user_id)
    
    database = await get_database()
    notebook_object_id = notebook["_id"]
    
    # Count pages before deletion for logging
    pages_count_before = await database.pages.count_documents({"notebook_id": notebook_object_id})
    
    # Delete all pages in the notebook
    pages_result = await database.pages.delete_many({"notebook_id": notebook_object_id})
    print(f"Deleted {pages_result.deleted_count} pages from notebook {notebook_id}")
    
    # Verify all pages were deleted
    pages_count_after = await database.pages.count_documents({"notebook_id": notebook_object_id})
    if pages_count_after > 0:
        print(f"Warning: {pages_count_after} pages still exist after deletion attempt")
    
    # Delete documents from vector store
    try:
        from src.rag.store import get_user_vector_store
        vector_store = get_user_vector_store(user_id)
        vector_store.delete_by_notebook(notebook_id)
        print(f"Deleted vector store chunks for notebook {notebook_id}")
    except Exception as e:
        print(f"Warning: Could not delete documents from vector store: {e}")
    
    # Delete the notebook
    result = await database.notebooks.delete_one({"_id": notebook_object_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete notebook"
        )
    
    print(f"Successfully deleted notebook {notebook_id} with {pages_result.deleted_count} pages")


# Page Endpoints

@router.get("/{notebook_id}/pages", response_model=List[PageResponse])
async def list_pages(
    notebook_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all pages for a notebook."""
    user_id = str(current_user["_id"])
    await get_notebook_or_404(notebook_id, user_id)
    
    database = await get_database()
    notebook_object_id = validate_object_id(notebook_id)
    
    # Fetch pages sorted by updated_at descending
    cursor = database.pages.find({"notebook_id": notebook_object_id}).sort("updated_at", -1)
    pages = []
    async for page_doc in cursor:
        pages.append(PageResponse.from_dict(page_doc))
    
    return pages


@router.get("/{notebook_id}/pages/{page_id}", response_model=PageResponse)
async def get_page(
    notebook_id: str,
    page_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a single page with its content."""
    user_id = str(current_user["_id"])
    page = await get_page_or_404(notebook_id, page_id, user_id)
    
    return PageResponse.from_dict(page)


@router.post("/{notebook_id}/pages", response_model=PageResponse, status_code=status.HTTP_201_CREATED)
async def create_page(
    notebook_id: str,
    page_data: PageCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new page in a notebook."""
    user_id = str(current_user["_id"])
    notebook = await get_notebook_or_404(notebook_id, user_id)
    
    database = await get_database()
    notebook_object_id = notebook["_id"]
    now = datetime.now(timezone.utc)
    
    # Create page
    page_dict = {
        "notebook_id": notebook_object_id,
        "user_id": user_id,
        "title": page_data.title or "Untitled Page",
        "content": None,
        "created_at": now,
        "updated_at": now,
    }
    
    result = await database.pages.insert_one(page_dict)
    created_page = await database.pages.find_one({"_id": result.inserted_id})
    
    if not created_page:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create page"
        )
    
    # Update notebook's updated_at
    await database.notebooks.update_one(
        {"_id": notebook_object_id},
        {"$set": {"updated_at": now}}
    )
    
    return PageResponse.from_dict(created_page)


@router.patch("/{notebook_id}/pages/{page_id}", response_model=PageResponse)
async def update_page(
    notebook_id: str,
    page_id: str,
    page_data: PageUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update page title and/or content."""
    user_id = str(current_user["_id"])
    page = await get_page_or_404(notebook_id, page_id, user_id)
    
    database = await get_database()
    page_object_id = page["_id"]
    notebook_object_id = validate_object_id(notebook_id)
    now = datetime.now(timezone.utc)
    
    # Build update data
    update_data = {"updated_at": now}
    
    if page_data.title is not None:
        update_data["title"] = page_data.title
    
    # Handle content update: content is stored as MongoDB document (dict), not stringified
    # Check if content was explicitly provided (even if None) using model_dump
    provided_fields = page_data.model_dump(exclude_unset=True)
    
    if "content" in provided_fields:
        # Content was explicitly provided - store as dict directly in MongoDB
        update_data["content"] = provided_fields["content"]
        # Update notebook's updated_at when content changes
        await database.notebooks.update_one(
            {"_id": notebook_object_id},
            {"$set": {"updated_at": now}}
        )
    
    # Update page
    await database.pages.update_one(
        {"_id": page_object_id},
        {"$set": update_data}
    )
    
    # Fetch updated page
    updated_page = await database.pages.find_one({"_id": page_object_id})
    
    return PageResponse.from_dict(updated_page)


@router.delete("/{notebook_id}/pages/{page_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_page(
    notebook_id: str,
    page_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a page from a notebook."""
    user_id = str(current_user["_id"])
    page = await get_page_or_404(notebook_id, page_id, user_id)
    
    database = await get_database()
    page_object_id = page["_id"]
    notebook_object_id = validate_object_id(notebook_id)
    
    # Delete page
    result = await database.pages.delete_one({"_id": page_object_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete page"
        )
    
    # Update notebook's updated_at
    await database.notebooks.update_one(
        {"_id": notebook_object_id},
        {"$set": {"updated_at": datetime.now(timezone.utc)}}
    )


@router.post("/{notebook_id}/pages/{page_id}/chat", response_model=ChatResponse)
async def chat_with_page(
    notebook_id: str,
    page_id: str,
    chat_request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """Chat with AI assistant using RAG for a specific page.
    
    Uses RAG to retrieve relevant context from user's documents and generates
    a response based on the conversation history and selected context.
    """
    user_id = str(current_user["_id"])
    
    # Verify notebook and page ownership
    notebook = await get_notebook_or_404(notebook_id, user_id)
    page = await get_page_or_404(notebook_id, page_id, user_id)
    
    # Generate or use existing conversation ID
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())
    
    # Build conversation history for context
    conversation_messages = []
    if chat_request.messages:
        for msg in chat_request.messages:
            conversation_messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # Add current user message
    conversation_messages.append({
        "role": "user",
        "content": chat_request.message
    })
    
    # Extract full page content if no context is provided
    page_content_text = ""
    if not chat_request.context and page.get("content"):
        # Convert TipTap JSON content to plain text
        def extract_text_from_tiptap(content_node):
            """Recursively extract text from TipTap JSON structure."""
            if isinstance(content_node, dict):
                text_parts = []
                # Extract text if present
                if "text" in content_node:
                    text_parts.append(content_node["text"])
                # Recursively process content array
                if "content" in content_node and isinstance(content_node["content"], list):
                    for child in content_node["content"]:
                        text_parts.append(extract_text_from_tiptap(child))
                return " ".join(text_parts)
            return ""
        
        page_content_text = extract_text_from_tiptap(page.get("content", {}))
        print(f"Extracted {len(page_content_text)} chars from page content")
    
    # Build query for RAG - prioritize context text for better retrieval
    # When user selects text and asks to summarize/explain, use that text for RAG search
    rag_query = chat_request.message
    if chat_request.context:
        # Use the selected context text for RAG search to find related content
        # This helps find additional relevant information from documents
        rag_query = chat_request.context
        print(f"Using context text for RAG query: {rag_query[:100]}...")
    elif page_content_text:
        # Use page content for RAG search if no context provided
        rag_query = page_content_text[:500]  # Use first 500 chars for RAG query
        print(f"Using page content for RAG query: {rag_query[:100]}...")
    
    # Query RAG system for relevant context
    try:
        print(f"Querying RAG with: {rag_query[:100]}...")
        rag_result = query_rag(
            user_id=user_id,
            query=rag_query,
            notebook_id=notebook_id,
            k=8,  # Get more chunks for comprehensive context
            return_raw_chunks=True  # Get raw chunks to combine with selected context
        )
        
        rag_chunks = rag_result.get("chunks", [])
        rag_sources = rag_result.get("sources", [])
        print(f"RAG retrieved {len(rag_chunks)} chunks")
        
        # Combine RAG chunks with selected context or full page content
        if chat_request.context:
            # Prepend selected context as it's most relevant
            combined_context = f"Selected Text from Page:\n{chat_request.context}\n\n"
            if rag_chunks:
                combined_context += "Related Content from Your Documents:\n" + "\n\n".join(rag_chunks[:5])  # Limit to top 5 chunks
            rag_context = combined_context
            print(f"Combined context length: {len(rag_context)} chars")
        elif page_content_text:
            # Use full page content with RAG chunks
            combined_context = f"Full Page Content:\n{page_content_text}\n\n"
            if rag_chunks:
                combined_context += "Related Content from Your Documents:\n" + "\n\n".join(rag_chunks[:5])  # Limit to top 5 chunks
            rag_context = combined_context
            print(f"Using full page content with RAG chunks, length: {len(rag_context)} chars")
        else:
            # Just use RAG chunks
            rag_context = "\n\n".join(rag_chunks) if rag_chunks else ""
            print(f"Using RAG chunks only, length: {len(rag_context)} chars")
            
    except Exception as e:
        print(f"Error querying RAG: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: use selected context, page content, or error message
        if chat_request.context:
            rag_context = chat_request.context
        elif page_content_text:
            rag_context = f"Full Page Content:\n{page_content_text}"
        else:
            rag_context = "Unable to retrieve context from documents."
        rag_sources = []
        print(f"Falling back to context: {len(rag_context)} chars")
    
    # Generate response using OpenAI with RAG context and conversation history
    try:
        from openai import OpenAI
        
        if not config.validate_api_key():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API key is not configured"
            )
        
        client = OpenAI(api_key=config.openai_api_key)
        
        # Build messages for OpenAI API
        openai_messages = []
        
        # System message
        system_content = """You are a helpful AI assistant that answers questions based on the provided context from the user's documents and selected text.

Use the provided context to give accurate, helpful answers. If the context doesn't contain enough information, say so clearly.
Be concise but comprehensive in your responses."""
        
        openai_messages.append({"role": "system", "content": system_content})
        
        # Add context as a separate user message if available
        if rag_context and rag_context.strip():
            openai_messages.append({
                "role": "user",
                "content": f"Context:\n{rag_context}"
            })
        
        # Add conversation history (excluding system messages)
        # Only add the last user message, not the full history to avoid duplication
        if conversation_messages:
            # Add the last user message (current one)
            openai_messages.append({
                "role": "user",
                "content": chat_request.message
            })
        
        # Generate response
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=openai_messages,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        usage_info = response.usage
        
        # Create response
        message_id = f"msg_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return ChatResponse(
            id=message_id,
            role="assistant",
            content=assistant_message,
            timestamp=timestamp,
            conversation_id=conversation_id,
            usage={
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.completion_tokens,
                "total_tokens": usage_info.total_tokens
            }
        )
        
    except Exception as e:
        print(f"Error generating chat response: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

