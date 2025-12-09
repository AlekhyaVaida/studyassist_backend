"""Pages router for managing notebook pages/sheets."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from bson.errors import InvalidId

from src.auth import get_current_user
from src.database import get_database
from src.models.notebook import PageContentCreate, PageContentResponse, Page

router = APIRouter(prefix="/pages", tags=["pages"])


def validate_object_id(id_str: str) -> ObjectId:
    """Validate and convert string ID to ObjectId."""
    try:
        return ObjectId(id_str)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid page ID format: {id_str}"
        )


@router.post("/{page_id}/content", response_model=PageContentResponse)
async def save_page_content(
    page_id: str,
    content_data: PageContentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Save or update page content.
    
    This endpoint accepts TipTap JSON content.
    If the page doesn't exist, it will be created. Otherwise, it will be updated.
    """
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    page_object_id = validate_object_id(page_id)
    
    # Check if page exists
    existing_page = await database.pages.find_one({"_id": page_object_id})
    
    now = datetime.now(timezone.utc)
    
    if existing_page:
        # Update existing page
        # Verify ownership
        if str(existing_page["user_id"]) != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this page"
            )
        
        # Update page content
        # Convert TipTapJSON to dict for MongoDB storage
        content_dict = content_data.content.model_dump()
        update_data = {
            "content": content_dict,
            "updated_at": now,
        }
        
        await database.pages.update_one(
            {"_id": page_object_id},
            {"$set": update_data}
        )
        
        # Fetch updated page
        updated_page = await database.pages.find_one({"_id": page_object_id})
        page = Page.from_dict(updated_page)
        
    else:
        # Create new page
        # Convert TipTapJSON to dict for MongoDB storage
        content_dict = content_data.content.model_dump()
        page_dict = {
            "user_id": user_id,
            "content": content_dict,
            "created_at": now,
            "updated_at": now,
        }
        
        result = await database.pages.insert_one(page_dict)
        created_page = await database.pages.find_one({"_id": result.inserted_id})
        page = Page.from_dict(created_page)
    
    return PageContentResponse(
        success=True,
        pageId=page.id,
        content=page.content,
        savedAt=page.updated_at.isoformat()
    )


@router.get("/{page_id}/content", response_model=PageContentResponse)
async def get_page_content(
    page_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get page content by ID."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    page_object_id = validate_object_id(page_id)
    
    page_doc = await database.pages.find_one({"_id": page_object_id})
    
    if not page_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page not found: {page_id}"
        )
    
    # Verify ownership
    if str(page_doc["user_id"]) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this page"
        )
    
    page = Page.from_dict(page_doc)
    
    return PageContentResponse(
        success=True,
        pageId=page.id,
        content=page.content,
        savedAt=page.updated_at.isoformat()
    )


@router.get("/", response_model=list[Page])
async def list_pages(
    notebook_id: str = None,
    current_user: dict = Depends(get_current_user)
):
    """List all pages for the current user, optionally filtered by notebook."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    
    # Build query
    query = {"user_id": user_id}
    if notebook_id:
        query["notebook_id"] = validate_object_id(notebook_id)
    
    # Fetch pages
    cursor = database.pages.find(query).sort("updated_at", -1)
    pages = []
    async for page_doc in cursor:
        pages.append(Page.from_dict(page_doc))
    
    return pages


@router.delete("/{page_id}")
async def delete_page(
    page_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a page."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    page_object_id = validate_object_id(page_id)
    
    # Check if page exists and user owns it
    page_doc = await database.pages.find_one({"_id": page_object_id})
    
    if not page_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page not found: {page_id}"
        )
    
    if str(page_doc["user_id"]) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this page"
        )
    
    # Delete page
    result = await database.pages.delete_one({"_id": page_object_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete page"
        )
    
    return {"success": True, "message": "Page deleted successfully"}

