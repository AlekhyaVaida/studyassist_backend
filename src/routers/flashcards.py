"""Flashcards router for managing flashcards."""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from bson import ObjectId
from bson.errors import InvalidId

from src.auth import get_current_user
from src.database import get_database
from src.config import config
from src.models.notebook import (
    FlashCardResponse,
    FlashCardCreateRequest,
    FlashCardSubmissionRequest,
    FlashCardSubmissionResponse,
)

router = APIRouter(prefix="/flashcards", tags=["flashcards"])


def validate_object_id(id_str: str) -> ObjectId:
    """Validate and convert string ID to ObjectId."""
    try:
        return ObjectId(id_str)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ID format: {id_str}"
        )


async def get_flashcard_or_404(flashcard_id: str, user_id: str):
    """Get flashcard by ID and verify ownership. Raises 404 if not found."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    flashcard_object_id = validate_object_id(flashcard_id)
    flashcard = await database.flashcards.find_one({"_id": flashcard_object_id})
    
    if not flashcard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flashcard not found"
        )
    
    # Verify ownership
    if str(flashcard.get("user_id")) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this flashcard"
        )
    
    return flashcard


def compare_answers(user_answer: str, correct_answer: str) -> dict:
    """Compare user answer with correct answer using LLM for semantic comparison.
    
    Args:
        user_answer: User's submitted answer
        correct_answer: Correct answer from flashcard
        
    Returns:
        dict with 'isCorrect' (bool), 'explanation' (str)
    """
    try:
        from openai import OpenAI
        
        if not config.validate_api_key():
            # Fallback to simple comparison if API key not available
            return {
                "isCorrect": user_answer.strip().lower() == correct_answer.strip().lower(),
                "explanation": "Answer comparison completed."
            }
        
        client = OpenAI(api_key=config.openai_api_key)
        
        prompt = f"""Compare the user's answer with the correct answer and determine if they are semantically equivalent.

Correct Answer: {correct_answer}

User's Answer: {user_answer}

Evaluate if the user's answer:
1. Contains the same key concepts and information as the correct answer
2. Is semantically equivalent (even if worded differently)
3. Demonstrates understanding of the topic

IMPORTANT: Be lenient and focus on correctness, not completeness. If the answer is correct but less detailed than the correct answer, still mark it as correct. Only mark as incorrect if:
- The answer contains factual errors
- The answer demonstrates misunderstanding of the core concept
- The answer is completely off-topic

Return a JSON object with this structure:
{{
  "isCorrect": true/false,
  "explanation": "Brief explanation of why the answer is correct or incorrect, and what the key points should be."
}}

Be lenient - if the user demonstrates understanding of the core concepts correctly, mark it as correct even if:
- The wording differs
- The answer is less detailed than the correct answer
- The answer is more concise than the correct answer

Only mark as incorrect if the answer is factually wrong or shows misunderstanding."""
        
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": "You are an educational assistant that evaluates student answers. Be fair and focus on correctness rather than completeness. If an answer is correct but less detailed, still mark it as correct. Only mark as incorrect if the answer contains factual errors or demonstrates misunderstanding."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        import json
        comparison = json.loads(result)
        
        return {
            "isCorrect": comparison.get("isCorrect", False),
            "explanation": comparison.get("explanation", "Answer evaluated.")
        }
        
    except Exception as e:
        print(f"Error comparing answers with LLM: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to simple comparison
        return {
            "isCorrect": user_answer.strip().lower() == correct_answer.strip().lower(),
            "explanation": "Answer comparison completed. (Note: Using fallback comparison method)"
        }


def generate_flashcard_from_content(content: str, notebook_id: str, user_id: str) -> dict:
    """Generate a flashcard question and answer from selected content using RAG and LLM.
    
    Args:
        content: Selected text content from editor
        notebook_id: Notebook ID for RAG context
        user_id: User ID for RAG query
        
    Returns:
        dict with 'question', 'answer', 'explanation'
    """
    try:
        from openai import OpenAI
        from src.rag.query import query_rag
        
        if not config.validate_api_key():
            raise ValueError("OpenAI API key is not configured")
        
        client = OpenAI(api_key=config.openai_api_key)
        
        # Get relevant context from RAG
        rag_context = ""
        try:
            rag_result = query_rag(
                user_id=user_id,
                query=content,
                notebook_id=notebook_id,
                k=5,
                return_raw_chunks=True
            )
            rag_chunks = rag_result.get("chunks", [])
            if rag_chunks:
                rag_context = "\n\n".join(rag_chunks[:3])  # Use top 3 chunks
        except Exception as e:
            print(f"Warning: Could not retrieve RAG context: {e}")
            rag_context = ""
        
        # Build prompt for LLM to generate flashcard
        system_prompt = """You are an educational assistant that creates effective flashcards from study material.
Create a flashcard with:
1. A clear, concise question that tests understanding of the key concept
2. A comprehensive answer that explains the concept clearly
3. A brief explanation that helps reinforce learning

The question should be specific and testable. The answer should be accurate and educational."""
        
        user_prompt = f"""Create a flashcard from the following content:

Selected Content:
{content}

"""
        
        if rag_context:
            user_prompt += f"""Additional Context from Documents:
{rag_context}

"""
        
        user_prompt += """Generate a JSON object with this structure:
{
  "question": "A clear, testable question about the key concept",
  "answer": "A comprehensive answer explaining the concept",
  "explanation": "A brief explanation that reinforces learning and provides additional context"
}

Focus on extracting the most important concept from the content and creating a question that tests understanding."""
        
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        import json
        flashcard_data = json.loads(result)
        
        return {
            "question": flashcard_data.get("question", "What is the key concept?"),
            "answer": flashcard_data.get("answer", ""),
            "explanation": flashcard_data.get("explanation", "")
        }
        
    except Exception as e:
        print(f"Error generating flashcard: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: create simple flashcard from content
        return {
            "question": "What is the key concept in this content?",
            "answer": content[:500],  # Use first 500 chars as answer
            "explanation": "This flashcard was created from the selected content."
        }


@router.post("", response_model=FlashCardResponse, status_code=status.HTTP_201_CREATED)
async def create_flashcard(
    request: FlashCardCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a flashcard from selected content using AI/RAG."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    
    # Validate notebook and page ownership
    notebook_object_id = validate_object_id(request.notebookId)
    page_object_id = validate_object_id(request.pageId)
    
    # Verify notebook belongs to user
    notebook = await database.notebooks.find_one({"_id": notebook_object_id})
    if not notebook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notebook not found"
        )
    if str(notebook.get("user_id")) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this notebook"
        )
    
    # Verify page belongs to notebook and user
    page = await database.pages.find_one({"_id": page_object_id})
    if not page:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Page not found"
        )
    if str(page.get("notebook_id")) != request.notebookId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page does not belong to the specified notebook"
        )
    if str(page.get("user_id")) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this page"
        )
    
    # Generate flashcard using AI/RAG
    flashcard_data = generate_flashcard_from_content(
        content=request.content,
        notebook_id=request.notebookId,
        user_id=user_id
    )
    
    # Create flashcard document
    now = datetime.now(timezone.utc)
    flashcard_dict = {
        "user_id": user_id,
        "notebook_id": notebook_object_id,
        "page_id": page_object_id,
        "question": flashcard_data["question"],
        "answer": flashcard_data["answer"],
        "explanation": flashcard_data["explanation"],
        "times_answered": 0,
        "times_answered_correctly": 0,
        "created_at": now,
        "updated_at": now,
    }
    
    result = await database.flashcards.insert_one(flashcard_dict)
    created_flashcard = await database.flashcards.find_one({"_id": result.inserted_id})
    
    if not created_flashcard:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create flashcard"
        )
    
    # Get notebook title for response
    notebook_title = notebook.get("title", "Unknown Notebook")
    
    created_at_str = created_flashcard.get("created_at").isoformat() if isinstance(created_flashcard.get("created_at"), datetime) else now.isoformat()
    updated_at_str = created_flashcard.get("updated_at").isoformat() if isinstance(created_flashcard.get("updated_at"), datetime) else now.isoformat()
    
    return FlashCardResponse(
        id=str(created_flashcard["_id"]),
        notebookId=request.notebookId,
        notebookTitle=notebook_title,
        question=created_flashcard.get("question", ""),
        answer=created_flashcard.get("answer"),  # Include answer in creation response
        explanation=created_flashcard.get("explanation"),  # Include explanation in creation response
        timesAnswered=created_flashcard.get("times_answered", 0),
        timesAnsweredCorrectly=created_flashcard.get("times_answered_correctly", 0),
        createdAt=created_at_str,
        updatedAt=updated_at_str
    )


@router.get("", response_model=List[FlashCardResponse])
async def get_flashcards(
    notebook_id: Optional[str] = Query(None, alias="notebook_id", description="Filter flashcards by notebook ID"),
    current_user: dict = Depends(get_current_user)
):
    """Get all flashcards for the authenticated user, optionally filtered by notebook ID."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    
    # Build query filter
    query_filter = {"user_id": user_id}
    
    # Add notebook filter if provided
    if notebook_id:
        try:
            notebook_object_id = validate_object_id(notebook_id)
            # Verify notebook belongs to user
            notebook = await database.notebooks.find_one({"_id": notebook_object_id})
            if not notebook:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Notebook not found"
                )
            if str(notebook.get("user_id")) != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to access this notebook"
                )
            query_filter["notebook_id"] = notebook_object_id
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid notebook_id format: {str(e)}"
            )
    
    # Fetch flashcards sorted by updated_at descending
    cursor = database.flashcards.find(query_filter).sort("updated_at", -1)
    flashcards = []
    
    async for flashcard_doc in cursor:
        flashcard_notebook_id = flashcard_doc.get("notebook_id")
        notebook_title = "Unknown Notebook"
        
        # Fetch notebook title
        if flashcard_notebook_id:
            notebook = await database.notebooks.find_one({"_id": flashcard_notebook_id})
            if notebook:
                notebook_title = notebook.get("title", "Unknown Notebook")
        
        created_at = flashcard_doc.get("created_at")
        updated_at = flashcard_doc.get("updated_at")
        
        created_at_str = created_at.isoformat() if isinstance(created_at, datetime) else datetime.now(timezone.utc).isoformat()
        updated_at_str = updated_at.isoformat() if isinstance(updated_at, datetime) else datetime.now(timezone.utc).isoformat()
        
        flashcards.append(FlashCardResponse(
            id=str(flashcard_doc["_id"]),
            notebookId=str(flashcard_notebook_id) if flashcard_notebook_id else "",
            notebookTitle=notebook_title,
            question=flashcard_doc.get("question", ""),
            answer=None,  # Hide answer in list view
            explanation=None,  # Hide explanation in list view
            timesAnswered=flashcard_doc.get("times_answered", 0),
            timesAnsweredCorrectly=flashcard_doc.get("times_answered_correctly", 0),
            createdAt=created_at_str,
            updatedAt=updated_at_str
        ))
    
    return flashcards


@router.post("/{flashcard_id}/submit", response_model=FlashCardSubmissionResponse)
async def submit_flashcard_answer(
    flashcard_id: str,
    request: FlashCardSubmissionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit answer for a flashcard and get feedback."""
    user_id = str(current_user["_id"])
    
    # Get flashcard and verify ownership
    flashcard = await get_flashcard_or_404(flashcard_id, user_id)
    
    correct_answer = flashcard.get("answer", "")
    stored_explanation = flashcard.get("explanation", "")
    
    if not correct_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Flashcard does not have a correct answer"
        )
    
    # Compare user's answer with correct answer using LLM
    comparison_result = compare_answers(request.answer, correct_answer)
    
    is_correct = comparison_result.get("isCorrect", False)
    ai_explanation = comparison_result.get("explanation", "")
    
    # Update statistics: increment times_answered and times_answered_correctly if correct
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    flashcard_object_id = validate_object_id(flashcard_id)
    
    update_data = {
        "$inc": {"times_answered": 1},
        "$set": {"updated_at": datetime.now(timezone.utc)}
    }
    
    if is_correct:
        update_data["$inc"]["times_answered_correctly"] = 1
    
    await database.flashcards.update_one(
        {"_id": flashcard_object_id},
        update_data
    )
    
    # Combine stored explanation with AI feedback
    if stored_explanation:
        explanation = f"{ai_explanation}\n\n{stored_explanation}"
    else:
        explanation = ai_explanation
    
    return FlashCardSubmissionResponse(
        isCorrect=is_correct,
        correctAnswer=correct_answer,
        explanation=explanation
    )


@router.delete("/{flashcard_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_flashcard(
    flashcard_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a flashcard."""
    database = await get_database()
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    user_id = str(current_user["_id"])
    
    # Get flashcard and verify ownership
    flashcard = await get_flashcard_or_404(flashcard_id, user_id)
    
    # Delete the flashcard
    flashcard_object_id = validate_object_id(flashcard_id)
    result = await database.flashcards.delete_one({"_id": flashcard_object_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete flashcard"
        )
    
    return None  # 204 No Content

