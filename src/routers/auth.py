"""Simple authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from src.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_user_by_email,
    get_current_user,
)
from src.database import get_database
from src.models.user import UserSignup, UserLogin, UserResponse
from datetime import datetime, timezone

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup):
    """Register a new user."""
    try:
        database = await get_database()
        if database is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database connection not available"
            )
        
        # Check if user already exists
        existing_user = await get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = hash_password(user_data.password)
        user_dict = {
            "email": user_data.email,
            "name": user_data.name,
            "hashed_password": hashed_password,
            "created_at": datetime.now(timezone.utc),
        }
        
        result = await database.users.insert_one(user_dict)
        created_user = await database.users.find_one({"_id": result.inserted_id})
        
        if not created_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        return UserResponse.from_dict(created_user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@router.post("/login")
async def login(user_data: UserLogin):
    """Login and get access token."""
    user = await get_user_by_email(user_data.email)
    
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(user_data.email)
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.from_dict(current_user)
