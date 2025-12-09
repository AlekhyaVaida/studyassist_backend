"""Simple user models for email/password authentication."""

from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr, Field


class UserSignup(BaseModel):
    """User signup schema."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=1, description="User's full name")


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    name: str
    created_at: str  # ISO format string
    
    @classmethod
    def from_dict(cls, user_dict: dict):
        """Create UserResponse from MongoDB document."""
        created_at = user_dict.get("created_at")
        if isinstance(created_at, datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = datetime.now(timezone.utc).isoformat()
        
        return cls(
            id=str(user_dict["_id"]),
            email=user_dict["email"],
            name=user_dict.get("name", ""),
            created_at=created_at_str
        )
