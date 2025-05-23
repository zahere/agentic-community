"""
Authentication and authorization for the Agentic API.

This module provides JWT-based authentication and API key management.
"""

import os
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps

from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from agentic_community.core.exceptions import ValidationError, APIError

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("AGENTIC_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    api_key: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)


class UserInDB(User):
    """User in database with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None


class AuthManager:
    """
    Manages authentication and authorization.
    
    In production, this would use a proper database.
    """
    
    def __init__(self):
        # In-memory storage for demo (use database in production)
        self.users: Dict[str, UserInDB] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> username
        self.revoked_tokens: set = set()
        
        # Create default admin user
        self.create_user(
            "admin",
            "admin@agentic.ai",
            "Admin User",
            "admin123",  # Change in production!
            scopes=["admin", "read", "write"]
        )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def create_user(
        self,
        username: str,
        email: str,
        full_name: str,
        password: str,
        scopes: List[str] = None
    ) -> User:
        """Create a new user."""
        if username in self.users:
            raise ValidationError("username", username, "User already exists")
        
        hashed_password = self.get_password_hash(password)
        api_key = secrets.token_urlsafe(32)
        
        user = UserInDB(
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
            api_key=api_key,
            scopes=scopes or ["read"]
        )
        
        self.users[username] = user
        self.api_keys[api_key] = username
        
        return User(**user.dict(exclude={"hashed_password"}))
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user with username and password."""
        user = self.users.get(username)
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self.users.get(username)
    
    def get_user_by_api_key(self, api_key: str) -> Optional[UserInDB]:
        """Get user by API key."""
        username = self.api_keys.get(api_key)
        if username:
            return self.users.get(username)
        return None
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "type": "access"
        })
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def create_refresh_token(self, data: dict) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "type": "refresh"
        })
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def decode_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            scopes: List[str] = payload.get("scopes", [])
            
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Check if token is revoked
            if token in self.revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return TokenData(username=username, scopes=scopes)
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str):
        """Revoke a token."""
        self.revoked_tokens.add(token)


# Global auth manager instance
auth_manager = AuthManager()


# Dependency functions

async def get_current_user_jwt(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> User:
    """Get current user from JWT token."""
    token = credentials.credentials
    token_data = auth_manager.decode_token(token)
    
    user = auth_manager.get_user(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is disabled"
        )
    
    return User(**user.dict(exclude={"hashed_password"}))


async def get_current_user_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """Get current user from API key."""
    if not api_key:
        return None
    
    user = auth_manager.get_user_by_api_key(api_key)
    if user is None:
        return None
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is disabled"
        )
    
    return User(**user.dict(exclude={"hashed_password"}))


async def get_current_user(
    jwt_user: Optional[User] = Depends(get_current_user_jwt),
    api_key_user: Optional[User] = Depends(get_current_user_api_key)
) -> User:
    """Get current user from either JWT or API key."""
    # Try JWT first
    try:
        if jwt_user:
            return jwt_user
    except HTTPException:
        pass
    
    # Try API key
    if api_key_user:
        return api_key_user
    
    # No valid authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )


def require_scopes(*required_scopes: str):
    """Decorator to require specific scopes."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_user), **kwargs):
            # Check if user has required scopes
            if "admin" in current_user.scopes:
                # Admin has all permissions
                return await func(*args, current_user=current_user, **kwargs)
            
            for scope in required_scopes:
                if scope not in current_user.scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required scope: {scope}"
                    )
            
            return await func(*args, current_user=current_user, **kwargs)
        
        return wrapper
    return decorator


# Rate limiting

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit."""
        now = datetime.utcnow()
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if (now - req_time).total_seconds() < self.window_seconds
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add this request
        self.requests[identifier].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


async def check_rate_limit(current_user: User = Depends(get_current_user)):
    """Check rate limit for current user."""
    if not rate_limiter.check_rate_limit(current_user.username):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return current_user


# Login endpoint models

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class CreateUserRequest(BaseModel):
    """Create user request model."""
    username: str
    email: str
    full_name: str
    password: str
    scopes: List[str] = Field(default_factory=lambda: ["read"])
