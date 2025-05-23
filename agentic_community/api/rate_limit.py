"""
Rate limiting implementation for API endpoints.

This module provides rate limiting functionality to prevent API abuse
and ensure fair usage across all users.
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Tuple, Optional, Callable

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from agentic_community.core.exceptions import RateLimitError


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    
    Uses a token bucket algorithm to limit requests per time window.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests allowed per minute
            requests_per_hour: Max requests allowed per hour
            burst_size: Max burst requests allowed
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        
        # Storage for rate limit data
        self._minute_buckets: Dict[str, Tuple[float, int]] = defaultdict(
            lambda: (time.time(), 0)
        )
        self._hour_buckets: Dict[str, Tuple[float, int]] = defaultdict(
            lambda: (time.time(), 0)
        )
        
    def _get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier from request.
        
        Uses IP address and optional API key for identification.
        """
        # Try to get real IP from proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0]
        else:
            client_ip = request.client.host if request.client else "unknown"
            
        # Include API key if present
        api_key = request.headers.get("X-API-Key", "")
        
        return f"{client_ip}:{api_key}"
        
    def _check_bucket(
        self,
        bucket: Dict[str, Tuple[float, int]],
        client_id: str,
        max_requests: int,
        time_window: float
    ) -> Tuple[bool, int]:
        """
        Check if request is allowed based on bucket state.
        
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        current_time = time.time()
        last_reset, request_count = bucket[client_id]
        
        # Reset bucket if time window has passed
        if current_time - last_reset > time_window:
            bucket[client_id] = (current_time, 1)
            return True, max_requests - 1
            
        # Check if under limit
        if request_count < max_requests:
            bucket[client_id] = (last_reset, request_count + 1)
            return True, max_requests - request_count - 1
            
        return False, 0
        
    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        client_id = self._get_client_id(request)
        
        # Check minute limit
        minute_allowed, minute_remaining = self._check_bucket(
            self._minute_buckets,
            client_id,
            self.requests_per_minute,
            60.0
        )
        
        # Check hour limit
        hour_allowed, hour_remaining = self._check_bucket(
            self._hour_buckets,
            client_id,
            self.requests_per_hour,
            3600.0
        )
        
        # Both limits must pass
        allowed = minute_allowed and hour_allowed
        
        rate_limit_info = {
            "minute_remaining": minute_remaining,
            "hour_remaining": hour_remaining,
            "minute_limit": self.requests_per_minute,
            "hour_limit": self.requests_per_hour
        }
        
        return allowed, rate_limit_info


class RateLimitMiddleware:
    """
    ASGI middleware for rate limiting.
    """
    
    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Create request object from ASGI scope
            request = Request(scope, receive)
            
            # Skip rate limiting for health checks
            if request.url.path in ["/health", "/metrics"]:
                await self.app(scope, receive, send)
                return
                
            # Check rate limit
            allowed, rate_limit_info = self.rate_limiter.check_rate_limit(request)
            
            if not allowed:
                # Send rate limit response
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "rate_limits": rate_limit_info
                    },
                    headers={
                        "X-RateLimit-Minute-Remaining": str(rate_limit_info["minute_remaining"]),
                        "X-RateLimit-Hour-Remaining": str(rate_limit_info["hour_remaining"]),
                        "X-RateLimit-Minute-Limit": str(rate_limit_info["minute_limit"]),
                        "X-RateLimit-Hour-Limit": str(rate_limit_info["hour_limit"]),
                        "Retry-After": "60"
                    }
                )
                
                await response(scope, receive, send)
                return
                
        await self.app(scope, receive, send)


def rate_limit(
    requests_per_minute: Optional[int] = None,
    requests_per_hour: Optional[int] = None
) -> Callable:
    """
    Decorator for rate limiting specific endpoints.
    
    Args:
        requests_per_minute: Override default minute limit
        requests_per_hour: Override default hour limit
    """
    def decorator(func: Callable) -> Callable:
        # Create endpoint-specific rate limiter
        endpoint_limiter = RateLimiter(
            requests_per_minute=requests_per_minute or 60,
            requests_per_hour=requests_per_hour or 1000
        )
        
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            allowed, rate_limit_info = endpoint_limiter.check_rate_limit(request)
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Minute-Remaining": str(rate_limit_info["minute_remaining"]),
                        "X-RateLimit-Hour-Remaining": str(rate_limit_info["hour_remaining"]),
                        "Retry-After": "60"
                    }
                )
                
            # Add rate limit info to response headers
            response = await func(request, *args, **kwargs)
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Minute-Remaining"] = str(
                    rate_limit_info["minute_remaining"]
                )
                response.headers["X-RateLimit-Hour-Remaining"] = str(
                    rate_limit_info["hour_remaining"]
                )
                
            return response
            
        return wrapper
    return decorator


class DistributedRateLimiter(RateLimiter):
    """
    Distributed rate limiter using Redis for multi-instance deployments.
    
    This is a placeholder for enterprise features.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In community edition, falls back to local rate limiting
        
    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, int]]:
        """
        Check rate limit using distributed storage.
        
        Community edition uses local storage only.
        """
        return super().check_rate_limit(request)
