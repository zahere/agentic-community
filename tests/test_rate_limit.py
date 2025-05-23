"""
Tests for rate limiting functionality.
"""

import asyncio
import time
from unittest.mock import Mock, MagicMock

import pytest
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient

from agentic_community.api.rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    rate_limit
)


class TestRateLimiter:
    """Test the RateLimiter class."""
    
    def test_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_hour=500,
            burst_size=5
        )
        
        assert limiter.requests_per_minute == 30
        assert limiter.requests_per_hour == 500
        assert limiter.burst_size == 5
        
    def test_get_client_id(self):
        """Test client ID extraction from request."""
        limiter = RateLimiter()
        
        # Mock request with IP
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock(host="192.168.1.1")
        
        client_id = limiter._get_client_id(request)
        assert client_id == "192.168.1.1:"
        
        # With API key
        request.headers = {"X-API-Key": "test-key"}
        client_id = limiter._get_client_id(request)
        assert client_id == "192.168.1.1:test-key"
        
        # With forwarded IP
        request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        client_id = limiter._get_client_id(request)
        assert client_id == "10.0.0.1:test-key"
        
    def test_check_bucket_under_limit(self):
        """Test bucket checking when under limit."""
        limiter = RateLimiter()
        bucket = {}
        client_id = "test-client"
        
        # First request
        allowed, remaining = limiter._check_bucket(
            bucket, client_id, 10, 60.0
        )
        assert allowed is True
        assert remaining == 9
        
        # Second request
        allowed, remaining = limiter._check_bucket(
            bucket, client_id, 10, 60.0
        )
        assert allowed is True
        assert remaining == 8
        
    def test_check_bucket_at_limit(self):
        """Test bucket checking when at limit."""
        limiter = RateLimiter()
        bucket = {"test-client": (time.time(), 10)}
        client_id = "test-client"
        
        allowed, remaining = limiter._check_bucket(
            bucket, client_id, 10, 60.0
        )
        assert allowed is False
        assert remaining == 0
        
    def test_check_bucket_reset(self):
        """Test bucket reset after time window."""
        limiter = RateLimiter()
        # Set bucket with old timestamp
        old_time = time.time() - 70  # 70 seconds ago
        bucket = {"test-client": (old_time, 10)}
        client_id = "test-client"
        
        allowed, remaining = limiter._check_bucket(
            bucket, client_id, 10, 60.0
        )
        assert allowed is True
        assert remaining == 9
        
    def test_check_rate_limit(self):
        """Test complete rate limit check."""
        limiter = RateLimiter(
            requests_per_minute=2,
            requests_per_hour=10
        )
        
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock(host="192.168.1.1")
        
        # First request - should pass
        allowed, info = limiter.check_rate_limit(request)
        assert allowed is True
        assert info["minute_remaining"] == 1
        assert info["hour_remaining"] == 9
        
        # Second request - should pass
        allowed, info = limiter.check_rate_limit(request)
        assert allowed is True
        assert info["minute_remaining"] == 0
        assert info["hour_remaining"] == 8
        
        # Third request - should fail (minute limit)
        allowed, info = limiter.check_rate_limit(request)
        assert allowed is False
        assert info["minute_remaining"] == 0


class TestRateLimitMiddleware:
    """Test the RateLimitMiddleware class."""
    
    @pytest.mark.asyncio
    async def test_middleware_allows_request(self):
        """Test middleware allows requests under limit."""
        app = MagicMock()
        limiter = RateLimiter(requests_per_minute=10)
        middleware = RateLimitMiddleware(app, limiter)
        
        # Mock ASGI scope
        scope = {
            "type": "http",
            "path": "/test",
            "headers": [],
            "method": "GET",
            "scheme": "http",
            "server": ("localhost", 8000),
            "client": ("127.0.0.1", 12345),
            "query_string": b""
        }
        
        receive = MagicMock()
        send = MagicMock()
        
        await middleware(scope, receive, send)
        
        # App should be called
        app.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_middleware_blocks_exceeded_requests(self):
        """Test middleware blocks requests over limit."""
        app = MagicMock()
        limiter = RateLimiter(requests_per_minute=0)  # No requests allowed
        middleware = RateLimitMiddleware(app, limiter)
        
        scope = {
            "type": "http",
            "path": "/test",
            "headers": [],
            "method": "GET",
            "scheme": "http",
            "server": ("localhost", 8000),
            "client": ("127.0.0.1", 12345),
            "query_string": b""
        }
        
        receive = MagicMock()
        send = MagicMock()
        
        await middleware(scope, receive, send)
        
        # App should not be called
        app.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_middleware_skips_health_check(self):
        """Test middleware skips rate limiting for health checks."""
        app = MagicMock()
        limiter = RateLimiter(requests_per_minute=0)  # No requests allowed
        middleware = RateLimitMiddleware(app, limiter)
        
        scope = {
            "type": "http",
            "path": "/health",
            "headers": [],
            "method": "GET",
            "scheme": "http",
            "server": ("localhost", 8000),
            "client": ("127.0.0.1", 12345),
            "query_string": b""
        }
        
        receive = MagicMock()
        send = MagicMock()
        
        await middleware(scope, receive, send)
        
        # App should be called despite rate limit
        app.assert_called_once()


class TestRateLimitDecorator:
    """Test the rate_limit decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_allows_request(self):
        """Test decorator allows requests under limit."""
        @rate_limit(requests_per_minute=10)
        async def test_endpoint(request: Request):
            return {"status": "ok"}
            
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock(host="192.168.1.1")
        
        result = await test_endpoint(request)
        assert result == {"status": "ok"}
        
    @pytest.mark.asyncio
    async def test_decorator_blocks_exceeded_requests(self):
        """Test decorator blocks requests over limit."""
        @rate_limit(requests_per_minute=1)
        async def test_endpoint(request: Request):
            return {"status": "ok"}
            
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock(host="192.168.1.1")
        
        # First request should pass
        result = await test_endpoint(request)
        assert result == {"status": "ok"}
        
        # Second request should fail
        with pytest.raises(HTTPException) as exc_info:
            await test_endpoint(request)
            
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail
        
    @pytest.mark.asyncio
    async def test_decorator_adds_headers(self):
        """Test decorator adds rate limit headers to response."""
        @rate_limit(requests_per_minute=10)
        async def test_endpoint(request: Request):
            response = Mock()
            response.headers = {}
            return response
            
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock(host="192.168.1.1")
        
        response = await test_endpoint(request)
        
        assert "X-RateLimit-Minute-Remaining" in response.headers
        assert "X-RateLimit-Hour-Remaining" in response.headers


class TestDistributedRateLimiter:
    """Test the DistributedRateLimiter class."""
    
    def test_fallback_to_local(self):
        """Test distributed limiter falls back to local in community edition."""
        limiter = DistributedRateLimiter()
        
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock(host="192.168.1.1")
        
        # Should work like regular rate limiter
        allowed, info = limiter.check_rate_limit(request)
        assert allowed is True
        assert "minute_remaining" in info
        assert "hour_remaining" in info
