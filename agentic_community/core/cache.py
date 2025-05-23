"""
Caching layer for tool results and LLM responses.

Provides a flexible caching system with support for different backends
and configurable TTL (Time To Live) settings.
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass
        
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in the cache with optional TTL in seconds."""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear all values from the cache."""
        pass
        
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache implementation using dictionaries."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = False  # Simple lock for thread safety
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        
        # Check if expired
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            del self._cache[key]
            return None
            
        # Update access time
        entry["last_accessed"] = time.time()
        entry["access_count"] = entry.get("access_count", 0) + 1
        
        return entry["value"]
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in the cache with optional TTL in seconds."""
        entry = {
            "value": value,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 0
        }
        
        if ttl:
            entry["expires_at"] = time.time() + ttl
            
        self._cache[key] = entry
        
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
        
    def clear(self):
        """Clear all values from the cache."""
        self._cache.clear()
        
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if key not in self._cache:
            return False
            
        entry = self._cache[key]
        
        # Check if expired
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            del self._cache[key]
            return False
            
        return True
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        total_accesses = 0
        expired_count = 0
        
        for key, entry in list(self._cache.items()):
            if entry.get("expires_at") and time.time() > entry["expires_at"]:
                del self._cache[key]
                expired_count += 1
            else:
                total_size += 1
                total_accesses += entry.get("access_count", 0)
                
        return {
            "total_entries": total_size,
            "total_accesses": total_accesses,
            "expired_removed": expired_count,
            "memory_usage_estimate": sum(
                len(str(k)) + len(str(v)) for k, v in self._cache.items()
            )
        }


class CacheManager:
    """Manages caching for the Agentic framework."""
    
    def __init__(self, backend: Optional[CacheBackend] = None, default_ttl: int = 3600):
        """
        Initialize the cache manager.
        
        Args:
            backend: Cache backend to use (defaults to InMemoryCache)
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.backend = backend or InMemoryCache()
        self.default_ttl = default_ttl
        self.enabled = True
        
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from prefix and arguments."""
        # Create a unique key from arguments
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": kwargs
        }
        
        # Use JSON serialization for consistent hashing
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Create a hash for the key
        hash_obj = hashlib.md5(key_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if not self.enabled:
            return None
            
        try:
            value = self.backend.get(key)
            if value is not None:
                logger.debug(f"Cache hit for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in the cache."""
        if not self.enabled:
            return
            
        try:
            ttl = ttl or self.default_ttl
            self.backend.set(key, value, ttl)
            logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        if not self.enabled:
            return False
            
        try:
            return self.backend.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    def clear(self):
        """Clear all values from the cache."""
        try:
            self.backend.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if not self.enabled:
            return False
            
        try:
            return self.backend.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False


# Global cache instance
_cache_manager = CacheManager()


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager


def cache_result(prefix: str, ttl: Optional[int] = None, 
                key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key
    
    Example:
        @cache_result("search", ttl=3600)
        def search_web(query):
            # Expensive search operation
            return results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache.generate_key(prefix, *args, **kwargs)
                
            # Try to get from cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
                
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(key, result, ttl)
            
            return result
            
        return wrapper
    return decorator


def invalidate_pattern(pattern: str):
    """
    Invalidate cache entries matching a pattern.
    
    This is a simple implementation that works with in-memory cache.
    For production use with Redis, use SCAN command.
    """
    cache = get_cache()
    if hasattr(cache.backend, '_cache'):
        # In-memory cache
        keys_to_delete = [
            k for k in cache.backend._cache.keys() 
            if pattern in k
        ]
        for key in keys_to_delete:
            cache.delete(key)
            
        logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching '{pattern}'")


class CachedTool:
    """Mixin for tools that support caching."""
    
    def __init__(self, *args, cache_ttl: int = 3600, **kwargs):
        """Initialize with caching support."""
        super().__init__(*args, **kwargs)
        self.cache_ttl = cache_ttl
        self._cache = get_cache()
        
    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Generate a cache key for a tool method."""
        return self._cache.generate_key(
            f"tool:{self.__class__.__name__}:{method_name}",
            *args, **kwargs
        )
        
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return self._cache.get(key)
        
    def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache."""
        ttl = ttl or self.cache_ttl
        self._cache.set(key, value, ttl)


# Example usage in tools
def cached_tool_method(ttl: Optional[int] = None):
    """Decorator for caching tool method results."""
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_cache'):
                # Tool doesn't support caching
                return method(self, *args, **kwargs)
                
            # Generate cache key
            key = self._get_cache_key(method.__name__, *args, **kwargs)
            
            # Try cache
            result = self._cache_get(key)
            if result is not None:
                return result
                
            # Execute method
            result = method(self, *args, **kwargs)
            
            # Cache result
            self._cache_set(key, result, ttl)
            
            return result
            
        return wrapper
    return decorator
