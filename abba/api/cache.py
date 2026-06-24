"""Caching layer for ABBA API queries."""

import hashlib
import json
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from ..config import ABBAConfig


class QueryCache:
    """Simple in-memory cache for query results."""

    def __init__(self, config: ABBAConfig):
        """Initialize cache.

        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = config.use_cache
        self.ttl = config.cache_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments.

        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create a stable string representation
        key_parts = [func_name]

        # Add all args - don't skip any
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            elif isinstance(arg, dict):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                # For complex objects, use their string representation
                key_parts.append(str(arg))

        # Add sorted kwargs
        if kwargs:
            key_parts.append(json.dumps(kwargs, sort_keys=True))

        # Create hash of the key parts
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            # Remove expired entry
            del self._cache[key]

        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if self.enabled:
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def remove_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0

        current_time = time.time()
        expired_keys = [key for key, (_, timestamp) in self._cache.items() if current_time - timestamp >= self.ttl]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)


def cached(cache_instance: Optional[QueryCache] = None):
    """Decorator to cache function results.

    Args:
        cache_instance: QueryCache instance to use

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If no cache instance or caching disabled, just call function
            if not cache_instance or not cache_instance.enabled:
                return func(*args, **kwargs)

            # Generate cache key
            func_name = getattr(func, "__name__", str(func))
            cache_key = cache_instance._make_key(func_name, args, kwargs)

            # Check cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)

            return result

        # Add cache management methods to wrapper
        wrapper.clear_cache = lambda: cache_instance.clear() if cache_instance else None  # type: ignore[attr-defined]
        wrapper.cache_info = lambda: {  # type: ignore[attr-defined]
            "enabled": cache_instance.enabled if cache_instance else False,
            "size": len(cache_instance._cache) if cache_instance else 0,
            "ttl": cache_instance.ttl if cache_instance else 0,
        }

        return wrapper

    return decorator


# Example usage patterns for common queries
class CachedSearchAPI:
    """Example of how to apply caching to SearchAPI methods."""

    def __init__(self, search_api, cache: QueryCache):
        """Initialize cached search API.

        Args:
            search_api: Original SearchAPI instance
            cache: QueryCache instance
        """
        self._api = search_api
        self._cache = cache

        # Define which methods should be cached
        cached_methods = ["get_verse", "search_strongs", "get_word_analysis"]
        uncached_methods = ["search_verses"]

        # Wrap methods that should be cached
        for method_name in cached_methods:
            if hasattr(self._api, method_name):
                original_method = getattr(self._api, method_name)
                cached_method = cached(cache)(original_method)
                setattr(self, method_name, cached_method)

        # Pass through methods that shouldn't be cached
        for method_name in uncached_methods:
            if hasattr(self._api, method_name):
                setattr(self, method_name, getattr(self._api, method_name))

        # Copy over other methods that might exist
        for attr_name in dir(self._api):
            if (
                not attr_name.startswith("_")
                and not hasattr(self, attr_name)
                and callable(getattr(self._api, attr_name))
            ):
                setattr(self, attr_name, getattr(self._api, attr_name))


class CachedAnalysisAPI:
    """Example of how to apply caching to AnalysisAPI methods."""

    def __init__(self, analysis_api, cache: QueryCache):
        """Initialize cached analysis API.

        Args:
            analysis_api: Original AnalysisAPI instance
            cache: QueryCache instance
        """
        self._api = analysis_api
        self._cache = cache

        # Cache heavy analysis methods
        self.analyze_morphology_patterns = cached(cache)(self._api.analyze_morphology_patterns)
        self.word_frequency_analysis = cached(cache)(self._api.word_frequency_analysis)
        self.find_hapax_legomena = cached(cache)(self._api.find_hapax_legomena)
        self.semantic_domain_analysis = cached(cache)(self._api.semantic_domain_analysis)

        # Methods with dynamic results might have shorter cache or no cache
        self.parallel_passage_detection = self._api.parallel_passage_detection
