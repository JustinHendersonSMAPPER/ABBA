"""Tests for the caching layer."""

import time
import unittest
from unittest.mock import MagicMock

from abba.api.cache import CachedSearchAPI, QueryCache, cached
from abba.config import ABBAConfig


class TestQueryCache(unittest.TestCase):
    """Test QueryCache functionality."""

    def setUp(self):
        """Set up test environment."""
        self.config = MagicMock(spec=ABBAConfig)
        self.config.use_cache = True
        self.config.cache_ttl = 1  # 1 second TTL for testing
        self.cache = QueryCache(self.config)

    def test_cache_disabled(self):
        """Test cache when disabled."""
        self.config.use_cache = False
        cache = QueryCache(self.config)

        cache.set("key", "value")
        self.assertIsNone(cache.get("key"))

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        # Test set and get
        self.cache.set("test_key", "test_value")
        self.assertEqual(self.cache.get("test_key"), "test_value")

        # Test non-existent key
        self.assertIsNone(self.cache.get("non_existent"))

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        self.cache.set("expire_key", "expire_value")
        self.assertEqual(self.cache.get("expire_key"), "expire_value")

        # Wait for TTL to expire
        time.sleep(1.1)
        self.assertIsNone(self.cache.get("expire_key"))

    def test_cache_key_generation(self):
        """Test cache key generation."""
        # Test with different argument types
        key1 = self.cache._make_key("func", (1, "test"), {"arg": True})
        key2 = self.cache._make_key("func", (1, "test"), {"arg": True})
        key3 = self.cache._make_key("func", (2, "test"), {"arg": True})

        # Same args should produce same key
        self.assertEqual(key1, key2)
        # Different args should produce different key
        self.assertNotEqual(key1, key3)

        # Test that keys are actually different with different args
        key4 = self.cache._make_key("func", (1, "test"), {})
        key5 = self.cache._make_key("func", (1, "different"), {})
        self.assertNotEqual(key4, key5)

    def test_clear_cache(self):
        """Test clearing cache."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.clear()

        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))

    def test_remove_expired(self):
        """Test removing expired entries."""
        # Add entries with different timestamps
        self.cache.set("keep", "value")

        # Manually add an expired entry
        self.cache._cache["expire"] = ("old_value", time.time() - 10)

        removed = self.cache.remove_expired()
        self.assertEqual(removed, 1)
        self.assertIsNone(self.cache.get("expire"))
        self.assertEqual(self.cache.get("keep"), "value")


class TestCachedDecorator(unittest.TestCase):
    """Test cached decorator functionality."""

    def setUp(self):
        """Set up test environment."""
        self.config = MagicMock(spec=ABBAConfig)
        self.config.use_cache = True
        self.config.cache_ttl = 60
        self.cache = QueryCache(self.config)

    def test_decorator_caching(self):
        """Test that decorator caches function results."""
        call_count = 0

        @cached(self.cache)
        def test_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call
        result1 = test_func(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)

        # Second call with same args should use cache
        result2 = test_func(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # No additional call

        # Different args should call function
        result3 = test_func(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)

    def test_decorator_methods(self):
        """Test decorator added methods."""

        @cached(self.cache)
        def test_func(x):
            return x * 2

        # Test cache_info
        info = test_func.cache_info()
        self.assertTrue(info["enabled"])
        self.assertEqual(info["ttl"], 60)

        # Test clear_cache
        test_func(5)
        # Get fresh info after calling function
        info2 = test_func.cache_info()
        self.assertGreater(info2["size"], 0)
        test_func.clear_cache()


class TestCachedAPIs(unittest.TestCase):
    """Test cached API wrappers."""

    def setUp(self):
        """Set up test environment."""
        self.config = MagicMock(spec=ABBAConfig)
        self.config.use_cache = True
        self.config.cache_ttl = 60
        self.cache = QueryCache(self.config)

    def test_cached_search_api(self):
        """Test CachedSearchAPI wrapper."""
        # Mock the underlying API
        mock_api = MagicMock()
        mock_api.get_verse.return_value = "Test verse"
        mock_api.search_strongs.return_value = ["Result 1", "Result 2"]

        # Create a regular function for search_verses (not a MagicMock)
        def mock_search_verses(query):
            return ["Search result"]

        mock_api.search_verses = mock_search_verses

        # Create cached wrapper
        cached_api = CachedSearchAPI(mock_api, self.cache)

        # Test that methods are wrapped
        self.assertTrue(hasattr(cached_api.get_verse, "cache_info"))
        self.assertTrue(hasattr(cached_api.search_strongs, "cache_info"))

        # Test that search_verses is not cached
        self.assertFalse(hasattr(cached_api.search_verses, "cache_info"))

    def test_cached_methods_work(self):
        """Test that cached methods still work correctly."""
        mock_api = MagicMock()
        mock_api.get_verse.return_value = "Genesis 1:1"

        cached_api = CachedSearchAPI(mock_api, self.cache)

        # First call
        result1 = cached_api.get_verse("KJV", 1, 1, 1)
        self.assertEqual(result1, "Genesis 1:1")
        self.assertEqual(mock_api.get_verse.call_count, 1)

        # Second call should use cache
        result2 = cached_api.get_verse("KJV", 1, 1, 1)
        self.assertEqual(result2, "Genesis 1:1")
        self.assertEqual(mock_api.get_verse.call_count, 1)  # No additional call


if __name__ == "__main__":
    unittest.main()
