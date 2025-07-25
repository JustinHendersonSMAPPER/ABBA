"""API modules for ABBA."""

from .analysis import AnalysisAPI, LexicalCluster, MorphologyPattern, WordFrequency
from .cache import CachedAnalysisAPI, CachedSearchAPI, QueryCache, cached
from .search import SearchAPI, VerseResult, WordResult

__all__ = [
    # Search API
    "SearchAPI",
    "VerseResult",
    "WordResult",
    # Analysis API
    "AnalysisAPI",
    "MorphologyPattern",
    "WordFrequency",
    "LexicalCluster",
    # Cache
    "QueryCache",
    "cached",
    "CachedSearchAPI",
    "CachedAnalysisAPI",
]
