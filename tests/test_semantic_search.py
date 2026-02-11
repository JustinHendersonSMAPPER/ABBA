"""Tests for semantic search API: similar verses, related words, hybrid search."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from abba.api.semantic_search import (
    HybridSearchResult,
    RelatedWordResult,
    SemanticSearchAPI,
    SemanticVerseResult,
)


@pytest.fixture
def mock_db():
    """Create a mock SQLiteManager."""
    db = MagicMock()
    db.search_verses.return_value = []
    db.get_verse.return_value = None
    return db


@pytest.fixture
def mock_chroma():
    """Create a mock ChromaManager."""
    chroma = MagicMock()
    chroma.get_collection.return_value = None
    return chroma


@pytest.fixture
def mock_models():
    """Create a mock EmbeddingModelManager."""
    import numpy as np

    models = MagicMock()
    # Return a fake 768-dim embedding
    models.encode_single.return_value = np.random.randn(768).astype("float32")
    return models


@pytest.fixture
def search_api(mock_db, mock_chroma, mock_models):
    """Create a SemanticSearchAPI with mocked dependencies."""
    return SemanticSearchAPI(mock_db, mock_chroma, mock_models)


# ------------------------------------------------------------------ #
#  search_similar_verses                                               #
# ------------------------------------------------------------------ #


class TestSearchSimilarVerses:
    """Tests for search_similar_verses."""

    def test_returns_results_sorted_by_similarity(self, search_api, mock_chroma):
        """Should return results sorted by descending similarity."""
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["001:001:001", "043:001:001", "019:023:001"]],
            "distances": [[0.3, 0.5, 0.1]],
            "metadatas": [
                [
                    {"book_id": 1, "chapter": 1, "verse": 1, "testament": "old", "language": "hebrew"},
                    {"book_id": 43, "chapter": 1, "verse": 1, "testament": "new", "language": "greek"},
                    {"book_id": 19, "chapter": 23, "verse": 1, "testament": "old", "language": "hebrew"},
                ]
            ],
        }
        mock_chroma.get_collection.return_value = collection

        results = search_api.search_similar_verses("In the beginning God created")

        assert len(results) == 3
        # Most similar first (lowest distance = highest similarity)
        assert results[0].book_id == 19  # distance 0.1 -> similarity 0.9
        assert results[0].similarity > results[1].similarity

    def test_filters_by_similarity_threshold(self, search_api, mock_chroma):
        """Should exclude results below the similarity threshold."""
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["001:001:001", "043:001:001"]],
            "distances": [[0.2, 0.9]],  # 0.9 distance -> 0.1 similarity
            "metadatas": [
                [
                    {"book_id": 1, "chapter": 1, "verse": 1, "testament": "old", "language": "hebrew"},
                    {"book_id": 43, "chapter": 1, "verse": 1, "testament": "new", "language": "greek"},
                ]
            ],
        }
        mock_chroma.get_collection.return_value = collection

        results = search_api.search_similar_verses("creation", similarity_threshold=0.5)

        assert len(results) == 1
        assert results[0].book_id == 1

    def test_returns_empty_when_no_collection(self, search_api, mock_chroma):
        """Should return empty list when collection doesn't exist."""
        mock_chroma.get_collection.return_value = None

        results = search_api.search_similar_verses("test query")
        assert results == []

    def test_returns_empty_when_no_models(self, mock_db, mock_chroma):
        """Should return empty list when no model manager is configured."""
        api = SemanticSearchAPI(mock_db, mock_chroma, model_manager=None)
        results = api.search_similar_verses("test query")
        assert results == []

    def test_testament_filter(self, search_api, mock_chroma):
        """Should pass testament filter to ChromaDB."""
        collection = MagicMock()
        collection.query.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        mock_chroma.get_collection.return_value = collection

        search_api.search_similar_verses("love", testament_filter="new")

        call_args = collection.query.call_args
        assert call_args.kwargs.get("where") == {"testament": "new"}

    def test_book_filter(self, search_api, mock_chroma):
        """Should pass book filter to ChromaDB."""
        collection = MagicMock()
        collection.query.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        mock_chroma.get_collection.return_value = collection

        search_api.search_similar_verses("love", book_filter=19)

        call_args = collection.query.call_args
        assert call_args.kwargs.get("where") == {"book_id": 19}

    def test_combined_filters(self, search_api, mock_chroma):
        """Should combine testament and book filters with $and."""
        collection = MagicMock()
        collection.query.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        mock_chroma.get_collection.return_value = collection

        search_api.search_similar_verses("love", testament_filter="old", book_filter=19)

        call_args = collection.query.call_args
        where = call_args.kwargs.get("where")
        assert "$and" in where


# ------------------------------------------------------------------ #
#  search_related_words                                                #
# ------------------------------------------------------------------ #


class TestSearchRelatedWords:
    """Tests for search_related_words."""

    def test_returns_related_words(self, search_api, mock_chroma):
        """Should return words sorted by similarity."""
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["H0430:HNcmpa", "G2316:GNnms"]],
            "distances": [[0.2, 0.4]],
            "metadatas": [
                [
                    {
                        "strongs": "H0430",
                        "word": "אֱלֹהִים",
                        "gloss": "God",
                        "language": "hebrew",
                        "transliteration": "elohim",
                        "part_of_speech": "noun",
                    },
                    {
                        "strongs": "G2316",
                        "word": "θεός",
                        "gloss": "God",
                        "language": "greek",
                        "transliteration": "theos",
                        "part_of_speech": "noun",
                    },
                ]
            ],
        }
        mock_chroma.get_collection.return_value = collection

        results = search_api.search_related_words("God, deity, divine")

        assert len(results) == 2
        assert results[0].strongs_number == "H0430"
        assert results[0].similarity > results[1].similarity

    def test_language_filter(self, search_api, mock_chroma):
        """Should pass language filter to ChromaDB."""
        collection = MagicMock()
        collection.query.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        mock_chroma.get_collection.return_value = collection

        search_api.search_related_words("love", language_filter="hebrew")

        call_args = collection.query.call_args
        assert call_args.kwargs.get("where") == {"language": "hebrew"}

    def test_returns_empty_when_no_collection(self, search_api, mock_chroma):
        """Should return empty when words collection doesn't exist."""
        mock_chroma.get_collection.return_value = None
        results = search_api.search_related_words("test")
        assert results == []


# ------------------------------------------------------------------ #
#  hybrid_search                                                       #
# ------------------------------------------------------------------ #


class TestHybridSearch:
    """Tests for hybrid_search."""

    def test_combines_exact_and_semantic_results(self, search_api, mock_db, mock_chroma):
        """Should combine FTS and semantic results."""
        # FTS results
        mock_db.search_verses.return_value = [
            {"book_id": 1, "chapter": 1, "verse": 1, "text": "In the beginning God created", "book_name": "Genesis"},
        ]

        # Semantic results (mock the collection)
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["001:001:001", "043:001:001"]],
            "distances": [[0.2, 0.3]],
            "metadatas": [
                [
                    {"book_id": 1, "chapter": 1, "verse": 1, "testament": "old", "language": "hebrew"},
                    {"book_id": 43, "chapter": 1, "verse": 1, "testament": "new", "language": "greek"},
                ]
            ],
        }
        mock_chroma.get_collection.return_value = collection
        mock_db.get_verse.return_value = {"text": "In the beginning was the Word"}

        results = search_api.hybrid_search("In the beginning")

        assert len(results) >= 1
        # Gen 1:1 should be "both" since it appears in FTS and semantic
        gen1 = next((r for r in results if r.book_id == 1 and r.chapter == 1 and r.verse == 1), None)
        assert gen1 is not None
        assert gen1.match_type == "both"

    def test_exact_only_results(self, search_api, mock_db, mock_chroma):
        """Should include FTS-only results."""
        mock_db.search_verses.return_value = [
            {"book_id": 1, "chapter": 1, "verse": 3, "text": "Let there be light", "book_name": "Genesis"},
        ]
        mock_chroma.get_collection.return_value = None  # No semantic search

        results = search_api.hybrid_search("let there be light")

        assert len(results) == 1
        assert results[0].match_type == "exact"

    def test_semantic_only_results(self, search_api, mock_db, mock_chroma):
        """Should include semantic-only results not found in FTS."""
        mock_db.search_verses.return_value = []  # No FTS results

        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["019:023:001"]],
            "distances": [[0.15]],
            "metadatas": [[{"book_id": 19, "chapter": 23, "verse": 1, "testament": "old", "language": "hebrew"}]],
        }
        mock_chroma.get_collection.return_value = collection
        mock_db.get_verse.return_value = {"text": "The Lord is my shepherd"}

        results = search_api.hybrid_search("shepherd caring for sheep")

        assert len(results) == 1
        assert results[0].match_type == "semantic"

    def test_respects_n_results_limit(self, search_api, mock_db, mock_chroma):
        """Should limit output to n_results."""
        mock_db.search_verses.return_value = [
            {"book_id": i, "chapter": 1, "verse": 1, "text": f"text {i}", "book_name": f"Book{i}"} for i in range(1, 20)
        ]
        mock_chroma.get_collection.return_value = None

        results = search_api.hybrid_search("test", n_results=5)
        assert len(results) <= 5


# ------------------------------------------------------------------ #
#  expand_query_with_strongs                                           #
# ------------------------------------------------------------------ #


class TestExpandQueryWithStrongs:
    """Tests for expand_query_with_strongs."""

    def test_finds_related_strongs(self, search_api, mock_chroma):
        """Should return related Strong's numbers."""
        collection = MagicMock()
        collection.get.return_value = {"embeddings": [[0.1] * 768]}
        collection.query.return_value = {
            "ids": [["H0430", "H0410", "H0433", "H7706"]],
            "distances": [[0.0, 0.2, 0.3, 0.4]],
            "metadatas": [
                [
                    {"strongs": "H0430"},
                    {"strongs": "H0410"},
                    {"strongs": "H0433"},
                    {"strongs": "H7706"},
                ]
            ],
        }
        mock_chroma.get_collection.return_value = collection

        related = search_api.expand_query_with_strongs("H0430", n_related=3)

        assert "H0430" not in related  # Should exclude self
        assert len(related) <= 3

    def test_returns_empty_when_strongs_not_found(self, search_api, mock_chroma):
        """Should return empty list when Strong's number has no embedding."""
        collection = MagicMock()
        collection.get.return_value = {"embeddings": []}
        mock_chroma.get_collection.return_value = collection

        related = search_api.expand_query_with_strongs("H9999")
        assert related == []

    def test_returns_empty_when_no_collection(self, search_api, mock_chroma):
        """Should return empty list when words collection doesn't exist."""
        mock_chroma.get_collection.return_value = None
        related = search_api.expand_query_with_strongs("H0430")
        assert related == []


# ------------------------------------------------------------------ #
#  _build_where_filter                                                 #
# ------------------------------------------------------------------ #


class TestBuildWhereFilter:
    """Tests for the where filter builder."""

    def test_no_filters(self):
        """Should return None when no filters specified."""
        assert SemanticSearchAPI._build_where_filter() is None

    def test_testament_only(self):
        """Should return simple testament filter."""
        result = SemanticSearchAPI._build_where_filter(testament_filter="old")
        assert result == {"testament": "old"}

    def test_book_only(self):
        """Should return simple book filter."""
        result = SemanticSearchAPI._build_where_filter(book_filter=19)
        assert result == {"book_id": 19}

    def test_combined(self):
        """Should combine with $and."""
        result = SemanticSearchAPI._build_where_filter(testament_filter="new", book_filter=43)
        assert "$and" in result
        conditions = result["$and"]
        assert {"testament": "new"} in conditions
        assert {"book_id": 43} in conditions


# ------------------------------------------------------------------ #
#  Data classes                                                        #
# ------------------------------------------------------------------ #


class TestDataClasses:
    """Tests for semantic search data classes."""

    def test_semantic_verse_result(self):
        """Should create SemanticVerseResult with defaults."""
        r = SemanticVerseResult(book_id=1, chapter=1, verse=1, similarity=0.95)
        assert r.testament == ""
        assert r.metadata == {}

    def test_related_word_result(self):
        """Should create RelatedWordResult with defaults."""
        r = RelatedWordResult(strongs_number="H0430", similarity=0.8)
        assert r.word == ""
        assert r.language == ""

    def test_hybrid_search_result(self):
        """Should create HybridSearchResult with defaults."""
        r = HybridSearchResult(book_id=1, chapter=1, verse=1)
        assert r.match_type == ""
        assert r.score == 0.0
