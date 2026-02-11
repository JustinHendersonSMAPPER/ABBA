"""Tests for Phase 2 embedding pipeline: dedup, legacy cleanup, original embeddings."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from abba.embeddings.chroma_manager import ChromaManager
from abba.embeddings.context_builder import ContextBuilder
from abba.embeddings.model_manager import EmbeddingModelManager


@pytest.fixture
def mock_db():
    """Create a mock database manager."""
    return MagicMock()


@pytest.fixture
def mock_chroma():
    """Create a mock ChromaManager."""
    chroma = MagicMock(spec=ChromaManager)
    chroma.get_database_stats.return_value = {"collections": {}}
    return chroma


@pytest.fixture
def mock_models():
    """Create a mock model manager."""
    return MagicMock(spec=EmbeddingModelManager)


@pytest.fixture
def mock_context():
    """Create a mock context builder."""
    return MagicMock(spec=ContextBuilder)


@pytest.fixture
def pipeline(mock_db, mock_chroma, mock_models, mock_context, tmp_path):
    """Create an EmbeddingPipeline with mocked dependencies."""
    with patch("abba.embeddings.embedding_pipeline.EmbeddingPipeline._log_gpu_status"):
        from abba.embeddings.embedding_pipeline import EmbeddingPipeline

        pipe = EmbeddingPipeline(
            db_manager=mock_db,
            chroma_manager=mock_chroma,
            model_manager=mock_models,
            context_builder=mock_context,
        )
        pipe.progress_file = tmp_path / ".embedding_progress.json"
        return pipe


class TestRemoveLegacyTranslationEmbeddings:
    """Tests for remove_legacy_translation_embeddings."""

    def test_removes_existing_legacy_collection(self, pipeline, mock_chroma):
        """Should remove legacy verses collection when it exists with data."""
        collection = MagicMock()
        collection.count.return_value = 5000
        mock_chroma.get_collection.return_value = collection

        result = pipeline.remove_legacy_translation_embeddings()

        assert result["removed"] is True
        assert result["count"] == 5000
        mock_chroma.delete_collection.assert_called_once_with("verses")

    def test_removes_empty_legacy_collection(self, pipeline, mock_chroma):
        """Should remove legacy collection even when empty."""
        collection = MagicMock()
        collection.count.return_value = 0
        mock_chroma.get_collection.return_value = collection

        result = pipeline.remove_legacy_translation_embeddings()

        assert result["removed"] is True
        assert result["count"] == 0
        mock_chroma.delete_collection.assert_called_once_with("verses")

    def test_no_legacy_collection(self, pipeline, mock_chroma):
        """Should handle case when no legacy collection exists."""
        mock_chroma.get_collection.return_value = None

        result = pipeline.remove_legacy_translation_embeddings()

        assert result["removed"] is False
        assert result["count"] == 0
        mock_chroma.delete_collection.assert_not_called()


class TestVerifyDeduplication:
    """Tests for verify_deduplication."""

    @staticmethod
    def _make_collection(count):
        coll = MagicMock()
        coll.count.return_value = count
        return coll

    def test_passes_when_no_legacy_and_reasonable_count(self, pipeline, mock_chroma):
        """Should pass when legacy removed and original count is reasonable."""
        mock_chroma.get_collection.side_effect = lambda name: {
            "verses": None,
            "original_verses": self._make_collection(31000),
        }.get(name)

        result = pipeline.verify_deduplication()

        assert result["passed"] is True
        assert len(result["checks"]) == 2
        assert all(c["passed"] for c in result["checks"])

    def test_fails_when_legacy_exists(self, pipeline, mock_chroma):
        """Should fail when legacy verses collection still has data."""
        legacy = MagicMock()
        legacy.count.return_value = 5000000

        mock_chroma.get_collection.side_effect = lambda name: {
            "verses": legacy,
            "original_verses": self._make_collection(31000),
        }.get(name)

        result = pipeline.verify_deduplication()

        assert result["passed"] is False
        legacy_check = next(c for c in result["checks"] if c["name"] == "legacy_removed")
        assert legacy_check["passed"] is False

    def test_fails_when_count_too_high(self, pipeline, mock_chroma):
        """Should fail when original_verses has too many embeddings (not deduplicated)."""
        mock_chroma.get_collection.side_effect = lambda name: {
            "verses": None,
            "original_verses": self._make_collection(500000),
        }.get(name)

        result = pipeline.verify_deduplication()

        assert result["passed"] is False

    def test_passes_when_original_not_yet_created(self, pipeline, mock_chroma):
        """Should pass when original_verses collection doesn't exist yet."""
        mock_chroma.get_collection.return_value = None

        result = pipeline.verify_deduplication()

        assert result["passed"] is True


class TestGetEmbeddingStats:
    """Tests for get_embedding_stats."""

    def test_returns_collections_and_models(self, pipeline, mock_chroma, mock_models):
        """Should return stats with collections and model info."""
        mock_chroma.get_database_stats.return_value = {
            "collections": {"original_verses": {"count": 31000}},
            "total_embeddings": 31000,
        }
        mock_models.get_model_info.return_value = {"name": "test-model", "loaded": False}

        stats = pipeline.get_embedding_stats()

        assert "collections" in stats
        assert "models" in stats
        assert "progress" in stats
        assert stats["collections"]["original_verses"]["count"] == 31000


class TestProgressTracking:
    """Tests for progress tracking."""

    def test_save_and_load_progress(self, pipeline, tmp_path):
        """Should persist progress to JSON file."""
        pipeline._update_progress("verses", "test", 500)
        pipeline._save_progress()

        loaded = json.loads(pipeline.progress_file.read_text())
        assert loaded["verses"]["test"]["last_count"] == 500

    def test_mark_translation_embedded(self, pipeline, mock_db):
        """Should mark a translation as complete."""
        mock_conn = MagicMock()
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        pipeline._mark_translation_embedded("engbsb")
        assert pipeline.progress["verses"]["engbsb"]["complete"] is True

    def test_is_translation_embedded(self, pipeline):
        """Should correctly report embedded status."""
        assert pipeline._is_translation_embedded("engbsb") is False

        pipeline.progress["verses"] = {"engbsb": {"complete": True}}
        assert pipeline._is_translation_embedded("engbsb") is True
