"""Tests for the OriginalLanguageEmbeddingPipeline."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_db():
    """Create a mock database manager."""
    return MagicMock()


@pytest.fixture
def mock_chroma():
    """Create a mock ChromaManager."""
    return MagicMock()


@pytest.fixture
def mock_models():
    """Create a mock model manager."""
    return MagicMock()


@pytest.fixture
def mock_context():
    """Create a mock context builder."""
    return MagicMock()


@pytest.fixture
def pipeline(mock_db, mock_chroma, mock_models, mock_context, tmp_path):
    """Create an OriginalLanguageEmbeddingPipeline with mocked dependencies."""
    with patch("abba.embeddings.original_language_pipeline.OriginalLanguageEmbeddingPipeline._log_gpu_status"):
        from abba.embeddings.original_language_pipeline import OriginalLanguageEmbeddingPipeline

        pipe = OriginalLanguageEmbeddingPipeline(
            db_manager=mock_db,
            chroma_manager=mock_chroma,
            model_manager=mock_models,
            context_builder=mock_context,
        )
        pipe.progress_file = tmp_path / ".embedding_progress.json"
        return pipe


class TestGetCanonicalVerses:
    """Tests for _get_canonical_verses."""

    def test_returns_canonical_verses(self, pipeline, mock_db):
        """Should return deduplicated canonical verses from stepbible_verses."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (
                1,
                1,
                1,
                "bereshit bara elohim",
                None,
                None,
                "H7225 H1254 H0430",
                "HR HVqp HNcmp",
                "beginning created God",
                3,
                "Gen",
                "hebrew",
            ),
            (
                43,
                1,
                1,
                None,
                "En arche en ho Logos",
                None,
                "G1722 G0746 G1510 G3588 G3056",
                "GP GNdfs GVIia GEdnms GNnms",
                "In beginning was the Word",
                5,
                "Jhn",
                "greek",
            ),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        verses = pipeline._get_canonical_verses()

        assert len(verses) == 2
        assert verses[0]["book_id"] == 1
        assert verses[1]["book_id"] == 43

    def test_returns_empty_when_no_data(self, pipeline, mock_db):
        """Should return empty list when no stepbible data exists."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        verses = pipeline._get_canonical_verses()
        assert verses == []


class TestBuildOriginalContext:
    """Tests for _build_original_context."""

    def test_builds_hebrew_context(self, pipeline):
        """Should build context string for Hebrew verse."""
        verse = {
            "book_id": 1,
            "chapter": 1,
            "verse": 1,
            "hebrew_text": "bereshit bara",
            "greek_text": None,
            "aramaic_text": None,
            "strongs_sequence": "H7225 H1254",
            "morphology_sequence": "HR HVqp",
            "english_gloss": "In beginning created",
            "book_name": "Gen",
        }

        context = pipeline._build_original_context(verse)

        assert context is not None
        assert "Hebrew:" in context
        assert "Gen 1:1" in context

    def test_builds_greek_context(self, pipeline):
        """Should build context string for Greek verse."""
        verse = {
            "book_id": 43,
            "chapter": 1,
            "verse": 1,
            "hebrew_text": None,
            "greek_text": "En arche en ho Logos",
            "aramaic_text": None,
            "strongs_sequence": "G1722 G0746",
            "morphology_sequence": "GP GNdfs",
            "english_gloss": "In beginning was the Word",
            "book_name": "Jhn",
        }

        context = pipeline._build_original_context(verse)

        assert context is not None
        assert "Greek:" in context

    def test_returns_none_for_empty_verse(self, pipeline):
        """Should return None when no original text is available."""
        verse = {
            "book_id": 1,
            "chapter": 1,
            "verse": 1,
            "hebrew_text": None,
            "greek_text": None,
            "aramaic_text": None,
        }

        assert pipeline._build_original_context(verse) is None


class TestProgressTracking:
    """Tests for progress tracking in original language pipeline."""

    def test_are_original_verses_embedded_false_initially(self, pipeline):
        """Should return False when no original verses have been embedded."""
        assert pipeline._are_original_verses_embedded() is False

    def test_mark_and_check_completion(self, pipeline):
        """Should mark completion and verify it."""
        pipeline._mark_original_verses_complete()
        assert pipeline.progress["original_verses"]["canonical"]["complete"] is True

    def test_update_progress(self, pipeline):
        """Should track intermediate progress."""
        pipeline._update_progress("original_verses", "canonical", 500)
        assert pipeline.progress["original_verses"]["canonical"]["last_count"] == 500
