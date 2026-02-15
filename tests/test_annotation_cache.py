"""Tests for the verse annotation cache and query optimization (Phase 7).

Covers:
- verse_annotations_cache table migration
- Range query index migration
- Cache builder (build_annotation_cache)
- Cache read/write via SQLiteManager
- Cache integration in API routes (_try_annotation_cache)
- Query profiling (slow query logging)
"""

import json
import logging
import sqlite3
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from abba.database import SQLiteManager
from abba.database.cache_builder import (
    _get_book_id_mapping,
    _query_active_genre,
    _query_cross_refs,
    _query_cultural_context,
    _query_literary_structures,
    _query_passage_info,
    _query_richness,
    _query_speaker,
    _query_words,
    build_annotation_cache,
)
from abba.database.migrations import (
    add_range_query_indexes,
    add_verse_annotations_cache_table,
    run_migrations,
)


@pytest.fixture
def cache_db(seeded_db):
    """Seeded database with migrations run (includes cache table)."""
    run_migrations(seeded_db)
    return seeded_db


@pytest.fixture
def cache_db_manager(cache_db):
    """SQLiteManager connected to cache-enabled test database."""
    return SQLiteManager(cache_db)


class TestCacheTableMigration:
    """Test verse_annotations_cache table creation migration."""

    def test_migration_creates_table(self, seeded_db):
        add_verse_annotations_cache_table(seeded_db)
        # May already exist from run_migrations in seeded_db setup
        # Just verify the table exists after
        with sqlite3.connect(str(seeded_db)) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='verse_annotations_cache'"
            )
            assert cursor.fetchone()[0] == 1

    def test_migration_idempotent(self, cache_db):
        # Table already exists from first migration run
        result = add_verse_annotations_cache_table(cache_db)
        assert result is False  # Already exists

    def test_table_has_correct_columns(self, cache_db):
        with sqlite3.connect(str(cache_db)) as conn:
            cursor = conn.execute("PRAGMA table_info(verse_annotations_cache)")
            columns = {r[1] for r in cursor.fetchall()}
            expected = {
                "id",
                "book_id",
                "chapter",
                "verse",
                "words_json",
                "richness_flags_json",
                "cross_references_json",
                "cultural_context_json",
                "passage_info_json",
                "literary_structures_json",
                "speaker_json",
                "active_genre",
                "cache_version",
                "created_at",
            }
            assert expected.issubset(columns)

    def test_unique_constraint_on_verse(self, cache_db):
        with sqlite3.connect(str(cache_db)) as conn:
            conn.execute("INSERT INTO verse_annotations_cache (book_id, chapter, verse) VALUES (1, 1, 1)")
            conn.commit()
            # Second insert with same key should replace (not raise)
            conn.execute(
                "INSERT OR REPLACE INTO verse_annotations_cache (book_id, chapter, verse, active_genre) "
                "VALUES (1, 1, 1, 'narrative')"
            )
            conn.commit()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM verse_annotations_cache WHERE book_id=1 AND chapter=1 AND verse=1"
            )
            assert cursor.fetchone()[0] == 1


class TestRangeQueryIndexes:
    """Test range query optimization indexes migration."""

    def test_migration_adds_indexes(self, cache_db):
        # Indexes should already be created via run_migrations
        with sqlite3.connect(str(cache_db)) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            index_names = {r[0] for r in cursor.fetchall()}
            # Check a few key indexes
            assert "idx_annotation_cache_verse" in index_names

    def test_migration_idempotent(self, cache_db):
        result = add_range_query_indexes(cache_db)
        assert result is False  # Already exists


class TestSQLiteManagerCache:
    """Test SQLiteManager cache methods."""

    def test_get_annotation_cache_miss(self, cache_db_manager):
        result = cache_db_manager.get_annotation_cache(999, 1, 1)
        assert result is None

    def test_upsert_and_get_annotation_cache(self, cache_db_manager):
        data = {
            "words_json": json.dumps([{"word_num": 1, "original_text": "test"}]),
            "richness_flags_json": None,
            "cross_references_json": json.dumps([{"target_reference": "Gen 1:2", "ref_type": "parallel"}]),
            "cultural_context_json": None,
            "passage_info_json": None,
            "literary_structures_json": None,
            "speaker_json": json.dumps({"speaker": "God", "context_note": "creation"}),
            "active_genre": "narrative",
        }
        cache_db_manager.upsert_annotation_cache(1, 1, 1, data)

        result = cache_db_manager.get_annotation_cache(1, 1, 1)
        assert result is not None
        assert result["active_genre"] == "narrative"
        words = json.loads(result["words_json"])
        assert words[0]["word_num"] == 1
        speaker = json.loads(result["speaker_json"])
        assert speaker["speaker"] == "God"

    def test_upsert_replaces_existing(self, cache_db_manager):
        data1 = {"active_genre": "narrative"}
        cache_db_manager.upsert_annotation_cache(1, 1, 1, data1)

        data2 = {"active_genre": "poetry"}
        cache_db_manager.upsert_annotation_cache(1, 1, 1, data2)

        result = cache_db_manager.get_annotation_cache(1, 1, 1)
        assert result["active_genre"] == "poetry"

    def test_invalidate_cache_all(self, cache_db_manager):
        cache_db_manager.upsert_annotation_cache(1, 1, 1, {"active_genre": "narrative"})
        cache_db_manager.upsert_annotation_cache(1, 1, 2, {"active_genre": "narrative"})
        cache_db_manager.upsert_annotation_cache(43, 1, 1, {"active_genre": "gospel"})

        deleted = cache_db_manager.invalidate_annotation_cache()
        assert deleted == 3

        assert cache_db_manager.get_annotation_cache(1, 1, 1) is None

    def test_invalidate_cache_by_book(self, cache_db_manager):
        cache_db_manager.upsert_annotation_cache(1, 1, 1, {"active_genre": "narrative"})
        cache_db_manager.upsert_annotation_cache(43, 1, 1, {"active_genre": "gospel"})

        deleted = cache_db_manager.invalidate_annotation_cache(book_id=1)
        assert deleted == 1

        # Book 1 gone, book 43 still there
        assert cache_db_manager.get_annotation_cache(1, 1, 1) is None
        assert cache_db_manager.get_annotation_cache(43, 1, 1) is not None

    def test_database_stats_includes_cache(self, cache_db_manager):
        cache_db_manager.upsert_annotation_cache(1, 1, 1, {"active_genre": "narrative"})
        stats = cache_db_manager.get_database_stats()
        assert "verse_annotations_cache" in stats
        assert stats["verse_annotations_cache"] >= 1


class TestCacheBuilderHelpers:
    """Test individual cache builder query functions."""

    def test_query_words(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        words = _query_words(conn, "Gen", 1, 1)
        assert len(words) == 7  # Gen 1:1 has 7 Hebrew words
        assert words[0]["original_text"] is not None
        assert words[0]["strongs_number"] == "H7225"
        conn.close()

    def test_query_words_missing_verse(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        words = _query_words(conn, "Gen", 99, 99)
        assert words == []
        conn.close()

    def test_query_richness_empty_table(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        richness = _query_richness(conn, "Gen", 1, 1)
        # Empty because word_richness table has no data in test fixtures
        assert isinstance(richness, list)
        conn.close()

    def test_query_cross_refs_empty(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        refs = _query_cross_refs(conn, 1, 1, 1)
        assert isinstance(refs, list)
        conn.close()

    def test_query_cultural_context_empty(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        ctx = _query_cultural_context(conn, 1)
        assert isinstance(ctx, list)
        conn.close()

    def test_query_passage_info_empty(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        info = _query_passage_info(conn, 1, 1, 1)
        # No passages in test fixtures
        assert info is None
        conn.close()

    def test_query_literary_structures_empty(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        structures = _query_literary_structures(conn, 1, 1, 1)
        assert isinstance(structures, list)
        conn.close()

    def test_query_speaker_empty(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        speaker = _query_speaker(conn, 1, 1, 1)
        assert speaker is None
        conn.close()

    def test_query_active_genre_empty(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        genre = _query_active_genre(conn, 1, 1, 1)
        # Falls back to book_metadata which doesn't exist in fixtures
        assert genre is None
        conn.close()

    def test_get_book_id_mapping(self, cache_db):
        conn = sqlite3.connect(str(cache_db))
        conn.row_factory = sqlite3.Row
        mapping = _get_book_id_mapping(conn)
        # Should map book names from words table to book IDs
        assert isinstance(mapping, dict)
        conn.close()


class TestBuildAnnotationCache:
    """Test the full cache build process."""

    def test_build_creates_cache_entries(self, cache_db):
        count = build_annotation_cache(cache_db)
        # Should cache entries for verses in words table (Gen 1:1, John 1:1)
        assert count >= 2

        # Verify entries exist
        conn = sqlite3.connect(str(cache_db))
        cursor = conn.execute("SELECT COUNT(*) FROM verse_annotations_cache")
        assert cursor.fetchone()[0] >= 2
        conn.close()

    def test_build_stores_words_json(self, cache_db):
        build_annotation_cache(cache_db)

        conn = sqlite3.connect(str(cache_db))
        cursor = conn.execute(
            "SELECT words_json FROM verse_annotations_cache WHERE book_id = 1 AND chapter = 1 AND verse = 1"
        )
        row = cursor.fetchone()
        if row and row[0]:
            words = json.loads(row[0])
            assert len(words) == 7  # Gen 1:1 has 7 words
            assert words[0]["strongs_number"] == "H7225"
        conn.close()

    def test_build_is_idempotent(self, cache_db):
        count1 = build_annotation_cache(cache_db)
        count2 = build_annotation_cache(cache_db)
        # Second run replaces, should produce same count
        assert count1 == count2


class TestQueryProfiling:
    """Test slow query logging."""

    def test_slow_query_logs_warning(self, cache_db_manager, caplog):
        """Verify that artificially slow queries trigger a warning log."""
        # Patch the threshold to be very low to trigger on any query
        with patch("abba.database.sqlite_manager._SLOW_QUERY_THRESHOLD_MS", 0.0):
            with caplog.at_level(logging.WARNING, logger="abba.database.sqlite_manager"):
                cache_db_manager.execute_query("SELECT COUNT(*) FROM verses")
            assert any("Slow query" in record.message for record in caplog.records)

    def test_fast_query_no_warning(self, cache_db_manager, caplog):
        """Normal-speed queries should not trigger warnings."""
        with patch("abba.database.sqlite_manager._SLOW_QUERY_THRESHOLD_MS", 10000.0):
            with caplog.at_level(logging.WARNING, logger="abba.database.sqlite_manager"):
                cache_db_manager.execute_query("SELECT 1")
            slow_warnings = [r for r in caplog.records if "Slow query" in r.message]
            assert len(slow_warnings) == 0


class TestCacheAPIIntegration:
    """Test that the API routes use the annotation cache."""

    def test_try_annotation_cache_returns_false_on_miss(self, cache_db):
        """When cache is empty, _try_annotation_cache returns False."""
        from abba.api.models import DepthLevel, VerseResponse
        from abba.api.routes import _try_annotation_cache, configure_db

        db = SQLiteManager(cache_db)
        configure_db(db)

        response = VerseResponse(
            reference="Gen 1:1",
            book_name="Genesis",
            chapter=1,
            verse=1,
            text="In the beginning...",
            translation_id="engbsb",
        )
        result = _try_annotation_cache(1, 1, 1, DepthLevel.DEEP, response, "engbsb", "Gen")
        assert result is False

    def test_try_annotation_cache_populates_from_cache(self, cache_db):
        """When cache has data, _try_annotation_cache populates the response."""
        from abba.api.models import DepthLevel, VerseResponse
        from abba.api.routes import _try_annotation_cache, configure_db

        db = SQLiteManager(cache_db)
        configure_db(db)

        # Insert cache data
        db.upsert_annotation_cache(
            1,
            1,
            1,
            {
                "words_json": json.dumps(
                    [
                        {
                            "word_num": 1,
                            "original_text": "test",
                            "transliteration": "t",
                            "english_gloss": "test",
                            "strongs_number": "H0001",
                            "morphology_code": "HN",
                            "language": "hebrew",
                        }
                    ]
                ),
                "richness_flags_json": None,
                "cross_references_json": json.dumps(
                    [{"target_reference": "Gen 1:2", "ref_type": "parallel", "confidence": 0.9, "notes": None}]
                ),
                "cultural_context_json": None,
                "passage_info_json": None,
                "literary_structures_json": None,
                "speaker_json": json.dumps({"speaker": "God", "context_note": "creation"}),
                "active_genre": "narrative",
            },
        )

        response = VerseResponse(
            reference="Gen 1:1",
            book_name="Genesis",
            chapter=1,
            verse=1,
            text="In the beginning...",
            translation_id="engbsb",
        )
        result = _try_annotation_cache(1, 1, 1, DepthLevel.DEEP, response, "engbsb", "Gen")
        assert result is True
        assert len(response.words) == 1
        assert response.words[0].strongs_number == "H0001"
        assert len(response.cross_references) == 1
        assert response.speaker.speaker == "God"
        assert response.genre == "narrative"
        assert response.is_descriptive is True

    def test_try_annotation_cache_standard_depth(self, cache_db):
        """Standard depth only uses words and richness from cache."""
        from abba.api.models import DepthLevel, VerseResponse
        from abba.api.routes import _try_annotation_cache, configure_db

        db = SQLiteManager(cache_db)
        configure_db(db)

        db.upsert_annotation_cache(
            1,
            1,
            1,
            {
                "words_json": json.dumps(
                    [
                        {
                            "word_num": 1,
                            "original_text": "x",
                            "transliteration": "x",
                            "english_gloss": "x",
                            "strongs_number": "H0001",
                            "morphology_code": "HN",
                            "language": "hebrew",
                        }
                    ]
                ),
                "richness_flags_json": None,
                "active_genre": "narrative",
            },
        )

        response = VerseResponse(
            reference="Gen 1:1",
            book_name="Genesis",
            chapter=1,
            verse=1,
            text="In the beginning...",
            translation_id="engbsb",
        )
        result = _try_annotation_cache(1, 1, 1, DepthLevel.STANDARD, response, "engbsb", "Gen")
        assert result is True
        assert len(response.words) == 1
        # Deep fields should not be populated at STANDARD depth
        assert response.cross_references is None
        assert response.speaker is None
