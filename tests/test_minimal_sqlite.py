"""
Tests for minimal SQLite export.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from abba.export.minimal_sqlite import (
    MinimalVerse,
    create_sqlite_database,
    query_verses,
    search_verses,
)


class TestMinimalSQLite:
    """Test minimal SQLite export functionality."""

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        return [
            MinimalVerse("GEN", 1, 1, "In the beginning God created the heavens and the earth."),
            MinimalVerse(
                "GEN",
                1,
                2,
                "And the earth was without form, and void; and darkness was upon the face of the deep.",
            ),
            MinimalVerse(
                "JOH", 3, 16, "For God so loved the world, that he gave his only begotten Son."
            ),
            MinimalVerse("PSA", 23, 1, "The LORD is my shepherd; I shall not want."),
        ]

    def test_create_database(self, sample_verses):
        """Test database creation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            # Create database
            create_sqlite_database(sample_verses, db_path)

            # Verify file exists
            assert Path(db_path).exists()

            # Check database structure
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            assert "verses" in tables
            assert "metadata" in tables
            assert "verses_fts" in tables

            # Check verse count
            cursor = conn.execute("SELECT COUNT(*) FROM verses")
            count = cursor.fetchone()[0]
            assert count == 4

            conn.close()

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_query_verses(self, sample_verses):
        """Test verse querying."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            create_sqlite_database(sample_verses, db_path)

            # Query all verses
            all_verses = query_verses(db_path)
            assert len(all_verses) == 4

            # Query by book
            gen_verses = query_verses(db_path, book="GEN")
            assert len(gen_verses) == 2
            assert all(v["book"] == "GEN" for v in gen_verses)

            # Query by book and chapter
            gen1_verses = query_verses(db_path, book="GEN", chapter=1)
            assert len(gen1_verses) == 2
            assert all(v["chapter"] == 1 for v in gen1_verses)

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_search_verses(self, sample_verses):
        """Test full-text search."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            create_sqlite_database(sample_verses, db_path, enable_fts=True)

            # Search for "God"
            results = search_verses(db_path, "God")
            assert len(results) >= 2  # Should find Genesis 1:1 and John 3:16

            # Search for "shepherd"
            results = search_verses(db_path, "shepherd")
            assert len(results) == 1
            assert results[0]["book"] == "PSA"

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_no_fts(self, sample_verses):
        """Test error when FTS is disabled."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            create_sqlite_database(sample_verses, db_path, enable_fts=False)

            # Search should raise error
            with pytest.raises(ValueError, match="Full-text search not enabled"):
                search_verses(db_path, "God")

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_empty_database(self):
        """Test creating empty database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            create_sqlite_database([], db_path)

            # Should still create tables
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            assert "verses" in tables
            assert "metadata" in tables

            conn.close()

        finally:
            Path(db_path).unlink(missing_ok=True)
