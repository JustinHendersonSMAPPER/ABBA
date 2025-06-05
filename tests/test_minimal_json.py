"""
Tests for minimal JSON export.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from abba.export.minimal_json import (
    MinimalVerse,
    create_verse_lookup,
    export_to_json_files,
    load_json_verses,
)


class TestMinimalJSON:
    """Test minimal JSON export functionality."""

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        return [
            MinimalVerse("GEN", 1, 1, "In the beginning God created the heavens and the earth."),
            MinimalVerse("GEN", 1, 2, "And the earth was without form, and void."),
            MinimalVerse("GEN", 2, 1, "Thus the heavens and the earth were finished."),
            MinimalVerse("JOH", 3, 16, "For God so loved the world."),
            MinimalVerse("PSA", 23, 1, "The LORD is my shepherd."),
        ]

    def test_single_file_export(self, sample_verses):
        """Test exporting all verses to a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = export_to_json_files(sample_verses, tmpdir, single_file=True)

            # Check stats
            assert stats["total_verses"] == 5
            assert stats["files_created"] == 1

            # Check file exists
            json_file = Path(tmpdir) / "verses.json"
            assert json_file.exists()

            # Load and verify content
            verses = load_json_verses(str(json_file))
            assert len(verses) == 5
            assert verses[0]["book"] == "GEN"
            assert verses[0]["chapter"] == 1
            assert verses[0]["verse"] == 1

    def test_by_chapter_export(self, sample_verses):
        """Test exporting verses by chapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = export_to_json_files(sample_verses, tmpdir, by_chapter=True)

            # Check stats
            assert stats["total_verses"] == 5
            assert stats["files_created"] == 5  # 4 chapter files + 1 index
            assert stats["book_count"] == 3
            assert stats["chapter_count"] == 4

            # Check index file
            index_file = Path(tmpdir) / "index.json"
            assert index_file.exists()

            with open(index_file) as f:
                index = json.load(f)

            assert len(index["books"]) == 3
            assert "GEN" in index["books"]
            assert index["structure"]["GEN"]["chapters"] == [1, 2]

            # Check chapter files
            gen1_file = Path(tmpdir) / "GEN" / "GEN_1.json"
            assert gen1_file.exists()

            with open(gen1_file) as f:
                chapter_data = json.load(f)

            assert chapter_data["book"] == "GEN"
            assert chapter_data["chapter"] == 1
            assert len(chapter_data["verses"]) == 2

    def test_by_book_export(self, sample_verses):
        """Test exporting verses by book only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = export_to_json_files(sample_verses, tmpdir, by_book=True, by_chapter=False)

            # Check stats
            assert stats["total_verses"] == 5
            assert stats["files_created"] == 4  # 3 book files + 1 index

            # Check book file
            gen_file = Path(tmpdir) / "GEN" / "GEN.json"
            assert gen_file.exists()

            with open(gen_file) as f:
                book_data = json.load(f)

            assert book_data["book"] == "GEN"
            assert len(book_data["verses"]) == 3

    def test_verse_lookup(self, sample_verses):
        """Test verse lookup creation."""
        lookup = create_verse_lookup(sample_verses)

        assert len(lookup) == 5
        assert "GEN_1_1" in lookup
        assert "JOH_3_16" in lookup

        verse = lookup["PSA_23_1"]
        assert verse.book == "PSA"
        assert verse.chapter == 23
        assert verse.verse == 1

    def test_empty_export(self):
        """Test exporting empty verse list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = export_to_json_files([], tmpdir, single_file=True)

            assert stats["total_verses"] == 0
            assert stats["files_created"] == 1

            json_file = Path(tmpdir) / "verses.json"
            assert json_file.exists()

            with open(json_file) as f:
                data = json.load(f)

            assert data["metadata"]["verse_count"] == 0
            assert len(data["verses"]) == 0

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        verses = [
            MinimalVerse("GEN", 1, 1, "בְּרֵאשִׁית בָּרָא אֱלֹהִים"),  # Hebrew
            MinimalVerse("JOH", 1, 1, "Ἐν ἀρχῇ ἦν ὁ λόγος"),  # Greek
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_to_json_files(verses, tmpdir, single_file=True)

            json_file = Path(tmpdir) / "verses.json"
            verses_loaded = load_json_verses(str(json_file))

            assert verses_loaded[0]["text"] == "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
            assert verses_loaded[1]["text"] == "Ἐν ἀρχῇ ἦν ὁ λόγος"
