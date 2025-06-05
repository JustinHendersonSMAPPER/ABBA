"""
Tests for simple SQLite export.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from abba.export.simple_sqlite_export import (
    SimpleSQLiteExporter,
    SimpleSQLiteConfig,
    export_to_sqlite
)
from abba.parsers.translation_parser import TranslationVerse
from abba.verse_id import VerseID


class TestSimpleSQLiteExport:
    """Test simple SQLite export functionality."""
    
    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        return [
            TranslationVerse(
                verse_id=VerseID("GEN", 1, 1),
                text="In the beginning God created the heavens and the earth.",
                original_book_name="Genesis",
                original_chapter=1,
                original_verse=1
            ),
            TranslationVerse(
                verse_id=VerseID("GEN", 1, 2),
                text="The earth was without form and void.",
                original_book_name="Genesis", 
                original_chapter=1,
                original_verse=2
            ),
            TranslationVerse(
                verse_id=VerseID("JHN", 1, 1),
                text="In the beginning was the Word.",
                original_book_name="John",
                original_chapter=1,
                original_verse=1
            ),
        ]
    
    def test_basic_export(self, sample_verses):
        """Test basic SQLite export."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            output_path = tmp.name
            
        try:
            # Export verses
            export_to_sqlite(sample_verses, output_path)
            
            # Verify database exists
            assert Path(output_path).exists()
            
            # Check contents
            conn = sqlite3.connect(output_path)
            
            # Check verse count
            count = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
            assert count == 3
            
            # Check books
            books = conn.execute("SELECT book_code, book_name FROM books ORDER BY book_code").fetchall()
            assert len(books) == 2
            assert books[0] == ("GEN", "Genesis")
            assert books[1] == ("JHN", "John")
            
            # Check specific verse
            verse = conn.execute(
                "SELECT text FROM verses WHERE verse_id = ?",
                ("GEN.1.1",)
            ).fetchone()
            assert verse[0] == "In the beginning God created the heavens and the earth."
            
            conn.close()
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_fts_search(self, sample_verses):
        """Test FTS5 search functionality."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            output_path = tmp.name
            
        try:
            # Export with FTS enabled
            config = SimpleSQLiteConfig(output_path=output_path, enable_fts=True)
            exporter = SimpleSQLiteExporter(config)
            exporter.export_verses(sample_verses)
            
            # Test search
            conn = sqlite3.connect(output_path)
            
            # Search for "beginning"
            results = conn.execute("""
                SELECT v.verse_id, v.text 
                FROM verses v
                JOIN verses_fts ON v.verse_id = verses_fts.verse_id
                WHERE verses_fts MATCH 'beginning'
                ORDER BY v.verse_id
            """).fetchall()
            
            assert len(results) == 2
            assert results[0][0] == "GEN.1.1"
            assert results[1][0] == "JHN.1.1"
            
            conn.close()
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_cross_references(self, sample_verses):
        """Test adding cross references."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            output_path = tmp.name
            
        try:
            # Export verses
            config = SimpleSQLiteConfig(output_path=output_path)
            exporter = SimpleSQLiteExporter(config)
            exporter.export_verses(sample_verses)
            
            # Add cross references
            references = [
                {'from_verse': 'GEN.1.1', 'to_verse': 'JHN.1.1', 'type': 'parallel'},
                {'from_verse': 'JHN.1.1', 'to_verse': 'GEN.1.1', 'type': 'allusion'},
            ]
            exporter.add_cross_references(references)
            
            # Check references
            conn = sqlite3.connect(output_path)
            refs = conn.execute(
                "SELECT from_verse, to_verse, reference_type FROM cross_references ORDER BY id"
            ).fetchall()
            
            assert len(refs) == 2
            assert refs[0] == ('GEN.1.1', 'JHN.1.1', 'parallel')
            assert refs[1] == ('JHN.1.1', 'GEN.1.1', 'allusion')
            
            conn.close()
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_metadata(self, sample_verses):
        """Test metadata storage."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            output_path = tmp.name
            
        try:
            # Export verses
            export_to_sqlite(sample_verses, output_path)
            
            # Check metadata
            conn = sqlite3.connect(output_path)
            metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
            
            assert metadata['version'] == '1.0'
            assert metadata['verse_count'] == '3'
            assert metadata['book_count'] == '2'
            
            conn.close()
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_indexes_created(self, sample_verses):
        """Test that indexes are created."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            output_path = tmp.name
            
        try:
            # Export verses
            export_to_sqlite(sample_verses, output_path)
            
            # Check indexes
            conn = sqlite3.connect(output_path)
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
            
            index_names = [idx[0] for idx in indexes]
            assert 'idx_verses_book_chapter' in index_names
            assert 'idx_xref_from' in index_names
            assert 'idx_xref_to' in index_names
            
            conn.close()
            
        finally:
            Path(output_path).unlink(missing_ok=True)