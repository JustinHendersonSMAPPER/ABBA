"""
Tests for ABBA SQLite exporter.

Test coverage for SQLite database export, schema creation,
data insertion, FTS5 indexing, and mobile optimization.
"""

import pytest
import asyncio
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Iterator
from unittest.mock import Mock, patch

from abba.export.sqlite_exporter import SQLiteExporter, SQLiteConfig
from abba.export.base import (
    ExportFormat,
    ExportResult,
    ExportStatus,
    CanonicalDataset,
    ValidationResult,
)
from abba.alignment.unified_reference import UnifiedVerse
from abba.annotations.models import Annotation, AnnotationType, AnnotationLevel, Topic
from abba.timeline.models import Event, TimePeriod, EventType, CertaintyLevel
from abba.verse_id import VerseID


class TestSQLiteConfig:
    """Test SQLite exporter configuration."""

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            config = SQLiteConfig(output_path=db_path, enable_fts5=True, batch_size=1000)

            assert config.output_path == db_path
            assert config.enable_fts5 is True
            assert config.batch_size == 1000
            assert config.format_type == ExportFormat.SQLITE
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = SQLiteConfig(output_path="/tmp/test.db")
        validation = config.validate()
        assert validation.is_valid

        # Invalid config - no output path
        config = SQLiteConfig(output_path="")
        validation = config.validate()
        assert not validation.is_valid

        # Invalid config - invalid batch size
        config = SQLiteConfig(output_path="/tmp/test.db", batch_size=0)
        validation = config.validate()
        assert not validation.is_valid

    def test_mobile_optimization_config(self):
        """Test mobile optimization configuration."""
        config = SQLiteConfig(
            output_path="/tmp/mobile.db",
            enable_wal_mode=True,
            vacuum_on_completion=True,
            compress_large_text=True,
            compression_threshold=512,
        )

        assert config.enable_wal_mode is True
        assert config.vacuum_on_completion is True
        assert config.compress_large_text is True
        assert config.compression_threshold == 512


class TestSQLiteExporter:
    """Test SQLite exporter functionality."""

    @pytest.fixture
    def config(self):
        """Create test SQLite configuration."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        yield SQLiteConfig(
            output_path=db_path,
            enable_fts5=True,
            enable_wal_mode=True,
            batch_size=100,
            vacuum_on_completion=True,
        )

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def exporter(self, config):
        """Create test SQLite exporter."""
        return SQLiteExporter(config)

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        verses = []
        for i in range(10):
            verse_id = VerseID("GEN", 1, i + 1)
            verse = UnifiedVerse(
                verse_id=verse_id,
                translations={
                    "ESV": f"This is verse {i + 1} in English Standard Version.",
                    "NIV": f"This is verse {i + 1} in New International Version.",
                },
                hebrew_tokens=(
                    [
                        {"word": "בְּרֵאשִׁית", "lemma": "רֵאשִׁית", "strongs": "H7225"},
                        {"word": "בָּרָא", "lemma": "בָּרָא", "strongs": "H1254"},
                    ]
                    if i == 0
                    else None
                ),
                greek_tokens=(
                    [
                        {"word": "Ἐν", "lemma": "ἐν", "strongs": "G1722"},
                        {"word": "ἀρχῇ", "lemma": "ἀρχή", "strongs": "G746"},
                    ]
                    if i == 0
                    else None
                ),
            )
            verses.append(verse)
        return verses

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotations for testing."""
        annotations = []
        for i in range(5):
            annotation = Annotation(
                id=f"ann_{i}",
                verse_id=VerseID("GEN", 1, i + 1),
                annotation_type=AnnotationType.TOPIC,
                level=AnnotationLevel.VERSE,
                confidence=0.8,
                topics=[Topic(id=f"topic_{i}", name=f"Topic {i}")],
            )
            annotations.append(annotation)
        return annotations

    @pytest.fixture
    def sample_events(self):
        """Create sample timeline events."""
        from abba.timeline.models import TimePoint, Location, Participant

        events = []
        for i in range(3):
            event = Event(
                id=f"event_{i}",
                name=f"Event {i}",
                description=f"Description of event {i}",
                event_type=EventType.HISTORICAL,
                certainty_level=CertaintyLevel.HIGH,
                categories=["biblical", "historical"],
                time_point=TimePoint(year=-2000 + i * 100),
                location=Location(name=f"Location {i}"),
                participants=[Participant(id=f"person_{i}", name=f"Person {i}")],
                verse_refs=[VerseID("GEN", 1, i + 1)],
            )
            events.append(event)
        return events

    @pytest.fixture
    def sample_dataset(self, sample_verses, sample_annotations, sample_events):
        """Create sample canonical dataset."""
        return CanonicalDataset(
            verses=iter(sample_verses),
            annotations=iter(sample_annotations),
            timeline_events=iter(sample_events),
            metadata={"format": "test", "version": "1.0"},
        )

    def test_exporter_initialization(self, exporter, config):
        """Test exporter initialization."""
        assert exporter.config == config
        assert exporter.db_path == Path(config.output_path)
        assert exporter.logger is not None

    def test_config_validation(self, exporter):
        """Test configuration validation."""
        validation = exporter.validate_config()
        assert validation.is_valid

    def test_supported_features(self, exporter):
        """Test supported features."""
        features = exporter.get_supported_features()
        assert "full_text_search" in features
        assert "offline_access" in features
        assert "mobile_optimized" in features
        assert "cross_references" in features

    @pytest.mark.asyncio
    async def test_database_creation(self, exporter, sample_dataset):
        """Test database creation and schema setup."""
        result = await exporter.export(sample_dataset)

        assert result.status == ExportStatus.COMPLETED
        assert Path(exporter.config.output_path).exists()

        # Check database schema
        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            )
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "verses",
                "original_language",
                "verse_translations",
                "annotations",
                "annotation_topics",
                "cross_references",
                "timeline_events",
                "timeline_periods",
                "export_metadata",
            }

            assert expected_tables.issubset(tables)

    @pytest.mark.asyncio
    async def test_verse_export(self, exporter, sample_dataset):
        """Test verse data export."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check verse count
            cursor.execute("SELECT COUNT(*) FROM verses")
            verse_count = cursor.fetchone()[0]
            assert verse_count == 10

            # Check verse data
            cursor.execute(
                """
                SELECT verse_id, book, chapter, verse 
                FROM verses 
                WHERE verse_id = 'GEN.1.1'
            """
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "GEN"
            assert row[2] == 1
            assert row[3] == 1

            # Check translations
            cursor.execute(
                """
                SELECT translation_id, text 
                FROM verse_translations 
                WHERE verse_id = 'GEN.1.1'
            """
            )
            translations = {row[0]: row[1] for row in cursor.fetchall()}
            assert "ESV" in translations
            assert "NIV" in translations
            assert "verse 1" in translations["ESV"]

    @pytest.mark.asyncio
    async def test_original_language_export(self, exporter, sample_dataset):
        """Test original language token export."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check Hebrew tokens
            cursor.execute(
                """
                SELECT word, lemma, strongs, language
                FROM original_language 
                WHERE verse_id = 'GEN.1.1' AND language = 'hebrew'
            """
            )
            hebrew_tokens = cursor.fetchall()
            assert len(hebrew_tokens) == 2
            assert hebrew_tokens[0][0] == "בְּרֵאשִׁית"
            assert hebrew_tokens[0][3] == "hebrew"

            # Check Greek tokens
            cursor.execute(
                """
                SELECT word, lemma, strongs, language
                FROM original_language 
                WHERE verse_id = 'GEN.1.1' AND language = 'greek'
            """
            )
            greek_tokens = cursor.fetchall()
            assert len(greek_tokens) == 2
            assert greek_tokens[0][0] == "Ἐν"
            assert greek_tokens[0][3] == "greek"

    @pytest.mark.asyncio
    async def test_annotation_export(self, exporter, sample_dataset):
        """Test annotation export."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check annotations
            cursor.execute("SELECT COUNT(*) FROM annotations")
            annotation_count = cursor.fetchone()[0]
            assert annotation_count == 5

            # Check annotation details
            cursor.execute(
                """
                SELECT annotation_id, verse_id, type, level, confidence
                FROM annotations 
                WHERE annotation_id = 'ann_0'
            """
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "GEN.1.1"
            assert row[2] == "topic"
            assert row[3] == "verse"
            assert row[4] == 0.8

            # Check topics
            cursor.execute(
                """
                SELECT topic_id, topic_name
                FROM annotation_topics 
                WHERE annotation_id = 'ann_0'
            """
            )
            topics = cursor.fetchall()
            assert len(topics) == 1
            assert topics[0][0] == "topic_0"

    @pytest.mark.asyncio
    async def test_timeline_export(self, exporter, sample_dataset):
        """Test timeline event export."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check events
            cursor.execute("SELECT COUNT(*) FROM timeline_events")
            event_count = cursor.fetchone()[0]
            assert event_count == 3

            # Check event details
            cursor.execute(
                """
                SELECT event_id, name, description, event_type, certainty_level
                FROM timeline_events 
                WHERE event_id = 'event_0'
            """
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "Event 0"
            assert row[3] == "historical"
            assert row[4] == "high"

    @pytest.mark.asyncio
    async def test_fts5_indexing(self, exporter, sample_dataset):
        """Test FTS5 full-text search indexing."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check if FTS5 table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE '%_fts'
            """
            )
            fts_tables = cursor.fetchall()
            assert len(fts_tables) > 0

            # Test FTS5 search
            try:
                cursor.execute(
                    """
                    SELECT verse_id FROM verses_fts 
                    WHERE text MATCH 'verse' 
                    LIMIT 5
                """
                )
                results = cursor.fetchall()
                assert len(results) > 0
            except sqlite3.OperationalError:
                # FTS5 might not be available in test environment
                pytest.skip("FTS5 not available in test environment")

    @pytest.mark.asyncio
    async def test_wal_mode_configuration(self, exporter, sample_dataset):
        """Test WAL mode configuration."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check journal mode
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode.upper() == "WAL"

    @pytest.mark.asyncio
    async def test_database_optimization(self, exporter, sample_dataset):
        """Test database optimization (VACUUM, ANALYZE)."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check that database file exists and has reasonable size
        db_path = Path(exporter.config.output_path)
        assert db_path.exists()
        assert db_path.stat().st_size > 0

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            assert integrity_result == "ok"

    @pytest.mark.asyncio
    async def test_batch_processing(self, exporter):
        """Test batch processing with large datasets."""
        # Create large dataset
        large_verses = []
        for i in range(1000):
            verse_id = VerseID("GEN", i // 100 + 1, i % 100 + 1)
            verse = UnifiedVerse(
                verse_id=verse_id, translations={"ESV": f"Large dataset verse {i}"}
            )
            large_verses.append(verse)

        large_dataset = CanonicalDataset(
            verses=iter(large_verses), metadata={"format": "test", "version": "1.0"}
        )

        result = await exporter.export(large_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM verses")
            count = cursor.fetchone()[0]
            assert count == 1000

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling during export."""
        # Create exporter with invalid path
        config.output_path = "/invalid/path/database.db"
        exporter = SQLiteExporter(config)

        sample_dataset = CanonicalDataset(verses=iter([]), metadata={"format": "test"})

        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_cross_references(self, exporter):
        """Test cross-reference export."""
        # Create dataset with cross-references
        verses = [
            UnifiedVerse(verse_id=VerseID("GEN", 1, 1), translations={"ESV": "In the beginning"})
        ]

        cross_refs = [
            {
                "source_verse_id": "GEN.1.1",
                "target_verse_id": "JOH.1.1",
                "type": "parallel",
                "confidence": 0.9,
            }
        ]

        dataset = CanonicalDataset(
            verses=iter(verses), cross_references=iter(cross_refs), metadata={"format": "test"}
        )

        result = await exporter.export(dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT source_verse_id, target_verse_id, type, confidence
                FROM cross_references
            """
            )
            refs = cursor.fetchall()
            assert len(refs) == 1
            assert refs[0][0] == "GEN.1.1"
            assert refs[0][1] == "JOH.1.1"
            assert refs[0][2] == "parallel"

    @pytest.mark.asyncio
    async def test_output_validation(self, exporter, sample_dataset):
        """Test export output validation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        validation = await exporter.validate_output(result)
        assert validation.is_valid
        assert len(validation.errors) == 0

    @pytest.mark.asyncio
    async def test_compression_feature(self, config, sample_dataset):
        """Test text compression feature for large text fields."""
        config.compress_large_text = True
        config.compression_threshold = 100  # Low threshold for testing

        exporter = SQLiteExporter(config)
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Verify compression was applied where appropriate
        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM verse_translations LIMIT 1")
            text = cursor.fetchone()[0]

            # Text should either be original or compressed
            # (In a real implementation, you'd check compression metadata)
            assert text is not None

    def test_export_statistics(self, exporter, sample_dataset):
        """Test export statistics tracking."""
        # This would test statistics collection during export
        # In a real implementation, you'd verify various metrics
        assert exporter.stats is not None

    @pytest.mark.asyncio
    async def test_metadata_export(self, exporter, sample_dataset):
        """Test export metadata storage."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        with sqlite3.connect(exporter.config.output_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT format, version, exported_at, exporter
                FROM export_metadata
            """
            )
            metadata = cursor.fetchone()
            assert metadata is not None
            assert metadata[0] == "sqlite"
            assert metadata[1] == "1.0"
            assert metadata[3] == "ABBA SQLite Exporter"


if __name__ == "__main__":
    pytest.main([__file__])
