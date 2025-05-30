"""
Tests for ABBA static JSON exporter.

Test coverage for hierarchical JSON file generation, progressive loading,
CDN optimization, and static website deployment features.
"""

import pytest
import asyncio
import json
import gzip
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from abba.export.json_exporter import StaticJSONExporter, JSONConfig
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


class TestJSONConfig:
    """Test JSON exporter configuration."""

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = JSONConfig(
                output_path=temp_dir, chunk_size=50, gzip_output=True, minify_json=True
            )

            assert config.output_path == temp_dir
            assert config.chunk_size == 50
            assert config.gzip_output is True
            assert config.minify_json is True
            assert config.format_type == ExportFormat.STATIC_JSON

    def test_progressive_loading_config(self):
        """Test progressive loading configuration."""
        config = JSONConfig(
            output_path="/tmp/json",
            enable_progressive_loading=True,
            manifest_version="2.0",
            create_search_indices=True,
        )

        assert config.enable_progressive_loading is True
        assert config.manifest_version == "2.0"
        assert config.create_search_indices is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = JSONConfig(output_path="/tmp/test")
        validation = config.validate()
        assert validation.is_valid

        # Invalid config - no output path
        config = JSONConfig(output_path="")
        validation = config.validate()
        assert not validation.is_valid

        # Invalid config - invalid chunk size
        config = JSONConfig(output_path="/tmp/test", chunk_size=0)
        validation = config.validate()
        assert not validation.is_valid


class TestStaticJSONExporter:
    """Test static JSON exporter functionality."""

    @pytest.fixture
    def config(self):
        """Create test JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield JSONConfig(
                output_path=temp_dir,
                chunk_size=5,  # Small for testing
                gzip_output=True,
                minify_json=True,
                enable_progressive_loading=True,
                create_search_indices=True,
                create_inverted_index=True,
            )

    @pytest.fixture
    def exporter(self, config):
        """Create test JSON exporter."""
        return StaticJSONExporter(config)

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        verses = []
        books = ["GEN", "EXO", "MAT"]

        for book_idx, book in enumerate(books):
            for chapter in range(1, 3):  # 2 chapters per book
                for verse in range(1, 6):  # 5 verses per chapter
                    verse_id = VerseID(book, chapter, verse)
                    verse_obj = UnifiedVerse(
                        verse_id=verse_id,
                        translations={
                            "ESV": f"This is {book} chapter {chapter} verse {verse} in ESV.",
                            "NIV": f"This is {book} chapter {chapter} verse {verse} in NIV.",
                        },
                        metadata={"source": "test"},
                    )
                    verses.append(verse_obj)

        return verses

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotations."""
        annotations = []
        for i in range(10):
            annotation = Annotation(
                id=f"ann_{i}",
                verse_id=VerseID("GEN", 1, (i % 5) + 1),
                annotation_type=AnnotationType.TOPIC,
                level=AnnotationLevel.VERSE,
                confidence=0.8,
                topics=[Topic(id=f"topic_{i}", name=f"Topic {i}")],
            )
            annotations.append(annotation)
        return annotations

    @pytest.fixture
    def sample_cross_references(self):
        """Create sample cross-references."""
        cross_refs = []
        for i in range(5):
            cross_refs.append(
                {
                    "source_verse_id": f"GEN.1.{i + 1}",
                    "target_verse_id": f"MAT.1.{i + 1}",
                    "type": "parallel",
                    "confidence": 0.9,
                }
            )
        return cross_refs

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
                verse_refs=[VerseID("GEN", 1, i + 1)],
            )
            events.append(event)
        return events

    @pytest.fixture
    def sample_dataset(
        self, sample_verses, sample_annotations, sample_cross_references, sample_events
    ):
        """Create sample canonical dataset."""
        return CanonicalDataset(
            verses=iter(sample_verses),
            annotations=iter(sample_annotations),
            cross_references=iter(sample_cross_references),
            timeline_events=iter(sample_events),
            metadata={"format": "test", "version": "1.0"},
        )

    def test_exporter_initialization(self, exporter, config):
        """Test exporter initialization."""
        assert exporter.config == config
        assert exporter.output_dir == Path(config.output_path)
        assert exporter.api_dir == exporter.output_dir / "api" / "v1"
        assert exporter.books_dir == exporter.api_dir / "books"

    def test_config_validation(self, exporter):
        """Test configuration validation."""
        validation = exporter.validate_config()
        assert validation.is_valid

    def test_supported_features(self, exporter):
        """Test supported features."""
        features = exporter.get_supported_features()
        assert "progressive_loading" in features
        assert "cdn_optimized" in features
        assert "client_side_search" in features
        assert "static_hosting" in features
        assert "mobile_friendly" in features

    @pytest.mark.asyncio
    async def test_directory_structure_creation(self, exporter, sample_dataset):
        """Test directory structure creation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check main directories
        assert exporter.api_dir.exists()
        assert exporter.meta_dir.exists()
        assert exporter.books_dir.exists()
        assert exporter.search_dir.exists()
        assert exporter.timeline_dir.exists()

        # Check book directories
        expected_books = ["GEN", "EXO", "MAT"]
        for book in expected_books:
            book_dir = exporter.books_dir / book
            assert book_dir.exists()
            assert (book_dir / "chapters").exists()

    @pytest.mark.asyncio
    async def test_metadata_file_generation(self, exporter, sample_dataset):
        """Test metadata file generation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check books metadata
        books_meta_file = exporter.meta_dir / "books.json.gz"
        assert books_meta_file.exists()

        with gzip.open(books_meta_file, "rt") as f:
            books_meta = json.load(f)

        assert "books" in books_meta
        assert "total_books" in books_meta
        assert "total_verses" in books_meta
        assert books_meta["total_books"] == 3

        # Check export metadata
        export_meta_file = exporter.meta_dir / "export.json.gz"
        assert export_meta_file.exists()

        with gzip.open(export_meta_file, "rt") as f:
            export_meta = json.load(f)

        assert export_meta["format"] == "static_json"
        assert "configuration" in export_meta

    @pytest.mark.asyncio
    async def test_book_file_generation(self, exporter, sample_dataset):
        """Test book and chapter file generation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check book metadata file
        gen_meta_file = exporter.books_dir / "GEN" / "meta.json.gz"
        assert gen_meta_file.exists()

        with gzip.open(gen_meta_file, "rt") as f:
            gen_meta = json.load(f)

        assert gen_meta["id"] == "GEN"
        assert "chapters" in gen_meta
        assert len(gen_meta["chapters"]) == 2  # 2 chapters

        # Check chapter files
        chapter_dir = exporter.books_dir / "GEN" / "chapters" / "1"
        assert chapter_dir.exists()

        verses_file = chapter_dir / "verses.json.gz"
        assert verses_file.exists()

        with gzip.open(verses_file, "rt") as f:
            chapter_data = json.load(f)

        assert "verses" in chapter_data
        assert len(chapter_data["verses"]) == 5  # 5 verses in chapter 1

    @pytest.mark.asyncio
    async def test_chunked_chapter_generation(self, exporter, sample_dataset):
        """Test chunked chapter file generation for large chapters."""
        # Create a large chapter (more than chunk_size)
        large_verses = []
        for i in range(10):  # More than chunk_size (5)
            verse_id = VerseID("PSA", 119, i + 1)
            verse = UnifiedVerse(
                verse_id=verse_id, translations={"ESV": f"Large chapter verse {i + 1}"}
            )
            large_verses.append(verse)

        large_dataset = CanonicalDataset(verses=iter(large_verses), metadata={"format": "test"})

        result = await exporter.export(large_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check chunked files
        chapter_dir = exporter.books_dir / "PSA" / "chapters" / "119"
        assert chapter_dir.exists()

        # Should have main verses.json file with chunk info
        verses_file = chapter_dir / "verses.json.gz"
        assert verses_file.exists()

        with gzip.open(verses_file, "rt") as f:
            chapter_index = json.load(f)

        assert chapter_index["is_chunked"] is True
        assert chapter_index["chunk_count"] == 2  # 10 verses / 5 chunk_size = 2 chunks

        # Check chunk files exist
        chunk1_file = chapter_dir / "chunk_1.json.gz"
        chunk2_file = chapter_dir / "chunk_2.json.gz"
        assert chunk1_file.exists()
        assert chunk2_file.exists()

    @pytest.mark.asyncio
    async def test_verse_serialization(self, exporter, sample_dataset):
        """Test verse serialization with all data types."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Read a verse file
        verses_file = exporter.books_dir / "GEN" / "chapters" / "1" / "verses.json.gz"
        with gzip.open(verses_file, "rt") as f:
            chapter_data = json.load(f)

        verse = chapter_data["verses"][0]

        # Check basic verse data
        assert verse["verse_id"] == "GEN.1.1"
        assert verse["book"] == "GEN"
        assert verse["chapter"] == 1
        assert verse["verse"] == 1

        # Check translations
        assert "translations" in verse
        assert "ESV" in verse["translations"]
        assert "NIV" in verse["translations"]

        # Check annotations (should be attached)
        if "annotations" in verse:
            assert len(verse["annotations"]) > 0
            annotation = verse["annotations"][0]
            assert "id" in annotation
            assert "type" in annotation
            assert "topics" in annotation

    @pytest.mark.asyncio
    async def test_search_index_generation(self, exporter, sample_dataset):
        """Test search index file generation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check main search index
        index_file = exporter.search_dir / "index.json.gz"
        assert index_file.exists()

        with gzip.open(index_file, "rt") as f:
            search_data = json.load(f)

        assert "books" in search_data
        assert "words" in search_data
        assert "translations" in search_data

        # Check book index
        assert "GEN" in search_data["books"]
        assert len(search_data["books"]["GEN"]) > 0

        # Check word index
        assert len(search_data["words"]) > 0

    @pytest.mark.asyncio
    async def test_timeline_file_generation(self, exporter, sample_dataset):
        """Test timeline file generation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check events file
        events_file = exporter.timeline_dir / "events.json.gz"
        assert events_file.exists()

        with gzip.open(events_file, "rt") as f:
            events_data = json.load(f)

        assert "events" in events_data
        assert "total_events" in events_data
        assert events_data["total_events"] == 3

        # Check combined timeline file
        timeline_file = exporter.timeline_dir / "timeline.json.gz"
        assert timeline_file.exists()

        with gzip.open(timeline_file, "rt") as f:
            timeline_data = json.load(f)

        assert "events" in timeline_data
        assert "metadata" in timeline_data

    @pytest.mark.asyncio
    async def test_progressive_loading_manifest(self, exporter, sample_dataset):
        """Test progressive loading manifest generation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check manifest file
        manifest_file = exporter.output_dir / "manifest.json.gz"
        assert manifest_file.exists()

        with gzip.open(manifest_file, "rt") as f:
            manifest = json.load(f)

        assert manifest["format"] == "static_json"
        assert manifest["progressive_loading"] is True
        assert "files" in manifest
        assert "statistics" in manifest
        assert "api_endpoints" in manifest

        # Check file categorization
        files = manifest["files"]
        categories = set(f["category"] for f in files)
        assert "metadata" in categories
        assert "content" in categories

        # Check file priorities
        priorities = set(f["priority"] for f in files)
        assert "high" in priorities
        assert "medium" in priorities

    @pytest.mark.asyncio
    async def test_compression_and_minification(self, exporter, sample_dataset):
        """Test JSON compression and minification."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check that gzipped files exist
        gzipped_files = list(exporter.output_dir.rglob("*.json.gz"))
        assert len(gzipped_files) > 0

        # Check a file is actually compressed
        test_file = gzipped_files[0]

        # Read compressed content
        with gzip.open(test_file, "rt") as f:
            content = f.read()

        # Should be valid JSON
        data = json.loads(content)
        assert data is not None

        # Should be minified (no unnecessary whitespace)
        assert "\n  " not in content  # No indentation

    @pytest.mark.asyncio
    async def test_cross_reference_attachment(self, exporter, sample_dataset):
        """Test cross-reference attachment to verses."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Read a verse that should have cross-references
        verses_file = exporter.books_dir / "GEN" / "chapters" / "1" / "verses.json.gz"
        with gzip.open(verses_file, "rt") as f:
            chapter_data = json.load(f)

        # Check if any verse has cross-references
        verse_with_refs = None
        for verse in chapter_data["verses"]:
            if "cross_references" in verse:
                verse_with_refs = verse
                break

        if verse_with_refs:
            refs = verse_with_refs["cross_references"]
            assert len(refs) > 0

            ref = refs[0]
            assert "target" in ref
            assert "type" in ref
            assert "confidence" in ref

    @pytest.mark.asyncio
    async def test_word_extraction_and_stemming(self, exporter):
        """Test word extraction and stemming for search."""
        # Test word extraction
        text = "The Lord God created the heavens and earth"
        words = exporter._extract_search_words(text)

        assert "lord" in words
        assert "god" in words
        assert "created" in words or "creat" in words  # Stemmed

        # Should not include stopwords if disabled
        if not exporter.config.index_stopwords:
            assert "the" not in words
            assert "and" not in words

    @pytest.mark.asyncio
    async def test_large_search_index_splitting(self, exporter):
        """Test splitting large search indices into separate files."""
        # Create a dataset that would generate a large word index
        large_verses = []
        for i in range(100):
            verse_id = VerseID("GEN", 1, i + 1)
            # Create verses with many unique words
            unique_words = [f"word{j}" for j in range(i * 10, (i + 1) * 10)]
            text = " ".join(unique_words)
            verse = UnifiedVerse(verse_id=verse_id, translations={"ESV": text})
            large_verses.append(verse)

        large_dataset = CanonicalDataset(verses=iter(large_verses), metadata={"format": "test"})

        result = await exporter.export(large_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check if word index was split
        word_manifest = exporter.search_dir / "words_manifest.json.gz"
        if word_manifest.exists():
            with gzip.open(word_manifest, "rt") as f:
                manifest = json.load(f)

            assert manifest["index_type"] == "alphabetical"
            assert "available_letters" in manifest

            # Check that letter files exist
            for letter in manifest["available_letters"][:3]:  # Check first few
                letter_file = exporter.search_dir / f"words_{letter}.json.gz"
                assert letter_file.exists()

    @pytest.mark.asyncio
    async def test_output_validation(self, exporter, sample_dataset):
        """Test export output validation."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        validation = await exporter.validate_output(result)
        assert validation.is_valid
        assert len(validation.errors) == 0

        # Validation should check directory structure
        assert exporter.api_dir.exists()
        assert exporter.meta_dir.exists()
        assert exporter.books_dir.exists()

    @pytest.mark.asyncio
    async def test_uncompressed_output(self, config, sample_dataset):
        """Test uncompressed JSON output."""
        config.gzip_output = False
        config.minify_json = False

        exporter = StaticJSONExporter(config)
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Check that regular JSON files exist (not gzipped)
        json_files = list(exporter.output_dir.rglob("*.json"))
        assert len(json_files) > 0

        # Should be no .gz files
        gz_files = list(exporter.output_dir.rglob("*.gz"))
        assert len(gz_files) == 0

        # Check content is formatted (not minified)
        test_file = json_files[0]
        with open(test_file, "r") as f:
            content = f.read()

        # Should have indentation
        assert "\n  " in content or "\n    " in content

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling during export."""
        # Test with invalid output path
        config.output_path = "/invalid/readonly/path"
        exporter = StaticJSONExporter(config)

        sample_dataset = CanonicalDataset(verses=iter([]), metadata={"format": "test"})

        # Should handle permission errors gracefully
        result = await exporter.export(sample_dataset)
        if result.status == ExportStatus.FAILED:
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_empty_dataset_handling(self, exporter):
        """Test handling of empty datasets."""
        empty_dataset = CanonicalDataset(verses=iter([]), metadata={"format": "test"})

        result = await exporter.export(empty_dataset)
        assert result.status == ExportStatus.COMPLETED

        # Should still create directory structure and metadata
        assert exporter.api_dir.exists()
        assert exporter.meta_dir.exists()

        # Books metadata should indicate no books
        books_meta_file = exporter.meta_dir / "books.json.gz"
        if books_meta_file.exists():
            with gzip.open(books_meta_file, "rt") as f:
                books_meta = json.load(f)
            assert books_meta["total_books"] == 0

    def test_file_size_calculation(self, exporter):
        """Test file size calculation for statistics."""
        # This would test file size tracking during export
        assert hasattr(exporter, "stats")

    @pytest.mark.asyncio
    async def test_api_endpoint_structure(self, exporter, sample_dataset):
        """Test API endpoint structure in manifest."""
        result = await exporter.export(sample_dataset)
        assert result.status == ExportStatus.COMPLETED

        manifest_file = exporter.output_dir / "manifest.json.gz"
        with gzip.open(manifest_file, "rt") as f:
            manifest = json.load(f)

        api_endpoints = manifest["api_endpoints"]
        assert "books" in api_endpoints
        assert "search" in api_endpoints
        assert "timeline" in api_endpoints

        # Endpoints should be valid paths
        for endpoint in api_endpoints.values():
            assert endpoint.startswith("/api/v1/")


if __name__ == "__main__":
    pytest.main([__file__])
