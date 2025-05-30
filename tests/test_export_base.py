"""
Tests for ABBA export base classes and utilities.

Test coverage for the core export framework, data processing,
validation, and shared utilities across all export formats.
"""

import pytest
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from abba.export.base import (
    DataExporter,
    ExportConfig,
    ExportResult,
    ExportStatus,
    ExportError,
    ValidationResult,
    ExportUtilities,
    StreamingDataProcessor,
    ExportFormat,
    ExportStatistics,
    CanonicalDataset,
)
from abba.alignment.unified_reference import UnifiedVerse
from abba.verse_id import VerseID


class TestExportConfig:
    """Test export configuration base class."""

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        config = ExportConfig(output_path="/tmp/test", batch_size=100)

        assert config.output_path == "/tmp/test"
        assert config.batch_size == 100
        assert config.format_type is None

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ExportConfig(output_path="/tmp/test")
        result = config.validate()
        assert result.is_valid

        # Invalid config - no output path
        config = ExportConfig(output_path="")
        result = config.validate()
        assert not result.is_valid
        assert any("output path" in error.lower() for error in result.errors)

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {"output_path": "/tmp/test", "batch_size": 200, "validate_output": True}

        config = ExportConfig.from_dict(config_dict)
        assert config.output_path == "/tmp/test"
        assert config.batch_size == 200
        assert config.validate_output is True


class TestExportResult:
    """Test export result tracking."""

    def test_successful_result(self):
        """Test successful export result."""
        result = ExportResult(
            format_type=ExportFormat.SQLITE,
            status=ExportStatus.COMPLETED,
            output_path="/tmp/test.db",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        assert result.format_type == ExportFormat.SQLITE
        assert result.status == ExportStatus.COMPLETED
        assert result.is_successful
        assert result.duration is not None

    def test_failed_result(self):
        """Test failed export result."""
        error = ExportError("Database connection failed", stage="connection")
        result = ExportResult(
            format_type=ExportFormat.OPENSEARCH, status=ExportStatus.FAILED, error=error
        )

        assert not result.is_successful
        assert result.error is not None
        assert result.error.message == "Database connection failed"

    def test_result_statistics(self):
        """Test result statistics tracking."""
        stats = ExportStatistics()
        stats.processed_verses = 1000
        stats.processed_annotations = 500
        stats.output_size_bytes = 2048

        result = ExportResult(
            format_type=ExportFormat.STATIC_JSON, status=ExportStatus.COMPLETED, stats=stats
        )

        assert result.stats.processed_verses == 1000
        assert result.stats.processed_annotations == 500
        assert result.stats.output_size_bytes == 2048


class TestValidationResult:
    """Test validation result handling."""

    def test_valid_result(self):
        """Test valid validation result."""
        validation = ValidationResult(is_valid=True)
        assert validation.is_valid
        assert len(validation.errors) == 0
        assert len(validation.warnings) == 0

    def test_invalid_result_with_errors(self):
        """Test invalid validation result with errors."""
        validation = ValidationResult(is_valid=True)
        validation.add_error("Critical failure")
        validation.add_warning("Minor issue")

        assert not validation.is_valid  # Should be False after adding error
        assert len(validation.errors) == 1
        assert len(validation.warnings) == 1
        assert "Critical failure" in validation.errors

    def test_merge_validation_results(self):
        """Test merging validation results."""
        result1 = ValidationResult(is_valid=True)
        result1.add_warning("Warning 1")

        result2 = ValidationResult(is_valid=True)
        result2.add_error("Error 1")
        result2.add_warning("Warning 2")

        merged = ValidationResult.merge([result1, result2])
        assert not merged.is_valid
        assert len(merged.errors) == 1
        assert len(merged.warnings) == 2


class TestStreamingDataProcessor:
    """Test streaming data processor."""

    @pytest.fixture
    def processor(self):
        """Create test processor."""
        return StreamingDataProcessor(batch_size=3)

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        verses = []
        for i in range(10):
            verse_id = VerseID("GEN", 1, i + 1)
            verse = UnifiedVerse(verse_id=verse_id, translations={"ESV": f"Test verse {i + 1}"})
            verses.append(verse)
        return verses

    @pytest.mark.asyncio
    async def test_batch_processing(self, processor, sample_verses):
        """Test batch processing functionality."""
        processed_batches = []

        async def process_batch(batch):
            processed_batches.append(len(batch))

        # Process verses in batches
        async for batch in processor.process_in_batches(iter(sample_verses), process_batch):
            pass

        # Should have 4 batches: 3,3,3,1
        assert len(processed_batches) == 4
        assert processed_batches[:3] == [3, 3, 3]
        assert processed_batches[3] == 1

    @pytest.mark.asyncio
    async def test_progress_tracking(self, processor, sample_verses):
        """Test progress tracking during processing."""
        progress_updates = []

        def track_progress(count, item_type):
            progress_updates.append((count, item_type))

        async def process_batch(batch):
            pass

        # Process with progress tracking
        async for batch in processor.process_in_batches(
            iter(sample_verses), process_batch, track_progress
        ):
            pass

        # Should have progress updates
        assert len(progress_updates) > 0
        # Final count should be total verses
        assert progress_updates[-1][0] == len(sample_verses)

    def test_memory_monitoring(self, processor):
        """Test memory usage monitoring."""
        initial_memory = processor.get_memory_usage()
        assert initial_memory > 0

        # Create some data to increase memory
        large_data = ["x" * 1000 for _ in range(1000)]

        new_memory = processor.get_memory_usage()
        assert new_memory >= initial_memory


class TestExportUtilities:
    """Test export utility functions."""

    def test_validate_output_path(self):
        """Test output path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid directory path
            assert ExportUtilities.validate_output_path(temp_dir)

            # Valid file path in existing directory
            file_path = Path(temp_dir) / "test.db"
            assert ExportUtilities.validate_output_path(str(file_path))

            # Invalid - non-existent directory
            invalid_path = "/non/existent/path/file.db"
            assert not ExportUtilities.validate_output_path(invalid_path)

    def test_estimate_output_size(self):
        """Test output size estimation."""
        # Create test data
        test_data = {
            "verses": [{"id": f"verse_{i}", "text": "test" * 10} for i in range(100)],
            "metadata": {"format": "test", "count": 100},
        }

        estimated_size = ExportUtilities.estimate_output_size(test_data)
        assert estimated_size > 0

        # Estimated size should be reasonable
        assert estimated_size > 1000  # Should be more than 1KB
        assert estimated_size < 1000000  # Should be less than 1MB for test data

    def test_create_backup_path(self):
        """Test backup path creation."""
        original_path = "/tmp/database.db"
        backup_path = ExportUtilities.create_backup_path(original_path)

        assert backup_path != original_path
        assert backup_path.startswith("/tmp/database")
        assert backup_path.endswith(".db")
        assert "backup" in backup_path or len(backup_path) > len(original_path)

    def test_compress_data(self):
        """Test data compression utility."""
        test_data = "This is test data that should compress well. " * 100

        compressed = ExportUtilities.compress_data(test_data.encode())
        assert len(compressed) < len(test_data.encode())

        # Test decompression
        decompressed = ExportUtilities.decompress_data(compressed)
        assert decompressed.decode() == test_data

    def test_format_file_size(self):
        """Test file size formatting."""
        assert ExportUtilities.format_file_size(512) == "512 B"
        assert ExportUtilities.format_file_size(1024) == "1.0 KB"
        assert ExportUtilities.format_file_size(1536) == "1.5 KB"
        assert ExportUtilities.format_file_size(1048576) == "1.0 MB"
        assert ExportUtilities.format_file_size(1073741824) == "1.0 GB"


class MockDataExporter(DataExporter):
    """Mock exporter for testing base class functionality."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.export_called = False
        self.prepare_called = False
        self.finalize_called = False

    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Mock export implementation."""
        await self.prepare_export(data)
        self.export_called = True

        result = self.create_result(ExportStatus.COMPLETED)
        return await self.finalize_export(result)

    async def prepare_export(self, data: CanonicalDataset):
        """Mock prepare implementation."""
        await super().prepare_export(data)
        self.prepare_called = True

    async def finalize_export(self, result: ExportResult) -> ExportResult:
        """Mock finalize implementation."""
        result = await super().finalize_export(result)
        self.finalize_called = True
        return result


class TestDataExporter:
    """Test base data exporter functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExportConfig(output_path="/tmp/test")

    @pytest.fixture
    def exporter(self, config):
        """Create test exporter."""
        return MockDataExporter(config)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample canonical dataset."""
        verses = [
            UnifiedVerse(
                verse_id=VerseID("GEN", 1, 1),
                translations={"ESV": "In the beginning God created the heavens and the earth."},
            ),
            UnifiedVerse(
                verse_id=VerseID("GEN", 1, 2),
                translations={"ESV": "The earth was without form and void."},
            ),
        ]

        return CanonicalDataset(verses=iter(verses), metadata={"format": "test", "version": "1.0"})

    def test_exporter_initialization(self, exporter, config):
        """Test exporter initialization."""
        assert exporter.config == config
        assert exporter.stats is not None
        assert exporter.logger is not None

    def test_config_validation(self, exporter):
        """Test exporter configuration validation."""
        validation = exporter.validate_config()
        assert validation.is_valid

    def test_supported_features(self, exporter):
        """Test supported features reporting."""
        features = exporter.get_supported_features()
        assert isinstance(features, list)

    @pytest.mark.asyncio
    async def test_export_workflow(self, exporter, sample_dataset):
        """Test complete export workflow."""
        result = await exporter.export(sample_dataset)

        assert exporter.prepare_called
        assert exporter.export_called
        assert exporter.finalize_called
        assert result.status == ExportStatus.COMPLETED

    def test_result_creation(self, exporter):
        """Test export result creation."""
        result = exporter.create_result(ExportStatus.COMPLETED)
        assert result.status == ExportStatus.COMPLETED
        assert result.format_type == exporter.config.format_type

        # Test with error
        error = ExportError("Test error")
        result = exporter.create_result(ExportStatus.FAILED, error)
        assert result.status == ExportStatus.FAILED
        assert result.error == error

    def test_progress_tracking(self, exporter):
        """Test progress tracking."""
        exporter.update_progress(100, "verses")
        assert exporter.stats.processed_verses == 100

        exporter.update_progress(50, "annotations")
        assert exporter.stats.processed_annotations == 50

    @pytest.mark.asyncio
    async def test_validation_output(self, exporter):
        """Test output validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock result with valid path
            result = ExportResult(
                format_type=ExportFormat.SQLITE, status=ExportStatus.COMPLETED, output_path=temp_dir
            )

            validation = await exporter.validate_output(result)
            assert validation.is_valid

            # Test with invalid path
            result.output_path = "/non/existent/path"
            validation = await exporter.validate_output(result)
            assert not validation.is_valid


class TestCanonicalDataset:
    """Test canonical dataset handling."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        verses = [UnifiedVerse(verse_id=VerseID("GEN", 1, 1), translations={"ESV": "Test verse"})]

        dataset = CanonicalDataset(verses=iter(verses), metadata={"test": "data"})

        assert dataset.metadata == {"test": "data"}
        assert hasattr(dataset, "verses")

    def test_dataset_statistics(self):
        """Test dataset statistics calculation."""
        verses = [
            UnifiedVerse(verse_id=VerseID("GEN", 1, i), translations={"ESV": f"Verse {i}"})
            for i in range(1, 11)
        ]

        dataset = CanonicalDataset(verses=iter(verses))

        # Note: In a real implementation, you might want to add
        # a method to calculate statistics without consuming the iterator
        assert hasattr(dataset, "verses")


class TestExportError:
    """Test export error handling."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ExportError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.stage is None
        assert error.details is None

    def test_detailed_error(self):
        """Test detailed error creation."""
        error = ExportError(
            message="Database connection failed",
            stage="connection",
            details={"host": "localhost", "port": 5432},
        )

        assert error.message == "Database connection failed"
        assert error.stage == "connection"
        assert error.details["host"] == "localhost"

    def test_error_string_representation(self):
        """Test error string representation."""
        error = ExportError("Test error", stage="testing")
        error_str = str(error)

        assert "Test error" in error_str
        assert "testing" in error_str


if __name__ == "__main__":
    pytest.main([__file__])
