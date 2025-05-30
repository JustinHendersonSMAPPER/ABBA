"""
Tests for ABBA export pipeline orchestration.

Test coverage for pipeline configuration, multi-format execution,
parallel processing, validation, and error handling.
"""

import pytest
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from abba.export.pipeline import (
    ExportPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineError,
    PipelineStatistics,
    ExecutionMode,
)
from abba.export.base import (
    ExportFormat,
    ExportResult,
    ExportStatus,
    ExportError,
    ValidationResult,
    CanonicalDataset,
    ExportConfig,
)
from abba.alignment.unified_reference import UnifiedVerse
from abba.verse_id import VerseID


class MockExporter:
    """Mock exporter for testing pipeline."""

    def __init__(self, format_type: ExportFormat, should_fail: bool = False):
        self.format_type = format_type
        self.should_fail = should_fail
        self.export_called = False
        self.validate_called = False

    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Mock export method."""
        self.export_called = True

        if self.should_fail:
            return ExportResult(
                format_type=self.format_type,
                status=ExportStatus.FAILED,
                error=ExportError(f"Mock {self.format_type.value} export failed"),
            )

        return ExportResult(
            format_type=self.format_type,
            status=ExportStatus.COMPLETED,
            output_path=f"/tmp/mock_{self.format_type.value}",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    async def validate_output(self, result: ExportResult) -> ValidationResult:
        """Mock validation method."""
        self.validate_called = True
        validation = ValidationResult(is_valid=True)

        if self.should_fail:
            validation.add_error("Mock validation error")

        return validation


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_basic_config_creation(self):
        """Test basic pipeline configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PipelineConfig(
                pipeline_name="Test Pipeline",
                output_base_path=temp_dir,
                formats=[ExportFormat.SQLITE, ExportFormat.STATIC_JSON],
                format_configs={
                    ExportFormat.SQLITE: {"batch_size": 1000},
                    ExportFormat.STATIC_JSON: {"chunk_size": 100},
                },
            )

            assert config.pipeline_name == "Test Pipeline"
            assert config.output_base_path == temp_dir
            assert len(config.formats) == 2
            assert ExportFormat.SQLITE in config.formats
            assert ExportFormat.STATIC_JSON in config.formats

    def test_config_validation(self):
        """Test pipeline configuration validation."""
        # Valid config
        config = PipelineConfig(
            pipeline_name="Valid Pipeline",
            output_base_path="/tmp/test",
            formats=[ExportFormat.SQLITE],
        )

        errors = config.validate()
        assert len(errors) == 0

        # Invalid config - no formats
        config = PipelineConfig(
            pipeline_name="Invalid Pipeline", output_base_path="/tmp/test", formats=[]
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("format" in error.lower() for error in errors)

    def test_config_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "pipeline_name": "Test Pipeline",
            "output_base_path": "/tmp/test",
            "formats": ["sqlite", "static_json"],
            "max_parallel_exports": 2,
            "validate_all_outputs": True,
            "format_configs": {"sqlite": {"batch_size": 1000}, "static_json": {"chunk_size": 100}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(config_data, f)
            config_file = f.name

        try:
            config = PipelineConfig.from_file(config_file)
            assert config.pipeline_name == "Test Pipeline"
            assert len(config.formats) == 2
            assert config.max_parallel_exports == 2
        finally:
            Path(config_file).unlink()

    def test_output_path_generation(self):
        """Test output path generation for formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PipelineConfig(
                pipeline_name="Test Pipeline",
                output_base_path=temp_dir,
                formats=[ExportFormat.SQLITE, ExportFormat.STATIC_JSON],
            )

            sqlite_path = config.get_output_path(ExportFormat.SQLITE)
            json_path = config.get_output_path(ExportFormat.STATIC_JSON)

            assert sqlite_path.endswith(".db")
            assert json_path != sqlite_path
            assert temp_dir in sqlite_path
            assert temp_dir in json_path


class TestPipelineResult:
    """Test pipeline result tracking."""

    def test_successful_pipeline_result(self):
        """Test successful pipeline result."""
        export_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/test.db",
            ),
            ExportResult(
                format_type=ExportFormat.STATIC_JSON,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/json_output",
            ),
        ]

        result = PipelineResult(
            pipeline_name="Test Pipeline",
            status=ExportStatus.COMPLETED,
            export_results=export_results,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        assert result.is_successful
        assert len(result.export_results) == 2
        assert result.duration is not None

    def test_failed_pipeline_result(self):
        """Test failed pipeline result."""
        export_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/test.db",
            ),
            ExportResult(
                format_type=ExportFormat.OPENSEARCH,
                status=ExportStatus.FAILED,
                error=ExportError("Connection failed"),
            ),
        ]

        result = PipelineResult(
            pipeline_name="Test Pipeline",
            status=ExportStatus.FAILED,
            export_results=export_results,
            error=PipelineError("One or more exports failed"),
        )

        assert not result.is_successful
        assert result.error is not None
        assert len(result.get_failed_exports()) == 1
        assert len(result.get_successful_exports()) == 1

    def test_result_statistics(self):
        """Test pipeline result statistics."""
        result = PipelineResult(
            pipeline_name="Test Pipeline", status=ExportStatus.COMPLETED, export_results=[]
        )

        stats = result.get_statistics()
        assert isinstance(stats, PipelineStatistics)
        assert stats.total_formats == 0
        assert stats.successful_formats == 0
        assert stats.failed_formats == 0


class TestExportPipeline:
    """Test export pipeline orchestration."""

    @pytest.fixture
    def sample_config(self):
        """Create sample pipeline configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield PipelineConfig(
                pipeline_name="Test Pipeline",
                output_base_path=temp_dir,
                formats=[ExportFormat.SQLITE, ExportFormat.STATIC_JSON],
                max_parallel_exports=2,
                validate_all_outputs=True,
                fail_fast=False,
            )

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

    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization."""
        pipeline = ExportPipeline(sample_config)
        assert pipeline.config == sample_config
        assert pipeline.logger is not None

    @pytest.mark.asyncio
    async def test_successful_parallel_execution(self, sample_config, sample_dataset):
        """Test successful parallel pipeline execution."""
        pipeline = ExportPipeline(sample_config)

        # Mock the exporter factory
        def mock_create_exporter(format_type, config):
            return MockExporter(format_type, should_fail=False)

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert result.is_successful
        assert len(result.export_results) == 2
        assert all(r.status == ExportStatus.COMPLETED for r in result.export_results)

    @pytest.mark.asyncio
    async def test_sequential_execution(self, sample_config, sample_dataset):
        """Test sequential pipeline execution."""
        sample_config.execution_mode = ExecutionMode.SEQUENTIAL
        pipeline = ExportPipeline(sample_config)

        def mock_create_exporter(format_type, config):
            return MockExporter(format_type, should_fail=False)

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert result.is_successful
        assert len(result.export_results) == 2

    @pytest.mark.asyncio
    async def test_fail_fast_behavior(self, sample_config, sample_dataset):
        """Test fail-fast behavior when enabled."""
        sample_config.fail_fast = True
        pipeline = ExportPipeline(sample_config)

        def mock_create_exporter(format_type, config):
            # Make SQLite fail, JSON succeed
            should_fail = format_type == ExportFormat.SQLITE
            return MockExporter(format_type, should_fail=should_fail)

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert not result.is_successful
        # With fail-fast, might not attempt all exports
        failed_exports = result.get_failed_exports()
        assert len(failed_exports) >= 1

    @pytest.mark.asyncio
    async def test_continue_on_failure(self, sample_config, sample_dataset):
        """Test continuing execution when fail-fast is disabled."""
        sample_config.fail_fast = False
        pipeline = ExportPipeline(sample_config)

        def mock_create_exporter(format_type, config):
            # Make SQLite fail, JSON succeed
            should_fail = format_type == ExportFormat.SQLITE
            return MockExporter(format_type, should_fail=should_fail)

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert not result.is_successful
        assert len(result.export_results) == 2  # Both should be attempted
        assert len(result.get_failed_exports()) == 1
        assert len(result.get_successful_exports()) == 1

    @pytest.mark.asyncio
    async def test_validation_execution(self, sample_config, sample_dataset):
        """Test validation execution."""
        sample_config.validate_all_outputs = True
        pipeline = ExportPipeline(sample_config)

        def mock_create_exporter(format_type, config):
            return MockExporter(format_type, should_fail=False)

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert result.is_successful
        # Validation should have been called
        # (In a real implementation, you'd check validation results)

    @pytest.mark.asyncio
    async def test_pipeline_cancellation(self, sample_config, sample_dataset):
        """Test pipeline cancellation."""
        pipeline = ExportPipeline(sample_config)

        # Mock slow exporter
        async def slow_export(data):
            await asyncio.sleep(10)  # Very slow operation
            return ExportResult(format_type=ExportFormat.SQLITE, status=ExportStatus.COMPLETED)

        def mock_create_exporter(format_type, config):
            exporter = MockExporter(format_type)
            exporter.export = slow_export
            return exporter

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            # Start pipeline and cancel it
            task = asyncio.create_task(pipeline.run(sample_dataset))
            await asyncio.sleep(0.1)  # Let it start
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_progress_tracking(self, sample_config, sample_dataset):
        """Test progress tracking during pipeline execution."""
        pipeline = ExportPipeline(sample_config)
        progress_updates = []

        def progress_callback(current, total, stage):
            progress_updates.append((current, total, stage))

        pipeline.set_progress_callback(progress_callback)

        def mock_create_exporter(format_type, config):
            return MockExporter(format_type, should_fail=False)

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert result.is_successful
        assert len(progress_updates) > 0

    def test_exporter_factory(self, sample_config):
        """Test exporter factory functionality."""
        pipeline = ExportPipeline(sample_config)

        # Test creation of different exporter types
        config_dict = {"batch_size": 1000}

        # This would test the actual factory method
        # In a real implementation, you'd verify correct exporter types are created
        with patch("abba.export.sqlite_exporter.SQLiteExporter") as mock_sqlite:
            with patch("abba.export.json_exporter.StaticJSONExporter") as mock_json:
                pipeline._create_exporter(ExportFormat.SQLITE, config_dict)
                pipeline._create_exporter(ExportFormat.STATIC_JSON, config_dict)

                mock_sqlite.assert_called_once()
                mock_json.assert_called_once()

    def test_output_path_management(self, sample_config):
        """Test output path management."""
        pipeline = ExportPipeline(sample_config)

        # Test path creation for different formats
        sqlite_path = pipeline._get_format_output_path(ExportFormat.SQLITE)
        json_path = pipeline._get_format_output_path(ExportFormat.STATIC_JSON)

        assert sqlite_path != json_path
        assert sample_config.output_base_path in sqlite_path
        assert sample_config.output_base_path in json_path

    @pytest.mark.asyncio
    async def test_error_aggregation(self, sample_config, sample_dataset):
        """Test error aggregation across multiple failed exports."""
        sample_config.fail_fast = False
        pipeline = ExportPipeline(sample_config)

        def mock_create_exporter(format_type, config):
            return MockExporter(format_type, should_fail=True)  # All fail

        with patch.object(pipeline, "_create_exporter", side_effect=mock_create_exporter):
            result = await pipeline.run(sample_dataset)

        assert not result.is_successful
        assert len(result.get_failed_exports()) == 2
        assert result.error is not None

        # Error should aggregate information from all failures
        error_message = str(result.error)
        assert "sqlite" in error_message.lower()
        assert "json" in error_message.lower()

    def test_pipeline_statistics(self, sample_config):
        """Test pipeline statistics calculation."""
        export_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
            ),
            ExportResult(
                format_type=ExportFormat.STATIC_JSON,
                status=ExportStatus.FAILED,
                error=ExportError("Failed"),
            ),
        ]

        result = PipelineResult(
            pipeline_name="Test",
            status=ExportStatus.COMPLETED,
            export_results=export_results,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        stats = result.get_statistics()
        assert stats.total_formats == 2
        assert stats.successful_formats == 1
        assert stats.failed_formats == 1
        assert 0 < stats.success_rate < 1


class TestPipelineError:
    """Test pipeline error handling."""

    def test_basic_pipeline_error(self):
        """Test basic pipeline error."""
        error = PipelineError("Pipeline failed")
        assert error.message == "Pipeline failed"
        assert error.failed_formats == []

    def test_pipeline_error_with_failed_formats(self):
        """Test pipeline error with failed format information."""
        failed_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.FAILED,
                error=ExportError("Database error"),
            )
        ]

        error = PipelineError(
            message="Export pipeline failed",
            failed_formats=[ExportFormat.SQLITE],
            failed_results=failed_results,
        )

        assert error.message == "Export pipeline failed"
        assert ExportFormat.SQLITE in error.failed_formats
        assert len(error.failed_results) == 1

    def test_error_aggregation(self):
        """Test error message aggregation."""
        failed_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.FAILED,
                error=ExportError("Database connection failed"),
            ),
            ExportResult(
                format_type=ExportFormat.OPENSEARCH,
                status=ExportStatus.FAILED,
                error=ExportError("Index creation failed"),
            ),
        ]

        error = PipelineError.from_failed_results(failed_results)

        assert "sqlite" in error.message.lower()
        assert "opensearch" in error.message.lower()
        assert len(error.failed_formats) == 2


class TestExecutionMode:
    """Test execution mode functionality."""

    def test_execution_modes(self):
        """Test execution mode enumeration."""
        assert ExecutionMode.PARALLEL == "parallel"
        assert ExecutionMode.SEQUENTIAL == "sequential"

        # Test all modes are defined
        modes = list(ExecutionMode)
        assert len(modes) == 2
        assert ExecutionMode.PARALLEL in modes
        assert ExecutionMode.SEQUENTIAL in modes


if __name__ == "__main__":
    pytest.main([__file__])
