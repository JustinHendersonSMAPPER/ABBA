"""
Tests for ABBA export validation and quality assurance.

Test coverage for export validation, integrity checking,
performance benchmarking, and cross-format consistency validation.
"""

import pytest
import asyncio
import tempfile
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from abba.export.validation import (
    ExportValidator,
    PerformanceMetrics,
    IntegrityCheckResult,
    IntegrityChecker,
    PerformanceBenchmark,
    SQLiteValidator,
    JSONValidator,
    OpenSearchValidator,
    Neo4jValidator,
    ArangoValidator,
)
from abba.export.base import (
    ExportFormat,
    ExportResult,
    ExportStatus,
    ValidationResult,
    ExportStatistics,
)


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    def test_basic_metrics_creation(self):
        """Test basic metrics creation."""
        metrics = PerformanceMetrics()
        assert metrics.query_times == []
        assert metrics.search_times == []
        assert metrics.storage_size_bytes == 0
        assert metrics.compression_ratio is None

    def test_query_time_tracking(self):
        """Test query time tracking."""
        metrics = PerformanceMetrics()

        # Add some query times
        metrics.add_query_time(0.1)
        metrics.add_query_time(0.2)
        metrics.add_query_time(0.15)

        assert len(metrics.query_times) == 3
        assert metrics.avg_query_time == 150.0  # Average in milliseconds

        # Test percentile calculation
        p95_time = metrics.p95_query_time
        assert p95_time > 0

    def test_search_time_tracking(self):
        """Test search time tracking."""
        metrics = PerformanceMetrics()

        metrics.add_search_time(0.05)
        metrics.add_search_time(0.08)

        assert len(metrics.search_times) == 2

    def test_concurrent_performance_tracking(self):
        """Test concurrent query performance tracking."""
        metrics = PerformanceMetrics()

        metrics.concurrent_query_performance[1] = 100.0  # 100 QPS with 1 thread
        metrics.concurrent_query_performance[4] = 300.0  # 300 QPS with 4 threads

        assert metrics.concurrent_query_performance[1] == 100.0
        assert metrics.concurrent_query_performance[4] == 300.0

    def test_empty_metrics(self):
        """Test metrics with no data."""
        metrics = PerformanceMetrics()

        assert metrics.avg_query_time == 0.0
        assert metrics.p95_query_time == 0.0


class TestIntegrityCheckResult:
    """Test integrity check result handling."""

    def test_basic_result_creation(self):
        """Test basic result creation."""
        result = IntegrityCheckResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_error_handling(self):
        """Test error handling."""
        result = IntegrityCheckResult()

        result.add_error("Critical data mismatch")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Critical data mismatch" in result.errors

    def test_warning_handling(self):
        """Test warning handling."""
        result = IntegrityCheckResult()

        result.add_warning("Minor inconsistency")
        assert result.is_valid  # Warnings don't invalidate
        assert len(result.warnings) == 1
        assert "Minor inconsistency" in result.warnings

    def test_count_validation(self):
        """Test count validation logic."""
        result = IntegrityCheckResult()

        # Set expected count
        result.set_expected_count("verses", 1000)

        # Test exact match
        result.set_actual_count("verses", 1000)
        assert result.is_valid
        assert len(result.errors) == 0

        # Test complete mismatch (0 actual)
        result.set_actual_count("annotations", 0)
        result.set_expected_count("annotations", 500)
        assert not result.is_valid
        assert len(result.errors) == 1

        # Test slight difference (within 5%)
        result.set_expected_count("cross_refs", 100)
        result.set_actual_count("cross_refs", 102)  # 2% difference
        assert len(result.warnings) > 0

    def test_format_consistency_tracking(self):
        """Test format consistency tracking."""
        result = IntegrityCheckResult()

        result.format_consistency["sqlite"] = True
        result.format_consistency["opensearch"] = False

        assert result.format_consistency["sqlite"] is True
        assert result.format_consistency["opensearch"] is False


class TestSQLiteValidator:
    """Test SQLite-specific validation."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary SQLite database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Create basic schema
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE verses (
                    verse_id TEXT PRIMARY KEY,
                    book TEXT,
                    chapter INTEGER,
                    verse INTEGER
                );
                
                CREATE TABLE verse_translations (
                    verse_id TEXT,
                    translation_id TEXT,
                    text TEXT
                );
                
                CREATE TABLE annotations (
                    annotation_id TEXT PRIMARY KEY,
                    verse_id TEXT,
                    type TEXT,
                    confidence REAL
                );
                
                CREATE TABLE timeline_events (
                    event_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT
                );
                
                CREATE TABLE export_metadata (
                    format TEXT,
                    version TEXT,
                    exported_at TEXT
                );
                
                -- Insert test data
                INSERT INTO verses VALUES ('GEN.1.1', 'GEN', 1, 1);
                INSERT INTO verses VALUES ('GEN.1.2', 'GEN', 1, 2);
                
                INSERT INTO verse_translations VALUES ('GEN.1.1', 'ESV', 'In the beginning...');
                INSERT INTO annotations VALUES ('ann_1', 'GEN.1.1', 'topic', 0.8);
                INSERT INTO timeline_events VALUES ('evt_1', 'Creation', 'God creates...');
                INSERT INTO export_metadata VALUES ('sqlite', '1.0', '2024-01-01T00:00:00');
            """
            )

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def sqlite_result(self, temp_db):
        """Create SQLite export result."""
        return ExportResult(
            format_type=ExportFormat.SQLITE,
            status=ExportStatus.COMPLETED,
            output_path=temp_db,
            stats=ExportStatistics(),
        )

    @pytest.mark.asyncio
    async def test_sqlite_validation_success(self, sqlite_result):
        """Test successful SQLite validation."""
        validator = SQLiteValidator()
        validation = await validator.validate(sqlite_result)

        assert validation.is_valid
        assert len(validation.errors) == 0

    @pytest.mark.asyncio
    async def test_sqlite_schema_validation(self, sqlite_result):
        """Test SQLite schema validation."""
        validator = SQLiteValidator()
        validation = await validator._validate_format_specific(
            sqlite_result, ValidationResult(is_valid=True)
        )

        assert validation.is_valid
        # Should not have missing table errors
        missing_table_errors = [
            error for error in validation.errors if "missing tables" in error.lower()
        ]
        assert len(missing_table_errors) == 0

    @pytest.mark.asyncio
    async def test_sqlite_integrity_check(self, temp_db):
        """Test SQLite integrity check."""
        # Corrupt the database slightly (in a way that doesn't break the test)
        # This is a simplified test - in practice you'd test with actual corruption

        validator = SQLiteValidator()
        result = ExportResult(
            format_type=ExportFormat.SQLITE, status=ExportStatus.COMPLETED, output_path=temp_db
        )

        validation = await validator.validate(result)
        # Should pass integrity check for our test database
        assert validation.is_valid

    @pytest.mark.asyncio
    async def test_sqlite_data_counts(self, sqlite_result):
        """Test SQLite data count retrieval."""
        validator = SQLiteValidator()
        counts = await validator.get_data_counts(sqlite_result)

        assert "verses" in counts
        assert counts["verses"] == 2  # We inserted 2 verses
        assert "annotations" in counts
        assert counts["annotations"] == 1
        assert "timeline_events" in counts
        assert counts["timeline_events"] == 1

    @pytest.mark.asyncio
    async def test_sqlite_performance_testing(self, temp_db):
        """Test SQLite performance testing."""
        validator = SQLiteValidator()

        with sqlite3.connect(temp_db) as conn:
            performance = await validator._test_sqlite_performance(conn)

            assert isinstance(performance, PerformanceMetrics)
            assert len(performance.query_times) > 0
            assert performance.avg_query_time >= 0

    @pytest.mark.asyncio
    async def test_missing_database_file(self):
        """Test validation with missing database file."""
        validator = SQLiteValidator()
        result = ExportResult(
            format_type=ExportFormat.SQLITE,
            status=ExportStatus.COMPLETED,
            output_path="/nonexistent/database.db",
        )

        validation = await validator.validate(result)
        assert not validation.is_valid
        assert any("not found" in error.lower() for error in validation.errors)


class TestJSONValidator:
    """Test JSON export validation."""

    @pytest.fixture
    def temp_json_export(self):
        """Create temporary JSON export structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)

            # Create directory structure
            api_dir = export_dir / "api" / "v1"
            meta_dir = api_dir / "meta"
            books_dir = api_dir / "books"

            meta_dir.mkdir(parents=True)
            books_dir.mkdir(parents=True)

            # Create essential files
            books_meta = {
                "books": [{"id": "GEN", "name": "Genesis", "chapters": [1, 2]}],
                "total_books": 1,
            }
            with open(meta_dir / "books.json", "w") as f:
                json.dump(books_meta, f)

            export_meta = {
                "format": "static_json",
                "version": "1.0",
                "generated_at": "2024-01-01T00:00:00",
            }
            with open(meta_dir / "export.json", "w") as f:
                json.dump(export_meta, f)

            # Create book structure
            gen_dir = books_dir / "GEN"
            gen_dir.mkdir()

            with open(gen_dir / "meta.json", "w") as f:
                json.dump({"id": "GEN", "name": "Genesis"}, f)

            # Create chapter structure
            chapter_dir = gen_dir / "chapters" / "1"
            chapter_dir.mkdir(parents=True)

            verses_data = {
                "verses": [{"verse_id": "GEN.1.1", "book": "GEN", "chapter": 1, "verse": 1}],
                "verse_count": 1,
            }
            with open(chapter_dir / "verses.json", "w") as f:
                json.dump(verses_data, f)

            # Create manifest
            manifest = {
                "format": "static_json",
                "progressive_loading": True,
                "files": [
                    {"path": "api/v1/meta/books.json", "category": "metadata"},
                    {"path": "api/v1/books/GEN/meta.json", "category": "content"},
                ],
            }
            with open(export_dir / "manifest.json", "w") as f:
                json.dump(manifest, f)

            yield str(export_dir)

    @pytest.fixture
    def json_result(self, temp_json_export):
        """Create JSON export result."""
        return ExportResult(
            format_type=ExportFormat.STATIC_JSON,
            status=ExportStatus.COMPLETED,
            output_path=temp_json_export,
            stats=ExportStatistics(),
        )

    @pytest.mark.asyncio
    async def test_json_validation_success(self, json_result):
        """Test successful JSON validation."""
        validator = JSONValidator()
        validation = await validator.validate(json_result)

        assert validation.is_valid
        assert len(validation.errors) == 0

    @pytest.mark.asyncio
    async def test_json_directory_structure(self, json_result):
        """Test JSON directory structure validation."""
        validator = JSONValidator()
        validation = await validator._validate_format_specific(
            json_result, ValidationResult(is_valid=True)
        )

        assert validation.is_valid
        # Should not have missing directory errors
        missing_dir_errors = [
            error for error in validation.errors if "directory missing" in error.lower()
        ]
        assert len(missing_dir_errors) == 0

    @pytest.mark.asyncio
    async def test_json_essential_files(self, json_result):
        """Test essential JSON files validation."""
        validator = JSONValidator()
        validation = await validator._validate_format_specific(
            json_result, ValidationResult(is_valid=True)
        )

        assert validation.is_valid
        # Should not have missing file errors
        missing_file_errors = [
            error for error in validation.errors if "file missing" in error.lower()
        ]
        assert len(missing_file_errors) == 0

    @pytest.mark.asyncio
    async def test_json_syntax_validation(self, json_result):
        """Test JSON syntax validation."""
        validator = JSONValidator()
        validation = await validator._validate_format_specific(
            json_result, ValidationResult(is_valid=True)
        )

        assert validation.is_valid
        # Should not have JSON syntax errors
        json_errors = [error for error in validation.errors if "invalid json" in error.lower()]
        assert len(json_errors) == 0

    @pytest.mark.asyncio
    async def test_json_performance_metrics(self, json_result):
        """Test JSON performance metrics collection."""
        validator = JSONValidator()
        validation = await validator._validate_format_specific(
            json_result, ValidationResult(is_valid=True)
        )

        assert "performance" in validation.metrics
        performance_data = validation.metrics["performance"]

        # Should have storage size
        assert "storage_size_bytes" in performance_data
        assert performance_data["storage_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_json_data_counts(self, json_result):
        """Test JSON data count retrieval."""
        validator = JSONValidator()
        counts = await validator.get_data_counts(json_result)

        assert "books" in counts
        assert counts["books"] == 1  # We created 1 book directory

    @pytest.mark.asyncio
    async def test_manifest_validation(self, temp_json_export):
        """Test manifest file validation."""
        validator = JSONValidator()
        manifest_path = Path(temp_json_export) / "manifest.json"
        validation = ValidationResult(is_valid=True)

        await validator._validate_manifest(manifest_path, validation)

        assert validation.is_valid
        assert len(validation.errors) == 0


class TestExportValidator:
    """Test main export validator orchestration."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ExportValidator()
        assert validator.logger is not None
        assert len(validator._validators) > 0

        # Check that validators are registered for each format
        assert ExportFormat.SQLITE in validator._validators
        assert ExportFormat.STATIC_JSON in validator._validators
        assert ExportFormat.OPENSEARCH in validator._validators

    @pytest.mark.asyncio
    async def test_validation_dispatch(self):
        """Test validation dispatch to format-specific validators."""
        validator = ExportValidator()

        # Mock SQLite result
        result = ExportResult(
            format_type=ExportFormat.SQLITE,
            status=ExportStatus.COMPLETED,
            output_path="/tmp/test.db",
        )

        # Mock the SQLite validator
        mock_validator = AsyncMock()
        mock_validation = ValidationResult(is_valid=True)
        mock_validator.validate.return_value = mock_validation

        validator._validators[ExportFormat.SQLITE] = mock_validator

        validation = await validator.validate_export(result)

        assert validation.is_valid
        mock_validator.validate.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_unknown_format_handling(self):
        """Test handling of unknown export formats."""
        validator = ExportValidator()

        # Create result with unknown format
        result = ExportResult(
            format_type="UNKNOWN_FORMAT", status=ExportStatus.COMPLETED  # Not a real format
        )

        validation = await validator.validate_export(result)

        assert not validation.is_valid
        assert any("no validator available" in error.lower() for error in validation.errors)

    @pytest.mark.asyncio
    async def test_cross_format_consistency(self):
        """Test cross-format consistency validation."""
        validator = ExportValidator()

        # Create multiple results
        results = [
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

        # Mock validators to return consistent data counts
        mock_sqlite_validator = AsyncMock()
        mock_sqlite_validator.get_data_counts.return_value = {"verses": 1000, "annotations": 500}

        mock_json_validator = AsyncMock()
        mock_json_validator.get_data_counts.return_value = {
            "verses": 1000,  # Consistent
            "annotations": 480,  # Slightly inconsistent
        }

        validator._validators[ExportFormat.SQLITE] = mock_sqlite_validator
        validator._validators[ExportFormat.STATIC_JSON] = mock_json_validator

        validation = await validator.validate_cross_format_consistency(results)

        # Should detect the inconsistency
        assert len(validation.warnings) > 0
        assert any("counts differ" in warning.lower() for warning in validation.warnings)

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test validation error handling."""
        validator = ExportValidator()

        # Mock validator that throws exception
        mock_validator = AsyncMock()
        mock_validator.validate.side_effect = Exception("Validation error")

        result = ExportResult(
            format_type=ExportFormat.SQLITE,
            status=ExportStatus.COMPLETED,
            output_path="/tmp/test.db",
        )

        validator._validators[ExportFormat.SQLITE] = mock_validator

        validation = await validator.validate_export(result)

        assert not validation.is_valid
        assert any("validation error" in error.lower() for error in validation.errors)


class TestIntegrityChecker:
    """Test data integrity checking."""

    def test_integrity_checker_initialization(self):
        """Test integrity checker initialization."""
        checker = IntegrityChecker()
        assert checker.logger is not None

    @pytest.mark.asyncio
    async def test_integrity_check_workflow(self):
        """Test complete integrity check workflow."""
        checker = IntegrityChecker()

        # Mock original data statistics
        original_stats = {"verses": 1000, "annotations": 500, "events": 100}

        # Mock export results
        export_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/test.db",
            )
        ]

        # Mock validator
        with patch("abba.export.validation.ExportValidator") as mock_validator_class:
            mock_validator = AsyncMock()
            mock_format_validator = AsyncMock()
            mock_format_validator.get_data_counts.return_value = {
                "verses": 1000,  # Matches original
                "annotations": 495,  # Slight difference
                "events": 100,  # Matches original
            }

            mock_validator._validators = {ExportFormat.SQLITE: mock_format_validator}
            mock_validator_class.return_value = mock_validator

            result = await checker.check_integrity(original_stats, export_results)

            assert isinstance(result, IntegrityCheckResult)
            # Should detect the annotation count difference
            assert len(result.warnings) > 0 or len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_consistency_tracking(self):
        """Test format consistency tracking."""
        checker = IntegrityChecker()

        original_stats = {"verses": 1000}
        export_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/test.db",
            )
        ]

        with patch("abba.export.validation.ExportValidator") as mock_validator_class:
            mock_validator = AsyncMock()
            mock_format_validator = AsyncMock()
            mock_format_validator.get_data_counts.return_value = {"verses": 1000}

            mock_validator._validators = {ExportFormat.SQLITE: mock_format_validator}
            mock_validator_class.return_value = mock_validator

            result = await checker.check_integrity(original_stats, export_results)

            # Should mark SQLite as consistent
            assert "sqlite" in result.format_consistency
            assert result.format_consistency["sqlite"] is True


class TestPerformanceBenchmark:
    """Test performance benchmarking."""

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = PerformanceBenchmark()
        assert benchmark.logger is not None

    @pytest.mark.asyncio
    async def test_benchmark_workflow(self):
        """Test benchmark workflow."""
        benchmark = PerformanceBenchmark()

        # Mock export results
        export_results = [
            ExportResult(
                format_type=ExportFormat.SQLITE,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/test.db",
                stats=ExportStatistics(output_size_bytes=1024000),
            ),
            ExportResult(
                format_type=ExportFormat.STATIC_JSON,
                status=ExportStatus.COMPLETED,
                output_path="/tmp/json_output",
                stats=ExportStatistics(output_size_bytes=2048000),
            ),
        ]

        with patch.object(benchmark, "_benchmark_format", return_value=PerformanceMetrics()):
            benchmarks = await benchmark.benchmark_exports(export_results)

            assert "sqlite" in benchmarks
            assert "static_json" in benchmarks
            assert isinstance(benchmarks["sqlite"], PerformanceMetrics)

    @pytest.mark.asyncio
    async def test_sqlite_benchmark(self, temp_db):
        """Test SQLite-specific benchmarking."""
        benchmark = PerformanceBenchmark()

        result = ExportResult(
            format_type=ExportFormat.SQLITE, status=ExportStatus.COMPLETED, output_path=temp_db
        )

        metrics = await benchmark._benchmark_sqlite(result)

        assert isinstance(metrics, PerformanceMetrics)
        assert len(metrics.query_times) > 0
        assert metrics.storage_size_bytes > 0
        assert len(metrics.concurrent_query_performance) > 0

    @pytest.mark.asyncio
    async def test_json_benchmark(self, temp_json_export):
        """Test JSON-specific benchmarking."""
        benchmark = PerformanceBenchmark()

        result = ExportResult(
            format_type=ExportFormat.STATIC_JSON,
            status=ExportStatus.COMPLETED,
            output_path=temp_json_export,
        )

        metrics = await benchmark._benchmark_json(result)

        assert isinstance(metrics, PerformanceMetrics)
        assert len(metrics.query_times) > 0  # File loading times
        assert metrics.storage_size_bytes > 0
        assert len(metrics.search_times) > 0  # Search simulation times


class TestOpenSearchValidator:
    """Test OpenSearch validator (simplified since it requires cluster access)."""

    @pytest.mark.asyncio
    async def test_opensearch_validation_placeholder(self):
        """Test OpenSearch validation placeholder."""
        validator = OpenSearchValidator()

        result = ExportResult(
            format_type=ExportFormat.OPENSEARCH,
            status=ExportStatus.COMPLETED,
            output_path="opensearch://localhost:9200/abba",
        )

        validation = await validator.validate(result)

        # Should add warning about requiring cluster access
        assert len(validation.warnings) > 0
        assert any("cluster access" in warning.lower() for warning in validation.warnings)


class TestNeo4jValidator:
    """Test Neo4j validator (simplified since it requires database access)."""

    @pytest.mark.asyncio
    async def test_neo4j_validation_placeholder(self):
        """Test Neo4j validation placeholder."""
        validator = Neo4jValidator()

        result = ExportResult(
            format_type=ExportFormat.NEO4J,
            status=ExportStatus.COMPLETED,
            output_path="neo4j://localhost:7687",
        )

        validation = await validator.validate(result)

        # Should add warning about requiring database access
        assert len(validation.warnings) > 0
        assert any("database access" in warning.lower() for warning in validation.warnings)


class TestArangoValidator:
    """Test ArangoDB validator (simplified since it requires database access)."""

    @pytest.mark.asyncio
    async def test_arango_validation_placeholder(self):
        """Test ArangoDB validation placeholder."""
        validator = ArangoValidator()

        result = ExportResult(
            format_type=ExportFormat.ARANGODB,
            status=ExportStatus.COMPLETED,
            output_path="arangodb://localhost:8529",
        )

        validation = await validator.validate(result)

        # Should add warning about requiring database access
        assert len(validation.warnings) > 0
        assert any("database access" in warning.lower() for warning in validation.warnings)


if __name__ == "__main__":
    pytest.main([__file__])
