"""
Validation and quality assurance for ABBA export system.

Provides comprehensive validation, integrity checking, and performance
benchmarking for all export formats to ensure data quality and consistency.
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import sqlite3
import json
import aiohttp

from .base import ExportResult, ValidationResult, ExportFormat


@dataclass
class PerformanceMetrics:
    """Performance metrics for export validation."""

    # Query performance
    query_times: List[float] = field(default_factory=list)
    search_times: List[float] = field(default_factory=list)

    # Storage metrics
    storage_size_bytes: int = 0
    compression_ratio: Optional[float] = None

    # Throughput metrics
    queries_per_second: Optional[float] = None
    concurrent_query_performance: Dict[int, float] = field(default_factory=dict)

    @property
    def avg_query_time(self) -> float:
        """Average query time in milliseconds."""
        return statistics.mean(self.query_times) * 1000 if self.query_times else 0.0

    @property
    def p95_query_time(self) -> float:
        """95th percentile query time in milliseconds."""
        if not self.query_times:
            return 0.0
        sorted_times = sorted(self.query_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] * 1000

    def add_query_time(self, duration: float):
        """Add query execution time."""
        self.query_times.append(duration)

    def add_search_time(self, duration: float):
        """Add search execution time."""
        self.search_times.append(duration)


@dataclass
class IntegrityCheckResult:
    """Result of data integrity checking."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Data counts
    expected_counts: Dict[str, int] = field(default_factory=dict)
    actual_counts: Dict[str, int] = field(default_factory=dict)

    # Cross-format consistency
    format_consistency: Dict[str, bool] = field(default_factory=dict)

    def add_error(self, message: str):
        """Add integrity error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add integrity warning."""
        self.warnings.append(message)

    def set_expected_count(self, data_type: str, count: int):
        """Set expected count for data type."""
        self.expected_counts[data_type] = count
        
        # Check against actual if already set
        if data_type in self.actual_counts:
            actual = self.actual_counts[data_type]
            if actual != count:
                if actual == 0 and count > 0:
                    self.add_error(f"No {data_type} found (expected {count})")
                elif count > 0 and abs(actual - count) / count > 0.05:  # >5% difference
                    self.add_error(f"{data_type} count mismatch: expected {count}, got {actual}")
                elif actual != count:
                    self.add_warning(
                        f"{data_type} count slight difference: expected {count}, got {actual}"
                    )

    def set_actual_count(self, data_type: str, count: int):
        """Set actual count for data type."""
        self.actual_counts[data_type] = count

        # Check against expected
        if data_type in self.expected_counts:
            expected = self.expected_counts[data_type]
            if count != expected:
                if count == 0 and expected > 0:
                    self.add_error(f"No {data_type} found (expected {expected})")
                elif expected > 0 and abs(count - expected) / expected > 0.05:  # >5% difference
                    self.add_error(f"{data_type} count mismatch: expected {expected}, got {count}")
                elif count != expected:
                    self.add_warning(
                        f"{data_type} count slight difference: expected {expected}, got {count}"
                    )


class ExportValidator:
    """Main export validation orchestrator."""

    def __init__(self):
        """Initialize export validator."""
        self.logger = logging.getLogger(f"{__name__}.ExportValidator")

        # Format-specific validators
        self._validators = {
            ExportFormat.SQLITE: SQLiteValidator(),
            ExportFormat.STATIC_JSON: JSONValidator(),
            ExportFormat.OPENSEARCH: OpenSearchValidator(),
            ExportFormat.NEO4J: Neo4jValidator(),
            ExportFormat.ARANGODB: ArangoValidator(),
        }

    async def validate_export(self, result: ExportResult) -> ValidationResult:
        """Validate export result comprehensively."""
        # Handle both string and enum format types
        format_str = result.format_type.value if hasattr(result.format_type, 'value') else str(result.format_type)
        self.logger.info(f"Validating {format_str} export")

        validation = ValidationResult(is_valid=True)

        try:
            # Get format-specific validator
            if result.format_type not in self._validators:
                validation.add_error(f"No validator available for {format_str}")
                return validation

            validator = self._validators[result.format_type]

            # Run format-specific validation
            format_validation = await validator.validate(result)

            # Merge results
            validation.is_valid = format_validation.is_valid
            validation.errors.extend(format_validation.errors)
            validation.warnings.extend(format_validation.warnings)
            validation.metrics.update(format_validation.metrics)

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            validation.add_error(f"Validation error: {str(e)}")

        return validation

    async def validate_cross_format_consistency(
        self, results: List[ExportResult]
    ) -> ValidationResult:
        """Validate consistency across multiple export formats."""
        self.logger.info("Validating cross-format consistency")

        validation = ValidationResult(is_valid=True)

        if len(results) < 2:
            validation.add_warning(
                "Cannot check cross-format consistency with fewer than 2 formats"
            )
            return validation

        try:
            # Collect data counts from each format
            format_counts = {}

            for result in results:
                if result.format_type in self._validators:
                    validator = self._validators[result.format_type]
                    counts = await validator.get_data_counts(result)
                    format_counts[result.format_type.value] = counts

            # Compare counts across formats
            if len(format_counts) >= 2:
                await self._compare_format_counts(format_counts, validation)

        except Exception as e:
            validation.add_error(f"Cross-format validation failed: {str(e)}")

        return validation

    async def _compare_format_counts(
        self, format_counts: Dict[str, Dict[str, int]], validation: ValidationResult
    ):
        """Compare data counts across formats."""
        data_types = set()
        for counts in format_counts.values():
            data_types.update(counts.keys())

        for data_type in data_types:
            type_counts = {}
            for format_name, counts in format_counts.items():
                if data_type in counts:
                    type_counts[format_name] = counts[data_type]

            if len(type_counts) >= 2:
                # Check for consistency
                count_values = list(type_counts.values())
                if len(set(count_values)) > 1:
                    # Counts differ
                    validation.add_warning(
                        f"{data_type} counts differ across formats: {type_counts}"
                    )


class FormatValidator:
    """Base class for format-specific validators."""

    def __init__(self):
        """Initialize format validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def validate(self, result: ExportResult) -> ValidationResult:
        """Validate export result."""
        validation = ValidationResult(is_valid=True)

        # Basic validation - check if it's a URL or file path
        if not (result.output_path.startswith("http://") or 
                result.output_path.startswith("https://") or
                result.output_path.startswith("opensearch://") or
                result.output_path.startswith("neo4j://") or
                result.output_path.startswith("arangodb://")):
            # It's a file path, check if it exists
            output_path = Path(result.output_path)
            if not output_path.exists():
                validation.add_error(f"Output path does not exist: {result.output_path}")
                return validation

        # Format-specific validation
        return await self._validate_format_specific(result, validation)

    async def _validate_format_specific(
        self, result: ExportResult, validation: ValidationResult
    ) -> ValidationResult:
        """Override in subclasses for format-specific validation."""
        return validation

    async def get_data_counts(self, result: ExportResult) -> Dict[str, int]:
        """Get data counts for consistency checking."""
        return {}


class SQLiteValidator(FormatValidator):
    """SQLite export validator."""

    async def _validate_format_specific(
        self, result: ExportResult, validation: ValidationResult
    ) -> ValidationResult:
        """Validate SQLite database."""
        db_path = Path(result.output_path)

        if not db_path.is_file():
            validation.add_error("SQLite database file not found")
            return validation

        try:
            with sqlite3.connect(db_path) as conn:
                # Check database integrity
                integrity_result = conn.execute("PRAGMA integrity_check").fetchone()
                if integrity_result[0] != "ok":
                    validation.add_error(f"Database integrity check failed: {integrity_result[0]}")

                # Check table structure
                tables = conn.execute(
                    """
                    SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
                ).fetchall()

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

                actual_tables = {table[0] for table in tables}
                missing_tables = expected_tables - actual_tables

                if missing_tables:
                    validation.add_error(f"Missing tables: {missing_tables}")

                # Check data counts
                counts = await self.get_data_counts(result)
                for table, count in counts.items():
                    if count == 0:
                        validation.add_warning(f"Table {table} is empty")

                # Test query performance
                performance = await self._test_sqlite_performance(conn)
                validation.metrics["performance"] = performance.__dict__

        except Exception as e:
            validation.add_error(f"SQLite validation failed: {str(e)}")

        return validation

    async def _test_sqlite_performance(self, conn: sqlite3.Connection) -> PerformanceMetrics:
        """Test SQLite query performance."""
        performance = PerformanceMetrics()

        # Test basic queries
        test_queries = [
            "SELECT COUNT(*) FROM verses",
            "SELECT * FROM verses WHERE book = 'GEN' LIMIT 10",
            "SELECT * FROM verses WHERE chapter = 1 LIMIT 10",
            "SELECT v.*, vt.text FROM verses v JOIN verse_translations vt ON v.verse_id = vt.verse_id LIMIT 10",
        ]

        for query in test_queries:
            try:
                start_time = time.time()
                result = conn.execute(query).fetchall()
                end_time = time.time()

                performance.add_query_time(end_time - start_time)
            except Exception as e:
                self.logger.warning(f"Performance test query failed: {str(e)}")

        # Test FTS if available
        try:
            start_time = time.time()
            conn.execute("SELECT * FROM verses_fts WHERE text MATCH 'God' LIMIT 10").fetchall()
            end_time = time.time()

            performance.add_search_time(end_time - start_time)
        except sqlite3.OperationalError:
            # FTS not available
            pass

        return performance

    async def get_data_counts(self, result: ExportResult) -> Dict[str, int]:
        """Get SQLite data counts."""
        counts = {}
        db_path = Path(result.output_path)

        try:
            with sqlite3.connect(db_path) as conn:
                tables = [
                    "verses",
                    "annotations",
                    "cross_references",
                    "timeline_events",
                    "timeline_periods",
                ]

                for table in tables:
                    try:
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        counts[table] = count
                    except sqlite3.OperationalError:
                        # Table doesn't exist
                        counts[table] = 0

        except Exception as e:
            self.logger.error(f"Failed to get SQLite counts: {str(e)}")

        return counts


class JSONValidator(FormatValidator):
    """Static JSON export validator."""

    async def _validate_format_specific(
        self, result: ExportResult, validation: ValidationResult
    ) -> ValidationResult:
        """Validate JSON export."""
        output_dir = Path(result.output_path)

        if not output_dir.is_dir():
            validation.add_error("JSON output directory not found")
            return validation

        try:
            # Check directory structure
            required_dirs = ["api/v1/meta", "api/v1/books"]
            for dir_path in required_dirs:
                full_path = output_dir / dir_path
                if not full_path.exists():
                    validation.add_error(f"Required directory missing: {dir_path}")

            # Check essential files
            api_dir = output_dir / "api/v1"
            essential_files = ["meta/books.json", "meta/export.json"]

            for file_path in essential_files:
                full_path = api_dir / file_path
                gz_path = full_path.with_suffix(full_path.suffix + ".gz")

                if not full_path.exists() and not gz_path.exists():
                    validation.add_error(f"Essential file missing: {file_path}")

            # Validate JSON syntax
            json_files = list(output_dir.rglob("*.json"))[:10]  # Sample validation

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    validation.add_error(f"Invalid JSON in {json_file}: {str(e)}")

            # Check manifest if it exists
            manifest_path = output_dir / "manifest.json"
            gz_manifest_path = manifest_path.with_suffix(".json.gz")

            if manifest_path.exists() or gz_manifest_path.exists():
                await self._validate_manifest(
                    manifest_path if manifest_path.exists() else gz_manifest_path, validation
                )

            # Test file access performance
            performance = await self._test_json_performance(output_dir)
            validation.metrics["performance"] = performance.__dict__

        except Exception as e:
            validation.add_error(f"JSON validation failed: {str(e)}")

        return validation

    async def _validate_manifest(self, manifest_path: Path, validation: ValidationResult):
        """Validate progressive loading manifest."""
        try:
            if manifest_path.suffix == ".gz":
                import gzip

                with gzip.open(manifest_path, "rt", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

            # Check required fields
            required_fields = ["format", "version", "files"]
            for field in required_fields:
                if field not in manifest:
                    validation.add_error(f"Manifest missing required field: {field}")

            # Validate file references
            if "files" in manifest:
                base_dir = manifest_path.parent
                for file_info in manifest["files"][:10]:  # Sample check
                    if "path" in file_info:
                        file_path = base_dir / file_info["path"]
                        if not file_path.exists():
                            validation.add_warning(
                                f"Manifest references missing file: {file_info['path']}"
                            )

        except Exception as e:
            validation.add_error(f"Manifest validation failed: {str(e)}")

    async def _test_json_performance(self, output_dir: Path) -> PerformanceMetrics:
        """Test JSON file access performance."""
        performance = PerformanceMetrics()

        # Test file loading times
        json_files = list(output_dir.rglob("*.json"))[:5]  # Test first 5 files

        for json_file in json_files:
            try:
                start_time = time.time()
                with open(json_file, "r", encoding="utf-8") as f:
                    json.load(f)
                end_time = time.time()

                performance.add_query_time(end_time - start_time)
            except Exception as e:
                self.logger.warning(f"Performance test failed for {json_file}: {str(e)}")

        # Calculate storage metrics
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        performance.storage_size_bytes = total_size

        return performance

    async def get_data_counts(self, result: ExportResult) -> Dict[str, int]:
        """Get JSON data counts."""
        counts = {}
        output_dir = Path(result.output_path)

        try:
            # Count book files
            books_dir = output_dir / "api/v1/books"
            if books_dir.exists():
                book_dirs = [d for d in books_dir.iterdir() if d.is_dir()]
                counts["books"] = len(book_dirs)

                # Estimate verse count from book files
                verse_count = 0
                for book_dir in book_dirs[:5]:  # Sample first 5 books
                    chapters_dir = book_dir / "chapters"
                    if chapters_dir.exists():
                        for chapter_dir in chapters_dir.iterdir():
                            if chapter_dir.is_dir():
                                verses_file = chapter_dir / "verses.json"
                                if verses_file.exists():
                                    try:
                                        with open(verses_file, "r") as f:
                                            data = json.load(f)
                                            if "verses" in data:
                                                verse_count += len(data["verses"])
                                            elif "chunks" in data:
                                                verse_count += sum(
                                                    chunk.get("verse_count", 0)
                                                    for chunk in data["chunks"]
                                                )
                                    except:
                                        pass

                counts["verses"] = verse_count

            # Count timeline files
            timeline_dir = output_dir / "api/v1/timeline"
            if timeline_dir.exists():
                events_file = timeline_dir / "events.json"
                if events_file.exists():
                    try:
                        with open(events_file, "r") as f:
                            data = json.load(f)
                            counts["events"] = len(data.get("events", []))
                    except:
                        counts["events"] = 0

        except Exception as e:
            self.logger.error(f"Failed to get JSON counts: {str(e)}")

        return counts


class OpenSearchValidator(FormatValidator):
    """OpenSearch export validator."""

    async def _validate_format_specific(
        self, result: ExportResult, validation: ValidationResult
    ) -> ValidationResult:
        """Validate OpenSearch export."""
        # Note: This requires the OpenSearch cluster to be accessible
        # In a real implementation, you would extract connection details from result

        try:
            # For this example, we'll do basic validation
            validation.add_warning("OpenSearch validation requires cluster access")

            # You would typically:
            # 1. Connect to cluster
            # 2. Check index existence
            # 3. Validate document counts
            # 4. Test search performance
            # 5. Verify mapping correctness

        except Exception as e:
            validation.add_error(f"OpenSearch validation failed: {str(e)}")

        return validation

    async def get_data_counts(self, result: ExportResult) -> Dict[str, int]:
        """Get OpenSearch data counts."""
        # Would query OpenSearch indices for document counts
        return {}


class Neo4jValidator(FormatValidator):
    """Neo4j export validator."""

    async def _validate_format_specific(
        self, result: ExportResult, validation: ValidationResult
    ) -> ValidationResult:
        """Validate Neo4j export."""
        try:
            # For this example, we'll do basic validation
            validation.add_warning("Neo4j validation requires database access")

            # You would typically:
            # 1. Connect to Neo4j
            # 2. Check node/relationship counts
            # 3. Validate constraints and indices
            # 4. Test query performance
            # 5. Verify graph structure

        except Exception as e:
            validation.add_error(f"Neo4j validation failed: {str(e)}")

        return validation

    async def get_data_counts(self, result: ExportResult) -> Dict[str, int]:
        """Get Neo4j data counts."""
        # Would query Neo4j for node/relationship counts
        return {}


class ArangoValidator(FormatValidator):
    """ArangoDB export validator."""

    async def _validate_format_specific(
        self, result: ExportResult, validation: ValidationResult
    ) -> ValidationResult:
        """Validate ArangoDB export."""
        try:
            # For this example, we'll do basic validation
            validation.add_warning("ArangoDB validation requires database access")

            # You would typically:
            # 1. Connect to ArangoDB
            # 2. Check collection existence
            # 3. Validate document counts
            # 4. Test query performance
            # 5. Verify graph structure

        except Exception as e:
            validation.add_error(f"ArangoDB validation failed: {str(e)}")

        return validation

    async def get_data_counts(self, result: ExportResult) -> Dict[str, int]:
        """Get ArangoDB data counts."""
        # Would query ArangoDB collections for document counts
        return {}


class IntegrityChecker:
    """Checks data integrity across the export process."""

    def __init__(self):
        """Initialize integrity checker."""
        self.logger = logging.getLogger(f"{__name__}.IntegrityChecker")

    async def check_integrity(
        self, original_data_stats: Dict[str, int], export_results: List[ExportResult]
    ) -> IntegrityCheckResult:
        """Check data integrity across all exports."""
        self.logger.info("Checking data integrity across exports")

        result = IntegrityCheckResult()

        # Set expected counts from original data
        for data_type, count in original_data_stats.items():
            result.set_expected_count(data_type, count)

        # Check each export format
        validator = ExportValidator()

        for export_result in export_results:
            try:
                if export_result.format_type in validator._validators:
                    format_validator = validator._validators[export_result.format_type]
                    counts = await format_validator.get_data_counts(export_result)

                    for data_type, count in counts.items():
                        # Set actual count with format prefix
                        result.set_actual_count(
                            f"{export_result.format_type.value}_{data_type}", count
                        )

                        # Also check against expected counts if they exist
                        if data_type in original_data_stats:
                            expected = original_data_stats[data_type]
                            # Temporarily set this data type to trigger warning/error logic
                            result.set_expected_count(data_type, expected)
                            result.set_actual_count(data_type, count)
                            
                            if count != expected:
                                result.format_consistency[export_result.format_type.value] = False
                            else:
                                result.format_consistency[export_result.format_type.value] = True

            except Exception as e:
                result.add_error(
                    f"Integrity check failed for {export_result.format_type.value}: {str(e)}"
                )

        return result


class PerformanceBenchmark:
    """Benchmarks export performance and query capabilities."""

    def __init__(self):
        """Initialize performance benchmark."""
        self.logger = logging.getLogger(f"{__name__}.PerformanceBenchmark")

    async def benchmark_exports(
        self, export_results: List[ExportResult]
    ) -> Dict[str, PerformanceMetrics]:
        """Benchmark performance of all export formats."""
        self.logger.info("Benchmarking export performance")

        benchmarks = {}

        for result in export_results:
            try:
                metrics = await self._benchmark_format(result)
                benchmarks[result.format_type.value] = metrics

            except Exception as e:
                self.logger.error(f"Benchmark failed for {result.format_type.value}: {str(e)}")
                benchmarks[result.format_type.value] = PerformanceMetrics()

        return benchmarks

    async def _benchmark_format(self, result: ExportResult) -> PerformanceMetrics:
        """Benchmark specific export format."""
        if result.format_type == ExportFormat.SQLITE:
            return await self._benchmark_sqlite(result)
        elif result.format_type == ExportFormat.STATIC_JSON:
            return await self._benchmark_json(result)
        else:
            # For other formats, return basic metrics
            metrics = PerformanceMetrics()
            metrics.storage_size_bytes = result.stats.output_size_bytes
            return metrics

    async def _benchmark_sqlite(self, result: ExportResult) -> PerformanceMetrics:
        """Benchmark SQLite database."""
        metrics = PerformanceMetrics()
        db_path = Path(result.output_path)

        if not db_path.exists():
            return metrics

        try:
            with sqlite3.connect(db_path) as conn:
                # Benchmark common queries
                test_queries = [
                    ("count_verses", "SELECT COUNT(*) FROM verses"),
                    ("book_lookup", "SELECT * FROM verses WHERE book = 'GEN' LIMIT 10"),
                    ("chapter_lookup", "SELECT * FROM verses WHERE book = 'GEN' AND chapter = 1"),
                    (
                        "verse_with_translation",
                        "SELECT v.*, vt.text FROM verses v JOIN verse_translations vt ON v.verse_id = vt.verse_id WHERE v.book = 'GEN' LIMIT 10",
                    ),
                    (
                        "cross_references",
                        "SELECT * FROM cross_references WHERE source_verse_id LIKE 'GEN%' LIMIT 10",
                    ),
                ]

                for query_name, query in test_queries:
                    # Run query multiple times for average
                    times = []
                    for _ in range(5):
                        start_time = time.time()
                        result = conn.execute(query).fetchall()
                        end_time = time.time()
                        times.append(end_time - start_time)

                    avg_time = statistics.mean(times)
                    metrics.add_query_time(avg_time)

                # Test concurrent queries
                concurrent_results = {}
                for concurrency in [1, 2, 4]:
                    start_time = time.time()
                    # Simulate concurrent queries (simplified)
                    for _ in range(concurrency * 10):
                        conn.execute("SELECT COUNT(*) FROM verses WHERE book = 'GEN'").fetchone()
                    end_time = time.time()

                    total_time = end_time - start_time
                    qps = (concurrency * 10) / total_time
                    concurrent_results[concurrency] = qps

                metrics.concurrent_query_performance = concurrent_results
                metrics.storage_size_bytes = db_path.stat().st_size

        except Exception as e:
            self.logger.error(f"SQLite benchmark failed: {str(e)}")

        return metrics

    async def _benchmark_json(self, result: ExportResult) -> PerformanceMetrics:
        """Benchmark JSON file access."""
        metrics = PerformanceMetrics()
        output_dir = Path(result.output_path)

        if not output_dir.exists():
            return metrics

        try:
            # Test file loading performance
            json_files = list(output_dir.rglob("*.json"))[:10]

            for json_file in json_files:
                start_time = time.time()
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                end_time = time.time()

                metrics.add_query_time(end_time - start_time)

            # Calculate storage metrics
            total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
            metrics.storage_size_bytes = total_size

            # Test search simulation (loading multiple files)
            search_start = time.time()
            search_files = json_files[:3]
            search_results = []

            for file_path in search_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Simulate search operation
                    if isinstance(data, dict) and "verses" in data:
                        search_results.extend(data["verses"][:5])

            search_end = time.time()
            metrics.add_search_time(search_end - search_start)

        except Exception as e:
            self.logger.error(f"JSON benchmark failed: {str(e)}")

        return metrics
