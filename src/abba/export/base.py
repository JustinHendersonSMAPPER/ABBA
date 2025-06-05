"""
Base classes and interfaces for the export system.

Provides abstract base classes, common data structures, and shared utilities
for implementing format-specific exporters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncIterator, Iterator, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
import asyncio
import logging

from ..verse_id import VerseID
from ..parsers.translation_parser import TranslationVerse
from ..annotations.models import Annotation
from ..timeline.models import Event, TimePeriod


class ExportFormat(Enum):
    """Supported export formats."""

    SQLITE = "sqlite"
    STATIC_JSON = "static_json"
    OPENSEARCH = "opensearch"
    NEO4J = "neo4j"
    ARANGODB = "arangodb"


class ExportStatus(Enum):
    """Export operation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportError(Exception):
    """Exception raised during export operations."""

    message: str
    format_type: Optional[ExportFormat] = None
    stage: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ExportConfig:
    """Base configuration for export operations."""

    # Output settings
    output_path: str
    format_type: ExportFormat

    # Processing settings
    batch_size: int = 1000
    parallel_workers: int = 4
    memory_limit_mb: int = 512

    # Quality settings
    validate_output: bool = True
    benchmark_performance: bool = False

    # Optimization settings
    compress_output: bool = True
    optimize_for_size: bool = False
    optimize_for_speed: bool = True

    # Metadata
    export_name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def validate(self) -> 'ValidationResult':
        """Validate configuration settings."""
        errors = []
        warnings = []
        
        # Validate output path
        if not self.output_path:
            errors.append("Output path is required")
            
        # Validate batch size
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
            
        # Validate parallel workers
        if self.parallel_workers <= 0:
            errors.append("Parallel workers must be positive")
            
        # Validate memory limit
        if self.memory_limit_mb <= 0:
            errors.append("Memory limit must be positive")
            
        # Check for conflicting optimization settings
        if self.optimize_for_size and self.optimize_for_speed:
            warnings.append("Both size and speed optimization enabled; speed will take precedence")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output_path": self.output_path,
            "format_type": self.format_type.value,
            "batch_size": self.batch_size,
            "parallel_workers": self.parallel_workers,
            "memory_limit_mb": self.memory_limit_mb,
            "validate_output": self.validate_output,
            "benchmark_performance": self.benchmark_performance,
            "compress_output": self.compress_output,
            "optimize_for_size": self.optimize_for_size,
            "optimize_for_speed": self.optimize_for_speed,
            "export_name": self.export_name,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class ValidationResult:
    """Result of export validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)


@dataclass
class ExportStats:
    """Statistics about export operation."""

    total_verses: int = 0
    total_annotations: int = 0
    total_events: int = 0
    total_cross_references: int = 0

    processed_verses: int = 0
    processed_annotations: int = 0
    processed_events: int = 0
    processed_cross_references: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Storage metrics
    output_size_bytes: int = 0
    compression_ratio: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate operation duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def verses_per_second(self) -> Optional[float]:
        """Calculate processing rate."""
        duration = self.duration_seconds
        if duration and duration > 0:
            return self.processed_verses / duration
        return None


@dataclass
class ExportResult:
    """Result of export operation."""

    format_type: ExportFormat
    status: ExportStatus
    output_path: str

    stats: ExportStats = field(default_factory=ExportStats)
    validation: Optional[ValidationResult] = None
    error: Optional[ExportError] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "format_type": self.format_type.value,
            "status": self.status.value,
            "output_path": self.output_path,
            "metadata": self.metadata,
        }

        # Add stats
        if self.stats:
            result["stats"] = {
                "total_verses": self.stats.total_verses,
                "processed_verses": self.stats.processed_verses,
                "duration_seconds": self.stats.duration_seconds,
                "verses_per_second": self.stats.verses_per_second,
                "output_size_bytes": self.stats.output_size_bytes,
            }

        # Add validation
        if self.validation:
            result["validation"] = {
                "is_valid": self.validation.is_valid,
                "error_count": len(self.validation.errors),
                "warning_count": len(self.validation.warnings),
            }

        # Add error info
        if self.error:
            result["error"] = {
                "message": self.error.message,
                "stage": self.error.stage,
                "details": self.error.details,
            }

        return result


@dataclass
class CanonicalDataset:
    """Container for all canonical data to be exported."""

    # Core biblical data
    verses: Iterator[TranslationVerse]

    # Enrichment data
    annotations: Optional[Iterator[Annotation]] = None
    cross_references: Optional[Iterator[Dict[str, Any]]] = None
    timeline_events: Optional[Iterator[Event]] = None
    timeline_periods: Optional[Iterator[TimePeriod]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Data counts (for progress tracking)
    total_verses: Optional[int] = None
    total_annotations: Optional[int] = None
    total_events: Optional[int] = None

    def get_estimated_total(self) -> int:
        """Get estimated total items to process."""
        return sum(filter(None, [self.total_verses, self.total_annotations, self.total_events]))


class DataExporter(ABC):
    """Abstract base class for format-specific exporters."""

    def __init__(self, config: ExportConfig):
        """Initialize exporter with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize statistics
        self.stats = ExportStats()

    @abstractmethod
    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Export canonical data to target format."""
        pass

    @abstractmethod
    def validate_config(self) -> ValidationResult:
        """Validate exporter configuration."""
        pass

    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """Get list of features supported by this exporter."""
        pass

    async def prepare_export(self, data: CanonicalDataset) -> None:
        """Prepare for export operation (setup, validation, etc.)."""
        self.stats.start_time = datetime.now()

        # Set total counts if available
        if data.total_verses:
            self.stats.total_verses = data.total_verses
        if data.total_annotations:
            self.stats.total_annotations = data.total_annotations
        if data.total_events:
            self.stats.total_events = data.total_events

        self.logger.info(
            f"Starting {self.config.format_type.value} export to {self.config.output_path}"
        )

    async def finalize_export(self, result: ExportResult) -> ExportResult:
        """Finalize export operation (cleanup, validation, etc.)."""
        self.stats.end_time = datetime.now()
        result.stats = self.stats

        # Calculate output size
        if Path(self.config.output_path).exists():
            if Path(self.config.output_path).is_file():
                result.stats.output_size_bytes = Path(self.config.output_path).stat().st_size
            else:
                # Directory - sum all files
                result.stats.output_size_bytes = sum(
                    f.stat().st_size
                    for f in Path(self.config.output_path).rglob("*")
                    if f.is_file()
                )

        # Validate if requested
        if self.config.validate_output:
            result.validation = await self.validate_output(result)

        duration = self.stats.duration_seconds
        self.logger.info(
            f"Completed {self.config.format_type.value} export in {duration:.2f}s. "
            f"Processed {self.stats.processed_verses} verses "
            f"({self.stats.verses_per_second:.1f} verses/sec)"
        )

        return result

    async def validate_output(self, result: ExportResult) -> ValidationResult:
        """Validate exported output."""
        validation = ValidationResult(is_valid=True)

        # Check output exists
        output_path = Path(result.output_path)
        if not output_path.exists():
            validation.add_error(f"Output path does not exist: {result.output_path}")
            return validation

        # Basic size check
        if result.stats.output_size_bytes == 0:
            validation.add_error("Output file is empty")
        elif result.stats.output_size_bytes < 1024:  # Less than 1KB
            validation.add_warning("Output file is very small, may be incomplete")

        return validation

    def update_progress(self, processed_items: int, item_type: str = "verses"):
        """Update processing progress."""
        if item_type == "verses":
            self.stats.processed_verses = processed_items
        elif item_type == "annotations":
            self.stats.processed_annotations = processed_items
        elif item_type == "events":
            self.stats.processed_events = processed_items
        elif item_type == "cross_references":
            self.stats.processed_cross_references = processed_items

    def create_result(
        self, status: ExportStatus, error: Optional[ExportError] = None
    ) -> ExportResult:
        """Create export result."""
        return ExportResult(
            format_type=self.config.format_type,
            status=status,
            output_path=self.config.output_path,
            stats=self.stats,
            error=error,
        )


class StreamingDataProcessor:
    """Utility class for streaming data processing."""

    def __init__(self, batch_size: int = 1000):
        """Initialize processor."""
        self.batch_size = batch_size

    async def process_in_batches(
        self,
        data_iterator: Iterator[Any],
        processor_func: callable,
        progress_callback: Optional[callable] = None,
    ) -> AsyncIterator[Any]:
        """Process data in batches to manage memory usage."""
        batch = []
        total_processed = 0

        for item in data_iterator:
            batch.append(item)

            if len(batch) >= self.batch_size:
                # Process batch
                result = await processor_func(batch)
                if result:
                    yield result

                total_processed += len(batch)
                if progress_callback:
                    progress_callback(total_processed)

                batch = []

        # Process remaining items
        if batch:
            result = await processor_func(batch)
            if result:
                yield result

            total_processed += len(batch)
            if progress_callback:
                progress_callback(total_processed)

    async def collect_batches(self, batch_iterator: AsyncIterator[List[Any]]) -> List[Any]:
        """Collect all items from batched iterator."""
        result = []
        async for batch in batch_iterator:
            result.extend(batch)
        return result


class ExportUtilities:
    """Shared utilities for exporters."""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary."""
        path_obj = Path(path)
        if path_obj.suffix:  # It's a file path
            path_obj = path_obj.parent

        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj

    @staticmethod
    def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 0.0
        return 1.0 - (compressed_size / original_size)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem safety."""
        import re

        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Remove leading/trailing periods and spaces
        sanitized = sanitized.strip(". ")
        # Ensure not empty
        if not sanitized:
            sanitized = "untitled"
        return sanitized

    @staticmethod
    def create_manifest(export_info: Dict[str, Any], output_path: Path) -> None:
        """Create export manifest file."""
        import json

        manifest = {
            "export_info": export_info,
            "created_at": datetime.now().isoformat(),
            "format_version": "1.0",
        }

        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
