"""
Export pipeline for orchestrating multi-format exports.

Coordinates the export of canonical data to multiple target formats,
handling dependencies, validation, and parallel processing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from datetime import datetime
import yaml
import json

from .base import (
    DataExporter,
    CanonicalDataset,
    ExportConfig,
    ExportResult,
    ExportFormat,
    ExportStatus,
    ExportError,
    ValidationResult,
)


@dataclass
class PipelineConfig:
    """Configuration for export pipeline."""

    # Global settings
    output_base_path: str
    formats: List[ExportFormat]

    # Processing settings
    max_parallel_exports: int = 2
    validate_all_outputs: bool = True
    fail_fast: bool = False  # Stop on first error

    # Format-specific configs
    format_configs: Dict[ExportFormat, Dict[str, Any]] = field(default_factory=dict)

    # Data source settings
    data_source_config: Dict[str, Any] = field(default_factory=dict)

    # Pipeline metadata
    pipeline_name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, config_path: str) -> "PipelineConfig":
        """Load pipeline configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Convert format strings to enums
        formats = [ExportFormat(fmt) for fmt in data.get("formats", [])]

        # Convert format config keys to enums
        format_configs = {}
        for fmt_str, config in data.get("format_configs", {}).items():
            format_configs[ExportFormat(fmt_str)] = config

        return cls(
            output_base_path=data["output_base_path"],
            formats=formats,
            max_parallel_exports=data.get("max_parallel_exports", 2),
            validate_all_outputs=data.get("validate_all_outputs", True),
            fail_fast=data.get("fail_fast", False),
            format_configs=format_configs,
            data_source_config=data.get("data_source_config", {}),
            pipeline_name=data.get("pipeline_name"),
            description=data.get("description"),
            tags=data.get("tags", []),
        )

    def to_file(self, config_path: str) -> None:
        """Save pipeline configuration to file."""
        data = {
            "output_base_path": self.output_base_path,
            "formats": [fmt.value for fmt in self.formats],
            "max_parallel_exports": self.max_parallel_exports,
            "validate_all_outputs": self.validate_all_outputs,
            "fail_fast": self.fail_fast,
            "format_configs": {fmt.value: config for fmt, config in self.format_configs.items()},
            "data_source_config": self.data_source_config,
            "pipeline_name": self.pipeline_name,
            "description": self.description,
            "tags": self.tags,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    pipeline_name: Optional[str]
    status: ExportStatus
    start_time: datetime
    end_time: Optional[datetime] = None

    # Export results by format
    export_results: Dict[ExportFormat, ExportResult] = field(default_factory=dict)

    # Pipeline-level validation
    pipeline_validation: Optional[ValidationResult] = None

    # Error information
    error: Optional[ExportError] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate pipeline duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def successful_exports(self) -> List[ExportFormat]:
        """Get list of successful exports."""
        return [
            fmt
            for fmt, result in self.export_results.items()
            if result.status == ExportStatus.COMPLETED
        ]

    @property
    def failed_exports(self) -> List[ExportFormat]:
        """Get list of failed exports."""
        return [
            fmt
            for fmt, result in self.export_results.items()
            if result.status == ExportStatus.FAILED
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "successful_exports": [fmt.value for fmt in self.successful_exports],
            "failed_exports": [fmt.value for fmt in self.failed_exports],
            "export_results": {
                fmt.value: result.to_dict() for fmt, result in self.export_results.items()
            },
            "pipeline_validation": (
                {
                    "is_valid": self.pipeline_validation.is_valid,
                    "error_count": len(self.pipeline_validation.errors),
                    "warning_count": len(self.pipeline_validation.warnings),
                }
                if self.pipeline_validation
                else None
            ),
            "error": (
                {"message": self.error.message, "stage": self.error.stage} if self.error else None
            ),
        }


class ExportPipeline:
    """Orchestrates multi-format export operations."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExportPipeline")

        # Registry of available exporters
        self._exporter_registry: Dict[ExportFormat, type] = {}
        self._register_default_exporters()

    def _register_default_exporters(self):
        """Register default exporter implementations."""
        # Import here to avoid circular dependencies
        try:
            from .sqlite_exporter import SQLiteExporter

            self._exporter_registry[ExportFormat.SQLITE] = SQLiteExporter
        except ImportError:
            self.logger.warning("SQLite exporter not available")

        try:
            from .json_exporter import StaticJSONExporter

            self._exporter_registry[ExportFormat.STATIC_JSON] = StaticJSONExporter
        except ImportError:
            self.logger.warning("Static JSON exporter not available")

        try:
            from .opensearch_exporter import OpenSearchExporter

            self._exporter_registry[ExportFormat.OPENSEARCH] = OpenSearchExporter
        except ImportError:
            self.logger.warning("OpenSearch exporter not available")

        try:
            from .graph_exporter import Neo4jExporter, ArangoExporter

            self._exporter_registry[ExportFormat.NEO4J] = Neo4jExporter
            self._exporter_registry[ExportFormat.ARANGODB] = ArangoExporter
        except ImportError:
            self.logger.warning("Graph database exporters not available")

    def register_exporter(self, format_type: ExportFormat, exporter_class: type):
        """Register custom exporter implementation."""
        self._exporter_registry[format_type] = exporter_class
        self.logger.info(f"Registered custom exporter for {format_type.value}")

    async def run(self, data: CanonicalDataset) -> PipelineResult:
        """Execute the export pipeline."""
        result = PipelineResult(
            pipeline_name=self.config.pipeline_name,
            status=ExportStatus.IN_PROGRESS,
            start_time=datetime.now(),
        )

        try:
            self.logger.info(f"Starting export pipeline: {self.config.pipeline_name}")
            self.logger.info(f"Target formats: {[fmt.value for fmt in self.config.formats]}")

            # Validate pipeline configuration
            pipeline_validation = await self._validate_pipeline()
            if not pipeline_validation.is_valid:
                raise ExportError(
                    "Pipeline configuration validation failed",
                    stage="validation",
                    details={"errors": pipeline_validation.errors},
                )

            # Prepare output directory
            self._prepare_output_directory()

            # Execute exports
            if self.config.max_parallel_exports > 1:
                await self._run_parallel_exports(data, result)
            else:
                await self._run_sequential_exports(data, result)

            # Validate pipeline results
            if self.config.validate_all_outputs:
                result.pipeline_validation = await self._validate_pipeline_results(result)

            # Determine final status
            if result.failed_exports and self.config.fail_fast:
                result.status = ExportStatus.FAILED
            elif result.failed_exports:
                result.status = ExportStatus.COMPLETED  # Partial success
            else:
                result.status = ExportStatus.COMPLETED

            # Generate pipeline report
            await self._generate_pipeline_report(result)

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            result.status = ExportStatus.FAILED
            result.error = ExportError(message=str(e), stage="pipeline_execution")

        finally:
            result.end_time = datetime.now()
            self.logger.info(
                f"Pipeline completed in {result.duration_seconds:.2f}s. "
                f"Status: {result.status.value}"
            )

        return result

    async def _run_parallel_exports(self, data: CanonicalDataset, result: PipelineResult):
        """Execute exports in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_parallel_exports)

        async def run_single_export(format_type: ExportFormat):
            async with semaphore:
                return await self._execute_single_export(format_type, data)

        # Create tasks for all exports
        tasks = [
            asyncio.create_task(run_single_export(fmt), name=f"export_{fmt.value}")
            for fmt in self.config.formats
        ]

        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for fmt, export_result in zip(self.config.formats, completed_results):
            if isinstance(export_result, Exception):
                result.export_results[fmt] = ExportResult(
                    format_type=fmt,
                    status=ExportStatus.FAILED,
                    output_path="",
                    error=ExportError(
                        message=str(export_result), format_type=fmt, stage="export_execution"
                    ),
                )
            else:
                result.export_results[fmt] = export_result

            if self.config.fail_fast and result.export_results[fmt].status == ExportStatus.FAILED:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break

    async def _run_sequential_exports(self, data: CanonicalDataset, result: PipelineResult):
        """Execute exports sequentially."""
        for format_type in self.config.formats:
            try:
                export_result = await self._execute_single_export(format_type, data)
                result.export_results[format_type] = export_result

                if self.config.fail_fast and export_result.status == ExportStatus.FAILED:
                    break

            except Exception as e:
                self.logger.error(f"Export failed for {format_type.value}: {str(e)}")
                result.export_results[format_type] = ExportResult(
                    format_type=format_type,
                    status=ExportStatus.FAILED,
                    output_path="",
                    error=ExportError(
                        message=str(e), format_type=format_type, stage="export_execution"
                    ),
                )

                if self.config.fail_fast:
                    break

    async def _execute_single_export(
        self, format_type: ExportFormat, data: CanonicalDataset
    ) -> ExportResult:
        """Execute export for a single format."""
        self.logger.info(f"Starting {format_type.value} export")

        # Get exporter class
        if format_type not in self._exporter_registry:
            raise ExportError(
                f"No exporter registered for format: {format_type.value}",
                format_type=format_type,
                stage="exporter_lookup",
            )

        exporter_class = self._exporter_registry[format_type]

        # Create exporter configuration
        export_config = self._create_export_config(format_type)

        # Create and run exporter
        exporter = exporter_class(export_config)

        # Validate exporter configuration
        config_validation = exporter.validate_config()
        if not config_validation.is_valid:
            raise ExportError(
                f"Exporter configuration validation failed: {config_validation.errors}",
                format_type=format_type,
                stage="config_validation",
            )

        # Execute export
        result = await exporter.export(data)

        self.logger.info(f"Completed {format_type.value} export: {result.status.value}")
        return result

    def _create_export_config(self, format_type: ExportFormat) -> ExportConfig:
        """Create export configuration for specific format."""
        # Get format-specific config
        format_config = self.config.format_configs.get(format_type, {})

        # Create output path
        output_path = Path(self.config.output_base_path) / format_type.value

        # Create base configuration
        config = ExportConfig(
            output_path=str(output_path), format_type=format_type, **format_config
        )

        return config

    async def _validate_pipeline(self) -> ValidationResult:
        """Validate pipeline configuration."""
        validation = ValidationResult(is_valid=True)

        # Check output base path
        if not self.config.output_base_path:
            validation.add_error("Output base path is required")

        # Check formats
        if not self.config.formats:
            validation.add_error("At least one export format must be specified")

        # Check exporter availability
        for format_type in self.config.formats:
            if format_type not in self._exporter_registry:
                validation.add_error(f"No exporter available for format: {format_type.value}")

        # Validate format-specific configurations
        for format_type in self.config.formats:
            if format_type in self._exporter_registry:
                try:
                    export_config = self._create_export_config(format_type)
                    exporter = self._exporter_registry[format_type](export_config)
                    format_validation = exporter.validate_config()

                    if not format_validation.is_valid:
                        for error in format_validation.errors:
                            validation.add_error(f"{format_type.value}: {error}")

                except Exception as e:
                    validation.add_error(f"{format_type.value}: Configuration error: {str(e)}")

        return validation

    def _prepare_output_directory(self):
        """Prepare output directory structure."""
        base_path = Path(self.config.output_base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Create format-specific directories
        for format_type in self.config.formats:
            format_path = base_path / format_type.value
            format_path.mkdir(exist_ok=True)

    async def _validate_pipeline_results(self, result: PipelineResult) -> ValidationResult:
        """Validate overall pipeline results."""
        validation = ValidationResult(is_valid=True)

        # Check that all requested formats completed
        for format_type in self.config.formats:
            if format_type not in result.export_results:
                validation.add_error(f"Missing export result for {format_type.value}")
            elif result.export_results[format_type].status == ExportStatus.FAILED:
                validation.add_error(f"Export failed for {format_type.value}")

        # Validate individual exports
        for format_type, export_result in result.export_results.items():
            if export_result.validation and not export_result.validation.is_valid:
                for error in export_result.validation.errors:
                    validation.add_error(f"{format_type.value}: {error}")

        return validation

    async def _generate_pipeline_report(self, result: PipelineResult):
        """Generate comprehensive pipeline report."""
        report_path = Path(self.config.output_base_path) / "pipeline_report.json"

        report_data = {
            "pipeline_config": {
                "name": self.config.pipeline_name,
                "description": self.config.description,
                "formats": [fmt.value for fmt in self.config.formats],
                "tags": self.config.tags,
            },
            "execution_summary": result.to_dict(),
            "format_details": {},
        }

        # Add detailed format information
        for format_type, export_result in result.export_results.items():
            report_data["format_details"][format_type.value] = {
                "status": export_result.status.value,
                "output_path": export_result.output_path,
                "stats": export_result.stats.__dict__ if export_result.stats else {},
                "validation": (
                    {
                        "is_valid": export_result.validation.is_valid,
                        "errors": export_result.validation.errors,
                        "warnings": export_result.validation.warnings,
                    }
                    if export_result.validation
                    else None
                ),
            }

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Pipeline report generated: {report_path}")
