"""
Multi-format export system for ABBA canonical data.

This package provides comprehensive export functionality to transform
canonical biblical data into optimized formats for different use cases:

- SQLite: Mobile apps, offline access, embedded systems
- Static JSON: CDNs, static websites, serverless architectures  
- OpenSearch: Full-text search, analytics, large-scale web apps
- Graph DB: Relationship traversal, research, complex queries
"""

from .base import DataExporter, ExportConfig, ExportResult, ValidationResult, ExportError

from .pipeline import ExportPipeline, PipelineConfig, PipelineResult

from .sqlite_exporter import SQLiteExporter, SQLiteConfig

from .json_exporter import StaticJSONExporter, JSONConfig

from .opensearch_exporter import OpenSearchExporter, OpenSearchConfig

from .graph_exporter import GraphExporter, GraphConfig, Neo4jExporter, ArangoExporter

from .validation import ExportValidator, IntegrityChecker, PerformanceBenchmark

__all__ = [
    # Base framework
    "DataExporter",
    "ExportConfig",
    "ExportResult",
    "ValidationResult",
    "ExportError",
    # Pipeline
    "ExportPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Format exporters
    "SQLiteExporter",
    "SQLiteConfig",
    "StaticJSONExporter",
    "JSONConfig",
    "OpenSearchExporter",
    "OpenSearchConfig",
    "GraphExporter",
    "GraphConfig",
    "Neo4jExporter",
    "ArangoExporter",
    # Validation
    "ExportValidator",
    "IntegrityChecker",
    "PerformanceBenchmark",
]
