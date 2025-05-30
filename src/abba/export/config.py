"""
Configuration management for ABBA export system.

Provides centralized configuration management, validation, and templating
for all export formats with support for environment variables and profiles.
"""

import os
import logging
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum

from .base import ExportFormat
from .sqlite_exporter import SQLiteConfig
from .json_exporter import JSONConfig
from .opensearch_exporter import OpenSearchConfig
from .graph_exporter import Neo4jConfig, ArangoConfig
from .pipeline import PipelineConfig


class ConfigProfile(Enum):
    """Predefined configuration profiles."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    RESEARCH = "research"
    MOBILE = "mobile"
    WEB = "web"


@dataclass
class ExportConfigTemplate:
    """Template for export configuration."""

    name: str
    description: str
    profile: ConfigProfile
    formats: List[ExportFormat]
    base_settings: Dict[str, Any] = field(default_factory=dict)
    format_settings: Dict[ExportFormat, Dict[str, Any]] = field(default_factory=dict)


class ConfigManager:
    """Manages export configurations with profiles and templates."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(f"{__name__}.ConfigManager")

        # Configuration directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / ".abba" / "export_configs"

        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Built-in templates
        self._templates = self._create_builtin_templates()

        # Environment variable mappings
        self._env_mappings = self._create_env_mappings()

    def _create_builtin_templates(self) -> Dict[str, ExportConfigTemplate]:
        """Create built-in configuration templates."""
        templates = {}

        # Development template
        templates["development"] = ExportConfigTemplate(
            name="Development",
            description="Fast development and testing configuration",
            profile=ConfigProfile.DEVELOPMENT,
            formats=[ExportFormat.SQLITE, ExportFormat.STATIC_JSON],
            base_settings={
                "batch_size": 500,
                "validate_output": True,
                "compress_output": False,
                "optimize_for_speed": True,
            },
            format_settings={
                ExportFormat.SQLITE: {
                    "enable_fts5": True,
                    "enable_wal_mode": True,
                    "vacuum_on_completion": False,
                },
                ExportFormat.STATIC_JSON: {
                    "chunk_size": 50,
                    "gzip_output": False,
                    "minify_json": False,
                    "create_search_indices": True,
                },
            },
        )

        # Production template
        templates["production"] = ExportConfigTemplate(
            name="Production",
            description="Optimized production configuration",
            profile=ConfigProfile.PRODUCTION,
            formats=[ExportFormat.SQLITE, ExportFormat.STATIC_JSON, ExportFormat.OPENSEARCH],
            base_settings={
                "batch_size": 1000,
                "validate_output": True,
                "compress_output": True,
                "optimize_for_size": True,
                "parallel_workers": 4,
            },
            format_settings={
                ExportFormat.SQLITE: {
                    "enable_fts5": True,
                    "enable_wal_mode": True,
                    "vacuum_on_completion": True,
                    "analyze_on_completion": True,
                    "compress_large_text": True,
                },
                ExportFormat.STATIC_JSON: {
                    "chunk_size": 100,
                    "gzip_output": True,
                    "minify_json": True,
                    "create_search_indices": True,
                    "progressive_loading": True,
                },
                ExportFormat.OPENSEARCH: {
                    "bulk_size": 1000,
                    "enable_biblical_analyzer": True,
                    "number_of_replicas": 1,
                    "refresh_after_import": True,
                },
            },
        )

        # Research template
        templates["research"] = ExportConfigTemplate(
            name="Research",
            description="Comprehensive configuration for biblical research",
            profile=ConfigProfile.RESEARCH,
            formats=[ExportFormat.SQLITE, ExportFormat.NEO4J, ExportFormat.OPENSEARCH],
            base_settings={
                "batch_size": 500,
                "validate_output": True,
                "benchmark_performance": True,
            },
            format_settings={
                ExportFormat.SQLITE: {
                    "enable_fts5": True,
                    "vacuum_on_completion": True,
                    "analyze_on_completion": True,
                },
                ExportFormat.NEO4J: {
                    "create_indices": True,
                    "compute_graph_metrics": True,
                    "enable_relationship_properties": True,
                },
                ExportFormat.OPENSEARCH: {
                    "enable_biblical_analyzer": True,
                    "enable_multilingual_support": True,
                    "bulk_size": 500,
                },
            },
        )

        # Mobile template
        templates["mobile"] = ExportConfigTemplate(
            name="Mobile",
            description="Optimized for mobile applications",
            profile=ConfigProfile.MOBILE,
            formats=[ExportFormat.SQLITE],
            base_settings={"batch_size": 1000, "compress_output": True, "optimize_for_size": True},
            format_settings={
                ExportFormat.SQLITE: {
                    "enable_fts5": True,
                    "vacuum_on_completion": True,
                    "compress_large_text": True,
                    "compression_threshold": 512,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
        )

        # Web template
        templates["web"] = ExportConfigTemplate(
            name="Web",
            description="Optimized for web applications",
            profile=ConfigProfile.WEB,
            formats=[ExportFormat.STATIC_JSON, ExportFormat.OPENSEARCH],
            base_settings={"batch_size": 1000, "compress_output": True, "parallel_workers": 4},
            format_settings={
                ExportFormat.STATIC_JSON: {
                    "chunk_size": 100,
                    "gzip_output": True,
                    "minify_json": True,
                    "create_search_indices": True,
                    "progressive_loading": True,
                },
                ExportFormat.OPENSEARCH: {
                    "bulk_size": 1000,
                    "enable_biblical_analyzer": True,
                    "refresh_after_import": True,
                },
            },
        )

        return templates

    def _create_env_mappings(self) -> Dict[str, str]:
        """Create environment variable mappings."""
        return {
            # Global settings
            "ABBA_EXPORT_BATCH_SIZE": "batch_size",
            "ABBA_EXPORT_PARALLEL_WORKERS": "parallel_workers",
            "ABBA_EXPORT_OUTPUT_PATH": "output_base_path",
            # SQLite settings
            "ABBA_SQLITE_ENABLE_FTS5": "sqlite.enable_fts5",
            "ABBA_SQLITE_WAL_MODE": "sqlite.enable_wal_mode",
            # JSON settings
            "ABBA_JSON_CHUNK_SIZE": "static_json.chunk_size",
            "ABBA_JSON_GZIP": "static_json.gzip_output",
            # OpenSearch settings
            "ABBA_OPENSEARCH_URL": "opensearch.cluster_url",
            "ABBA_OPENSEARCH_USERNAME": "opensearch.username",
            "ABBA_OPENSEARCH_PASSWORD": "opensearch.password",
            "ABBA_OPENSEARCH_BULK_SIZE": "opensearch.bulk_size",
            # Neo4j settings
            "ABBA_NEO4J_URL": "neo4j.server_url",
            "ABBA_NEO4J_USERNAME": "neo4j.username",
            "ABBA_NEO4J_PASSWORD": "neo4j.password",
            "ABBA_NEO4J_DATABASE": "neo4j.database",
            # ArangoDB settings
            "ABBA_ARANGO_URL": "arangodb.server_url",
            "ABBA_ARANGO_USERNAME": "arangodb.username",
            "ABBA_ARANGO_PASSWORD": "arangodb.password",
            "ABBA_ARANGO_DATABASE": "arangodb.database",
        }

    def get_template(self, template_name: str) -> Optional[ExportConfigTemplate]:
        """Get configuration template by name."""
        return self._templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List available configuration templates."""
        return list(self._templates.keys())

    def create_pipeline_config(
        self, template_name: str, output_path: str, overrides: Optional[Dict[str, Any]] = None
    ) -> PipelineConfig:
        """Create pipeline configuration from template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        self.logger.info(f"Creating pipeline config from template: {template_name}")

        # Start with base settings
        config_data = {
            "output_base_path": output_path,
            "formats": template.formats,
            "pipeline_name": f"ABBA Export - {template.name}",
            "description": template.description,
            **template.base_settings,
        }

        # Add format-specific configurations
        format_configs = {}
        for format_type in template.formats:
            format_config = template.format_settings.get(format_type, {})

            # Apply environment variable overrides
            format_config = self._apply_env_overrides(format_config, format_type)

            format_configs[format_type] = format_config

        config_data["format_configs"] = format_configs

        # Apply user overrides
        if overrides:
            config_data = self._merge_configs(config_data, overrides)

        # Create pipeline config
        return PipelineConfig(**config_data)

    def _apply_env_overrides(
        self, config: Dict[str, Any], format_type: ExportFormat
    ) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        config = config.copy()

        for env_var, config_path in self._env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Check if this environment variable applies to this format
                if config_path.startswith(f"{format_type.value}."):
                    key = config_path.split(".", 1)[1]

                    # Convert string values to appropriate types
                    if key in ["enable_fts5", "enable_wal_mode", "gzip_output", "minify_json"]:
                        config[key] = value.lower() in ("true", "1", "yes")
                    elif key in ["chunk_size", "bulk_size", "batch_size", "parallel_workers"]:
                        config[key] = int(value)
                    else:
                        config[key] = value

        return config

    def _merge_configs(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        result = base.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, config: PipelineConfig, name: str) -> Path:
        """Save pipeline configuration to file."""
        config_file = self.config_dir / f"{name}.yaml"

        self.logger.info(f"Saving configuration: {config_file}")

        # Convert to dictionary
        config_dict = {
            "pipeline_name": config.pipeline_name,
            "description": config.description,
            "output_base_path": config.output_base_path,
            "formats": [fmt.value for fmt in config.formats],
            "max_parallel_exports": config.max_parallel_exports,
            "validate_all_outputs": config.validate_all_outputs,
            "fail_fast": config.fail_fast,
            "format_configs": {fmt.value: cfg for fmt, cfg in config.format_configs.items()},
            "data_source_config": config.data_source_config,
            "tags": config.tags,
        }

        # Write to file
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        return config_file

    def load_config(self, name: str) -> PipelineConfig:
        """Load pipeline configuration from file."""
        config_file = self.config_dir / f"{name}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        self.logger.info(f"Loading configuration: {config_file}")

        return PipelineConfig.from_file(str(config_file))

    def list_saved_configs(self) -> List[str]:
        """List saved configuration files."""
        if not self.config_dir.exists():
            return []

        config_files = self.config_dir.glob("*.yaml")
        return [f.stem for f in config_files]

    def validate_config(self, config: PipelineConfig) -> List[str]:
        """Validate pipeline configuration."""
        errors = []

        # Check required fields
        if not config.output_base_path:
            errors.append("Output base path is required")

        if not config.formats:
            errors.append("At least one export format must be specified")

        # Validate format-specific configurations
        for format_type in config.formats:
            format_config = config.format_configs.get(format_type, {})
            format_errors = self._validate_format_config(format_type, format_config)
            errors.extend([f"{format_type.value}: {error}" for error in format_errors])

        return errors

    def _validate_format_config(
        self, format_type: ExportFormat, config: Dict[str, Any]
    ) -> List[str]:
        """Validate format-specific configuration."""
        errors = []

        if format_type == ExportFormat.SQLITE:
            if "batch_size" in config and config["batch_size"] <= 0:
                errors.append("Batch size must be positive")

        elif format_type == ExportFormat.STATIC_JSON:
            if "chunk_size" in config and config["chunk_size"] <= 0:
                errors.append("Chunk size must be positive")

        elif format_type == ExportFormat.OPENSEARCH:
            if "cluster_url" in config and not config["cluster_url"]:
                errors.append("OpenSearch cluster URL is required")
            if "bulk_size" in config and config["bulk_size"] <= 0:
                errors.append("Bulk size must be positive")

        elif format_type == ExportFormat.NEO4J:
            if "server_url" in config and not config["server_url"]:
                errors.append("Neo4j server URL is required")

        elif format_type == ExportFormat.ARANGODB:
            if "server_url" in config and not config["server_url"]:
                errors.append("ArangoDB server URL is required")

        return errors

    def create_format_config(self, format_type: ExportFormat, config_dict: Dict[str, Any]) -> Any:
        """Create format-specific configuration object."""
        if format_type == ExportFormat.SQLITE:
            return SQLiteConfig(**config_dict)
        elif format_type == ExportFormat.STATIC_JSON:
            return JSONConfig(**config_dict)
        elif format_type == ExportFormat.OPENSEARCH:
            return OpenSearchConfig(**config_dict)
        elif format_type == ExportFormat.NEO4J:
            return Neo4jConfig(**config_dict)
        elif format_type == ExportFormat.ARANGODB:
            return ArangoConfig(**config_dict)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def export_template_as_config(self, template_name: str, output_path: str) -> str:
        """Export template as a configuration file example."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Create example configuration
        config = {
            "# ABBA Export Configuration": None,
            "# Generated from template": template_name,
            "pipeline_name": f"ABBA Export - {template.name}",
            "description": template.description,
            "output_base_path": output_path or "/path/to/output",
            "formats": [fmt.value for fmt in template.formats],
            **template.base_settings,
            "format_configs": {
                fmt.value: settings for fmt, settings in template.format_settings.items()
            },
        }

        # Convert to YAML string
        yaml_str = yaml.dump(config, default_flow_style=False, indent=2)

        # Add comments for environment variables
        env_comment = "\n# Environment Variables:\n"
        for env_var, config_path in self._env_mappings.items():
            env_comment += f"# {env_var} -> {config_path}\n"

        return yaml_str + env_comment

    def create_docker_compose_config(
        self, template_name: str, include_databases: bool = True
    ) -> str:
        """Create Docker Compose configuration for export infrastructure."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        services = {}

        # Add database services based on formats
        if include_databases:
            if ExportFormat.OPENSEARCH in template.formats:
                services["opensearch"] = {
                    "image": "opensearchproject/opensearch:2.11.0",
                    "environment": [
                        "discovery.type=single-node",
                        "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin123",
                        "DISABLE_SECURITY_PLUGIN=true",
                    ],
                    "ports": ["9200:9200"],
                    "volumes": ["opensearch_data:/usr/share/opensearch/data"],
                }

            if ExportFormat.NEO4J in template.formats:
                services["neo4j"] = {
                    "image": "neo4j:5.13",
                    "environment": ["NEO4J_AUTH=neo4j/password", 'NEO4J_PLUGINS=["apoc"]'],
                    "ports": ["7474:7474", "7687:7687"],
                    "volumes": ["neo4j_data:/data"],
                }

            if ExportFormat.ARANGODB in template.formats:
                services["arangodb"] = {
                    "image": "arangodb:3.11",
                    "environment": ["ARANGO_ROOT_PASSWORD=password"],
                    "ports": ["8529:8529"],
                    "volumes": ["arango_data:/var/lib/arangodb3"],
                }

        # Create volumes
        volumes = {}
        if "opensearch" in services:
            volumes["opensearch_data"] = {}
        if "neo4j" in services:
            volumes["neo4j_data"] = {}
        if "arangodb" in services:
            volumes["arango_data"] = {}

        # Create docker-compose structure
        docker_compose = {"version": "3.8", "services": services}

        if volumes:
            docker_compose["volumes"] = volumes

        return yaml.dump(docker_compose, default_flow_style=False, indent=2)
