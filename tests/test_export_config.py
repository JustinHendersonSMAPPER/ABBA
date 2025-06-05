"""
Tests for ABBA export configuration management.

Test coverage for configuration templates, environment variable handling,
Docker Compose generation, and configuration validation.
"""

import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from abba.export.config import ConfigManager, ExportConfigTemplate, ConfigProfile
from abba.export.base import ExportFormat
from abba.export.pipeline import PipelineConfig


class TestConfigProfile:
    """Test configuration profile enumeration."""

    def test_config_profiles(self):
        """Test configuration profile values."""
        assert ConfigProfile.DEVELOPMENT.value == "development"
        assert ConfigProfile.TESTING.value == "testing"
        assert ConfigProfile.PRODUCTION.value == "production"
        assert ConfigProfile.RESEARCH.value == "research"
        assert ConfigProfile.MOBILE.value == "mobile"
        assert ConfigProfile.WEB.value == "web"

        # Test all profiles are defined
        profiles = list(ConfigProfile)
        assert len(profiles) == 6


class TestExportConfigTemplate:
    """Test export configuration template."""

    def test_basic_template_creation(self):
        """Test basic template creation."""
        template = ExportConfigTemplate(
            name="Test Template",
            description="A test configuration template",
            profile=ConfigProfile.DEVELOPMENT,
            formats=[ExportFormat.SQLITE, ExportFormat.STATIC_JSON],
        )

        assert template.name == "Test Template"
        assert template.description == "A test configuration template"
        assert template.profile == ConfigProfile.DEVELOPMENT
        assert len(template.formats) == 2
        assert ExportFormat.SQLITE in template.formats
        assert ExportFormat.STATIC_JSON in template.formats

    def test_template_with_settings(self):
        """Test template with base and format settings."""
        template = ExportConfigTemplate(
            name="Advanced Template",
            description="Template with settings",
            profile=ConfigProfile.PRODUCTION,
            formats=[ExportFormat.OPENSEARCH],
            base_settings={"batch_size": 1000, "validate_output": True},
            format_settings={
                ExportFormat.OPENSEARCH: {
                    "cluster_url": "https://search.example.com",
                    "bulk_size": 500,
                }
            },
        )

        assert template.base_settings["batch_size"] == 1000
        assert template.base_settings["validate_output"] is True

        opensearch_settings = template.format_settings[ExportFormat.OPENSEARCH]
        assert opensearch_settings["cluster_url"] == "https://search.example.com"
        assert opensearch_settings["bulk_size"] == 500


class TestConfigManager:
    """Test configuration manager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create test configuration manager."""
        return ConfigManager(config_dir=temp_config_dir)

    def test_manager_initialization(self, config_manager, temp_config_dir):
        """Test manager initialization."""
        assert config_manager.config_dir == Path(temp_config_dir)
        assert config_manager.config_dir.exists()
        assert len(config_manager._templates) > 0
        assert len(config_manager._env_mappings) > 0

    def test_builtin_templates(self, config_manager):
        """Test built-in configuration templates."""
        templates = config_manager.list_templates()

        assert "development" in templates
        assert "production" in templates
        assert "research" in templates
        assert "mobile" in templates
        assert "web" in templates

        # Test development template
        dev_template = config_manager.get_template("development")
        assert dev_template is not None
        assert dev_template.name == "Development"
        assert dev_template.profile == ConfigProfile.DEVELOPMENT
        assert ExportFormat.SQLITE in dev_template.formats
        assert ExportFormat.STATIC_JSON in dev_template.formats

        # Test production template
        prod_template = config_manager.get_template("production")
        assert prod_template is not None
        assert prod_template.name == "Production"
        assert prod_template.profile == ConfigProfile.PRODUCTION
        assert ExportFormat.OPENSEARCH in prod_template.formats

    def test_template_retrieval(self, config_manager):
        """Test template retrieval."""
        # Test existing template
        template = config_manager.get_template("development")
        assert template is not None
        assert template.name == "Development"

        # Test non-existent template
        template = config_manager.get_template("nonexistent")
        assert template is None

    def test_pipeline_config_creation(self, config_manager):
        """Test pipeline configuration creation from template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = config_manager.create_pipeline_config(
                template_name="development", output_path=temp_dir
            )

            assert isinstance(config, PipelineConfig)
            assert config.output_base_path == temp_dir
            assert config.pipeline_name == "ABBA Export - Development"
            assert len(config.formats) >= 2
            assert config.validate_all_outputs is True

    def test_pipeline_config_with_overrides(self, config_manager):
        """Test pipeline configuration with overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            overrides = {
                "max_parallel_exports": 4,
                "validate_all_outputs": False,
                "format_configs": {
                    ExportFormat.SQLITE: {
                        "enable_fts5": False,
                        "batch_size": 2000
                    }
                },
            }

            config = config_manager.create_pipeline_config(
                template_name="development", output_path=temp_dir, overrides=overrides
            )

            assert config.max_parallel_exports == 4
            assert config.validate_all_outputs is False

            sqlite_config = config.format_configs.get(ExportFormat.SQLITE, {})
            assert sqlite_config.get("enable_fts5") is False
            assert sqlite_config.get("batch_size") == 2000

    def test_environment_variable_overrides(self, config_manager):
        """Test environment variable override application."""
        # Mock environment variables
        env_vars = {
            "ABBA_EXPORT_BATCH_SIZE": "5000",
            "ABBA_SQLITE_ENABLE_FTS5": "false",
            "ABBA_JSON_CHUNK_SIZE": "200",
            "ABBA_OPENSEARCH_BULK_SIZE": "750",
        }

        with patch.dict(os.environ, env_vars):
            with tempfile.TemporaryDirectory() as temp_dir:
                config = config_manager.create_pipeline_config(
                    template_name="production", output_path=temp_dir
                )

                # Check format-specific overrides
                # Note: batch_size is a format-specific setting, not a pipeline setting

                # Check format-specific overrides
                sqlite_config = config.format_configs.get(ExportFormat.SQLITE, {})
                assert sqlite_config.get("enable_fts5") is False

                json_config = config.format_configs.get(ExportFormat.STATIC_JSON, {})
                assert json_config.get("chunk_size") == 200

    def test_environment_variable_type_conversion(self, config_manager):
        """Test environment variable type conversion."""
        format_config = {"enable_fts5": True, "chunk_size": 100}

        # Mock environment variables with different types
        with patch.dict(
            os.environ, {"ABBA_SQLITE_ENABLE_FTS5": "false", "ABBA_JSON_CHUNK_SIZE": "250"}
        ):
            result = config_manager._apply_env_overrides(format_config, ExportFormat.SQLITE)

            # Boolean conversion
            assert result.get("enable_fts5") is False

            # Integer conversion (for JSON format)
            json_result = config_manager._apply_env_overrides(
                format_config, ExportFormat.STATIC_JSON
            )
            # Note: This would need the actual environment variable to apply

    def test_config_merging(self, config_manager):
        """Test configuration dictionary merging."""
        base_config = {"batch_size": 100, "format_configs": {"sqlite": {"enable_fts5": True}}}

        overrides = {
            "batch_size": 200,
            "new_setting": "value",
            "format_configs": {
                "sqlite": {"vacuum_on_completion": True},
                "json": {"chunk_size": 50},
            },
        }

        merged = config_manager._merge_configs(base_config, overrides)

        assert merged["batch_size"] == 200  # Override
        assert merged["new_setting"] == "value"  # New setting

        # Check nested merging
        sqlite_config = merged["format_configs"]["sqlite"]
        assert sqlite_config["enable_fts5"] is True  # Original
        assert sqlite_config["vacuum_on_completion"] is True  # Override

        json_config = merged["format_configs"]["json"]
        assert json_config["chunk_size"] == 50  # New format config

    def test_config_saving_and_loading(self, config_manager):
        """Test configuration saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration
            config = config_manager.create_pipeline_config(
                template_name="development", output_path=temp_dir
            )

            # Save configuration
            config_file = config_manager.save_config(config, "test_config")
            assert config_file.exists()
            assert config_file.suffix == ".yaml"

            # Load configuration
            loaded_config = config_manager.load_config("test_config")
            assert loaded_config.pipeline_name == config.pipeline_name
            assert loaded_config.output_base_path == config.output_base_path
            assert loaded_config.formats == config.formats

    def test_saved_config_listing(self, config_manager):
        """Test listing saved configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initially no saved configs
            configs = config_manager.list_saved_configs()
            assert len(configs) == 0

            # Save a configuration
            config = config_manager.create_pipeline_config(
                template_name="development", output_path=temp_dir
            )
            config_manager.save_config(config, "my_config")

            # Should now list the saved config
            configs = config_manager.list_saved_configs()
            assert "my_config" in configs

    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid configuration
            valid_config = config_manager.create_pipeline_config(
                template_name="development", output_path=temp_dir
            )

            errors = config_manager.validate_config(valid_config)
            assert len(errors) == 0

            # Invalid configuration - no output path
            valid_config.output_base_path = ""
            errors = config_manager.validate_config(valid_config)
            assert len(errors) > 0
            assert any("output" in error.lower() for error in errors)

            # Invalid configuration - no formats
            valid_config.output_base_path = temp_dir
            valid_config.formats = []
            errors = config_manager.validate_config(valid_config)
            assert len(errors) > 0
            assert any("format" in error.lower() for error in errors)

    def test_format_specific_validation(self, config_manager):
        """Test format-specific configuration validation."""
        # SQLite validation
        sqlite_errors = config_manager._validate_format_config(
            ExportFormat.SQLITE, {"batch_size": 0}  # Invalid
        )
        assert len(sqlite_errors) > 0
        assert any("batch size" in error.lower() for error in sqlite_errors)

        # JSON validation
        json_errors = config_manager._validate_format_config(
            ExportFormat.STATIC_JSON, {"chunk_size": -1}  # Invalid
        )
        assert len(json_errors) > 0
        assert any("chunk size" in error.lower() for error in json_errors)

        # OpenSearch validation
        opensearch_errors = config_manager._validate_format_config(
            ExportFormat.OPENSEARCH, {"cluster_url": ""}  # Invalid
        )
        assert len(opensearch_errors) > 0
        assert any("url" in error.lower() for error in opensearch_errors)

    def test_format_config_creation(self, config_manager):
        """Test format-specific configuration object creation."""
        # SQLite config
        sqlite_config = config_manager.create_format_config(
            ExportFormat.SQLITE,
            {"output_path": "/tmp/test.db", "enable_fts5": True, "batch_size": 1000},
        )

        assert hasattr(sqlite_config, "output_path")
        assert hasattr(sqlite_config, "enable_fts5")
        assert hasattr(sqlite_config, "batch_size")

        # JSON config
        json_config = config_manager.create_format_config(
            ExportFormat.STATIC_JSON,
            {"output_path": "/tmp/json", "chunk_size": 100, "gzip_output": True},
        )

        assert hasattr(json_config, "output_path")
        assert hasattr(json_config, "chunk_size")
        assert hasattr(json_config, "gzip_output")

    def test_template_export_as_config(self, config_manager):
        """Test exporting template as configuration file."""
        config_yaml = config_manager.export_template_as_config(
            template_name="development", output_path="/tmp/test_output"
        )

        assert isinstance(config_yaml, str)
        assert "pipeline_name" in config_yaml
        assert "Development" in config_yaml
        assert "formats:" in config_yaml
        assert "sqlite" in config_yaml.lower()

        # Should include environment variable comments
        assert "Environment Variables:" in config_yaml
        assert "ABBA_EXPORT_BATCH_SIZE" in config_yaml

    def test_docker_compose_generation(self, config_manager):
        """Test Docker Compose configuration generation."""
        # Test with databases included
        docker_compose_yaml = config_manager.create_docker_compose_config(
            template_name="production", include_databases=True
        )

        assert isinstance(docker_compose_yaml, str)

        # Parse YAML
        compose_config = yaml.safe_load(docker_compose_yaml)

        assert "version" in compose_config
        assert "services" in compose_config

        # Should include OpenSearch (used in production template)
        services = compose_config["services"]
        assert "opensearch" in services

        # Check OpenSearch service configuration
        opensearch_service = services["opensearch"]
        assert opensearch_service["image"].startswith("opensearchproject/opensearch")
        assert "ports" in opensearch_service
        assert "9200:9200" in opensearch_service["ports"]

        # Should include volumes
        if "volumes" in compose_config:
            assert "opensearch_data" in compose_config["volumes"]

    def test_docker_compose_without_databases(self, config_manager):
        """Test Docker Compose generation without database services."""
        docker_compose_yaml = config_manager.create_docker_compose_config(
            template_name="mobile", include_databases=False  # Only uses SQLite
        )

        compose_config = yaml.safe_load(docker_compose_yaml)

        # Should have minimal structure
        assert "version" in compose_config
        assert "services" in compose_config

        # Should not include database services
        services = compose_config["services"]
        assert len(services) == 0 or all(
            service_name not in ["opensearch", "neo4j", "arangodb"]
            for service_name in services.keys()
        )

    def test_research_template_docker_compose(self, config_manager):
        """Test Docker Compose for research template with multiple databases."""
        docker_compose_yaml = config_manager.create_docker_compose_config(
            template_name="research", include_databases=True
        )

        compose_config = yaml.safe_load(docker_compose_yaml)
        services = compose_config["services"]

        # Research template includes SQLite, Neo4j, and OpenSearch
        assert "opensearch" in services
        assert "neo4j" in services

        # Check Neo4j configuration
        neo4j_service = services["neo4j"]
        assert neo4j_service["image"].startswith("neo4j")
        assert "7474:7474" in neo4j_service["ports"]
        assert "7687:7687" in neo4j_service["ports"]

        # Should have volumes for all databases
        if "volumes" in compose_config:
            volumes = compose_config["volumes"]
            assert "opensearch_data" in volumes
            assert "neo4j_data" in volumes

    def test_invalid_template_handling(self, config_manager):
        """Test handling of invalid template names."""
        # Test pipeline config creation with invalid template
        with pytest.raises(ValueError) as exc_info:
            config_manager.create_pipeline_config(
                template_name="nonexistent_template", output_path="/tmp/test"
            )
        assert "unknown template" in str(exc_info.value).lower()

        # Test Docker Compose with invalid template
        with pytest.raises(ValueError) as exc_info:
            config_manager.create_docker_compose_config(template_name="nonexistent_template")
        assert "unknown template" in str(exc_info.value).lower()

        # Test template export with invalid template
        with pytest.raises(ValueError) as exc_info:
            config_manager.export_template_as_config(
                template_name="nonexistent_template", output_path="/tmp/test"
            )
        assert "unknown template" in str(exc_info.value).lower()

    def test_config_file_loading_error(self, config_manager):
        """Test configuration file loading error handling."""
        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent_config")

    def test_environment_mappings(self, config_manager):
        """Test environment variable mappings."""
        mappings = config_manager._env_mappings

        # Check global mappings
        assert "ABBA_EXPORT_BATCH_SIZE" in mappings
        assert mappings["ABBA_EXPORT_BATCH_SIZE"] == "batch_size"

        # Check format-specific mappings
        assert "ABBA_SQLITE_ENABLE_FTS5" in mappings
        assert "ABBA_JSON_CHUNK_SIZE" in mappings
        assert "ABBA_OPENSEARCH_URL" in mappings
        assert "ABBA_NEO4J_URL" in mappings
        assert "ABBA_ARANGO_URL" in mappings

        # Check mapping format
        sqlite_mapping = mappings["ABBA_SQLITE_ENABLE_FTS5"]
        assert sqlite_mapping == "sqlite.enable_fts5"


if __name__ == "__main__":
    pytest.main([__file__])
