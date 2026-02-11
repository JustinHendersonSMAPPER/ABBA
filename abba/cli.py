"""Command line interface for ABBA."""

import argparse
from pathlib import Path
from typing import List, Optional


class CLIConfig:
    """Command line interface configuration."""

    def __init__(self):
        self.parser = self._create_parser()
        self.args = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="ABBA - Bible Data Extractor",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  abba                                    # Download and extract all translations
  abba --translations KJV ASV            # Extract only KJV and ASV
  abba --data-dir /path/to/data          # Use custom data directory
  abba --list                            # List available translations
  abba --no-download                     # Skip download, only extract
            """,
        )

        # Data directory
        parser.add_argument(
            "--data-dir", type=Path, help="Data directory for bible.db and translations (default: bible_data)"
        )

        # Translation selection
        parser.add_argument("--translations", nargs="*", help="Specific translations to extract (default: all)")

        # Actions
        parser.add_argument("--list", action="store_true", help="List available translations and exit")

        parser.add_argument("--no-download", action="store_true", help="Skip downloading bible.db if it exists")

        parser.add_argument("--force-download", action="store_true", help="Force download bible.db even if it exists")

        parser.add_argument(
            "--english-only",
            action="store_true",
            help="Use bible.eng.db (475 MB, English translations only) instead of bible.db (11.8 GB, all languages)",
        )

        # Database options
        parser.add_argument("--db-path", type=Path, help="Override ABBA database location")

        parser.add_argument(
            "--rebuild-db", action="store_true", help="Rebuild database - remove and reimport all translations"
        )

        parser.add_argument(
            "--rebuild-stepbible",
            action="store_true",
            help="Rebuild STEPBible data - remove and reimport Hebrew/Greek texts",
        )

        parser.add_argument(
            "--rebuild-embeddings",
            action="store_true",
            help="Rebuild embeddings - remove and regenerate all embeddings",
        )

        parser.add_argument(
            "--purge-all",
            action="store_true",
            help=(
                "Remove all data, databases, embeddings, and tracking files "
                "before starting (WARNING: This deletes everything!)"
            ),
        )

        parser.add_argument("--no-cache", action="store_true", help="Disable query caching")

        # Embedding options
        parser.add_argument("--embed-verses", action="store_true", help="Generate embeddings for verses")

        parser.add_argument("--embed-words", action="store_true", help="Generate embeddings for words")

        parser.add_argument("--embed-all", action="store_true", help="Generate all embeddings (verses and words)")

        parser.add_argument("--embedding-batch-size", type=int, default=100, help="Batch size for embedding generation")

        # Ollama options
        parser.add_argument("--ollama-host", help="Override Ollama API endpoint (default: http://localhost:11434)")

        parser.add_argument("--ollama-models", help="Comma-separated list of Ollama models for semantic analysis")

        parser.add_argument("--ollama-consensus", type=float, help="Set consensus threshold for multi-model agreement")

        # Concept mapping options
        parser.add_argument("--concepts-file", type=Path, help="Path to user-defined concepts YAML file")

        parser.add_argument("--validate-concepts", action="store_true", help="Run LLM validation on concepts")

        parser.add_argument(
            "--validate-concept-data",
            action="store_true",
            help="Validate Hebrew/Greek terms and Strong's numbers exist in databases",
        )

        parser.add_argument("--concept-report", action="store_true", help="Generate detailed concept validation report")

        parser.add_argument(
            "--map-concepts", action="store_true", help="Map all concepts to verses using semantic concordance"
        )

        parser.add_argument("--search-concept", help="Search for a specific biblical concept (e.g., love, faith)")

        parser.add_argument("--export-concept-mappings", help="Export concept mappings to file (CSV or JSON)")

        # Performance options
        parser.add_argument(
            "--parallel-workers",
            type=int,
            default=None,
            help="Number of parallel workers for import (default: auto-detect CPU count)",
        )

        parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing for imports")

        parser.add_argument(
            "--use-processes", action="store_true", help="Use processes instead of threads for parallel import"
        )

        parser.add_argument("--verify", action="store_true", help="Verify imports using hash validation after import")

        parser.add_argument(
            "--check-for-updates",
            action="store_true",
            help="Check for updates to STEPBible data and re-import if changed",
        )

        # Configuration
        parser.add_argument("--env-file", type=Path, help="Path to .env file (default: .env)")

        parser.add_argument("--config-file", type=Path, help="Path to configuration file")

        # Output options
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output (equivalent to --log-level DEBUG)"
        )

        parser.add_argument(
            "--log-level",
            choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set logging level (default: INFO, --verbose sets DEBUG)",
        )

        parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts (assume yes)")

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        self.args = self.parser.parse_args(args)
        result: argparse.Namespace = self.args
        return result

    def get_data_dir(self) -> Optional[Path]:
        """Get data directory from CLI args."""
        return self.args.data_dir if self.args else None

    def get_translations(self) -> Optional[List[str]]:
        """Get translations list from CLI args."""
        return self.args.translations if self.args else None

    def get_env_file(self) -> Optional[Path]:
        """Get env file path from CLI args."""
        return self.args.env_file if self.args else None

    def get_config_file(self) -> Optional[Path]:
        """Get config file path from CLI args."""
        return self.args.config_file if self.args else None

    def should_list(self) -> bool:
        """Check if should list translations."""
        return self.args.list if self.args else False

    def should_download(self) -> Optional[bool]:
        """Check download preference from CLI args."""
        if not self.args:
            return None

        if self.args.no_download:
            return False
        if self.args.force_download:
            return True
        return None  # Use default logic

    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.args.verbose if self.args else False

    def is_english_only(self) -> bool:
        """Check if english-only mode is enabled."""
        return self.args.english_only if self.args else False

    def get_db_path(self) -> Optional[Path]:
        """Get database path from CLI args."""
        return self.args.db_path if self.args else None

    def should_rebuild_db(self) -> bool:
        """Check if should rebuild database."""
        return self.args.rebuild_db if self.args else False

    def should_rebuild_stepbible(self) -> bool:
        """Check if should rebuild STEPBible data."""
        return self.args.rebuild_stepbible if self.args else False

    def should_rebuild_embeddings(self) -> bool:
        """Check if should rebuild embeddings."""
        return self.args.rebuild_embeddings if self.args else False

    def should_purge_all(self) -> bool:
        """Check if should purge all data and start fresh."""
        return self.args.purge_all if self.args else False

    def skip_confirmations(self) -> bool:
        """Check if confirmation prompts should be skipped."""
        return self.args.yes if self.args else False

    def should_check_for_updates(self) -> bool:
        """Check if should check for STEPBible data updates."""
        return self.args.check_for_updates if self.args else False

    def should_use_cache(self) -> Optional[bool]:
        """Check cache preference from CLI args."""
        if not self.args:
            return None
        return not self.args.no_cache

    def should_embed_verses(self) -> bool:
        """Check if should generate verse embeddings."""
        return self.args.embed_verses if self.args else False

    def should_embed_words(self) -> bool:
        """Check if should generate word embeddings."""
        return self.args.embed_words if self.args else False

    def should_embed_all(self) -> bool:
        """Check if should generate all embeddings."""
        return self.args.embed_all if self.args else False

    def get_embedding_batch_size(self) -> int:
        """Get embedding batch size."""
        return self.args.embedding_batch_size if self.args else 100

    def get_parallel_workers(self) -> Optional[int]:
        """Get number of parallel workers."""
        if self.args and hasattr(self.args, "parallel_workers"):
            result: Optional[int] = self.args.parallel_workers
            return result
        return None

    def should_use_parallel(self) -> bool:
        """Check if parallel processing should be used."""
        if self.args and hasattr(self.args, "no_parallel"):
            return not self.args.no_parallel
        return True  # Default to parallel

    def should_use_processes(self) -> bool:
        """Check if processes should be used instead of threads."""
        if self.args and hasattr(self.args, "use_processes"):
            result: bool = self.args.use_processes
            return result
        return False  # Default to threads

    def get_log_level(self) -> str:
        """Get logging level from CLI args, considering --verbose flag."""
        if not self.args:
            return "INFO"

        # --verbose overrides --log-level
        if self.args.verbose:
            return "DEBUG"
        level: str = self.args.log_level
        return level

    def get_ollama_host(self) -> Optional[str]:
        """Get Ollama host from CLI args."""
        return self.args.ollama_host if self.args else None

    def get_ollama_models(self) -> Optional[List[str]]:
        """Get Ollama models list from CLI args."""
        if self.args and self.args.ollama_models:
            return [model.strip() for model in self.args.ollama_models.split(",")]
        return None

    def get_ollama_consensus(self) -> Optional[float]:
        """Get Ollama consensus threshold from CLI args."""
        return self.args.ollama_consensus if self.args else None

    def get_concepts_file(self) -> Optional[Path]:
        """Get concepts file path from CLI args."""
        return self.args.concepts_file if self.args else None

    def should_validate_concepts(self) -> bool:
        """Check if should validate concepts."""
        return self.args.validate_concepts if self.args else False

    def should_validate_concept_data(self) -> bool:
        """Check if should validate concept data against databases."""
        return self.args.validate_concept_data if self.args else False

    def get_search_concept(self) -> Optional[str]:
        """Get concept to search for."""
        return self.args.search_concept if self.args else None

    def get_export_concept_mappings(self) -> Optional[str]:
        """Get export file path for concept mappings."""
        return self.args.export_concept_mappings if self.args else None

    def should_generate_concept_report(self) -> bool:
        """Check if should generate concept report."""
        return self.args.concept_report if self.args else False

    def should_map_concepts(self) -> bool:
        """Check if should map concepts to verses."""
        return self.args.map_concepts if self.args else False


# Global CLI configuration instance
cli_config = CLIConfig()
