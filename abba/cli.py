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

        # Database options
        parser.add_argument("--db-path", type=Path, help="Override ABBA database location")

        parser.add_argument("--rebuild-db", action="store_true", help="Force database rebuild")

        parser.add_argument("--no-cache", action="store_true", help="Disable query caching")

        # Embedding options
        parser.add_argument("--embed-verses", action="store_true", help="Generate embeddings for verses")
        
        parser.add_argument("--embed-words", action="store_true", help="Generate embeddings for words")
        
        parser.add_argument("--embed-all", action="store_true", help="Generate all embeddings (verses and words)")
        
        parser.add_argument("--force-reembed", action="store_true", help="Force regeneration of embeddings")
        
        parser.add_argument("--embedding-batch-size", type=int, default=100, help="Batch size for embedding generation")

        # Configuration
        parser.add_argument("--env-file", type=Path, help="Path to .env file (default: .env)")

        parser.add_argument("--config-file", type=Path, help="Path to configuration file")

        # Output options
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

        parser.add_argument("--quiet", "-q", action="store_true", help="Enable quiet mode (minimal output)")

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        self.args = self.parser.parse_args(args)
        return self.args

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
        elif self.args.force_download:
            return True
        else:
            return None  # Use default logic

    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.args.verbose if self.args else False

    def is_quiet(self) -> bool:
        """Check if quiet mode is enabled."""
        return self.args.quiet if self.args else False

    def get_db_path(self) -> Optional[Path]:
        """Get database path from CLI args."""
        return self.args.db_path if self.args else None

    def should_rebuild_db(self) -> bool:
        """Check if should rebuild database."""
        return self.args.rebuild_db if self.args else False

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

    def should_force_reembed(self) -> bool:
        """Check if should force regeneration of embeddings."""
        return self.args.force_reembed if self.args else False

    def get_embedding_batch_size(self) -> int:
        """Get embedding batch size."""
        return self.args.embedding_batch_size if self.args else 100


# Global CLI configuration instance
cli_config = CLIConfig()
