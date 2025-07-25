"""Configuration management for ABBA."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .cli import cli_config
from .env import env_config


@dataclass
class ABBAConfig:
    """ABBA configuration settings."""

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("bible_data"))

    # Translation settings
    translations: Optional[List[str]] = None  # None means all translations

    # Download settings
    download_enabled: bool = True
    force_download: bool = False
    bible_db_url: str = "https://bible.helloao.org/bible.db"

    # Output settings
    verbose: bool = False
    quiet: bool = False

    # Database settings
    database_path: Optional[Path] = None
    use_cache: bool = True
    cache_ttl: int = 3600

    # Vector database settings
    vector_db_type: str = "chromadb"
    vector_db_path: Optional[Path] = None
    vector_dimensions: int = 768
    similarity_metric: str = "cosine"

    # Embedding model settings
    embedding_library: str = "sentence-transformers"
    embedding_model_english: str = "intfloat/e5-large-v2"
    embedding_model_multilingual: str = "intfloat/multilingual-e5-base"
    embedding_context_mode: str = "enhanced"
    embedding_cache_dir: Optional[Path] = None

    # Search settings
    max_results: int = 50
    similarity_threshold: float = 0.7
    enable_query_expansion: bool = True

    # Performance settings
    parallel_workers: int = 4
    connection_pool_size: int = 10

    # File paths
    env_file: Optional[Path] = None
    config_file: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization to ensure Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if self.env_file and isinstance(self.env_file, str):
            self.env_file = Path(self.env_file)
        if self.config_file and isinstance(self.config_file, str):
            self.config_file = Path(self.config_file)
        if self.database_path and isinstance(self.database_path, str):
            self.database_path = Path(self.database_path)
        if self.vector_db_path and isinstance(self.vector_db_path, str):
            self.vector_db_path = Path(self.vector_db_path)
        if self.embedding_cache_dir and isinstance(self.embedding_cache_dir, str):
            self.embedding_cache_dir = Path(self.embedding_cache_dir)

    @property
    def db_path(self) -> Path:
        """Get path to bible.db file."""
        return self.data_dir / "bible.db"

    @property
    def abba_db_path(self) -> Path:
        """Get path to ABBA SQLite database."""
        if self.database_path:
            return self.database_path
        return self.data_dir / "abba.db"

    @property
    def translations_dir(self) -> Path:
        """Get path to translations directory."""
        return self.data_dir / "translations"

    @property
    def vectors_path(self) -> Path:
        """Get path to vector database."""
        if self.vector_db_path:
            return self.vector_db_path
        return self.data_dir / "vectors"

    @property
    def models_path(self) -> Path:
        """Get path to embedding models cache."""
        if self.embedding_cache_dir:
            return self.embedding_cache_dir
        return self.data_dir / "models"

    def should_download(self) -> bool:
        """Check if should download bible.db."""
        if self.force_download:
            return True
        if not self.download_enabled:
            return False
        return not self.db_path.exists()

    def create_directories(self):
        """Create necessary directories."""
        self.data_dir.mkdir(exist_ok=True)
        self.translations_dir.mkdir(exist_ok=True)


class ConfigManager:
    """Manages configuration from multiple sources with priority: CLI > env > config file > defaults."""

    def __init__(self):
        self.config = ABBAConfig()
        self._loaded = False

    def load_config(self, cli_args: Optional[List[str]] = None) -> ABBAConfig:
        """Load configuration from all sources."""
        if self._loaded:
            return self.config

        # Parse CLI arguments
        cli_config.parse_args(cli_args)

        # Load environment configuration
        env_file = cli_config.get_env_file()
        if env_file:
            env_config.env_file = str(env_file)
            env_config._load_env()

        # Load configuration file if specified
        config_file = cli_config.get_config_file()
        if config_file:
            self._load_config_file(config_file)

        # Apply settings with priority: CLI > env > config file > defaults
        self._apply_settings()

        self._loaded = True
        return self.config

    def _load_config_file(self, config_file: Path):
        """Load configuration from JSON file."""
        if not config_file.exists():
            return

        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Apply config file settings to defaults
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")

    def _apply_settings(self):
        """Apply settings with priority: CLI > env > current config."""

        # Data directory
        cli_data_dir = cli_config.get_data_dir()
        env_data_dir = env_config.get_path("ABBA_DATA_DIR")

        if cli_data_dir:
            self.config.data_dir = cli_data_dir
        elif env_data_dir:
            self.config.data_dir = env_data_dir

        # Translations
        cli_translations = cli_config.get_translations()
        env_translations = env_config.get_list("ABBA_TRANSLATIONS")

        if cli_translations is not None:
            self.config.translations = cli_translations
        elif env_translations is not None:
            self.config.translations = env_translations

        # Download settings
        cli_download = cli_config.should_download()
        env_download = env_config.get_bool("ABBA_DOWNLOAD_ENABLED")
        env_force_download = env_config.get_bool("ABBA_FORCE_DOWNLOAD")

        if cli_download is not None:
            self.config.download_enabled = cli_download
            if cli_config.args and cli_config.args.force_download:
                self.config.force_download = True
        elif env_download is not None:
            self.config.download_enabled = env_download

        if env_force_download is not None:
            self.config.force_download = env_force_download

        # Bible DB URL
        env_url = env_config.get_str("ABBA_BIBLE_DB_URL")
        if env_url:
            self.config.bible_db_url = env_url

        # Output settings
        if cli_config.is_verbose():
            self.config.verbose = True
        elif env_config.get_bool("ABBA_VERBOSE"):
            self.config.verbose = True

        if cli_config.is_quiet():
            self.config.quiet = True
        elif env_config.get_bool("ABBA_QUIET"):
            self.config.quiet = True

        # File paths
        cli_env_file = cli_config.get_env_file()
        if cli_env_file:
            self.config.env_file = cli_env_file

        cli_config_file = cli_config.get_config_file()
        if cli_config_file:
            self.config.config_file = cli_config_file

        # Database settings
        cli_db_path = cli_config.get_db_path()
        env_db_path = env_config.get_path("ABBA_DATABASE_PATH")
        env_use_cache = env_config.get_bool("ABBA_USE_CACHE")
        env_cache_ttl = env_config.get_int("ABBA_CACHE_TTL")

        if cli_db_path:
            self.config.database_path = cli_db_path
        elif env_db_path:
            self.config.database_path = env_db_path

        cli_use_cache = cli_config.should_use_cache()
        if cli_use_cache is not None:
            self.config.use_cache = cli_use_cache
        elif env_use_cache is not None:
            self.config.use_cache = env_use_cache

        if env_cache_ttl is not None:
            self.config.cache_ttl = env_cache_ttl

        # Vector database settings
        env_vector_db_type = env_config.get_str("ABBA_VECTOR_DB_TYPE")
        env_vector_db_path = env_config.get_path("ABBA_VECTOR_DB_PATH")
        env_vector_dimensions = env_config.get_int("ABBA_VECTOR_DIMENSIONS")
        env_similarity_metric = env_config.get_str("ABBA_SIMILARITY_METRIC")

        if env_vector_db_type:
            self.config.vector_db_type = env_vector_db_type
        if env_vector_db_path:
            self.config.vector_db_path = env_vector_db_path
        if env_vector_dimensions is not None:
            self.config.vector_dimensions = env_vector_dimensions
        if env_similarity_metric:
            self.config.similarity_metric = env_similarity_metric

        # Embedding model settings
        env_embedding_library = env_config.get_str("ABBA_EMBEDDING_LIBRARY")
        env_embedding_model_english = env_config.get_str("ABBA_EMBEDDING_MODEL_ENGLISH")
        env_embedding_model_multilingual = env_config.get_str("ABBA_EMBEDDING_MODEL_MULTILINGUAL")
        env_embedding_context_mode = env_config.get_str("ABBA_EMBEDDING_CONTEXT_MODE")
        env_embedding_cache_dir = env_config.get_path("ABBA_EMBEDDING_CACHE_DIR")

        if env_embedding_library:
            self.config.embedding_library = env_embedding_library
        if env_embedding_model_english:
            self.config.embedding_model_english = env_embedding_model_english
        if env_embedding_model_multilingual:
            self.config.embedding_model_multilingual = env_embedding_model_multilingual
        if env_embedding_context_mode:
            self.config.embedding_context_mode = env_embedding_context_mode
        if env_embedding_cache_dir:
            self.config.embedding_cache_dir = env_embedding_cache_dir

        # Search settings
        env_max_results = env_config.get_int("ABBA_MAX_RESULTS")
        env_similarity_threshold = env_config.get_float("ABBA_SIMILARITY_THRESHOLD")
        env_enable_query_expansion = env_config.get_bool("ABBA_ENABLE_QUERY_EXPANSION")

        if env_max_results is not None:
            self.config.max_results = env_max_results
        if env_similarity_threshold is not None:
            self.config.similarity_threshold = env_similarity_threshold
        if env_enable_query_expansion is not None:
            self.config.enable_query_expansion = env_enable_query_expansion

        # Performance settings
        env_parallel_workers = env_config.get_int("ABBA_PARALLEL_WORKERS")
        env_connection_pool_size = env_config.get_int("ABBA_CONNECTION_POOL_SIZE")

        if env_parallel_workers is not None:
            self.config.parallel_workers = env_parallel_workers
        if env_connection_pool_size is not None:
            self.config.connection_pool_size = env_connection_pool_size

    def save_config(self, config_file: Optional[Path] = None):
        """Save current configuration to file."""
        if not config_file:
            config_file = self.config.config_file or Path("abba_config.json")

        config_dict = {
            "data_dir": str(self.config.data_dir),
            "translations": self.config.translations,
            "download_enabled": self.config.download_enabled,
            "force_download": self.config.force_download,
            "bible_db_url": self.config.bible_db_url,
            "verbose": self.config.verbose,
            "quiet": self.config.quiet,
            "database_path": str(self.config.database_path) if self.config.database_path else None,
            "use_cache": self.config.use_cache,
            "cache_ttl": self.config.cache_ttl,
        }

        try:
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config file {config_file}: {e}")

    def get_config(self) -> ABBAConfig:
        """Get current configuration."""
        return self.config


# Global configuration manager
config_manager = ConfigManager()
