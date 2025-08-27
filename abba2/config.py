"""
ABBA (Annotated Bible and Background Analysis) 2.0 Configuration System
Centralized configuration with environment variable support
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import os


class Settings(BaseSettings):
    """Application settings with validation and environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ABBA_",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Project paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".abba2" / "data",
        description="Directory for downloaded data"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".abba2" / "cache", 
        description="Directory for cache files"
    )
    database_path: Path = Field(
        default_factory=lambda: Path.home() / ".abba2" / "abba2.db",
        description="SQLite database path"
    )
    
    # Data sources
    sources_manifest: Path = Field(
        default=Path("sources.yaml"),
        description="Path to sources manifest file"
    )
    verify_checksums: bool = Field(
        default=True,
        description="Verify file checksums after download"
    )
    max_download_retries: int = Field(
        default=3,
        description="Maximum download retry attempts"
    )
    download_timeout: int = Field(
        default=300,
        description="Download timeout in seconds"
    )
    
    # Processing settings
    parallel_workers: Optional[int] = Field(
        default=None,
        description="Number of parallel workers (None for auto-detect)"
    )
    batch_size: int = Field(
        default=1000,
        description="Batch size for processing"
    )
    memory_limit_mb: int = Field(
        default=2048,
        description="Memory limit in MB for processing"
    )
    
    # Synthesis settings
    min_source_agreement: float = Field(
        default=0.6,
        description="Minimum source agreement for consensus (0-1)"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for definitions"
    )
    semantic_cluster_eps: float = Field(
        default=0.3,
        description="DBSCAN epsilon for semantic clustering"
    )
    
    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    api_port: int = Field(
        default=8000,
        description="API port"
    )
    api_workers: int = Field(
        default=4,
        description="Number of API workers"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model for semantic search"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    embedding_cache_enabled: bool = Field(
        default=True,
        description="Enable embedding cache"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path (None for stdout only)"
    )
    
    # Development settings
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    testing: bool = Field(
        default=False,
        description="Testing mode"
    )
    profile: bool = Field(
        default=False,
        description="Enable profiling"
    )
    
    @field_validator("data_dir", "cache_dir", mode="after")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("parallel_workers", mode="after")
    @classmethod
    def set_parallel_workers(cls, v: Optional[int]) -> int:
        """Auto-detect CPU count if not specified"""
        if v is None:
            return min(os.cpu_count() or 1, 8)
        return v
    
    @field_validator("log_level", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    def get_source_path(self, source_name: str) -> Path:
        """Get path for a downloaded source file"""
        return self.data_dir / "sources" / source_name
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for a cache file"""
        return self.cache_dir / cache_key
    
    def get_database_url(self) -> str:
        """Get SQLAlchemy database URL"""
        return f"sqlite:///{self.database_path}"


# Global settings instance
settings = Settings()