"""Environment variable handling for ABBA configuration."""

import os
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv


class EnvConfig:
    """Handles loading configuration from .env files."""

    def __init__(self, env_file: Optional[str] = None):
        """Initialize environment configuration.

        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
        """
        self.env_file = env_file or ".env"
        self.loaded = False
        self._load_env()

    def _load_env(self):
        """Load environment variables from .env file."""
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)
            self.loaded = True

    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get string value from environment."""
        return os.getenv(key, default)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Get boolean value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Get float value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def get_list(self, key: str, default: Optional[List[str]] = None, separator: str = ",") -> Optional[List[str]]:
        """Get list value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        return [item.strip() for item in value.split(separator) if item.strip()]

    def get_path(self, key: str, default: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Get path value from environment."""
        value = os.getenv(key)
        if value is None:
            return Path(default) if default else None
        return Path(value)


# Global environment configuration instance
env_config = EnvConfig()
