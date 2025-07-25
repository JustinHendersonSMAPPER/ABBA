"""Tests for configuration management."""

import tempfile
import unittest
from pathlib import Path

from abba.config import ABBAConfig, ConfigManager


class TestABBAConfig(unittest.TestCase):
    """Test ABBAConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ABBAConfig()
        
        self.assertEqual(config.data_dir, Path("bible_data"))
        self.assertIsNone(config.translations)
        self.assertTrue(config.download_enabled)
        self.assertFalse(config.force_download)
        self.assertFalse(config.verbose)
        self.assertFalse(config.quiet)

    def test_db_path_property(self):
        """Test db_path property."""
        config = ABBAConfig(data_dir=Path("/test/path"))
        expected_path = Path("/test/path") / "bible.db"
        self.assertEqual(config.db_path, expected_path)

    def test_abba_db_path_property(self):
        """Test abba_db_path property with custom path."""
        custom_path = Path("/custom/abba.db")
        config = ABBAConfig(database_path=custom_path)
        self.assertEqual(config.abba_db_path, custom_path)

    def test_abba_db_path_default(self):
        """Test abba_db_path property with default path."""
        config = ABBAConfig(data_dir=Path("/test/path"))
        expected_path = Path("/test/path") / "abba.db"
        self.assertEqual(config.abba_db_path, expected_path)

    def test_should_download_conditions(self):
        """Test download decision logic."""
        # Force download
        config = ABBAConfig(force_download=True)
        self.assertTrue(config.should_download())
        
        # Download disabled
        config = ABBAConfig(download_enabled=False)
        self.assertFalse(config.should_download())


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = self.config_manager.load_config([])
        
        self.assertIsInstance(config, ABBAConfig)
        self.assertEqual(config.data_dir, Path("bible_data"))

    def test_save_and_load_config_file(self):
        """Test saving and loading configuration file."""
        config_file = Path(self.temp_dir) / "test_config.json"
        
        # Modify config and save
        self.config_manager.config.verbose = True
        self.config_manager.config.data_dir = Path("/custom/path")
        self.config_manager.save_config(config_file)
        
        # Load new manager and verify
        new_manager = ConfigManager()
        new_manager._load_config_file(config_file)
        
        self.assertTrue(new_manager.config.verbose)
        self.assertEqual(str(new_manager.config.data_dir), "/custom/path")


if __name__ == "__main__":
    unittest.main()