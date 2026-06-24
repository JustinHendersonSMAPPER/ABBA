"""Tests for BibleExtractor functionality."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from abba.bible_extractor import BibleExtractor


class TestBibleExtractor(unittest.TestCase):
    """Test BibleExtractor functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.extractor = BibleExtractor(str(self.data_dir))

    def test_init(self):
        """Test extractor initialization."""
        self.assertEqual(Path(self.extractor.data_dir), self.data_dir)
        self.assertEqual(str(self.extractor.db_path), str(self.data_dir / "bible.db"))

    def test_data_dir_creation(self):
        """Test data directory is created."""
        new_dir = Path(self.temp_dir) / "new_data"
        _extractor = BibleExtractor(str(new_dir))  # noqa: F841

        # Directory should be created during initialization
        self.assertTrue(new_dir.exists())

    @patch("abba.bible_extractor.requests.get")
    def test_download_bible_db_success(self, mock_get):
        """Test successful bible.db download."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_response

        result = self.extractor.download_bible_db()

        self.assertTrue(result)
        mock_get.assert_called_once()

    @patch("abba.bible_extractor.requests.get")
    def test_download_bible_db_failure(self, mock_get):
        """Test failed bible.db download."""
        # Mock failed response
        mock_get.side_effect = Exception("Download failed")

        result = self.extractor.download_bible_db()

        self.assertFalse(result)

    def test_list_translations_no_db(self):
        """Test listing translations when database doesn't exist."""
        translations = self.extractor.list_translations()
        self.assertEqual(translations, [])

    @patch("abba.bible_extractor.sqlite3.connect")
    def test_list_translations_with_db(self, mock_connect):
        """Test listing translations from database."""
        # Create fake db file
        db_path = self.data_dir / "bible.db"
        db_path.touch()

        # Mock database response
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("ESV", "English Standard Version", "English Standard Version", "en"),
            ("NIV", "New International Version", "New International Version", "en"),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        translations = self.extractor.list_translations()

        self.assertEqual(len(translations), 2)
        self.assertEqual(translations[0]["id"], "ESV")
        self.assertEqual(translations[1]["id"], "NIV")

    def test_extract_translation_no_db(self):
        """Test extract_translation when database doesn't exist."""
        result = self.extractor.extract_translation("ESV")
        # Method returns False when db doesn't exist
        self.assertFalse(result)

    def test_stepbible_data_directory_structure(self):
        """Test STEPBible data directory creation."""
        stepbible_dir = self.data_dir / "stepbible"

        # Should be created during download attempt
        self.extractor.download_stepbible_data()

        self.assertTrue(stepbible_dir.exists())

    @patch("abba.bible_extractor.requests.get")
    def test_stepbible_morphology_urls_include_subdirectory(self, mock_get):
        """Regression: TEHMC/TEGMC morphology files live under the 'Morphology codes/'
        subdirectory of the STEPBible-Data repo. Omitting it returns HTTP 404, which
        leaves hebrew_morphology.txt/greek_morphology.txt undownloaded and the
        morphology table empty (parse_stepbible_morphology then finds no file)."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"content-length": "0"}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        self.extractor.download_stepbible_data()

        requested_urls = [call.args[0] for call in mock_get.call_args_list]
        morphology_urls = [u for u in requested_urls if "TEHMC" in u or "TEGMC" in u]

        self.assertEqual(len(morphology_urls), 2, "expected both Hebrew (TEHMC) and Greek (TEGMC) morphology downloads")
        for url in morphology_urls:
            self.assertIn("Morphology codes/", url, f"morphology URL missing 'Morphology codes/' subdirectory: {url}")


if __name__ == "__main__":
    unittest.main()
