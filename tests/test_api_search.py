"""Tests for SearchAPI functionality."""

import tempfile
import unittest
from pathlib import Path

from abba.api.search import SearchAPI
from abba.database import SQLiteManager


class TestSearchAPI(unittest.TestCase):
    """Test SearchAPI functionality."""

    def setUp(self):
        """Set up test database and search API."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db_manager = SQLiteManager(self.db_path)
        self.db_manager.initialize_database()
        self.search_api = SearchAPI(self.db_manager)

        # Insert test data
        self._insert_test_data()

    def tearDown(self):
        """Clean up test database."""
        if self.db_path.exists():
            self.db_path.unlink()

    def _insert_test_data(self):
        """Insert test data for searching."""
        # Insert translation
        translation_data = {
            "id": "ESV",
            "name": "English Standard Version",
            "english_name": "English Standard Version",
            "language": "en",
        }
        self.db_manager.insert_translation(translation_data)

        # Insert verses
        verses = [
            {
                "translation_id": "ESV",
                "book_id": 1,
                "chapter": 1,
                "verse": 1,
                "text": "In the beginning, God created the heavens and the earth.",
            },
            {
                "translation_id": "ESV",
                "book_id": 1,
                "chapter": 1,
                "verse": 2,
                "text": "The earth was without form and void, and darkness was over the face of the deep.",
            },
        ]

        for verse in verses:
            self.db_manager.insert_verse(verse)

        # Insert test word
        word_data = {
            "book": "Genesis",
            "chapter": 1,
            "verse": 1,
            "word_num": 1,
            "word_ref": "Gen.1.1.1",
            "hebrew_text": "בְּרֵאשִׁית",
            "transliteration": "b'reshit",
            "translation": "beginning",
            "strongs_primary": "H7225",
            "language": "hebrew",
        }
        self.db_manager.insert_word(word_data)

    def test_get_verse(self):
        """Test getting a specific verse."""
        result = self.search_api.get_verse("ESV", 1, 1, 1)

        self.assertIsNotNone(result)
        self.assertEqual(result.translation_id, "ESV")
        self.assertEqual(result.book_id, 1)
        self.assertEqual(result.chapter, 1)
        self.assertEqual(result.verse, 1)
        self.assertIn("beginning", result.text)

    def test_get_verse_not_found(self):
        """Test getting a verse that doesn't exist."""
        result = self.search_api.get_verse("ESV", 1, 1, 999)
        self.assertIsNone(result)

    def test_get_words_for_verse(self):
        """Test getting words for a verse."""
        words = self.search_api.get_words_for_verse("Genesis", 1, 1)

        self.assertEqual(len(words), 1)
        word = words[0]
        self.assertEqual(word.book, "Genesis")
        self.assertEqual(word.chapter, 1)
        self.assertEqual(word.verse, 1)
        self.assertEqual(word.hebrew_text, "בְּרֵאשִׁית")

    def test_search_strongs(self):
        """Test searching by Strong's number."""
        words = self.search_api.search_strongs("H7225")

        self.assertEqual(len(words), 1)
        word = words[0]
        self.assertEqual(word.strongs_primary, "H7225")
        self.assertEqual(word.hebrew_text, "בְּרֵאשִׁית")

    def test_get_word_analysis(self):
        """Test getting complete word analysis."""
        analysis = self.search_api.get_word_analysis("Genesis", 1, 1, 1)

        self.assertIsNotNone(analysis)
        self.assertIn("word", analysis)
        self.assertIn("lexicon", analysis)
        self.assertIn("morphology", analysis)

        word_info = analysis["word"]
        self.assertEqual(word_info["book"], "Genesis")
        self.assertEqual(word_info["hebrew_text"], "בְּרֵאשִׁית")

    def test_get_word_analysis_not_found(self):
        """Test getting analysis for non-existent word."""
        analysis = self.search_api.get_word_analysis("Genesis", 1, 1, 999)
        self.assertIsNone(analysis)


if __name__ == "__main__":
    unittest.main()
