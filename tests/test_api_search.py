"""Tests for SearchAPI functionality."""

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

from abba.api.search import SearchAPI
from abba.database import SQLiteManager
from abba.database.lexical_strongs_populator import populate_lexical_strongs


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
        """Clean up test database.

        On Windows, lingering SQLite connections (e.g. migration connections
        opened via ``with sqlite3.connect(...)``, which commit but do not close)
        keep the file locked, raising ``PermissionError [WinError 32]`` on
        unlink. Force a GC pass so those connections are finalized before
        removing the temp directory, and tolerate any residual lock.
        """
        self.search_api = None
        self.db_manager = None
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _insert_test_data(self):
        """Insert test data for searching.

        Original-language word lookups (``get_words_for_verse``,
        ``search_strongs``, ``get_word_analysis``) read from the
        ``stepbible_verses`` table, so seed that table -- not the legacy
        ``words`` table -- to exercise the real query paths.
        """
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

        # Insert lexicon entry so get_word_analysis can attach lexicon info.
        self.db_manager.insert_lexicon_entry(
            {
                "strongs_number": "H7225",
                "original_word": "רֵאשִׁית",
                "transliteration": "reshit",
                "part_of_speech": "noun",
                "gloss": "beginning",
                "definition": "beginning, chief, first",
                "language": "hebrew",
            }
        )

        # Insert an original-language word into stepbible_verses (the source the
        # SearchAPI actually queries). STEP uses 3-letter book codes ("Gen").
        self.db_manager.execute_update(
            "INSERT OR REPLACE INTO stepbible_verses "
            "(source_file, book, chapter, verse, word_number, original_word, transliteration, english, "
            "strongs_raw, strongs_primary, morphology, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test", "Genesis", 1, 1, 1, "בְּרֵאשִׁית", "b'reshit", "beginning", "{H7225}", "H7225", None, "hebrew"),
        )

        # Populate the normalized lexical_strongs key so search_strongs works.
        populate_lexical_strongs(self.db_path)

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
        """Test getting words for a verse from the original-language source.

        Exercises ``SQLiteManager.get_words_for_verse`` -- the query path the
        ``/words`` route uses in production (routes.py) -- which reads from the
        ``stepbible_verses`` table.
        """
        words = self.db_manager.get_words_for_verse("Genesis", 1, 1)

        self.assertEqual(len(words), 1)
        word = words[0]
        self.assertEqual(word["word_number"], 1)
        self.assertEqual(word["original_word"], "בְּרֵאשִׁית")
        self.assertEqual(word["strongs_primary"], "H7225")
        self.assertEqual(word["language"], "hebrew")

    def test_search_strongs(self):
        """Test searching by Strong's number.

        Exercises ``SQLiteManager.search_strongs`` -- the query path the
        ``/strongs`` route uses in production (routes.py) -- which returns the
        distinct (book, chapter, verse) tuples containing the Strong's number.
        """
        rows = self.db_manager.search_strongs("H7225")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["book"], "Genesis")
        self.assertEqual(row["chapter"], 1)
        self.assertEqual(row["verse"], 1)

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
