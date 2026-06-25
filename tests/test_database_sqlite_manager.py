"""Tests for SQLiteManager database operations."""

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

from abba.database import SQLiteManager


class TestSQLiteManager(unittest.TestCase):
    """Test SQLiteManager functionality."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db_manager = SQLiteManager(self.db_path)
        self.db_manager.initialize_database()

    def tearDown(self):
        """Clean up test database.

        On Windows, lingering SQLite connections (e.g. migration connections
        opened via ``with sqlite3.connect(...)``, which commit but do not close)
        keep the file locked, raising ``PermissionError [WinError 32]`` on
        unlink. Force a GC pass so those connections are finalized before
        removing the temp directory, and tolerate any residual lock.
        """
        self.db_manager = None
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        """Test database is created and schema is applied."""
        self.assertTrue(self.db_path.exists())

        # Check that tables exist
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = ["words", "lexicon", "morphology", "translations", "books", "verses"]
            for table in expected_tables:
                self.assertIn(table, tables)

    def test_insert_and_get_translation(self):
        """Test translation insertion and retrieval."""
        translation_data = {
            "id": "ESV",
            "name": "English Standard Version",
            "english_name": "English Standard Version",
            "language": "en",
        }

        self.db_manager.insert_translation(translation_data)

        # Verify insertion
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM translations WHERE id = ?", ("ESV",))
            result = cursor.fetchone()

            self.assertIsNotNone(result)
            self.assertEqual(result["id"], "ESV")
            self.assertEqual(result["name"], "English Standard Version")

    def test_insert_and_get_verse(self):
        """Test verse insertion and retrieval."""
        # First insert a translation
        translation_data = {
            "id": "ESV",
            "name": "English Standard Version",
            "english_name": "English Standard Version",
            "language": "en",
        }
        self.db_manager.insert_translation(translation_data)

        # Insert a verse
        verse_data = {
            "translation_id": "ESV",
            "book_id": 1,
            "chapter": 1,
            "verse": 1,
            "text": "In the beginning, God created the heavens and the earth.",
        }
        self.db_manager.insert_verse(verse_data)

        # Retrieve verse
        result = self.db_manager.get_verse("ESV", 1, 1, 1)
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "In the beginning, God created the heavens and the earth.")

    def test_insert_word(self):
        """Test word insertion."""
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

        # Verify insertion against the ``words`` table (insert_word's target).
        # get_words_for_verse reads the separate stepbible_verses table.
        words = self.db_manager.execute_query(
            "SELECT * FROM words WHERE book = ? AND chapter = ? AND verse = ?",
            ("Genesis", 1, 1),
        )
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["hebrew_text"], "בְּרֵאשִׁית")

    def test_database_stats(self):
        """Test database statistics."""
        stats = self.db_manager.get_database_stats()

        # Should be a dict with counts
        self.assertIsInstance(stats, dict)
        self.assertIn("words", stats)
        self.assertIn("verses", stats)
        self.assertIn("translations", stats)

        # All should be 0 initially
        for table in ["words", "verses", "translations"]:
            self.assertEqual(stats[table], 0)


if __name__ == "__main__":
    unittest.main()
