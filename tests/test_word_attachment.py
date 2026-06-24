"""Tests for word attachment from stepbible_verses and lexicon normalization."""

import os
import tempfile

from abba.database.sqlite_manager import SQLiteManager
from abba.strongs import extract_lexical_strongs


class TestGetWordsForVerse:
    def setup_method(self) -> None:
        """Set up a fresh SQLite database for each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = SQLiteManager(self.tmp.name)
        self.db.initialize_database()

    def teardown_method(self) -> None:
        """Clean up temporary database."""
        try:
            os.unlink(self.tmp.name)
        except Exception:  # noqa: BLE001, S110
            pass

    def _insert_stepbible_verse(
        self,
        book: str,
        chapter: int,
        verse: int,
        word_number: int,
        original_word: str,
        transliteration: str,
        english: str,
        strongs_raw: str,
        strongs_primary: str,
        morphology: str,
        language: str,
    ) -> None:
        """Helper to insert a stepbible_verses row."""
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO stepbible_verses
                   (source_file, book, chapter, verse, word_number, original_word, transliteration,
                    english, strongs_raw, strongs_primary, morphology, language)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "test_source",
                    book,
                    chapter,
                    verse,
                    word_number,
                    original_word,
                    transliteration,
                    english,
                    strongs_raw,
                    strongs_primary,
                    morphology,
                    language,
                ),
            )
            conn.commit()

    def test_get_words_jhn_1_1_greek(self) -> None:
        """Test retrieving Greek words from John 1:1."""
        self._insert_stepbible_verse(
            book="Jhn",
            chapter=1,
            verse=1,
            word_number=1,
            original_word="Ἐν",
            transliteration="En",
            english="In",
            strongs_raw="G1722",
            strongs_primary="G0746",
            morphology="P",
            language="greek",
        )
        rows = self.db.get_words_for_verse("Jhn", 1, 1)
        assert len(rows) == 1
        row = rows[0]
        assert row["word_number"] == 1

    def test_get_words_gen_1_1_hebrew(self) -> None:
        """Test retrieving Hebrew words from Genesis 1:1."""
        self._insert_stepbible_verse(
            book="Gen",
            chapter=1,
            verse=1,
            word_number=1,
            original_word="בְּרֵאשִׁית",
            transliteration="bereshit",
            english="In the beginning",
            strongs_raw="H9003/{H7225G}",
            strongs_primary="H9003",
            morphology="HR/Ncfsa",
            language="hebrew",
        )
        rows = self.db.get_words_for_verse("Gen", 1, 1)
        assert len(rows) == 1

    def test_extract_strongs_greek(self) -> None:
        """Test Strong's extraction for Greek word (source/padded form; lookups normalize)."""
        result = extract_lexical_strongs("G0746", "G0746")
        assert result == "G0746"

    def test_extract_strongs_hebrew_step_prefix(self) -> None:
        """Test Strong's extraction for Hebrew word with STEP prefix."""
        result = extract_lexical_strongs("H9003", "H9003/{H7225G}")
        assert result == "H7225"

    def test_get_words_empty_verse(self) -> None:
        """Test that non-existent verse returns empty list."""
        rows = self.db.get_words_for_verse("Rev", 22, 21)
        assert len(rows) == 0

    def test_get_words_returns_correct_columns(self) -> None:
        """Test that returned rows have the expected column names."""
        self._insert_stepbible_verse(
            book="Gen",
            chapter=1,
            verse=2,
            word_number=1,
            original_word="וְהָאָרֶץ",
            transliteration="veha-aretz",
            english="and the earth",
            strongs_raw="{H0776}",
            strongs_primary="H0776",
            morphology="HC/Ncbsa",
            language="hebrew",
        )
        rows = self.db.get_words_for_verse("Gen", 1, 2)
        assert len(rows) == 1
        row = rows[0]
        # Verify expected columns are present
        assert row["word_number"] == 1
        assert row["original_word"] == "וְהָאָרֶץ"
        assert row["transliteration"] == "veha-aretz"
        assert row["english"] == "and the earth"
        assert row["strongs_primary"] == "H0776"
        assert row["language"] == "hebrew"


class TestLexiconNormalization:
    def setup_method(self) -> None:
        """Set up a fresh SQLite database for each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = SQLiteManager(self.tmp.name)
        self.db.initialize_database()

    def teardown_method(self) -> None:
        """Clean up temporary database."""
        try:
            os.unlink(self.tmp.name)
        except Exception:  # noqa: BLE001, S110
            pass

    def _insert_lexicon_entry(self, strongs_number: str, original_word: str, gloss: str, language: str) -> None:
        """Helper to insert a lexicon row."""
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO lexicon (strongs_number, original_word, gloss, definition, language)
                   VALUES (?, ?, ?, ?, ?)""",
                (strongs_number, original_word, gloss, gloss, language),
            )
            conn.commit()

    def test_lookup_by_exact_strongs(self) -> None:
        """Direct lookup returns the stored entry."""
        self._insert_lexicon_entry("H430", "אֱלֹהִים", "God, gods", "hebrew")
        result = self.db.get_lexicon_entry("H430")
        assert result is not None
        assert result["strongs_number"] == "H430"

    def test_lookup_by_padded_strongs(self) -> None:
        """Padded lookup H0430 resolves when lexicon has H430."""
        self._insert_lexicon_entry("H430", "אֱלֹהִים", "God, gods", "hebrew")
        result_padded = self.db.get_lexicon_entry("H0430")
        assert result_padded is not None
        assert result_padded["strongs_number"] == "H430"

    def test_lookup_greek_exact(self) -> None:
        """Direct Greek lookup works."""
        self._insert_lexicon_entry("G746", "ἀρχή", "beginning, origin", "greek")
        result = self.db.get_lexicon_entry("G746")
        assert result is not None

    def test_lookup_greek_padded(self) -> None:
        """Padded Greek lookup G0746 resolves when lexicon has G746."""
        self._insert_lexicon_entry("G746", "ἀρχή", "beginning, origin", "greek")
        result_padded = self.db.get_lexicon_entry("G0746")
        assert result_padded is not None
        assert result_padded["strongs_number"] == "G746"

    def test_lookup_missing_returns_none(self) -> None:
        """Missing entry returns None."""
        result = self.db.get_lexicon_entry("H9999")
        assert result is None
