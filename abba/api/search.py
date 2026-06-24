"""Search API for ABBA biblical analysis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..database import SQLiteManager
from ..strongs import extract_lexical_strongs


@dataclass
class VerseResult:
    """Represents a verse search result."""

    translation_id: str
    book_id: int
    chapter: int
    verse: int
    text: str
    book_name: Optional[str] = None


@dataclass
class WordResult:
    """Represents a word analysis result."""

    book: str
    chapter: int
    verse: int
    word_num: int
    word_ref: str
    hebrew_text: Optional[str]
    greek_text: Optional[str]
    transliteration: Optional[str]
    translation: Optional[str]
    strongs_primary: Optional[str]
    morphology_code: Optional[str]
    language: str


class SearchAPI:
    """Provides search functionality for biblical texts."""

    def __init__(self, db_manager: SQLiteManager):
        """Initialize search API.

        Args:
            db_manager: SQLite database manager
        """
        self.db_manager = db_manager

    def get_verse(self, translation_id: str, book_id: int, chapter: int, verse: int) -> Optional[VerseResult]:
        """Get a specific verse.

        Args:
            translation_id: Translation identifier
            book_id: Book identifier
            chapter: Chapter number
            verse: Verse number

        Returns:
            VerseResult or None if not found
        """
        result = self.db_manager.get_verse(translation_id, book_id, chapter, verse)
        if result:
            return VerseResult(
                translation_id=result["translation_id"],
                book_id=result["book_id"],
                chapter=result["chapter"],
                verse=result["verse"],
                text=result["text"],
            )
        return None

    def search_verses(self, translation_id: str, search_text: str, limit: int = 50) -> List[VerseResult]:
        """Search verses using full-text search.

        Args:
            translation_id: Translation identifier
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching verses
        """
        results = self.db_manager.search_verses(translation_id, search_text, limit)
        return [
            VerseResult(
                translation_id=row["translation_id"],
                book_id=row["book_id"],
                chapter=row["chapter"],
                verse=row["verse"],
                text=row["text"],
            )
            for row in results
        ]

    def get_words_for_verse(self, book: str, chapter: int, verse: int) -> List[WordResult]:
        """Get all words for a specific verse from original language texts.

        Args:
            book: Book name
            chapter: Chapter number
            verse: Verse number

        Returns:
            List of word analysis results
        """
        results = self.db_manager.get_words_for_verse(book, chapter, verse)
        return [
            WordResult(
                book=row["book"],
                chapter=row["chapter"],
                verse=row["verse"],
                word_num=row["word_num"],
                word_ref=row["word_ref"],
                hebrew_text=row["hebrew_text"],
                greek_text=row["greek_text"],
                transliteration=row["transliteration"],
                translation=row["translation"],
                strongs_primary=row["strongs_primary"],
                morphology_code=row["morphology_code"],
                language=row["language"],
            )
            for row in results
        ]

    def search_strongs(self, strongs_number: str) -> List[WordResult]:
        """Search for verses containing a specific Strong's number.

        Args:
            strongs_number: Strong's number (e.g., "H0430")

        Returns:
            List of word occurrences
        """
        results = self.db_manager.search_strongs(strongs_number)
        return [
            WordResult(
                book=row["book"],
                chapter=row["chapter"],
                verse=row["verse"],
                word_num=row["word_num"],
                word_ref=row["word_ref"],
                hebrew_text=row["hebrew_text"],
                greek_text=row["greek_text"],
                transliteration=row["transliteration"],
                translation=row["translation"],
                strongs_primary=row["strongs_primary"],
                morphology_code=row["morphology_code"],
                language=row["language"],
            )
            for row in results
        ]

    def get_word_analysis(self, book: str, chapter: int, verse: int, word_num: int) -> Optional[Dict[str, Any]]:
        """Get complete analysis for a specific word.

        Args:
            book: Book name
            chapter: Chapter number
            verse: Verse number
            word_num: Word number in verse

        Returns:
            Complete word analysis with lexicon and morphology info
        """
        # Get word data (original-language words from stepbible_verses via get_words_for_verse)
        words = self.db_manager.get_words_for_verse(book, chapter, verse)
        word = next((w for w in words if w["word_number"] == word_num), None)

        if not word:
            return None

        language = word["language"]
        original = word["original_word"]
        is_hebrew = language in ("hebrew", "aramaic")
        lexical_strongs = extract_lexical_strongs(word["strongs_primary"], word["strongs_raw"])
        analysis = {
            "word": {
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "word_num": word["word_number"],
                "word_ref": f"{book}.{chapter}.{verse}#{word['word_number']:02d}",
                "hebrew_text": original if is_hebrew else None,
                "greek_text": original if language == "greek" else None,
                "transliteration": word["transliteration"],
                "translation": word["english"],
                "language": language,
                "strongs_number": lexical_strongs,
            },
            "lexicon": None,
            "morphology": None,
        }

        # Get lexicon entry (normalized lookup resolves padded/prefixed Strong's codes)
        if lexical_strongs:
            lexicon = self.db_manager.get_lexicon_entry(lexical_strongs)
            if lexicon:
                analysis["lexicon"] = {
                    "strongs_number": lexicon["strongs_number"],
                    "original_word": lexicon["original_word"],
                    "transliteration": lexicon["transliteration"],
                    "part_of_speech": lexicon["part_of_speech"],
                    "gloss": lexicon["gloss"],
                    "definition": lexicon["definition"],
                }

        # Get morphology info
        if word["morphology"]:
            morphology = self.db_manager.get_morphology_info(word["morphology"])
            if morphology:
                analysis["morphology"] = {
                    "code": morphology["code"],
                    "description": morphology["description"],
                    "components": morphology["components"],
                }

        return analysis
