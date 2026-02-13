"""SQLite database manager for ABBA."""

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Query profiling threshold (log queries slower than this)
_SLOW_QUERY_THRESHOLD_MS = 50.0


class SQLiteManager:
    """Manages SQLite database operations for ABBA."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.schema_path = Path(__file__).parent / "schema.sql"
        self._transaction_conn = None

    def initialize_database(self) -> None:
        """Initialize database with schema if it doesn't exist."""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Creating new database at %s", self.db_path)

        self._execute_schema()

        # Run migrations for existing databases
        from .migrations import run_migrations

        run_migrations(self.db_path)

        logger.info("Database initialized successfully")

    def _execute_schema(self) -> None:
        """Execute the schema SQL file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            # Enable foreign key constraints and concurrency settings
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Database error: %s", e)
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result rows
        """
        start = time.perf_counter()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > _SLOW_QUERY_THRESHOLD_MS:
            logger.warning("Slow query (%.1fms): %s", elapsed_ms, query.strip()[:120])
        return result  # type: ignore[no-any-return]

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an INSERT, UPDATE, or DELETE query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.rowcount  # type: ignore[no-any-return]

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute a query multiple times with different parameters.

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount  # type: ignore[no-any-return]

    @contextmanager
    def transaction(self):
        """Context manager for explicit transaction control.

        Usage:
            with db_manager.transaction():
                db_manager.insert_translation(...)
                db_manager.insert_verses(...)
                # Automatically commits on success, rolls back on exception
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN TRANSACTION")

            # Store the connection for use in nested operations
            self._transaction_conn = conn
            yield self

            conn.execute("COMMIT")
        except Exception as e:
            if conn:
                conn.execute("ROLLBACK")
            logger.error("Transaction failed: %s", e)
            raise
        finally:
            self._transaction_conn = None
            if conn:
                conn.close()

    def _get_connection_for_transaction(self):
        """Get the current transaction connection if in a transaction."""
        return getattr(self, "_transaction_conn", None)

    def get_verse(self, translation_id: str, book_id: int, chapter: int, verse: int) -> Optional[sqlite3.Row]:
        """Get a specific verse.

        Args:
            translation_id: Translation identifier
            book_id: Book identifier
            chapter: Chapter number
            verse: Verse number

        Returns:
            Verse row or None if not found
        """
        query = """
            SELECT * FROM verses
            WHERE translation_id = ? AND book_id = ? AND chapter = ? AND verse = ?
        """
        results = self.execute_query(query, (translation_id, book_id, chapter, verse))
        return results[0] if results else None

    def search_verses(self, translation_id: str, search_text: str, limit: int = 50) -> List[sqlite3.Row]:
        """Search verses using full-text search.

        Args:
            translation_id: Translation identifier
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching verses
        """
        query = """
            SELECT * FROM verses_fts
            WHERE verses_fts MATCH ? AND translation_id = ?
            LIMIT ?
        """
        return self.execute_query(query, (search_text, translation_id, limit))

    def get_words_for_verse(self, book: str, chapter: int, verse: int) -> List[sqlite3.Row]:
        """Get all words for a specific verse from original language texts.

        Args:
            book: Book name
            chapter: Chapter number
            verse: Verse number

        Returns:
            List of word records
        """
        query = """
            SELECT * FROM words
            WHERE book = ? AND chapter = ? AND verse = ?
            ORDER BY word_num
        """
        return self.execute_query(query, (book, chapter, verse))

    def search_strongs(self, strongs_number: str) -> List[sqlite3.Row]:
        """Search for verses containing a specific Strong's number.

        Args:
            strongs_number: Strong's number (e.g., "H0430")

        Returns:
            List of word records
        """
        query = """
            SELECT * FROM words
            WHERE strongs_primary = ? OR strongs_raw LIKE ?
            ORDER BY book, chapter, verse, word_num
        """
        like_pattern = f"%{strongs_number}%"
        return self.execute_query(query, (strongs_number, like_pattern))

    def get_lexicon_entry(self, strongs_number: str) -> Optional[sqlite3.Row]:
        """Get lexicon entry for a Strong's number.

        Args:
            strongs_number: Strong's number

        Returns:
            Lexicon entry or None if not found
        """
        query = "SELECT * FROM lexicon WHERE strongs_number = ?"
        results = self.execute_query(query, (strongs_number,))
        return results[0] if results else None

    def get_morphology_info(self, morphology_code: str) -> Optional[sqlite3.Row]:
        """Get morphology information for a code.

        Args:
            morphology_code: Morphology code

        Returns:
            Morphology info or None if not found
        """
        query = "SELECT * FROM morphology WHERE code = ?"
        results = self.execute_query(query, (morphology_code,))
        return results[0] if results else None

    def insert_translation(self, translation_data: Dict[str, Any]) -> None:
        """Insert translation metadata.

        Args:
            translation_data: Translation information
        """
        # Detect canon if not provided
        canon = translation_data.get("canon")
        if not canon:
            from ..parallel_import import get_translation_canon

            canon_enum = get_translation_canon(translation_data["id"], str(self.db_path.parent / "bible.db"))
            canon = canon_enum.value

        query = """
            INSERT OR REPLACE INTO translations (id, name, english_name, language, canon)
            VALUES (?, ?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                translation_data["id"],
                translation_data["name"],
                translation_data.get("english_name"),
                translation_data.get("language"),
                canon,
            ),
        )

    def update_translation_partial_canon(self, translation_id: str, is_partial: bool, apocrypha_count: int) -> None:
        """Update partial canon information for a translation.

        Args:
            translation_id: Translation ID
            is_partial: Whether the canon is partial
            apocrypha_count: Number of apocryphal books included
        """
        query = """
            UPDATE translations
            SET is_partial_canon = ?, apocrypha_count = ?
            WHERE id = ?
        """
        self.execute_update(query, (is_partial, apocrypha_count, translation_id))

    def insert_verse(self, verse_data: Dict[str, Any]) -> None:
        """Insert a verse.

        Args:
            verse_data: Verse information
        """
        query = """
            INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text)
            VALUES (?, ?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                verse_data["translation_id"],
                verse_data["book_id"],
                verse_data["chapter"],
                verse_data["verse"],
                verse_data["text"],
            ),
        )

    def insert_word(self, word_data: Dict[str, Any]) -> None:
        """Insert a word from original language texts.

        Args:
            word_data: Word information
        """
        query = """
            INSERT OR REPLACE INTO words (
                book, chapter, verse, word_num, word_ref,
                hebrew_text, greek_text, transliteration, translation,
                strongs_raw, morphology_code, strongs_primary, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                word_data["book"],
                word_data["chapter"],
                word_data["verse"],
                word_data["word_num"],
                word_data["word_ref"],
                word_data.get("hebrew_text"),
                word_data.get("greek_text"),
                word_data.get("transliteration"),
                word_data.get("translation"),
                word_data.get("strongs_raw"),
                word_data.get("morphology_code"),
                word_data.get("strongs_primary"),
                word_data["language"],
            ),
        )

    def insert_lexicon_entry(self, lexicon_data: Dict[str, Any]) -> None:
        """Insert a lexicon entry.

        Args:
            lexicon_data: Lexicon information
        """
        query = """
            INSERT OR REPLACE INTO lexicon (
                strongs_number, extended_strongs, disambiguated_strongs,
                unified_strongs, original_word, transliteration,
                part_of_speech, gloss, definition, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                lexicon_data["strongs_number"],
                lexicon_data.get("extended_strongs"),
                lexicon_data.get("disambiguated_strongs"),
                lexicon_data.get("unified_strongs"),
                lexicon_data.get("original_word"),
                lexicon_data.get("transliteration"),
                lexicon_data.get("part_of_speech"),
                lexicon_data.get("gloss"),
                lexicon_data.get("definition"),
                lexicon_data["language"],
            ),
        )

    def insert_lexicon_definition(self, definition_data: Dict[str, Any]) -> None:
        """Insert a supplementary lexicon definition.

        Stores definitions from additional lexicon sources (BDB, Dodson, etc.)
        enabling multi-source comparison for the same Strong's number.

        Args:
            definition_data: Definition information with keys:
                strongs_number, source_lexicon, original_word, transliteration,
                part_of_speech, gloss, definition, language
        """
        query = """
            INSERT OR REPLACE INTO lexicon_definitions (
                strongs_number, source_lexicon, original_word, transliteration,
                part_of_speech, gloss, definition, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                definition_data["strongs_number"],
                definition_data["source_lexicon"],
                definition_data.get("original_word"),
                definition_data.get("transliteration"),
                definition_data.get("part_of_speech"),
                definition_data.get("gloss"),
                definition_data.get("definition"),
                definition_data["language"],
            ),
        )

    def get_lexicon_definitions(self, strongs_number: str) -> List[sqlite3.Row]:
        """Get all supplementary lexicon definitions for a Strong's number.

        Returns definitions from all available sources (BDB, Dodson, etc.)
        for the given Strong's number, enabling multi-source comparison.

        Args:
            strongs_number: Strong's number (e.g., "H0430", "G0026")

        Returns:
            List of definition rows from all sources
        """
        query = "SELECT * FROM lexicon_definitions WHERE strongs_number = ? ORDER BY source_lexicon"
        return self.execute_query(query, (strongs_number,))

    def insert_morphology_entry(self, morphology_data: Dict[str, Any]) -> None:
        """Insert a morphology entry.

        Args:
            morphology_data: Morphology information
        """
        query = """
            INSERT OR REPLACE INTO morphology (code, description, components, language)
            VALUES (?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                morphology_data["code"],
                morphology_data["description"],
                morphology_data.get("components"),
                morphology_data["language"],
            ),
        )

    def get_annotation_cache(self, book_id: int, chapter: int, verse: int) -> Optional[sqlite3.Row]:
        """Get precomputed annotation cache for a verse.

        Args:
            book_id: Book identifier
            chapter: Chapter number
            verse: Verse number

        Returns:
            Cache row or None if not cached
        """
        query = (
            "SELECT words_json, richness_flags_json, cross_references_json, "
            "cultural_context_json, passage_info_json, literary_structures_json, "
            "speaker_json, active_genre "
            "FROM verse_annotations_cache "
            "WHERE book_id = ? AND chapter = ? AND verse = ?"
        )
        try:
            results = self.execute_query(query, (book_id, chapter, verse))
            return results[0] if results else None
        except sqlite3.OperationalError:
            return None

    def upsert_annotation_cache(self, book_id: int, chapter: int, verse: int, data: Dict[str, Any]) -> None:
        """Insert or update precomputed annotation cache for a verse.

        Args:
            book_id: Book identifier
            chapter: Chapter number
            verse: Verse number
            data: Dict with JSON-serialized annotation fields
        """
        query = """
            INSERT OR REPLACE INTO verse_annotations_cache (
                book_id, chapter, verse,
                words_json, richness_flags_json, cross_references_json,
                cultural_context_json, passage_info_json, literary_structures_json,
                speaker_json, active_genre
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute_update(
            query,
            (
                book_id,
                chapter,
                verse,
                data.get("words_json"),
                data.get("richness_flags_json"),
                data.get("cross_references_json"),
                data.get("cultural_context_json"),
                data.get("passage_info_json"),
                data.get("literary_structures_json"),
                data.get("speaker_json"),
                data.get("active_genre"),
            ),
        )

    def invalidate_annotation_cache(self, book_id: Optional[int] = None) -> int:
        """Invalidate (delete) annotation cache entries.

        Args:
            book_id: If provided, only invalidate for this book. Otherwise invalidate all.

        Returns:
            Number of rows deleted
        """
        if book_id is not None:
            return self.execute_update(
                "DELETE FROM verse_annotations_cache WHERE book_id = ?",
                (book_id,),
            )
        return self.execute_update("DELETE FROM verse_annotations_cache")

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with table row counts
        """
        stats = {}
        tables = [
            "words",
            "lexicon",
            "lexicon_definitions",
            "morphology",
            "translations",
            "books",
            "verses",
            "stepbible_verses",
            "verse_annotations_cache",
        ]

        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query)
            stats[table] = result[0]["count"] if result else 0

        # For backward compatibility, add stepbible_verses count to words count
        if "stepbible_verses" in stats and stats["stepbible_verses"] > 0:
            stats["words"] = stats.get("words", 0) + stats["stepbible_verses"]

        return stats
