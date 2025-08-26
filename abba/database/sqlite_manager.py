"""SQLite database manager for ABBA."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SQLiteManager:
    """Manages SQLite database operations for ABBA."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.schema_path = Path(__file__).parent / "schema.sql"

    def initialize_database(self) -> None:
        """Initialize database with schema if it doesn't exist."""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating new database at {self.db_path}")

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
            logger.error(f"Database error: {e}")
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

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
            return cursor.rowcount

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
            return cursor.rowcount
    
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
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            self._transaction_conn = None
            if conn:
                conn.close()
    
    def _get_connection_for_transaction(self):
        """Get the current transaction connection if in a transaction."""
        return getattr(self, '_transaction_conn', None)

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
            canon_enum = get_translation_canon(
                translation_data["id"], 
                str(self.db_path.parent / "bible.db")
            )
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

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with table row counts
        """
        stats = {}
        tables = ["words", "lexicon", "morphology", "translations", "books", "verses", "stepbible_verses"]

        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query)
            stats[table] = result[0]["count"] if result else 0

        # For backward compatibility, add stepbible_verses count to words count
        if "stepbible_verses" in stats and stats["stepbible_verses"] > 0:
            stats["words"] = stats.get("words", 0) + stats["stepbible_verses"]

        return stats
