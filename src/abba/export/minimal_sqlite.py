"""
Minimal SQLite export for ABBA data.

A dependency-minimal implementation for creating SQLite databases
from biblical verse data.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MinimalVerse:
    """Minimal verse representation."""

    book: str
    chapter: int
    verse: int
    text: str
    translation: str = "KJV"


class MinimalSQLiteExporter:
    """Simple SQLite exporter for verse data."""
    
    def __init__(self, output_path: str):
        """Initialize exporter with output path."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection and initialize database
        self.conn = sqlite3.connect(str(self.output_path))
        self._create_tables()
        self.verse_count = 0
    
    def _create_tables(self) -> None:
        """Create database tables."""
        # Set pragmas for performance
        self.conn.execute("PRAGMA page_size = 4096")
        self.conn.execute("PRAGMA journal_mode = WAL")
        
        # Create verses table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS verses (
                id INTEGER PRIMARY KEY,
                verse_id TEXT UNIQUE NOT NULL,
                book TEXT NOT NULL,
                chapter INTEGER NOT NULL,
                verse INTEGER NOT NULL,
                text TEXT NOT NULL,
                translation TEXT DEFAULT 'KJV'
            )
        """)
        
        # Create indices
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_book_chapter_verse 
            ON verses(book, chapter, verse)
        """)
        
        self.conn.commit()
    
    def add_verse(self, verse_id: str, book: str, chapter: int, 
                  verse: int, text: str, translation: str = "KJV") -> None:
        """Add a verse to the database."""
        self.conn.execute("""
            INSERT OR REPLACE INTO verses 
            (verse_id, book, chapter, verse, text, translation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (verse_id, book, chapter, verse, text, translation))
        self.verse_count += 1
        
        # Commit every 1000 verses for performance
        if self.verse_count % 1000 == 0:
            self.conn.commit()
    
    def finalize(self) -> None:
        """Finalize the export and close database."""
        # Final commit
        self.conn.commit()
        
        # Create full-text search table
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
                verse_id, book, chapter, verse, text,
                content='verses',
                content_rowid='id'
            )
        """)
        
        # Populate FTS table
        self.conn.execute("""
            INSERT INTO verses_fts(verse_id, book, chapter, verse, text)
            SELECT verse_id, book, chapter, verse, text FROM verses
        """)
        
        # Commit and close
        self.conn.commit()
        self.conn.close()
        
        # VACUUM must be done with a new connection outside of any transaction
        vacuum_conn = sqlite3.connect(str(self.output_path))
        vacuum_conn.execute("VACUUM")
        vacuum_conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.finalize()


def create_sqlite_database(
    verses: List[MinimalVerse], output_path: str, enable_fts: bool = True
) -> None:
    """
    Create a SQLite database from verse data.

    Args:
        verses: List of verses to export
        output_path: Path to output SQLite file
        enable_fts: Whether to enable full-text search
    """
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create connection
    conn = sqlite3.connect(str(output_file))
    conn.execute("PRAGMA page_size = 4096")
    conn.execute("PRAGMA journal_mode = WAL")

    # Create verses table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verses (
            id INTEGER PRIMARY KEY,
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            text TEXT NOT NULL,
            translation TEXT NOT NULL,
            UNIQUE(book, chapter, verse, translation)
        )
    """
    )

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_book_chapter ON verses(book, chapter)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_translation ON verses(translation)")

    # Insert verses
    for v in verses:
        conn.execute(
            "INSERT OR REPLACE INTO verses (book, chapter, verse, text, translation) VALUES (?, ?, ?, ?, ?)",
            (v.book, v.chapter, v.verse, v.text, v.translation),
        )

    # Create FTS table if enabled
    if enable_fts:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
                book, chapter, verse, text, translation,
                content='verses',
                content_rowid='id'
            )
        """
        )

        # Populate FTS
        conn.execute(
            """
            INSERT INTO verses_fts(verses_fts) VALUES ('rebuild')
        """
        )

    # Create metadata table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """
    )

    # Add metadata
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("verse_count", str(len(verses))),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("fts_enabled", "true" if enable_fts else "false"),
    )

    # Commit and close
    conn.commit()
    conn.close()


def query_verses(
    db_path: str, book: Optional[str] = None, chapter: Optional[int] = None
) -> List[Dict]:
    """
    Query verses from SQLite database.

    Args:
        db_path: Path to SQLite database
        book: Optional book filter
        chapter: Optional chapter filter

    Returns:
        List of verse dictionaries
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM verses WHERE 1=1"
    params = []

    if book:
        query += " AND book = ?"
        params.append(book)
    if chapter is not None:
        query += " AND chapter = ?"
        params.append(chapter)

    query += " ORDER BY book, chapter, verse"

    cursor = conn.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return results


def search_verses(db_path: str, search_term: str) -> List[Dict]:
    """
    Full-text search for verses.

    Args:
        db_path: Path to SQLite database
        search_term: Search term

    Returns:
        List of matching verses
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if FTS is available
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='verses_fts'")
    if not cursor.fetchone():
        conn.close()
        raise ValueError("Full-text search not enabled for this database")

    # Perform search
    query = """
        SELECT v.* FROM verses v
        JOIN verses_fts fts ON v.id = fts.rowid
        WHERE verses_fts MATCH ?
        ORDER BY rank
    """

    cursor = conn.execute(query, (search_term,))
    results = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return results
