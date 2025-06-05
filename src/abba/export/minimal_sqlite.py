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
