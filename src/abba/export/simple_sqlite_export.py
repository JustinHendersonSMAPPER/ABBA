"""
Simple SQLite export for ABBA canonical data.

A straightforward, synchronous implementation that creates a mobile-friendly
SQLite database with verses, cross-references, and basic search.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..verse_id import VerseID
from ..parsers.translation_parser import TranslationVerse


@dataclass
class SimpleSQLiteConfig:
    """Configuration for simple SQLite export."""
    
    output_path: str
    enable_fts: bool = True
    page_size: int = 4096  # Optimized for mobile
    

class SimpleSQLiteExporter:
    """Simple SQLite exporter for biblical data."""
    
    def __init__(self, config: SimpleSQLiteConfig):
        """Initialize the exporter."""
        self.config = config
        self.conn: Optional[sqlite3.Connection] = None
        
    def export_verses(self, verses: List[TranslationVerse]) -> None:
        """
        Export verses to SQLite database.
        
        Args:
            verses: List of verses to export
        """
        # Ensure output directory exists
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database
        self.conn = sqlite3.connect(str(output_path))
        self.conn.execute(f"PRAGMA page_size = {self.config.page_size}")
        self.conn.execute("PRAGMA journal_mode = WAL")
        
        try:
            self._create_schema()
            self._insert_verses(verses)
            if self.config.enable_fts:
                self._create_search_index()
            self._create_indexes()
            self.conn.commit()
        finally:
            self.conn.close()
            self.conn = None
            
    def _create_schema(self) -> None:
        """Create database schema."""
        # Books table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                book_code TEXT PRIMARY KEY,
                book_name TEXT NOT NULL,
                testament TEXT NOT NULL,
                chapter_count INTEGER NOT NULL
            )
        """)
        
        # Verses table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS verses (
                verse_id TEXT PRIMARY KEY,
                book_code TEXT NOT NULL,
                chapter INTEGER NOT NULL,
                verse INTEGER NOT NULL,
                text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                FOREIGN KEY (book_code) REFERENCES books(book_code)
            )
        """)
        
        # Cross references table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_verse TEXT NOT NULL,
                to_verse TEXT NOT NULL,
                reference_type TEXT DEFAULT 'standard',
                FOREIGN KEY (from_verse) REFERENCES verses(verse_id),
                FOREIGN KEY (to_verse) REFERENCES verses(verse_id)
            )
        """)
        
        # Metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
    def _insert_verses(self, verses: List[TranslationVerse]) -> None:
        """Insert verses into database."""
        # Track books
        books_data = {}
        
        # Prepare verse data
        verse_data = []
        for verse in verses:
            verse_id_str = str(verse.verse_id)
            book_code = verse.verse_id.book
            
            # Track book info
            if book_code not in books_data:
                books_data[book_code] = {
                    'name': verse.original_book_name,
                    'max_chapter': verse.verse_id.chapter
                }
            else:
                books_data[book_code]['max_chapter'] = max(
                    books_data[book_code]['max_chapter'],
                    verse.verse_id.chapter
                )
            
            # Count words (simple split)
            word_count = len(verse.text.split())
            
            verse_data.append((
                verse_id_str,
                book_code,
                verse.verse_id.chapter,
                verse.verse_id.verse,
                verse.text,
                word_count
            ))
        
        # Insert books
        from ..book_codes import get_book_info, get_book_order
        book_rows = []
        for book_code, book_data in books_data.items():
            book_info = get_book_info(book_code)
            book_order = get_book_order(book_code)
            if book_info and book_order:
                testament = "OT" if book_order <= 39 else "NT"
                book_rows.append((
                    book_code,
                    book_data['name'],
                    testament,
                    book_data['max_chapter']
                ))
        
        self.conn.executemany(
            "INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?)",
            book_rows
        )
        
        # Insert verses
        self.conn.executemany(
            "INSERT OR REPLACE INTO verses VALUES (?, ?, ?, ?, ?, ?)",
            verse_data
        )
        
        # Insert metadata
        metadata = [
            ('version', '1.0'),
            ('verse_count', str(len(verses))),
            ('book_count', str(len(books_data))),
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            metadata
        )
        
    def _create_search_index(self) -> None:
        """Create FTS5 search index."""
        # Create FTS5 table
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
                verse_id UNINDEXED,
                text,
                content=verses,
                content_rowid=rowid
            )
        """)
        
        # Populate FTS index
        self.conn.execute("""
            INSERT INTO verses_fts(verse_id, text)
            SELECT verse_id, text FROM verses
        """)
        
    def _create_indexes(self) -> None:
        """Create database indexes."""
        # Book and chapter index for fast lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_verses_book_chapter 
            ON verses(book_code, chapter)
        """)
        
        # Cross reference indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_xref_from 
            ON cross_references(from_verse)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_xref_to 
            ON cross_references(to_verse)
        """)
        
    def add_cross_references(self, references: List[Dict[str, str]]) -> None:
        """
        Add cross references to existing database.
        
        Args:
            references: List of dicts with 'from_verse', 'to_verse', and optional 'type'
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.config.output_path)
            
        try:
            reference_data = [
                (ref['from_verse'], ref['to_verse'], ref.get('type', 'standard'))
                for ref in references
            ]
            
            self.conn.executemany(
                "INSERT INTO cross_references (from_verse, to_verse, reference_type) VALUES (?, ?, ?)",
                reference_data
            )
            self.conn.commit()
        finally:
            self.conn.close()
            self.conn = None


def export_to_sqlite(verses: List[TranslationVerse], output_path: str) -> None:
    """
    Convenience function to export verses to SQLite.
    
    Args:
        verses: List of verses to export
        output_path: Path to output SQLite file
    """
    config = SimpleSQLiteConfig(output_path=output_path)
    exporter = SimpleSQLiteExporter(config)
    exporter.export_verses(verses)