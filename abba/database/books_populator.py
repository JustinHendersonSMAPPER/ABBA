"""Populate the books table from verse data.

This module derives book metadata (name, chapter count, testament) from the
verses table and inserts rows into the books table.  It is idempotent: it
clears existing rows before inserting so it can be run repeatedly.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical 66-book Protestant canon names keyed by numeric book_id
CANONICAL_BOOK_NAMES: dict[int, str] = {
    1: "Genesis",
    2: "Exodus",
    3: "Leviticus",
    4: "Numbers",
    5: "Deuteronomy",
    6: "Joshua",
    7: "Judges",
    8: "Ruth",
    9: "1 Samuel",
    10: "2 Samuel",
    11: "1 Kings",
    12: "2 Kings",
    13: "1 Chronicles",
    14: "2 Chronicles",
    15: "Ezra",
    16: "Nehemiah",
    17: "Esther",
    18: "Job",
    19: "Psalms",
    20: "Proverbs",
    21: "Ecclesiastes",
    22: "Song of Solomon",
    23: "Isaiah",
    24: "Jeremiah",
    25: "Lamentations",
    26: "Ezekiel",
    27: "Daniel",
    28: "Hosea",
    29: "Joel",
    30: "Amos",
    31: "Obadiah",
    32: "Jonah",
    33: "Micah",
    34: "Nahum",
    35: "Habakkuk",
    36: "Zephaniah",
    37: "Haggai",
    38: "Zechariah",
    39: "Malachi",
    40: "Matthew",
    41: "Mark",
    42: "Luke",
    43: "John",
    44: "Acts",
    45: "Romans",
    46: "1 Corinthians",
    47: "2 Corinthians",
    48: "Galatians",
    49: "Ephesians",
    50: "Philippians",
    51: "Colossians",
    52: "1 Thessalonians",
    53: "2 Thessalonians",
    54: "1 Timothy",
    55: "2 Timothy",
    56: "Titus",
    57: "Philemon",
    58: "Hebrews",
    59: "James",
    60: "1 Peter",
    61: "2 Peter",
    62: "1 John",
    63: "2 John",
    64: "3 John",
    65: "Jude",
    66: "Revelation",
}


def _testament_for(book_id: int) -> str:
    """Return 'old' for books 1-39, 'new' for books 40-66 (matches DB CHECK constraint)."""
    return "old" if book_id <= 39 else "new"


def populate_books(db_path: Path) -> int:
    """Populate the books table from data in the verses table.

    Derives one row per (translation_id, book_id) pair, computing the maximum
    chapter number as number_of_chapters.  Only Protestant canon book IDs
    (1–66) are included.

    This function is idempotent: it deletes existing rows before inserting.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The number of rows inserted.
    """
    conn: sqlite3.Connection = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")

        # Discover (translation_id, book_id, max_chapter) from verses
        cursor = conn.execute(
            """
            SELECT translation_id, book_id, MAX(chapter) AS number_of_chapters
            FROM verses
            WHERE book_id >= 1 AND book_id <= 66
            GROUP BY translation_id, book_id
            ORDER BY translation_id, book_id
            """
        )
        source_rows = cursor.fetchall()

        if not source_rows:
            logger.warning("populate_books: no verse data found; books table will remain empty")
            return 0

        rows_to_insert: list[tuple[str, int, str, str, int, int, str]] = []
        for translation_id, book_id, number_of_chapters in source_rows:
            if book_id not in CANONICAL_BOOK_NAMES:
                logger.debug("populate_books: skipping unknown book_id %d", book_id)
                continue
            name = CANONICAL_BOOK_NAMES[book_id]
            testament = _testament_for(book_id)
            rows_to_insert.append(
                (
                    translation_id,
                    book_id,
                    name,
                    name,  # common_name == canonical name
                    book_id,  # book_order == book_id
                    number_of_chapters,
                    testament,
                )
            )

        with conn:  # transaction
            conn.execute("DELETE FROM books")
            conn.executemany(
                """
                INSERT INTO books
                    (translation_id, book_id, name, common_name, book_order, number_of_chapters, testament)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

        inserted = len(rows_to_insert)
        logger.info("populate_books: inserted %d rows into books table", inserted)
        return inserted
    finally:
        conn.close()
