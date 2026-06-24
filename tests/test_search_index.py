"""Tests for search_index.rebuild_search_index."""

import sqlite3
from pathlib import Path

import pytest

from abba.database import SQLiteManager
from abba.database.search_index import rebuild_search_index

TRANSLATION_ID = "BSB"


def _make_db(tmp_path: Path) -> Path:
    """Create a test database with schema and a few seeded verses."""
    db_path = tmp_path / "t.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # Insert translation (required by FK constraint)
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        (TRANSLATION_ID, "Berean Standard Bible", "Berean Standard Bible", "en", "protestant"),
    )

    # John 3:16 — "God is love" variant for search testing
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        (TRANSLATION_ID, 43, 3, 16, "For God so loved the world that he gave his one and only Son."),
    )
    # 1 John 4:8 — contains "love"
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        (TRANSLATION_ID, 62, 4, 8, "Whoever does not love does not know God, because God is love."),
    )
    # Genesis 1:1 — does NOT contain "love"
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        (TRANSLATION_ID, 1, 1, 1, "In the beginning God created the heavens and the earth."),
    )

    return db_path


# ---------------------------------------------------------------------------
# Core rebuild tests
# ---------------------------------------------------------------------------


def test_rebuild_returns_positive_count(tmp_path: Path) -> None:
    """rebuild_search_index returns a positive integer after a successful build."""
    db_path = _make_db(tmp_path)
    count = rebuild_search_index(db_path)
    assert count >= 1, f"Expected at least 1 indexed document, got {count}"


def test_fts_match_finds_verses_after_rebuild(tmp_path: Path) -> None:
    """After rebuild, a MATCH query on 'love' finds the expected verses."""
    db_path = _make_db(tmp_path)
    rebuild_search_index(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM verses_fts WHERE verses_fts MATCH ? AND translation_id = ?",
            ("love", TRANSLATION_ID),
        ).fetchone()
        assert row is not None
        assert row[0] >= 1, f"Expected at least 1 match for 'love', got {row[0]}"
    finally:
        conn.close()


def test_fts_match_translation_scoped(tmp_path: Path) -> None:
    """MATCH with translation_id filter returns 0 for a non-existent translation."""
    db_path = _make_db(tmp_path)
    rebuild_search_index(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM verses_fts WHERE verses_fts MATCH ? AND translation_id = ?",
            ("love", "ESV"),
        ).fetchone()
        assert row is not None
        assert row[0] == 0, f"Expected 0 matches for unknown translation, got {row[0]}"
    finally:
        conn.close()


def test_fts_no_match_for_absent_word(tmp_path: Path) -> None:
    """After rebuild, a MATCH query for a word not in any verse returns 0."""
    db_path = _make_db(tmp_path)
    rebuild_search_index(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM verses_fts WHERE verses_fts MATCH ? AND translation_id = ?",
            ("xyzzy", TRANSLATION_ID),
        ).fetchone()
        assert row is not None
        assert row[0] == 0
    finally:
        conn.close()


def test_rebuild_is_idempotent(tmp_path: Path) -> None:
    """Calling rebuild_search_index twice does not raise and count stays consistent."""
    db_path = _make_db(tmp_path)
    count1 = rebuild_search_index(db_path)
    count2 = rebuild_search_index(db_path)
    assert count1 == count2, "Idempotent rebuild should yield same document count"


def test_rebuild_returns_zero_if_fts_table_missing(tmp_path: Path) -> None:
    """If verses_fts was dropped, rebuild_search_index returns 0 without crashing."""
    db_path = _make_db(tmp_path)

    # Drop the FTS table to simulate a schema-less DB
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS verses_fts")
    conn.commit()
    conn.close()

    result = rebuild_search_index(db_path)
    assert result == 0, f"Expected 0 when FTS table is absent, got {result}"
