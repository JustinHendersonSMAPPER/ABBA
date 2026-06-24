"""Tests for books_populator.populate_books."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.api.constants import DEFAULT_TRANSLATION_ID
from abba.database import SQLiteManager
from abba.database.books_populator import populate_books


def _make_db(tmp_path: Path, translation_id: str = "BSB") -> Path:
    """Create and seed a test database with verses for books 1 and 43."""
    db_path = tmp_path / "t.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # Insert translation (required by FK constraint)
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        (translation_id, "Berean Standard Bible", "Berean Standard Bible", "en", "protestant"),
    )

    # Genesis (book_id=1): 3 chapters
    genesis_verses = [
        (1, 1, "In the beginning God created the heavens and the earth."),
        (1, 2, "The earth was without form and void."),
        (2, 1, "Thus the heavens and the earth were finished."),
        (3, 1, "Now the serpent was more crafty than any other beast."),
    ]
    for ch, vs, text in genesis_verses:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (translation_id, 1, ch, vs, text),
        )

    # John (book_id=43): 2 chapters
    john_verses = [
        (1, 1, "In the beginning was the Word."),
        (1, 2, "He was with God in the beginning."),
        (2, 1, "On the third day a wedding took place at Cana in Galilee."),
    ]
    for ch, vs, text in john_verses:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (translation_id, 43, ch, vs, text),
        )

    return db_path


# ---------------------------------------------------------------------------
# Unit tests for populate_books()
# ---------------------------------------------------------------------------


def test_populate_books_returns_count(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)
    count = populate_books(db_path)
    assert count == 2  # Genesis + John


def test_populate_books_genesis_metadata(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)
    populate_books(db_path)

    db = SQLiteManager(db_path)
    rows = db.execute_query(
        "SELECT name, common_name, number_of_chapters, testament, book_order FROM books WHERE book_id = 1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "Genesis"
    assert row[1] == "Genesis"
    assert row[2] == 3
    assert row[3] == "old"
    assert row[4] == 1


def test_populate_books_john_metadata(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)
    populate_books(db_path)

    db = SQLiteManager(db_path)
    rows = db.execute_query(
        "SELECT name, common_name, number_of_chapters, testament, book_order FROM books WHERE book_id = 43",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "John"
    assert row[1] == "John"
    assert row[2] == 2
    assert row[3] == "new"
    assert row[4] == 43


def test_populate_books_idempotent(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)
    count1 = populate_books(db_path)
    count2 = populate_books(db_path)
    assert count1 == count2

    db = SQLiteManager(db_path)
    rows = db.execute_query("SELECT COUNT(*) FROM books")
    assert rows[0][0] == count1  # no duplicate rows


def test_populate_books_empty_db_returns_zero(tmp_path: Path) -> None:
    """No verses → no books rows, returns 0."""
    db_path = tmp_path / "empty.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("BSB", "Berean Standard Bible", "Berean Standard Bible", "en", "protestant"),
    )

    count = populate_books(db_path)
    assert count == 0


def test_populate_books_skips_out_of_range_book_id(tmp_path: Path) -> None:
    """book_id outside 1-66 should be ignored."""
    db_path = tmp_path / "apocrypha.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("BSB", "Berean Standard Bible", "Berean Standard Bible", "en", "protestant"),
    )
    # book_id=70 is outside the Protestant canon range
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        ("BSB", 70, 1, 1, "Apocryphal verse."),
    )

    count = populate_books(db_path)
    assert count == 0


# ---------------------------------------------------------------------------
# Integration test: get_books endpoint with seeded DB
# ---------------------------------------------------------------------------


def test_get_books_endpoint_returns_default_translation_books(tmp_path: Path) -> None:
    """GET /api/v1/books returns books for DEFAULT_TRANSLATION_ID only, no duplicates."""
    db_path = _make_db(tmp_path, translation_id=DEFAULT_TRANSLATION_ID)
    populate_books(db_path)

    client = TestClient(create_app(db_path=db_path))
    resp = client.get("/api/v1/books")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2  # Genesis + John

    book_ids = [b["book_id"] for b in data]
    assert book_ids == sorted(book_ids), "books should be ordered by book_id"
    assert 1 in book_ids
    assert 43 in book_ids

    genesis = next(b for b in data if b["book_id"] == 1)
    assert genesis["name"] == "Genesis"
    assert genesis["chapter_count"] == 3

    john = next(b for b in data if b["book_id"] == 43)
    assert john["name"] == "John"
    assert john["chapter_count"] == 2


def test_get_books_endpoint_empty_when_no_books_populated(tmp_path: Path) -> None:
    """Before populate_books runs, /books returns empty list."""
    db_path = tmp_path / "fresh.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    client = TestClient(create_app(db_path=db_path))
    resp = client.get("/api/v1/books")
    assert resp.status_code == 200
    assert resp.json() == []
