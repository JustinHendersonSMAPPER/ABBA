"""Tests for /books endpoint fallback when default translation is absent."""

from pathlib import Path

from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.database import SQLiteManager
from abba.database.books_populator import populate_books


def _make_db_non_default(tmp_path: Path, translation_id: str = "eng_test") -> Path:
    """Create and seed a test database with a non-default translation."""
    db_path = tmp_path / "fallback.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # Insert translation
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        (translation_id, "Test Translation", "Test Translation", "en", "protestant"),
    )

    # Genesis (book_id=1): 3 chapters
    for ch, vs, text in [
        (1, 1, "In the beginning God created the heavens and the earth."),
        (2, 1, "Thus the heavens and the earth were finished."),
        (3, 1, "Now the serpent was more crafty than any other beast."),
    ]:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (translation_id, 1, ch, vs, text),
        )

    # John (book_id=43): 2 chapters
    for ch, vs, text in [
        (1, 1, "In the beginning was the Word."),
        (2, 1, "On the third day a wedding took place at Cana in Galilee."),
    ]:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (translation_id, 43, ch, vs, text),
        )

    return db_path


def test_books_endpoint_fallback_when_no_default_translation(tmp_path: Path) -> None:
    """GET /api/v1/books falls back to any translation when default (BSB) is absent."""
    db_path = _make_db_non_default(tmp_path, translation_id="eng_test")
    populate_books(db_path)

    client = TestClient(create_app(db_path=db_path))
    resp = client.get("/api/v1/books")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2, f"Expected 2 books (Genesis + John), got {len(data)}"

    book_ids = [b["book_id"] for b in data]
    assert book_ids == sorted(book_ids), "Books should be ordered by book_id"
    assert 1 in book_ids, "Genesis (book_id=1) should be in fallback results"
    assert 43 in book_ids, "John (book_id=43) should be in fallback results"

    genesis = next(b for b in data if b["book_id"] == 1)
    assert genesis["name"] == "Genesis"
    assert genesis["chapter_count"] == 3

    john = next(b for b in data if b["book_id"] == 43)
    assert john["name"] == "John"
    assert john["chapter_count"] == 2


def test_books_endpoint_fallback_returns_distinct_books(tmp_path: Path) -> None:
    """Fallback returns one row per book_id, not duplicates across translations."""
    db_path = tmp_path / "multi.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    for tid, tname in [("eng_test1", "Test 1"), ("eng_test2", "Test 2")]:
        db.execute_update(
            "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
            (tid, tname, tname, "en", "protestant"),
        )
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (tid, 1, 1, 1, "Genesis verse in " + tname),
        )

    populate_books(db_path)

    client = TestClient(create_app(db_path=db_path))
    resp = client.get("/api/v1/books")
    assert resp.status_code == 200

    data = resp.json()
    book_ids = [b["book_id"] for b in data]
    assert book_ids.count(1) == 1, "book_id=1 must appear exactly once even with multiple translations"
