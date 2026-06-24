"""Tests for GET /api/v1/translations endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.database import SQLiteManager


def _seed_db(tmp_path: Path) -> Path:
    """Create and seed a test database with two translations, each with verses."""
    db_path = tmp_path / "trans_test.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # Insert BSB (default) with a verse
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("BSB", "Berean Standard Bible", "Berean Standard Bible", "eng", "protestant"),
    )
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        ("BSB", 43, 1, 1, "In the beginning was the Word."),
    )

    # Insert a second English translation with a verse
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("KJV", "King James Version", "King James Version", "eng", "protestant"),
    )
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        ("KJV", 43, 1, 1, "In the beginning was the Word."),
    )

    return db_path


def _seed_db_with_empty(tmp_path: Path) -> Path:
    """Create a db where one translation has no verses."""
    db_path = tmp_path / "trans_empty.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # BSB with verse
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("BSB", "Berean Standard Bible", "Berean Standard Bible", "eng", "protestant"),
    )
    db.execute_update(
        "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        ("BSB", 1, 1, 1, "In the beginning God created the heavens and the earth."),
    )

    # Translation with NO verses — should NOT appear in results
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("EMPTY", "Empty Translation", "Empty Translation", "eng", "protestant"),
    )

    return db_path


def test_translations_endpoint_returns_list(tmp_path: Path) -> None:
    """GET /api/v1/translations returns a list of TranslationInfo objects."""
    db_path = _seed_db(tmp_path)
    client = TestClient(create_app(db_path=db_path))

    resp = client.get("/api/v1/translations")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_translations_endpoint_bsb_is_first(tmp_path: Path) -> None:
    """BSB (default translation) must appear first in the response."""
    db_path = _seed_db(tmp_path)
    client = TestClient(create_app(db_path=db_path))

    resp = client.get("/api/v1/translations")
    assert resp.status_code == 200

    data = resp.json()
    assert data[0]["id"] == "BSB", f"Expected BSB first, got {data[0]['id']}"


def test_translations_endpoint_response_shape(tmp_path: Path) -> None:
    """Each item must have id, name, and language fields."""
    db_path = _seed_db(tmp_path)
    client = TestClient(create_app(db_path=db_path))

    resp = client.get("/api/v1/translations")
    assert resp.status_code == 200

    for item in resp.json():
        assert "id" in item
        assert "name" in item
        assert "language" in item


def test_translations_excludes_empty(tmp_path: Path) -> None:
    """Translations with no verses must not appear in the list."""
    db_path = _seed_db_with_empty(tmp_path)
    client = TestClient(create_app(db_path=db_path))

    resp = client.get("/api/v1/translations")
    assert resp.status_code == 200

    data = resp.json()
    ids = [t["id"] for t in data]
    assert "EMPTY" not in ids, "Translation with no verses should be excluded"
    assert "BSB" in ids


def test_translations_ordering(tmp_path: Path) -> None:
    """Default first, then English, then others, each group alphabetical by name."""
    db_path = tmp_path / "order_test.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    for tid, name, lang in [
        ("BSB", "Berean Standard Bible", "eng"),
        ("KJV", "King James Version", "eng"),
        ("SPA", "Reina Valera", "spa"),
    ]:
        db.execute_update(
            "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
            (tid, name, name, lang, "protestant"),
        )
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (tid, 1, 1, 1, "Test verse."),
        )

    client = TestClient(create_app(db_path=db_path))
    resp = client.get("/api/v1/translations")
    assert resp.status_code == 200

    data = resp.json()
    ids = [t["id"] for t in data]

    # BSB must be first (it's the default)
    assert ids[0] == "BSB", f"Expected BSB first, got {ids}"

    # KJV (English) must come before SPA (Spanish)
    assert ids.index("KJV") < ids.index("SPA"), "English translations should precede non-English"
