"""Tests for the Strong's concordance feature.

Covers:
- add_stepbible_lexical_strongs_column migration (idempotent)
- populate_lexical_strongs populator
- SQLiteManager.search_strongs / count_strongs_occurrences
- /api/v1/search/strongs/{strongs_number} endpoint
"""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from abba.database.migrations import add_stepbible_lexical_strongs_column
from abba.database.sqlite_manager import SQLiteManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_stepbible_verse(
    conn: sqlite3.Connection,
    *,
    book: str,
    chapter: int,
    verse: int,
    word_number: int,
    strongs_primary: str,
    strongs_raw: str,
    language: str = "greek",
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO stepbible_verses
            (source_file, book, chapter, verse, word_number,
             original_word, transliteration, english,
             strongs_raw, strongs_primary, morphology, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "test_source",
            book,
            chapter,
            verse,
            word_number,
            "word",
            "translit",
            "english",
            strongs_raw,
            strongs_primary,
            "V-PIA-3S",
            language,
        ),
    )


def _insert_verse(
    conn: sqlite3.Connection,
    *,
    translation_id: str,
    book_id: int,
    chapter: int,
    verse: int,
    text: str,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
        (translation_id, book_id, chapter, verse, text),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_with_data(tmp_path: Path) -> SQLiteManager:
    """Fresh DB with migration, test STEPBible rows, and BSB verse texts."""
    db_path = tmp_path / "test.db"
    mgr = SQLiteManager(db_path)
    mgr.initialize_database()

    with sqlite3.connect(str(db_path)) as conn:
        # --- STEPBible rows ---
        # Jhn 1:1 — "In the beginning was the Word" (logos, G3056)
        _insert_stepbible_verse(
            conn,
            book="Jhn",
            chapter=1,
            verse=1,
            word_number=1,
            strongs_primary="G3056",
            strongs_raw="{G3056}",
        )
        # Jhn 1:14 — second occurrence of logos (G3056)
        _insert_stepbible_verse(
            conn,
            book="Jhn",
            chapter=1,
            verse=14,
            word_number=1,
            strongs_primary="G3056",
            strongs_raw="{G3056}",
        )
        # Jhn 1:1 word 2 — different Greek word (theos, G2316) — shouldn't match G3056
        _insert_stepbible_verse(
            conn,
            book="Jhn",
            chapter=1,
            verse=1,
            word_number=2,
            strongs_primary="G2316",
            strongs_raw="{G2316}",
        )
        # Gen 1:1 — Hebrew word "God" (H0430 padded)
        _insert_stepbible_verse(
            conn,
            book="Gen",
            chapter=1,
            verse=1,
            word_number=1,
            strongs_primary="H0430",
            strongs_raw="{H0430}",
            language="hebrew",
        )

        # --- BSB verse texts ---
        # Translation row first
        conn.execute(
            "INSERT OR IGNORE INTO translations (id, name, language) VALUES (?, ?, ?)",
            ("BSB", "Berean Standard Bible", "eng"),
        )
        _insert_verse(conn, translation_id="BSB", book_id=43, chapter=1, verse=1, text="In the beginning was the Word")
        _insert_verse(conn, translation_id="BSB", book_id=43, chapter=1, verse=14, text="The Word became flesh")
        _insert_verse(conn, translation_id="BSB", book_id=1, chapter=1, verse=1, text="In the beginning God created")

        conn.commit()

    # Run the populator (migration already added column via initialize_database)
    from abba.database.lexical_strongs_populator import populate_lexical_strongs

    populate_lexical_strongs(db_path)

    return mgr


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


class TestMigration:
    def test_column_added_by_initialize_database(self, tmp_path: Path) -> None:
        """initialize_database runs migrations including lexical_strongs column."""
        db_path = tmp_path / "m.db"
        mgr = SQLiteManager(db_path)
        mgr.initialize_database()

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("PRAGMA table_info(stepbible_verses)")
            columns = {row[1] for row in cursor.fetchall()}
        assert "lexical_strongs" in columns

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        """Running the migration twice does not raise and returns False on 2nd call."""
        db_path = tmp_path / "idem.db"
        mgr = SQLiteManager(db_path)
        mgr.initialize_database()

        result = add_stepbible_lexical_strongs_column(db_path)
        assert result is False  # already exists

    def test_migration_returns_false_if_no_table(self, tmp_path: Path) -> None:
        """If stepbible_verses doesn't exist the migration returns False silently."""
        # Create an empty SQLite file without any schema
        db_path = tmp_path / "empty.db"
        sqlite3.connect(str(db_path)).close()
        result = add_stepbible_lexical_strongs_column(db_path)
        assert result is False


# ---------------------------------------------------------------------------
# Populator tests
# ---------------------------------------------------------------------------


class TestPopulator:
    def test_populate_returns_non_empty_count(self, db_with_data: SQLiteManager) -> None:
        """populate_lexical_strongs returns the count of rows with a non-NULL key."""
        # Already ran in fixture; re-run to verify idempotence
        from abba.database.lexical_strongs_populator import populate_lexical_strongs

        count = populate_lexical_strongs(db_with_data.db_path)
        # 4 rows total, all have valid Strong's numbers
        assert count == 4

    def test_keys_written_correctly(self, tmp_path: Path) -> None:
        """Verifies the actual key values stored in the column."""
        db_path = tmp_path / "keys.db"
        mgr = SQLiteManager(db_path)
        mgr.initialize_database()

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO stepbible_verses "
                "(source_file, book, chapter, verse, word_number, original_word, "
                "transliteration, english, strongs_raw, strongs_primary, morphology, language) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("src", "Jhn", 1, 1, 1, "w", "t", "e", "{G3056}", "G3056", "N", "greek"),
            )
            conn.commit()

        from abba.database.lexical_strongs_populator import populate_lexical_strongs

        populate_lexical_strongs(db_path)

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT lexical_strongs FROM stepbible_verses WHERE book='Jhn'").fetchone()

        assert row is not None
        assert row[0] == "G3056"  # normalized: no leading zeros


# ---------------------------------------------------------------------------
# SQLiteManager.search_strongs tests
# ---------------------------------------------------------------------------


class TestSearchStrongs:
    def test_returns_two_john_verses(self, db_with_data: SQLiteManager) -> None:
        rows = db_with_data.search_strongs("G3056")
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        refs = {(r["book"], r["chapter"], r["verse"]) for r in rows}
        assert ("Jhn", 1, 1) in refs
        assert ("Jhn", 1, 14) in refs

    def test_padded_input_matches(self, db_with_data: SQLiteManager) -> None:
        """G03056 should normalize to G3056 and match the same verses."""
        rows = db_with_data.search_strongs("G03056")
        assert len(rows) == 2

    def test_lowercase_input_matches(self, db_with_data: SQLiteManager) -> None:
        rows = db_with_data.search_strongs("g3056")
        assert len(rows) == 2

    def test_hebrew_not_matched_by_greek_search(self, db_with_data: SQLiteManager) -> None:
        """Searching G3056 must not return Hebrew H3056 (different prefix)."""
        rows = db_with_data.search_strongs("H3056")
        assert len(rows) == 0

    def test_greek_not_matched_by_hebrew_search(self, db_with_data: SQLiteManager) -> None:
        """Searching H430 must only return OT verse, not NT verse."""
        rows = db_with_data.search_strongs("H430")
        assert len(rows) == 1
        assert rows[0]["book"] == "Gen"

    def test_h0430_padded_matches_h430(self, db_with_data: SQLiteManager) -> None:
        """H0430 and H430 should resolve to the same key."""
        rows_padded = db_with_data.search_strongs("H0430")
        rows_unpadded = db_with_data.search_strongs("H430")
        assert len(rows_padded) == len(rows_unpadded) == 1

    def test_limit_respected(self, db_with_data: SQLiteManager) -> None:
        rows = db_with_data.search_strongs("G3056", limit=1)
        assert len(rows) == 1

    def test_no_results_for_unknown(self, db_with_data: SQLiteManager) -> None:
        rows = db_with_data.search_strongs("G9999")
        assert rows == []


# ---------------------------------------------------------------------------
# SQLiteManager.count_strongs_occurrences tests
# ---------------------------------------------------------------------------


class TestCountStrongsOccurrences:
    def test_count_two_for_logos(self, db_with_data: SQLiteManager) -> None:
        assert db_with_data.count_strongs_occurrences("G3056") == 2

    def test_count_ignores_limit(self, db_with_data: SQLiteManager) -> None:
        """count_strongs_occurrences has no limit; stays 2 regardless."""
        assert db_with_data.count_strongs_occurrences("G3056") == 2

    def test_count_zero_for_unknown(self, db_with_data: SQLiteManager) -> None:
        assert db_with_data.count_strongs_occurrences("G9999") == 0

    def test_count_hebrew(self, db_with_data: SQLiteManager) -> None:
        assert db_with_data.count_strongs_occurrences("H430") == 1


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client(db_with_data: SQLiteManager) -> TestClient:
    """TestClient wired to the db_with_data fixture."""
    from fastapi import FastAPI

    from abba.api.routes import configure_db, router

    configure_db(db_with_data)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestStrongsEndpoint:
    def test_returns_occurrences(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G3056")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_response_shape(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G3056")
        item = resp.json()[0]
        assert "strongs_number" in item
        assert "book_id" in item
        assert "book_name" in item
        assert "chapter" in item
        assert "verse" in item
        assert "text" in item

    def test_strongs_number_in_response_is_normalized(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G03056")
        data = resp.json()
        # All items should carry the normalized key
        assert all(d["strongs_number"] == "G3056" for d in data)

    def test_text_populated_from_bsb(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G3056")
        texts = {d["text"] for d in resp.json()}
        assert "In the beginning was the Word" in texts
        assert "The Word became flesh" in texts

    def test_sorted_by_book_chapter_verse(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G3056")
        data = resp.json()
        keys = [(d["book_id"], d["chapter"], d["verse"]) for d in data]
        assert keys == sorted(keys)

    def test_book_id_is_numeric_for_john(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G3056")
        for item in resp.json():
            assert item["book_id"] == 43  # John

    def test_limit_query_param(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/G3056?limit=1")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_hebrew_concordance(self, test_client: TestClient) -> None:
        resp = test_client.get("/api/v1/search/strongs/H430")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["book_id"] == 1  # Genesis

    def test_no_cross_prefix_contamination(self, test_client: TestClient) -> None:
        """H3056 must return nothing (H and G prefixes are distinct)."""
        resp = test_client.get("/api/v1/search/strongs/H3056")
        assert resp.status_code == 200
        assert resp.json() == []
