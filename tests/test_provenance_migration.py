"""Tests for the provenance table migration."""

import sqlite3
from pathlib import Path

from abba.database.migrations import add_provenance_table


def test_add_provenance_table_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    sqlite3.connect(db_path).close()  # create an empty database file

    assert add_provenance_table(db_path) is True

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='provenance'").fetchone()
    assert row is not None


def test_add_provenance_table_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    sqlite3.connect(db_path).close()

    add_provenance_table(db_path)
    assert add_provenance_table(db_path) is False
