"""Tests for the public-domain dictionary -> verse linker (decision D5)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from abba.database.dictionary_linker import (
    OSIS_TO_BOOK_ID,
    link_dictionary_entries,
    parse_osis_ref,
)
from abba.database.migrations import add_dictionary_entries_table


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("Exod.6.20", (2, 6, 20)),
        ("Gen.1.1", (1, 1, 1)),
        ("Rev.22.21", (66, 22, 21)),
        ("Ps.23.1", (19, 23, 1)),
        ("Gen.1.27-Gen.1.30", (1, 1, 27)),  # range -> start verse
        ("Num.12", None),  # chapter-only is too broad
        ("Foo.1.1", None),  # unknown book
        ("", None),
        ("garbage", None),
        ("Gen.x.y", None),  # non-numeric
    ],
)
def test_parse_osis_ref(ref: str, expected: object) -> None:
    assert parse_osis_ref(ref) == expected


def test_osis_map_covers_66_books() -> None:
    """Every Protestant book id 1..66 is reachable from at least one OSIS code."""
    assert set(OSIS_TO_BOOK_ID.values()) == set(range(1, 67))


def _seed_entries(tmp_path: Path) -> str:
    db_path = str(tmp_path / "dict.db")
    add_dictionary_entries_table(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        for headword, norm, refs in [
            ("Aaron", "AARON", ["Exod.6.20", "Num.12", "Gen.1.27-Gen.1.30"]),  # 2 links + 1 skip
            ("Moriah", "MORIAH", ["Gen.22.2"]),  # 1 link
            ("NoRefs", "NOREFS", None),  # ignored (no ref_targets)
        ]:
            conn.execute(
                "INSERT INTO dictionary_entries "
                "(headword, headword_normalized, article, ref_targets, source, license) VALUES (?, ?, ?, ?, ?, ?)",
                (headword, norm, f"{headword} article", json.dumps(refs) if refs else None, "Easton's", "PD"),
            )
        conn.commit()
    return db_path


def test_link_dictionary_entries(tmp_path: Path) -> None:
    db_path = _seed_entries(tmp_path)
    stats = link_dictionary_entries(db_path)

    assert stats["entries"] == 2  # only the two with ref_targets
    assert stats["skipped"] == 1  # Num.12 chapter-only
    assert stats["links"] == 3  # Exod.6.20, Gen.1.27 (range start), Gen.22.2

    with closing(sqlite3.connect(db_path)) as conn:
        exod = conn.execute(
            "SELECT d.headword FROM verse_dictionary_links l JOIN dictionary_entries d ON d.entry_id = l.entry_id "
            "WHERE l.book_id = 2 AND l.chapter = 6 AND l.verse = 20"
        ).fetchall()
        assert [r[0] for r in exod] == ["Aaron"]
        moriah = conn.execute(
            "SELECT d.headword FROM verse_dictionary_links l JOIN dictionary_entries d ON d.entry_id = l.entry_id "
            "WHERE l.book_id = 1 AND l.chapter = 22 AND l.verse = 2"
        ).fetchall()
        assert [r[0] for r in moriah] == ["Moriah"]
        # chapter-only Num.12 produced no link
        num12 = conn.execute(
            "SELECT COUNT(*) FROM verse_dictionary_links WHERE book_id = 4 AND chapter = 12"
        ).fetchone()[0]
        assert num12 == 0


def test_link_is_idempotent(tmp_path: Path) -> None:
    db_path = _seed_entries(tmp_path)
    first = link_dictionary_entries(db_path)
    second = link_dictionary_entries(db_path)
    assert first["links"] == 3
    assert second["links"] == 0  # nothing new on re-run
