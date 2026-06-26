"""Link public-domain dictionary entries to verses via their own verse citations.

The conservative, ambiguity-free v1 linker (decision D5): an Easton entry's article
cites specific verses (`ref_targets`, OSIS refs like ``Exod.6.20``). When the dictionary
itself points at a verse, that is a source-vouched link — no NER, no disambiguation, no
LLM. We parse those citations into ``verse_dictionary_links`` so a verse can surface the
exact PD articles that reference it (Tier A — verbatim fact, fully auditable).

Ranges (``Gen.1.27-Gen.1.30``) link their start verse; chapter-only refs (``Num.12``) are
skipped as too broad. This is intentionally precise — a wrong context is worse than none.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OSIS book abbreviation -> numeric book_id (Protestant 66). Includes a few CCEL/legacy
# alternates seen in the Easton module alongside the OSIS standard.
OSIS_TO_BOOK_ID: dict[str, int] = {
    "Gen": 1, "Exod": 2, "Lev": 3, "Num": 4, "Deut": 5, "Josh": 6, "Judg": 7, "Ruth": 8,
    "1Sam": 9, "2Sam": 10, "1Kgs": 11, "2Kgs": 12, "1Chr": 13, "2Chr": 14, "Ezra": 15,
    "Neh": 16, "Esth": 17, "Job": 18, "Ps": 19, "Psa": 19, "Prov": 20, "Eccl": 21,
    "Song": 22, "Sng": 22, "Isa": 23, "Jer": 24, "Lam": 25, "Ezek": 26, "Dan": 27,
    "Hos": 28, "Joel": 29, "Amos": 30, "Obad": 31, "Jonah": 32, "Mic": 33, "Nah": 34,
    "Hab": 35, "Zeph": 36, "Hag": 37, "Zech": 38, "Mal": 39, "Matt": 40, "Mark": 41,
    "Luke": 42, "John": 43, "Acts": 44, "Rom": 45, "1Cor": 46, "2Cor": 47, "Gal": 48,
    "Eph": 49, "Phil": 50, "Col": 51, "1Thess": 52, "2Thess": 53, "1Tim": 54, "2Tim": 55,
    "Titus": 56, "Phlm": 57, "Phm": 57, "Heb": 58, "Jas": 59, "1Pet": 60, "2Pet": 61,
    "1John": 62, "2John": 63, "3John": 64, "Jude": 65, "Rev": 66,
}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS verse_dictionary_links (
    link_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id      INTEGER NOT NULL,
    chapter      INTEGER NOT NULL,
    verse        INTEGER NOT NULL,
    entry_id     INTEGER NOT NULL,
    match_method TEXT NOT NULL,
    confidence   REAL NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_id, chapter, verse, entry_id),
    FOREIGN KEY (entry_id) REFERENCES dictionary_entries(entry_id)
)
"""
_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_vdl_verse ON verse_dictionary_links(book_id, chapter, verse)"
)

# Citations to a verse this many times or more across distinct entries are unremarkable
# (e.g. very common verses) — not filtered here, but display is capped in the API.


def parse_osis_ref(ref: str) -> Optional[tuple[int, int, int]]:
    """Parse a single OSIS reference into ``(book_id, chapter, verse)``.

    Handles single verses (``Exod.6.20``) and takes the start of a range
    (``Gen.1.27-Gen.1.30`` -> Gen 1:27). Returns None for chapter-only refs
    (``Num.12``), unknown books, or malformed input.

    Args:
        ref: An OSIS reference string (the part after ``Bible:``).

    Returns:
        ``(book_id, chapter, verse)`` or None if it cannot be resolved to a verse.
    """
    if not ref:
        return None
    start = ref.split("-", 1)[0].strip()  # range -> start verse
    parts = start.split(".")
    if len(parts) != 3:  # need book.chapter.verse; chapter-only is too broad
        return None
    book, chap, verse = parts
    book_id = OSIS_TO_BOOK_ID.get(book)
    if book_id is None:
        return None
    try:
        return book_id, int(chap), int(verse)
    except ValueError:
        return None


def create_link_table(db_path: str | Path) -> None:
    """Create the ``verse_dictionary_links`` table + index if absent (idempotent)."""
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()


def link_dictionary_entries(db_path: str | Path, confidence: float = 1.0) -> dict[str, int]:
    """Build verse->entry links from every dictionary entry's cited verses.

    Source-vouched, exact, idempotent (``INSERT OR IGNORE`` on the unique tuple). Reads
    ``dictionary_entries.ref_targets`` (a JSON array of OSIS refs), resolves each to a
    verse, and inserts a ``match_method='ref_target'`` link.

    Args:
        db_path: Path to the ABBA SQLite database.
        confidence: Link confidence (default 1.0 — the dictionary itself cites the verse).

    Returns:
        Stats dict: ``entries`` (with ref_targets), ``refs`` (parsed), ``skipped``
        (unparseable/too-broad), ``links`` (rows inserted this run).
    """
    create_link_table(db_path)
    stats = {"entries": 0, "refs": 0, "skipped": 0, "links": 0}

    insert_sql = (
        "INSERT OR IGNORE INTO verse_dictionary_links "
        "(book_id, chapter, verse, entry_id, match_method, confidence) VALUES (?, ?, ?, ?, 'ref_target', ?)"
    )

    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        rows = conn.execute(
            "SELECT entry_id, ref_targets FROM dictionary_entries WHERE ref_targets IS NOT NULL"
        ).fetchall()

        batch: list[tuple[int, int, int, int, float]] = []
        for entry_id, ref_targets_json in rows:
            try:
                refs = json.loads(ref_targets_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not refs:
                continue
            stats["entries"] += 1
            seen: set[tuple[int, int, int]] = set()
            for ref in refs:
                stats["refs"] += 1
                parsed = parse_osis_ref(ref)
                if parsed is None:
                    stats["skipped"] += 1
                    continue
                if parsed in seen:
                    continue
                seen.add(parsed)
                batch.append((parsed[0], parsed[1], parsed[2], entry_id, confidence))

        if batch:
            cursor = conn.executemany(insert_sql, batch)
            stats["links"] = cursor.rowcount
            conn.commit()

    logger.info("Dictionary linking complete: %s", stats)
    return stats
