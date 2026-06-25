"""Easton's Bible Dictionary importer.

Reads the CrossWire SWORD "Easton" dictionary module (public domain, 1897) via
:mod:`abba.sources.easton` and populates the ``dictionary_entries`` table, plus
one authoritative (Tier A) provenance record per entry.

All operations are idempotent: an ``INSERT OR IGNORE`` against the
``UNIQUE(source, headword)`` constraint means a second run inserts zero rows.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from ..sources.easton import DictionaryEntry, iter_easton_entries

logger = logging.getLogger(__name__)

# Stable identifiers recorded on every row + provenance record.
SOURCE_NAME = "Easton's Bible Dictionary (1897)"
SOURCE_LICENSE = "Public Domain"
SOURCE_DETAIL = (
    "CrossWire SWORD module 'Easton' (DistributionLicense=Public Domain, "
    "TextSource=CCEL); M.G. Easton, Illustrated Bible Dictionary, 3rd ed., "
    "Thomas Nelson, 1897. Re-fetch: "
    "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/Easton.zip"
)
PIPELINE_VERSION = "easton-import-v1"

_BATCH_SIZE = 500

# Headword normalization: strip trailing parenthetical disambiguators
# (e.g. "Zuph, Land of" -> "ZUPH LAND OF"), collapse punctuation/whitespace,
# uppercase. Used for high-confidence EXACT-match entity linking (decision D5).
_PAREN = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_headword(headword: str) -> str:
    """Normalize a headword to a canonical exact-match key.

    Uppercases, removes parenthetical asides, replaces punctuation (commas,
    hyphens, apostrophes) with spaces, and collapses whitespace. This is the key
    used for conservative exact-match linking of a verse's proper nouns / key
    terms to dictionary headwords.

    Args:
        headword: Raw headword from the dictionary (e.g. ``"Zuph, Land of"``).

    Returns:
        The normalized key (e.g. ``"ZUPH LAND OF"``). May be empty if the
        headword contained no alphanumerics.
    """
    text = _PAREN.sub(" ", headword)
    text = _NON_ALNUM.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text.upper()


def _record_provenance(conn: sqlite3.Connection, entry: DictionaryEntry) -> None:
    """Upsert a Tier-A provenance record for one dictionary entry.

    Written directly on the importer's connection (not via SQLiteManager) so the
    whole import shares one transaction/connection. The schema mirrors
    :class:`abba.provenance.Provenance`.

    Args:
        conn: Open SQLite connection (provenance table assumed to exist).
        entry: The dictionary entry being recorded.
    """
    entity_id = f"{SOURCE_NAME}:{entry.headword}"
    grounding = {
        "headword": entry.headword,
        "ref_targets": list(entry.ref_targets),
    }
    conn.execute(
        """
        INSERT INTO provenance (
            entity_type, entity_id, source, source_detail, trust_tier,
            trust_rationale, generated_by, grounding_json, confidence, pipeline_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
            source=excluded.source,
            source_detail=excluded.source_detail,
            trust_tier=excluded.trust_tier,
            trust_rationale=excluded.trust_rationale,
            generated_by=excluded.generated_by,
            grounding_json=excluded.grounding_json,
            confidence=excluded.confidence,
            pipeline_version=excluded.pipeline_version
        """,
        (
            "dictionary_entry",
            entity_id,
            SOURCE_NAME,
            SOURCE_DETAIL,
            "A",  # AUTHORITATIVE: verbatim text from a public-domain reference work
            "Verbatim entry from a public-domain (1897) reference work; ingested unaltered.",
            None,  # not AI-generated
            json.dumps(grounding, ensure_ascii=False, sort_keys=True),
            None,  # no confidence: it is a fact, not an inference
            PIPELINE_VERSION,
        ),
    )


def import_easton_entries(db_path: str | Path, zip_path: str | Path) -> int:
    """Import Easton's Bible Dictionary entries into ``dictionary_entries``.

    Ensures the ``dictionary_entries`` table exists (applies its migration if
    needed), then bulk-inserts every entry from the Easton SWORD module with
    ``INSERT OR IGNORE`` (idempotent). A Tier-A provenance record is upserted for
    each entry if the ``provenance`` table is present.

    Args:
        db_path: Path to the ABBA SQLite database.
        zip_path: Path to Easton.zip (CrossWire SWORD rawzip module).

    Returns:
        Number of new ``dictionary_entries`` rows inserted (0 on a repeated run).
    """
    db_path = Path(db_path)
    zip_path = Path(zip_path)

    from .migrations import add_dictionary_entries_table  # noqa: PLC0415

    add_dictionary_entries_table(db_path)

    insert_sql = """
        INSERT OR IGNORE INTO dictionary_entries (
            headword, headword_normalized, article, ref_targets, source, license
        ) VALUES (?, ?, ?, ?, ?, ?)
    """

    total_inserted = 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

        has_provenance = (
            conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='provenance'").fetchone()[0]
            > 0
        )

        batch: list[tuple[str, str, str, str | None, str, str]] = []
        pending: list[DictionaryEntry] = []

        def flush() -> None:
            nonlocal total_inserted, batch, pending
            if not batch:
                return
            cursor = conn.executemany(insert_sql, batch)
            total_inserted += cursor.rowcount
            if has_provenance:
                for entry in pending:
                    _record_provenance(conn, entry)
            conn.commit()
            batch = []
            pending = []

        for entry in iter_easton_entries(zip_path):
            ref_targets_json = json.dumps(list(entry.ref_targets), ensure_ascii=False) if entry.ref_targets else None
            batch.append(
                (
                    entry.headword,
                    normalize_headword(entry.headword),
                    entry.article,
                    ref_targets_json,
                    SOURCE_NAME,
                    SOURCE_LICENSE,
                )
            )
            pending.append(entry)
            if len(batch) >= _BATCH_SIZE:
                flush()

        flush()

    logger.info("Easton import complete: %d new rows inserted into dictionary_entries", total_inserted)
    return total_inserted
