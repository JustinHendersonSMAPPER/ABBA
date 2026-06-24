"""TSK cross-reference importer.

Reads the Treasury of Scripture Knowledge SWORD module and populates
the ``cross_reference_candidates`` staging table.  All operations are
idempotent: running the importer twice produces no duplicate rows thanks
to the UNIQUE constraint + INSERT OR IGNORE strategy.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ..sources.tsk import iter_tsk_cross_references

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000


def import_tsk_candidates(db_path: str | Path, zip_path: str | Path) -> int:
    """Import TSK cross-references into the cross_reference_candidates table.

    Ensures the staging table exists (runs the migration if needed), then
    bulk-inserts every cross-reference from the TSK module using
    INSERT OR IGNORE so repeated runs add zero rows.

    Args:
        db_path: Path to the ABBA SQLite database.
        zip_path: Path to TSK.zip (CrossWire SWORD zCom module).

    Returns:
        Number of new rows inserted (0 on a repeated run).
    """
    db_path = Path(db_path)
    zip_path = Path(zip_path)

    # Ensure the staging table exists
    from .migrations import add_cross_reference_candidates_table  # noqa: PLC0415

    add_cross_reference_candidates_table(db_path)

    insert_sql = """
        INSERT OR IGNORE INTO cross_reference_candidates (
            source_book_id, source_chapter, source_verse,
            target_book_id, target_chapter, target_verse,
            anchor_phrase, source_dataset
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    total_inserted = 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

        batch: list[tuple[int, int, int, int, int, int, str, str]] = []

        for rec in iter_tsk_cross_references(zip_path):
            batch.append(
                (
                    rec["source_book_id"],
                    rec["source_chapter"],
                    rec["source_verse"],
                    rec["target_book_id"],
                    rec["target_chapter"],
                    rec["target_verse"],
                    rec["anchor_phrase"],
                    "TSK",
                )
            )
            if len(batch) >= _BATCH_SIZE:
                cursor = conn.executemany(insert_sql, batch)
                total_inserted += cursor.rowcount
                conn.commit()
                batch = []

        if batch:
            cursor = conn.executemany(insert_sql, batch)
            total_inserted += cursor.rowcount
            conn.commit()

    logger.info("TSK import complete: %d new rows inserted into cross_reference_candidates", total_inserted)
    return total_inserted
