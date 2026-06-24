"""Rebuild the FTS5 full-text search index for the verses table.

The ``verses_fts`` table is an *external-content* FTS5 table backed by
``verses``.  Bulk inserts into ``verses`` do **not** automatically update
the FTS index, so ``rebuild_search_index`` must be called after a bulk
import to make text search functional.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def rebuild_search_index(db_path: Path) -> int:
    """Rebuild the FTS5 search index for the verses table.

    Runs ``INSERT INTO verses_fts(verses_fts) VALUES('rebuild')`` which
    causes SQLite to repopulate the FTS index from the content table
    (``verses``).  This is idempotent and safe to run multiple times.

    If ``verses_fts`` does not exist (e.g. schema was not yet applied),
    logs a warning and returns 0 rather than raising.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The number of rows in ``verses_fts`` after the rebuild (a rough
        signal that indexing succeeded), or 0 if the table was absent.
    """
    conn: sqlite3.Connection = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode = WAL")

        # Check whether the FTS table exists before attempting the rebuild.
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='verses_fts'")
        if cursor.fetchone() is None:
            logger.warning(
                "rebuild_search_index: verses_fts table not found in %s; skipping rebuild",
                db_path,
            )
            return 0

        # Rebuild the FTS index from the content table.
        conn.execute("INSERT INTO verses_fts(verses_fts) VALUES('rebuild')")
        conn.commit()

        # Return indexed document count as a "did it build" signal.
        row = conn.execute("SELECT COUNT(*) FROM verses_fts").fetchone()
        count: int = row[0] if row else 0
        logger.info("rebuild_search_index: FTS index rebuilt from verses table — %d documents now indexed", count)
        return count
    finally:
        conn.close()
