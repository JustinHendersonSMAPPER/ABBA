"""Populate the lexical_strongs column in stepbible_verses.

Run once after the STEPBible import (or at any time as a backfill).  Idempotent:
rows whose lexical_strongs already matches the computed key are skipped via the
batch UPDATE.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..strongs import extract_lexical_strongs, normalize_strongs

logger = logging.getLogger(__name__)

_BATCH_SIZE = 5_000


def populate_lexical_strongs(db_path: Union[str, Path]) -> int:
    """Compute and store the normalized lexical Strong's key for every stepbible_verses row.

    For each row the key is::

        key = normalize_strongs(extract_lexical_strongs(strongs_primary, strongs_raw))

    Rows with an empty key get NULL.  The function operates in batches of
    :data:`_BATCH_SIZE` rows and is safe to run multiple times (idempotent).

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Number of rows given a non-empty (non-NULL) key.
    """
    db_path = Path(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        cursor = conn.cursor()

        # Guard: table may not exist yet
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='stepbible_verses'")
        if cursor.fetchone()[0] == 0:
            logger.warning("stepbible_verses table not found; nothing to populate")
            return 0

        # Guard: column may not exist yet (migration not run)
        cursor.execute("PRAGMA table_info(stepbible_verses)")
        columns = {row[1] for row in cursor.fetchall()}
        if "lexical_strongs" not in columns:
            logger.warning("lexical_strongs column not present; run add_stepbible_lexical_strongs_column first")
            return 0

        # Fetch all rows we need to process
        cursor.execute("SELECT id, strongs_primary, strongs_raw FROM stepbible_verses")
        rows = cursor.fetchall()

        non_empty = 0
        updates: List[Tuple[Optional[str], int]] = []

        for row_id, strongs_primary, strongs_raw in rows:
            key = normalize_strongs(extract_lexical_strongs(strongs_primary, strongs_raw))
            if key:
                non_empty += 1
                updates.append((key, row_id))
            else:
                updates.append((None, row_id))

        # Write in batches to keep memory and WAL manageable
        for i in range(0, len(updates), _BATCH_SIZE):
            batch = updates[i : i + _BATCH_SIZE]
            conn.executemany(
                "UPDATE stepbible_verses SET lexical_strongs = ? WHERE id = ?",
                batch,
            )
            conn.commit()

        logger.info(
            "populate_lexical_strongs: processed %d rows, %d with non-empty key",
            len(rows),
            non_empty,
        )
        return non_empty
