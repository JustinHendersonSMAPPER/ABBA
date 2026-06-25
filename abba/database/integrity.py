"""Read-only data-integrity checks for ABBA's "no dead data / fully auditable" invariant.

ABBA's trust model requires that every cross-reference surfaced to a reader is both
*explained* and *auditable*:

* a non-empty explanation in ``cross_references.notes`` (no dead data), and
* a matching provenance record (``provenance.entity_type = 'cross_reference'`` with
  ``entity_id`` equal to the ``cross_references.ref_id`` rendered as TEXT).

This module provides **read-only** detectors for violations of that invariant. Every
function either accepts an already-open :class:`sqlite3.Connection` (and never writes
to it) or, when given a database path, opens its own connection in SQLite read-only
URI mode (``mode=ro``) so it is safe to run against a live database that another
process is concurrently writing.

Detection is done in SQL (anti-joins / ``NOT EXISTS``) rather than Python row loops,
so the checks stay efficient on a full corpus.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union

# Cap on how many offending ids we collect per check (the full counts are always exact;
# the samples are just enough to investigate without dumping the whole corpus).
SAMPLE_LIMIT = 20

# The provenance.entity_type value used for cross-reference attribution records.
CROSS_REFERENCE_ENTITY_TYPE = "cross_reference"


@dataclass
class CheckResult:
    """Result of a single integrity check.

    Attributes:
        name: Stable identifier for the check (e.g. ``"missing_notes"``).
        description: Human-readable description of what the check detects.
        count: Total number of offending rows (exact, not capped).
        sample_ids: Up to :data:`SAMPLE_LIMIT` offending ids, as strings, for
            investigation. ``cross_references`` checks report ``ref_id`` values;
            the orphan-provenance check reports offending ``entity_id`` values.
    """

    name: str
    description: str
    count: int
    sample_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` if the check found no violations."""
        return self.count == 0

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict view (JSON-friendly) of this result."""
        return {
            "name": self.name,
            "description": self.description,
            "count": self.count,
            "sample_ids": list(self.sample_ids),
            "ok": self.ok,
        }


# A db argument is either a path (str/Path) or an already-open connection.
DbArg = Union[str, Path, sqlite3.Connection]


@contextmanager
def _readonly_connection(db: DbArg) -> Iterator[sqlite3.Connection]:
    """Yield a connection for ``db``, opening read-only if a path was given.

    If ``db`` is already an open :class:`sqlite3.Connection`, it is yielded as-is and
    left open (the caller owns its lifecycle); this module never writes through it. If
    ``db`` is a path, a new connection is opened in SQLite read-only URI mode
    (``file:<abspath>?mode=ro``) and closed when the context exits.

    Args:
        db: A database path or an already-open connection.

    Yields:
        A usable SQLite connection.
    """
    if isinstance(db, sqlite3.Connection):
        yield db
        return

    abspath = Path(db).resolve()
    uri = f"file:{abspath.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        yield conn
    finally:
        conn.close()


def _scalar_count(conn: sqlite3.Connection, sql: str) -> int:
    """Run a ``SELECT COUNT(*)`` style query and return the integer result."""
    with closing(conn.cursor()) as cur:
        cur.execute(sql)
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _sample_ids(conn: sqlite3.Connection, sql: str) -> list[str]:
    """Run a query returning a single id column and collect the ids as strings."""
    with closing(conn.cursor()) as cur:
        cur.execute(sql)
        return [str(row[0]) for row in cur.fetchall()]


# --- SQL fragments -------------------------------------------------------------------

# cross_references whose notes (the explanation) is NULL or blank/whitespace-only.
_MISSING_NOTES_WHERE = "cr.notes IS NULL OR TRIM(cr.notes) = ''"

# cross_references with no matching provenance row (anti-join via NOT EXISTS).
_MISSING_PROVENANCE_WHERE = (
    "NOT EXISTS ("
    "  SELECT 1 FROM provenance p"
    f"  WHERE p.entity_type = '{CROSS_REFERENCE_ENTITY_TYPE}'"
    "    AND p.entity_id = CAST(cr.ref_id AS TEXT)"
    ")"
)

# provenance cross_reference rows whose entity_id does not match any cross_references.ref_id.
_ORPHAN_PROVENANCE_WHERE = (
    f"p.entity_type = '{CROSS_REFERENCE_ENTITY_TYPE}' "
    "AND NOT EXISTS ("
    "  SELECT 1 FROM cross_references cr"
    "  WHERE CAST(cr.ref_id AS TEXT) = p.entity_id"
    ")"
)


def check_cross_references_missing_notes(db: DbArg) -> CheckResult:
    """Find ``cross_references`` rows whose explanation (``notes``) is missing.

    A missing explanation is NULL or blank/whitespace-only. Such a cross-reference
    would be "dead data" if surfaced, so it violates the no-dead-data invariant.

    Args:
        db: A database path (opened read-only) or an open connection.

    Returns:
        A :class:`CheckResult` with the total count and up to :data:`SAMPLE_LIMIT`
        offending ``ref_id`` values.
    """
    with _readonly_connection(db) as conn:
        count = _scalar_count(
            conn,
            f"SELECT COUNT(*) FROM cross_references cr WHERE {_MISSING_NOTES_WHERE}",
        )
        samples = _sample_ids(
            conn,
            f"SELECT cr.ref_id FROM cross_references cr WHERE {_MISSING_NOTES_WHERE} "
            f"ORDER BY cr.ref_id LIMIT {SAMPLE_LIMIT}",
        )
    return CheckResult(
        name="missing_notes",
        description="cross_references rows with NULL/empty notes (explanation missing)",
        count=count,
        sample_ids=samples,
    )


def check_cross_references_missing_provenance(db: DbArg) -> CheckResult:
    """Find ``cross_references`` rows with no matching provenance record.

    Every cross-reference must be auditable via a provenance row keyed by
    (``entity_type='cross_reference'``, ``entity_id = ref_id`` as TEXT). Rows with no
    such provenance record are not auditable and violate the invariant.

    Args:
        db: A database path (opened read-only) or an open connection.

    Returns:
        A :class:`CheckResult` with the total count and up to :data:`SAMPLE_LIMIT`
        offending ``ref_id`` values.
    """
    with _readonly_connection(db) as conn:
        count = _scalar_count(
            conn,
            f"SELECT COUNT(*) FROM cross_references cr WHERE {_MISSING_PROVENANCE_WHERE}",
        )
        samples = _sample_ids(
            conn,
            f"SELECT cr.ref_id FROM cross_references cr WHERE {_MISSING_PROVENANCE_WHERE} "
            f"ORDER BY cr.ref_id LIMIT {SAMPLE_LIMIT}",
        )
    return CheckResult(
        name="missing_provenance",
        description="cross_references rows with no matching provenance record",
        count=count,
        sample_ids=samples,
    )


def check_orphan_cross_reference_provenance(db: DbArg) -> CheckResult:
    """Find orphan cross-reference provenance rows.

    These are ``provenance`` rows with ``entity_type='cross_reference'`` whose
    ``entity_id`` does not correspond to any existing ``cross_references.ref_id``.
    They point at a cross-reference that no longer exists (dangling audit trail).

    Args:
        db: A database path (opened read-only) or an open connection.

    Returns:
        A :class:`CheckResult` with the total count and up to :data:`SAMPLE_LIMIT`
        offending ``entity_id`` values.
    """
    with _readonly_connection(db) as conn:
        count = _scalar_count(
            conn,
            f"SELECT COUNT(*) FROM provenance p WHERE {_ORPHAN_PROVENANCE_WHERE}",
        )
        samples = _sample_ids(
            conn,
            f"SELECT p.entity_id FROM provenance p WHERE {_ORPHAN_PROVENANCE_WHERE} "
            f"ORDER BY p.entity_id LIMIT {SAMPLE_LIMIT}",
        )
    return CheckResult(
        name="orphan_provenance",
        description=(
            "provenance rows (entity_type='cross_reference') whose entity_id has no matching cross_references.ref_id"
        ),
        count=count,
        sample_ids=samples,
    )


def _count_cross_references(conn: sqlite3.Connection) -> int:
    """Return the total number of ``cross_references`` rows."""
    return _scalar_count(conn, "SELECT COUNT(*) FROM cross_references")


def _count_cross_reference_provenance(conn: sqlite3.Connection) -> int:
    """Return the number of provenance rows with entity_type='cross_reference'."""
    return _scalar_count(
        conn,
        f"SELECT COUNT(*) FROM provenance WHERE entity_type = '{CROSS_REFERENCE_ENTITY_TYPE}'",
    )


def summarize_integrity(db: DbArg) -> dict[str, object]:
    """Run every cross-reference integrity check and aggregate the results.

    Opens a single read-only connection (when given a path) and runs all three
    detectors plus totals through it, so the whole summary is one consistent snapshot.

    Args:
        db: A database path (opened read-only) or an open connection.

    Returns:
        A dict with:

        * ``total_cross_references``: total ``cross_references`` rows.
        * ``total_cross_reference_provenance``: provenance rows for cross-references.
        * ``checks``: mapping of check name to its :meth:`CheckResult.to_dict`.
        * ``ok``: ``True`` if every check passed (no violations).
    """
    with _readonly_connection(db) as conn:
        results = [
            check_cross_references_missing_notes(conn),
            check_cross_references_missing_provenance(conn),
            check_orphan_cross_reference_provenance(conn),
        ]
        summary: dict[str, object] = {
            "total_cross_references": _count_cross_references(conn),
            "total_cross_reference_provenance": _count_cross_reference_provenance(conn),
            "checks": {r.name: r.to_dict() for r in results},
            "ok": all(r.ok for r in results),
        }
    return summary
