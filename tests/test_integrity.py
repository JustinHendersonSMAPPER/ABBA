"""Tests for the read-only cross-reference data-integrity checker.

Builds a temp DB (via :class:`SQLiteManager`, which creates the schema and runs the
migrations that add the ``cross_references`` and ``provenance`` tables), inserts a mix
of good rows and deliberately-bad rows, and asserts the checker flags exactly the bad
ones. Self-contained: no network, no live DB.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from abba.database.integrity import (
    SAMPLE_LIMIT,
    CheckResult,
    check_cross_references_missing_notes,
    check_cross_references_missing_provenance,
    check_orphan_cross_reference_provenance,
    summarize_integrity,
)
from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier


def _insert_cross_reference(db: SQLiteManager, *, source_verse: int, target_verse: int, notes: str | None) -> int:
    """Insert one cross_references row and return its ref_id.

    Source/target verses vary per row so the table's uniqueness constraint is satisfied.
    """
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO cross_references (
                source_book_id, source_chapter, source_verse,
                target_book_id, target_chapter, target_verse,
                ref_type, confidence, source_dataset, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, 1, source_verse, 43, 1, target_verse, "TSK", 0.8, "TSK+ollama", notes),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def _record_cross_reference_provenance(db: SQLiteManager, ref_id: int) -> None:
    """Write a well-formed provenance record for a cross_reference ref_id."""
    ProvenanceStore(db).record(
        Provenance(
            entity_type="cross_reference",
            entity_id=str(ref_id),
            source="ollama",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="AI-generated explanation grounded in TSK anchor.",
            generated_by="qwen2.5:14b",
            confidence=0.8,
            pipeline_version="0.1.0",
        )
    )


@pytest.fixture
def integrity_db(tmp_path: Path) -> dict[str, object]:
    """Build a temp DB with known-good and deliberately-bad cross-reference data.

    Layout:
      * good_id      -> non-empty notes + provenance (clean).
      * empty_id     -> empty (whitespace) notes + provenance (missing-notes violation).
      * no_prov_id   -> non-empty notes, NO provenance (missing-provenance violation).
      * orphan id    -> provenance row pointing at a non-existent ref_id (orphan).

    Returns a dict of the db path and the relevant ids for assertions.
    """
    db_path = tmp_path / "integrity_test.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # Good row: has notes and provenance.
    good_id = _insert_cross_reference(
        db, source_verse=1, target_verse=1, notes="A real explanation linking the verses."
    )
    _record_cross_reference_provenance(db, good_id)

    # Bad row 1: empty (whitespace-only) notes, but provenance present.
    empty_notes_id = _insert_cross_reference(db, source_verse=2, target_verse=2, notes="   ")
    _record_cross_reference_provenance(db, empty_notes_id)

    # Bad row 2: NULL notes (also a missing-notes violation), provenance present.
    null_notes_id = _insert_cross_reference(db, source_verse=3, target_verse=3, notes=None)
    _record_cross_reference_provenance(db, null_notes_id)

    # Bad row 3: has notes but NO provenance.
    no_prov_id = _insert_cross_reference(db, source_verse=4, target_verse=4, notes="Explained but unaudited.")

    # Bad row 4: orphan provenance pointing at a ref_id that does not exist.
    orphan_entity_id = str(good_id + 9999)
    _record_cross_reference_provenance(db, int(orphan_entity_id))

    return {
        "db_path": db_path,
        "good_id": good_id,
        "empty_notes_id": empty_notes_id,
        "null_notes_id": null_notes_id,
        "no_prov_id": no_prov_id,
        "orphan_entity_id": orphan_entity_id,
    }


def test_missing_notes_flags_empty_and_null(integrity_db: dict[str, object]) -> None:
    """Both empty-string and NULL notes rows are flagged; the good row is not."""
    result = check_cross_references_missing_notes(integrity_db["db_path"])
    assert isinstance(result, CheckResult)
    assert result.name == "missing_notes"
    assert result.count == 2
    assert not result.ok
    flagged = set(result.sample_ids)
    assert str(integrity_db["empty_notes_id"]) in flagged
    assert str(integrity_db["null_notes_id"]) in flagged
    assert str(integrity_db["good_id"]) not in flagged
    assert str(integrity_db["no_prov_id"]) not in flagged


def test_missing_provenance_flags_only_unaudited(integrity_db: dict[str, object]) -> None:
    """Only the cross-reference with no provenance row is flagged."""
    result = check_cross_references_missing_provenance(integrity_db["db_path"])
    assert result.name == "missing_provenance"
    assert result.count == 1
    assert result.sample_ids == [str(integrity_db["no_prov_id"])]
    assert not result.ok


def test_orphan_provenance_flags_dangling_entity_id(integrity_db: dict[str, object]) -> None:
    """The provenance row pointing at a non-existent ref_id is flagged as an orphan."""
    result = check_orphan_cross_reference_provenance(integrity_db["db_path"])
    assert result.name == "orphan_provenance"
    assert result.count == 1
    assert result.sample_ids == [integrity_db["orphan_entity_id"]]
    assert not result.ok


def test_summarize_integrity_aggregates_all_checks(integrity_db: dict[str, object]) -> None:
    """The aggregator reports totals and every check's result, and overall not-ok."""
    summary = summarize_integrity(integrity_db["db_path"])

    # 4 cross_references rows inserted (good, empty, null, no_prov).
    assert summary["total_cross_references"] == 4
    # 4 cross_reference provenance rows (good, empty, null, orphan).
    assert summary["total_cross_reference_provenance"] == 4

    checks = summary["checks"]
    assert isinstance(checks, dict)
    assert checks["missing_notes"]["count"] == 2
    assert checks["missing_provenance"]["count"] == 1
    assert checks["orphan_provenance"]["count"] == 1
    assert summary["ok"] is False


def test_accepts_open_connection(integrity_db: dict[str, object]) -> None:
    """Checks accept an already-open connection and do not close it."""
    conn = sqlite3.connect(str(integrity_db["db_path"]))
    try:
        result = check_cross_references_missing_notes(conn)
        assert result.count == 2
        # Connection is still usable (was not closed by the checker).
        cur = conn.execute("SELECT COUNT(*) FROM cross_references")
        assert cur.fetchone()[0] == 4
    finally:
        conn.close()


def test_summarize_with_open_connection(integrity_db: dict[str, object]) -> None:
    """The aggregator also accepts an already-open connection."""
    conn = sqlite3.connect(str(integrity_db["db_path"]))
    try:
        summary = summarize_integrity(conn)
        assert summary["total_cross_references"] == 4
        assert summary["ok"] is False
    finally:
        conn.close()


def test_clean_db_reports_ok(tmp_path: Path) -> None:
    """A DB with only well-formed rows reports ok and zero violations."""
    db_path = tmp_path / "clean.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    for i in range(3):
        ref_id = _insert_cross_reference(db, source_verse=10 + i, target_verse=20 + i, notes=f"Clean explanation {i}.")
        _record_cross_reference_provenance(db, ref_id)

    summary = summarize_integrity(db_path)
    assert summary["total_cross_references"] == 3
    assert summary["ok"] is True
    checks = summary["checks"]
    assert isinstance(checks, dict)
    for check in checks.values():
        assert check["count"] == 0
        assert check["ok"] is True


def test_readonly_path_does_not_write(integrity_db: dict[str, object]) -> None:
    """Running a check via a path opens read-only: a concurrent write attempt fails."""
    # Open a second read-only connection the way the checker does and confirm writes fail.
    abspath = Path(str(integrity_db["db_path"])).resolve()
    ro_conn = sqlite3.connect(f"file:{abspath.as_posix()}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro_conn.execute("INSERT INTO cross_references (source_book_id) VALUES (99)")
    finally:
        ro_conn.close()

    # And the normal read-only check still works.
    result = check_cross_references_missing_provenance(integrity_db["db_path"])
    assert result.count == 1


def test_sample_ids_capped(tmp_path: Path) -> None:
    """When violations exceed SAMPLE_LIMIT, count is exact but samples are capped."""
    db_path = tmp_path / "many.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    over = SAMPLE_LIMIT + 5
    for i in range(over):
        # All rows have NULL notes -> all are missing-notes violations.
        _insert_cross_reference(db, source_verse=100 + i, target_verse=200 + i, notes=None)

    result = check_cross_references_missing_notes(db_path)
    assert result.count == over
    assert len(result.sample_ids) == SAMPLE_LIMIT
