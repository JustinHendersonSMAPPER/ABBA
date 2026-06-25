"""Tests for the read-only data-transparency endpoint (GET /stats -> DataStats)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from abba.api.constants import DEFAULT_TRANSLATION_ID
from abba.api.models import DataStats
from abba.api.routes import configure_db, get_data_stats
from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier


def _seed_db(tmp_path: Path) -> SQLiteManager:
    """Build a temp DB with 1 translation, 2 verses, 2 candidates, and 2 cross-refs.

    One cross-reference is explained (non-empty notes) + has a provenance record; the
    other is deliberately unexplained (empty notes) to exercise the no-dead-data signal.
    """
    db_path = str(tmp_path / "stats.db")
    db = SQLiteManager(db_path)
    db.initialize_database()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT OR IGNORE INTO translations (id, name, language) VALUES (?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, "Berean Standard Bible", "en"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 43, 3, 16, "For God so loved the world..."),
        )
        conn.execute(
            "INSERT OR IGNORE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 1, 1, 1, "In the beginning..."),
        )
        for tb, tc, tv in [(1, 1, 1), (1, 22, 12)]:
            conn.execute(
                "INSERT OR IGNORE INTO cross_reference_candidates "
                "(source_book_id, source_chapter, source_verse, target_book_id, target_chapter, target_verse, "
                "anchor_phrase, source_dataset) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (43, 3, 16, tb, tc, tv, "gave", "TSK"),
            )
        # explained cross-reference (has notes)
        conn.execute(
            "INSERT INTO cross_references "
            "(source_book_id, source_chapter, source_verse, target_book_id, target_chapter, target_verse, "
            "ref_type, confidence, source_dataset, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (43, 3, 16, 1, 1, 1, "TSK", 0.8, "TSK+ollama", "An explanation of the link."),
        )
        # unexplained cross-reference (empty notes) — should count toward unexplained
        conn.execute(
            "INSERT INTO cross_references "
            "(source_book_id, source_chapter, source_verse, target_book_id, target_chapter, target_verse, "
            "ref_type, confidence, source_dataset, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (43, 3, 16, 1, 22, 12, "TSK", 0.7, "TSK+ollama", ""),
        )
        conn.commit()
        explained_ref_id = conn.execute(
            "SELECT ref_id FROM cross_references WHERE target_chapter = 1 AND target_verse = 1"
        ).fetchone()[0]

    ProvenanceStore(db).record(
        Provenance(
            entity_type="cross_reference",
            entity_id=str(explained_ref_id),
            source="ollama",
            source_detail="model=test",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="grounded",
            generated_by="test-model",
            grounding={"anchor_phrase": "gave"},
            confidence=0.8,
            pipeline_version="0.1.0",
        )
    )
    return db


def test_stats_counts_and_no_dead_data_signal(tmp_path: Path) -> None:
    """The endpoint returns accurate counts incl. the race-free unexplained signal."""
    db = _seed_db(tmp_path)
    configure_db(db)

    stats = asyncio.run(get_data_stats())

    assert isinstance(stats, DataStats)
    assert stats.translations == 1
    assert stats.verses == 2
    assert stats.cross_reference_candidates == 2
    assert stats.cross_references == 2
    assert stats.cross_references_explained == 1
    assert stats.cross_references_unexplained == 1  # the empty-notes row
    assert stats.explained_coverage_pct == 50.0  # 1 explained / 2 candidates
    assert stats.provenance_records == 1
    assert stats.cross_references_by_tier == {"B": 1}
    assert stats.cross_references_by_source == {"TSK+ollama": 2}
    assert stats.avg_confidence == 0.75  # avg of 0.8 and 0.7
    assert stats.dictionary_entries == 0  # table not created


def test_stats_empty_db(tmp_path: Path) -> None:
    """A DB with no cross-references reports zeros, not errors (coverage = 0)."""
    db_path = str(tmp_path / "empty.db")
    db = SQLiteManager(db_path)
    db.initialize_database()
    configure_db(db)

    stats = asyncio.run(get_data_stats())

    assert stats.cross_references == 0
    assert stats.cross_references_unexplained == 0
    assert stats.explained_coverage_pct == 0.0
    assert stats.avg_confidence is None
