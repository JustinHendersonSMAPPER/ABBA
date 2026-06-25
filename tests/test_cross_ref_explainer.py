"""Tests for abba.semantic.cross_ref_explainer.

All tests are self-contained (tmp_path fixture) and never make real network
calls: _ollama_generate is patched throughout.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from abba.api.constants import DEFAULT_TRANSLATION_ID
from abba.database.sqlite_manager import SQLiteManager
from abba.semantic.cross_ref_explainer import (
    CONFIDENCE_THRESHOLD,
    build_prompt,
    compute_shared_strongs,
    generate_explanations,
    is_meaningful_anchor,
    score_confidence,
)

# ---------------------------------------------------------------------------
# Helper: make_test_db
# ---------------------------------------------------------------------------

_JOHN_TEXT = "For God so loved the world that He gave His one and only Son."
_GEN_TEXT = "In the beginning God created the heavens and the earth."
_SHARED_STRONGS = "H3588"  # a common function word present in both test verses


def make_test_db(tmp_path: Path) -> str:
    """Create a minimal test database with all required tables and seed data.

    Tables created via SQLiteManager.initialize_database() (full schema + migrations).
    Additional seed rows are inserted directly.

    Args:
        tmp_path: pytest tmp_path directory.

    Returns:
        Absolute path to the new SQLite file as a string.
    """
    db_path = str(tmp_path / "test.db")
    db = SQLiteManager(db_path)
    db.initialize_database()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")  # relax FK for minimal test data

        # translations row so FK on verses is satisfied
        conn.execute(
            "INSERT OR IGNORE INTO translations (id, name, language) VALUES (?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, "Berean Standard Bible", "en"),
        )

        # books rows for John (43) and Genesis (1)
        conn.execute(
            "INSERT OR IGNORE INTO books (translation_id, book_id, name, book_order, testament) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 43, "John", 43, "new"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO books (translation_id, book_id, name, book_order, testament) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 1, "Genesis", 1, "old"),
        )

        # verses: John 3:16 (book_id=43) and Genesis 1:1 (book_id=1)
        conn.execute(
            "INSERT OR IGNORE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 43, 3, 16, _JOHN_TEXT),
        )
        conn.execute(
            "INSERT OR IGNORE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 1, 1, 1, _GEN_TEXT),
        )

        # stepbible_verses: seed a shared Strong's number in both verses
        # John 3:16 → book="Jhn"
        conn.execute(
            "INSERT OR IGNORE INTO stepbible_verses "
            "(source_file, book, chapter, verse, word_number, language, lexical_strongs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("tagnt_mat_jhn.txt", "Jhn", 3, 16, 1, "greek", _SHARED_STRONGS),
        )
        # Genesis 1:1 → book="Gen"
        conn.execute(
            "INSERT OR IGNORE INTO stepbible_verses "
            "(source_file, book, chapter, verse, word_number, language, lexical_strongs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("tahot_gen_deu.txt", "Gen", 1, 1, 1, "hebrew", _SHARED_STRONGS),
        )

        # cross_reference_candidates: John 3:16 → Genesis 1:1
        conn.execute(
            "INSERT OR IGNORE INTO cross_reference_candidates "
            "(source_book_id, source_chapter, source_verse, "
            "target_book_id, target_chapter, target_verse, "
            "anchor_phrase, source_dataset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (43, 3, 16, 1, 1, 1, "gave", "TSK"),
        )

        conn.commit()

    return db_path


# ---------------------------------------------------------------------------
# Unit tests: score_confidence
# ---------------------------------------------------------------------------


def test_score_confidence_anchor_no_strongs() -> None:
    """Anchor phrase present, no shared Strong's → base 0.7."""
    result = score_confidence("gave", [])
    assert result == pytest.approx(0.7)


def test_score_confidence_anchor_three_strongs() -> None:
    """Anchor phrase present + 3 shared Strong's → 0.7 + 0.3 = 1.0."""
    result = score_confidence("gave", ["H1234", "H5678", "H9012"])
    assert result == pytest.approx(1.0)


def test_score_confidence_no_anchor() -> None:
    """No anchor phrase → base 0.3."""
    result = score_confidence("", [])
    assert result == pytest.approx(0.3)


def test_score_confidence_no_anchor_none() -> None:
    """None anchor phrase → base 0.3."""
    result = score_confidence(None, [])
    assert result == pytest.approx(0.3)


def test_score_confidence_bonus_capped() -> None:
    """Bonus capped at 0.3 even with many shared Strong's numbers."""
    result = score_confidence("", ["H1", "H2", "H3", "H4", "H5"])
    assert result == pytest.approx(0.3 + 0.3)


def test_score_confidence_stopword_anchor_treated_as_no_anchor() -> None:
    """A function-word-only anchor ('that') gets the no-anchor base, not 0.7."""
    assert score_confidence("that", []) == pytest.approx(0.3)
    assert score_confidence("I will", []) == pytest.approx(0.3)
    # but it can still be promoted if backed by shared Strong's
    assert score_confidence("that", ["H1", "H2", "H3"]) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Unit tests: is_meaningful_anchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor,expected",
    [
        ("gave", True),
        ("the Lord", True),
        ("pure gold", True),
        ("Manasseh", True),
        (None, False),
        ("", False),
        ("   ", False),
        ("that", False),
        ("they", False),
        ("I will", False),
        ("for it is", False),
        ("thou shalt", False),
        ("See on Matt 4:1", False),
        ("cf. Genesis 1:1", False),
        ("compare John 1", False),
    ],
)
def test_is_meaningful_anchor(anchor: Any, expected: bool) -> None:
    assert is_meaningful_anchor(anchor) is expected


# ---------------------------------------------------------------------------
# Unit tests: compute_shared_strongs
# ---------------------------------------------------------------------------


def test_compute_shared_strongs(tmp_path: Path) -> None:
    """Shared Strong's seeded in both verses should appear in the result."""
    db_path = make_test_db(tmp_path)
    db = SQLiteManager(db_path)
    # John 3:16 (book_id=43, Jhn) and Genesis 1:1 (book_id=1, Gen)
    shared = compute_shared_strongs(db, 43, 3, 16, 1, 1, 1)
    assert _SHARED_STRONGS in shared


def test_compute_shared_strongs_no_overlap(tmp_path: Path) -> None:
    """Two verses with no lexical_strongs in common return an empty list."""
    db_path = make_test_db(tmp_path)
    db = SQLiteManager(db_path)
    # John 3:16 vs itself but use an unknown book_id that has no rows
    shared = compute_shared_strongs(db, 43, 3, 16, 99, 1, 1)
    assert shared == []


def test_compute_shared_strongs_unknown_book_id(tmp_path: Path) -> None:
    """Unknown book_id (not in BOOK_ID_TO_STEP_CODE) returns empty list immediately."""
    db_path = make_test_db(tmp_path)
    db = SQLiteManager(db_path)
    shared = compute_shared_strongs(db, 999, 1, 1, 1, 1, 1)
    assert shared == []


# ---------------------------------------------------------------------------
# Unit tests: build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_with_anchor() -> None:
    """Anchor phrase is quoted in the shared_idea."""
    prompt = build_prompt("John 3:16", "For God so loved...", "Gen 1:1", "In the beginning...", "gave")
    assert 'the word/idea "gave"' in prompt
    assert "John 3:16" in prompt
    assert "Gen 1:1" in prompt


def test_build_prompt_no_anchor() -> None:
    """No anchor → 'a related theme'."""
    prompt = build_prompt("John 3:16", "text_a", "Gen 1:1", "text_b", None)
    assert "a related theme" in prompt


def test_build_prompt_stopword_anchor_uses_generic_theme() -> None:
    """A junk/stopword anchor is not quoted; falls back to 'a related theme'."""
    prompt = build_prompt("A 1:1", "ta", "B 2:2", "tb", "See on Matt 4:1")
    assert "a related theme" in prompt
    assert "See on" not in prompt


def test_build_prompt_requests_english() -> None:
    """The prompt explicitly asks for an English answer (anti-drift)."""
    prompt = build_prompt("A 1:1", "ta", "B 2:2", "tb", "gave")
    assert "English" in prompt


# ---------------------------------------------------------------------------
# Unit tests: _generate_english (CJK drift handling)
# ---------------------------------------------------------------------------


def test_generate_english_retries_then_succeeds() -> None:
    """A first CJK-contaminated response is rejected; a clean retry is returned."""
    from abba.semantic import cross_ref_explainer as eng

    with patch.object(eng, "_ollama_generate", side_effect=["这是中文回答", "A clean English answer."]):
        out = eng._generate_english("p", "m", "u", attempts=3)
    assert out == "A clean English answer."


def test_generate_english_gives_up_on_persistent_cjk() -> None:
    """If every attempt contains CJK, return None (candidate is deferred)."""
    from abba.semantic import cross_ref_explainer as eng

    with patch.object(eng, "_ollama_generate", return_value="始终是中文"):
        out = eng._generate_english("p", "m", "u", attempts=3)
    assert out is None


# ---------------------------------------------------------------------------
# Integration tests: generate_explanations
# ---------------------------------------------------------------------------

_MOCK_EXPLANATION = "These passages share the concept of giving."


def test_generate_explanations_promotes_row(tmp_path: Path) -> None:
    """Happy path: one candidate with anchor and shared Strong's → promoted."""
    db_path = make_test_db(tmp_path)

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value=_MOCK_EXPLANATION):
        stats = generate_explanations(
            db_path,
            model="test-model",
            url="http://mock-ollama",
            threshold=0.6,
        )

    assert stats["promoted"] == 1
    assert stats["processed"] == 1
    assert stats["skipped_low_conf"] == 0
    assert stats["skipped_no_text"] == 0
    assert stats["skipped_existing"] == 0

    # Verify the cross_reference row was inserted with the explanation
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT notes, confidence FROM cross_references "
            "WHERE source_book_id=43 AND source_chapter=3 AND source_verse=16 "
            "AND target_book_id=1 AND target_chapter=1 AND target_verse=1"
        ).fetchone()
    assert row is not None
    assert row[0] == _MOCK_EXPLANATION
    assert row[1] >= 0.6

    # Verify provenance record exists
    with sqlite3.connect(db_path) as conn:
        prov_row = conn.execute(
            "SELECT entity_type, trust_tier FROM provenance WHERE entity_type='cross_reference'"
        ).fetchone()
    assert prov_row is not None
    assert prov_row[1] == "B"  # TrustTier.GENERATED


def test_generate_explanations_idempotent(tmp_path: Path) -> None:
    """Running the engine twice should not duplicate rows."""
    db_path = make_test_db(tmp_path)

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value=_MOCK_EXPLANATION):
        stats1 = generate_explanations(db_path, model="m", url="u", threshold=0.6)
    assert stats1["promoted"] == 1

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value=_MOCK_EXPLANATION):
        stats2 = generate_explanations(db_path, model="m", url="u", threshold=0.6)
    assert stats2["promoted"] == 0
    assert stats2["skipped_existing"] == 1


def test_generate_explanations_low_confidence_skipped(tmp_path: Path) -> None:
    """Candidate with no anchor and no shared Strong's (conf=0.3) is skipped at threshold 0.6."""
    db_path = make_test_db(tmp_path)

    # Add a second candidate: John 3:16 → Genesis 1:2, no anchor, no stepbible rows for Gen 1:2
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cross_reference_candidates "
            "(source_book_id, source_chapter, source_verse, "
            "target_book_id, target_chapter, target_verse, "
            "anchor_phrase, source_dataset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (43, 3, 16, 1, 1, 2, None, "TSK"),
        )
        # verses row for Gen 1:2 so it doesn't hit skipped_no_text
        conn.execute(
            "INSERT OR IGNORE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_TRANSLATION_ID, 1, 1, 2, "Now the earth was formless and empty."),
        )
        conn.commit()

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value=_MOCK_EXPLANATION):
        stats = generate_explanations(db_path, model="m", url="u", threshold=0.6)

    # First candidate (anchor="gave", shared_strongs present) → promoted
    # Second candidate (no anchor, no shared_strongs) → 0.3 < 0.6 → skipped_low_conf
    assert stats["promoted"] == 1
    assert stats["skipped_low_conf"] == 1


def test_generate_explanations_skips_missing_text(tmp_path: Path) -> None:
    """Candidate whose target verse has no text row → skipped_no_text."""
    db_path = make_test_db(tmp_path)

    # Add a candidate pointing to a verse that has no row in verses
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cross_reference_candidates "
            "(source_book_id, source_chapter, source_verse, "
            "target_book_id, target_chapter, target_verse, "
            "anchor_phrase, source_dataset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (43, 3, 16, 2, 1, 1, "anchor", "TSK"),  # Exodus 1:1 has no verse row
        )
        conn.commit()

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value=_MOCK_EXPLANATION):
        stats = generate_explanations(db_path, model="m", url="u", threshold=0.6)

    assert stats["skipped_no_text"] == 1
    assert stats["promoted"] == 1  # the original John 3:16 → Gen 1:1 still promoted


def test_generate_explanations_drops_cjk_explanation(tmp_path: Path) -> None:
    """A candidate whose explanation is always Chinese is dropped, not promoted."""
    db_path = make_test_db(tmp_path)

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value="全部都是中文的解释"):
        stats = generate_explanations(db_path, model="m", url="u", threshold=0.6)

    assert stats["promoted"] == 0
    assert stats["skipped_no_text"] == 1  # explain_candidate returned None (no clean English)


def test_generate_explanations_filter_by_verse(tmp_path: Path) -> None:
    """source_book_id / source_chapter / source_verse filters work."""
    db_path = make_test_db(tmp_path)

    with patch("abba.semantic.cross_ref_explainer._ollama_generate", return_value=_MOCK_EXPLANATION):
        stats = generate_explanations(
            db_path,
            model="m",
            url="u",
            threshold=0.6,
            source_book_id=43,
            source_chapter=3,
            source_verse=16,
        )

    assert stats["processed"] == 1
    assert stats["promoted"] == 1


# ---------------------------------------------------------------------------
# Integration test: CrossRef model shape via _get_cross_refs
# ---------------------------------------------------------------------------


def test_get_cross_refs_shape(tmp_path: Path) -> None:
    """_get_cross_refs returns CrossRef objects with the new extended fields."""
    from abba.api.models import CrossRef
    from abba.api.routes import _get_cross_refs, configure_db

    db_path = make_test_db(tmp_path)
    db = SQLiteManager(db_path)

    # Insert a cross_reference row directly (bypassing the engine)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cross_references "
            "(source_book_id, source_chapter, source_verse, "
            "target_book_id, target_chapter, target_verse, "
            "ref_type, confidence, source_dataset, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (43, 3, 16, 1, 1, 1, "TSK", 0.75, "TSK+ollama", "Explanation text here."),
        )
        conn.commit()

    # Inject the test db into the routes module
    configure_db(db)
    refs = _get_cross_refs(43, 3, 16)

    assert len(refs) == 1
    ref = refs[0]
    assert isinstance(ref, CrossRef)
    assert ref.id is not None
    assert ref.book_id == 1
    assert ref.chapter == 1
    assert ref.verse == 1
    assert ref.book_name == "Genesis"  # from books table
    assert ref.note == "Explanation text here."
    assert ref.label is not None
    assert "Genesis" in ref.label
