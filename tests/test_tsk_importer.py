"""Tests for the TSK cross-reference importer.

Covers:
- parse_verse_refs unit tests (accuracy gate for reference grammar)
- build_abbrev_map (basic sanity)
- Integration tests guarded by @pytest.mark.skipif when TSK.zip is absent
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from abba.sources.tsk import (
    _FALLBACK_ABBREV_MAP,
    build_abbrev_map,
    iter_tsk_cross_references,
    open_tsk_bible,
    parse_verse_refs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TSK_ZIP = Path("bible_data/sources/TSK.zip")
TSK_AVAILABLE = TSK_ZIP.exists()

_ABBREV_MAP = dict(_FALLBACK_ABBREV_MAP)  # use fallback for unit tests


# ---------------------------------------------------------------------------
# Unit tests: parse_verse_refs
# ---------------------------------------------------------------------------


class TestParseVerseRefs:
    """Accuracy tests for the reference-notation grammar parser."""

    def test_multi_book_range_and_inheritance(self) -> None:
        """Genesis 1:1 cross-refs: multi-book list with ranges + inheritance.

        Input ThML:  beginning.<br /><scripRef>Pr 8:22-24; 16:4; Mr 13:19;
                     Joh 1:1-3; Heb 1:10; 1Jo 1:1</scripRef>
        Source: Gen (1) 1:1
        Expected targets (in order):
            (20,8,22), (20,8,23), (20,8,24)  ← Pr 8:22-24
            (20,16,4)                          ← 16:4 (inherits Pr)
            (41,13,19)                         ← Mr 13:19
            (43,1,1), (43,1,2), (43,1,3)      ← Joh 1:1-3
            (58,1,10)                          ← Heb 1:10
            (62,1,1)                           ← 1Jo 1:1
        """
        thml = "beginning.<br /><scripRef>Pr 8:22-24; 16:4; Mr 13:19; Joh 1:1-3; Heb 1:10; 1Jo 1:1</scripRef>"
        groups = parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP)
        assert len(groups) == 1
        anchor, targets = groups[0]
        assert anchor == "beginning"

        expected = [
            (20, 8, 22),
            (20, 8, 23),
            (20, 8, 24),
            (20, 16, 4),
            (41, 13, 19),
            (43, 1, 1),
            (43, 1, 2),
            (43, 1, 3),
            (58, 1, 10),
            (62, 1, 1),
        ]
        assert targets == expected

    def test_chapter_inheritance_with_comma_list(self) -> None:
        """John 3:16 cross-refs: chapter inheritance + comma verse list.

        Input: <scripRef>1:14,18; Ge 22:12</scripRef>
        Source: John (43) ch 3
        Expected:
            (43,1,14), (43,1,18)  ← 1:14,18 inherits book=John
            (1,22,12)             ← Ge 22:12
        """
        thml = "gave.<br /><scripRef>1:14,18; Ge 22:12</scripRef>"
        groups = parse_verse_refs(thml, source_book_id=43, source_chapter=3, abbrev_map=_ABBREV_MAP)
        assert len(groups) == 1
        anchor, targets = groups[0]
        assert anchor == "gave"
        assert (43, 1, 14) in targets
        assert (43, 1, 18) in targets
        assert (1, 22, 12) in targets
        # Exactly those three
        assert set(targets) == {(43, 1, 14), (43, 1, 18), (1, 22, 12)}

    def test_bare_verse_and_new_book(self) -> None:
        """John 3:16 'whosoever' group: bare verse inherits book+chapter.

        Input: <scripRef>15; Mt 9:13</scripRef>
        Source: John (43) ch 3
        Expected:
            (43,3,15)  ← bare 15 inherits book=43, chapter=3
            (40,9,13)  ← Mt 9:13
        """
        thml = "that whosoever.<br /><scripRef>15; Mt 9:13</scripRef>"
        groups = parse_verse_refs(thml, source_book_id=43, source_chapter=3, abbrev_map=_ABBREV_MAP)
        assert len(groups) == 1
        anchor, targets = groups[0]
        assert anchor == "that whosoever"
        assert (43, 3, 15) in targets
        assert (40, 9, 13) in targets
        assert len(targets) == 2

    def test_chapter_summary_scripref_skipped(self) -> None:
        """Chapter-summary scripRefs (with passage= attr) must be skipped entirely."""
        thml = (
            '<scripRef passage="Ge 1:1">1</scripRef> God creates heaven and earth;<br />'
            '<scripRef passage="Ge 1:3">3</scripRef> the light;<br />'
            "beginning.<br />"
            "<scripRef>Pr 8:22-24; 16:4</scripRef>"
        )
        groups = parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP)
        # Only the non-passage scripRef should produce a group
        assert len(groups) == 1
        anchor, targets = groups[0]
        assert anchor == "beginning"
        assert (20, 8, 22) in targets
        assert (20, 8, 23) in targets
        assert (20, 8, 24) in targets
        assert (20, 16, 4) in targets

    def test_empty_thml_returns_empty(self) -> None:
        """Empty or whitespace ThML produces no results."""
        assert parse_verse_refs("", source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP) == []
        assert parse_verse_refs("  \n\n  ", source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP) == []

    def test_only_chapter_summary_returns_empty(self) -> None:
        """ThML with only chapter-summary scripRefs produces no cross-refs."""
        thml = (
            '<scripRef passage="Ge 1:1">1</scripRef> God creates heaven;<br />'
            '<scripRef passage="Ge 1:3">3</scripRef> the light'
        )
        assert parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP) == []

    def test_multiple_anchor_groups(self) -> None:
        """Multiple anchor groups in one verse are parsed independently."""
        thml = "God.<br /><scripRef>Ex 20:11; Ne 9:6</scripRef><br />created.<br /><scripRef>Ps 33:6</scripRef>"
        groups = parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP)
        assert len(groups) == 2
        anchors = [g[0] for g in groups]
        assert "God" in anchors
        assert "created" in anchors

        god_targets = next(t for a, t in groups if a == "God")
        assert (2, 20, 11) in god_targets  # Ex 20:11
        assert (16, 9, 6) in god_targets  # Ne 9:6

        created_targets = next(t for a, t in groups if a == "created")
        assert (19, 33, 6) in created_targets  # Ps 33:6

    def test_unknown_book_abbreviation_skipped(self) -> None:
        """References to unknown book abbreviations are silently skipped."""
        thml = "anchor.<br /><scripRef>Xyz 1:1; Ge 1:1</scripRef>"
        groups = parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP)
        assert len(groups) == 1
        _, targets = groups[0]
        # Xyz is unknown; only Ge 1:1 should appear
        assert (1, 1, 1) in targets
        # No target with book 0 or similar garbage
        assert all(b > 0 for b, _, _ in targets)

    def test_verse_range_correct_expansion(self) -> None:
        """Verse ranges are fully expanded."""
        thml = "test.<br /><scripRef>Ge 1:1-5</scripRef>"
        groups = parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP)
        _, targets = groups[0]
        assert targets == [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5)]

    def test_gen_1_1_actual_format(self) -> None:
        """Simulate the actual Gen 1:1 ThML structure from the module."""
        thml = (
            '<br /><scripRef passage="Ge 1:1">1</scripRef> God creates heaven and earth;<br />'
            '<scripRef passage="Ge 1:3">3</scripRef> the light;<br />'
            "\n<br />\n<br />beginning.<br />"
            "<scripRef>Pr 8:22-24; 16:4; Mr 13:19; Joh 1:1-3; Heb 1:10; 1Jo 1:1</scripRef><br />"
            "God.<br /><scripRef>Ex 20:11; Ps 33:6,9</scripRef>\n"
        )
        groups = parse_verse_refs(thml, source_book_id=1, source_chapter=1, abbrev_map=_ABBREV_MAP)
        # Should have 2 groups: "beginning" and "God"
        anchors = {g[0] for g in groups}
        assert "beginning" in anchors
        assert "God" in anchors

        beg_targets = next(t for a, t in groups if a == "beginning")
        # Check key expected targets
        assert (43, 1, 1) in beg_targets  # Joh 1:1
        assert (58, 1, 10) in beg_targets  # Heb 1:10

        god_targets = next(t for a, t in groups if a == "God")
        assert (2, 20, 11) in god_targets  # Ex 20:11
        # Ps 33:6,9 should produce both verses
        assert (19, 33, 6) in god_targets
        assert (19, 33, 9) in god_targets


# ---------------------------------------------------------------------------
# Unit tests: build_abbrev_map (fallback only)
# ---------------------------------------------------------------------------


class TestFallbackAbbrevMap:
    def test_fallback_map_covers_all_66_books(self) -> None:
        book_ids = set(_FALLBACK_ABBREV_MAP.values())
        assert book_ids == set(range(1, 67)), f"Missing book IDs: {set(range(1, 67)) - book_ids}"

    def test_genesis_is_1(self) -> None:
        assert _FALLBACK_ABBREV_MAP["Ge"] == 1

    def test_revelation_is_66(self) -> None:
        assert _FALLBACK_ABBREV_MAP["Re"] == 66

    def test_john_is_43(self) -> None:
        assert _FALLBACK_ABBREV_MAP["Joh"] == 43

    def test_hebrews_is_58(self) -> None:
        assert _FALLBACK_ABBREV_MAP["Heb"] == 58


# ---------------------------------------------------------------------------
# Integration tests (guarded — require TSK.zip to exist)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TSK_AVAILABLE, reason="bible_data/sources/TSK.zip not present")
class TestTSKIntegration:
    """Live integration tests against the actual TSK SWORD module."""

    def test_build_abbrev_map_from_module(self) -> None:
        """build_abbrev_map discovers abbreviations from the real module."""
        bible = open_tsk_bible(TSK_ZIP)
        amap = build_abbrev_map(bible)
        # Must cover all 66 books
        assert max(amap.values()) == 66
        assert min(amap.values()) == 1
        # Key abbreviations present
        assert amap["Ge"] == 1
        assert amap["Re"] == 66
        assert amap["Joh"] == 43

    def test_gen_1_1_has_john_1_1_and_heb_1_10(self) -> None:
        """Gen 1:1 must have cross-refs to John 1:1 (43,1,1) and Heb 1:10 (58,1,10)."""
        found_john = False
        found_heb = False

        for rec in iter_tsk_cross_references(TSK_ZIP):
            if rec["source_book_id"] != 1 or rec["source_chapter"] != 1 or rec["source_verse"] != 1:
                continue
            if (rec["target_book_id"], rec["target_chapter"], rec["target_verse"]) == (43, 1, 1):
                found_john = True
            if (rec["target_book_id"], rec["target_chapter"], rec["target_verse"]) == (58, 1, 10):
                found_heb = True
            if found_john and found_heb:
                break

        assert found_john, "Gen 1:1 should have a cross-ref to John 1:1 (book 43, ch 1, v 1)"
        assert found_heb, "Gen 1:1 should have a cross-ref to Heb 1:10 (book 58, ch 1, v 10)"

    def test_importer_populates_table_and_is_idempotent(self) -> None:
        """import_tsk_candidates fills the table; second run adds zero rows."""
        from abba.database.tsk_importer import import_tsk_candidates

        # ignore_cleanup_errors avoids Windows WinError 32 (file-still-open) on WAL files
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # First run
            count1 = import_tsk_candidates(db_path, TSK_ZIP)
            assert count1 > 0, "Should insert at least one row on first run"

            # Verify rows in DB
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute("SELECT COUNT(*) FROM cross_reference_candidates").fetchone()
                assert row[0] == count1

            # Second run must be idempotent
            count2 = import_tsk_candidates(db_path, TSK_ZIP)
            assert count2 == 0, f"Second run should insert 0 rows, got {count2}"

            # Row count unchanged
            with sqlite3.connect(str(db_path)) as conn:
                row2 = conn.execute("SELECT COUNT(*) FROM cross_reference_candidates").fetchone()
                assert row2[0] == count1

    def test_total_cross_ref_count_is_substantial(self) -> None:
        """TSK should yield hundreds of thousands of cross-references."""
        from abba.database.tsk_importer import import_tsk_candidates

        # ignore_cleanup_errors avoids Windows WinError 32 on WAL files
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            count = import_tsk_candidates(db_path, TSK_ZIP)
            # TSK is famously large — conservative lower bound
            assert count > 100_000, f"Expected >100k cross-refs, got {count}"
