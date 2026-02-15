"""Tests for Phase 5-8 enrichment modules: passages, cultural context, literary structures, reading plans."""

import json
import sqlite3

import pytest

from abba.api.semantic_search import LRUCache
from abba.enrichment.cultural_context import BOOK_INTRODUCTIONS, CulturalContextPopulator
from abba.enrichment.literary_structures import CURATED_STRUCTURES, LiteraryStructurePopulator
from abba.enrichment.passages import CURATED_PASSAGES, PassagePopulator
from abba.enrichment.reading_plans import PLAN_ENTRIES, READING_PLANS, ReadingPlanPopulator


@pytest.fixture
def enrichment_db(tmp_path):
    """Create a temporary database with enrichment tables."""
    from abba.database import SQLiteManager
    from abba.database.migrations import (
        add_cultural_context_table,
        add_literary_structures_table,
        add_passages_table,
    )

    db_path = tmp_path / "test_enrichment.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    add_passages_table(db_path)
    add_cultural_context_table(db_path)
    add_literary_structures_table(db_path)

    return db_path


class TestPassagesData:
    """Tests for curated passage data."""

    def test_passages_have_required_fields(self):
        """All passages have book_id, title, genre, and literary type."""
        for p in CURATED_PASSAGES:
            book_id, s_ch, s_vs, e_ch, e_vs, title, genre, lit_type, order = p
            assert 1 <= book_id <= 66, f"Invalid book_id: {book_id}"
            assert s_ch >= 1
            assert s_vs >= 1
            assert e_ch >= s_ch
            assert title, "Title must not be empty"
            assert genre, "Genre must not be empty"

    def test_passages_cover_multiple_books(self):
        """Passages span many books of the Bible."""
        book_ids = {p[0] for p in CURATED_PASSAGES}
        assert len(book_ids) >= 20

    def test_populate_and_query(self, enrichment_db):
        """Passages can be populated and queried."""
        inserted = PassagePopulator(enrichment_db).populate()
        assert inserted > 50

        passages = PassagePopulator.get_passages_for_chapter(enrichment_db, 1, 1)
        assert len(passages) > 0
        assert passages[0]["title"] is not None

    def test_get_passage_for_verse(self, enrichment_db):
        """Can find passages containing a specific verse."""
        PassagePopulator(enrichment_db).populate()
        passages = PassagePopulator.get_passage_for_verse(enrichment_db, 40, 5, 3)
        # Matt 5:3 is in Sermon on the Mount and Beatitudes
        assert len(passages) >= 1
        titles = [p["title"] for p in passages]
        assert any("Beatitude" in t or "Sermon" in t for t in titles)


class TestCulturalContextData:
    """Tests for cultural context data."""

    def test_introductions_have_required_fields(self):
        """All introductions have required content."""
        for entry in BOOK_INTRODUCTIONS:
            book_id, ctx_type, title, summary, detailed, time_period, geo, conf = entry
            assert 1 <= book_id <= 66
            assert ctx_type == "historical_background"
            assert title, "Title must not be empty"
            assert summary, "Summary must not be empty"
            assert detailed, "Detailed content must not be empty"

    def test_populate_and_query(self, enrichment_db):
        """Cultural context can be populated and queried."""
        inserted = CulturalContextPopulator(enrichment_db).populate()
        assert inserted > 10

        contexts = CulturalContextPopulator.get_context_for_book(enrichment_db, 1)
        assert len(contexts) > 0
        assert "Genesis" in contexts[0]["title"]

    def test_idempotent_populate(self, enrichment_db):
        """Second populate doesn't duplicate data."""
        first = CulturalContextPopulator(enrichment_db).populate()
        second = CulturalContextPopulator(enrichment_db).populate()
        assert first > 0
        assert second == 0


class TestLiteraryStructuresData:
    """Tests for literary structure annotations."""

    def test_structures_have_valid_types(self):
        """All structures have valid types."""
        valid_types = {"chiasmus", "acrostic", "parallelism", "inclusio"}
        for s in CURATED_STRUCTURES:
            assert s[5] in valid_types, f"Invalid structure type: {s[5]}"

    def test_structures_have_parseable_elements(self):
        """All structure element JSON is valid."""
        for s in CURATED_STRUCTURES:
            elements = json.loads(s[8])
            assert isinstance(elements, list)
            assert len(elements) > 0
            for elem in elements:
                assert "label" in elem
                assert "ref" in elem

    def test_structures_have_scholarly_sources(self):
        """All structures cite scholarly sources."""
        for s in CURATED_STRUCTURES:
            assert s[9], f"Missing scholarly source for structure in book {s[0]}"

    def test_populate_and_query(self, enrichment_db):
        """Literary structures can be populated and queried."""
        inserted = LiteraryStructurePopulator(enrichment_db).populate()
        assert inserted > 10

        # John 1:1 should be in John's Prologue chiasmus
        structures = LiteraryStructurePopulator.get_structures_for_verse(enrichment_db, 43, 1, 1)
        assert len(structures) > 0
        assert structures[0]["structure_type"] == "chiasmus"


class TestReadingPlanData:
    """Tests for reading plan data."""

    def test_plans_have_required_fields(self):
        """All plans have slug, name, and category."""
        for slug, name, _desc, category, days in READING_PLANS:
            assert slug, "Slug must not be empty"
            assert name, "Name must not be empty"
            assert category in ("beginner", "devotional", "topical", "overview")
            assert days > 0

    def test_entries_reference_valid_books(self):
        """All plan entries reference valid book IDs."""
        for entry in PLAN_ENTRIES:
            plan_slug, day, book_id, s_ch, s_vs, e_ch, e_vs, title, refl = entry
            assert 1 <= book_id <= 66
            assert day >= 1
            assert title, "Title must not be empty"
            assert refl, "Reflection question must not be empty"

    def test_start_here_plan_has_7_days(self):
        """Start Here plan has exactly 7 entries."""
        start_here_entries = [e for e in PLAN_ENTRIES if e[0] == "start-here"]
        assert len(start_here_entries) == 7
        days = [e[1] for e in start_here_entries]
        assert days == list(range(1, 8))

    def test_populate_and_query(self, enrichment_db):
        """Reading plans can be populated and queried."""
        pop = ReadingPlanPopulator(enrichment_db)
        counts = pop.populate()
        assert counts["plans"] > 0
        assert counts["entries"] > 0

        plans = ReadingPlanPopulator.get_plans(enrichment_db)
        assert len(plans) > 0
        assert plans[0]["slug"] is not None

        entries = ReadingPlanPopulator.get_plan_entries(enrichment_db, "start-here")
        assert len(entries) == 7

    def test_idempotent_populate(self, enrichment_db):
        """Second populate doesn't duplicate plans."""
        pop = ReadingPlanPopulator(enrichment_db)
        first = pop.populate()
        second = pop.populate()
        assert first["plans"] > 0
        assert second["plans"] == 0


class TestLRUCache:
    """Tests for the search result LRU cache."""

    def test_basic_get_put(self):
        """Cache stores and retrieves values."""
        cache = LRUCache(max_size=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        """Missing keys return None."""
        cache = LRUCache(max_size=10)
        assert cache.get("missing") is None

    def test_eviction(self):
        """Oldest entry is evicted when cache is full."""
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_order(self):
        """Accessing an entry moves it to front (prevents eviction)."""
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # refresh "a"
        cache.put("c", 3)  # should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_stats_tracking(self):
        """Cache tracks hits and misses."""
        cache = LRUCache(max_size=10)
        cache.put("x", 1)
        cache.get("x")  # hit
        cache.get("y")  # miss
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.size == 1

    def test_clear(self):
        """Clear empties the cache and resets stats."""
        cache = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.get("a") is None
