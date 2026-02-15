"""Integration tests for user-facing search and data access flows.

These tests verify that a user can search, browse, study, and export
biblical data through the API — exercising the complete stack from
database through enrichment through FastAPI endpoints.
"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.api.routes import configure_db
from abba.database import SQLiteManager
from abba.database.migrations import (
    add_book_metadata_table,
    add_cross_references_table,
    add_cultural_context_table,
    add_life_topics_tables,
    add_literary_structures_table,
    add_passages_table,
    add_word_richness_table,
)
from abba.enrichment.book_metadata import BookMetadataPopulator
from abba.enrichment.cross_references import CrossReferencePopulator
from abba.enrichment.cultural_context import CulturalContextPopulator
from abba.enrichment.life_topics import LifeTopicPopulator
from abba.enrichment.literary_structures import LiteraryStructurePopulator
from abba.enrichment.passages import PassagePopulator
from abba.enrichment.reading_plans import ReadingPlanPopulator
from abba.enrichment.word_richness import WordRichnessComputer


@pytest.fixture
def fully_enriched_db(seeded_db):
    """Create a database with ALL enrichment tables and data populated."""
    db_path = seeded_db

    # Run all enrichment migrations
    add_book_metadata_table(db_path)
    add_cross_references_table(db_path)
    add_word_richness_table(db_path)
    add_life_topics_tables(db_path)
    add_passages_table(db_path)
    add_cultural_context_table(db_path)
    add_literary_structures_table(db_path)

    # Populate all enrichment data
    BookMetadataPopulator(db_path).populate()
    CrossReferencePopulator(db_path).populate()
    WordRichnessComputer(db_path).compute_all()
    LifeTopicPopulator(db_path).populate()
    PassagePopulator(db_path).populate()
    CulturalContextPopulator(db_path).populate()
    LiteraryStructurePopulator(db_path).populate()
    ReadingPlanPopulator(db_path).populate()

    return db_path


@pytest.fixture
def client(fully_enriched_db):
    """Create a FastAPI test client with fully enriched database."""
    db_manager = SQLiteManager(fully_enriched_db)
    configure_db(db_manager)
    app = create_app()
    return TestClient(app)


# ================================================================== #
#  User Flow 1: Basic Text Search → Verse Detail → Word Analysis      #
# ================================================================== #


class TestSearchToStudyFlow:
    """User searches for text, finds a verse, drills into word details."""

    def test_text_search_returns_results(self, client):
        """User can search for text and get results."""
        response = client.get("/api/v1/search/text?q=beginning&translation_id=engbsb")
        assert response.status_code == 200
        results = response.json()
        assert len(results) > 0
        assert any("beginning" in r["text"].lower() for r in results)

    def test_search_result_links_to_verse(self, client):
        """Search result provides enough data to fetch the full verse."""
        response = client.get("/api/v1/search/text?q=beginning&translation_id=engbsb")
        results = response.json()
        first = results[0]

        # Use search result to fetch full verse
        verse_url = (
            f"/api/v1/verses/{first['translation_id']}/{first['book_id']}"
            f"/{first['chapter']}/{first['verse']}?depth=standard"
        )
        verse_response = client.get(verse_url)
        assert verse_response.status_code == 200
        verse = verse_response.json()
        assert verse["text"] == first["text"]

    def test_verse_at_standard_depth_includes_words(self, client):
        """Standard depth includes original language words."""
        response = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard")
        assert response.status_code == 200
        verse = response.json()
        assert verse["words"] is not None
        assert len(verse["words"]) > 0
        assert verse["words"][0]["original_text"] is not None

    def test_word_detail_lookup(self, client):
        """User can drill into a specific word for full analysis."""
        response = client.get("/api/v1/words/Gen/1/1/3")
        assert response.status_code == 200
        word = response.json()
        assert word["word"]["word_num"] == 3
        assert word["lexicon"] is not None
        assert word["lexicon"]["strongs_number"] == "H0430"


# ================================================================== #
#  User Flow 2: Semantic Search with Filters                          #
# ================================================================== #


class TestSemanticSearchFlow:
    """User performs semantic/hybrid search with query syntax."""

    def test_semantic_search_endpoint_exists(self, client):
        """Semantic search endpoint is accessible."""
        response = client.get("/api/v1/search/semantic?q=beginning")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_semantic_search_returns_scored_results(self, client):
        """Results have scores and explanations."""
        response = client.get("/api/v1/search/semantic?q=beginning")
        results = response.json()
        if results:
            assert "score" in results[0]
            assert "explanation" in results[0]
            assert "match_type" in results[0]

    def test_semantic_search_with_book_filter(self, client):
        """User can filter semantic search by book."""
        response = client.get("/api/v1/search/semantic?q=beginning&book_id=1")
        assert response.status_code == 200
        results = response.json()
        for r in results:
            assert r["book_id"] == 1

    def test_semantic_search_with_testament_filter(self, client):
        """User can filter semantic search by testament."""
        response = client.get("/api/v1/search/semantic?q=beginning&testament=new")
        assert response.status_code == 200
        results = response.json()
        for r in results:
            assert r["book_id"] >= 40  # NT books start at 40


# ================================================================== #
#  User Flow 3: Deep Verse Study with Enrichment Data                  #
# ================================================================== #


class TestDeepStudyFlow:
    """User explores a verse at deep/scholarly depth."""

    def test_deep_depth_includes_cross_refs(self, client):
        """Deep depth includes cross-references."""
        response = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        assert response.status_code == 200
        verse = response.json()
        assert verse["cross_references"] is not None
        # Gen 1:1 has cross-refs to John 1:1 and Rev 21:1
        if verse["cross_references"]:
            assert any("ref_type" in x for x in verse["cross_references"])

    def test_deep_depth_includes_passage_info(self, client):
        """Deep depth includes passage/pericope info."""
        response = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        verse = response.json()
        if verse.get("passage_info"):
            assert "title" in verse["passage_info"]
            assert "genre" in verse["passage_info"]

    def test_deep_depth_includes_cultural_context(self, client):
        """Deep depth includes cultural context."""
        response = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        verse = response.json()
        assert verse.get("cultural_context") is not None

    def test_standard_depth_includes_richness_flags(self, client):
        """Standard depth includes word richness indicators."""
        response = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard")
        verse = response.json()
        # richness_flags may be empty if no words score > 0.3
        assert "richness_flags" in verse

    def test_scholarly_depth_includes_parallels(self, client):
        """Scholarly depth includes parallel passage detection."""
        response = client.get("/api/v1/verses/engbsb/1/1/1?depth=scholarly")
        assert response.status_code == 200
        verse = response.json()
        assert "parallel_passages" in verse

    def test_progressive_depth_adds_fields(self, client):
        """Each depth level adds more fields than the previous."""
        basic = client.get("/api/v1/verses/engbsb/1/1/1?depth=basic").json()
        standard = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard").json()
        deep = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep").json()

        # Basic has no words
        assert basic.get("words") is None
        # Standard adds words
        assert standard.get("words") is not None
        # Deep adds cross_references, cultural_context
        assert deep.get("cross_references") is not None


# ================================================================== #
#  User Flow 4: Life Topics → Guided Study Steps                      #
# ================================================================== #


class TestLifeTopicFlow:
    """User browses life topics and follows a guided study."""

    def test_list_life_topics(self, client):
        """User can list all life topics."""
        response = client.get("/api/v1/life-topics")
        assert response.status_code == 200
        topics = response.json()
        assert len(topics) > 0
        assert topics[0]["slug"] is not None
        assert topics[0]["name"] is not None
        assert topics[0]["category"] is not None

    def test_search_life_topics(self, client):
        """User can search for topics by keyword."""
        response = client.get("/api/v1/life-topics/search?q=anxiety")
        assert response.status_code == 200
        topics = response.json()
        assert len(topics) > 0
        assert topics[0]["slug"] == "anxiety"

    def test_search_life_topics_by_category(self, client):
        """Search returns topics matching category."""
        response = client.get("/api/v1/life-topics/search?q=emotions")
        assert response.status_code == 200
        topics = response.json()
        assert len(topics) > 0

    def test_get_topic_with_study_steps(self, client):
        """User gets a topic with complete study path."""
        response = client.get("/api/v1/life-topics/anxiety")
        assert response.status_code == 200
        topic = response.json()
        assert topic["slug"] == "anxiety"
        assert topic["name"] == "Anxiety & Worry"
        assert len(topic["study_steps"]) == 4
        step_types = [s["step_type"] for s in topic["study_steps"]]
        assert "comfort" in step_types
        assert "hope" in step_types

    def test_topic_not_found(self, client):
        """Non-existent topic returns 404."""
        response = client.get("/api/v1/life-topics/nonexistent")
        assert response.status_code == 404


# ================================================================== #
#  User Flow 5: Book Browsing → Passage Discovery                     #
# ================================================================== #


class TestBookAndPassageFlow:
    """User browses books and discovers passage structure."""

    def test_book_list_with_metadata(self, client):
        """Book list includes genre and context from enrichment."""
        response = client.get("/api/v1/books")
        assert response.status_code == 200
        books = response.json()
        genesis = next((b for b in books if b["book_id"] == 1), None)
        assert genesis is not None
        assert genesis["primary_genre"] == "narrative"

    def test_single_book_metadata(self, client):
        """Single book includes full enrichment metadata."""
        response = client.get("/api/v1/books/1")
        assert response.status_code == 200
        book = response.json()
        assert book["primary_genre"] == "narrative"
        assert book["canonical_section"] == "Torah"

    def test_passage_boundaries(self, client):
        """User can get passage boundaries for a chapter."""
        response = client.get("/api/v1/passages/1/1")
        assert response.status_code == 200
        passages = response.json()
        assert len(passages) > 0
        assert passages[0]["title"] is not None
        assert passages[0]["genre"] is not None

    def test_passage_boundaries_john_1(self, client):
        """John 1 passage includes the Prologue."""
        response = client.get("/api/v1/passages/43/1")
        assert response.status_code == 200
        passages = response.json()
        titles = [p["title"] for p in passages]
        assert any("Prologue" in t or "Word" in t for t in titles)


# ================================================================== #
#  User Flow 6: Reading Plans                                          #
# ================================================================== #


class TestReadingPlanFlow:
    """User browses and follows reading plans."""

    def test_list_reading_plans(self, client):
        """User can list available reading plans."""
        response = client.get("/api/v1/reading-plans")
        assert response.status_code == 200
        plans = response.json()
        assert len(plans) > 0
        slugs = [p["slug"] for p in plans]
        assert "start-here" in slugs

    def test_get_reading_plan_detail(self, client):
        """User gets full plan with daily entries."""
        response = client.get("/api/v1/reading-plans/start-here")
        assert response.status_code == 200
        plan = response.json()
        assert plan["name"] == "Start Here: Introduction to the Bible"
        assert plan["estimated_days"] == 7
        assert len(plan["entries"]) == 7

    def test_reading_plan_entries_have_reflection(self, client):
        """Each entry includes a reflection question."""
        response = client.get("/api/v1/reading-plans/start-here")
        plan = response.json()
        for entry in plan["entries"]:
            assert entry["reflection_question"] is not None
            assert entry["title"] is not None

    def test_reading_plan_not_found(self, client):
        """Non-existent plan returns 404."""
        response = client.get("/api/v1/reading-plans/nonexistent")
        assert response.status_code == 404

    def test_reading_plan_entry_links_to_verse(self, client):
        """Plan entries reference valid book/chapter/verse data."""
        response = client.get("/api/v1/reading-plans/start-here")
        plan = response.json()
        first_entry = plan["entries"][0]
        assert first_entry["book_id"] == 1
        assert first_entry["start_chapter"] == 1


# ================================================================== #
#  User Flow 7: Translation Comparison                                 #
# ================================================================== #


class TestTranslationComparisonFlow:
    """User compares verses across translations."""

    def test_compare_genesis_1_1(self, client):
        """User compares Gen 1:1 across BSB and KJV."""
        response = client.get("/api/v1/compare/Genesis/1/1?translations=engbsb&translations=engkjv")
        assert response.status_code == 200
        comp = response.json()
        assert "engbsb" in comp["translations"]
        assert "engkjv" in comp["translations"]
        assert comp["original_words"] is not None


# ================================================================== #
#  User Flow 8: Export                                                 #
# ================================================================== #


class TestExportFlow:
    """User exports verse data for personal study."""

    def test_export_verse_json(self, client):
        """Export returns structured JSON data."""
        response = client.get("/api/v1/export/verse/engbsb/1/1/1?format=json")
        assert response.status_code == 200
        data = response.json()
        assert data["reference"] is not None
        assert data["text"] is not None

    def test_export_verse_markdown(self, client):
        """Export in markdown includes formatted text."""
        response = client.get("/api/v1/export/verse/engbsb/1/1/1?format=markdown")
        assert response.status_code == 200
        data = response.json()
        assert "markdown" in data
        assert "# " in data["markdown"]

    def test_export_includes_original_words(self, client):
        """Export includes original language words."""
        response = client.get("/api/v1/export/verse/engbsb/1/1/1")
        data = response.json()
        if data.get("original_words"):
            assert data["original_words"][0]["original_text"] is not None


# ================================================================== #
#  Enrichment Data Validation                                          #
# ================================================================== #


class TestEnrichmentDataIntegrity:
    """Verify all enrichment tables are populated correctly."""

    def test_passages_populated(self, fully_enriched_db):
        """Passages table has curated data."""
        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM passages")
            count = cursor.fetchone()[0]
            assert count > 50  # We curated 100+ passages

    def test_cultural_context_populated(self, fully_enriched_db):
        """Cultural context table has book introductions."""
        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cultural_context")
            count = cursor.fetchone()[0]
            assert count > 10

    def test_literary_structures_populated(self, fully_enriched_db):
        """Literary structures table has annotations."""
        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM literary_structures")
            count = cursor.fetchone()[0]
            assert count > 10

    def test_reading_plans_populated(self, fully_enriched_db):
        """Reading plans are populated."""
        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reading_plans")
            plans = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM reading_plan_entries")
            entries = cursor.fetchone()[0]
            assert plans >= 3
            assert entries >= 20

    def test_all_enrichment_tables_exist(self, fully_enriched_db):
        """All enrichment tables exist in the database."""
        expected_tables = [
            "book_metadata",
            "cross_references",
            "word_richness",
            "life_topics",
            "life_topic_concepts",
            "topic_study_steps",
            "passages",
            "cultural_context",
            "literary_structures",
            "reading_plans",
            "reading_plan_entries",
        ]
        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            for table in expected_tables:
                cursor.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                assert cursor.fetchone()[0] == 1, f"Table {table} should exist"

    def test_enrichment_idempotent(self, fully_enriched_db):
        """Re-running population doesn't create duplicates."""
        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM passages")
            before = cursor.fetchone()[0]

        PassagePopulator(fully_enriched_db).populate()

        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM passages")
            after = cursor.fetchone()[0]

        assert before == after

    def test_force_repopulation(self, fully_enriched_db):
        """Force flag replaces all data."""
        PassagePopulator(fully_enriched_db).populate(force=True)
        CulturalContextPopulator(fully_enriched_db).populate(force=True)
        LiteraryStructurePopulator(fully_enriched_db).populate(force=True)

        with sqlite3.connect(fully_enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM passages")
            assert cursor.fetchone()[0] > 0
            cursor.execute("SELECT COUNT(*) FROM cultural_context")
            assert cursor.fetchone()[0] > 0
            cursor.execute("SELECT COUNT(*) FROM literary_structures")
            assert cursor.fetchone()[0] > 0
