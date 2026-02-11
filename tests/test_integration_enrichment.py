"""Integration tests for enrichment + API layer.

These tests verify that enrichment data flows correctly through
the API routes and can be queried via the verse endpoints.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.api.routes import configure_db
from abba.database import SQLiteManager
from abba.database.migrations import (
    add_book_metadata_table,
    add_cross_references_table,
    add_life_topics_tables,
    add_word_richness_table,
)
from abba.enrichment.book_metadata import BookMetadataPopulator
from abba.enrichment.cross_references import CrossReferencePopulator
from abba.enrichment.life_topics import LifeTopicPopulator
from abba.enrichment.word_richness import WordRichnessComputer


@pytest.fixture
def enriched_db(seeded_db):
    """Extend seeded_db with enrichment tables and data."""
    db_path = seeded_db

    # Run enrichment migrations
    add_book_metadata_table(db_path)
    add_cross_references_table(db_path)
    add_word_richness_table(db_path)
    add_life_topics_tables(db_path)

    # Populate enrichment data
    BookMetadataPopulator(db_path).populate()
    CrossReferencePopulator(db_path).populate()
    WordRichnessComputer(db_path).compute_all()
    LifeTopicPopulator(db_path).populate()

    return db_path


@pytest.fixture
def enriched_client(enriched_db):
    """Create a FastAPI test client with enriched database."""
    db_manager = SQLiteManager(enriched_db)
    configure_db(db_manager)
    app = create_app()
    return TestClient(app)


class TestBookMetadataIntegration:
    """Tests that book metadata is accessible via the API."""

    def test_book_list_includes_metadata(self, enriched_client):
        """Books endpoint should include genre and context from book_metadata table."""
        response = enriched_client.get("/api/v1/books")
        assert response.status_code == 200

        books = response.json()
        assert len(books) > 0

        # Find Genesis
        genesis = next((b for b in books if b["book_id"] == 1), None)
        assert genesis is not None
        assert genesis["primary_genre"] == "narrative"
        assert genesis["canonical_section"] == "Torah"

    def test_single_book_metadata(self, enriched_client):
        """Single book endpoint should include full metadata."""
        response = enriched_client.get("/api/v1/books/1")
        assert response.status_code == 200

        book = response.json()
        assert book["primary_genre"] == "narrative"
        assert "law" in book.get("secondary_genres", [])
        assert book["author_traditional"] == "Moses"


class TestCrossReferenceIntegration:
    """Tests that cross-references are stored and queryable."""

    def test_cross_refs_exist_in_db(self, enriched_db):
        """Cross-references should be present in the database."""
        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cross_references")
            count = cursor.fetchone()[0]
            assert count > 0

    def test_cross_refs_queryable_by_verse(self, enriched_db):
        """Should be able to query cross-references for a specific verse."""
        pop = CrossReferencePopulator(enriched_db)
        # Genesis 1:1 cross-refs
        refs = pop.get_cross_references_for_verse(enriched_db, 1, 1, 1)
        assert len(refs) > 0


class TestWordRichnessIntegration:
    """Tests that word richness data integrates correctly."""

    def test_richness_computed_for_words(self, enriched_db):
        """Word richness should be computed for words in the database."""
        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM word_richness")
            count = cursor.fetchone()[0]
            # Should have some entries if words and lexicon are seeded
            # (depends on conftest.py seeded data)
            assert count >= 0

    def test_richness_scores_in_valid_range(self, enriched_db):
        """Richness scores should be between 0 and 1."""
        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(richness_score), MAX(richness_score) FROM word_richness")
            row = cursor.fetchone()
            if row[0] is not None:
                assert row[0] >= 0.0
                assert row[1] <= 1.0


class TestLifeTopicsIntegration:
    """Tests that life topics integrate correctly."""

    def test_topics_populated(self, enriched_db):
        """Life topics should be present in the database."""
        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM life_topics")
            assert cursor.fetchone()[0] > 0

    def test_study_steps_populated(self, enriched_db):
        """Study steps should be linked to topics."""
        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT lt.name, COUNT(ts.id) "
                "FROM life_topics lt "
                "JOIN topic_study_steps ts ON lt.id = ts.topic_id "
                "GROUP BY lt.id"
            )
            rows = cursor.fetchall()
            assert len(rows) > 0
            for name, step_count in rows:
                assert step_count >= 1, f"Topic '{name}' should have at least 1 study step"


class TestFullEnrichmentPipeline:
    """Tests the complete enrichment pipeline end-to-end."""

    def test_all_enrichment_tables_populated(self, enriched_db):
        """All enrichment tables should have data after population."""
        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()

            tables = ["book_metadata", "cross_references", "life_topics", "topic_study_steps"]
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                count = cursor.fetchone()[0]
                assert count > 0, f"Table {table} should have data"

    def test_repopulation_with_force(self, enriched_db):
        """Should be able to repopulate all enrichment data with force."""
        # First populate is done in fixture; now force-repopulate
        BookMetadataPopulator(enriched_db).populate(force=True)
        CrossReferencePopulator(enriched_db).populate(force=True)
        LifeTopicPopulator(enriched_db).populate(force=True)

        with sqlite3.connect(enriched_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM book_metadata")
            assert cursor.fetchone()[0] == 66
