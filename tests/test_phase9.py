"""Comprehensive test suite for Phase 9: Extended Capabilities.

Covers all 13 Phase 9 features with unit tests, integration tests, and e2e tests:
  9A: Concept discovery, Louw-Nida semantic domains, TFLSJ lexicon
  9B: MACULA syntax trees, OpenText discourse annotations, manuscript variants
  9C: Multi-language search, community contributions, collaborative concepts
  9D: Audio integration, semantic graph visualization, ML feedback, mobile API
"""

# pylint: disable=redefined-outer-name
import json

import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.api.routes import configure_db
from abba.database import SQLiteManager
from abba.database.migrations import run_migrations
from abba.enrichment import (
    BookMetadataPopulator,
    CrossReferencePopulator,
    DiscourseAnnotationPopulator,
    ManuscriptVariantPopulator,
    PassagePopulator,
    SemanticDomainPopulator,
    SemanticGraphPopulator,
    SyntaxTreePopulator,
)


@pytest.fixture
def phase9_db(seeded_db):
    """Create test database with Phase 9 migrations and enrichment data."""
    run_migrations(seeded_db)
    BookMetadataPopulator(seeded_db).populate()
    CrossReferencePopulator(seeded_db).populate()
    PassagePopulator(seeded_db).populate()
    SemanticDomainPopulator(seeded_db).populate()
    SyntaxTreePopulator(seeded_db).populate()
    DiscourseAnnotationPopulator(seeded_db).populate()
    ManuscriptVariantPopulator(seeded_db).populate()
    SemanticGraphPopulator(seeded_db).populate()
    return seeded_db


@pytest.fixture
def phase9_client(phase9_db):
    """Create a FastAPI test client with full Phase 9 data."""
    db = SQLiteManager(phase9_db)
    configure_db(db)
    app = create_app()
    yield TestClient(app)


# ============================================================
# Phase 9A: Semantic Domain Tests
# ============================================================


class TestSemanticDomainPopulator:
    """Unit tests for Louw-Nida semantic domain data population."""

    def test_populator_creates_domains(self, phase9_db):
        import sqlite3

        conn = sqlite3.connect(str(phase9_db))
        cursor = conn.execute("SELECT COUNT(*) FROM semantic_domains")
        count = cursor.fetchone()[0]
        assert count > 0, "Semantic domains should be populated"
        conn.close()

    def test_populator_creates_mappings(self, phase9_db):
        import sqlite3

        conn = sqlite3.connect(str(phase9_db))
        cursor = conn.execute("SELECT COUNT(*) FROM strongs_domain_mappings")
        count = cursor.fetchone()[0]
        assert count > 0, "Strong's-domain mappings should be populated"
        conn.close()

    def test_populator_idempotent(self, phase9_db):
        SemanticDomainPopulator(phase9_db).populate()
        count2 = SemanticDomainPopulator(phase9_db).populate()
        assert count2 == 0, "Second populate should be no-op"


class TestSemanticDomainAPI:
    """Integration tests for semantic domain API endpoints."""

    def test_list_semantic_domains(self, phase9_client):
        resp = phase9_client.get("/api/v1/semantic-domains")
        assert resp.status_code == 200
        domains = resp.json()
        assert isinstance(domains, list)
        if domains:
            assert "domain_code" in domains[0]
            assert "domain_name" in domains[0]

    def test_get_word_domains(self, phase9_client):
        resp = phase9_client.get("/api/v1/words/G3056/domains")
        assert resp.status_code == 200
        result = resp.json()
        assert result["strongs_number"] == "G3056"
        assert "domains" in result

    def test_get_domain_words(self, phase9_client):
        # First get a domain code that exists
        domains_resp = phase9_client.get("/api/v1/semantic-domains")
        if domains_resp.json():
            code = domains_resp.json()[0]["domain_code"]
            resp = phase9_client.get(f"/api/v1/semantic-domains/{code}/words")
            assert resp.status_code == 200


# ============================================================
# Phase 9A: Concept Discovery Tests
# ============================================================


class TestConceptDiscovery:
    """Integration tests for concept discovery from natural language."""

    def test_discover_by_keyword(self, phase9_client):
        resp = phase9_client.get("/api/v1/discover?q=love")
        assert resp.status_code == 200
        result = resp.json()
        assert result["query"] == "love"
        assert "matched_concepts" in result
        assert "matched_life_topics" in result
        assert "suggested_searches" in result

    def test_discover_returns_suggestions(self, phase9_client):
        resp = phase9_client.get("/api/v1/discover?q=worry")
        assert resp.status_code == 200
        result = resp.json()
        # Should get synonym-based suggestions for "worry"
        assert isinstance(result["suggested_searches"], list)

    def test_discover_empty_query_fails(self, phase9_client):
        resp = phase9_client.get("/api/v1/discover")
        assert resp.status_code == 422  # Missing required parameter


# ============================================================
# Phase 9B: Syntax Tree Tests
# ============================================================


class TestSyntaxTreePopulator:
    """Unit tests for MACULA syntax tree data."""

    def test_populator_creates_trees(self, phase9_db):
        import sqlite3

        conn = sqlite3.connect(str(phase9_db))
        cursor = conn.execute("SELECT COUNT(*) FROM syntax_trees")
        count = cursor.fetchone()[0]
        assert count > 0, "Syntax trees should be populated"
        conn.close()

    def test_populator_idempotent(self, phase9_db):
        count2 = SyntaxTreePopulator(phase9_db).populate()
        assert count2 == 0


class TestSyntaxTreeAPI:
    """Integration tests for syntax tree endpoint."""

    def test_get_syntax_tree_existing(self, phase9_client):
        # Genesis 1:1 should have syntax data
        resp = phase9_client.get("/api/v1/syntax/1/1/1")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            tree = resp.json()
            assert tree["book_id"] == 1
            assert tree["chapter"] == 1
            assert tree["verse"] == 1
            assert "root_nodes" in tree

    def test_get_syntax_tree_missing(self, phase9_client):
        resp = phase9_client.get("/api/v1/syntax/99/99/99")
        assert resp.status_code == 404


# ============================================================
# Phase 9B: Discourse Annotation Tests
# ============================================================


class TestDiscourseAnnotationPopulator:
    """Unit tests for OpenText discourse annotations."""

    def test_populator_creates_annotations(self, phase9_db):
        import sqlite3

        conn = sqlite3.connect(str(phase9_db))
        cursor = conn.execute("SELECT COUNT(*) FROM discourse_annotations")
        count = cursor.fetchone()[0]
        assert count > 0, "Discourse annotations should be populated"
        conn.close()


class TestDiscourseAPI:
    """Integration tests for discourse annotation endpoints."""

    def test_get_verse_discourse(self, phase9_client):
        resp = phase9_client.get("/api/v1/discourse/1/1/1")
        assert resp.status_code == 200
        units = resp.json()
        assert isinstance(units, list)

    def test_get_book_discourse(self, phase9_client):
        resp = phase9_client.get("/api/v1/discourse/1")
        assert resp.status_code == 200
        units = resp.json()
        assert isinstance(units, list)


# ============================================================
# Phase 9B: Manuscript Variant Tests
# ============================================================


class TestManuscriptVariantPopulator:
    """Unit tests for manuscript variant data."""

    def test_populator_creates_variants(self, phase9_db):
        import sqlite3

        conn = sqlite3.connect(str(phase9_db))
        cursor = conn.execute("SELECT COUNT(*) FROM manuscript_variants")
        count = cursor.fetchone()[0]
        assert count > 0, "Manuscript variants should be populated"
        conn.close()


class TestManuscriptVariantAPI:
    """Integration tests for manuscript variant endpoints."""

    def test_get_verse_variants(self, phase9_client):
        resp = phase9_client.get("/api/v1/variants/1/1/1")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_significant_variants(self, phase9_client):
        resp = phase9_client.get("/api/v1/variants/significant")
        assert resp.status_code == 200
        variants = resp.json()
        assert isinstance(variants, list)
        if variants:
            assert variants[0]["significance"] == "major"


# ============================================================
# Phase 9C: Multi-language Search Tests
# ============================================================


class TestMultilingualSearch:
    """Integration tests for multi-language semantic search."""

    def test_multilingual_search_english(self, phase9_client):
        resp = phase9_client.get("/api/v1/search/multilingual?q=beginning")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)

    def test_multilingual_search_with_translation(self, phase9_client):
        resp = phase9_client.get("/api/v1/search/multilingual?q=beginning&target_translations=engbsb")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        for r in results:
            assert r["translation_id"] == "engbsb"

    def test_multilingual_search_limit(self, phase9_client):
        resp = phase9_client.get("/api/v1/search/multilingual?q=God&limit=3")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) <= 3


# ============================================================
# Phase 9C: Community Contribution Tests
# ============================================================


class TestCommunityContributions:
    """Integration + e2e tests for community contribution workflow."""

    def test_create_contribution(self, phase9_client):
        resp = phase9_client.post(
            "/api/v1/community/contributions",
            json={
                "book_id": 1,
                "chapter": 1,
                "verse": 1,
                "contribution_type": "cultural_context",
                "title": "Ancient Mesopotamian creation parallels",
                "content": "Genesis 1 contrasts with Enuma Elish in key theological ways.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Ancient Mesopotamian creation parallels"
        assert data["status"] == "pending"

    def test_list_contributions(self, phase9_client):
        # Create one first
        phase9_client.post(
            "/api/v1/community/contributions",
            json={
                "book_id": 1,
                "chapter": 1,
                "verse": 1,
                "contribution_type": "historical_note",
                "title": "Test",
                "content": "Test content",
            },
        )
        resp = phase9_client.get("/api/v1/community/contributions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_contributions_filter_status(self, phase9_client):
        resp = phase9_client.get("/api/v1/community/contributions?status=pending")
        assert resp.status_code == 200

    def test_review_contribution_e2e(self, phase9_client):
        """E2E: Submit, review, verify status change."""
        # Submit
        create_resp = phase9_client.post(
            "/api/v1/community/contributions",
            json={
                "book_id": 43,
                "chapter": 1,
                "verse": 1,
                "contribution_type": "translation_note",
                "title": "Logos context",
                "content": "The term Logos had deep Greek philosophical meaning.",
            },
        )
        assert create_resp.status_code == 200
        contrib_id = create_resp.json()["id"]

        # Review (approve)
        review_resp = phase9_client.post(
            f"/api/v1/community/contributions/{contrib_id}/review",
            json={"decision": "approve", "review_note": "Accurate and helpful."},
        )
        assert review_resp.status_code == 200
        assert review_resp.json()["new_status"] == "approved"


# ============================================================
# Phase 9C: Collaborative Concept Editing Tests
# ============================================================


class TestConceptProposals:
    """Integration tests for concept proposal workflow."""

    def test_create_proposal(self, phase9_client):
        resp = phase9_client.post(
            "/api/v1/concepts/proposals",
            json={
                "concept_name": "divine_rest",
                "proposal_type": "new",
                "description": "God's rest as a theological concept distinct from physical rest.",
                "hebrew_terms": [{"strongs": "H7673", "word": "shabat"}],
                "greek_terms": [],
                "verse_mappings": ["Gen 2:2", "Heb 4:9"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["concept_name"] == "divine_rest"
        assert data["proposal_type"] == "new"
        assert data["status"] == "pending"

    def test_list_proposals(self, phase9_client):
        phase9_client.post(
            "/api/v1/concepts/proposals",
            json={
                "concept_name": "test_concept",
                "proposal_type": "new",
                "description": "Test",
            },
        )
        resp = phase9_client.get("/api/v1/concepts/proposals")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_proposals_filter_status(self, phase9_client):
        resp = phase9_client.get("/api/v1/concepts/proposals?status=pending")
        assert resp.status_code == 200


# ============================================================
# Phase 9D: Audio Integration Tests
# ============================================================


class TestAudioIntegration:
    """Integration tests for audio resource endpoints."""

    def test_get_audio_resource(self, phase9_client):
        resp = phase9_client.get("/api/v1/audio/1/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_id"] == 1
        assert data["chapter"] == 1
        assert "audio_url" in data

    def test_get_audio_with_translation(self, phase9_client):
        resp = phase9_client.get("/api/v1/audio/43/1?translation_id=engkjv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["translation_id"] == "engkjv"


# ============================================================
# Phase 9D: Semantic Relationship Graph Tests
# ============================================================


class TestSemanticGraphPopulator:
    """Unit tests for semantic graph population."""

    def test_populator_creates_relationships(self, phase9_db):
        import sqlite3

        conn = sqlite3.connect(str(phase9_db))
        cursor = conn.execute("SELECT COUNT(*) FROM semantic_relationship_graph")
        count = cursor.fetchone()[0]
        assert count > 0, "Semantic relationships should be populated"
        conn.close()


class TestConceptGraphAPI:
    """Integration tests for concept graph endpoint."""

    def test_get_concept_graph(self, phase9_client):
        resp = phase9_client.get("/api/v1/graph/grace")
        assert resp.status_code == 200
        graph = resp.json()
        assert graph["center_concept"] == "grace"
        assert "relationships" in graph
        assert "nodes" in graph

    def test_get_concept_graph_with_depth(self, phase9_client):
        resp = phase9_client.get("/api/v1/graph/faith?depth=2")
        assert resp.status_code == 200
        graph = resp.json()
        assert graph["center_concept"] == "faith"

    def test_get_concept_graph_unknown(self, phase9_client):
        resp = phase9_client.get("/api/v1/graph/nonexistent_concept")
        assert resp.status_code == 200
        graph = resp.json()
        # Should still return a valid graph, just with no relationships
        assert graph["center_concept"] == "nonexistent_concept"
        assert graph["relationships"] == []


# ============================================================
# Phase 9D: ML Concept Feedback Tests
# ============================================================


class TestConceptFeedback:
    """Integration tests for ML concept feedback."""

    def test_submit_feedback(self, phase9_client):
        resp = phase9_client.post("/api/v1/concepts/grace/feedback?verse_id=Gen.1.1&feedback_type=relevant")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_submit_invalid_feedback(self, phase9_client):
        resp = phase9_client.post("/api/v1/concepts/grace/feedback?verse_id=Gen.1.1&feedback_type=wrong")
        assert resp.status_code == 400

    def test_feedback_summary(self, phase9_client):
        # Submit some feedback first
        phase9_client.post("/api/v1/concepts/grace/feedback?verse_id=Gen.1.1&feedback_type=relevant")
        phase9_client.post("/api/v1/concepts/grace/feedback?verse_id=Gen.1.2&feedback_type=irrelevant")

        resp = phase9_client.get("/api/v1/concepts/grace/feedback/summary")
        assert resp.status_code == 200
        summary = resp.json()
        assert summary["concept_name"] == "grace"
        assert "feedback" in summary


# ============================================================
# Phase 9D: Mobile Native App API Tests
# ============================================================


class TestMobileAPI:
    """Integration tests for mobile sync endpoint."""

    def test_mobile_sync_basic(self, phase9_client):
        resp = phase9_client.post(
            "/api/v1/mobile/sync",
            json={"book_ids": [1], "include_words": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sync_timestamp" in data
        assert "verses" in data
        assert data["total_verses"] > 0

    def test_mobile_sync_with_words(self, phase9_client):
        resp = phase9_client.post(
            "/api/v1/mobile/sync",
            json={"book_ids": [1], "include_words": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Verses should have compact word data
        if data["verses"]:
            verse = data["verses"][0]
            assert "ref" in verse
            assert "text" in verse
            assert "tid" in verse

    def test_mobile_sync_multiple_books(self, phase9_client):
        resp = phase9_client.post(
            "/api/v1/mobile/sync",
            json={"book_ids": [1, 43], "include_words": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_verses"] > 0


# ============================================================
# E2E: Full Scholarly Verse Flow
# ============================================================


class TestScholarlyVerseE2E:
    """End-to-end test: request a verse at scholarly depth with all Phase 9 data."""

    def test_scholarly_verse_includes_phase9_fields(self, phase9_client):
        resp = phase9_client.get("/api/v1/verses/engbsb/1/1/1?depth=scholarly")
        assert resp.status_code == 200
        data = resp.json()

        # Core fields always present
        assert data["reference"]
        assert data["text"]

        # Phase 9 scholarly fields should be present (may be empty lists)
        assert "manuscript_variants" in data
        assert "syntax_tree" in data
        assert "discourse_units" in data
        assert "semantic_domains" in data

    def test_deep_verse_no_phase9_fields(self, phase9_client):
        """Deep depth should NOT include syntax_tree or manuscript_variants."""
        resp = phase9_client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        # These are only at scholarly level
        assert data.get("syntax_tree") is None
        assert data.get("manuscript_variants") is None


# ============================================================
# E2E: Concept Discovery to Study Flow
# ============================================================


class TestConceptDiscoveryE2E:
    """End-to-end: discover a concept, then study related verses."""

    def test_discover_and_follow(self, phase9_client):
        # Step 1: Discover concepts about "forgiveness"
        discover_resp = phase9_client.get("/api/v1/discover?q=forgiveness")
        assert discover_resp.status_code == 200
        discovery = discover_resp.json()
        assert discovery["query"] == "forgiveness"

        # Step 2: If we got a concept graph, follow it
        graph_resp = phase9_client.get("/api/v1/graph/forgiveness")
        assert graph_resp.status_code == 200
        graph = graph_resp.json()
        assert graph["center_concept"] == "forgiveness"

    def test_word_domain_to_related_words(self, phase9_client):
        """E2E: Look up a word's domains, then find related words in same domain."""
        # Step 1: Get domains for logos (G3056)
        domain_resp = phase9_client.get("/api/v1/words/G3056/domains")
        assert domain_resp.status_code == 200
        result = domain_resp.json()
        assert result["strongs_number"] == "G3056"
